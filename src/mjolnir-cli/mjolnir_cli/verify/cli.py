"""``mjolnir verify`` — correctness, benchmarks and profiles of the Mjolnir
kernel, every run saved as structured raw data (``benchmarks/raw/verify``).

    mjolnir verify correctness            # kernel vs emulator matrix
    mjolnir verify sweep                  # Mjolnir vs FlashInfer vs FA2 at every L (ncu)
    mjolnir verify hndsweep|kvfmt|quant|ring|prefetch|pdl|ab|fp6prof|timing|vllm
    mjolnir verify bench                  # every benchmark stage, default parameters
    mjolnir verify all                    # correctness + every benchmark stage
    mjolnir verify show [stage]           # re-print the latest saved run(s)
    mjolnir verify list                   # stages + saved runs

Every stage defaults to the full L axis: 128 4096 8192 16384 32768 65536
131072 262144 (``--ls`` overrides). A run with non-default parameters is
saved too, but marked ``default: false`` so it never replaces a headline
chart (``mjolnir plot`` reads the latest default run of each stage).
"""

from __future__ import annotations

import shutil
import sys
from typing import Optional

import typer

from mjolnir_cli.config import load_layout, resolve
from mjolnir_cli.verify import results, stages
from mjolnir_cli.verify.docker import Runner, ensure_image
from mjolnir_cli.verify.report import console, show

app = typer.Typer(
    help="Kernel correctness + benchmarks (Mjolnir vs FlashInfer vs FA2), saved as raw data for the charts.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

_LS_HELP = "L values (default: 128 4096 8192 16384 32768 65536 131072 262144)"


def _parse_ls(ls: Optional[str]) -> tuple[int, ...]:
    if not ls:
        return stages.ALL_LS
    return tuple(int(x) for x in ls.replace(",", " ").split())


def _kernel_info(layout) -> dict:
    info: dict = {}
    try:
        import mjolnir  # the kernel package (workspace / installed)

        info["version"] = getattr(mjolnir, "__version__", "?")
    except ImportError:
        pass
    if layout.repo_root is not None:
        src = layout.repo_root / "src" / "mjolnir" / "mjolnir" / "__init__.py"
        if src.exists():
            import re

            m = re.search(r'__version__\s*=\s*"([^"]+)"', src.read_text())
            if m:
                info["version"] = m.group(1)
        info.update(results.git_info(layout.repo_root))
    return info


# docker missing / image unavailable: nothing can run, stop the whole sequence
PREFLIGHT_RC = 2


def execute(
    stage: str,
    params: dict,
    defaults: dict,
    *,
    image: Optional[str],
    ls: Optional[str],
    iters: int,
    gate: bool,
    dry_run: bool,
    timeout: float,
    save: bool,
    plot: bool,
    page: int = 128,
) -> tuple[int, dict | None]:
    layout = load_layout()
    s = resolve(None, None, image, None)
    if shutil.which("docker") is None and not dry_run:
        console.print("[red]docker not found on PATH[/red]")
        return PREFLIGHT_RC, None
    if not dry_run:
        err = ensure_image(s.image)
        if err:
            console.print(f"[red]✗ {err}[/red]")
            return PREFLIGHT_RC, None
    run_id = results.new_run_id()
    logdir = layout.benchmarks_dir / "logs" / f"verify-{run_id}"
    kernel_src = None
    if layout.repo_root is not None:
        cand = layout.repo_root / "src" / "mjolnir"
        if (cand / "mjolnir" / "t5_decode.py").exists():
            kernel_src = cand
    runner = Runner(image=s.image, logdir=logdir, kernel_src=kernel_src, timeout=timeout, dry_run=dry_run, page=page)
    ctx = stages.Ctx(
        runner=runner,
        run_id=run_id,
        logdir=logdir,
        metrics_url=s.metrics_url,
        gate=gate,
        ls=_parse_ls(ls),
        iters=iters,
        out=lambda m: console.print(m, highlight=False),
    )
    st = stages.STAGES[stage]
    console.print(f"\n[bold cyan]━━ verify {stage}[/bold cyan] — {st.summary}")
    console.print(
        f"[dim]   image {s.image} · kernel {'checkout ' + str(kernel_src) if kernel_src else 'baked into the image'}"
        f" · L {' '.join(str(x) for x in ctx.ls)} · logs {logdir}[/dim]"
    )
    if not dry_run:
        from mjolnir_cli import cachectl

        cachectl.prepare_bench()  # clean page cache before measuring (no-op without the grant)
    kwargs = dict(params)
    if stage == "vllm":
        kwargs["repo"] = layout.require_repo("verify vllm")
    body = st.fn(ctx, **kwargs)
    is_default = params == defaults and ctx.ls == stages.ALL_LS and iters == stages.DEFAULT_ITERS and page == 128
    doc = results.header(
        stage,
        run_id,
        image=s.image,
        default=is_default,
        params={
            **body.pop("params", {}),
            "ls": list(ctx.ls),
            "iters": iters,
            "page": page,
            "hq": runner.hq,
            "hk": runner.hk,
            "d": runner.d,
        },
        kernel=_kernel_info(layout),
        gate=ctx.gate_log,
    )
    doc.update(body)
    doc["summary"] = stages.summarize(stage, doc)
    if dry_run:
        console.print("[dim]   (dry run — nothing saved)[/dim]")
        return 0, None
    show(doc)
    if save:
        p = results.save(layout.verify_dir, doc)
        console.print(f"   📝 [bold]{p}[/bold]{'' if is_default else '  (custom parameters: not a headline run)'}")
        if plot and layout.is_repo:
            _replot(layout)
    failed = [c for c in doc.get("cases", []) if not c.get("passed")]
    empty = st.kind in ("ncu",) and not any(s_.get("points") for s_ in doc.get("series", []))
    return (1 if failed or empty else 0), doc


def _execute(*a, **kw) -> int:
    return execute(*a, **kw)[0]


def _replot(layout) -> None:
    try:
        from mjolnir_cli import charts, readme

        charts.render_all(layout)
        readme.render_readme(layout)
    except Exception as e:  # noqa: BLE001 — charts are a convenience here
        console.print(f"[yellow]   (charts/README not re-rendered: {e})[/yellow]")


# ── shared options (each command declares them; typer needs explicit params) ──
IMAGE = typer.Option(None, "--image", help="vLLM image (default: the active one, see `mjolnir image`)")
LS = typer.Option(None, "--ls", help=_LS_HELP)
ITERS = typer.Option(stages.DEFAULT_ITERS, "--iters", help="profiled iterations per point (ncu stages)")
GATE = typer.Option(True, "--gate/--no-gate", help="wait for a clean window when the live vLLM server is up")
DRY = typer.Option(False, "--dry-run", help="print the docker commands, run nothing")
TIMEOUT = typer.Option(180.0, "--timeout", help="per-container timeout (s); a hung kernel's container is killed")
SAVE = typer.Option(True, "--save/--no-save", help="write the raw result JSON")
PLOT = typer.Option(False, "--plot/--no-plot", help="re-render charts + README after saving (checkout only)")
PAGE = typer.Option(128, "--page-size", help="KV page size for every leg")


def _run(stage: str, params: dict, defaults: dict, image, ls, iters, gate, dry_run, timeout, save, plot, page=128):
    raise typer.Exit(
        _execute(
            stage,
            params,
            defaults,
            image=image,
            ls=ls,
            iters=iters,
            gate=gate,
            dry_run=dry_run,
            timeout=timeout,
            save=save,
            plot=plot,
            page=page,
        )
    )


@app.command()
def correctness(
    dtypes: str = typer.Option("fp8 bf16", "--dtypes", help="Q dtypes to test"),
    match: str = typer.Option("", "--match", help="only cases whose label contains this"),
    quick: bool = typer.Option(False, "--quick", help="the quick subset (one case per group; what autoresearch runs)"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    image: Optional[str] = IMAGE,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
):
    """Kernel vs the bit-faithful emulator: every layout, KV format, page size, q_len and Q dtype."""
    p = {"dtypes": dtypes, "match": match, "fail_fast": fail_fast, "timeout": timeout, "quick": quick}
    d = {"dtypes": "fp8 bf16", "match": "", "fail_fast": False, "timeout": 180.0, "quick": False}
    _run("correctness", p, d, image, None, stages.DEFAULT_ITERS, False, dry_run, timeout, save, False)


@app.command()
def sweep(
    legs: str = typer.Option(
        "fi,fa2,mjolnir-fp8,mjolnir", "--legs", help="legs: fi, fi-hnd, fa2, mjolnir-fp8, mjolnir"
    ),
    image: Optional[str] = IMAGE,
    ls: Optional[str] = LS,
    iters: int = ITERS,
    gate: bool = GATE,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
    plot: bool = PLOT,
    page: int = PAGE,
):
    """The headline: Mjolnir vs FlashInfer vs FlashAttention 2 at every L (ncu, cold L2, base clocks)."""
    _run(
        "sweep",
        {"legs": legs},
        {"legs": "fi,fa2,mjolnir-fp8,mjolnir"},
        image,
        ls,
        iters,
        gate,
        dry_run,
        timeout,
        save,
        plot,
        page,
    )


@app.command()
def hndsweep(
    legs: str = typer.Option("fi,fi-hnd,mjolnir-fp8", "--legs"),
    image: Optional[str] = IMAGE,
    ls: Optional[str] = LS,
    iters: int = ITERS,
    gate: bool = GATE,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
    plot: bool = PLOT,
    page: int = PAGE,
):
    """FlashInfer on NHD and HND pages vs Mjolnir (fp8 KV) on HND: the fair layout comparison."""
    _run(
        "hndsweep",
        {"legs": legs},
        {"legs": "fi,fi-hnd,mjolnir-fp8"},
        image,
        ls,
        iters,
        gate,
        dry_run,
        timeout,
        save,
        plot,
        page,
    )


@app.command()
def kvfmt(
    variants: str = typer.Option(stages.KVFMT_DEFAULT, "--variants", help="name:flags|name:flags"),
    refs: str = typer.Option("fi,fi-hnd", "--refs", help="baseline legs"),
    wall_iters: int = typer.Option(200, "--wall-iters", help="harness wall-clock iterations (0 = skip)"),
    image: Optional[str] = IMAGE,
    ls: Optional[str] = LS,
    iters: int = ITERS,
    gate: bool = GATE,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
    plot: bool = PLOT,
):
    """KV layouts x formats: fp8 / fp6 (e2m3, e3m2) / fp4 (e2m1) / nvfp4 on NHD / interleaved / HND."""
    _run(
        "kvfmt",
        {"variants": variants, "refs": refs, "wall_iters": wall_iters},
        {"variants": stages.KVFMT_DEFAULT, "refs": "fi,fi-hnd", "wall_iters": 200},
        image,
        ls,
        iters,
        gate,
        dry_run,
        timeout,
        save,
        plot,
    )


@app.command()
def quant(
    formats: str = typer.Option(stages.QUANT_FORMATS, "--formats"),
    batches: int = typer.Option(2, "--batches"),
    image: Optional[str] = IMAGE,
    ls: Optional[str] = LS,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
    plot: bool = PLOT,
):
    """Accuracy per KV format: relative L2 error vs attention over unquantized KV."""
    _run(
        "quant",
        {"formats": formats, "batches": batches},
        {"formats": stages.QUANT_FORMATS, "batches": 2},
        image,
        ls,
        stages.DEFAULT_ITERS,
        False,
        dry_run,
        timeout,
        save,
        plot,
    )


@app.command()
def ring(
    depths: str = typer.Option(stages.RING_DEFAULT, "--stages", help="--kv_stages values"),
    base: str = typer.Option("", "--base", help="harness flags every variant gets"),
    refs: str = typer.Option("fi", "--refs"),
    image: Optional[str] = IMAGE,
    ls: Optional[str] = LS,
    iters: int = ITERS,
    gate: bool = GATE,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
    plot: bool = PLOT,
):
    """KV ring depth (TMA → MMA stages) at every L."""
    _run(
        "ring",
        {"stages": depths, "base": base, "refs": refs},
        {"stages": stages.RING_DEFAULT, "base": "", "refs": "fi"},
        image,
        ls,
        iters,
        gate,
        dry_run,
        timeout,
        save,
        plot,
    )


@app.command()
def prefetch(
    distances: str = typer.Option("0 1 2 4", "--distances", help="--l2_prefetch tiles ahead"),
    base: str = typer.Option("--kv_format fp8", "--base"),
    refs: str = typer.Option("fi", "--refs"),
    image: Optional[str] = IMAGE,
    ls: Optional[str] = LS,
    iters: int = ITERS,
    gate: bool = GATE,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
    plot: bool = PLOT,
):
    """TMA L2-prefetch distance sweep."""
    _run(
        "prefetch",
        {"distances": distances, "base": base, "refs": refs},
        {"distances": "0 1 2 4", "base": "--kv_format fp8", "refs": "fi"},
        image,
        ls,
        iters,
        gate,
        dry_run,
        timeout,
        save,
        plot,
    )


@app.command()
def pdl(
    batches: str = typer.Option("1 8", "--batches"),
    dtypes: str = typer.Option("fp8", "--dtypes"),
    wall_iters: int = typer.Option(200, "--wall-iters"),
    image: Optional[str] = IMAGE,
    ls: Optional[str] = LS,
    gate: bool = GATE,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
    plot: bool = PLOT,
):
    """Programmatic dependent launch on/off (wall clock)."""
    _run(
        "pdl",
        {"batches": batches, "dtypes": dtypes, "iters": wall_iters},
        {"batches": "1 8", "dtypes": "fp8", "iters": 200},
        image,
        ls,
        stages.DEFAULT_ITERS,
        gate,
        dry_run,
        timeout,
        save,
        plot,
    )


@app.command()
def ab(
    variants: str = typer.Option(stages.AB_DEFAULT, "--variants", help="name:flags|name:flags"),
    refs: str = typer.Option("fi", "--refs"),
    wall_iters: int = typer.Option(200, "--wall-iters"),
    image: Optional[str] = IMAGE,
    ls: Optional[str] = LS,
    iters: int = ITERS,
    gate: bool = GATE,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
    plot: bool = PLOT,
):
    """A/B any harness flag sets (ncu + wall clock). Default: evict-first x PDL."""
    _run(
        "ab",
        {"variants": variants, "refs": refs, "wall_iters": wall_iters},
        {"variants": stages.AB_DEFAULT, "refs": "fi", "wall_iters": 200},
        image,
        ls,
        iters,
        gate,
        dry_run,
        timeout,
        save,
        plot,
    )


@app.command()
def fp6prof(
    variants: str = typer.Option(stages.PROF_DEFAULT, "--variants"),
    count: int = typer.Option(3, "--count", help="profiled launches per point"),
    section_set: str = typer.Option("full", "--set"),
    image: Optional[str] = IMAGE,
    ls: Optional[str] = LS,
    gate: bool = GATE,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
    plot: bool = PLOT,
):
    """Full ncu profiles per KV format (bytes vs requests vs time, warp stalls)."""
    _run(
        "fp6prof",
        {"variants": variants, "count": count, "section_set": section_set},
        {"variants": stages.PROF_DEFAULT, "count": 3, "section_set": "full"},
        image,
        ls,
        stages.DEFAULT_ITERS,
        gate,
        dry_run,
        timeout,
        save,
        plot,
    )


@app.command()
def timing(
    dtypes: str = typer.Option("fp8 bf16", "--dtypes"),
    ctas: str = typer.Option("1 2", "--ctas"),
    wall_iters: int = typer.Option(50, "--wall-iters"),
    image: Optional[str] = IMAGE,
    ls: Optional[str] = LS,
    gate: bool = GATE,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
):
    """Harness-timed medians, 1 vs 2 CTAs/SM, fp8 and bf16 Q (no ncu)."""
    _run(
        "timing",
        {"dtypes": dtypes, "ctas": ctas, "iters": wall_iters},
        {"dtypes": "fp8 bf16", "ctas": "1 2", "iters": 50},
        image,
        ls,
        stages.DEFAULT_ITERS,
        gate,
        dry_run,
        timeout,
        save,
        False,
    )


@app.command()
def vllm(
    smoke_modes: str = typer.Option("graph eager", "--smoke-modes", help="graph | eager | spec<N>; '' = none"),
    model: str = typer.Option("Qwen/Qwen3-0.6B", "--model"),
    tokens: int = typer.Option(128, "--tokens"),
    image: Optional[str] = IMAGE,
    dry_run: bool = DRY,
    timeout: float = typer.Option(1800.0, "--timeout"),
    save: bool = SAVE,
):
    """The vLLM plugin: runner tests + greedy MJOLNIR-vs-FLASHINFER smoke runs (checkout only)."""
    _run(
        "vllm",
        {"smoke_modes": smoke_modes, "model": model, "tokens": tokens, "timeout": timeout},
        {"smoke_modes": "graph eager", "model": "Qwen/Qwen3-0.6B", "tokens": 128, "timeout": 1800.0},
        image,
        None,
        stages.DEFAULT_ITERS,
        False,
        dry_run,
        timeout,
        save,
        False,
    )


def _sequence(names: list[str], image, gate, dry_run, plot) -> int:
    rc = 0
    for name in names:
        defaults = _DEFAULTS[name]
        r = _execute(
            name,
            dict(defaults),
            dict(defaults),
            image=image,
            ls=None,
            iters=stages.DEFAULT_ITERS,
            gate=gate,
            dry_run=dry_run,
            timeout=180.0,
            save=True,
            plot=False,
        )
        if r == PREFLIGHT_RC:
            return r  # no image / no docker: every later stage would fail the same way
        if r != 0:
            console.print(f"[red]   verify {name} exit {r} — continuing[/red]")
            rc = r
    if plot and not dry_run:
        _replot(load_layout())
    return rc


_DEFAULTS: dict[str, dict] = {
    "correctness": {"dtypes": "fp8 bf16", "match": "", "fail_fast": False, "timeout": 180.0, "quick": False},
    "sweep": {"legs": "fi,fa2,mjolnir-fp8,mjolnir"},
    "hndsweep": {"legs": "fi,fi-hnd,mjolnir-fp8"},
    "kvfmt": {"variants": stages.KVFMT_DEFAULT, "refs": "fi,fi-hnd", "wall_iters": 200},
    "quant": {"formats": stages.QUANT_FORMATS, "batches": 2},
    "ring": {"stages": stages.RING_DEFAULT, "base": "", "refs": "fi"},
    "prefetch": {"distances": "0 1 2 4", "base": "--kv_format fp8", "refs": "fi"},
    "pdl": {"batches": "1 8", "dtypes": "fp8", "iters": 200},
    "ab": {"variants": stages.AB_DEFAULT, "refs": "fi", "wall_iters": 200},
    "fp6prof": {"variants": stages.PROF_DEFAULT, "count": 3, "section_set": "full"},
}


@app.command()
def bench(
    image: Optional[str] = IMAGE,
    gate: bool = GATE,
    dry_run: bool = DRY,
    plot: bool = typer.Option(True, "--plot/--no-plot"),
):
    """Every benchmark stage with default parameters (one headline run each), then re-render the charts."""
    raise typer.Exit(_sequence(stages.BENCH_ORDER, image, gate, dry_run, plot))


@app.command("all")
def all_(
    image: Optional[str] = IMAGE,
    gate: bool = GATE,
    dry_run: bool = DRY,
    plot: bool = typer.Option(True, "--plot/--no-plot"),
):
    """Correctness + every benchmark stage, then re-render the charts."""
    raise typer.Exit(_sequence(stages.ALL_ORDER, image, gate, dry_run, plot))


@app.command("show")
def show_cmd(
    stage: Optional[str] = typer.Argument(None, help="stage (default: every stage with a saved run)"),
    any_run: bool = typer.Option(False, "--any", help="latest run of any parameters, not just default runs"),
):
    """Re-print the latest saved run of a stage (or every stage)."""
    layout = load_layout()
    names = [stage] if stage else list(stages.STAGES)
    found = False
    for n in names:
        doc = results.latest(layout.verify_dir, n, default_only=not any_run)
        if doc:
            found = True
            show(doc)
    if not found:
        console.print(f"[yellow]no saved verify runs under {layout.verify_dir} — run: mjolnir verify sweep[/yellow]")


@app.command("list")
def list_cmd():
    """Stages and their saved runs."""
    layout = load_layout()
    for name, st in stages.STAGES.items():
        rs = results.runs(layout.verify_dir, name)
        last = rs[-1]["run_id"] if rs else "—"
        n_def = sum(1 for r in rs if r.get("default"))
        console.print(
            f"  [bold]{name:<12}[/bold] [dim]{st.kind:<11}[/dim] runs {len(rs):>3} (default {n_def:>2}) "
            f"last {last:<16} {st.summary}"
        )
    console.print(f"[dim]  raw results: {layout.verify_dir}[/dim]")


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
