"""``mjolnir verify`` — correctness, benchmarks and profiles of the Mjolnir
kernel, every run saved as structured raw data (``benchmarks/raw/verify``).

    mjolnir verify correctness            # kernel vs emulator matrix
    mjolnir verify sweep                  # Mjolnir vs FlashInfer vs FA4 hd256 at every L (ncu)
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

import typer

from mjolnir_cli import legs as legs_
from mjolnir_cli import retention
from mjolnir_cli.cli_shared import _unset_flag
from mjolnir_cli.config import DEFAULT_IMAGE, load_layout, resolve
from mjolnir_cli.legs import KERNEL, SMOKE_MODES
from mjolnir_cli.platforms import DEFAULT_PLATFORM, gpu_name
from mjolnir_cli.verify import results, stages
from mjolnir_cli.verify.kvcache import DEFAULT_LEGS as KV_LEGS
from mjolnir_cli.verify.profile import DEFAULT_LEGS as PROFILE_LEGS
from mjolnir_cli.verify.spec import DEFAULT_DRAFTS, DEFAULT_LEGS as SPEC_LEGS
from mjolnir_cli.verify.docker import Runner, ensure_image
from mjolnir_cli.verify.report import console, show

app = typer.Typer(
    help="Kernel correctness + benchmarks (Mjolnir vs FlashInfer vs FA4 hd256), saved as raw data for the charts.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

_LS_HELP = "L values (default: 128 4096 8192 16384 32768 65536 131072 262144)"


def _parse_ls(ls: str | None) -> tuple[int, ...]:
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


def _platform_legs(
    stage: str, params: dict, defaults: dict, platform: str | None, legs_unset: bool
) -> tuple[dict, dict]:
    """The verify legs sentinel (plan §0.5): the stage defaults are the
    default platform's leg table, so on another platform an **unset**
    ``--legs`` re-resolves there — and stays the stage's default parameters
    *for that platform*, so ``is_default`` keeps meaning "stage defaults".
    Unset is the typer parameter source (``cli_shared._unset_flag``), not
    string equality: an explicitly passed flag wins unchanged even when it
    spells the default-platform table. Stages without a ``--legs`` option
    pass ``legs_unset=True`` (nothing explicit to win); an unresolved
    Settings (platform ``None``) serves the default table."""
    if platform is None or platform == DEFAULT_PLATFORM or not legs_unset:
        return params, defaults
    name = f"verify.{stage}"  # stages are registered bare; the legs tables are verify.-prefixed
    if name in legs_.SERVING:
        resolved: str | None = legs_.verify_legs(name, platform)
    elif name in legs_.KERNEL:
        resolved = legs_.kernel_legs(name, platform)
    else:
        return params, defaults
    return {**params, "legs": resolved}, {**defaults, "legs": resolved}


def execute(
    stage: str,
    params: dict,
    defaults: dict,
    *,
    image: str | None,
    ls: str | None,
    iters: int,
    gate: bool,
    dry_run: bool,
    timeout: float,
    save: bool,
    plot: bool,
    page: int = 128,
    legs_unset: bool = True,
) -> tuple[int, dict | None]:
    layout = load_layout()
    s = resolve(None, None, image, None)
    params, defaults = _platform_legs(stage, params, defaults, s.platform, legs_unset)
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
        if (cand / "mjolnir" / "arch" / "thor" / "decode.py").exists():
            kernel_src = cand
    runner = Runner(
        image=s.image,
        logdir=logdir,
        platform=s.platform or DEFAULT_PLATFORM,
        kernel_src=kernel_src,
        timeout=timeout,
        dry_run=dry_run,
        page=page,
    )
    ctx = stages.Ctx(
        runner=runner,
        run_id=run_id,
        logdir=logdir,
        metrics_url=s.metrics_url,
        gate=gate,
        ls=_parse_ls(ls),
        iters=iters,
        out=lambda m: console.print(m, highlight=False, soft_wrap=True),  # rows never hard-wrap
    )
    st = stages.STAGES[stage]
    console.print(f"\n[bold cyan]━━ verify {stage}[/bold cyan] — {st.summary}")
    console.print(
        f"[dim]   image {s.image}{'' if s.image == DEFAULT_IMAGE else f' (not the default {DEFAULT_IMAGE})'} · kernel {'checkout ' + str(kernel_src) if kernel_src else 'baked into the image'}"
        f" · L {' '.join(str(x) for x in ctx.ls)} · logs {logdir}[/dim]"
    )
    console.print(
        f"[dim]   FlashInfer references run on the official image {runner.ref_image}; the FA4 leg runs on the Mjolnir image (its patched FA4)[/dim]"
    )
    from mjolnir_cli import cachectl

    if not dry_run:
        cachectl.prepare_bench()  # clean page cache before measuring (no-op without the grant)
    kwargs = dict(params)
    if stage == "vllm":
        kwargs["repo"] = layout.require_repo("verify vllm")
    assert st.fn is not None, f"stage {stage} has no body"
    try:
        body = st.fn(ctx, **kwargs)
    finally:
        if not dry_run:  # Thor keeps the page cache the stage's containers filled
            cachectl.prepare_bench()
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
        platform=s.platform,
        gpu=gpu_name(),
    )
    doc["ref_image"] = runner.ref_image  # where the FlashInfer legs ran (stock vLLM; FA4 runs on our image)
    doc.update(body)
    doc["summary"] = stages.summarize(stage, doc)
    if dry_run:
        console.print("[dim]   (dry run — nothing saved)[/dim]")
        return 0, None
    show(doc)
    if save:
        p = results.save(layout.verify_dir, doc)
        if p.parent.parent == layout.verify_dir:
            console.print(f"   📝 [bold]{p}[/bold]  (the stage's evidence — replaces the previous default run)")
        else:
            why = "uncharted stage" if is_default else "custom parameters"
            console.print(f"   📝 [bold]{p}[/bold]  (untracked — {why}: stays local, never in a chart)")
        rep = retention.sweep(layout)  # the writer prunes: expired staging/logs, silent when nothing freed
        if rep.removed:
            console.print(f"   🧹 [dim]{rep.summary()}[/dim]")
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


def _run(
    stage: str,
    params: dict,
    defaults: dict,
    image,
    ls,
    iters,
    gate,
    dry_run,
    timeout,
    save,
    plot,
    page=128,
    legs_unset: bool = True,
):
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
            legs_unset=legs_unset,
        )
    )


@app.command()
def correctness(
    dtypes: str = typer.Option("fp8 bf16", "--dtypes", help="Q dtypes to test"),
    match: str = typer.Option("", "--match", help="only cases whose label contains this"),
    quick: bool = typer.Option(False, "--quick", help="the quick subset (one case per group; what autoresearch runs)"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    image: str | None = IMAGE,
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
    ctx: typer.Context,
    legs: str = typer.Option(KERNEL["verify.sweep"], "--legs", help="legs: fi, fi-hnd, fa2, fa4, mjolnir-fp8, mjolnir"),
    image: str | None = IMAGE,
    ls: str | None = LS,
    iters: int = ITERS,
    gate: bool = GATE,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
    plot: bool = PLOT,
    page: int = PAGE,
    batch: int = typer.Option(1, "--batch", help="sequences per step, each of L tokens (serving: 1-4)"),
    q_len: int = typer.Option(1, "--q-len", help="query tokens per sequence (MTP=3 verification: 4)"),
):
    """The headline: Mjolnir vs FlashInfer vs FlashAttention 4 hd256 at every L (ncu, cold L2, base clocks).

    `--batch 4 --q_len 4 --legs fi,mjolnir-fp8` measures the shape vLLM serves
    with MTP=3 at 4 concurrent requests (saved as a non-default run)."""
    _run(
        "sweep",
        {"legs": legs, "batch": batch, "q_len": q_len},
        {"legs": KERNEL["verify.sweep"], "batch": 1, "q_len": 1},
        image,
        ls,
        iters,
        gate,
        dry_run,
        timeout,
        save,
        plot,
        page,
        legs_unset=_unset_flag(ctx, "legs"),
    )


@app.command()
def hndsweep(
    ctx: typer.Context,
    legs: str = typer.Option(KERNEL["verify.hndsweep"], "--legs"),
    image: str | None = IMAGE,
    ls: str | None = LS,
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
        {"legs": KERNEL["verify.hndsweep"]},
        image,
        ls,
        iters,
        gate,
        dry_run,
        timeout,
        save,
        plot,
        page,
        legs_unset=_unset_flag(ctx, "legs"),
    )


@app.command()
def kvfmt(
    variants: str = typer.Option("", "--variants", help="name:flags|name:flags (default: the platform's set)"),
    refs: str = typer.Option(KERNEL["verify.kvfmt.refs"], "--refs", help="baseline legs"),
    wall_iters: int = typer.Option(200, "--wall-iters", help="harness wall-clock iterations (0 = skip)"),
    image: str | None = IMAGE,
    ls: str | None = LS,
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
        {"variants": "", "refs": KERNEL["verify.kvfmt.refs"], "wall_iters": 200},
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
    formats: str = typer.Option("", "--formats", help="default: the platform's set"),
    batches: int = typer.Option(2, "--batches"),
    image: str | None = IMAGE,
    ls: str | None = LS,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
    plot: bool = PLOT,
):
    """Accuracy per KV format: relative L2 error vs attention over unquantized KV."""
    _run(
        "quant",
        {"formats": formats, "batches": batches},
        {"formats": "", "batches": 2},
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
    image: str | None = IMAGE,
    ls: str | None = LS,
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
    image: str | None = IMAGE,
    ls: str | None = LS,
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
    image: str | None = IMAGE,
    ls: str | None = LS,
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
    variants: str = typer.Option("", "--variants", help="name:flags|name:flags (default: the platform's set)"),
    refs: str = typer.Option("fi", "--refs"),
    wall_iters: int = typer.Option(200, "--wall-iters"),
    image: str | None = IMAGE,
    ls: str | None = LS,
    iters: int = ITERS,
    gate: bool = GATE,
    dry_run: bool = DRY,
    timeout: float = TIMEOUT,
    save: bool = SAVE,
    plot: bool = PLOT,
):
    """A/B any harness flag sets (ncu + wall clock). Default: the platform's
    own A/B — the Thor launch knobs (evict-first × PDL) / the Orin KV formats."""
    _run(
        "ab",
        {"variants": variants, "refs": refs, "wall_iters": wall_iters},
        {"variants": "", "refs": "fi", "wall_iters": 200},
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
    variants: str = typer.Option("", "--variants", help="name:flags|name:flags (default: the platform's set)"),
    count: int = typer.Option(3, "--count", help="profiled launches per point"),
    section_set: str = typer.Option("full", "--set"),
    image: str | None = IMAGE,
    ls: str | None = LS,
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
        {"variants": "", "count": 3, "section_set": "full"},
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
    image: str | None = IMAGE,
    ls: str | None = LS,
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


@app.command("profile")
def profile(
    ctx: typer.Context,
    legs: str = typer.Option(
        PROFILE_LEGS,
        "--legs",
        help="side:config, comma-separated (mjolnir:NVFP4 = FlashInfer on the Mjolnir image)",
    ),
    drafts: int = typer.Option(3, "--drafts", help="num_speculative_tokens (0 = the config's own)"),
    context_words: int = typer.Option(6000, "--context-words", help="prompt size (~1.3 tokens/word; 0 = short)"),
    concurrency: int = typer.Option(1, "--concurrency"),
    requests: int = typer.Option(2, "--requests", help="request rounds inside the profiled window"),
    max_tokens: int = typer.Option(128, "--max-tokens"),
    model: str | None = typer.Option(None, "--model"),
    restart: bool = typer.Option(False, "--restart", help="stop a running vLLM server first"),
    image: str | None = IMAGE,
    dry_run: bool = DRY,
    timeout: float = typer.Option(1800.0, "--timeout"),
    save: bool = SAVE,
):
    """Profile a decode workload per config (torch profiler): graph replays, eager kernels, GPU idle, top kernels."""
    p = {
        "legs": legs,
        "drafts": drafts,
        "context_words": context_words,
        "concurrency": concurrency,
        "requests": requests,
        "max_tokens": max_tokens,
        "model": model,
        "restart": restart,
        "timeout": timeout,
    }
    _run(
        "profile",
        p,
        {**p, "legs": PROFILE_LEGS},
        image,
        None,
        stages.DEFAULT_ITERS,
        False,
        dry_run,
        timeout,
        save,
        False,
        legs_unset=_unset_flag(ctx, "legs"),
    )


@app.command("spec")
def spec(
    ctx: typer.Context,
    legs: str = typer.Option(
        SPEC_LEGS,
        "--legs",
        help="side:config, comma-separated; official = the stock vLLM image, mjolnir = --image",
    ),
    drafts: str = typer.Option(DEFAULT_DRAFTS, "--drafts", help="num_speculative_tokens values to serve"),
    concurrency: str = typer.Option("1 4", "--concurrency", help="requests in flight per block"),
    max_tokens: int = typer.Option(256, "--max-tokens", help="generated tokens per request"),
    long_words: int = typer.Option(6000, "--long-words", help="the long-context prompt's size (0 = none)"),
    model: str | None = typer.Option(None, "--model", help="model (default: the active one)"),
    restart: bool = typer.Option(
        False, "--restart", help="stop a running vLLM server (default: refuse — the GPU must be yours)"
    ),
    image: str | None = IMAGE,
    dry_run: bool = DRY,
    timeout: float = typer.Option(1800.0, "--timeout", help="per-point server start timeout (s)"),
    save: bool = SAVE,
    plot: bool = PLOT,
):
    """MTP per config and draft count: tokens/step, per-position acceptance, decode t/s (restarts the server per point)."""
    defaults = {
        "legs": SPEC_LEGS,
        "drafts": DEFAULT_DRAFTS,
        "concurrency": "1 4",
        "max_tokens": 256,
        "long_words": 6000,
        "model": None,
        "restart": restart,
        "timeout": 1800.0,
    }
    _run(
        "spec",
        {
            "legs": legs,
            "drafts": drafts,
            "concurrency": concurrency,
            "max_tokens": max_tokens,
            "long_words": long_words,
            "model": model,
            "restart": restart,
            "timeout": timeout,
        },
        defaults,
        image,
        None,
        stages.DEFAULT_ITERS,
        False,
        dry_run,
        timeout,
        save,
        plot,
        legs_unset=_unset_flag(ctx, "legs"),
    )


@app.command("kv-cache-size")
def kv_cache_size(
    ctx: typer.Context,
    legs: str = typer.Option(
        KV_LEGS,
        "--legs",
        help="side:config, comma-separated; official = the stock vLLM image, mjolnir = --image",
    ),
    model: str | None = typer.Option(None, "--model", help="model (default: the active one)"),
    restart: bool = typer.Option(
        False, "--restart", help="stop a running vLLM server (default: refuse — the GPU must be yours)"
    ),
    image: str | None = IMAGE,
    dry_run: bool = DRY,
    timeout: float = typer.Option(1800.0, "--timeout", help="per-leg wait for the KV sizing (s)"),
    save: bool = SAVE,
    plot: bool = PLOT,
):
    """KV cache capacity per config: serves each leg (restarting the server), reads vLLM's KV sizing, stops it."""
    _run(
        "kv-cache-size",
        {"legs": legs, "model": model, "restart": restart, "timeout": timeout},
        {
            "legs": KV_LEGS,
            "model": None,
            "restart": restart,
            "timeout": 1800.0,
        },
        image,
        None,
        stages.DEFAULT_ITERS,
        False,
        dry_run,
        timeout,
        save,
        plot,
        legs_unset=_unset_flag(ctx, "legs"),
    )


@app.command()
def vllm(
    smoke_modes: str = typer.Option(
        SMOKE_MODES,
        "--smoke-modes",
        help="graph | eager | e2m3 (fp6 KV) | lmh8 (fp8 lm_head) | spec<N>; '' = none",
    ),
    model: str = typer.Option("Qwen/Qwen3-0.6B", "--model"),
    tokens: int = typer.Option(128, "--tokens"),
    image: str | None = IMAGE,
    dry_run: bool = DRY,
    timeout: float = typer.Option(1800.0, "--timeout"),
    save: bool = SAVE,
):
    """The vLLM plugin: runner tests + greedy MJOLNIR-vs-FLASHINFER smoke runs (checkout only)."""
    _run(
        "vllm",
        {"smoke_modes": smoke_modes, "model": model, "tokens": tokens, "timeout": timeout},
        {"smoke_modes": SMOKE_MODES, "model": "Qwen/Qwen3-0.6B", "tokens": 128, "timeout": 1800.0},
        image,
        None,
        stages.DEFAULT_ITERS,
        False,
        dry_run,
        timeout,
        save,
        False,
    )


def _from(order: list[str], start: str | None) -> list[str]:
    if not start:
        return list(order)
    if start not in order:
        raise typer.BadParameter(f"--from must be one of: {', '.join(order)}")
    return order[order.index(start) :]


def _sequence(names: list[str], image, gate, dry_run, plot) -> int:
    rc = 0
    for name in names:
        defaults = _DEFAULTS[name]
        try:
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
        except Exception as e:  # noqa: BLE001 — one broken stage must not cost the rest of the suite
            console.print(f"[red]   verify {name} crashed: {type(e).__name__}: {e} — continuing[/red]")
            r = 1
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
    "sweep": {"legs": KERNEL["verify.sweep"]},
    "hndsweep": {"legs": KERNEL["verify.hndsweep"]},
    "kvfmt": {"variants": "", "refs": KERNEL["verify.kvfmt.refs"], "wall_iters": 200},
    "quant": {"formats": "", "batches": 2},
    "ring": {"stages": stages.RING_DEFAULT, "base": "", "refs": "fi"},
    "prefetch": {"distances": "0 1 2 4", "base": "--kv_format fp8", "refs": "fi"},
    "pdl": {"batches": "1 8", "dtypes": "fp8", "iters": 200},
    "ab": {"variants": "", "refs": "fi", "wall_iters": 200},
    "fp6prof": {"variants": "", "count": 3, "section_set": "full"},
}


@app.command()
def bench(
    image: str | None = IMAGE,
    gate: bool = GATE,
    dry_run: bool = DRY,
    plot: bool = typer.Option(True, "--plot/--no-plot"),
    start: str | None = typer.Option(None, "--from", help="resume the sequence at this stage"),
):
    """Every benchmark stage with default parameters (one headline run each), then re-render the charts."""
    raise typer.Exit(_sequence(_from(stages.BENCH_ORDER, start), image, gate, dry_run, plot))


@app.command("all")
def all_(
    image: str | None = IMAGE,
    gate: bool = GATE,
    dry_run: bool = DRY,
    plot: bool = typer.Option(True, "--plot/--no-plot"),
    start: str | None = typer.Option(None, "--from", help="resume the sequence at this stage"),
):
    """Correctness + every benchmark stage, then re-render the charts."""
    raise typer.Exit(_sequence(_from(stages.ALL_ORDER, start), image, gate, dry_run, plot))


@app.command("show")
def show_cmd(
    stage: str | None = typer.Argument(None, help="stage (default: every stage with a saved run)"),
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
