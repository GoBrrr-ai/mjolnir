"""`mjolnir bench` + `mjolnir history`/`plot` commands (split out of cli.py)."""

from __future__ import annotations

import json
import shlex

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from mjolnir_cli import benchy, charts, dockerctl, e2e, readme
from mjolnir_cli import legs as legs_
from mjolnir_cli.cli_shared import _HELP_CTX, _settings
from mjolnir_cli.config import (
    DEFAULT_IMAGE,
    DEFAULT_QUANT,
    E2E_DEPTHS,
    STOCK_IMAGE,
    VLLM_VERSION,
    ConfigEntry,
    Settings,
    load_layout,
    scan_all_configs,
)
from mjolnir_cli.e2e import MAX_CTX as _E2E_MAX_CTX

app = typer.Typer(
    help="gated end-to-end benchmarks: perf / ab / accuracy / e2e / compare, plus history + plot",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
bench_app = app


@bench_app.command("perf")
def bench_perf(
    runs: int = typer.Option(6, "--runs", help="measured runs per cell (llama-benchy --runs)"),
    warmup_runs: int = typer.Option(2, "--warmup-runs"),
    repeat: int = typer.Option(1, "--repeat", help="independent gated sweeps (default: one sweep, 6 runs per cell)"),
    depths: str = typer.Option(",".join(map(str, E2E_DEPTHS)), "--depths", help="context depths"),
    concurrency: str = typer.Option("1,2,4", "--concurrency"),
    pp: int = typer.Option(2048, "--pp"),
    tg: int = typer.Option(128, "--tg"),
    exact_tg: bool = typer.Option(
        True, "--exact-tg/--no-exact-tg", help="pin output length to --tg (kills EOS variance)"
    ),
    gate: bool = typer.Option(True, "--gate/--no-gate"),
    confirm: int = typer.Option(3, "--confirm", help="clean-window confirm samples"),
    gate_timeout: float = typer.Option(3600.0, "--gate-timeout"),
    label: str | None = typer.Option(None, "--label", help="backend label for the history log"),
    api_key: str | None = typer.Option(None, "--api-key"),
    model: str | None = typer.Option(None, "--model"),
    config: str | None = typer.Option(None, "--config"),
    image: str | None = typer.Option(None, "--image"),
    port: int | None = typer.Option(None, "--port"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
):
    """End-to-end perf bench (gated clean window, raw JSON, history log)."""
    s = _settings(model, config, image, port)
    layout = load_layout()

    def split_ints(v):
        return [int(x) for x in str(v).split(",") if x.strip()]

    try:
        benchy.run_perf(
            s,
            layout,
            runs=runs,
            warmup_runs=warmup_runs,
            exact_tg=exact_tg,
            repeat=repeat,
            depths=split_ints(depths),
            concurrency=split_ints(concurrency),
            pp=pp,
            tg=tg,
            gate=gate,
            confirm=confirm,
            gate_timeout_s=gate_timeout,
            api_key=api_key,
            label=label,
        )
    except benchy.BenchError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    if plot:
        try:
            charts.render_all(layout)
        except RuntimeError as e:
            typer.secho(f"(plots skipped: {e})", fg=typer.colors.YELLOW, err=True)
        try:
            readme.render_readme(layout)
        except (RuntimeError, FileNotFoundError) as e:
            typer.secho(f"(readme skipped: {e})", fg=typer.colors.YELLOW, err=True)


def _ab_legs(
    specs: str, default_model: str, default_image: str, entries: list[ConfigEntry]
) -> list[tuple[str, str, str]]:
    """Parse ``--backends`` into ``(model, image, quant)`` legs.

    Each element is one of:
      * ``<config>``            — the default model's config, served from
                                   ``default_image``;
      * ``<image>:<config>``    — the default model's config pinned to an
                                   image (the last ``:`` splits, so the image
                                   may carry its own registry port / tag);
      * ``<model>/<config>``    — a different model (``vendor/Model``) with
                                   its config, served from ``default_image``;
      * ``<image>:<model>/<config>`` — the same, with the image pinned too
                                   (the last ``:`` splits the image off).
    """
    quants = {e.quant for e in entries if e.model == default_model}
    model_quants = {(e.model, e.quant) for e in entries}
    legs: list[tuple[str, str, str]] = []
    for spec in (b.strip() for b in specs.split(",")):
        if not spec:
            continue
        if ":" in spec:
            image, _, spec = spec.rpartition(":")
            if not image.strip():
                raise ValueError(f"bad leg '{spec}' — expected '<config>', '<model>/<config>' or '<image>:<config>'")
        else:
            image = None
        if "/" in spec:
            model, _, quant = spec.rpartition("/")
            if not model or not quant or model.count("/") != 1:
                raise ValueError(f"bad leg '{spec}' — expected '<config>', '<model>/<config>' or '<image>:<config>'")
            if (model, quant) not in model_quants:
                raise ValueError(f"unknown model config '{model}/{quant}' — available configs: mjolnir model list")
            legs.append((model, image or default_image, quant))
        else:
            if spec not in quants:
                raise ValueError(
                    f"unknown config '{spec}' for model {default_model} — available configs: mjolnir model list"
                )
            legs.append((default_model, image or default_image, spec))
    if not legs:
        raise ValueError("--backends is empty")
    return legs


@bench_app.command("ab")
def bench_ab(
    backends: str = typer.Option(
        legs_.bench_legs("bench.ab"),
        "--backends",
        help="comma-separated legs: each is "
        "'<config>' (the --model/state "
        "model, served from --image), "
        "'<image>:<config>', "
        "'<model>/<config>' or "
        "'<image>:<model>/<config>' — "
        "available: mjolnir model list",
    ),
    depths: str = typer.Option(
        ",".join(str(d) for d in E2E_DEPTHS if d <= _E2E_MAX_CTX),
        "--depths",
        help="context depths (tokens already in the KV); defaults to the 32K e2e scope —"
        " pass deeper values (e.g. 65536,131072) for deliberate deep-cell runs",
    ),
    runs: int = typer.Option(6, "--runs"),
    repeat: int = typer.Option(1, "--repeat"),
    gate: bool = typer.Option(True, "--gate/--no-gate"),
    model: str | None = typer.Option(None, "--model"),
    image: str | None = typer.Option(
        None, "--image", help="image for bare-config legs (default: $MJOLNIR_IMAGE / built-in default)"
    ),
    port: int | None = typer.Option(None, "--port"),
):
    """Headline A/B: restart the server per leg, gated perf bench for each,
    one history row-set per leg.

    Each leg is a model config (quant), optionally model-qualified and/or
    image-pinned: '<config>', '<image>:<config>', '<model>/<config>' or
    '<image>:<model>/<config>'. The image carries the backend (kernel stack
    is baked in); the config (configs/<model>/<quant>/config.yaml, repo or local)
    carries the serving parameters. Bare configs use --model/--image.
    Compare images on one model and/or models on one image. Charts
    re-render at the end."""
    s = _settings(model, None, image, port)
    layout = load_layout()
    entries = scan_all_configs(layout)
    try:
        legs = _ab_legs(backends, s.model, s.image, entries)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4) from e

    for _m, img, q in legs:
        if q != DEFAULT_QUANT and e2e.is_mjolnir({"image": img}):
            typer.secho(
                f"note: {q} on {img} is not native vLLM (the Mjolnir image patches FlashInfer too) — it is kept in "
                f"history but never charted; measure native backends on {STOCK_IMAGE}",
                fg=typer.colors.YELLOW,
            )
    _run_legs(
        s,
        layout,
        [(m, i, q, None) for m, i, q in legs],
        runs=runs,
        repeat=repeat,
        gate=gate,
        depths=[int(x) for x in depths.replace(" ", ",").split(",") if x.strip()],
    )
    _rerender(layout)


@bench_app.command(
    "accuracy",
    context_settings={**_HELP_CTX, "allow_extra_args": True, "ignore_unknown_options": True},
)
def bench_accuracy(
    ctx: typer.Context,
    backends: str = typer.Option(
        legs_.bench_legs("bench.accuracy"),
        "--backends",
        help="comma-separated legs, as in 'bench ab': '<config>', '<image>:<config>', '<model>/<config>' or "
        "'<image>:<model>/<config>' (default: mjolnir_cli.legs)",
    ),
    restart: bool = typer.Option(
        True,
        "--restart/--no-restart",
        help="serve each leg (restarts the server; the GPU must be yours) — --no-restart measures the running "
        "server as one leg (pass its config with --backends)",
    ),
    model: str | None = typer.Option(None, "--model"),
    image: str | None = typer.Option(None, "--image", help="image for bare-config legs"),
    port: int | None = typer.Option(None, "--port"),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the legs and the tool-eval-bench command"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
):
    """Model quality per serving config with tool-eval-bench: does fp6 KV /
    the fp8 lm_head keep the model's answers?

    Default per leg: the 69 standard tool-calling scenarios + the 28 Hard Mode
    ones (`tool-eval-bench run --hardmode --seed 42 --timeout 600`). Anything
    after `--` replaces that and is forwarded verbatim (the leg's --base-url is
    added), e.g. `-- run --hardmode --no-think --trials 3` or
    `-- --gsm8k --mmlu --ifeval --skip-tool-eval --gsm8k-limit 200`.
    Results: benchmarks/raw/accuracy/<run>/accuracy.json (committed; the
    tool's SQLite, reports and the server log stay next to it, untracked);
    docs/kernel/accuracy.md is rendered from the newest result per config and
    suite."""
    from mjolnir_cli import accuracy

    s = _settings(model, None, image, port)
    layout = load_layout()
    entries = scan_all_configs(layout)
    try:
        legs = _ab_legs(backends, s.model, s.image, entries)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4) from e
    passthrough = list(ctx.args)
    if not restart and len(legs) != 1:
        typer.secho("--no-restart measures the running server: give exactly one leg", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    if dry_run:
        try:
            tool = accuracy.find_tool()
        except accuracy.AccuracyError as e:
            tool = [f"<{e}>"]
        for m_, img, q in legs:
            typer.echo(f"{m_}/{q} on {img}:")
            typer.echo("  " + shlex.join(tool + accuracy.tool_args(passthrough, s.base_url)))
        return
    if restart:
        typer.secho(
            "bench accuracy restarts the vLLM server for each leg — the GPU must be yours for the whole run.",
            fg=typer.colors.YELLOW,
        )
    elif not dockerctl.container_running():
        typer.secho("--no-restart: no vLLM server is running (mjolnir serve up)", fg=typer.colors.RED, err=True)
        raise typer.Exit(3)
    try:
        out = accuracy.run_accuracy(s, layout, legs, passthrough, restart=restart)
    except accuracy.AccuracyError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    _print_accuracy(json.loads(out.read_text()))
    typer.secho(f"wrote {out}", fg=typer.colors.GREEN)
    if plot:
        _rerender(layout)


def _print_accuracy(doc: dict) -> None:
    from mjolnir_cli import accuracy

    rows = doc.get("legs", [])
    suites = accuracy.suites_in([{"suites": r.get("suites", {})} for r in rows])
    t = Table(box=box.SIMPLE_HEAD, title=f"accuracy · run {doc['run_id']} · tool-eval-bench {' '.join(doc['args'])}")
    t.add_column("config")
    for n in suites:
        t.add_column(f"{accuracy.suite_label(n)} %", justify="right")
    t.add_column("completion", justify="right")
    bases = accuracy.baselines(rows)  # a run may hold several models: each vs its own
    for r, base in zip(rows, bases, strict=True):
        cells = []
        for n in suites:
            v = (r.get("suites", {}).get(n) or {}).get("score")
            cells.append("—" if v is None else f"{v:.1f}{accuracy.fmt_delta(accuracy.delta(r, base, n))}")
        comp = [x.get("completion_rate") for x in r.get("suites", {}).values() if x.get("completion_rate") is not None]
        t.add_row(r.get("label", r.get("id", "")), *cells, f"{min(comp):.0f}%" if comp else "—")
    if accuracy.any_excluded(rows, bases, suites):
        t.caption = accuracy.EXCLUDED_NOTE
    Console().print(t)


def _rerender(layout) -> None:
    try:
        charts.render_all(layout)
    except RuntimeError as e:
        typer.secho(f"(plots skipped: {e})", fg=typer.colors.YELLOW, err=True)
    try:
        readme.render_readme(layout)
    except (RuntimeError, FileNotFoundError) as e:
        typer.secho(f"(readme skipped: {e})", fg=typer.colors.YELLOW, err=True)


def _run_legs(s: Settings, layout, legs: list[tuple], **perf) -> None:
    """Serve each (model, image, config, label) leg in turn — the server is
    restarted per leg on purpose — run one gated perf bench on it, and stop
    the leg's server when it is done (or fails), so no vLLM stays up holding
    the GPU after the bench."""
    for leg_model, leg_image, quant, label in legs:
        s.model, s.image, s.quant = leg_model, leg_image, quant
        s.label = label
        typer.secho(f"\n{'=' * 62}\n  leg: {leg_model}/{quant}  (image {leg_image})\n{'=' * 62}", fg=typer.colors.CYAN)
        dockerctl.serve_down(s)
        try:
            dockerctl.serve_up(s, wait=True)
            benchy.run_perf(s, layout, warmup_runs=2, exact_tg=True, **perf)
        except benchy.BenchError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from e
        finally:
            typer.secho(f"  leg done: stopping the vLLM server ({leg_image})", fg=typer.colors.CYAN)
            dockerctl.serve_down(s)


def _print_compare(
    layout,
    mjolnir_config: str | None = None,
    as_json: bool = False,
    official_config: str | None = None,
    model: str | None = None,
) -> int:
    off, mj = e2e.latest_pair(layout.history_file, mjolnir_config, official_config, model)
    if not off or not mj:
        missing = "the official vLLM image" if not off else "a Mjolnir image"
        typer.secho(f"no run on {missing} in {layout.history_file} — run: mjolnir bench e2e", fg=typer.colors.YELLOW)
        return 4
    rows = e2e.compare(off, mj)
    if as_json:
        typer.echo(
            json.dumps({"official": e2e.side_label(off), "mjolnir": e2e.side_label(mj), "cells": rows}, indent=2)
        )
        return 0
    con = Console()
    t = Table(
        box=box.SIMPLE_HEAD,
        title="official vLLM image → Mjolnir image (gain: + = Mjolnir better)",
        caption=f"official: {e2e.side_label(off)}\nmjolnir:  {e2e.side_label(mj)}\nboth: {e2e.spec_label(off, mj)}"
        + e2e.acceptance_line(off, mj),
    )
    t.add_column("ctx", justify="right")
    t.add_column("c", justify="right")
    for m in e2e.METRICS:
        t.add_column(
            f"{m.label} {m.unit} {'↑' if m.higher_better else '↓'} ({'throughput' if m.higher_better else 'latency'} change)",
            justify="right",
        )
    for r in rows:
        cells = []
        for m in e2e.METRICS:
            a, b, g = r[m.key]
            if a is None or b is None:
                cells.append("—")
                continue
            color = "green" if g >= e2e.E2E_NOISE_PCT else ("red" if g <= -e2e.E2E_NOISE_PCT else "grey50")
            cells.append(
                f"{m.fmt.format(a)} → {m.fmt.format(b)} [{color}]{e2e.pct(g if m.higher_better else -g)}[/{color}]"
            )
        t.add_row(e2e.ctx_label(r["ctx"]), str(r["conc"]), *cells)
    con.print(t)
    h = e2e.hero(off, mj)
    if h:
        con.print(
            f"[bold]1 request @ {e2e.ctx_label(h['ctx'])}:[/bold] decode {e2e.pct(h['tg_tps'][2])} · "
            f"prefill {e2e.pct(h['pp_tps'][2])} · TTFT {e2e.pct(-h['ttft_ms'][2])} · ITL {e2e.pct(-h['itl_ms'][2])} "
            "(throughput: % more tokens/s; latency: % change in ms)"
        )
    return 0


@bench_app.command("e2e")
def bench_e2e(
    official_image: str = typer.Option(STOCK_IMAGE, "--official-image", help="the upstream vLLM image (baseline)"),
    official_config: str = typer.Option(
        legs_.side_configs("bench.e2e", legs_.OFFICIAL)[0],
        "--official-config",
        help="config served on the official image (stock backends only)",
    ),
    image: str = typer.Option(DEFAULT_IMAGE, "--image", help="the Mjolnir image"),
    config: str = typer.Option(
        legs_.side_configs("bench.e2e", legs_.MJOLNIR)[0], "--config", help="config served on the Mjolnir image"
    ),
    model: str | None = typer.Option(None, "--model"),
    runs: int = typer.Option(6, "--runs", help="measured runs per cell"),
    repeat: int = typer.Option(1, "--repeat", help="independent gated sweeps per leg"),
    depths: str = typer.Option(
        ",".join(str(d) for d in E2E_DEPTHS if d <= _E2E_MAX_CTX),
        "--depths",
        help="context depths (tokens already in the KV); the comparison shows up to 32K",
    ),
    concurrency: str = typer.Option("1,2,4", "--concurrency"),
    pp: int = typer.Option(2048, "--pp", help="prompt tokens per request (prefill)"),
    tg: int = typer.Option(128, "--tg", help="generated tokens per request (decode)"),
    gate: bool = typer.Option(True, "--gate/--no-gate"),
    port: int | None = typer.Option(None, "--port"),
    only: str | None = typer.Option(
        None,
        "--only",
        help="run one leg: 'mjolnir' or 'official'; the comparison pairs it with the other side's latest record",
    ),
    also: str = typer.Option(
        ",".join(legs_.side_configs("bench.e2e", legs_.MJOLNIR)[1:]),
        "--also",
        help="more legs to measure after the pair, comma-separated: a config (Mjolnir image) or "
        "official:<config> / mjolnir:<config>; '' = none. They land in the history (bench compare "
        "--config <cfg>); the README table, charts and infographic stay on --config",
    ),
    optional: bool = typer.Option(
        False,
        "--optional",
        help="also measure the opt-in legs (legs.OPTIONAL: DSpark and DFlash2 on the official and the Mjolnir "
        "image) — hours per leg",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the serve commands, run nothing"),
    allow_stale_image: bool = typer.Option(
        False, "--allow-stale-image", help="benchmark a Mjolnir image older than the checkout's kernel"
    ),
):
    """Official vLLM image vs the Mjolnir image, end to end: decode and prefill
    throughput, time to first token and inter-token latency per context x
    concurrency. Serves each image in turn (the server is restarted per leg),
    benchmarks it with llama-benchy, then prints the comparison and re-renders
    the README (infographic + table)."""
    s = _settings(model, None, None, port)
    layout = load_layout()

    def ints(v: str) -> list[int]:
        return [int(x) for x in v.replace(" ", ",").split(",") if x.strip()]

    legs = [
        (s.model, official_image, official_config, f"vLLM {VLLM_VERSION} (official image)"),
        (s.model, image, config, None),
    ]
    if only:
        side = only.strip().lower()
        if side not in ("mjolnir", "official"):
            typer.secho(f"--only must be 'mjolnir' or 'official', got {only!r}", fg=typer.colors.RED, err=True)
            raise typer.Exit(2)
        legs = [legs[1] if side == "mjolnir" else legs[0]]
    extra = [legs_.parse_leg(q) for q in also.split(",") if q.strip()]
    if optional:
        extra += list(legs_.OPTIONAL["bench.e2e"])
    side_only = only.strip().lower() if only else None
    for side, q in dict.fromkeys(extra):
        if side_only and side != side_only:
            continue
        if side == legs_.OFFICIAL and q != official_config:
            legs.append((s.model, official_image, q, f"vLLM {VLLM_VERSION} (official image)"))
        elif side == legs_.MJOLNIR and q != config:
            legs.append((s.model, image, q, None))
    stale = dockerctl.stale_image(image) if any(leg[1] == image for leg in legs) else None
    if stale and not dry_run:
        if not allow_stale_image:
            typer.secho(f"{stale} (or --allow-stale-image)", fg=typer.colors.RED, err=True)
            raise typer.Exit(2)
        typer.secho(f"warning: {stale}", fg=typer.colors.YELLOW, err=True)
    if dry_run:
        for m_, img, q, _ in legs:
            s.model, s.image, s.quant = m_, img, q
            dockerctl.serve_up(s, wait=False, dry_run=True)
        return
    typer.secho(
        "bench e2e restarts the vLLM server for each image — the GPU must be yours for the whole run.",
        fg=typer.colors.YELLOW,
    )
    _run_legs(
        s,
        layout,
        legs,
        runs=runs,
        repeat=repeat,
        gate=gate,
        depths=ints(depths),
        concurrency=ints(concurrency),
        pp=pp,
        tg=tg,
    )
    _print_compare(layout, config, official_config=official_config, model=s.model)
    measured = [(img, q) for _, img, q, _ in legs if q not in (config, official_config)]
    if measured:
        typer.echo("optional legs — compare them with:")
        offs = [q for img, q in measured if img == official_image]
        for img, q in measured:
            if img != image:
                continue
            twin = next((o for o in offs if o.split("_")[-1] == q.split("_")[-1]), None)
            typer.echo(
                f"  mjolnir bench compare --model {s.model} --config {q}"
                + (f"   |   --official-config {twin} --config {q}" if twin else "")
            )
    _rerender(layout)


@bench_app.command("compare")
def bench_compare(
    config: str | None = typer.Option(None, "--config", help="Mjolnir-side config (default: the Mjolnir backend)"),
    official_config: str | None = typer.Option(
        None, "--official-config", help="official-side config (default: the stock baseline, NVFP4)"
    ),
    as_json: bool = typer.Option(False, "--json", "-j"),
    model: str | None = typer.Option(None, "--model", help="whose runs to pair (default: the active model)"),
):
    """Official vLLM image vs Mjolnir image from the history — no new run.
    `--official-config NVFP4_DSPARK --config NVFP4_MJOLNIR_E2M3_DSPARK` compares
    the same drafter on both images; both sides are runs of one model."""
    model = _settings(model, None, None, None).model
    raise typer.Exit(_print_compare(load_layout(), config, as_json, official_config, model))
