"""The ``mjolnir`` CLI — one command surface for the whole repo.

mjolnir hw setup|status             pre-configure the Thor host (HW
                                    provisioning, idempotent)
mjolnir serve up|down|status|logs    drive the vLLM container
mjolnir model [use|list]            active model/config (bare = arrow-key
                                    picker over configs/)
mjolnir image [use|list|build|gates] active vLLM image (bare = arrow-key
                                    picker over local docker images)
mjolnir verify <stage>              kernel correctness + ncu benchmarks (Mjolnir vs
                                    FlashInfer vs FA2) -> benchmarks/raw/verify
mjolnir autoresearch ...            the autonomous kernel-research loop (OpenCode
                                    agents, every experiment on the ledger)
mjolnir gate                        the clean-window gate
mjolnir bench perf|ab               end-to-end perf / A/B (vLLM server)
mjolnir plot|history                charts (+ README.md with --readme) + the raw-data log
mjolnir litellm up|down|status|logs  the LiteLLM proxy stack (independent
                                    of the vLLM server)
mjolnir perses up|down|status|logs   the Perses + Prometheus observability
                                    stack (independent of the vLLM server)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import typer
from rich import box
from rich.style import Style
from rich.table import Table
from typer.core import TyperGroup

from mjolnir_cli import __version__
from mjolnir_cli.config import (
    BACKEND_LABELS,
    DEFAULT_IMAGE,
    ConfigEntry,
    load_layout,
    configs_root,
    scan_all_configs,
    user_configs_dir,
)
from mjolnir_cli import cachectl, charts, dockerctl, litellm, perses, picker, readme, tasks, thor
from mjolnir_cli.cli_platform import platform_app
from mjolnir_cli.verify.cli import app as verify_app
from mjolnir_cli.autoresearch import app as autoresearch_app
from mjolnir_cli.cli_bench import app as bench_app

from mjolnir_cli.cli_shared import _HELP_CTX, _console, _load_state, _save_state, _settings, _state_file
from mjolnir_cli.platforms import UnknownPlatform, resolve_platform


def _picker_group(bare_fn) -> type[TyperGroup]:
    """A group that, when invoked bare (no subcommand), runs ``bare_fn``
    (the arrow-key picker) instead of erroring out."""

    class PickerGroup(TyperGroup):
        def invoke(self, ctx):
            if not ctx._protected_args:
                bare_fn()
                return None
            return super().invoke(ctx)

    return PickerGroup


_NEXT_STEPS = """
Next steps:
  mjolnir serve up         serve the model (baked-in defaults: Qwen3.8-27B NVFP4,
                           --attention-backend MJOLNIR, port 6001)
  mjolnir serve status     container + health + request queue
  mjolnir verify sweep     Mjolnir vs FlashInfer vs FA2 at every L (ncu) -> raw data
  mjolnir verify all       correctness + every benchmark stage -> charts + README
  mjolnir bench ab         end-to-end A/B: Mjolnir vs FlashInfer vs FA2 configs
  mjolnir model / image    pick another model config / serving image
"""


def _root_group() -> type[TyperGroup]:
    """The root group: bare ``mjolnir`` prints the help plus a next-steps
    guide instead of just the help."""

    class RootGroup(TyperGroup):
        def invoke(self, ctx):
            if not ctx._protected_args:
                ctx.command.get_help(ctx)
                print(_NEXT_STEPS, flush=True)
                return None
            return super().invoke(ctx)

    return RootGroup


serve_app = typer.Typer(
    help="Drive the vLLM container (the patched image).", no_args_is_help=True, context_settings=_HELP_CTX
)
model_app = typer.Typer(
    help="Pick the active model / config.", context_settings=_HELP_CTX, cls=_picker_group(lambda: model_use(None, None))
)
image_app = typer.Typer(
    help="Pick the active vLLM image / build the patched one.",
    context_settings=_HELP_CTX,
    cls=_picker_group(lambda: image_use(None)),
)
litellm_app = typer.Typer(
    help="The LiteLLM proxy stack (proxy + postgres + redis) — independent "
    "of the vLLM server: 'litellm down' leaves vLLM running, and "
    "'serve down' leaves the proxy up. Reads your template from the "
    "litellm data dir, ~/.local/share/mjolnir/litellm/ (see "
    "docker/litellm/).",
    no_args_is_help=True,
    context_settings=_HELP_CTX,
)
perses_app = typer.Typer(
    help="The Perses + Prometheus observability stack for the vLLM server — "
    "independent of the vLLM server: 'perses down' leaves the server "
    "running, and 'serve down' leaves the stack up. Prometheus scrapes "
    "the server's /metrics (host.docker.internal + the resolved port); "
    "Perses serves the two vLLM dashboards (vendored from "
    "vllm-project/vllm, docker/perses/). Data: "
    "~/.local/share/mjolnir/perses/.",
    no_args_is_help=True,
    context_settings=_HELP_CTX,
)
hw_app = typer.Typer(
    help="Thor host pre-configuration — one-time, idempotent provisioning "
    "(sudo where needed): headless boot target, apt upgrade, Docker + "
    "NVIDIA default runtime, pip, jtop, swap, fan profiles "
    "(recommended + max installed, default selected), service "
    "cleanup, locked clocks, MAXN power mode. Run 'hw setup' before "
    "serving or benchmarking.",
    no_args_is_help=True,
    context_settings=_HELP_CTX,
)

app = typer.Typer(
    context_settings=_HELP_CTX,
    cls=_root_group(),
    help="Mjolnir — a decode-attention kernel for Jetson AGX Thor (sm_110) and the "
    "vLLM image that serves it: serve / verify / bench / plot.",
)
app.add_typer(serve_app, name="serve")
app.add_typer(bench_app, name="bench")
app.add_typer(model_app, name="model")
app.add_typer(image_app, name="image")
app.add_typer(platform_app, name="platform")
app.add_typer(verify_app, name="verify")
app.add_typer(autoresearch_app, name="autoresearch")
app.add_typer(litellm_app, name="litellm")
app.add_typer(perses_app, name="perses")
app.add_typer(hw_app, name="hw")


# ── serve ───────────────────────────────────────────────────────────────────


def _local_images() -> list[str]:
    """Every local docker image (``repo:tag``), deduped, order preserved."""
    try:
        p = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    out: list[str] = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if line and line != "<none>:<none>" and line not in out:
            out.append(line)
    return out


def _active_image(st: dict) -> str:
    return st.get("image") or _settings(None, None, None, None).image


@image_app.command("use")
def image_use(image: str | None = typer.Argument(None, help="docker image reference (omit to pick interactively)")):
    """Remember the active vLLM image (bare ``mjolnir image`` or no args
    here = arrow-key picker over local docker images).

    Persisted to the state file ($MJOLNIR_STATE, default
    ~/.mjolnir-state.json) — serve/bench pick it up from there."""
    st = _load_state()
    imgs = _local_images()
    if not imgs:
        typer.secho("no local docker images found — build or pull one first", fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    if image:
        if image not in imgs:
            typer.secho(
                f"image '{image}' not found locally ({len(imgs)} image(s) available: mjolnir image list)",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(4)
        chosen = image
    else:
        if not sys.stdin.isatty():
            for i in imgs:
                typer.echo(f"  {'●' if i == _active_image(st) else ' '}{i}")
            typer.secho("non-interactive session — pass the image explicitly", fg=typer.colors.RED, err=True)
            raise typer.Exit(4)
        try:
            chosen = picker.select("Pick the vLLM image:", imgs, default=_active_image(st))
        except picker.Cancelled:
            typer.secho("cancelled — active image unchanged", fg=typer.colors.YELLOW, err=True)
            raise typer.Exit(1) from None
    _save_state({"image": chosen})
    typer.secho(f"active image: {chosen}", fg=typer.colors.GREEN)
    typer.echo(f"  state: {_state_file()}")


@image_app.command("list")
def image_list():
    """List local docker images (● = the active one)."""
    st = _load_state()
    active = _active_image(st)
    for i in _local_images():
        if i == active:
            typer.secho(f"  ● {i}  (active)", fg=typer.colors.GREEN)
        else:
            typer.echo(f"    {i}")


@image_app.command("build")
def image_build(
    tag: str | None = typer.Option(
        None,
        "--tag",
        help=f"image tag (default: {DEFAULT_IMAGE} — the repo + the vLLM version the Dockerfile builds on)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Build the Mjolnir vLLM image: the patch stack + the mjolnir-kernel
    package from this checkout (repo-root build context).

    The default tag is always the canonical one (IMAGE_REPO:v<vLLM version>),
    never the remembered active image — that may be an older build."""
    tag = tag or DEFAULT_IMAGE
    try:
        root = load_layout().require_repo("image build")
    except FileNotFoundError as e:
        typer.secho(f"{e} — or pull the published image: docker pull {DEFAULT_IMAGE}", fg=typer.colors.RED, err=True)
        raise typer.Exit(4) from e
    rev = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    cmd = ["docker", "build", "-f", str(root / "docker" / "vllm-thor" / "Dockerfile"), "-t", tag]
    if rev.returncode == 0 and rev.stdout.strip():
        cmd += ["--build-arg", f"MJOLNIR_REVISION={rev.stdout.strip()}"]
    cmd.append(str(root))
    if dry_run:
        print(" ".join(cmd))
        return
    rc = subprocess.call(cmd)
    if rc == 0:
        typer.secho(f"built {tag} — canary:  mjolnir image gates --image {tag}", fg=typer.colors.GREEN)
        active = _settings(None, None, None, None).image
        if active != tag:
            typer.secho(
                f"note: the active image is {active} — make the new build active:  mjolnir image use {tag}",
                fg=typer.colors.YELLOW,
            )
    raise typer.Exit(rc)


@image_app.command("gates")
def image_gates(image: str | None = typer.Option(None, "--image"), dry_run: bool = typer.Option(False, "--dry-run")):
    """Run the sm_110 gate-probe canaries against the image (fresh container,
    no server needed)."""
    s = _settings(None, None, image, None)
    raise typer.Exit(tasks.run_task(s, "gates", [], dry_run=dry_run))


@litellm_app.command("up")
def litellm_up(
    model: str | None = typer.Option(None, "--model"),
    config: str | None = typer.Option(None, "--config", help="config quant name, e.g. NVFP4_FA4hd256"),
    port: int | None = typer.Option(
        None, "--port", help=f"host port of the proxy (default {litellm.DEFAULT_PROXY_PORT})"
    ),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="wait for the proxy's liveliness before returning"),
    wait_timeout: float = typer.Option(600.0, "--wait-timeout"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Render the config from your template (in the litellm data dir,
    ~/.local/share/mjolnir/litellm/) with the currently selected
    model/config, and start the stack (proxy + postgres + redis). The vLLM
    server is reached via the host — it may be up or down."""
    s = _settings(model, config, None, None)
    try:
        litellm.up(s, proxy_port=port, wait=wait, wait_timeout_s=wait_timeout, dry_run=dry_run)
    except (litellm.LitellmError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4 if "not found" in str(e) else 1) from e
    typer.secho("\n  proxy:  mjolnir litellm status", fg=typer.colors.GREEN)


@litellm_app.command("down")
def litellm_down(
    purge: bool = typer.Option(False, "--purge", help="also wipe the stack data (postgres + redis)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Stop the stack (the vLLM server keeps running; data kept by
    default, --purge wipes postgres + redis state)."""
    raise typer.Exit(litellm.down(purge=purge, dry_run=dry_run))


@litellm_app.command("status")
def litellm_status(port: int | None = typer.Option(None, "--port")):
    """Container state + proxy liveliness + the rendered config path."""
    s = _settings(None, None, None, None)
    try:
        st = litellm.status(s, port)
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4) from e
    typer.echo(json.dumps(st, indent=2))


@litellm_app.command("logs")
def litellm_logs(follow: bool = typer.Option(False, "--follow", "-f"), tail: int = typer.Option(100, "--tail")):
    """Tail the proxy container logs."""
    raise typer.Exit(litellm.logs(follow=follow, tail=tail))


@litellm_app.command("config")
def litellm_config(
    model: str | None = typer.Option(None, "--model"),
    config: str | None = typer.Option(None, "--config"),
    show: bool = typer.Option(False, "--show", help="print the rendered config instead of just validating it"),
):
    """Render the template with the current model/config (no containers) —
    validates the variables and the YAML."""
    s = _settings(model, config, None, None)
    try:
        p = litellm.render_config(s)
    except (litellm.LitellmError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4 if "not found" in str(e) else 1) from e
    typer.secho(f"rendered ok: {p}  (model: {s.served_model})", fg=typer.colors.GREEN)
    if show:
        typer.echo(p.read_text())


@perses_app.command("up")
def perses_up(
    port: int | None = typer.Option(None, "--port", help=f"host port of the Perses UI (default {perses.DEFAULT_PORT})"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="wait for the Perses API before returning"),
    wait_timeout: float = typer.Option(300.0, "--wait-timeout"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Start the observability stack (prometheus + perses) with the two vLLM
    dashboards provisioned. The vLLM scrape target = the resolved server port
    (state/$MJOLNIR_PORT — the same one `mjolnir serve` uses; override with
    PERSER_VLLM_TARGET in the perses env file). The server may be up or down."""
    s = _settings(None, None, None, None)
    try:
        perses.up(s, port=port, wait=wait, wait_timeout_s=wait_timeout, dry_run=dry_run)
    except (perses.PersesError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4 if "not found" in str(e) else 1) from e
    typer.secho(
        f"\n  dashboards:  mjolnir perses status  (UI: http://127.0.0.1:{port or perses.DEFAULT_PORT})",
        fg=typer.colors.GREEN,
    )


@perses_app.command("down")
def perses_down(
    purge: bool = typer.Option(False, "--purge", help="also wipe the stack data (prometheus TSDB + perses db)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Stop the stack (the vLLM server keeps running; data kept by default,
    --purge wipes the prometheus TSDB + the perses db)."""
    raise typer.Exit(perses.down(purge=purge, dry_run=dry_run))


@perses_app.command("status")
def perses_status(port: int | None = typer.Option(None, "--port")):
    """Container state + API health + the vLLM scrape target + the data dir."""
    s = _settings(None, None, None, None)
    typer.echo(json.dumps(perses.status(s, port), indent=2))


@perses_app.command("logs")
def perses_logs(
    follow: bool = typer.Option(False, "--follow", "-f"),
    tail: int = typer.Option(100, "--tail"),
    service: str | None = typer.Option(None, "--service", help="perses (default) | prometheus"),
):
    """Tail a stack service's logs (default: perses)."""
    raise typer.Exit(perses.logs(follow=follow, tail=tail, service=service))


@hw_app.command("setup")
def hw_setup(
    yes: bool = typer.Option(False, "--yes", "-y", help="don't prompt to continue"),
    upgrade: bool = typer.Option(
        True,
        "--upgrade/--no-upgrade",
        help="apt update + full-upgrade (default on — the original script's NVIDIA-carrier path)",
    ),
    keep_gui: bool = typer.Option(
        False, "--keep-gui", help="keep the graphical boot target (default: headless multi-user)"
    ),
    fan_profile: str = typer.Option(
        "recommended",
        "--fan-profile",
        help="recommended (default) | max | cool | quiet — recommended/max are installed by this step if missing",
    ),
    swap_size: int = typer.Option(32, "--swap-size", help="swap file size in GB (default 32 — the 128 GB Thor SoM)"),
    skip: str = typer.Option(
        "", "--skip", help="comma-separated steps to skip: gui,upgrade,pip,docker,jtop,memory,fan,host,clocks,power"
    ),
    reboot: bool = typer.Option(False, "--reboot", help="reboot at the end if a step required one (non-interactive)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the plan, change nothing"),
):
    """Pre-configure the Thor host (port of the ~/thor HW setup script).

    Runs ``scripts/hw/setup-thor.sh``: headless boot target, apt upgrade,
    Docker + NVIDIA default runtime, pip, jtop, 32 GB swap (zRAM off),
    the fan profiles (``recommended`` + ``max`` installed per-profile
    gated, with the .bck rollback backup; default selected —
    ``recommended`` by default), service cleanup, locked max clocks,
    MAXN power mode. Idempotent — safe to re-run; the script asks for
    sudo."""
    try:
        rc = thor.dispatch(
            resolve_platform(None),
            yes=yes,
            upgrade=upgrade,
            keep_gui=keep_gui,
            fan_profile=fan_profile,
            swap_size=swap_size,
            skip=skip,
            reboot=reboot,
            dry_run=dry_run,
        )
    except UnknownPlatform as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None
    except (ValueError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4) from e
    raise typer.Exit(rc)


@hw_app.command("status")
def hw_status():
    """Installed fan profiles and the selected default (read-only, no sudo).

    The daemon's live state (which profile it is actually running) needs
    sudo — check it with `sudo nvfancontrol -q`."""
    conf = Path(os.environ.get("MJOLNIR_NVFANCONF", "/etc/nvfancontrol.conf"))
    if not conf.exists():
        typer.secho(f"no conf at {conf}", fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    text = conf.read_text()
    profiles = re.findall(r"^[ \t]*FAN_PROFILE\s+(\w+)", text, re.MULTILINE)
    m = re.search(r"^[ \t]*FAN_DEFAULT_PROFILE\s+(\S+)", text, re.MULTILINE)
    bck = conf.with_name(conf.name + ".bck")
    print(f"conf:     {conf}")
    print(f"profiles: {', '.join(profiles)}")
    print(f"default:  {m.group(1) if m else '<unset>'}")
    print(f"backup:   {bck}" + (" (exists)" if bck.exists() else " (missing)"))


@serve_app.command("up")
def serve_up(
    model: str | None = typer.Option(None, "--model"),
    config: str | None = typer.Option(None, "--config", help="config quant name, e.g. NVFP4_FA4hd256"),
    image: str | None = typer.Option(None, "--image"),
    port: int | None = typer.Option(None, "--port"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="wait for /health before returning"),
    wait_timeout: float = typer.Option(1800.0, "--wait-timeout"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Start the vLLM server (custom image, config from --model/--config)."""
    s = _settings(model, config, image, port)
    if not s.config_path.exists():
        typer.secho(f"config not found: {s.config_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    stale = dockerctl.stale_image(s.image)
    if stale:
        typer.secho(f"warning: {stale}", fg=typer.colors.YELLOW, err=True)
    try:
        dockerctl.serve_up(s, wait=wait, wait_timeout_s=wait_timeout, dry_run=dry_run)
    except (dockerctl.DockerError, TimeoutError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    typer.secho(f"\n  serving {s.model} ({s.quant}, {s.backend_label}) at {s.base_url}", fg=typer.colors.GREEN)
    typer.echo("  gate:  mjolnir gate --once")
    typer.echo("  bench: mjolnir bench perf   ·   kernel: mjolnir verify sweep")


@serve_app.command("down")
def serve_down(
    model: str | None = typer.Option(None, "--model"),
    config: str | None = typer.Option(None, "--config"),
    image: str | None = typer.Option(None, "--image"),
    port: int | None = typer.Option(None, "--port"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Stop the vLLM container."""
    s = _settings(model, config, image, port)
    dockerctl.serve_down(s, dry_run=dry_run)


@serve_app.command("status")
def serve_status(port: int | None = typer.Option(None, "--port")):
    """Container state + /health + current queue load."""
    s = _settings(None, None, None, port)
    st = dockerctl.status(s)
    typer.echo(json.dumps(st, indent=2))
    from mjolnir_cli.gate import server_load

    load = server_load(s.metrics_url)
    if load is not None:
        typer.echo(f"  queue: running={load[0]:g} waiting={load[1]:g}")
    pport = perses.running_port()
    if pport:
        typer.echo(f"  dashboard: http://127.0.0.1:{pport} (mjolnir perses status)")


@serve_app.command("logs")
def serve_logs(
    port: int | None = typer.Option(None, "--port"),
    follow: bool = typer.Option(False, "--follow", "-f"),
    tail: int = typer.Option(100, "--tail"),
):
    """Tail the vLLM container logs."""
    s = _settings(None, None, None, port)
    raise typer.Exit(dockerctl.logs(s, follow=follow, tail=tail))


# ── model ───────────────────────────────────────────────────────────────────


def _config_table(entries: list[ConfigEntry], active_model, active_quant) -> Table:
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    t.add_column("", justify="center")
    t.add_column("model", style="bold cyan")
    t.add_column("quant", style="green")
    t.add_column("backend")
    t.add_column("where")
    for e in entries:
        active = e.model == active_model and e.quant == active_quant
        t.add_row(
            "[green]●[/green]" if active else "",
            e.model,
            e.quant,
            e.backend,
            e.source,
            style=Style(bold=True) if active else None,
        )
    return t


def _config_labels(entries: list[ConfigEntry]) -> list[str]:
    return [f"{e.model} / {e.quant}" for e in entries]


def _pick_config(
    entries: list[ConfigEntry],
    st: dict,
    candidates: list[ConfigEntry] | None = None,
    title: str = "Pick the active model / config:",
) -> ConfigEntry:
    """Arrow-key pick over the candidate configs (default: the active one)."""
    pool = candidates or entries
    labels = _config_labels(pool)
    cur = st.get("model") and st.get("quant")
    default = f"{cur[0]} / {cur[1]}" if cur and f"{cur[0]} / {cur[1]}" in labels else None
    if not sys.stdin.isatty():
        _console.print(_config_table(entries, st.get("model"), st.get("quant")))
        typer.secho("non-interactive session — pass model + quant explicitly", fg=typer.colors.RED, err=True)
        raise typer.Exit(4)
    try:
        pick = picker.select(title, labels, default=default)
    except picker.Cancelled:
        typer.secho("cancelled — active config unchanged", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(1) from None
    return pool[labels.index(pick)]


@model_app.command("use")
def model_use(
    model: str | None = typer.Argument(None, help="vendor/Model, e.g. Qwen3.8-27B (omit to pick interactively)"),
    quant: str | None = typer.Argument(None, help="config name, e.g. NVFP4_FA4hd256 (omit to pick interactively)"),
):
    """Remember the active model/config (bare ``mjolnir model`` or no args
    here = arrow-key picker over configs: repo configs/ + user-local
    ~/.local/share/mjolnir/configs).

    Persisted to a small state file ($MJOLNIR_STATE, default
    ~/.mjolnir-state.json) — not an env file."""
    layout = load_layout()
    entries = scan_all_configs(layout)
    if not entries:
        typer.secho(
            f"no configs found under {configs_root()} or {user_configs_dir()}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(4)
    st = _load_state()

    if model and quant:
        if not any(e.model == model and e.quant == quant for e in entries):
            typer.secho(
                f"config not found: {model}/{quant} (available: mjolnir model list)", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(4)
        chosen = next(e for e in entries if e.model == model and e.quant == quant)
    elif model or quant:
        candidates = [e for e in entries if (not model or e.model == model) and (not quant or e.quant == quant)]
        if not candidates:
            typer.secho(
                "no config matches the given model/quant (available: mjolnir model list)", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(4)
        chosen = candidates[0] if len(candidates) == 1 else _pick_config(entries, st, candidates)
    else:
        chosen = _pick_config(entries, st)

    _save_state({"model": chosen.model, "quant": chosen.quant})
    typer.secho(
        f"active: {chosen.model} / {chosen.quant}  (→ {BACKEND_LABELS.get(chosen.quant, chosen.quant)})",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"  state: {_state_file()}")


@model_app.command("list")
def model_list(as_json: bool = typer.Option(False, "--json", "-j", help="emit JSON (for scripts)")):
    """List every config (repo configs/ + the user-local dir; ● = the active
    one; the state file is printed below the table)."""
    layout = load_layout()
    entries = scan_all_configs(layout)
    st = _load_state()
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "active": {"model": st.get("model"), "quant": st.get("quant")},
                    "state_file": str(_state_file()),
                    "local_configs_dir": str(user_configs_dir()),
                    "configs": [
                        {
                            "model": e.model,
                            "quant": e.quant,
                            "backend": e.backend,
                            "source": e.source,
                            "path": str(e.path),
                        }
                        for e in entries
                    ],
                },
                indent=2,
            )
        )
        return
    if not entries:
        typer.secho("no configs found", fg=typer.colors.YELLOW)
        return
    _console.print(_config_table(entries, st.get("model"), st.get("quant")))
    if any(e.source == "local" for e in entries):
        typer.echo(f"    local = {user_configs_dir()}")
    if st.get("model"):
        typer.secho(f"  ● active: {st['model']} / {st['quant']}", fg=typer.colors.GREEN)
        typer.echo(f"    state: {_state_file()}  (change: mjolnir model)")
    else:
        typer.echo("  no active config yet — run 'mjolnir model' to pick one")


# ── image ───────────────────────────────────────────────────────────────────


# ── litellm ─────────────────────────────────────────────────────────────────


# ── perses ──────────────────────────────────────────────────────────────────


# ── hw ─────────────────────────────────────────────────────────────────────


# ── gate ────────────────────────────────────────────────────────────────────


@app.command()
def gate(
    port: int | None = typer.Option(None, "--port"),
    once: bool = typer.Option(False, "--once", help="check now and exit"),
    confirm: int = typer.Option(6, "--confirm"),
    timeout: float = typer.Option(3600.0, "--timeout"),
):
    """The clean-window gate (vLLM request queue 0/0 × N consecutive samples)."""
    s = _settings(None, None, None, port)
    from mjolnir_cli.tasks import _run_gate

    args = ["--once"] if once else ["--confirm", str(confirm), "--timeout", str(timeout)]
    raise typer.Exit(_run_gate(s, args, False))


# ── calibrate (fp6 KV scales) ───────────────────────────────────────────────


@app.command()
def calibrate(
    model: str | None = typer.Option(None, "--model", help="vendor/Model (default: the active one)"),
    config: str | None = typer.Option(None, "--config", help="the config to calibrate (default: the active one)"),
    image: str | None = typer.Option(None, "--image", help="the Mjolnir image"),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="default: the model's kv_scales/<format>.json next to the config dirs — kv_scales/<drafter>.kv_scales.json "
        "for a config whose own drafter (DSpark) runs on Mjolnir (the legacy single file is written too, one "
        "compat cycle)",
    ),
    prompts: int = typer.Option(48, "--prompts", help="short prompts"),
    long: int = typer.Option(6, "--long", help="long-context prompts"),
    max_model_len: int = typer.Option(32768, "--max-model-len"),
    draft_vocab: bool = typer.Option(
        False,
        "--draft-vocab",
        help="instead: count the target's generated tokens -> draft_vocab.json (VocabTrim, MJOLNIR_DRAFT_VOCAB)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the docker command, run nothing"),
):
    """Calibrate per-(layer, KV head, channel) K/V scales for the fp6 (e2m3) KV cache
    (or, with --draft-vocab, the drafter's vocabulary for VocabTrim).

    Runs the config's model on fp8 KV in the Mjolnir image, records the
    per-channel amax of every K/V written to the cache over a varied prompt set,
    and writes the per-format ``kv_scales/<format>.json`` next to the config —
    what an ``kv-cache-dtype: fp6_e2m3`` config is served with (plus the
    pre-migration single file, one compat cycle). Needs the GPU to
    itself: refuses while the vLLM server is up (stop it with
    ``mjolnir serve down`` in a window you own)."""
    s = _settings(model, config, image, None)
    if not dry_run and dockerctl.container_running():
        typer.secho(
            "the vLLM server is running — calibration loads the model itself. Stop the server "
            "first (mjolnir serve down), in a window you own.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(3)
    target, legacy = dockerctl.calibrate_targets(s, out, draft_vocab)
    target.parent.mkdir(parents=True, exist_ok=True)
    mounts = [(str(target.parent.resolve()), "/out")]
    env = []
    try:  # a checkout's kernel package over the image's (calibrate.py may be newer)
        from mjolnir_cli.config import find_repo_root

        src = find_repo_root() / "src" / "mjolnir"
        if (src / "mjolnir" / ("draftvocab.py" if draft_vocab else "calibrate.py")).exists():
            mounts.append((str(src), "/opt/mjolnir-src:ro"))
            env.append("PYTHONPATH=/opt/mjolnir-src")
    except FileNotFoundError:
        pass
    cmd = (
        f"python3 -m mjolnir.{'draftvocab' if draft_vocab else 'calibrate'} "
        f"--config {dockerctl.config_model_dir(s)}/{s.quant}/config.yaml --out /out/{target.name} "
        f"--download-dir /data/models/huggingface --max-model-len {max_model_len} "
        f"--prompts {prompts} --long {long}"
    )
    argv = dockerctl.build_run_args(
        s,
        name="mjolnir-calibrate",
        detach=False,
        command=cmd,
        extra_env=tuple(env),
        extra_mounts=tuple(mounts),
        chown=(f"/out/{target.name}",),
    )
    if dry_run:
        typer.echo(" ".join(argv))
        return
    typer.secho(f"calibrating {s.model}/{s.quant} on {s.image} -> {target}", fg=typer.colors.CYAN)
    rc = dockerctl.run_once(argv, "mjolnir-calibrate")
    if rc == 0:
        if legacy is not None:  # compat window: the legacy file feeds older wheels
            dockerctl.mirror_legacy_scales(target, legacy)
        typer.secho(f"wrote {target}", fg=typer.colors.GREEN)
    raise typer.Exit(rc)


# ── permissions / cache ─────────────────────────────────────────────────────


@app.command("grant-permissions")
def grant_permissions(
    yes: bool = typer.Option(False, "--yes", "-y", help="don't prompt to confirm the install"),
):
    """Check for — and, if missing, install — the NOPASSWD sudo grant the
    kernel benches use to clear the page cache.

    Idempotent: a present grant is left untouched. The install adds only the
    cache-clear command (``sysctl -w vm.drop_caches=3``) — never full sudo —
    and asks for the sudo password exactly once."""
    raise typer.Exit(cachectl.grant_drop_caches(assume_yes=yes))


@app.command("drop-caches")
def drop_caches():
    """Clear the host page cache (``sysctl -w vm.drop_caches=3``) for a clean
    kernel-bench baseline. Uses the NOPASSWD grant (run
    ``mjolnir grant-permissions`` if it's missing); never prompts."""
    if not cachectl.drop_caches_granted():
        typer.secho(
            "drop-caches grant missing — run: mjolnir grant-permissions",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(4)
    rc = cachectl.drop_caches()
    if rc == 0:
        typer.secho("page cache cleared (vm.drop_caches=3)", fg=typer.colors.GREEN)
    else:
        typer.secho(f"drop_caches failed (exit {rc})", fg=typer.colors.RED, err=True)
    raise typer.Exit(rc)


# ── bench ───────────────────────────────────────────────────────────────────


# ── plot / history ──────────────────────────────────────────────────────────


@app.command()
def plot(
    render_readme: bool = typer.Option(
        False,
        "--readme",
        help="also render README.md from the README.md.j2 template "
        "(every table + number computed from the committed raw "
        "data)",
    ),
    readme_template: Path | None = typer.Option(
        None, "--readme-template", help="template override (default: <repo>/README.md.j2)"
    ),
    readme_out: Path | None = typer.Option(None, "--readme-out", help="output override (default: <repo>/README.md)"),
    site: Path | None = typer.Option(
        None, "--site", help="also build the interactive benchmark site (index.html) into this directory"
    ),
):
    """Render every chart (dark-only PNGs in assets/benchmarks/, from the
    latest default `mjolnir verify` run of each stage + the e2e history).
    --readme also renders README.md and docs/**/*.md from their .md.j2
    templates; --site builds the interactive GitHub Pages site. Everything is
    computed from the committed raw data."""
    layout = load_layout()
    try:
        charts.render_all(layout)
    except RuntimeError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e
    if layout.repo_root:  # research/progress.png: the live ledger, else the newest exported run
        from mjolnir_cli import autoresearch

        autoresearch._render_progress(layout.repo_root, autoresearch.latest_rows(layout.repo_root))
    if render_readme:
        try:
            readme.render_readme(layout, template=readme_template, out=readme_out)
        except (RuntimeError, FileNotFoundError, OSError) as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from e
    if site is not None:
        from mjolnir_cli import interactive

        interactive.build(layout, site)


def _ctx_label(ctx: int | None) -> str:
    if not ctx:
        return "0"
    return f"{ctx // 1024}K" if ctx % 1024 == 0 else str(ctx)


def _history_record_table(r: dict) -> Table:
    """One history record → a small tg-t/s (decode throughput) table."""
    by: dict[int | None, dict] = {}
    for c in r.get("cells", []):
        tg = (c.get("tg_tps") or {}).get("mean")
        if tg is None:
            continue
        by.setdefault(c.get("context"), {})[c.get("concurrency")] = tg
    concs = sorted({cc for v in by.values() for cc in v}, key=lambda x: (x is None, x or 0))
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    t.add_column("ctx")
    for cc in concs:
        t.add_column(f"c={cc} · t/s")
    for ctx in sorted(by, key=lambda x: (x is None, x or 0)):
        t.add_row(_ctx_label(ctx), *[f"{by[ctx][cc]:.1f}" if cc in by[ctx] else "—" for cc in concs])
    return t


@app.command()
def history(
    limit: int = typer.Option(5, "--limit", help="how many recent runs to show"),
    as_json: bool = typer.Option(False, "--json", "-j", help="emit the raw JSON records (for scripts)"),
):
    """Bench results at a glance: decode throughput (t/s) per recent run.

    The raw substrate (every sweep, every cell) is committed in
    benchmarks/history.jsonl — ``--json`` dumps the records."""
    layout = load_layout()
    from mjolnir_cli.history import load_records

    recs = load_records(layout.history_file)[-limit:]
    if not recs:
        typer.secho("no history yet — run: mjolnir bench perf", fg=typer.colors.YELLOW)
        return
    if as_json:
        typer.echo(json.dumps(recs, indent=2))
        return
    for r in recs:
        ts = str(r.get("ts", ""))[:16].replace("T", " ")
        _console.print(f"  {ts}   {r.get('backend', '')} — {r.get('image', '')}")
        _console.print(_history_record_table(r))
        _console.print()


@app.command()
def version():
    """Print the version."""
    typer.echo(f"mjolnir {__version__}")


def main() -> None:
    """The console-script entry point: a host that detection cannot name must
    fail every resolving command with the probe instructions (exit 2), never
    a raw traceback (review cycle 2, IMPROVEMENT 3). ``platform show`` and
    ``hw`` keep their own handlers and exits; a bad flag/env value keeps its
    ``ValueError``."""
    try:
        app()
    except UnknownPlatform as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    sys.exit(main())
