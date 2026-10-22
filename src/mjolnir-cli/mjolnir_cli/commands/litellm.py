"""``mjolnir litellm`` — the LiteLLM proxy stack (proxy + postgres + redis)."""

from __future__ import annotations

import json

import typer

from mjolnir_cli import litellm
from mjolnir_cli.commands._common import HELP_CTX, settings

app = typer.Typer(
    help="The LiteLLM proxy stack (proxy + postgres + redis) — independent "
    "of the vLLM server: 'litellm down' leaves vLLM running, and "
    "'serve down' leaves the proxy up. Reads your template from the "
    "litellm data dir, ~/.local/share/mjolnir/litellm/ (see "
    "docker/litellm/).",
    no_args_is_help=True,
    context_settings=HELP_CTX,
)


@app.command("up")
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
    s = settings(model, config, None, None)
    try:
        litellm.up(s, proxy_port=port, wait=wait, wait_timeout_s=wait_timeout, dry_run=dry_run)
    except (litellm.LitellmError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4 if "not found" in str(e) else 1) from e
    typer.secho("\n  proxy:  mjolnir litellm status", fg=typer.colors.GREEN)


@app.command("down")
def litellm_down(
    purge: bool = typer.Option(False, "--purge", help="also wipe the stack data (postgres + redis)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Stop the stack (the vLLM server keeps running; data kept by
    default, --purge wipes postgres + redis state)."""
    raise typer.Exit(litellm.down(purge=purge, dry_run=dry_run))


@app.command("status")
def litellm_status(port: int | None = typer.Option(None, "--port")):
    """Container state + proxy liveliness + the rendered config path."""
    s = settings(None, None, None, None)
    try:
        st = litellm.status(s, port)
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4) from e
    typer.echo(json.dumps(st, indent=2))


@app.command("logs")
def litellm_logs(follow: bool = typer.Option(False, "--follow", "-f"), tail: int = typer.Option(100, "--tail")):
    """Tail the proxy container logs."""
    raise typer.Exit(litellm.logs(follow=follow, tail=tail))


@app.command("config")
def litellm_config(
    model: str | None = typer.Option(None, "--model"),
    config: str | None = typer.Option(None, "--config"),
    show: bool = typer.Option(False, "--show", help="print the rendered config instead of just validating it"),
):
    """Render the template with the current model/config (no containers) —
    validates the variables and the YAML."""
    s = settings(model, config, None, None)
    try:
        p = litellm.render_config(s)
    except (litellm.LitellmError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(4 if "not found" in str(e) else 1) from e
    typer.secho(f"rendered ok: {p}  (model: {s.served_model})", fg=typer.colors.GREEN)
    if show:
        typer.echo(p.read_text())
