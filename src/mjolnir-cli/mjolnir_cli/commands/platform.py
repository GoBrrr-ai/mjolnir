"""``mjolnir platform`` — the active platform: list the registry, show the
resolved one with its detection provenance. Read-only: the platform is
detected (flag > env > host), never remembered — mounted next to
``model``/``image`` in ``cli.py``."""

from __future__ import annotations

import typer

from mjolnir_cli import platforms
from mjolnir_cli.commands._common import HELP_CTX

app = typer.Typer(
    help="The active platform (thor supported; spark / orin research): list, show.",
    no_args_is_help=True,
    context_settings=HELP_CTX,
)


@app.command("list")
def platform_list() -> None:
    """Every shipped platform: architecture, images, maturity."""
    for p in platforms.REGISTRY.values():
        typer.echo(f"  {p.key:<7} {p.name:<16} {p.sm_family:<7} {p.status:<10} {p.default_image}")


@app.command("show")
def platform_show() -> None:
    """The active platform (flag > env `MJOLNIR_PLATFORM` > detection), its
    detection confidence and one provenance line per source that probed."""
    try:
        d = platforms.detect()
    except (ValueError, platforms.UnknownPlatform) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    p = platforms.get(d.platform)
    typer.echo(f"platform: {p.key} — {p.name} ({p.sm_family}, {p.status})")
    typer.echo(f"confidence: {d.confidence}")
    for source, raw in d.provenance:
        typer.echo(f"  {source}: {raw}")
    if not d.provenance:
        typer.echo(f"  (no probes ran — {d.confidence})")
    # the same source instances detect() used (or would have): one nvidia-smi
    # spawn for the whole command
    typer.echo(f"gpu: {platforms.gpu_name(platforms.DEFAULT_SOURCES) or '(not detected on this host)'}")
    typer.echo(f"image: {p.default_image}")
