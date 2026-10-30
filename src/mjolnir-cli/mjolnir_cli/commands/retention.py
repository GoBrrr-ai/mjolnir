"""``mjolnir gc`` — run the retention sweep on demand (it already runs at the
end of every ``mjolnir verify`` stage and ``mjolnir bench`` sweep)."""

from __future__ import annotations

import typer

from mjolnir_cli import retention
from mjolnir_cli.commands._common import console
from mjolnir_cli.config import load_layout


def gc(
    days: int = typer.Option(
        None,
        "--days",
        help="grace period in days (default: $MJOLNIR_RETENTION_DAYS or 14; 0 keeps everything)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="report what would be freed, delete nothing"),
) -> None:
    """Free the untracked exhaust: expired experiment runs, superseded perf
    dirs and verify logs — anything past the grace period that nothing cites."""
    if days is not None and days < 0:
        typer.secho("--days must be >= 0", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    layout = load_layout()
    rep = retention.sweep(layout, days=days, dry_run=dry_run)
    if dry_run:
        console.print(f"[mjolnir] (dry run) {rep.summary()}")
    else:
        console.print(f"[mjolnir] {rep.summary()}")
    for rel in rep.removed:
        console.print(f"  [dim]freed {rel}[/dim]")
    if rep.kept_cited:
        console.print(f"  [dim]kept {rep.kept_cited} cited past the grace period[/dim]")
