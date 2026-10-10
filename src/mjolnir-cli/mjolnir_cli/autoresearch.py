"""``mjolnir autoresearch`` — placeholder (filled in by the autoresearch task)."""

import typer

app = typer.Typer(help="Autonomous kernel-research loop.", no_args_is_help=True)


@app.command()
def status():
    """Show the ledger."""
