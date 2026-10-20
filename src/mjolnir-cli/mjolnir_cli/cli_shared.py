"""CLI state + settings helpers shared by the command modules."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console

from mjolnir_cli.config import Settings, resolve
from mjolnir_cli.platforms import DEFAULT_PLATFORM

try:  # typer >= 0.20 vendors its own click
    from typer._click.core import ParameterSource
except ImportError:  # pragma: no cover
    from click.core import ParameterSource  # ty: ignore[unresolved-import]

_console = Console()

_HELP_CTX = {"help_option_names": ["-h", "--help"]}


def _state_file() -> Path:
    return Path(os.environ.get("MJOLNIR_STATE", str(Path.home() / ".mjolnir-state.json"))).expanduser()


def _load_state() -> dict:
    p = _state_file()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {}


def _save_state(patch: dict) -> None:
    """Merge ``patch`` into the state file (the active model/config/image)."""
    st = _load_state()
    st.update(patch)
    _state_file().parent.mkdir(parents=True, exist_ok=True)
    _state_file().write_text(json.dumps(st, indent=2))


def _settings(model: str | None, quant: str | None, image: str | None, port: int | None) -> Settings:
    st = _load_state()
    return resolve(model or st.get("model"), quant or st.get("quant"), image or st.get("image"), port)


def _unset_flag(ctx: typer.Context, name: str) -> bool:
    """True when the typer option ``name`` was left at its default (neither
    passed on the command line nor set through env/config)."""
    return ctx.get_parameter_source(name) == ParameterSource.DEFAULT


def _platform_leg(
    ctx: typer.Context, name: str, platform: str | None, given: str | None, resolved: str | None
) -> str | None:
    """The legs sentinel (plan §0.5), shared by bench ab/accuracy/e2e and
    verify: the typer defaults are bound at import time to the *default*
    platform's leg table, so on another platform an **unset** flag re-resolves
    there (``resolved``) — an explicitly passed flag always wins, even when it
    names a default-platform leg. An unresolved Settings (platform ``None``)
    serves the default table."""
    if platform is None or platform == DEFAULT_PLATFORM or not _unset_flag(ctx, name):
        return given
    return resolved
