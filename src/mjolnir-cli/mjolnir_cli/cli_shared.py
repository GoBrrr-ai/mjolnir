"""Typer parameter-source + platform-legs helpers shared by ``bench`` and
``verify``.

This stays outside the ``commands`` package so ``verify`` (which must not
import the command bodies) can use the legs sentinel; the state file /
settings / console helpers every command module shares live in
``mjolnir_cli.commands._common``."""

from __future__ import annotations

import typer

from mjolnir_cli.platforms import DEFAULT_PLATFORM

try:  # typer >= 0.20 vendors its own click
    from typer._click.core import ParameterSource
except ImportError:  # pragma: no cover
    from click.core import ParameterSource  # ty: ignore[unresolved-import]


def _unset_flag(ctx: typer.Context, name: str) -> bool:
    """True when the typer option ``name`` was left at its default (neither
    passed on the command line nor set through env/config)."""
    return ctx.get_parameter_source(name) == ParameterSource.DEFAULT


def platform_leg(
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
