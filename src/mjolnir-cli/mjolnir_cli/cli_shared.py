"""CLI state + settings helpers shared by the command modules."""

from __future__ import annotations

import json
import os
from pathlib import Path

from rich.console import Console

from mjolnir_cli.config import Settings, resolve

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
