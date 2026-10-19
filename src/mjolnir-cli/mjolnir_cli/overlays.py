"""Config platform overlays: ``configs/<vendor>/<model>/<quant>/platforms/
<platform>.yaml`` deep-merges over the config's ``config.yaml``.

The rules (plan §0.3): dicts merge recursively, lists replace, scalars
override. A config dir **without** a ``platforms/`` dir is a legacy
thor-only config (the pre-overlay layout — zero data churn); with the dir,
only the listed overlays exist. :func:`validate` checks the merged config
against the ``kv-cache-dtype`` x attention-backend availability table
(``mjolnir_cli.availability`` — the cycle-2 inline matrix became its data)
and every error names the platform, the key, the value and why — a bad
overlay is fixable from the message alone.

Pure merge/validate logic plus thin path I/O; serving code merges
``load_overlay`` over the config yaml before serving a leg.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from mjolnir_cli import availability
from mjolnir_cli.availability import KV_AVAILABILITY, UNAVAILABLE, kv_state
from mjolnir_cli.platforms import DEFAULT_PLATFORM


def overlay_dir(config_dir: Path) -> Path:
    """The config's per-platform overlay directory."""
    return Path(config_dir) / "platforms"


def overlay_platforms(config_dir: Path) -> tuple[str, ...]:
    """The platforms the config carries overlays for (``()`` when there is no
    ``platforms/`` dir — the legacy thor-only layout)."""
    d = overlay_dir(config_dir)
    if not d.is_dir():
        return ()
    return tuple(sorted(p.stem for p in d.glob("*.yaml")))


def load_overlay(config_dir: Path, platform: str) -> dict | None:
    """The parsed overlay for ``platform`` — None when it has none; ValueError
    naming the file when the yaml is malformed or not a mapping."""
    p = overlay_dir(config_dir) / f"{platform}.yaml"
    if not p.is_file():
        return None
    try:
        data = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"malformed platform overlay {p.name} ({p}): {e}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"malformed platform overlay {p.name} ({p}): expected a yaml mapping, got {type(data).__name__}"
        )
    return data


def merge_config(base: dict, overlay: dict) -> dict:
    """The overlay deep-merged over the base: nested dicts merge recursively,
    lists replace wholesale, scalars override. Neither input is mutated."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = value
    return out


def backend_class(merged: dict) -> str:
    """The attention-backend class a merged config selects: the
    ``attention-backend`` field, else ``attention-config.backend`` — FA4
    (``flash_attn`` with ``flash_attn_version: 4``) is its own class, named
    ``FLASH_ATTN@4`` so the field string maps onto :data:`FA4` and errors can
    quote it. An undeclared backend stays ``""`` so validate can skip."""
    ac = merged.get("attention-config")
    ac = ac if isinstance(ac, dict) else {}
    backend = str(merged.get("attention-backend") or ac.get("backend") or "").upper()
    if backend == "FLASH_ATTN" and ac.get("flash_attn_version") == 4:
        return "FLASH_ATTN@4"
    return backend


def validate(platform: str, merged: dict) -> None:
    """Check the merged config against the availability table
    (:func:`mjolnir_cli.availability.kv_state` decides); ValueError names the
    platform, the key, the value and why when the combination cannot be
    served. A config without ``kv-cache-dtype``, with a dtype the table does
    not know, or **without a declared attention backend** has nothing to
    contradict — the table validates declared (dtype, backend) pairs; with no
    backend declared, vLLM auto-selects a servable one."""
    kv = merged.get("kv-cache-dtype")
    if not kv:
        return
    kv = str(kv)
    if kv not in KV_AVAILABILITY:  # unknown dtype: nothing to contradict
        return
    backend = backend_class(merged)
    if not backend:  # no declared backend: vLLM auto-selects a servable one
        return
    cell = kv_state(kv, availability.backend_class(backend), platform)
    if cell.state == UNAVAILABLE:
        raise ValueError(
            f"{platform}: kv-cache-dtype {kv!r} cannot be served with attention-backend {backend!r} — {cell.why}"
        )


def config_exists_on(config_dir: Path, platform: str) -> bool:
    """Does the config exist on ``platform``: a ``config.yaml`` must be there,
    and — the overlay rule — a dir without ``platforms/`` is a legacy
    thor-only config, while a dir with overlays lists its platforms exactly
    (legs skip a config rather than guess)."""
    d = Path(config_dir)
    if not (d / "config.yaml").is_file():
        return False
    plats = overlay_platforms(d)
    if not plats:
        return platform == DEFAULT_PLATFORM
    return platform in plats


def validate_serving(config_path: Path, platform: str | None) -> None:
    """The serve-time gate: the config merged with ``platform``'s overlay must
    be a servable combination (the kv-cache-dtype x attention-backend matrix)
    — ValueError names the platform, the key, the value and why. A missing or
    unparseable base config is not this check's error (the caller reports
    that); the default platform (or an unresolved Settings) never
    re-validates: its configs are the reviewed ones."""
    if platform is None or platform == DEFAULT_PLATFORM:
        return
    config_path = Path(config_path)
    if not config_path.is_file():
        return
    try:
        base = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError:
        return
    if not isinstance(base, dict):
        return
    overlay = load_overlay(config_path.parent, platform) or {}
    validate(platform, merge_config(base, overlay))
