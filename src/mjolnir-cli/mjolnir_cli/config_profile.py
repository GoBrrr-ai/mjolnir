"""The config README's serving-profile table (cycle 3e): the curated
meta-table behind :attr:`mjolnir_cli.readme_targets.ConfigContext.profile`.

:func:`_profile_rows` turns each serving platform's merged config dict into
one row per scalar serving-capacity key — dict-valued keys never appear (they
live in config.yaml, one click away), identical-across-platform values repeat
per column, a key absent from every platform's config omits its row, and
``served-model-name`` is omitted when it equals the page's model. Pure data
rules over ``(model, quant, merged)``; the template renders.

Host-side data leaf like its only consumer: stdlib + yaml, no torch/vllm, no
matplotlib, no jinja2 (import purity is pinned in
``tests/cli/test_host_import.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import yaml

from mjolnir_cli.config import BACKEND_LABELS
from mjolnir_cli.overlays import backend_class


@dataclass(frozen=True)
class ProfileRow:
    """One curated serving-profile row: the label plus one cell per platform."""

    label: str
    cells: tuple[str, ...]


def _cell_str(v: object) -> str:
    """One config scalar as the table renders it: floats at the config's
    written precision (0.80), lists/dicts as flow yaml, None as empty."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, dict | list):
        return yaml.safe_dump(v, default_flow_style=True).strip()
    return str(v)


def _context_str(v: object) -> str:
    """A context/token-count cell: a plain int gets the repo's L label."""
    if isinstance(v, int) and not isinstance(v, bool):
        from mjolnir_cli.charts_base import L_label  # lazy: this module stays a stdlib leaf

        return L_label(v)
    return _cell_str(v)


def _row(label: str, cells: list[str | None]) -> ProfileRow | None:
    """The row when any platform carries the value; absent everywhere → None."""
    if all(c is None for c in cells):
        return None
    return ProfileRow(label, tuple("—" if c is None else c for c in cells))


def _backend_label(quant: str, merged: list[dict]) -> str:
    """The friendly format line: BACKEND_LABELS, else the config's
    ``quantization`` key, else the quant name itself."""
    quantization = next((m.get("quantization") for m in merged if m.get("quantization")), None)
    return BACKEND_LABELS.get(quant) or (str(quantization) if quantization else quant)


def _profile_rows(model: str, quant: str, merged: list[dict]) -> list[ProfileRow]:
    """The curated meta-table (plan §1): the scalar serving-capacity keys of
    each platform's merged config, one cell per platform. Dict-valued keys
    (compilation-config, generation overrides, chat template) never appear —
    they live in config.yaml; identical-across-platform values repeat per
    column; ``served-model-name`` is omitted when it equals the page's model."""

    def col(key: str) -> list[object]:
        return [m.get(key) for m in merged]

    def fmt(key: str, f: Callable[[object], str] = _cell_str) -> list[str | None]:
        return [None if v is None else f(v) for v in col(key)]

    def backticked(key: str) -> list[str | None]:
        return [None if v is None else f"`{v}`" for v in col(key)]

    def backend(m: dict) -> str | None:
        """The canonical derivation (:func:`mjolnir_cli.overlays.backend_class`
        — the same one validate and the availability matrix read), None when
        the config declares no backend."""
        return backend_class(m) or None

    def spec(v: object) -> str | None:
        if not isinstance(v, dict) or not v.get("num_speculative_tokens"):
            return None
        n = int(v["num_speculative_tokens"])
        method = str(v.get("method") or "").upper()
        return f"{method} ×{n}" if method else f"×{n}"

    def kv(m: dict) -> str | None:
        dtype = m.get("kv-cache-dtype")
        if dtype is None:
            return None
        skip = m.get("kv-cache-dtype-skip-layers")
        return str(dtype) + (f" · skip: {_cell_str(skip)}" if skip else "")

    def parsing(m: dict) -> str | None:
        parts = [str(m[k]) for k in ("tool-call-parser", "reasoning-parser") if m.get(k)]
        return " / ".join(parts) or None

    rows: list[ProfileRow | None] = [
        _row("Checkpoint", backticked("model")),
        _row("Format", [_backend_label(quant, merged)] * len(merged)),
        _row("Max context", fmt("max-model-len", _context_str)),
        _row("Max concurrency", fmt("max-num-seqs")),
        _row("Prefill chunk", fmt("max-num-batched-tokens", _context_str)),
        _row("KV cache", [kv(m) for m in merged]),
        _row("Attention backend", [backend(m) for m in merged]),
        _row("Spec decode", [spec(v) for v in col("speculative-config")]),
        _row("GPU memory util", fmt("gpu-memory-utilization")),
        _row("Parsing", [parsing(m) for m in merged]),
    ]
    served = col("served-model-name")
    if any(v is not None and v != model for v in served):
        rows.insert(1, _row("Served as", backticked("served-model-name")))
    return [r for r in rows if r is not None]
