"""Static per-(layer, KV head) scales for the fp6 (e2m3) KV cache.

e2m3 spans only 0 .. ±7.5 (with 0.125 steps below 1), so every KV head needs
its own scale, ``amax / 7.5``. vLLM 0.31 has no runtime KV-scale calculation
and checkpoints carry fp8 scales at best, so the amax values come from a
calibration run (``python -m mjolnir.calibrate``, or ``mjolnir calibrate``):
serve the model on fp8 KV with ``MJOLNIR_ATTN_KV_CALIBRATE=1``, run prompts,
and dump the running per-head max of every K and V written to the cache.

File (``MJOLNIR_ATTN_KV_SCALES``)::

    {"format": "mjolnir.kv_scales/1", "model": "...", "tokens": 123456,
     "amax": {"<layer name>": {"k": [h_k floats], "v": [h_k floats]}, ...}}
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import torch

FORMAT = "mjolnir.kv_scales/1"
E2M3_MAX = 7.5

# ── loading ───────────────────────────────────────────────────────────────
_FILES: dict[str, dict] = {}


def scales_path() -> str | None:
    for name in ("MJOLNIR_ATTN_KV_SCALES", "THOR_T5_KV_SCALES"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return None


def load(path: str | os.PathLike) -> dict:
    key = str(path)
    if key not in _FILES:
        doc = json.loads(Path(path).read_text())
        if doc.get("format") != FORMAT or not isinstance(doc.get("amax"), dict):
            raise ValueError(f"{path}: not a {FORMAT} file")
        _FILES[key] = doc
    return _FILES[key]


def headroom() -> float:
    """Scale multiplier (>= 1 leaves room above the calibrated amax; values
    beyond it saturate at ±7.5 * scale). MJOLNIR_ATTN_KV_SCALE_HEADROOM."""
    v = os.environ.get("MJOLNIR_ATTN_KV_SCALE_HEADROOM", "").strip()
    return float(v) if v else 1.0


def e2m3_scales(doc: dict, layer_name: str, h_k: int) -> tuple[list[float], list[float]]:
    """(k scales, v scales), h_k each: amax / 7.5 * headroom."""
    entry = doc["amax"].get(layer_name)
    if entry is None:
        raise KeyError(layer_name)
    out = []
    for side in ("k", "v"):
        amax = [float(x) for x in entry[side]]
        if len(amax) == 1:
            amax = amax * h_k
        if len(amax) != h_k:
            raise ValueError(f"{layer_name}: {side} has {len(amax)} heads, the layer has {h_k}")
        out.append([max(a, 1e-6) / E2M3_MAX * headroom() for a in amax])
    return out[0], out[1]


def missing_layers(doc: dict, layer_names) -> list[str]:
    return [n for n in layer_names if n not in doc["amax"]]


# ── calibration (in-process accumulators) ─────────────────────────────────
_CALIB: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_CALIB_TOKENS: dict[str, torch.Tensor] = {}
_LOCK = threading.Lock()


def calibrating() -> bool:
    return os.environ.get("MJOLNIR_ATTN_KV_CALIBRATE", "").strip().lower() in ("1", "true", "yes")


def observe(layer_name: str, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    """Fold the K/V of real tokens (slot >= 0) into the layer's running
    per-head amax. Graph-safe: in-place on persistent tensors."""
    T = slot_mapping.shape[0]
    if T == 0:
        return
    h_k = key.shape[1]
    with _LOCK:
        if layer_name not in _CALIB:
            dev = key.device
            _CALIB[layer_name] = (
                torch.zeros(h_k, dtype=torch.float32, device=dev),
                torch.zeros(h_k, dtype=torch.float32, device=dev),
            )
            _CALIB_TOKENS[layer_name] = torch.zeros((), dtype=torch.int64, device=dev)
    k_amax, v_amax = _CALIB[layer_name]
    live = (slot_mapping[:T] >= 0).unsqueeze(1)  # (T, 1)
    ka = torch.where(live, key[:T].float().abs().amax(dim=2), 0.0).amax(dim=0)
    va = torch.where(live, value[:T].float().abs().amax(dim=2), 0.0).amax(dim=0)
    torch.maximum(k_amax, ka, out=k_amax)
    torch.maximum(v_amax, va, out=v_amax)
    _CALIB_TOKENS[layer_name].add_(live.sum())


def reset() -> None:
    """Zero the accumulators (after engine warm-up and profiling runs).

    They were created inside vLLM's forward (torch.inference_mode), so they
    are inference tensors: in-place updates are only allowed in that mode."""
    with _LOCK, torch.inference_mode():
        for k, v in _CALIB.values():
            k.zero_()
            v.zero_()
        for t in _CALIB_TOKENS.values():
            t.zero_()


def snapshot(model: str = "") -> dict:
    with _LOCK, torch.inference_mode():
        amax = {name: {"k": k.tolist(), "v": v.tolist()} for name, (k, v) in sorted(_CALIB.items())}
        tokens = max((int(t.item()) for t in _CALIB_TOKENS.values()), default=0)
    return {"format": FORMAT, "model": model, "tokens": tokens, "amax": amax}


def dump(path: str | os.PathLike, model: str = "") -> dict:
    doc = snapshot(model)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1) + "\n")
    return doc
