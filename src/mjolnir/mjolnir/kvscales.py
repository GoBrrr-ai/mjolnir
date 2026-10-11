"""Static per-(layer, KV head, channel) scales for the fp6 (e2m3) KV cache.

e2m3 spans only 0 .. ±7.5 with 0.125 steps below 1 — about 60x of dynamic
range. One scale per KV head is not enough on real models: K has outlier
channels (Qwen3-0.6B layer 0: K amax 486 in a head whose other channels are
a few units), and a head-wide scale of 486 / 7.5 rounds every small channel
to zero. So every (layer, KV head, channel) gets its own scale,
``amax / 7.5``. They cost nothing at attention time — a K channel scale folds
into Q (q . k = sum_c (q_c s_c) code_c), a V channel scale into the output
(O_c = s_c sum_t p_t code_tc) — see ``mjolnir.backend``.

vLLM 0.31 has no runtime KV-scale calculation and checkpoints carry fp8
scales at best, so the amax values come from a calibration run
(``python -m mjolnir.calibrate``, or ``mjolnir calibrate``): serve the model
on fp8 KV with ``MJOLNIR_ATTN_KV_CALIBRATE=1``, run prompts, and dump the
running per-channel max of every K and V written to the cache.

File (``MJOLNIR_ATTN_KV_SCALES``)::

    {"format": "mjolnir.kv_scales/2", "model": "...", "tokens": 123456,
     "amax": {"<layer name>": {"k": [[d floats] x h_k], "v": [[d floats] x h_k]}, ...}}

(``mjolnir.kv_scales/1`` files — one amax per head — still load; each head's
amax then applies to all its channels.)
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import torch

FORMAT = "mjolnir.kv_scales/2"
FORMATS = ("mjolnir.kv_scales/1", FORMAT)
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
        fmt = doc.get("format")
        if fmt not in FORMATS or not isinstance(doc.get("amax"), dict):
            hint = (
                " — the file is newer than this mjolnir package (rebuild the image: mjolnir image build)"
                if isinstance(fmt, str) and fmt.startswith("mjolnir.kv_scales/")
                else ""
            )
            raise ValueError(f"{path}: format {fmt!r}, this mjolnir reads {', '.join(FORMATS)}{hint}")
        _FILES[key] = doc
    return _FILES[key]


def headroom() -> float:
    """Scale multiplier (>= 1 leaves room above the calibrated amax; values
    beyond it saturate at ±7.5 * scale). MJOLNIR_ATTN_KV_SCALE_HEADROOM."""
    v = os.environ.get("MJOLNIR_ATTN_KV_SCALE_HEADROOM", "").strip()
    return float(v) if v else 1.0


def e2m3_scales(doc: dict, layer_name: str, h_k: int, d: int) -> tuple[list[list[float]], list[list[float]]]:
    """(k scales, v scales), each (h_k, d) nested lists: amax / 7.5 * headroom.
    Per-head entries (format /1, or one value) broadcast over channels."""
    entry = doc["amax"].get(layer_name)
    if entry is None:
        raise KeyError(layer_name)
    out = []
    for side in ("k", "v"):
        rows = entry[side]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{layer_name}: bad {side} entry")
        if not isinstance(rows[0], list):  # per head (or one value)
            rows = [[float(a)] * d for a in (rows * h_k if len(rows) == 1 else rows)]
        if len(rows) != h_k or any(len(r) != d for r in rows):
            raise ValueError(f"{layer_name}: {side} is {len(rows)}x{len(rows[0])}, the layer is {h_k}x{d}")
        out.append([[max(float(a), 1e-6) / E2M3_MAX * headroom() for a in r] for r in rows])
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
    per-(KV head, channel) amax. Graph-safe: in-place on persistent tensors."""
    T = slot_mapping.shape[0]
    if T == 0:
        return
    h_k, d = key.shape[1], key.shape[2]
    with _LOCK:
        if layer_name not in _CALIB:
            dev = key.device
            _CALIB[layer_name] = (
                torch.zeros(h_k, d, dtype=torch.float32, device=dev),
                torch.zeros(h_k, d, dtype=torch.float32, device=dev),
            )
            _CALIB_TOKENS[layer_name] = torch.zeros((), dtype=torch.int64, device=dev)
    k_amax, v_amax = _CALIB[layer_name]
    live = (slot_mapping[:T] >= 0).view(T, 1, 1)
    ka = torch.where(live, key[:T].float().abs(), 0.0).amax(dim=0)  # (h_k, d)
    va = torch.where(live, value[:T].float().abs(), 0.0).amax(dim=0)
    torch.maximum(k_amax, ka, out=k_amax)
    torch.maximum(v_amax, va, out=v_amax)
    _CALIB_TOKENS[layer_name].add_(live.sum())


def per_channel(doc: dict, layer_name: str) -> bool:
    """Does the file carry per-channel amax for this layer?"""
    k = doc["amax"].get(layer_name, {}).get("k")
    return bool(k) and isinstance(k[0], list)


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
