"""Shared decode-runner machinery, family-independent by construction.

Everything the vLLM backend and every arch family's runner needs that is
not vLLM-specific and not kernel-specific: the fp8 helpers, the
compiled-kernel cache (in-process + the on-disk object cache, keyed by
the *family's* kernel module so two families can never reuse each
other's objects), the kernel env knobs, the StepPlan dataclass and the
KV-cache views. The per-family Runner classes live in
``mjolnir.arch.<family>.runner`` and import from here.

Nothing here imports vLLM, so tests/kernel/test_runner.py exercises it
directly.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import torch


FP8_MAX = 448.0
_log = logging.getLogger("vllm.mjolnir")
SUPPORTED_PAGE_SIZES = (16, 32, 64, 128)

_TORCH_TO_CUTLASS = None


def _cutlass_dtype(dt: torch.dtype):
    global _TORCH_TO_CUTLASS
    if _TORCH_TO_CUTLASS is None:
        import cutlass

        _TORCH_TO_CUTLASS = {
            torch.bfloat16: cutlass.BFloat16,
            torch.float16: cutlass.Float16,
            torch.float32: cutlass.Float32,
            torch.int32: cutlass.Int32,
            torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        }
    return _TORCH_TO_CUTLASS[dt]


def _ptr(t: torch.Tensor):
    """Raw cute pointer for a tensor (cheap; no dlpack round trip)."""
    import cutlass.cute as cute

    assert t.data_ptr() % 16 == 0, "T5 kernel needs 16-byte aligned base pointers"
    dtype = _cutlass_dtype(t.dtype)
    try:
        from cutlass.cute.runtime import make_ptr

        return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    except ImportError:  # older DSL: go through dlpack
        from cutlass.cute.runtime import from_dlpack

        src = t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t
        ct = from_dlpack(src, assumed_align=16)
        ct.element_type = dtype
        return ct.iterator


def _stream():
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


# ----------------------------------------------------------------------------
# Compiled-kernel cache (process wide; shared by all layers / builders)
# ----------------------------------------------------------------------------
_COMPILED: dict = {}
_COMPILE_LOCK = threading.Lock()

# ----------------------------------------------------------------------------
# On-disk kernel cache: a compiled variant is exported as an object file
# (CuTe DSL export_to_c) and loaded back on the next server start
# (cute.runtime.load_module), skipping the ~2-3 s compile per variant.
# MJOLNIR_ATTN_KERNEL_CACHE: unset/1 = on (under $VLLM_CACHE_ROOT/mjolnir or
# ~/.cache/mjolnir), 0 = off, a path = that directory. The entry key covers
# the variant, the kernel source, the CuTe DSL stack and the GPU, so a new
# kernel, image or DSL never loads a stale object. Best effort: any failure
# compiles in-process as before and turns the cache off for the process.
# ----------------------------------------------------------------------------
_DISK_MODULES: dict = {}  # keep loaded modules alive
CACHE_STATS = {"loaded": 0, "compiled": 0}
_disk_cache_ok = True


def kernel_cache_dir():
    from pathlib import Path

    v = env_flag("MJOLNIR_ATTN_KERNEL_CACHE")
    if v.lower() in ("0", "false", "no", "off"):
        return None
    if v and v.lower() not in ("1", "true", "yes", "on"):
        return Path(v)
    base = os.environ.get("VLLM_CACHE_ROOT") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "mjolnir" / "kernels"


_FINGERPRINTS: dict[str, str] = {}


def _cache_fingerprint(kernel_path: str) -> str:
    """Everything a compiled object depends on besides the variant key:
    the *family's* kernel source (passed by the runner — never a global
    module reference), the package version, the CuTe DSL stack and the
    device's compute capability."""
    fp = _FINGERPRINTS.get(kernel_path)
    if fp is None:
        import hashlib
        import importlib.metadata

        from . import __version__

        h = hashlib.sha256()
        with open(kernel_path, "rb") as f:
            h.update(f.read())
        dsl = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in importlib.metadata.distributions()
            if (d.metadata["Name"] or "").lower().startswith("nvidia-cutlass-dsl")
        )
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        h.update(f"{__version__}|{';'.join(dsl)}|sm{cap[0]}{cap[1]}".encode())
        fp = h.hexdigest()[:20]
        _FINGERPRINTS[kernel_path] = fp
    return fp


def _cache_entry(key, kernel_path: str, family: str):
    import hashlib

    d = kernel_cache_dir()
    if d is None:
        return None
    kh = hashlib.sha256(repr(key).encode()).hexdigest()[:20]
    return d / _cache_fingerprint(kernel_path) / f"{family}_{kh}.o", f"mjolnir_{family}_{kh}"


def _disk_load(key, kernel_path: str, family: str):
    global _disk_cache_ok
    if not _disk_cache_ok:
        return None
    try:
        entry = _cache_entry(key, kernel_path, family)
        if entry is None or not entry[0].exists():
            return None
        import cutlass.cute as cute

        module = cute.runtime.load_module(str(entry[0]))
        fn = getattr(module, entry[1])
        _DISK_MODULES[key] = module
        _log.debug("Mjolnir: loaded %s from %s", key, entry[0])
        return fn
    except Exception as e:  # noqa: BLE001 - any failure: compile instead
        _disk_cache_ok = False
        _log.warning("Mjolnir: kernel cache load failed (%r); compiling, cache off for this process", e)
        return None


def _export(fn, path, symbol: str) -> None:
    """Write ``fn``'s object file to ``path`` (atomically), exported under
    ``symbol``. CuTe DSL 4.7 (vLLM 0.31's): ``export_to_c(dir, stem,
    function_prefix=...)`` writes ``<dir>/<stem>.o`` + ``.h``; later DSLs:
    ``export_to_c(path, function_name=...)``."""
    import inspect

    params = inspect.signature(fn.export_to_c).parameters
    stem = f"{path.stem}.tmp{os.getpid()}"
    tmp_o, tmp_h = path.parent / f"{stem}.o", path.parent / f"{stem}.h"
    try:
        if "function_prefix" in params:
            fn.export_to_c(str(path.parent), stem, function_prefix=symbol)
        elif "function_name" in params:
            fn.export_to_c(str(tmp_o), function_name=symbol)
        else:
            raise TypeError(f"unknown export_to_c signature {tuple(params)}")
        os.replace(tmp_o, path)
    finally:
        for t in (tmp_o, tmp_h):
            if t.exists():
                t.unlink()


def _disk_store(key, fn, kernel_path: str, family: str) -> None:
    global _disk_cache_ok
    if not _disk_cache_ok:
        return
    try:
        entry = _cache_entry(key, kernel_path, family)
        if entry is None:
            return
        path, symbol = entry
        path.parent.mkdir(parents=True, exist_ok=True)
        _export(fn, path, symbol)
    except Exception as e:  # noqa: BLE001 - persistence is best effort
        _disk_cache_ok = False
        _log.warning("Mjolnir: kernel cache store failed (%r); cache off for this process", e)


@dataclass
class StepPlan:
    """Per-step plan shared by every layer of one KV-cache group."""

    num_seqs: int
    q_len: int
    n_q: int
    num_work: int  # == grid.x, fixed per (q_len) capacity
    with_reduction: bool
    block_table: torch.Tensor  # (>= num_seqs, max_blocks) int32, device
    seq_lens: torch.Tensor  # (>= num_seqs,) int32, device
    work: torch.Tensor  # (num_work, 4) int32 view of a persistent buffer
    n_splits: torch.Tensor  # (num_seqs,) int32 view of a persistent buffer
    chunk_tiles: int
    num_chunks: int
    causal: bool = True


ENV_PREFIX = "MJOLNIR_ATTN_"
LEGACY_ENV_PREFIX = "THOR_T5_"


def env_flag(name: str, default: str = "") -> str:
    """Read a kernel knob. ``name`` is the canonical ``MJOLNIR_ATTN_*`` name;
    the pre-rename ``THOR_T5_*`` spelling is still honored as a fallback."""
    if name in os.environ:
        return os.environ[name].strip()
    if name.startswith(ENV_PREFIX):
        legacy = LEGACY_ENV_PREFIX + name[len(ENV_PREFIX) :]
        if legacy in os.environ:
            return os.environ[legacy].strip()
    return default.strip()


# ----------------------------------------------------------------------------
# KV-cache views
# ----------------------------------------------------------------------------
KV_LAYOUTS = ("BHN2D", "BNH2D", "B2NHD", "B2HND", "2BNHD", "2BHND")


def _canon_singleton(t: torch.Tensor) -> torch.Tensor:
    """Size-1 dims may carry arbitrary strides (e.g. h_k == 1 under TP);
    TMA wants 16-byte multiples -> zero them (never used for indexing)."""
    if all(s != 1 for s in t.shape):
        return t
    st = [0 if n == 1 else s for n, s in zip(t.shape, t.stride(), strict=True)]
    return t.as_strided(t.shape, st, t.storage_offset())


def detect_kv_layout(shape, num_kv_heads: int, page_size: int, head_size: int) -> str:
    """Name the logical layout of a vLLM per-layer KV cache tensor.

    vLLM main:            (B, H, N, 2D)  K|V interleaved in the last dim
    older FlashInfer NHD: (B, 2, N, H, D);  HND: (B, 2, H, N, D)
    older FlashAttention: (2, B, N, H, D)
    Ambiguous (H == N) shapes resolve to the vLLM-main / NHD reading;
    MJOLNIR_ATTN_KV_LAYOUT overrides.
    """
    forced = env_flag("MJOLNIR_ATTN_KV_LAYOUT").upper()
    if forced:
        assert forced in KV_LAYOUTS, f"MJOLNIR_ATTN_KV_LAYOUT must be one of {KV_LAYOUTS}"
        return forced
    H, N, D = num_kv_heads, page_size, head_size
    s = tuple(shape)
    if len(s) == 4 and s[3] == 2 * D:
        if (s[1], s[2]) == (H, N):
            return "BHN2D"
        if (s[1], s[2]) == (N, H):
            return "BNH2D"
    if len(s) == 5 and s[4] == D:
        if s[1] == 2 and (s[2], s[3]) == (N, H):
            return "B2NHD"
        if s[1] == 2 and (s[2], s[3]) == (H, N):
            return "B2HND"
        if s[0] == 2 and (s[2], s[3]) == (N, H):
            return "2BNHD"
        if s[0] == 2 and (s[2], s[3]) == (H, N):
            return "2BHND"
    raise ValueError(
        f"unrecognized KV cache shape {s} for H={H} N={N} D={D}; set MJOLNIR_ATTN_KV_LAYOUT to one of {KV_LAYOUTS}"
    )


def split_kv_cache(kv_cache: torch.Tensor, layout: str, head_size: int):
    """Zero-copy logical (num_pages, page_size, h_k, d) K and V views of any
    supported layout. Physical stride order (NHD/HND, layer-major, ...) is
    carried in the strides, which the kernel takes as runtime arguments."""
    D = head_size
    if layout == "BHN2D":
        k, v = kv_cache[..., :D].transpose(1, 2), kv_cache[..., D:].transpose(1, 2)
    elif layout == "BNH2D":
        k, v = kv_cache[..., :D], kv_cache[..., D:]
    elif layout == "B2NHD":
        k, v = kv_cache[:, 0], kv_cache[:, 1]
    elif layout == "B2HND":
        k, v = kv_cache[:, 0].transpose(1, 2), kv_cache[:, 1].transpose(1, 2)
    elif layout == "2BNHD":
        k, v = kv_cache[0], kv_cache[1]
    elif layout == "2BHND":
        k, v = kv_cache[0].transpose(1, 2), kv_cache[1].transpose(1, 2)
    else:
        raise ValueError(layout)
    return _canon_singleton(k), _canon_singleton(v)
