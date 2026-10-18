"""Shared kernel constants and the harness note helper (leaf module).

t5_decode (the traced kernels) and t5_call (the host-side builder) both need
these; they live here so neither imports the other.
"""

from __future__ import annotations

import os
from functools import partial

import cutlass.cute as cute

VERBOSE = os.environ.get("MJOLNIR_ATTN_VERBOSE", os.environ.get("THOR_T5_VERBOSE", "")).strip() in ("1", "true", "yes")


def _note(msg: str) -> None:
    if VERBOSE:
        print(msg)


# Kernel invariants
mma_modes = (0, 1, 2)
mma_dice = (None, None, None)  # (MMA, #MMA_M, #MMA_K)
cpy_dice = (None,) + mma_dice  # (CPY, #CPY_MMA, #CPY_M, #CPY_K)
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128

# SW128 smem atoms repeat every 1024 B and both TMA and the UMMA descriptor
# (base_offset = 0) assume the tile base is 1024-B aligned.
swizzle_align = 1024

# fp8 P is stored as e4m3(p * FP8_P_SCALE): p in [0, 1] then spans e4m3's
# normal range instead of flushing small probabilities to subnormals / 0.
# 1 / FP8_P_SCALE is folded into the output scale.
FP8_P_SCALE = 448.0

log2_e = 1.4426950408889634


# Math helpers
fadd2 = cute.arch.add_packed_f32x2
fmul2 = cute.arch.mul_packed_f32x2
ffma2 = cute.arch.fma_packed_f32x2
exp2 = partial(cute.math.exp2, fastmath=True)


def _warp_fmax_shfl(value, mask=0xFFFFFFFF):
    """Warp-wide fmax via a butterfly shfl. ``warp_redux_sync(kind="fmax")``
    emits ``redux.f32``, which ptxas rejects on sm_110a (redux is int-only)."""
    for offset in (16, 8, 4, 2, 1):
        value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=offset, mask=mask), nan=True)
    return value


warp_fmax = _warp_fmax_shfl
smem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")
