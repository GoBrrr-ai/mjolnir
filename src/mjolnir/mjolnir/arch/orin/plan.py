"""Host-side split-KV planning for the Orin GEMV decode kernel.

Pure, torch-free, and **board-agnostic**: every board in the Orin family
(AGX Orin 64/32 GB, Orin NX 16/8 GB, Orin Nano 8/4 GB) is the same
``sm_87`` GA10B arch with a different SM count and memory system, so the
planner takes ``sm_count`` from the caller — which queries the device at
runtime (``torch.cuda.get_device_properties().multi_processor_count`` or
the DSL's ``HardwareInfo``), never a per-board constant — and the plan
adapts itself. The kernel's grid is ``(num_kv_heads, num_splits,
batch)``, so the split count is the only knob that trades
partial-write traffic against SM fill.
"""

from __future__ import annotations

# FlashInfer Alg. 1 (scheduler.cuh:176-187): fill up to
# ``BLOCKS_PER_SM * sm_count`` CTAs with (kv_head, split, batch) blocks.
# 4 is the conservative occupancy estimate for the 96-thread GEMV CTA
# (~100 regs, ~6 KiB smem -> 4-6 resident CTAs/SM); it keeps the
# partial-write traffic small. The runner may override per call
# (``force_splits``) after measuring.
BLOCKS_PER_SM = 4

# The LSE combine's slot capacity (the fp32 partial buffers are indexed
# by the split dim; the in-tree combine kernel supports up to 64).
COMBINE_MAX_SPLITS = 64

# Rows per split floor: a split shorter than this hides no DRAM latency
# (one cp.async round trip is ~128 rows at 128-bit loads) and the extra
# partials only add combine traffic.
KV_CHUNK_FLOOR = 128


def cap_num_splits(ns: int, kv_len: int) -> int:
    """The caller-side caps: the combine's slot max and the 128-row
    kv-chunk floor (never more splits than there are 128-row chunks)."""
    return max(1, min(ns, COMBINE_MAX_SPLITS, (kv_len + KV_CHUNK_FLOOR - 1) // KV_CHUNK_FLOOR))


def auto_num_splits(kv_len: int, batch: int, num_kv_heads: int, sm_count: int) -> int:
    """FlashInfer Alg. 1: pick ``num_splits`` so the grid
    ``(num_kv_heads, num_splits, batch)`` fills up to
    ``BLOCKS_PER_SM * sm_count`` CTAs."""
    if kv_len <= 0 or batch <= 0 or num_kv_heads <= 0 or sm_count <= 0:
        return 1
    max_grid = sm_count * BLOCKS_PER_SM
    ns = (max_grid + num_kv_heads * batch - 1) // (num_kv_heads * batch)
    return cap_num_splits(ns, kv_len)


def plan_splits(
    kv_len: int,
    batch: int,
    num_kv_heads: int,
    sm_count: int,
    force_splits: int = 0,
) -> int:
    """The split count for a decode call.

    Precedence: an explicit ``force_splits > 0`` (a measured override)
    wins over the Alg. 1 auto plan; both are capped by
    :func:`cap_num_splits`. ``kv_len`` is the (max) KV length of the call
    — per-sequence varlen lengths are handled by the per-CTA chunk logic
    in the kernel (empty chunks take the ``kv_len == 0`` path)."""
    if force_splits > 0:
        return cap_num_splits(force_splits, kv_len)
    return auto_num_splits(kv_len, batch, num_kv_heads, sm_count)
