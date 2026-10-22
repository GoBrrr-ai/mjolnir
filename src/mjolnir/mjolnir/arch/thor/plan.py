"""Host-side split-KV planning and stride helpers (split out of t5_decode.py).

`t5_plan.plan_split_kv` is the occupancy-pinned planner a vLLM backend calls;
`kernel_strides` builds the int32 stride vectors the kernel launch consumes.
"""

from __future__ import annotations

import math

import torch

tile_s_rows = 128  # KV rows per tile (= GEMM1 M)

STRIDE_FIELDS = (
    "q_seq",
    "o_seq",
    "k_page",
    "k_row",
    "k_head",
    "v_page",
    "v_row",
    "v_head",
    "qd_seq",
    "qd_head",
    "kd_seq",
    "kd_head",
    "vd_seq",
    "vd_head",
    "ks_page",
    "ks_row",
    "ks_head",
    "vs_page",
    "vs_row",
    "vs_head",
)


def _descale_strides(t, num_seqs, heads_k):
    """(seq, head) element strides of a descale tensor: a 1-element tensor
    is a per-tensor scale (0, 0); (h_k,) per head; (b, h_k) per seq+head."""
    if t.numel() == 1:
        return 0, 0
    if t.dim() == 1 and t.shape[0] == heads_k:
        return 0, t.stride(0)
    assert t.dim() == 2 and t.shape[1] == heads_k, f"descale shape {tuple(t.shape)}"
    return t.stride(0), t.stride(1)


def _kv_elem_strides(t, kv_bits, name):
    """(page, row, head) strides of a KV view in units of the KV ELEMENT.
    8-bit formats: the view's own strides. Sub-byte formats (fp6 / fp4 /
    nvfp4) are passed as uint8 BYTE views (pages, PS, h_k, d * bits / 8);
    their byte strides are converted to element strides."""
    assert t.stride(3) == 1, f"{name}: head dim must be contiguous"
    byte_strides = [x * t.element_size() for x in t.stride()[:3]]
    assert all(x % 16 == 0 for x in byte_strides), f"{name}: TMA needs 16-byte-multiple strides, got {byte_strides} B"
    if kv_bits == 8:
        return tuple(t.stride()[:3])
    assert t.element_size() == 1, f"{name}: sub-byte KV must be a uint8 byte view"
    assert all((x * 8) % kv_bits == 0 for x in byte_strides), (
        f"{name}: byte strides {byte_strides} are not whole {kv_bits}-bit elements"
    )
    return tuple(x * 8 // kv_bits for x in byte_strides)


def kernel_strides(q, o, k_cache, v_cache, q_descale, k_descale, v_descale, kv_bits=8, k_scales=None, v_scales=None):
    """Element strides for the kernel's `strides` argument.

    q, o      : (b, h_k, n_q, d) views (rows d-contiguous; seq stride free)
    k/v_cache : (num_pages, page_size, h_k, d) LOGICAL views of any physical
                layout with d contiguous (vLLM NHD, interleaved K|V, FlashInfer,
                head-major HND); for sub-byte formats uint8 byte views with
                last dim d * kv_bits / 8
    *_descale : 1-element, (h_k,) or (b, h_k) fp32 tensors
    k/v_scales: nvfp4 only, (num_pages, page_size, h_k, d / 16) e4m3 (uint8)
    """
    b, h_k, n_q, d = q.shape
    for name, t in (("q", q), ("o", o)):
        assert t.stride(3) == 1 and t.stride(2) == d and t.stride(1) == n_q * d, (
            f"{name}: rows must be packed (n_q, h_k) with d contiguous, got {t.stride()}"
        )
    ks = (0, 0, 0)
    vs = (0, 0, 0)
    if k_scales is not None:
        assert k_scales.stride(3) == 1 and v_scales.stride(3) == 1
        ks = tuple(k_scales.stride()[:3])
        vs = tuple(v_scales.stride()[:3])
    vals = (
        q.stride(0),
        o.stride(0),
        *_kv_elem_strides(k_cache, kv_bits, "k_cache"),
        *_kv_elem_strides(v_cache, kv_bits, "v_cache"),
        *_descale_strides(q_descale, b, h_k),
        *_descale_strides(k_descale, b, h_k),
        *_descale_strides(v_descale, b, h_k),
        *ks,
        *vs,
    )
    assert all(0 <= x < 2**31 for x in vals), f"stride out of int32 range: {vals}"
    return tuple(int(x) for x in vals)


class SplitPlan:
    """Host-side occupancy-pinned split-KV plan (see plan_split_kv)."""

    def __init__(self, work, n_splits, chunk_tiles, slots, counter_len=0):
        self.work = work  # list of (seq, tile_begin, tile_end, split_slot)
        self.n_splits = n_splits  # chunks per sequence
        self.chunk_tiles = chunk_tiles  # tiles per chunk (last chunk may be short)
        self.slots = slots  # work items that fit one wave
        self.counter_len = counter_len  # fused-merge counter slots appended to n_splits

    @property
    def num_work(self):
        return len(self.work)

    def work_tensor(self, device="cpu"):
        t = torch.tensor(self.work if self.work else [(0, 0, 0, 0)], dtype=torch.int32)
        return t.reshape(-1, 4).contiguous().to(device)

    def n_splits_tensor(self, device="cpu"):
        t = torch.tensor(self.n_splits, dtype=torch.int32)
        if self.counter_len:
            # Fused-merge election counters, one per (q tile, kv head, seq);
            # zeroed once at plan build, self-reset by the last CTA afterwards
            # (graph-safe: a replayed launch must not re-zero).
            t = torch.cat([t, torch.zeros(self.counter_len, dtype=torch.int32)])
        return t.to(device)

    def __repr__(self):
        real = sum(1 for w in self.work if w[2] > w[1])
        return (
            f"SplitPlan(chunk={self.chunk_tiles} tiles, work={real}"
            f"{f'+{self.num_work - real} pad' if self.num_work > real else ''}, "
            f"slots/wave={self.slots}, max chunks/seq={max(self.n_splits)})"
        )


def plan_split_kv(
    seq_lens,
    heads_k,
    n_q_tiles,
    sm_count,
    ctas_per_sm=1,
    max_splits=16,
    force_splits=0,
    min_chunk_tiles=1,
    pad_to=0,
    with_counter=False,
):
    """Occupancy-pinned split-KV plan (the FlashInfer PrefillPlan / FlashDecoding
    scheme), in units of 128-row KV tiles.

    The grid is (work items, q tiles, kv heads), so one wave holds
        slots = sm_count * ctas_per_sm // (heads_k * n_q_tiles)
    work items. Pick the SMALLEST contiguous chunk size such that the total
    number of chunks over all sequences fits in one wave; every sequence is
    then cut into ceil(tiles / chunk) contiguous chunks. Long sequences get
    more chunks, short ones fewer, so varlen batches balance, and each CTA
    streams one contiguous KV range front to back.

    If the batch alone already fills the wave, nothing is split (each
    sequence is one work item, ordered longest first for the waves).
    force_splits > 0 cuts every sequence into that many chunks instead.
    pad_to > 0 pads the work list to a fixed length with empty items
    (begin == end, they exit at once) — keeps the grid fixed for CUDA graphs.
    """
    tiles = [max(1, math.ceil(int(L) / tile_s_rows)) for L in seq_lens]
    T = max(tiles)
    slots = max(1, (sm_count * ctas_per_sm) // (heads_k * n_q_tiles))
    lo = max(min_chunk_tiles, math.ceil(T / max_splits))  # cap chunks/seq
    if force_splits > 0:
        chunk = max(lo, math.ceil(T / force_splits))
    elif len(tiles) >= slots or max_splits == 1:
        chunk = T
    else:
        hi = T
        while lo < hi:  # smallest chunk whose total work fits one wave
            mid = (lo + hi) // 2
            if sum(math.ceil(t / mid) for t in tiles) <= slots:
                hi = mid
            else:
                lo = mid + 1
        chunk = lo
    work, n_splits = [], []
    for b, t in enumerate(tiles):
        k = math.ceil(t / chunk)
        n_splits.append(k)
        work.extend((b, j * chunk, min((j + 1) * chunk, t), j) for j in range(k))
    work.sort(key=lambda w: -(w[2] - w[1]))  # longest first (stable)
    if pad_to:
        assert pad_to >= len(work), f"pad_to {pad_to} < {len(work)} work items"
        work += [(0, 0, 0, 0)] * (pad_to - len(work))
    assert max(n_splits) <= max_splits
    counter_len = n_q_tiles * heads_k * len(seq_lens) if with_counter else 0
    return SplitPlan(work, n_splits, chunk, slots, counter_len)
