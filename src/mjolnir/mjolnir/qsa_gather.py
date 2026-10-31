"""The fused QSA scratch-gather copy: selection rows -> page-shaped scratch.

The scratch-gather decode copies each sequence's SELECTED rows (the QSA
indexer's top-k blocks, ratio 4) into a contiguous page-shaped scratch so
the existing dense kernel runs unchanged (``mjolnir.arch.thor.qsa`` owns
the plan math). The torch path (``qsa.scratch_copy``) is one advanced-
indexing kernel per (sequence, K/V) — ~300 µs of launch overhead per step
at batch 2. ``gather_rows`` is ONE Triton launch for the whole step: one
program per (sequence, KV head) streams its count rows of D bytes each.

Both pools are addressed through the FLAT row view: token row ``r`` of a
sequence lives at pool flat row ``page * PS + r``; the byte address is
``flat_row * row_stride + head * head_stride`` (elements of the uint8
view). The scratch is ours: contiguous ``(pages, PS, h_k, d_bytes)``.
"""

from __future__ import annotations

import torch

try:  # Triton ships with vLLM / torch on CUDA; the harness may run without it
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - no Triton: the torch path is used
    triton = None
    tl = None


def available() -> bool:
    return triton is not None


if triton is not None:

    @triton.jit
    def _gather_rows_kernel(
        src_ptr,  # uint8 view of the pool
        dst_ptr,  # uint8 view of the scratch
        src_idx_ptr,  # (b, max_rows) int64 flat pool rows
        dst_idx_ptr,  # (b, max_rows) int64 flat scratch rows
        counts_ptr,  # (b,) int32 valid rows per sequence (device)
        src_row_stride,  # elements (bytes) between flat pool rows
        src_head_stride,  # elements between heads
        dst_row_stride,
        dst_head_stride,
        D: tl.constexpr,  # bytes per (row, head), power of 2
        DREAL: tl.constexpr,  # actual bytes (D == next_pow2(DREAL))
        MAX_ROWS: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        offs = tl.arange(0, D)
        mask = offs < DREAL
        count = tl.load(counts_ptr + b)
        for j in range(count):
            src_flat = tl.load(src_idx_ptr + b * MAX_ROWS + j)
            dst_flat = tl.load(dst_idx_ptr + b * MAX_ROWS + j)
            src = src_ptr + src_flat * src_row_stride + h * src_head_stride
            dst = dst_ptr + dst_flat * dst_row_stride + h * dst_head_stride
            vals = tl.load(src + offs, mask=mask)
            tl.store(dst + offs, vals, mask=mask)


def gather_rows(
    pool: torch.Tensor,
    scratch: torch.Tensor,
    src_rows: torch.Tensor,
    dst_rows: torch.Tensor,
    counts: torch.Tensor,
    row_stride: int,
    head_stride: int,
) -> None:
    """One-launch row gather. ``pool``/``scratch`` are uint8 byte views;
    ``row_stride``/``head_stride`` in bytes of the FLAT row view (the
    logical (pages, PS, h_k, d) strides of the fp8/e2m3 tensor: fp8 is
    1 byte/elt, packed fp6 views are pre-flattened to bytes). ``counts``
    must live on the device (hoist it — no per-call sync)."""
    assert pool.dtype == torch.uint8 and scratch.dtype == torch.uint8, "pass uint8 byte views"
    assert src_rows.is_cuda and counts.is_cuda, "indices and counts on device"
    max_rows = src_rows.shape[1]
    d = pool.shape[-1]
    d_pow2 = triton.next_power_of_2(d)
    _gather_rows_kernel[(src_rows.shape[0], pool.shape[2])](
        pool,
        scratch,
        src_rows.to(torch.int64),
        dst_rows.to(torch.int64),
        counts.to(torch.int32),
        row_stride,
        head_stride,
        scratch.stride(1),  # our scratch: (pages, PS, h_k, d_bytes)
        scratch.stride(2),
        D=d_pow2,
        DREAL=d,
        MAX_ROWS=max_rows,
        num_warps=1,
    )
