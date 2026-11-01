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
        page_size,  # the pool's page size (splits the flat row index)
        src_page_stride,  # elements (bytes) between pool PAGES
        src_row_stride,  # elements between rows within a page
        src_head_stride,  # elements between heads
        dst_row_stride,
        dst_head_stride,
        D: tl.constexpr,  # bytes per (row, head), power of 2
        DREAL: tl.constexpr,  # actual bytes (D == next_pow2(DREAL))
        MAX_ROWS: tl.constexpr,
    ):
        # one program per (sequence, head, ROW): the row loop of a 2D grid
        # serializes 2048 dependent iterations behind 8 CTAs (~4 GB/s);
        # a 3D grid makes the copy bandwidth-bound again
        b = tl.program_id(0)
        h = tl.program_id(1)
        j = tl.program_id(2)
        count = tl.load(counts_ptr + b)
        if j < count:
            src_flat = tl.load(src_idx_ptr + b * MAX_ROWS + j)
            dst_flat = tl.load(dst_idx_ptr + b * MAX_ROWS + j)
            # the pool's (page, row) dims need NOT be adjacent in memory: the
            # serving pool is an HND transpose view whose stride(page) is
            # H*PS*stride(row) — decompose the flat index instead of assuming
            # stride(page) == page_size * stride(row) (which silently misreads
            # every page >= 1 when h_kv > 1)
            page = src_flat // page_size
            row = src_flat % page_size
            offs = tl.arange(0, D)
            mask = offs < DREAL
            src = src_ptr + page * src_page_stride + row * src_row_stride + h * src_head_stride
            dst = dst_ptr + dst_flat * dst_row_stride + h * dst_head_stride
            tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def gather_rows(
    pool: torch.Tensor,
    scratch: torch.Tensor,
    src_rows: torch.Tensor,
    dst_rows: torch.Tensor,
    counts: torch.Tensor,
    page_size: int,
    page_stride: int,
    row_stride: int,
    head_stride: int,
) -> None:
    """One-launch row gather. ``pool``/``scratch`` are uint8 byte views;
    the strides are in bytes of the logical ``(pages, PS, h_k, d)`` layout
    the view indexes (fp8 is 1 byte/elt, packed fp6 views are pre-flattened
    to bytes). ``counts`` must live on the device (hoist it — no per-call
    sync). The scratch must satisfy the flat-row contract
    ``stride(page) == PS * stride(row)`` (ours is contiguous with the head
    dim inside the row) — asserted, never assumed."""
    assert pool.dtype == torch.uint8 and scratch.dtype == torch.uint8, "pass uint8 byte views"
    assert src_rows.is_cuda and counts.is_cuda, "indices and counts on device"
    assert scratch.stride(0) == scratch.shape[1] * scratch.stride(1), (
        "scratch must satisfy stride(page) == PS * stride(row) (contiguous (pages, PS, h, d))"
    )
    max_rows = src_rows.shape[1]
    d = pool.shape[-1]
    d_pow2 = triton.next_power_of_2(d)
    _gather_rows_kernel[(src_rows.shape[0], pool.shape[2], max_rows)](
        pool,
        scratch,
        src_rows.to(torch.int64),
        dst_rows.to(torch.int64),
        counts.to(torch.int32),
        page_size,
        page_stride,
        row_stride,
        head_stride,
        scratch.stride(1),  # our scratch: (pages, PS, h_k, d_bytes)
        scratch.stride(2),
        D=d_pow2,
        DREAL=d,
        MAX_ROWS=max_rows,
        num_warps=1,
    )
