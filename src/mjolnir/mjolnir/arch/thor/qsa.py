"""Qwen4 QSA (sparse-attention indexer) host-side plumbing for the Thor kernel.

Probe-gated: everything here is inert until ``MJOLNIR_ATTN_QSA`` is set —
the vLLM-side wiring arrives with the backend patch, no default flips.

Basis: ``task/research/qwen4-27b-upstream-prep/`` (the recovered Intel
``csrc/qwen38`` kernels + the qwen4_exp family config). Two additions over
today's kernel, both shape-compatible:

1. **Indexer decode** — the QSA indexer scores tokens for the sparse
   selection with a small MQA over a *second* paged pool. It reuses
   today's kernel unchanged at ``(b, h_k=1, n_q=indexer_heads*q_len,
   d=indexer_dim)``: the pool's pooled rows are the single KV "head" and
   the indexer q heads pack into ``n_q`` rows (``heads_q % heads_k == 0``,
   ``head_size % 128 == 0`` both hold).
2. **Gather decode** — the main attention reads a *selected* token set
   (budget 2048 tokens = 512 blocks x compress ratio 4) instead of the
   whole block table. Host side, the selection is coalesced into runs of
   consecutive rows per page (:func:`selection_runs`); the device loader
   that consumes runs is the only traced-body change and is flag-gated.

Under the budget (L <= budget tokens) the selection is every token, so QSA
is bit-identical to dense attention — the correctness oracle for both
paths (llama.cpp measured max logit delta 0.0 over all 2051 rows).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjolnir.runner import env_flag

# Family constants (qwen4_exp config + the recovered kernels; the indexer
# geometry is hardcoded family-wide: qsa_sycl.cpp kIndexHeads/kIndexDim).
INDEXER_HEADS = 4  # MQA query heads
INDEXER_DIM = 128  # indexer head dim (the main KV head dim is 256)
COMPRESS_RATIO = 4  # tokens pooled per compressed slot ("requires ratio=4")
BUDGET_TOKENS = 2048  # selection budget ("512 blocks or 2048 tokens")
BUDGET_BLOCKS = BUDGET_TOKENS // COMPRESS_RATIO
POOL_ROWS_PER_SLOT = 2  # raw key | pooled key (indexer_kpool_row = 2)
QSA_PAGE_SIZES = (64, 128)  # compressed-pool slots per page (qsa_sycl.cpp:2084)

POOLED_ROW = 1  # the pooled key's row index inside a slot (raw is 0)


def qsa_enabled() -> bool:
    """``MJOLNIR_ATTN_QSA`` gate (default off): the whole QSA path is opt-in."""
    return env_flag("MJOLNIR_ATTN_QSA").lower() not in ("", "0", "false", "off")


@dataclass(frozen=True)
class IndexerSpec:
    """Per-model QSA indexer geometry and its derived shapes.

    ``page_size`` is the *compressed* pool's slots-per-page; the main KV
    pool keeps its own page size (any of SUPPORTED_PAGE_SIZES, as long as
    ``main_page_size % ratio == 0`` so a block never crosses a page).
    """

    heads: int = INDEXER_HEADS
    dim: int = INDEXER_DIM
    ratio: int = COMPRESS_RATIO
    budget_tokens: int = BUDGET_TOKENS
    page_size: int = 128
    rows_per_slot: int = POOL_ROWS_PER_SLOT

    def __post_init__(self):
        assert self.ratio == COMPRESS_RATIO, "group compression requires ratio=4 (qsa_sycl.cpp:905)"
        assert self.budget_tokens % self.ratio == 0, "budget must be a whole number of blocks"
        assert self.page_size in QSA_PAGE_SIZES, f"compressed page size {self.page_size} unsupported"
        assert self.dim % 128 == 0, "indexer head dim must be a multiple of 128 (kernel headdim rule)"
        assert self.rows_per_slot == POOL_ROWS_PER_SLOT, "raw | pooled rows per slot"

    @property
    def budget_blocks(self) -> int:
        return self.budget_tokens // self.ratio

    def slots(self, tokens: int) -> int:
        """Compressed slots (KV rows of the indexer decode) for ``tokens``."""
        return math.ceil(tokens / self.ratio)

    def pool_pages(self, tokens: int) -> int:
        return math.ceil(self.slots(tokens) / self.page_size)

    def score_rows(self, q_len: int) -> int:
        """``n_q`` of the indexer decode: q tokens x indexer heads, packed."""
        return q_len * self.heads

    def check_main_page_size(self, main_page_size: int) -> None:
        assert main_page_size % self.ratio == 0, (
            f"main page size {main_page_size} must be a multiple of the compress "
            f"ratio {self.ratio} or a selected block would cross a page"
        )


def effective_seq_lens(selection_blocks: torch.Tensor, seq_lens: torch.Tensor, spec: IndexerSpec) -> torch.Tensor:
    """The gather call's ``seq_lens``: tokens actually covered by the
    selection (blocks clipped to L). The kernel's S masking uses it, so a
    block id beyond L contributes nothing and a partial last block clips."""
    ids = selection_blocks.to(torch.int64)
    valid = ids >= 0
    lo = ids.clamp_min(0) * spec.ratio
    hi = (ids.clamp_min(0) + 1) * spec.ratio
    L = seq_lens.to(ids.device, torch.int64).view(-1, 1)
    contrib = (hi.clamp_max(L) - lo.clamp_max(L)).clamp_min(0) * valid
    return contrib.sum(dim=1).to(torch.int32)


def scratch_plan(
    selection_blocks: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    spec: IndexerSpec,
    main_page_size: int,
    scratch_page_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Everything the scratch-gather copy and the follow-up dense decode need.

    The scratch packs each sequence's SELECTED rows contiguously into
    page-shaped scratch pages, so the EXISTING dense kernel + plan run on
    the scratch unchanged (zero traced-body changes for P0; the in-loader
    per-block TMA gather is the P2 refinement if ncu ever shows the copy).

    ``main_page_size`` is the POOL's page size (a hybrid stack may force it
    large — GDN mamba pages set the floor); ``scratch_page_size`` is the
    kernel's (default: same as the pool). Returns
    ``(src_rows, dst_rows, scratch_block_table, pages_per_seq_max)``:
    src_rows/dst_rows are (b, max_rows) int64 gather/scatter indices into
    the pool's and scratch's flat ``(pages * PS, ...)`` row space (rows
    beyond a sequence's count are -1: skip), and scratch_block_table is the
    (b, max_pages) int32 page table of the scratch pool.
    """
    spec.check_main_page_size(main_page_size)
    sps = scratch_page_size or main_page_size
    b = selection_blocks.shape[0]
    srcs, dsts, pages_max = [], [], 0
    for i in range(b):
        L = int(seq_lens[i])
        ids = selection_blocks[i]
        ids = ids[ids >= 0].long()
        # pool flat row of each selected token row: token row r of the
        # sequence lives at bt[r // PS] * PS + r % PS; a block's rows sit in
        # ONE page (main_page_size % ratio == 0, checked above)
        bt = block_table[i].long()
        src = []
        for bid in ids.tolist():
            tok0 = min(int(bid) * spec.ratio, L)
            n_rows = min((int(bid) + 1) * spec.ratio, L) - tok0
            if n_rows > 0:
                page = bt[tok0 // main_page_size]
                src.append(page * main_page_size + tok0 % main_page_size + torch.arange(n_rows, dtype=torch.int64))
        src = torch.cat(src) if src else torch.empty(0, dtype=torch.int64)
        dst = torch.arange(len(src), dtype=torch.int64)
        pages_max = max(pages_max, (len(src) + sps - 1) // sps)
        srcs.append(src)
        dsts.append(dst)
    max_rows = max(1, max(len(s) for s in srcs))
    src_rows = torch.full((b, max_rows), -1, dtype=torch.int64)
    dst_rows = torch.full((b, max_rows), -1, dtype=torch.int64)
    scratch_bt = torch.zeros(b, pages_max, dtype=torch.int32)
    start = 0
    for i in range(b):
        n = len(srcs[i])
        src_rows[i, :n] = srcs[i]
        dst_rows[i, :n] = dsts[i] + start * sps
        n_pages = (n + sps - 1) // sps
        scratch_bt[i, :n_pages] = torch.arange(start, start + n_pages, dtype=torch.int32)
        start += n_pages
    return src_rows, dst_rows, scratch_bt, pages_max


def scratch_copy(
    pool: torch.Tensor,
    scratch: torch.Tensor,
    src_rows: torch.Tensor,
    dst_rows: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    """Row-gather pool -> scratch on the flat ``(pages * PS, ...)`` row view.
    ``counts`` (host ints, from ``effective_seq_lens``) fix each sequence's
    valid-row count so the copy is ONE index kernel with no device sync
    (a data-dependent mask would sync every call). Works for fp8 element
    views and packed uint8 byte views alike (rows are the unit)."""
    pf = pool.view(-1, *pool.shape[2:])
    sf = scratch.view(-1, *scratch.shape[2:])
    for i, n in enumerate(counts.tolist()):
        n = int(n)
        if n:
            sf[dst_rows[i, :n].long()] = pf[src_rows[i, :n].long()]


def indexer_seq_lens(seq_lens: torch.Tensor, spec: IndexerSpec) -> torch.Tensor:
    """The indexer decode's ``seq_lens``: one KV row per compressed slot."""
    return (seq_lens.to(torch.int64) + spec.ratio - 1).div(spec.ratio).to(torch.int32)


def indexer_block_table(seq_lens: torch.Tensor, spec: IndexerSpec) -> torch.Tensor:
    """(b, max_pages) int32 block table of a CONTIGUOUSLY appended indexer
    pool: slots append in token order, so page ids are dense per sequence.
    The real pool reuses freed pages; a device-side table replaces this in
    the vLLM wiring (same shape contract)."""
    pages_per_seq = [(spec.slots(int(L)) + spec.page_size - 1) // spec.page_size for L in seq_lens.tolist()]
    max_pages = max(1, max(pages_per_seq))
    out = torch.zeros(len(seq_lens), max_pages, dtype=torch.int32)
    start = 0
    for i, n in enumerate(pages_per_seq):
        if n:
            out[i, :n] = torch.arange(start, start + n, dtype=torch.int32)
        start += n
    return out


def selection_runs(
    selected_blocks: torch.Tensor,
    seq_lens: torch.Tensor,
    spec: IndexerSpec,
    main_page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coalesce a per-sequence selection of *blocks* into per-page row runs.

    ``selected_blocks``: (b, <= budget_blocks) int32, -1 padded, UNORDERED
    (a top-k emits arbitrary order). Block ``i`` covers tokens
    ``[i*ratio, (i+1)*ratio)``; since ``main_page_size % ratio == 0`` a
    block sits inside one page, so consecutive selected blocks in a page
    merge into one contiguous run — the unit the gather loader TMA-copies.

    Returns ``(runs, counts)``: runs is (b, max_runs, 3) int32 columns
    ``(page, row, len)`` in ROWS of the main pool, len a multiple of the
    ratio; counts is (b,) the number of valid runs per sequence. Padding
    rows are zeroed.
    """
    spec.check_main_page_size(main_page_size)
    b = selected_blocks.shape[0]
    assert selected_blocks.shape[1] <= spec.budget_blocks, "selection exceeds the budget"
    max_runs = 1
    runs_out = torch.zeros(b, spec.budget_blocks, 3, dtype=torch.int32)
    counts = torch.zeros(b, dtype=torch.int32)
    for i in range(b):
        n_tok = int(seq_lens[i])
        ids = selected_blocks[i]
        ids = ids[ids >= 0].to(torch.int64)
        ids, _ = ids.sort()
        # a block may extend past L on the last partial block: clip its rows
        rows = (ids * spec.ratio).clamp_max(max(n_tok - 1, 0))
        pages = rows // main_page_size
        # run boundary: page changes or block ids are not consecutive
        new_run = torch.ones(len(ids) + 1, dtype=torch.bool)
        if len(ids):
            new_run[1:-1] = (pages[1:] != pages[:-1]) | (ids[1:] != ids[:-1] + 1)
            boundaries = new_run[:-1].nonzero().flatten()
            b_pages = pages[boundaries]
            b_rows = rows[boundaries]
            # a run ends at the page edge or at L, whichever comes first
            upper = torch.minimum(
                torch.full_like(b_rows, main_page_size) - (b_rows % main_page_size),
                torch.full_like(b_rows, n_tok) - b_rows,
            )
            ends = torch.cat([boundaries[1:], torch.tensor([len(ids)])])
            lens = ((ends - boundaries) * spec.ratio).clamp_max(upper)
            n = len(boundaries)
            runs_out[i, :n, 0] = b_pages.to(torch.int32)
            runs_out[i, :n, 1] = (b_rows % main_page_size).to(torch.int32)
            runs_out[i, :n, 2] = lens.to(torch.int32)
            counts[i] = n
            max_runs = max(max_runs, n)
    return runs_out[:, :max_runs], counts


def identity_selection(seq_lens: torch.Tensor, spec: IndexerSpec) -> torch.Tensor:
    """The under-budget selection: every block of every sequence (QSA is
    then bit-identical to dense attention — the correctness oracle)."""
    b = len(seq_lens)
    out = torch.full((b, spec.budget_blocks), -1, dtype=torch.int32)
    for i, L in enumerate(seq_lens.tolist()):
        n_blocks = min(spec.slots(int(L)), spec.budget_blocks)
        out[i, :n_blocks] = torch.arange(n_blocks, dtype=torch.int32)
    return out


def decoder_kwargs(spec: IndexerSpec) -> dict:
    """``T5DecodeRunner`` constructor args for the indexer decode: the same
    kernel, re-parameterized (MQA 4x128 over the pooled rows)."""
    return {
        "num_heads": spec.heads,
        "num_kv_heads": 1,
        "head_size": spec.dim,
        "page_size": spec.page_size,
        "q_mode": "fp8",
        "kv_format": "fp8",
    }
