"""QSA (Qwen4 sparse-attention indexer) host-side plumbing — pure torch.

Covers the :mod:`mjolnir.arch.thor.qsa` spec invariants and the selection
machinery the gather decode will consume: the under-budget identity
selection, block->run coalescing (merging, page edges, unsorted input,
last-partial-block clipping), and the indexer-pool shapes. No GPU needed —
every function is host torch; the device-side gather loader is a later,
flag-gated commit (see task/research/qwen4-27b-upstream-prep/
qwen4-27b-sm110-paper-tuning.md).

    python -m pytest -q tests/kernel/test_qsa.py      # or
    python tests/kernel/test_qsa.py                   # same, no pytest
"""

from __future__ import annotations

import os
import sys

import torch


def _add_pkg_path():
    """Make the kernel package `mjolnir` importable: an installed
    mjolnir-kernel, $MJOLNIR_SRC, or the nearest parent directory that
    holds src/mjolnir/mjolnir/ (a checkout)."""
    import importlib.util

    spec = importlib.util.find_spec("mjolnir")
    if (
        spec is not None
        and spec.origin
        and os.path.isfile(os.path.join(os.path.dirname(spec.origin), "arch/thor/decode.py"))
    ):
        return
    env = os.environ.get("MJOLNIR_SRC")
    d = os.path.dirname(os.path.abspath(__file__))
    cands = ([env] if env else []) + [os.path.join(d, "..", "..", "..", "src", "mjolnir")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "mjolnir", "arch/thor/qsa.py")):
            sys.path.insert(0, c)
            return
    raise SystemExit("the mjolnir kernel package was not found: set MJOLNIR_SRC to src/mjolnir")


_add_pkg_path()

from mjolnir.arch.thor import qsa  # noqa: E402

PS = 128  # main-KV page size (the default harness/runner page)
SPEC = qsa.IndexerSpec()


def test_spec_invariants():
    assert SPEC.heads == 4 and SPEC.dim == 128 and SPEC.ratio == 4
    assert SPEC.budget_tokens == 2048 and SPEC.budget_blocks == 512
    assert SPEC.page_size in qsa.QSA_PAGE_SIZES
    assert SPEC.slots(1) == 1 and SPEC.slots(4) == 1 and SPEC.slots(5) == 2
    assert SPEC.pool_pages(2048) == 8 and SPEC.pool_pages(1) == 1
    assert SPEC.score_rows(1) == 4 and SPEC.score_rows(5) == 20
    for bad in ({"ratio": 8}, {"page_size": 256}, {"dim": 96}, {"budget_tokens": 1025}):
        try:
            qsa.IndexerSpec(**bad)
        except AssertionError:
            continue
        raise AssertionError(f"IndexerSpec accepted {bad}")


def test_identity_selection_is_every_block():
    seq_lens = torch.tensor([128, 10, 2048])
    sel = qsa.identity_selection(seq_lens, SPEC)
    assert sel.shape == (3, SPEC.budget_blocks)
    for i, L in enumerate(seq_lens.tolist()):
        n = min(SPEC.slots(L), SPEC.budget_blocks)
        assert (sel[i, :n] == torch.arange(n)).all() and (sel[i, n:] == -1).all()


def test_selection_runs_identity_covers_the_sequence():
    seq_lens = torch.tensor([128, 10, 2048])
    runs, counts = qsa.selection_runs(qsa.identity_selection(seq_lens, SPEC), seq_lens, SPEC, PS)
    for i, L in enumerate(seq_lens.tolist()):
        assert int(runs[i, : counts[i], 2].sum()) == min(L, SPEC.budget_tokens)
        # identity selection over a contiguous prefix merges into ONE run per
        # page boundary; every run starts where the previous ended
        r = runs[i, : counts[i]]
        for j in range(1, counts[i]):
            assert int(r[j, 0]) * PS + int(r[j, 1]) == int(r[j - 1, 0]) * PS + int(r[j - 1, 1]) + int(r[j - 1, 2])


def test_selection_runs_merge_and_page_edges():
    # blocks 0-2 merge; 7 and 31 stay alone; 32-33 merge across the page
    # boundary into page 1; 100 is a lone block in page 3
    sel = torch.full((1, SPEC.budget_blocks), -1, dtype=torch.int32)
    sel[0, :8] = torch.tensor([0, 1, 2, 7, 31, 32, 33, 100])
    runs, counts = qsa.selection_runs(sel, torch.tensor([2048]), SPEC, PS)
    assert runs[0, : counts[0]].tolist() == [[0, 0, 12], [0, 28, 4], [0, 124, 4], [1, 0, 8], [3, 16, 4]]


def test_selection_runs_sorts_unordered_blocks():
    sel = torch.full((1, SPEC.budget_blocks), -1, dtype=torch.int32)
    sel[0, :3] = torch.tensor([33, 0, 32])
    runs, counts = qsa.selection_runs(sel, torch.tensor([2048]), SPEC, PS)
    assert counts[0] == 2
    assert runs[0, 0].tolist() == [0, 0, 4] and runs[0, 1].tolist() == [1, 0, 8]


def test_selection_runs_clips_last_partial_block():
    # L=10: blocks 0,1 are full (rows 0..7); block 2 covers rows 8..9 only
    sel = torch.full((1, SPEC.budget_blocks), -1, dtype=torch.int32)
    sel[0, :3] = torch.tensor([0, 1, 2])
    runs, counts = qsa.selection_runs(sel, torch.tensor([10]), SPEC, PS)
    assert counts[0] == 1 and runs[0, 0].tolist() == [0, 0, 10]


def test_selection_runs_rejects_over_budget():
    sel = torch.full((1, SPEC.budget_blocks + 1), -1, dtype=torch.int32)
    try:
        qsa.selection_runs(sel, torch.tensor([262144]), SPEC, PS)
    except AssertionError:
        return
    raise AssertionError("selection over budget accepted")


def test_indexer_shapes():
    seq_lens = torch.tensor([128, 10, 2048])
    assert qsa.indexer_seq_lens(seq_lens, SPEC).tolist() == [32, 3, 512]
    bt = qsa.indexer_block_table(seq_lens, SPEC)
    # contiguously appended pool: dense page ids per sequence
    assert bt[0, 0].item() == 0 and bt[1, 0].item() == 1 and bt[2, 0].item() == 2
    assert bt.shape[0] == 3 and bt.shape[1] >= SPEC.pool_pages(2048)


def test_gate_defaults_off():
    saved = os.environ.pop("MJOLNIR_ATTN_QSA", None)
    try:
        assert not qsa.qsa_enabled()
        os.environ["MJOLNIR_ATTN_QSA"] = "1"
        assert qsa.qsa_enabled()
        os.environ["MJOLNIR_ATTN_QSA"] = "false"
        assert not qsa.qsa_enabled()
    finally:
        if saved is not None:
            os.environ["MJOLNIR_ATTN_QSA"] = saved
        else:
            os.environ.pop("MJOLNIR_ATTN_QSA", None)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
    print("all ok")
