"""The QSA gather oracle: emulate_decode's selection path vs the dense path.

Pure-torch checks (no kernel launch) that pin the semantics the gather
loader must reproduce:

* an ALL-blocks selection at q_len == 1 is BIT-IDENTICAL to the dense path
  (the identity oracle — the loader's correctness anchor);
* a partial selection equals dense attention restricted to the selected
  rows (the top-k semantics);
* blocks are clipped to L and out-of-range ids contribute nothing;
* ``effective_seq_lens`` counts exactly the covered rows.

Needs the kernel package importable (torch + cutlass on the import chain of
``mjolnir.arch.thor.harness``) — device-first like test_runner.py; the math
itself is host-checked in tests/kernel/test_qsa.py's stub harness.

    python -m pytest -q tests/kernel/test_qsa_gather.py   # or python this file
"""

from __future__ import annotations

import math
import os
import sys

import torch


def _add_pkg_path():
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

from mjolnir.arch.thor.harness import emulate_decode  # noqa: E402
from mjolnir.arch.thor import qsa  # noqa: E402

PS = 128
SPEC = qsa.IndexerSpec()
D, HEADS_K, HEADS_Q = 64, 2, 4  # small shapes: the oracle is dtype-shape agnostic


def _fixtures(L: int, b: int = 2, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    n_pages = math.ceil(L / PS) * b + 4
    k_pool = torch.randn(n_pages, PS, HEADS_K, D, generator=g)
    v_pool = torch.randn(n_pages, PS, HEADS_K, D, generator=g)
    q = torch.randn(b, HEADS_K, HEADS_Q, D, generator=g)  # packed (b, h_k, n_q, d)
    one = torch.ones(1)
    qd = one.expand(b, HEADS_K).clone()
    kd = one.expand(b, HEADS_K).clone()
    vd = one.expand(b, HEADS_K).clone()
    seq_lens = torch.full((b,), L, dtype=torch.int32)
    bt = torch.arange(n_pages, dtype=torch.int32).reshape(b, -1)[:, : math.ceil(L / PS)]
    return q, k_pool, v_pool, bt, seq_lens, qd, kd, vd


def _run(q, k, v, bt, sl, qd, kd, vd, **kw):
    return emulate_decode(
        q,
        k,
        v,
        bt,
        sl,
        qd,
        kd,
        vd,
        scale_qs=1.0 / math.sqrt(D),
        scale_o=1.0,
        q_len=kw.pop("q_len", 1),
        chunk_tiles=2,
        fp8=False,
        **kw,
    )


def test_identity_selection_is_bit_identical_dense():
    L = 1000
    q, k, v, bt, sl, qd, kd, vd = _fixtures(L)
    o_dense, lse_dense, _ = _run(q, k, v, bt, sl, qd, kd, vd, causal=True)
    sel = qsa.identity_selection(sl, SPEC)
    o_sel, lse_sel, _ = _run(q, k, v, bt, sl, qd, kd, vd, selection_blocks=sel)
    assert torch.equal(o_dense, o_sel), (o_dense - o_sel).abs().max()
    assert torch.equal(lse_dense, lse_sel)


def test_partial_selection_equals_masked_dense():
    L = 1000
    q, k, v, bt, sl, qd, kd, vd = _fixtures(L)
    _o_dense, _, _ = _run(q, k, v, bt, sl, qd, kd, vd, causal=True)
    sel = torch.full((2, SPEC.budget_blocks), -1, dtype=torch.int32)
    sel[0, :4] = torch.tensor([0, 5, 6, 250])
    sel[1, :3] = torch.tensor([1, 2, 249])
    o_sel, _, _ = _run(q, k, v, bt, sl, qd, kd, vd, selection_blocks=sel)
    # reference: dense scores restricted to the selected rows (per sequence,
    # through that sequence's OWN block table)
    for bi in range(2):
        bt_i = bt[bi][: math.ceil(L / PS)].long()
        Kb = k[bt_i].reshape(-1, HEADS_K, D)[:L].permute(1, 0, 2)
        Vb = v[bt_i].reshape(-1, HEADS_K, D)[:L].permute(1, 0, 2)
        S = torch.einsum("gnd,gld->gnl", q[bi], Kb) / math.sqrt(D)
        ids = sel[bi][sel[bi] >= 0].long()
        mask = torch.zeros(L, dtype=torch.bool)
        for i in ids.tolist():
            mask[i * SPEC.ratio : min((i + 1) * SPEC.ratio, L)] = True
        P = torch.softmax(S[..., mask], -1)
        o_ref = torch.einsum("gnl,gld->gnd", P, Vb[:, mask, :])
        # the oracle rounds P to bf16 per chunk (the kernel's bf16-P path) —
        # the reference is exact fp32, so the tolerance is the rounding noise
        assert (o_sel[bi] - o_ref).abs().max() < 5e-3, bi


def test_blocks_clip_to_l():
    L = 10  # block 2 covers rows 8..11 -> clipped to 8..9
    q, k, v, bt, sl, qd, kd, vd = _fixtures(L, b=1)
    sel = torch.full((1, SPEC.budget_blocks), -1, dtype=torch.int32)
    sel[0, :3] = torch.tensor([0, 1, 2])
    sel[0, 5] = 999  # entirely beyond L: contributes nothing
    o_sel, _, _ = _run(q, k, v, bt, sl, qd, kd, vd, selection_blocks=sel)
    o_ref, _, _ = _run(q, k, v, bt, sl, qd, kd, vd, causal=True)
    # the selection covers ALL of L -> identical to dense
    assert torch.equal(o_sel, o_ref)
    eff = qsa.effective_seq_lens(sel, sl, SPEC)
    assert eff.tolist() == [10], eff


def test_effective_seq_lens():
    sl = torch.tensor([10, 2048])
    sel = torch.full((2, SPEC.budget_blocks), -1, dtype=torch.int32)
    sel[0, :3] = torch.tensor([0, 1, 2])  # 8 + 2 clipped rows
    sel[0, 4] = 999  # beyond L: 0
    sel[1, :2] = torch.tensor([0, 511])  # rows 0..3 + 2044..2047
    eff = qsa.effective_seq_lens(sel, sl, SPEC)
    assert eff.tolist() == [10, 8], eff


def test_gather_hnd_pool_pages_beyond_first():
    """REGRESSION (day-0 CR): the gather indexed the pool through a flat-row
    contract (stride(page) == PS * stride(row)) that the PRODUCTION HND
    transpose view violates by H_kv on the page term — every page >= 1 read
    wrong bytes, silently (random-weight smokes cannot see it; the harness
    oracle built PSH-contiguous pools where the contract holds). The kernel
    now takes the page stride explicitly; this pins the exact production
    derivation: kv HND -> transpose+split views -> the impl's stride args."""
    if not torch.cuda.is_available():
        return  # device-first; the host CI runs the math tests above
    from mjolnir import qsa_gather

    if not qsa_gather.available():
        return
    dev = torch.device("cuda")
    P, H, ps, d2 = 8, HEADS_K, PS, 2 * D
    g = torch.Generator().manual_seed(11)
    kv = (torch.randn(P, H, ps, d2, generator=g) * 0.3).to(torch.float8_e4m3fn).to(dev)
    key, value = kv.transpose(1, 2).split(D, dim=-1)  # qsa_impl._split_cache
    sel = torch.full((1, SPEC.budget_blocks), -1, dtype=torch.int32)
    sel[0, :3] = torch.tensor([0, 33, 66])  # ratio blocks landing on pages 0, 1, 2
    sl = torch.tensor([P * ps])
    bt = torch.arange(P, dtype=torch.int32).view(1, P)
    src, dst, _sbt, _pmax = qsa.scratch_plan(sel, sl, bt, SPEC, ps)
    eff = qsa.effective_seq_lens(sel, sl, SPEC)
    scratch = torch.zeros(2, ps, H, D, dtype=torch.float8_e4m3fn, device=dev)
    for kb, sk in (
        (key.view(torch.uint8), scratch[0:1].view(torch.uint8)),
        (value.view(torch.uint8), scratch[1:2].view(torch.uint8)),
    ):
        qsa_gather.gather_rows(
            kb, sk, src.to(dev), dst.to(dev), eff.to(dev), ps, kb.stride(0), kb.stride(1), kb.stride(2)
        )
    for kb, sk in ((key.contiguous(), scratch[0:1]), (value.contiguous(), scratch[1:2])):
        ref = torch.zeros_like(sk)
        qsa.scratch_copy(kb, ref, src.cpu(), dst.cpu(), eff)  # the reference
        assert torch.equal(sk.view(torch.uint8), ref.view(torch.uint8)), (
            "gather != scratch_copy on the HND pool (pages >= 1 misread)"
        )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
    print("all ok")
