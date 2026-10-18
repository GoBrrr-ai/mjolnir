"""T5DecodeRunner vs the bit-faithful emulator — no vLLM needed.

Covers what the vLLM backend feeds the runner: every KV-cache layout
(vLLM main (B,H,N,2D), FlashInfer NHD/HND, FlashAttention (2,B,...)),
uint8-typed fp8 caches, a strided query (a view of a fused QKV buffer),
a 2-D output buffer, per-tensor / per-head / per-channel KV scales, page sizes 16..128,
q_len 1 and 2 (spec decode), bf16 and fp8 Q, h_k == 1, CUDA-graph padding
rows (seq_len 0), and CUDA-graph capture -> replay with a changed plan.

    python -m pytest -q tests/test_runner.py          # or
    python tests/test_runner.py                       # same, no pytest
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch


def _add_pkg_path():
    """Make the kernel package `mjolnir` importable from wherever the tests
    live: an installed mjolnir-kernel, $MJOLNIR_SRC, or the nearest parent
    directory that holds src/mjolnir/mjolnir/ (a checkout)."""
    import importlib.util

    spec = importlib.util.find_spec("mjolnir")
    if spec is not None and spec.origin and os.path.isfile(os.path.join(os.path.dirname(spec.origin), "t5_decode.py")):
        return
    env = os.environ.get("MJOLNIR_SRC") or os.environ.get("THOR_T5_SRC")
    cands = [env] if env else []
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cands += [d, os.path.join(d, "src", "mjolnir")]
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "mjolnir", "t5_decode.py")):
            sys.path.insert(0, c)
            return
    raise SystemExit("the mjolnir kernel package was not found: set MJOLNIR_SRC to src/mjolnir")


_add_pkg_path()

from mjolnir.t5_harness import emulate_decode  # noqa: E402 — the bit-faithful reference (moved out of t5_decode by the harness split)
from _util import (  # noqa: E402
    DEV,
    _fp6_cache,
    _graph_replay,
    _outlier_channel_scales,
    run_case,
)
from mjolnir.runner import (  # noqa: E402
    T5DecodeRunner,
    detect_kv_layout,
    split_kv_cache,
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
def test_vllm_main_layout_bf16():
    run_case(layout="BHN2D", page_size=16, q_mode="bf16")


def test_vllm_main_layout_fp8():
    run_case(layout="BHN2D", page_size=16, q_mode="fp8")


def test_flashinfer_nhd_per_head_fp8():
    run_case(layout="B2NHD", page_size=32, q_mode="fp8", scales="per_head", as_uint8=False)


def test_flashinfer_hnd_page128():
    run_case(layout="B2HND", page_size=128, q_mode="bf16", seq_lens=(5000, 70))


def test_flash_attn_layout_q2_bf16():
    run_case(layout="2BNHD", page_size=64, q_len=2, q_mode="bf16", out_2d=False)


def test_flash_attn_layout_q2_fp8():
    run_case(layout="2BHND", page_size=64, q_len=2, q_mode="fp8", fused_qkv=False)


def test_single_kv_head():
    run_case(layout="BHN2D", h_q=8, h_k=1, page_size=16, q_mode="fp8")


def test_head_dim_128():
    run_case(layout="BNH2D", d=128, h_q=32, h_k=8, page_size=16, q_mode="bf16")


def test_full_wave_no_split():
    # batch >= slots -> one work item per sequence, no reduction launch
    run_case(layout="BHN2D", seq_lens=tuple(range(100, 100 + 40 * 37, 37)), max_num_seqs=48, q_mode="fp8")


def test_padding_rows_are_zero():
    # CUDA-graph padding requests: seq_len 0 -> zero output, not NaN
    run_case(layout="BHN2D", seq_lens=(700, 33, 0, 0), q_mode="fp8", check_zero_rows=(2, 3))


def test_mtp_verify_shape_fp8():
    # what MTP=3 verification sends on Qwen3.8-27B: q_len 4, 24/4 heads, d 256, 128-token pages
    run_case(layout="BHN2D", h_q=24, h_k=4, page_size=128, q_len=4, q_mode="fp8", seq_lens=(5000, 300, 9000, 129))


def test_plan_margin_covers_draft_steps():
    # fused draft decode: the plan is made for seq_len + margin, the kernel runs
    # on the shorter live seq_lens; chunks past a sequence's end must be empty
    # (127 + 3 and 255 + 3 cross a 128-row tile boundary)
    run_case(layout="BHN2D", seq_lens=(127, 1, 255, 2046), q_mode="fp8", plan_margin=3)
    run_case(layout="BHN2D", seq_lens=(127, 128, 129), q_mode="fp8", plan_margin=3, page_size=128, h_q=24, h_k=4)


def test_fused_qpack_matches_torch():
    from mjolnir import qpack

    if not qpack.available():
        print("   (no Triton: skipped)")
        return
    gen = torch.Generator().manual_seed(3)
    for b, ql, h_k, h_r, d in ((1, 1, 4, 6, 256), (4, 4, 4, 6, 256), (3, 2, 8, 2, 128)):
        tokens = b * ql
        qkv = (torch.randn(tokens, (h_k * h_r + 2 * h_k) * d, generator=gen) * 3).to(torch.bfloat16).to(DEV)
        q3 = qkv[:, : h_k * h_r * d].view(tokens, h_k * h_r, d)  # strided, like vLLM's qkv.split()
        out = torch.empty(b, h_k, ql * h_r, d, dtype=torch.float8_e4m3fn, device=DEV)
        qd = torch.empty(b, h_k, dtype=torch.float32, device=DEV)
        qpack.pack_q_fp8(q3, b, ql, h_k, h_r, out, qd)
        ref, ref_qd = qpack.pack_q_fp8_reference(q3, b, ql, h_k, h_r)
        assert torch.allclose(qd, ref_qd, rtol=1e-6, atol=0), "descales differ"
        diff = (out.float() - ref.float()).abs()
        # same scale, same value -> same e4m3 code; allow one rounding step
        # where fp32 division order differs (|x| <= 448: one e4m3 ulp <= 32)
        assert (diff <= ref.float().abs() * 0.0625 + 1e-6).all(), f"pack differs (max {diff.max().item()})"
        # with a per-(q head, channel) multiplier (fp6 K channel scales folded into Q)
        mul = (torch.rand(h_k * h_r, d, generator=gen) * 20 + 0.01).to(DEV)
        qpack.pack_q_fp8(q3, b, ql, h_k, h_r, out, qd, mul)
        ref, ref_qd = qpack.pack_q_fp8_reference(q3, b, ql, h_k, h_r, mul)
        assert torch.allclose(qd, ref_qd, rtol=1e-6, atol=0), "descales differ (mul)"
        diff = (out.float() - ref.float()).abs()
        assert (diff <= ref.float().abs() * 0.0625 + 1e-6).all(), f"pack differs with mul (max {diff.max().item()})"


# ---- fp6 (e2m3) KV: the vLLM write path, the fallback expansion, T5 on it ----


def test_fp6_write_matches_reference():
    from mjolnir import kvfp6

    if not kvfp6.triton:
        print("   (no Triton: skipped)")
        return
    gen = torch.Generator().manual_seed(11)
    h_k, d, PS, T = 4, 256, 16, 37
    for head_major in (True, False):
        cache = _fp6_cache(6, h_k, PS, d, head_major)
        cache.fill_(0xA5)  # untouched slots must keep this
        key = (torch.randn(T, h_k, d, generator=gen) * 3).to(torch.bfloat16).to(DEV)
        value = (torch.randn(T, h_k, d, generator=gen) * 0.3).to(torch.bfloat16).to(DEV)
        slots = torch.randperm(6 * PS, generator=gen)[:T].to(torch.int64)
        slots[[3, 17]] = -1  # padding tokens
        ks = torch.tensor([0.5, 0.4, 0.7, 0.45], device=DEV)
        vs = torch.tensor([0.05], device=DEV)  # per tensor
        kvfp6.write_kv(key, value, cache, slots.to(DEV), ks, vs)
        rb = kvfp6.row_bytes(d)
        ref_k = kvfp6.quantize_rows_reference(key.float(), ks)
        ref_v = kvfp6.quantize_rows_reference(value.float(), vs)
        got = cache.cpu()
        for t, slot in enumerate(slots.tolist()):
            if slot < 0:
                continue
            blk, off = divmod(slot, PS)
            assert torch.equal(got[blk, :, off, :rb], ref_k[t]), f"K row {t} differs"
            assert torch.equal(got[blk, :, off, rb:], ref_v[t]), f"V row {t} differs"
        written = {s for s in slots.tolist() if s >= 0}
        untouched = [s for s in range(6 * PS) if s not in written]
        for slot in untouched[:20]:
            blk, off = divmod(slot, PS)
            assert (got[blk, :, off] == 0xA5).all(), "wrote a slot it was not given"


def test_fp6_expand_is_exact():
    from mjolnir import kvfp6

    if not kvfp6.triton:
        print("   (no Triton: skipped)")
        return
    gen = torch.Generator().manual_seed(12)
    h_k, d, PS = 2, 256, 32
    cache = _fp6_cache(9, h_k, PS, d, head_major=False)
    cache.copy_(torch.randint(0, 256, cache.shape, generator=gen, dtype=torch.uint8).to(DEV))
    src = torch.tensor([7, 2, 2, 0], dtype=torch.int32, device=DEV)
    out = torch.full((6, h_k, PS, 2 * d), 0x7F, dtype=torch.uint8, device=DEV)
    kvfp6.expand_pages_to_fp8(cache, src, out, d)
    rb = kvfp6.row_bytes(d)
    for i, page in enumerate(src.tolist()):
        want_k = kvfp6.decode_rows_reference(cache[page, :, :, :rb], d)
        want_v = kvfp6.decode_rows_reference(cache[page, :, :, rb:], d)
        got = out[i].view(torch.float8_e4m3fn).float().cpu()
        assert torch.equal(got[..., :d], want_k) and torch.equal(got[..., d:], want_v), f"page {i} differs"
    assert (out[4:] == 0x7F).all(), "expanded past the pages it was given"


def test_fp6_write_per_channel():
    from mjolnir import kvfp6

    if not kvfp6.triton:
        print("   (no Triton: skipped)")
        return
    gen = torch.Generator().manual_seed(21)
    h_k, d, PS, T = 4, 256, 16, 29
    cache = _fp6_cache(4, h_k, PS, d)
    ks = _outlier_channel_scales(gen, h_k, d)
    vs = torch.rand(h_k, d, generator=gen) * 0.2 + 0.01
    key = (torch.randn(T, h_k, d, generator=gen) * ks * 7.5 / 3).to(torch.bfloat16).to(DEV)
    value = (torch.randn(T, h_k, d, generator=gen) * vs * 7.5 / 3).to(torch.bfloat16).to(DEV)
    slots = torch.randperm(4 * PS, generator=gen)[:T].to(torch.int64)
    slots[5] = -1
    kvfp6.write_kv(key, value, cache, slots.to(DEV), ks.to(DEV), vs.to(DEV))
    rb = kvfp6.row_bytes(d)
    ref_k = kvfp6.quantize_rows_reference(key.float(), ks)
    ref_v = kvfp6.quantize_rows_reference(value.float(), vs)
    got = cache.cpu()
    for t, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        blk, off = divmod(slot, PS)
        assert torch.equal(got[blk, :, off, :rb], ref_k[t]), f"K row {t} differs"
        assert torch.equal(got[blk, :, off, rb:], ref_v[t]), f"V row {t} differs"
    # per-channel resolution: small channels survive next to the outliers
    k_back = kvfp6.decode_rows_reference(got[slots[0] // PS, :, slots[0] % PS, :rb], d) * ks
    rel = ((k_back - key[0].float().cpu()).norm() / key[0].float().cpu().norm()).item()
    assert rel < 0.08, f"per-channel K round trip rel err {rel:.3f}"


def test_fp6_expand_with_ratio():
    from mjolnir import kvfp6

    if not kvfp6.triton:
        print("   (no Triton: skipped)")
        return
    gen = torch.Generator().manual_seed(22)
    h_k, d, PS = 2, 256, 32
    cache = _fp6_cache(5, h_k, PS, d, head_major=False)
    cache.copy_(torch.randint(0, 256, cache.shape, generator=gen, dtype=torch.uint8).to(DEV))
    ch_k = _outlier_channel_scales(gen, h_k, d)
    ch_v = torch.rand(h_k, d, generator=gen) + 0.1
    kr = ch_k / (ch_k.amax(1, keepdim=True) * 7.5 / 448)
    vr = ch_v / (ch_v.amax(1, keepdim=True) * 7.5 / 448)
    src = torch.tensor([4, 1, 1], dtype=torch.int32, device=DEV)
    out = torch.full((4, h_k, PS, 2 * d), 0x7F, dtype=torch.uint8, device=DEV)
    kvfp6.expand_pages_to_fp8(cache, src, out, d, kr.to(DEV), vr.to(DEV))
    rb = kvfp6.row_bytes(d)
    for i, page in enumerate(src.tolist()):
        want_k = (kvfp6.decode_rows_reference(cache[page, :, :, :rb], d) * kr[:, None, :]).to(torch.float8_e4m3fn)
        want_v = (kvfp6.decode_rows_reference(cache[page, :, :, rb:], d) * vr[:, None, :]).to(torch.float8_e4m3fn)
        got = out[i].view(torch.float8_e4m3fn).float().cpu()
        assert torch.equal(got[..., :d], want_k.float()), f"K page {i} differs"
        assert torch.equal(got[..., d:], want_v.float()), f"V page {i} differs"
    assert (out[3:] == 0x7F).all(), "expanded past the pages it was given"


def test_fp6_runner_per_channel():
    """T5 on an e2m3 cache with per-channel scales, applied as the backend does:
    K channel scales folded into Q (q_mul), V's into O (o_mul), kernel descales 1.
    Checked against exact attention on the dequantized cache (code * s_c)."""
    from mjolnir import kvfp6

    if not kvfp6.triton:
        print("   (no Triton: skipped)")
        return
    gen = torch.Generator().manual_seed(23)
    h_q, h_k, d, PS, ql = 24, 4, 256, 128, 4
    h_r = h_q // h_k
    seq_lens = (900, 130)
    b = len(seq_lens)
    n_blocks = [math.ceil(L / PS) for L in seq_lens]
    num_pages = sum(n_blocks) + 1
    bt = torch.zeros(4, max(n_blocks) + 1, dtype=torch.int32)
    perm = torch.randperm(num_pages, generator=gen).tolist()
    i = 0
    for r, nb in enumerate(n_blocks):
        bt[r, :nb] = torch.tensor(perm[i : i + nb])
        i += nb
    block_table = bt.to(DEV)
    sl = torch.zeros(4, dtype=torch.int32)
    sl[:b] = torch.tensor(seq_lens)
    cache = _fp6_cache(num_pages, h_k, PS, d)
    ks = _outlier_channel_scales(gen, h_k, d, base=0.3, outlier=40.0)
    vs = torch.rand(h_k, d, generator=gen) * 0.1 + 0.02
    for r, L in enumerate(seq_lens):
        slots = torch.tensor([int(bt[r, p // PS]) * PS + p % PS for p in range(L)], device=DEV)
        key = (torch.randn(L, h_k, d, generator=gen) * ks * 2).to(torch.bfloat16).to(DEV)
        value = (torch.randn(L, h_k, d, generator=gen) * vs * 2).to(torch.bfloat16).to(DEV)
        kvfp6.write_kv(key, value, cache, slots, ks.to(DEV), vs.to(DEV))
    rb = kvfp6.row_bytes(d)
    k_cache, v_cache = split_kv_cache(cache, "BHN2D", rb)
    runner = T5DecodeRunner(
        num_heads=h_q,
        num_kv_heads=h_k,
        head_size=d,
        page_size=PS,
        out_dtype=torch.bfloat16,
        max_num_seqs=4,
        max_q_len=ql,
        q_mode="fp8",
        kv_format="e2m3",
    )
    tokens = b * ql
    # small Q on the outlier channels (as in real models: q . k stays O(10))
    query = (torch.randn(tokens, h_q, d, generator=gen) / ks.repeat_interleave(h_r, 0) * 0.5).to(torch.bfloat16)
    query = query.to(DEV)
    out = torch.full((tokens, h_q * d), float("nan"), dtype=torch.bfloat16, device=DEV)
    one = torch.ones(1, device=DEV)
    plan = runner.plan(sl.tolist(), b, ql, block_table, sl.to(DEV))
    runner.ensure_compiled(ql, plan.with_reduction)
    q_mul = ks.repeat_interleave(h_r, 0).contiguous().to(DEV)
    o_mul = vs.repeat_interleave(h_r, 0).contiguous().to(DEV)
    runner.run(plan, query, out, k_cache, v_cache, one, one, 1.0 / math.sqrt(d), q_mul=q_mul, o_mul=o_mul)
    torch.cuda.synchronize()
    # exact reference: causal attention over the dequantized cache
    k_vals = kvfp6.decode_rows_reference(k_cache.contiguous(), d) * ks[None, None]  # (pages, PS, h_k, d)
    v_vals = kvfp6.decode_rows_reference(v_cache.contiguous(), d) * vs[None, None]
    q = query.float().cpu().view(b, ql, h_k, h_r, d)
    o = out.view(tokens, h_q, d).float().cpu().view(b, ql, h_k, h_r, d)
    worst = 0.0
    for r, L in enumerate(seq_lens):
        pages = bt[r, : n_blocks[r]].long()
        K = k_vals[pages].reshape(-1, h_k, d)[:L]
        V = v_vals[pages].reshape(-1, h_k, d)[:L]
        for qi in range(ql):
            n = L - ql + qi + 1  # causal within the MTP verify window
            s = torch.einsum("khd,tkd->kht", q[r, qi], K[:n]) / math.sqrt(d)
            ref = torch.einsum("kht,tkd->khd", s.softmax(-1), V[:n])
            err = ((o[r, qi] - ref).norm() / ref.norm()).item()
            worst = max(worst, err)
    assert worst < 0.05, f"per-channel e2m3 T5 vs exact attention: rel err {worst:.4f}"


def test_fp6_runner_vs_emulator():
    """T5 on an e2m3 cache written by the vLLM write path, in vLLM's (B,H,N,2*row)
    layout, MTP verify shape, per-head scales — vs the emulator on the decoded values."""
    from mjolnir import kvfp6

    if not kvfp6.triton:
        print("   (no Triton: skipped)")
        return
    gen = torch.Generator().manual_seed(13)
    h_q, h_k, d, PS, ql = 24, 4, 256, 128, 4
    seq_lens = (3000, 129, 700)
    b = len(seq_lens)
    n_blocks = [math.ceil(L / PS) for L in seq_lens]
    num_pages = sum(n_blocks) + 2
    perm = torch.randperm(num_pages, generator=gen).tolist()
    bt = torch.zeros(8, max(n_blocks) + 1, dtype=torch.int32)
    i = 0
    for r, nb in enumerate(n_blocks):
        bt[r, :nb] = torch.tensor(perm[i : i + nb])
        i += nb
    block_table = bt.to(DEV)
    sl = torch.zeros(8, dtype=torch.int32)
    sl[:b] = torch.tensor(seq_lens)
    seq_lens_d = sl.to(DEV)
    cache = _fp6_cache(num_pages, h_k, PS, d)
    ks = torch.tensor([0.6, 0.5, 0.8, 0.55], device=DEV)
    vs = torch.tensor([0.12, 0.1, 0.15, 0.09], device=DEV)
    for r, L in enumerate(seq_lens):  # write every token of every sequence
        slots = torch.tensor([int(bt[r, p // PS]) * PS + p % PS for p in range(L)], device=DEV)
        key = (torch.randn(L, h_k, d, generator=gen) * 2).to(torch.bfloat16).to(DEV)
        value = (torch.randn(L, h_k, d, generator=gen) * 0.5).to(torch.bfloat16).to(DEV)
        kvfp6.write_kv(key, value, cache, slots, ks, vs)
    rb = kvfp6.row_bytes(d)
    assert detect_kv_layout(cache.shape, h_k, PS, rb) == "BHN2D"
    k_cache, v_cache = split_kv_cache(cache, "BHN2D", rb)
    runner = T5DecodeRunner(
        num_heads=h_q,
        num_kv_heads=h_k,
        head_size=d,
        page_size=PS,
        out_dtype=torch.bfloat16,
        max_num_seqs=8,
        max_q_len=ql,
        q_mode="fp8",
        kv_format="e2m3",
    )
    tokens = b * ql
    qkv = (torch.randn(tokens, (h_q + 2 * h_k) * d, generator=gen) * 0.5).to(torch.bfloat16).to(DEV)
    query = qkv[:, : h_q * d].view(tokens, h_q, d)
    out = torch.full((tokens + 2, h_q * d), float("nan"), dtype=torch.bfloat16, device=DEV)
    plan = runner.plan(sl.tolist(), b, ql, block_table, seq_lens_d, margin=3)
    runner.ensure_compiled(ql, plan.with_reduction)
    runner.run(plan, query, out, k_cache, v_cache, ks, vs, 1.0 / math.sqrt(d))
    torch.cuda.synchronize()
    assert torch.isnan(out[tokens:]).all(), "wrote past the batch"
    k_vals = kvfp6.decode_rows_reference(k_cache.contiguous(), d)  # (pages, PS, h_k, d) code values
    v_vals = kvfp6.decode_rows_reference(v_cache.contiguous(), d)
    n_q = ql * (h_q // h_k)
    q_packed = runner.q_pack[: b * h_k * n_q * d].view(b, h_k, n_q, d).float().cpu()
    qd = runner.q_descale[:b].float().cpu()
    o_emu, _, _ = emulate_decode(
        q_packed,
        k_vals,
        v_vals,
        block_table[:b].cpu(),
        seq_lens_d[:b].cpu(),
        qd,
        ks.cpu()[None].expand(b, h_k).clone(),
        vs.cpu()[None].expand(b, h_k).clone(),
        1.0 / math.sqrt(d),
        1.0,
        ql,
        plan.chunk_tiles,
        True,
    )
    ref = o_emu.view(b, h_k, ql, h_q // h_k, d).permute(0, 2, 1, 3, 4).reshape(tokens, h_q, d)
    o = out[:tokens].view(tokens, h_q, d).float().cpu()
    err = (o - ref).abs().max().item()
    tol = 2e-2 * max(1.0, ref.abs().max().item())
    assert math.isfinite(err) and err <= tol, f"e2m3 T5 vs emulator: max|err| {err:.4g} > {tol:.4g}"


def test_kernel_disk_cache_roundtrip():
    """A variant exported to the on-disk kernel cache and loaded back in a
    fresh process state gives bit-identical output to the in-process compile."""
    import tempfile

    from mjolnir import runner as R

    gen = torch.Generator().manual_seed(31)
    h_q, h_k, d, PS, ql = 16, 2, 256, 16, 1
    seq_lens = (300, 77)
    b = len(seq_lens)
    n_blocks = [math.ceil(L / PS) for L in seq_lens]
    num_pages = sum(n_blocks) + 1
    bt = torch.zeros(4, max(n_blocks) + 1, dtype=torch.int32)
    i = 0
    for r, nb in enumerate(n_blocks):
        bt[r, :nb] = torch.arange(i, i + nb)
        i += nb
    sl = torch.zeros(4, dtype=torch.int32)
    sl[:b] = torch.tensor(seq_lens)
    kv = torch.randn(num_pages, h_k, PS, 2 * d, generator=gen).to(torch.float8_e4m3fn).to(DEV)
    k_cache, v_cache = split_kv_cache(kv, "BHN2D", d)
    query = (torch.randn(b * ql, h_q, d, generator=gen) * 0.5).to(torch.bfloat16).to(DEV)
    one = torch.ones(1, device=DEV)

    def run_once():
        runner = T5DecodeRunner(
            num_heads=h_q,
            num_kv_heads=h_k,
            head_size=d,
            page_size=PS,
            out_dtype=torch.bfloat16,
            max_num_seqs=4,
            max_q_len=ql,
            q_mode="fp8",
        )
        out = torch.zeros(b * ql, h_q * d, dtype=torch.bfloat16, device=DEV)
        plan = runner.plan(sl.tolist(), b, ql, bt.to(DEV), sl.to(DEV))
        runner.ensure_compiled(ql, plan.with_reduction)
        runner.run(plan, query, out, k_cache, v_cache, one, one, 1.0 / math.sqrt(d))
        torch.cuda.synchronize()
        return out.clone()

    saved = (dict(R._COMPILED), os.environ.get("MJOLNIR_ATTN_KERNEL_CACHE"), R._disk_cache_ok)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.environ["MJOLNIR_ATTN_KERNEL_CACHE"] = tmp
            R._disk_cache_ok = True
            R._COMPILED.clear()
            c0 = dict(R.CACHE_STATS)
            first = run_once()
            assert R.CACHE_STATS["compiled"] > c0["compiled"], "expected a fresh compile"
            files = list(Path(tmp).rglob("*.o"))
            assert files, f"nothing exported to {tmp} (cache off: {not R._disk_cache_ok})"
            R._COMPILED.clear()  # a new server start: only the disk cache is left
            c1 = dict(R.CACHE_STATS)
            second = run_once()
            assert R.CACHE_STATS["loaded"] > c1["loaded"], f"not loaded from disk (cache off: {not R._disk_cache_ok})"
            assert R.CACHE_STATS["compiled"] == c1["compiled"], "recompiled despite a cache entry"
            assert torch.equal(first, second), "disk-cached kernel output differs"
        finally:
            R._COMPILED.clear()
            R._COMPILED.update(saved[0])
            R._disk_cache_ok = saved[2]
            if saved[1] is None:
                os.environ.pop("MJOLNIR_ATTN_KERNEL_CACHE", None)
            else:
                os.environ["MJOLNIR_ATTN_KERNEL_CACHE"] = saved[1]


# ---- opt-in lm_head experiments (mjolnir.lmhead) ----
def test_lmhead_fp8_matches_reference():
    from mjolnir import lmhead

    gen = torch.Generator().manual_seed(41)
    N, K = 4160, 512  # N not a multiple of the 64-row tile
    w = (torch.randn(N, K, generator=gen) * 0.02).to(torch.bfloat16).to(DEV)
    w8, scale = lmhead.quantize_rows(w)
    for M in (1, 4, 16, 40, 100):  # 100 > SMALL_M: the dequantized-slab path
        x = torch.randn(M, K, generator=gen).to(torch.bfloat16).to(DEV)
        got = lmhead.fp8_linear(x, w8, scale).float()
        ref = lmhead.fp8_linear_reference(x, w8, scale).float()
        exact = x.float() @ w.float().t()
        e_ref = ((got - ref).norm() / ref.norm()).item()
        e_bf16 = ((got - exact).norm() / exact.norm()).item()
        assert e_ref < 1e-2, f"M={M}: fp8 head vs its reference {e_ref:.4f}"
        assert e_bf16 < 0.05, f"M={M}: fp8 head vs bf16 {e_bf16:.4f}"
        top = (got.argmax(-1) == exact.argmax(-1)).float().mean().item()
        assert top >= 0.75, f"M={M}: argmax agreement {top:.2f}"


def test_lmhead_fp8_is_faster_than_bf16():
    """The point of the experiment, at the real shape (Qwen3.8 lm_head)."""
    from mjolnir import lmhead

    N, K = 248320, 5120
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.02
    w8, scale = lmhead.quantize_rows(w)
    out = []
    for M in (1, 4, 16):
        x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)

        def t(fn, it=20):
            fn()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(it):
                fn()
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b) / it

        bf = t(lambda x=x, w=w: torch.nn.functional.linear(x, w))
        f8 = t(lambda x=x, w8=w8, scale=scale: lmhead.fp8_linear(x, w8, scale))
        out.append((M, bf, f8))
        print(f"   lm_head M={M}: bf16 {bf:.2f} ms ({N * K * 2 / bf / 1e6:.0f} GB/s) · fp8 {f8:.2f} ms "
              f"({N * K / f8 / 1e6:.0f} GB/s) · {bf / f8:.2f}x")  # fmt: skip
    assert all(f8 < bf for _, bf, f8 in out), f"the fp8 head is not faster: {out}"


def test_lmhead_quantize_rows_chunked_matches_whole():
    from mjolnir import lmhead

    w = (torch.randn(1000, 256, device=DEV) * 0.05).to(torch.bfloat16)
    q1, s1 = lmhead.quantize_rows(w, chunk=7)
    q2, s2 = lmhead.quantize_rows(w, chunk=4096)
    assert torch.equal(q1.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(s1, s2)
    assert lmhead._self_test(w, q1, s1) < 0.05


def test_draft_vocab_trim():
    from mjolnir import lmhead

    class _LP:
        org_vocab_size, scale, soft_cap = 1000, 1.0, None

    class _Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter((torch.randn(1024, 64) * 0.1).to(torch.bfloat16).to(DEV), False)

    class _Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head, self.logits_processor = _Head(), _LP()

        def compute_logits(self, h):
            return (h @ self.lm_head.weight.t())[..., :1000]

        def get_top_tokens(self, h):
            return self.compute_logits(h).argmax(-1)

    d = _Draft()
    keep = list(range(0, 1000, 3))
    h = torch.randn(5, 64, device=DEV, dtype=torch.bfloat16)
    full = d.compute_logits(h)
    assert lmhead.install_draft_vocab(d, keep)
    trimmed = d.compute_logits(h)
    k = torch.tensor(keep, device=DEV)
    # same rows of the same weight (different GEMM shapes may round differently)
    assert torch.allclose(trimmed[:, k].float(), full[:, k].float(), rtol=2e-2, atol=2e-2), "kept logits differ"
    drop = torch.ones(1000, dtype=torch.bool, device=DEV)
    drop[k] = False
    assert torch.isinf(trimmed[:, drop]).all() and (trimmed[:, drop] < 0).all()
    assert torch.equal(d.get_top_tokens(h), trimmed.argmax(-1)), "top tokens must index the full vocabulary"
    assert lmhead.pad_vocab([5, 3, 5, 2000], 6, 10) == [5, 3, 0, 1, 2, 4], "pad with the lowest unused ids"
    assert lmhead.pad_vocab([5, 3, 1], 2, 10) == [5, 3]


def test_draft_vocab_quantized_head():
    """A quantized head stores its weight transposed (hidden, vocab), not
    (vocab, hidden); install_draft_vocab dequantizes it to [out, in] first, so
    it keeps the right rows instead of building a (kept, vocab) matrix.
    Regression for the ~40 GB / OOM on ModelOpt-quantized heads (issue #25)."""
    from mjolnir import lmhead

    class _QM:  # a non-"Unquantized" quant_method (e.g. ModelOptLinearMethod)
        pass

    class _LP:
        org_vocab_size, scale, soft_cap = 1000, 1.0, None

    class _HeadQ:  # stores the weight transposed (hidden, vocab), like per-tensor FP8
        def __init__(self, wT):
            self.weight, self.quant_method = wT, _QM()

    vocab, hidden = 1000, 64
    W = (torch.randn(vocab, hidden, device=DEV) * 0.1).to(torch.bfloat16)
    d = torch.nn.Module()
    d.logits_processor = _LP()
    d.lm_head = _HeadQ(W.t().contiguous())  # (hidden, vocab) — the layout that used to blow up
    keep = list(range(0, 1000, 3))

    saved = lmhead._dequant_weight
    saved_qr = lmhead.quantize_rows
    lmhead._dequant_weight = lambda h, dt: W.to(dt)  # ty: ignore[invalid-assignment] — test stub
    qr_calls = []

    def _qr_spy(w, *a, **kw):  # proves the quantized path re-quantizes to fp8, never bf16
        qr_calls.append(w.shape)
        return saved_qr(w, *a, **kw)

    try:
        lmhead.quantize_rows = _qr_spy  # ty: ignore[invalid-assignment] — test stub
        assert lmhead.install_draft_vocab(d, keep)
    finally:
        lmhead._dequant_weight = saved
        lmhead.quantize_rows = saved_qr

    assert len(qr_calls) == 1, "a quantized head must re-quant the trimmed rows to fp8"
    assert qr_calls[0] == (len(keep), hidden), "the fp8 rows must be the kept tokens"

    h = torch.randn(5, hidden, device=DEV, dtype=torch.bfloat16)
    got = d.compute_logits(h)
    k = torch.tensor(keep, device=DEV)
    ref = (h @ W.t())[:, k]
    # fp8 re-quant rounds the kept rows (the spy above proves fp8 is used)
    assert torch.allclose(got[:, k].float(), ref.float(), rtol=1e-1, atol=1e-1), "kept logits differ"
    drop = torch.ones(vocab, dtype=torch.bool, device=DEV)
    drop[k] = False
    assert torch.isinf(got[:, drop]).all() and (got[:, drop] < 0).all(), "dropped tokens must be -inf"


def test_draft_vocab_native_head():
    """VT a native-quant (NVFP4) drafter head keeps the kept rows bit-exact to the
    source head — no dequant->bf16->fp8 re-rounding. That's what lets a 4-bit
    drafter head be vocab-trimmed without perturbing its logits (acceptance) and
    keeps the draft step a 4-bit GEMM. The fp8 path re-rounds the kept rows.
    CUDA-only (the in-kernel GEMM is Triton)."""
    from mjolnir import lmhead
    from mjolnir.lmhead import _nvfp4_native_gemm

    torch.manual_seed(0)
    vocab, hidden = 96, 64  # hidden must divide 64 (the GEMM's K-block tiling)
    packed_cpu = torch.randint(0, 256, (vocab, hidden // 2), dtype=torch.uint8)
    bs_f = (torch.rand(vocab, hidden // 16) + 0.5).to(torch.float8_e4m3fn).float()  # the e4m3 block scales
    bs = bs_f.to(torch.float8_e4m3fn).to(DEV)
    packed = packed_cpu.to(DEV)
    gs = torch.tensor(1.0, device=DEV)
    LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    # exact dequant BY CONSTRUCTION (independent of the torch reference): the
    # source head's rows, nibble -> e2m1 -> * block scale -> * global
    W = torch.empty(vocab, hidden)
    for i in range(vocab):
        for j in range(hidden):
            c = packed_cpu[i, j // 2].item()
            c = (c & 0xF) if j % 2 == 0 else (c >> 4) & 0xF
            W[i, j] = LUT[c] * bs_f[i, j // 16].item() * 1.0
    idx = torch.tensor(sorted(set(range(0, vocab, 3))), device=DEV)
    h = torch.randn(3, hidden, device=DEV, dtype=torch.bfloat16)
    ref = (h.float() @ W.to(DEV).index_select(0, idx).float().t()).to(torch.bfloat16)
    # native VT = the source head's kept rows, bit-exact (the capability we need)
    got = _nvfp4_native_gemm(h, packed.index_select(0, idx), bs.index_select(0, idx), gs)
    assert got is not None, "native in-kernel GEMM unavailable (triton?)"
    assert torch.allclose(got.float(), ref.float(), rtol=5e-2, atol=5e-2), (
        "native VT must equal the source head's kept rows"
    )
    # the fp8 VT re-rounds the kept rows -> that perturbation is what native avoids
    sub8, sub_scale = lmhead.quantize_rows(W.to(DEV).index_select(0, idx))
    fp8_deq = sub8.float() * sub_scale[:, None]
    assert not torch.equal(fp8_deq, W.to(DEV).index_select(0, idx).float()), (
        "fp8 re-quant of a 4-bit head is expected to round (why native is needed)"
    )


def test_fp8_native_slice():
    """VT of an already-fp8 head keeps the kept rows' e4m3 bytes + scale (no
    dequant->re-quant). Per-row and per-tensor scales slice natively; a
    block-scale (2-D) scale is not row-aligned -> None (the caller dequants).
    CPU."""
    from mjolnir.lmhead import _fp8_native_slice

    torch.manual_seed(0)
    vocab, hidden = 64, 32
    idx = torch.tensor(sorted(set(range(0, vocab, 3))))
    w8 = torch.randn(vocab, hidden).to(torch.float8_e4m3fn)

    class _H:
        pass

    # per-row (N,) scale: row-aligned with the weights -> sliced too
    h = _H()
    h.weight = w8
    h.weight_scale = torch.rand(vocab)
    k8, ksc = _fp8_native_slice(h, idx)
    assert k8 is not None and k8.shape[0] == idx.numel()
    assert torch.equal(k8, w8.index_select(0, idx).contiguous())
    assert torch.equal(ksc, h.weight_scale.index_select(0, idx).float().contiguous())
    # per-tensor scalar scale: broadcast to a per-row scale (still the native bytes)
    h2 = _H()
    h2.weight = w8
    h2.weight_scale = torch.tensor(1.5)
    k8b, kscb = _fp8_native_slice(h2, idx)
    assert k8b is not None and k8b.shape[0] == idx.numel()
    assert kscb.shape[0] == idx.numel() and (kscb == 1.5).all()
    # block-scale (2-D) scale: not row-aligned -> None (the dequant fallback)
    h3 = _H()
    h3.weight = w8
    h3.weight_scale = torch.rand(vocab, hidden // 16)
    assert _fp8_native_slice(h3, idx) is None
    # a bf16 weight is not fp8-native
    h4 = _H()
    h4.weight = torch.randn(vocab, hidden)
    h4.weight_scale = torch.rand(vocab)
    assert _fp8_native_slice(h4, idx) is None


def test_draft_vocab_dispatch():
    """VT dispatches on the drafter head's original precision (no bf16 upscale):
    a bf16 head stays bf16, but the single precision exception (bf16 -> fp8)
    engages only when the fp8 env is set. The kept logits match the source
    head's kept rows (bit-exact for bf16, a tiny re-round for fp8). CPU (the fp8
    path uses the dequant-slab GEMM at M > 64)."""
    from mjolnir.lmhead import ENV_LM_HEAD, install_draft_vocab

    torch.manual_seed(0)
    vocab, hidden = 256, 64
    ids = sorted(set(range(0, vocab, 3)))
    W = torch.randn(vocab, hidden)
    idx = torch.tensor(ids)
    drop = torch.tensor(sorted(set(range(vocab)) - set(ids)))
    h = torch.randn(70, hidden)  # 70 rows: > SMALL_M so the fp8 path uses the dequant slab
    kept_ref = h.float() @ W.index_select(0, idx).float().t()

    class _LP:
        org_vocab_size, scale, soft_cap = vocab, 1.0, None

    class _Head:
        def __init__(self):
            self.weight = W
            self.quant_method = None
            self.device = W.device

    class _Draft:
        def __init__(self):
            self.lm_head = _Head()
            self.logits_processor = _LP()

    def kept(out):
        return out.index_select(1, idx)

    # no fp8 env -> a bf16 head stays bf16 (bit-exact to the source head's kept rows).
    # This runner is a plain script (not pytest), so the env var is managed by hand.
    saved = os.environ.get(ENV_LM_HEAD)
    try:
        os.environ.pop(ENV_LM_HEAD, None)
        d = _Draft()
        assert install_draft_vocab(d, ids)
        out = d.compute_logits(h)
        assert torch.isinf(out.index_select(1, drop)).all(), "non-kept tokens must be -inf"
        assert torch.allclose(kept(out), kept_ref, rtol=5e-2, atol=5e-2), (
            "bf16 VT must equal the source head's kept rows"
        )
        # the single exception: bf16 -> fp8 when the fp8 env is set (a tiny re-round)
        os.environ[ENV_LM_HEAD] = "fp8"
        d2 = _Draft()
        assert install_draft_vocab(d2, ids)
        out8 = d2.compute_logits(h)
        assert torch.isinf(out8.index_select(1, drop)).all(), "non-kept tokens must be -inf"
        # the fp8 re-round is ~2-6% per element (e4m3); compare mean error to the
        # logits' own magnitude so the check is scale-robust (a broken re-round would
        # blow this up), and confirm it actually differs from the bit-exact bf16 VT
        diff8 = (kept(out8) - kept_ref).abs()
        assert diff8.mean() < 0.1 * kept_ref.abs().mean(), "bf16->fp8 VT should be a close re-round of the source head"
        assert not torch.equal(kept(out8), kept(out)), "the fp8 VT re-rounds (a bit-exact bf16 VT would not)"
    finally:
        if saved is None:
            os.environ.pop(ENV_LM_HEAD, None)
        else:
            os.environ[ENV_LM_HEAD] = saved


def test_cuda_graph_replay_with_new_plan():
    _graph_replay(pdl=False, kv_evict_first=False)


def test_cuda_graph_replay_pdl_evict_first():
    # the vLLM fp8 defaults, captured with programmatic dependencies
    _graph_replay(pdl=True, kv_evict_first=True)


def test_pdl():
    run_case(layout="BHN2D", page_size=16, q_mode="fp8", pdl=True)


def test_l2_prefetch_page16():
    run_case(layout="BHN2D", page_size=16, q_mode="fp8", l2_prefetch=2, seq_lens=(9000, 1, 2049, 300))


def test_l2_prefetch_page128_bf16_pdl():
    run_case(layout="B2HND", page_size=128, q_mode="bf16", l2_prefetch=3, pdl=True, seq_lens=(20000, 70))


def test_kv_evict_first():
    run_case(layout="BHN2D", page_size=16, q_mode="bf16", kv_evict_first=True)


def test_layout_detection():
    H, N, D = 4, 16, 256
    assert detect_kv_layout((10, H, N, 2 * D), H, N, D) == "BHN2D"
    assert detect_kv_layout((10, N, H, 2 * D), H, N, D) == "BNH2D"
    assert detect_kv_layout((10, 2, N, H, D), H, N, D) == "B2NHD"
    assert detect_kv_layout((10, 2, H, N, D), H, N, D) == "B2HND"
    assert detect_kv_layout((2, 10, N, H, D), H, N, D) == "2BNHD"
    assert detect_kv_layout((2, 10, H, N, D), H, N, D) == "2BHND"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    only = sys.argv[1:]
    failed = 0
    for name, fn in tests:
        if only and not any(o in name for o in only):
            continue
        try:
            fn()
            print(f"  ✅ {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed" if not only else f"{failed} failed")
    sys.exit(1 if failed else 0)
