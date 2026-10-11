"""T5DecodeRunner vs the bit-faithful emulator — no vLLM needed.

Covers what the vLLM backend feeds the runner: every KV-cache layout
(vLLM main (B,H,N,2D), FlashInfer NHD/HND, FlashAttention (2,B,...)),
uint8-typed fp8 caches, a strided query (a view of a fused QKV buffer),
a 2-D output buffer, per-tensor / per-head KV scales, page sizes 16..128,
q_len 1 and 2 (spec decode), bf16 and fp8 Q, h_k == 1, CUDA-graph padding
rows (seq_len 0), and CUDA-graph capture -> replay with a changed plan.

    python -m pytest -q tests/test_runner.py          # or
    python tests/test_runner.py                       # same, no pytest
"""

from __future__ import annotations

import math
import os
import sys

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

from mjolnir import t5_decode as t5  # noqa: E402
from mjolnir.runner import (  # noqa: E402
    T5DecodeRunner,
    detect_kv_layout,
    split_kv_cache,
)

DEV = "cuda"
FP8_NAN = 0x7F  # e4m3fn NaN byte, poisons pages no sequence owns


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def make_kv_cache(layout, num_pages, page_size, h_k, d, as_uint8):
    shapes = {
        "BHN2D": (num_pages, h_k, page_size, 2 * d),
        "BNH2D": (num_pages, page_size, h_k, 2 * d),
        "B2NHD": (num_pages, 2, page_size, h_k, d),
        "B2HND": (num_pages, 2, h_k, page_size, d),
        "2BNHD": (2, num_pages, page_size, h_k, d),
        "2BHND": (2, num_pages, h_k, page_size, d),
    }
    raw = torch.full(shapes[layout], FP8_NAN, dtype=torch.uint8, device=DEV)
    return raw if as_uint8 else raw.view(torch.float8_e4m3fn)


def fill_pages(view, pages, gen):
    """Random e4m3 values (|x| <~ 2) in the given pages of a logical
    (num_pages, PS, h_k, d) view (any strides)."""
    pages = list(pages)
    x = (torch.randn((len(pages), *view.shape[1:]), generator=gen) * 0.7).to(torch.float8_e4m3fn)
    idx = torch.tensor(pages, dtype=torch.long, device=DEV)
    view[idx] = x.to(DEV).view(view.dtype)


def run_case(
    *,
    layout="BHN2D",
    seq_lens=(300, 1, 2049, 128),
    q_len=1,
    h_q=16,
    h_k=2,
    d=256,
    page_size=16,
    q_mode="bf16",
    scales="per_tensor",  # per_tensor | per_head
    as_uint8=True,
    fused_qkv=True,
    out_2d=True,
    max_num_seqs=8,
    seed=0,
    check_zero_rows=(),
    pdl=False,
    l2_prefetch=0,
    kv_evict_first=False,
    plan_margin=0,
):
    gen = torch.Generator().manual_seed(seed)
    # vLLM's seq_len counts the step's own q_len tokens, so a live sequence
    # always has seq_len >= q_len (0 = CUDA-graph padding request)
    seq_lens = tuple(max(L, q_len) if L > 0 else 0 for L in seq_lens)
    b = len(seq_lens)
    tokens = b * q_len
    n_blocks = [max(1, math.ceil(L / page_size)) for L in seq_lens]
    max_blocks = max(n_blocks) + 1
    num_pages = sum(n_blocks) + 3  # spare pages stay NaN
    perm = torch.randperm(num_pages, generator=gen).tolist()
    bt = torch.zeros(max_num_seqs, max_blocks, dtype=torch.int32)
    used, i = [], 0
    for s, nb in enumerate(n_blocks):
        bt[s, :nb] = torch.tensor(perm[i : i + nb])
        used += perm[i : i + nb]
        i += nb
    block_table = bt.to(DEV)
    seq_lens_t = torch.zeros(max_num_seqs, dtype=torch.int32)
    seq_lens_t[:b] = torch.tensor(seq_lens)
    seq_lens_d = seq_lens_t.to(DEV)

    kv = make_kv_cache(layout, num_pages, page_size, h_k, d, as_uint8)
    assert detect_kv_layout(kv.shape, h_k, page_size, d) == layout or h_k == page_size
    k_cache, v_cache = split_kv_cache(kv, layout, d)
    fill_pages(k_cache, used, gen)
    fill_pages(v_cache, used, gen)

    # query: a strided view of a fused QKV buffer, like vLLM's qkv.split()
    if fused_qkv:
        qkv = (torch.randn(tokens, (h_q + 2 * h_k) * d, generator=gen) * 0.5).to(torch.bfloat16).to(DEV)
        query = qkv[:, : h_q * d].view(tokens, h_q, d)
        assert query.stride(0) == (h_q + 2 * h_k) * d
    else:
        query = (torch.randn(tokens, h_q, d, generator=gen) * 0.5).to(torch.bfloat16).to(DEV)
    out = torch.full(
        (tokens + 2, h_q * d) if out_2d else (tokens + 2, h_q, d), float("nan"), dtype=torch.bfloat16, device=DEV
    )

    if scales == "per_tensor":
        k_scale = torch.tensor([0.75], device=DEV)
        v_scale = torch.tensor([1.25], device=DEV)
    else:
        k_scale = (torch.rand(h_k, generator=gen) + 0.5).to(DEV)
        v_scale = (torch.rand(h_k, generator=gen) + 0.5).to(DEV)
    softmax_scale = 1.0 / math.sqrt(d)

    runner = T5DecodeRunner(
        num_heads=h_q,
        num_kv_heads=h_k,
        head_size=d,
        page_size=page_size,
        out_dtype=torch.bfloat16,
        max_num_seqs=max_num_seqs,
        max_q_len=max(2, q_len),
        q_mode=q_mode,
        pdl=pdl,
        l2_prefetch=l2_prefetch,
        kv_evict_first=kv_evict_first,
    )
    plan = runner.plan(seq_lens_t.tolist(), b, q_len, block_table, seq_lens_d, margin=plan_margin)
    runner.ensure_compiled(q_len, plan.with_reduction)
    runner.run(plan, query, out, k_cache, v_cache, k_scale, v_scale, softmax_scale)
    torch.cuda.synchronize()

    o = out[:tokens].view(tokens, h_q, d).float().cpu()
    assert torch.isnan(out[tokens:]).all(), "wrote past the batch"
    ref = emulate(
        runner, plan, query[:tokens], k_cache, v_cache, block_table, seq_lens_d, k_scale, v_scale, softmax_scale
    )
    for s in check_zero_rows:
        rows = o[s * q_len : (s + 1) * q_len]
        assert torch.isfinite(rows).all() and rows.abs().max() == 0, f"padding seq {s} not zero"
    live = [s for s in range(b) if seq_lens[s] > 0]
    sel = torch.cat([torch.arange(s * q_len, (s + 1) * q_len) for s in live])
    err = (o[sel] - ref[sel]).abs().max().item()
    tol = 2e-2 * max(1.0, ref[sel].abs().max().item())
    assert math.isfinite(err) and err <= tol, (
        f"{layout} q_len={q_len} {q_mode}: max|err| {err:.4g} > {tol:.4g} "
        f"(splits {plan.n_splits.tolist()}, chunk {plan.chunk_tiles})"
    )
    return err, plan


def emulate(runner, plan, query, k_cache, v_cache, block_table, seq_lens, k_scale, v_scale, scale):
    b, ql, h_k, h_r, d = plan.num_seqs, plan.q_len, runner.h_k, runner.h_r, runner.d
    n_q = plan.n_q
    if runner.fp8_q:  # the exact fp8 values + descales the kernel read
        q_packed = runner.q_pack[: b * h_k * n_q * d].view(b, h_k, n_q, d).float().cpu()
        qd = runner.q_descale[:b].float().cpu()
    else:
        q_packed = query.reshape(b, ql, h_k, h_r, d).permute(0, 2, 1, 3, 4).reshape(b, h_k, n_q, d).float().cpu()
        qd = torch.ones(b, h_k)

    def expand(t):
        t = t.float().cpu().reshape(-1)
        return t.expand(h_k).expand(b, h_k).clone() if t.numel() == 1 else t[None].expand(b, h_k).clone()

    kf = k_cache.view(torch.float8_e4m3fn).float().cpu() if k_cache.dtype == torch.uint8 else k_cache.float().cpu()
    vf = v_cache.view(torch.float8_e4m3fn).float().cpu() if v_cache.dtype == torch.uint8 else v_cache.float().cpu()
    # padding rows (seq_len 0) have no reference; emulate them as length
    # q_len and never compare them (run_case checks they are exactly 0)
    lens = seq_lens[:b].cpu().clamp_min(ql)
    o_emu, _, _ = t5.emulate_decode(
        q_packed,
        kf,
        vf,
        block_table[:b].cpu(),
        lens,
        qd,
        expand(k_scale),
        expand(v_scale),
        scale,
        1.0,
        ql,
        plan.chunk_tiles,
        runner.fp8_q,
    )
    # (b, h_k, ql*h_r, d) -> (b*ql, h_q, d)
    return o_emu.view(b, h_k, ql, h_r, d).permute(0, 2, 1, 3, 4).reshape(b * ql, h_k * h_r, d)


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


# ---- fp6 (e2m3) KV: the vLLM write path, the fallback expansion, T5 on it ----
def _fp6_cache(num_pages, h_k, page_size, d, head_major=True):
    """A vLLM-shaped (B, H, N, 2 * row) uint8 layer view; head_major=False
    stores it token-major underneath (LBNHC-like strides), as a layout check."""
    from mjolnir import kvfp6

    C = 2 * kvfp6.row_bytes(d)
    if head_major:
        return torch.zeros(num_pages, h_k, page_size, C, dtype=torch.uint8, device=DEV)
    return torch.zeros(num_pages, page_size, h_k, C, dtype=torch.uint8, device=DEV).transpose(1, 2)


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
    o_emu, _, _ = t5.emulate_decode(
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


def test_cuda_graph_replay_with_new_plan():
    _graph_replay(pdl=False, kv_evict_first=False)


def test_cuda_graph_replay_pdl_evict_first():
    # the vLLM fp8 defaults, captured with programmatic dependencies
    _graph_replay(pdl=True, kv_evict_first=True)


def _graph_replay(pdl, kv_evict_first):
    h_q, h_k, d, PS, b = 16, 2, 256, 16, 3
    gen = torch.Generator().manual_seed(7)
    max_blocks, num_pages = 600, 1800
    kv = make_kv_cache("BHN2D", num_pages, PS, h_k, d, True)
    k_cache, v_cache = split_kv_cache(kv, "BHN2D", d)
    fill_pages(k_cache, range(num_pages), gen)
    fill_pages(v_cache, range(num_pages), gen)
    block_table = torch.randperm(num_pages, generator=gen)[: b * max_blocks].view(b, max_blocks).int().to(DEV)
    seq_lens = torch.tensor([4000, 50, 1000], dtype=torch.int32, device=DEV)
    query = (torch.randn(b, h_q, d, generator=gen) * 0.5).to(torch.bfloat16).to(DEV)
    out = torch.empty(b, h_q * d, dtype=torch.bfloat16, device=DEV)
    ks = torch.tensor([1.0], device=DEV)
    runner = T5DecodeRunner(
        num_heads=h_q,
        num_kv_heads=h_k,
        head_size=d,
        page_size=PS,
        out_dtype=torch.bfloat16,
        max_num_seqs=b,
        q_mode="fp8",
        pdl=pdl,
        kv_evict_first=kv_evict_first,
    )
    plan = runner.plan(seq_lens.tolist(), b, 1, block_table, seq_lens)
    runner.ensure_compiled(1, plan.with_reduction)
    runner.run(plan, query, out, k_cache, v_cache, ks, ks, 0.0625)  # warm-up (eager)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        runner.run(plan, query, out, k_cache, v_cache, ks, ks, 0.0625)

    # next "step": new lengths and query, re-planned outside the graph
    for lens in ([9000, 17, 300], [128, 128, 128], [1, 8191, 2]):
        seq_lens.copy_(torch.tensor(lens, dtype=torch.int32))
        query.copy_((torch.randn(b, h_q, d, generator=gen) * 0.5).to(torch.bfloat16))
        new = runner.plan(lens, b, 1, block_table, seq_lens)
        assert new.with_reduction == plan.with_reduction and new.num_work == plan.num_work
        g.replay()
        torch.cuda.synchronize()
        ref = emulate(runner, new, query, k_cache, v_cache, block_table, seq_lens, ks, ks, 0.0625)
        err = (out.view(b, h_q, d).float().cpu() - ref).abs().max().item()
        assert err <= 2e-2 * max(1.0, ref.abs().max().item()), f"graph replay {lens}: {err}"


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
