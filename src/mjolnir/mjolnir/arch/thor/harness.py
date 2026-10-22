"""The standalone Mjolnir decode harness: correctness reference, benchmark driver, CLI.

The Thor (sm_110) harness: the standalone benchmark driver + the
bit-faithful emulator. Run as ``python -m mjolnir.arch.thor.harness``
(formerly ``python -m mjolnir.t5_decode``).
"""

import argparse
import math
import os

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from mjolnir.quant import (
    NVFP4_GROUP,
    minifloat_quantize,
    minifloat_table,
    nvfp4_quantize,
    pack_q,
    pack_subbyte,
    quantize_q_fp8,
    unpack_o,
)
from mjolnir.arch.thor.plan import STRIDE_FIELDS, kernel_strides, plan_split_kv, tile_s_rows
from mjolnir.arch.thor.const import FP8_P_SCALE, log2_e
from mjolnir.arch.thor.decode import (
    HARNESS_KV_FORMAT_FP8Q,
    HARNESS_KV_LAYOUT,
    KV_FORMAT_BITS,
    KV_FORMATS,
    MixedInputFusedMultiHeadAttentionDecode,
)


def _to_cute(t, dtype=None):
    """torch -> cute tensor; fp8 goes through a uint8 view (dlpack-safe)."""
    if t.dtype == torch.float8_e4m3fn:
        ct = from_dlpack(t.view(torch.uint8), assumed_align=16)
        ct.element_type = cutlass.Float8E4M3FN
        return ct
    ct = from_dlpack(t, assumed_align=16)
    if dtype is not None:
        ct.element_type = dtype
    return ct


def emulate_decode(
    q_packed,  # (b, h_k, n_q, d) float — the exact values the kernel reads
    k_pool,  # (num_pages, PS, h_k, d) float
    v_pool,
    block_table,  # (b, max_blocks) int
    seq_lens,  # (b,) int
    q_descale,
    k_descale,
    v_descale,  # (b, h_k) float
    scale_qs,
    scale_o,
    q_len,
    chunk_tiles,
    fp8,
    causal=True,
):
    """Bit-faithful model of the kernel's arithmetic (contiguous chunk split, running
    max, -inf guard, P rounding to e4m3(p * 448) or bf16, descale folding).
    Also returns a plain fp32 attention for the fp8-vs-exact gap."""
    b, h_k, n_q, d = q_packed.shape
    PS = k_pool.shape[1]
    h_r = n_q // q_len
    p_scale = FP8_P_SCALE if fp8 else 1.0
    o_emu = torch.zeros(b, h_k, n_q, d)
    lse_emu = torch.zeros(b, h_k, n_q)
    o_exact = torch.zeros(b, h_k, n_q, d)
    for bi in range(b):
        L = int(seq_lens[bi])
        n_blk = math.ceil(L / PS)
        pages = block_table[bi, :n_blk].long()
        Kb = k_pool[pages].reshape(-1, h_k, d)[:L].permute(1, 0, 2)  # (h_k, L, d)
        Vb = v_pool[pages].reshape(-1, h_k, d)[:L].permute(1, 0, 2)
        S = torch.einsum("gnd,gld->gnl", q_packed[bi], Kb)  # raw scores
        qi = torch.arange(n_q) // h_r
        lim = L - q_len + qi if causal else torch.full((n_q,), L - 1)  # (n_q,) last key each row sees
        masked = torch.arange(L)[None, :] > lim[:, None]  # (n_q, L)
        S = S.masked_fill(masked[None], -math.inf)
        qk = q_descale[bi] * k_descale[bi]  # (h_k,)
        c = (scale_qs * log2_e * qk)[:, None]  # (h_k, 1)
        cta_scale = (scale_qs * qk)[:, None]
        vd = v_descale[bi][:, None, None]

        tiles = math.ceil(L / tile_s_rows)
        o_parts, m_parts, l_parts = [], [], []
        n_chunks = math.ceil(tiles / chunk_tiles)
        for j in range(n_chunks):  # contiguous chunks, as in the plan
            m = torch.full((h_k, n_q), -math.inf)
            l = torch.zeros(h_k, n_q)
            o = torch.zeros(h_k, n_q, d)
            for t in range(j * chunk_tiles, min((j + 1) * chunk_tiles, tiles)):
                sl = slice(t * tile_s_rows, min((t + 1) * tile_s_rows, L))
                St = S[:, :, sl]
                m_new = torch.maximum(m, St.amax(dim=-1))
                m_use = torch.where(m_new == -math.inf, torch.zeros_like(m_new), m_new)
                p = torch.exp2((St - m_use[..., None]) * c[..., None])
                corr = torch.exp2((m - m_use) * c)
                l = l * corr + p.sum(dim=-1)
                pq = (p * p_scale).to(torch.float8_e4m3fn).float() if fp8 else p.to(torch.bfloat16).float()
                o = o * corr[..., None] + torch.einsum("gnl,gld->gnd", pq, Vb[:, sl])
                m = m_new
            o_parts.append(o * vd / p_scale)
            m_parts.append(m * cta_scale)
            l_parts.append(l)
        m_all = torch.stack(m_parts).amax(dim=0)
        w = [torch.exp(mp - m_all) for mp in m_parts]
        o_sum = sum(wi[..., None] * op for wi, op in zip(w, o_parts, strict=True))
        l_sum = sum(wi * lp for wi, lp in zip(w, l_parts, strict=True))
        o_emu[bi] = scale_o * o_sum / l_sum[..., None]
        lse_emu[bi] = m_all + torch.log(l_sum)

        # exact fp32 attention (no P rounding) for reference
        P = torch.softmax(S * cta_scale[..., None], dim=-1)
        o_exact[bi] = scale_o * torch.einsum("gnl,gld->gnd", P, Vb) * vd
    return o_emu, lse_emu, o_exact


def run(
    batches: int = 1,
    seqlen: int = 1024,
    varlen: bool = False,
    q_len: int = 1,
    heads_q: int = 24,
    heads_k: int = 4,
    headdim: int = 256,
    page_size: int = 128,
    kv_splits: int = 0,
    q_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
    kv_dtype: type[cutlass.Numeric] = cutlass.Float8E4M3FN,
    tolerance: float = 0.05,
    scale_s: float = 0.0,
    scale_o: float = 1.0,
    warmup_iterations: int = 0,
    iterations: int = 0,
    skip_ref_check: bool = False,
    seed: int = 1111,
    ctas_per_sm: int = 2,
    no_l2_flush: bool = False,
    kv_stages: int = 0,
    contiguous_pages: bool = False,
    max_splits: int = 16,
    pad_work_to: int = 0,
    kv_layout: str = "nhd",
    strided_qo: bool = False,
    descale: str = "per_seq_head",
    pdl: bool = False,
    l2_prefetch: int = 0,
    kv_evict_first: int = -1,  # -1 auto: on for fp8 Q (vLLM default), off for bf16 Q
    kv_format: str = "fp8",
    kv_from_float: bool = False,
    non_causal: bool = False,
    fused_merge: bool = False,
    **kwargs,
):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")
    cap = torch.cuda.get_device_capability()
    if cap not in ((10, 0), (11, 0)):
        raise RuntimeError(
            f"This harness drives the Thor (sm_110) T5 tcgen05 kernel; the device is "
            f"sm_{cap[0]}{cap[1]}. The Orin (sm_87) family has its own driver under "
            "mjolnir/arch/orin (see its README)."
        )
    torch.manual_seed(seed)
    dev = "cuda"

    fp8 = q_dtype.width == 8
    num_seqs = batches
    if kv_evict_first < 0:
        # matches the vLLM backend default: EVICT_FIRST wins on the fp8 path,
        # but costs the bf16 path 4-18% at 32K-256K
        kv_evict_first = 1 if fp8 else 0
    h_r = heads_q // heads_k
    n_q = q_len * h_r
    PS = page_size

    # Kernel config. Default 2 CTAs/SM (40-CTA one-wave plan for one
    # sequence) with a one-tile KV ring: measured best on Thor. At 2 CTAs/SM
    # the bf16 path uses a single cvt warpgroup so two CTAs fit the
    # 64K-register file.
    assert kv_format in KV_FORMATS
    if kv_format != "fp8":
        assert fp8, f"kv_format {kv_format} runs on the fp8-Q path (--q_dtype Float8E4M3FN)"
        kv_dtype = {
            "e3m2": cutlass.Float6E3M2FN,
            "e2m3": cutlass.Float6E2M3FN,
            "e2m1": cutlass.Float4E2M1FN,
            "nvfp4": cutlass.Float4E2M1FN,
        }[kv_format]
    kv_bits = KV_FORMAT_BITS[kv_format]
    from_float = kv_format != "fp8" or kv_from_float
    if fp8:
        convert_warpgroups = 1 if kv_format == "nvfp4" else 0
    elif headdim > 128 and ctas_per_sm == 1:
        convert_warpgroups = 2
    else:
        convert_warpgroups = 1
    ght_base = 16 if fp8 else 8
    # Rows per CTA tile: next power of two >= n_q (min 8 bf16 / 16 fp8, max 32;
    # wider n_q is tiled across CTAs). TMEM ld/st widths must be powers of two.
    grouped_head_tile = max(ght_base, min(32, 1 << max(0, (n_q - 1).bit_length())))
    n_tiles = math.ceil(n_q / grouped_head_tile)

    # Sequence lengths (include the q_len new tokens). A decode step always
    # has at least the new token(s) in the cache, so L < q_len is clamped.
    if seqlen < q_len:
        print(f"\tseqlen {seqlen} < q_len {q_len}: using seqlen = {q_len}")
        seqlen = q_len
    if varlen:
        seq_lens = torch.randint(q_len, seqlen + 1, (num_seqs,), dtype=torch.int32)
        seq_lens[0] = seqlen  # keep the max for the split heuristic
    else:
        seq_lens = torch.full((num_seqs,), seqlen, dtype=torch.int32)
    max_seqlen = int(seq_lens.max())

    hw = cutlass.utils.HardwareInfo()
    sm_count = hw.get_device_multiprocessor_count()
    sm_count = 20 if sm_count <= 0 else sm_count  # fallback: Thor's 20 SMs (the query rarely fails)
    # Occupancy-pinned split-KV plan (contiguous chunks, one wave).
    plan = plan_split_kv(
        seq_lens.tolist(),
        heads_k,
        n_tiles,
        sm_count,
        ctas_per_sm=ctas_per_sm,
        max_splits=max_splits,
        force_splits=kv_splits,
        pad_to=pad_work_to,
        with_counter=fused_merge,
    )

    print(f"Paged FMHA decode (sm_{cap[0]}{cap[1]}) with:")
    print(f"\tnum_seqs: {num_seqs}, max seqlen: {max_seqlen}, varlen: {varlen}")
    print(f"\tseq_lens: {seq_lens.tolist()[:8]}{' ...' if num_seqs > 8 else ''}")
    print(f"\tq_len: {q_len}, heads_q: {heads_q}, heads_k: {heads_k}, headdim: {headdim}")
    grid_ctas = plan.num_work * heads_k * n_tiles
    print(
        f"\tpage_size: {PS}, sm_count: {sm_count}, ctas_per_sm: {ctas_per_sm}, "
        f"grid: {grid_ctas} CTAs, pages: {'contiguous' if contiguous_pages else 'shuffled'}"
    )
    print(f"\tplan: {plan}")
    print(f"\tq_dtype: {q_dtype}, kv_dtype: {kv_dtype}")

    fmha = MixedInputFusedMultiHeadAttentionDecode(
        headdim=headdim,
        grouped_head_tile=grouped_head_tile,
        convert_warpgroups=convert_warpgroups,
        page_size=PS,
        q_len=q_len,
        max_splits=max_splits,
        ctas_per_sm=ctas_per_sm,
        max_kv_stages=kv_stages,
        with_reduction=max(plan.n_splits) > 1 and not fused_merge,
        pdl=pdl,
        l2_prefetch=l2_prefetch,
        kv_evict_first=bool(kv_evict_first),
        kv_format=kv_format,
        causal=not non_causal,
        fused_merge=fused_merge,
    )
    fmha.can_implement(num_seqs, heads_q, heads_k, headdim, q_dtype, kv_dtype)

    if scale_s == 0.0:
        scale_s = 1.0 / math.sqrt(headdim)

    #
    # Paged KV pool with a SHUFFLED page assignment; unreferenced pages are
    # poisoned with e4m3 NaN so any stray read shows up in the output.
    #
    pages_per_seq = [math.ceil(int(L) / PS) for L in seq_lens]
    max_blocks = max(pages_per_seq)
    used_pages = sum(pages_per_seq)
    num_pages = used_pages + max(4, used_pages // 4)
    # Shuffled pages exercise the paging logic; contiguous pages (identity-
    # ordered block table) match a freshly allocated cache and the FI driver.
    perm = torch.arange(num_pages) if contiguous_pages else torch.randperm(num_pages)
    block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32)
    cursor = 0
    for bi, n_blk in enumerate(pages_per_seq):
        block_table[bi, :n_blk] = perm[cursor : cursor + n_blk].to(torch.int32)
        cursor += n_blk
    unused = perm[cursor:]

    # Descales: per (seq, kv head), per kv head, or one per-tensor scale
    # (vLLM's layer._k_scale) — the kernel takes them via strides.
    def make_descale():
        if descale == "per_tensor":
            return torch.rand(1) * 1.5 + 0.5
        if descale == "per_head":
            return torch.rand(heads_k) * 1.5 + 0.5
        return torch.rand(num_seqs, heads_k) * 1.5 + 0.5

    def expand_descale(t):  # logical (b, h_k) for the reference
        t = t.float()
        if t.numel() == 1:
            return t.reshape(1, 1).expand(num_seqs, heads_k)
        if t.dim() == 1:
            return t.reshape(1, heads_k).expand(num_seqs, heads_k)
        return t

    row_bytes = headdim * kv_bits // 8

    def kv_pool():
        """-> (bytes (pages, PS, h_k, row_bytes) uint8, kernel values (float,
        what the MMA multiplies), descale, nvfp4 scale bytes or None,
        unquantized float pool or None). Unreferenced pages are poisoned:
        e4m3 NaN for fp8, the largest code (and scale) otherwise."""
        shape = (num_pages, PS, heads_k, headdim)
        if not from_float:  # legacy fp8 test data: raw e4m3 values, random descales
            x8 = (torch.randn(shape) * 2.0).to(torch.float8_e4m3fn)
            xb = x8.view(torch.uint8).clone()
            xb[unused] = 0x7F  # e4m3fn NaN
            return xb, xb.view(torch.float8_e4m3fn).float(), make_descale(), None, None
        x = torch.randn(shape)  # the "real" KV the cache quantizes
        if kv_format == "nvfp4":
            codes, sc, _gs, vals, desc = nvfp4_quantize(x)
            codes[unused] = 7  # +6.0
            sc = sc.view(torch.uint8).clone()
            sc[unused] = 0x7E  # e4m3 448
            vals = vals.clone()
            vals[unused] = float("nan")  # never read by the reference
            return pack_subbyte(codes, 4), vals, torch.tensor([desc]), sc, x
        amax = x.abs().amax(dim=(0, 1, 3)).clamp(min=1e-6)  # per kv head
        if kv_format == "fp8":
            kd = amax / 448.0
            x8 = (x / kd.view(1, 1, -1, 1)).to(torch.float8_e4m3fn)
            xb = x8.view(torch.uint8).clone()
            xb[unused] = 0x7F
            return xb, xb.view(torch.float8_e4m3fn).float(), kd, None, x
        fmax = float(minifloat_table(kv_format).max())
        kd = amax / fmax
        codes, vals = minifloat_quantize(x / kd.view(1, 1, -1, 1), kv_format)
        codes[unused] = (1 << (kv_bits - 1)) - 1  # largest positive code
        return pack_subbyte(codes, kv_bits), vals, kd, None, x

    k_bytes, k_pool, k_descale, k_sc, k_true = kv_pool()
    v_bytes, v_pool, v_descale, v_sc, v_true = kv_pool()
    if from_float:
        print(
            f"\tkv: format {kv_format} ({kv_bits} bit) quantized from float data; "
            f"descale {'per-tensor (nvfp4 global)' if kv_format == 'nvfp4' else 'per kv head'}"
            f" (--descale ignored)"
        )

    q_vllm = torch.randn(num_seqs * q_len, heads_q, headdim).to(torch.bfloat16)
    if fp8:
        q_in, q_descale = quantize_q_fp8(q_vllm, num_seqs, q_len, heads_k)
    else:
        q_in, q_descale = q_vllm, torch.ones(1)  # per-tensor 1.0
    q_packed = pack_q(q_in, num_seqs, q_len, heads_k).contiguous()

    # Device tensors. Q/O optionally live inside wider rows (like vLLM's
    # fused-QKV `query` view); the KV cache optionally uses another vLLM
    # physical layout — the kernel sees only strided logical views.
    row = heads_k * n_q * headdim
    pad = (2 * heads_k * headdim) if strided_qo else 0

    def strided_rows(src_packed, dtype):
        buf = torch.zeros(num_seqs, row + pad, dtype=dtype, device=dev)
        view = buf[:, :row].view(num_seqs, heads_k, n_q, headdim)
        if src_packed is not None:
            if src_packed.dtype == torch.float8_e4m3fn:
                view.view(torch.uint8).copy_(src_packed.view(torch.uint8).to(dev))
            else:
                view.copy_(src_packed.to(dev))
        return view

    q_d = strided_rows(q_packed, q_packed.dtype)
    # KV bytes in the chosen physical layout; the kernel sees logical
    # (pages, PS, h_k, row) views. fp8 views are retyped e4m3, sub-byte
    # formats stay uint8 byte views (strides converted to elements).
    rb = row_bytes
    if kv_layout == "nhd":  # separate (pages, PS, h_k, row) K and V tensors
        k_d, v_d = k_bytes.to(dev), v_bytes.to(dev)
    elif kv_layout == "hnd":  # head-major: (pages, 2, h_k, PS, row)
        buf = torch.empty(num_pages, 2, heads_k, PS, rb, dtype=torch.uint8, device=dev)
        buf[:, 0] = k_bytes.permute(0, 2, 1, 3).to(dev)
        buf[:, 1] = v_bytes.permute(0, 2, 1, 3).to(dev)
        k_d, v_d = buf[:, 0].transpose(1, 2), buf[:, 1].transpose(1, 2)
    elif kv_layout == "interleaved":  # vLLM main FA: (pages, h_k, PS, 2*row)
        buf = torch.empty(num_pages, heads_k, PS, 2 * rb, dtype=torch.uint8, device=dev)
        buf[..., :rb] = k_bytes.permute(0, 2, 1, 3).to(dev)
        buf[..., rb:] = v_bytes.permute(0, 2, 1, 3).to(dev)
        kv = buf.transpose(1, 2)  # (pages, PS, h_k, 2*row)
        k_d, v_d = kv[..., :rb], kv[..., rb:]
    elif kv_layout == "flashinfer":  # FlashInfer NHD: (pages, 2, PS, h_k, row)
        buf = torch.empty(num_pages, 2, PS, heads_k, rb, dtype=torch.uint8, device=dev)
        buf[:, 0] = k_bytes.to(dev)
        buf[:, 1] = v_bytes.to(dev)
        k_d, v_d = buf[:, 0], buf[:, 1]
    else:
        raise ValueError(f"unknown kv_layout {kv_layout}")
    if kv_bits == 8:
        k_d, v_d = k_d.view(torch.float8_e4m3fn), v_d.view(torch.float8_e4m3fn)
    if kv_format == "nvfp4":
        ks_d = k_sc.to(dev).view(torch.float8_e4m3fn)
        vs_d = v_sc.to(dev).view(torch.float8_e4m3fn)
    else:  # unused by the kernel; any valid e4m3 pointer
        ks_d = vs_d = torch.zeros(1, 1, 1, NVFP4_GROUP, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    bt_d = block_table.to(dev)
    sl_d = seq_lens.to(dev)
    work_d = plan.work_tensor(dev)
    ns_d = plan.n_splits_tensor(dev)
    qd_d = q_descale.float().contiguous().to(dev)
    kd_d = k_descale.float().contiguous().to(dev)
    vd_d = v_descale.float().contiguous().to(dev)
    o_d = strided_rows(None, torch.bfloat16)
    strides = kernel_strides(
        q_d,
        o_d,
        k_d,
        v_d,
        qd_d,
        kd_d,
        vd_d,
        kv_bits=kv_bits,
        k_scales=ks_d if kv_format == "nvfp4" else None,
        v_scales=vs_d if kv_format == "nvfp4" else None,
    )
    print(
        f"\tlayout: kv={kv_layout} strided_qo={strided_qo} descale={descale}; "
        f"strides={dict(zip(STRIDE_FIELDS, strides, strict=True))}"
    )
    lse_d = torch.empty(num_seqs, heads_k, n_q, dtype=torch.float32, device=dev)
    # Workspace: deliberately NOT initialized (the kernel must not need it)
    op_d = torch.full((max_splits, num_seqs, heads_k, n_q, headdim), float("nan"), device=dev)
    mp_d = torch.full((max_splits, num_seqs, heads_k, n_q), float("nan"), device=dev)
    lp_d = torch.full((max_splits, num_seqs, heads_k, n_q), float("nan"), device=dev)

    def kv_ptr(t):
        if kv_bits == 8:
            return _to_cute(t).iterator
        from cutlass.cute.runtime import make_ptr  # sub-byte element pointer

        return make_ptr(kv_dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    kv_ptrs = [kv_ptr(k_d), kv_ptr(v_d), _to_cute(ks_d).iterator, _to_cute(vs_d).iterator]
    cts = [
        _to_cute(q_d),
        _to_cute(bt_d),
        _to_cute(sl_d),
        _to_cute(work_d),
        _to_cute(ns_d),
        _to_cute(qd_d),
        _to_cute(kd_d),
        _to_cute(vd_d),
        _to_cute(o_d),
        _to_cute(lse_d),
        _to_cute(op_d),
        _to_cute(mp_d),
        _to_cute(lp_d),
    ]
    stream = cutlass_torch.default_stream()
    args = (
        (num_seqs, heads_k, n_q, headdim),
        max_blocks,
        num_pages,
        max(1, plan.num_work),
        strides,
        cts[0].iterator,
        *kv_ptrs,
        *[t.iterator for t in cts[1:]],
        scale_s,
        scale_o,
        stream,
    )
    compiled = cute.compile(fmha, *args, options="--opt-level 2")
    print("Finished Compiling")

    if not skip_ref_check:
        compiled(*args)
        torch.cuda.synchronize()
        print("Verifying results...")
        o_emu, lse_emu, o_exact = emulate_decode(
            q_packed.float(),
            k_pool.float(),
            v_pool.float(),
            block_table,
            seq_lens,
            expand_descale(q_descale.float()),
            expand_descale(k_descale.float()),
            expand_descale(v_descale.float()),
            scale_s,
            scale_o,
            q_len,
            plan.chunk_tiles,
            fp8,
            causal=not non_causal,
        )
        o_kern = o_d.float().cpu()
        lse_kern = lse_d.cpu()
        if from_float:
            # what the cache format costs: kernel vs attention over the
            # UNQUANTIZED float KV (same q, P rounding aside)
            ones = torch.ones(num_seqs, heads_k)
            _, _, o_true = emulate_decode(
                q_packed.float(),
                k_true,
                v_true,
                block_table,
                seq_lens,
                expand_descale(q_descale.float()),
                ones,
                ones,
                scale_s,
                scale_o,
                q_len,
                plan.chunk_tiles,
                fp8,
                causal=not non_causal,
            )
            d_true = (o_kern - o_true).abs()
            rel = (d_true.norm() / o_true.norm().clamp(min=1e-12)).item()
            print(
                f"kv quantization error ({kv_format}): |kern-fp32 unquantized| "
                f"max={d_true.max().item():.4g} mean={d_true.mean().item():.4g} "
                f"rel_l2={rel:.4g}"
            )
        d_emu = (o_kern - o_emu).abs()
        d_exact = (o_kern - o_exact).abs()
        d_lse = (lse_kern - lse_emu).abs()
        print(
            f"check: |kern-emu| max={d_emu.max().item():.4g} mean={d_emu.mean().item():.4g}; "
            f"|lse-emu| max={d_lse.max().item():.4g}; "
            f"|kern-exact fp32| max={d_exact.max().item():.4g} "
            f"(P rounding to {'e4m3' if fp8 else 'bf16'})"
        )
        if torch.isnan(o_kern).any():
            raise AssertionError("NaN in output (stray read of a poisoned page?)")
        torch.testing.assert_close(o_kern, o_emu, atol=tolerance, rtol=2e-2)
        torch.testing.assert_close(lse_kern, lse_emu, atol=1e-2, rtol=1e-3)
        # vLLM-layout round trip sanity
        _ = unpack_o(o_d, num_seqs, q_len, heads_k)

    exec_us = 0.0
    if iterations > 0:
        # Cold L2 (32 MiB on Thor): flush between timed launches, unless an
        # external profiler controls the cache (ncu --cache-control all) —
        # then the flush kernel is just noise in the profile.
        flush = None
        if not no_l2_flush:
            flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
        for _ in range(warmup_iterations):
            compiled(*args)
        times = []
        for _ in range(iterations):
            if flush is not None:
                flush.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            compiled(*args)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        times.sort()
        exec_us = times[len(times) // 2]
        kv_bytes = int(seq_lens.sum()) * heads_k * headdim * 2  # K + V, 1 B each
        io_bytes = q_d.numel() * q_d.element_size() + o_d.numel() * 2
        gbps = (kv_bytes + io_bytes) / (exec_us * 1e-6) / 1e9
        cache = "hot/unflushed L2" if flush is None else "cold L2"
        if os.environ.get("CUDA_INJECTION64_PATH"):
            # ncu / nsys inject a library and serialize + replay every launch:
            # host-side event timing then measures the profiler, not the kernel.
            print(
                f"median {exec_us:.1f} us host-timed UNDER A PROFILER (includes "
                f"replay/flush overhead) — use the profiler's gpu__time_duration"
            )
        else:
            print(f"median {exec_us:.2f} us, {gbps:.3g} GB/s ({cache}, {kv_bytes + io_bytes} B of KV+Q+O)")

    return exec_us


def main():
    from mjolnir.arch.thor import decode as _t5

    _t5.VERBOSE = True  # the harness prints the kernel's compile-time notes
    parser = argparse.ArgumentParser(description="Paged fp8 decode attention (the Thor sm_110 T5 kernel).")
    parser.add_argument("--batches", "--batch", "--b", "--num_seqs", type=int, default=1)
    parser.add_argument(
        "--seqlen", "--seqlen_k", "--s", type=int, default=1024, help="(max) kv sequence length incl. new tokens"
    )
    parser.add_argument("--varlen", action="store_true", help="random per-sequence lengths in [q_len, seqlen]")
    parser.add_argument("--q_len", type=int, default=1, help="query tokens per sequence (spec decode)")
    parser.add_argument("--heads_q", "--h_q", type=int, default=24)
    parser.add_argument("--heads_k", "--h_k", type=int, default=4)
    parser.add_argument("--headdim", "--d", type=int, default=256)
    parser.add_argument("--page_size", type=int, default=128)
    parser.add_argument(
        "--ctas_per_sm",
        type=int,
        default=2,
        choices=(1, 2),
        help="co-resident CTAs per SM (2 = 40 CTAs/wave, measured best on Thor with the default ring depth)",
    )
    parser.add_argument(
        "--kv_stages",
        type=int,
        default=0,
        help="KV ring depth (0 = default: 1 KV tile at 2 CTAs/SM, 2 tiles at 1 CTA/SM)",
    )
    parser.add_argument(
        "--contiguous_pages", action="store_true", help="identity-ordered block table (default: shuffled pages)"
    )
    parser.add_argument(
        "--kv_splits",
        "--splits",
        type=int,
        default=0,
        help="force N contiguous chunks per sequence (0 = occupancy-pinned plan)",
    )
    parser.add_argument("--max_splits", type=int, default=16, help="partial-slot capacity per sequence (compile-time)")
    parser.add_argument(
        "--pad_work_to", type=int, default=0, help="pad the work list to N items (fixed grid, CUDA graphs)"
    )
    parser.add_argument(
        "--kv_layout",
        default=HARNESS_KV_LAYOUT,
        choices=("nhd", "hnd", "interleaved", "flashinfer"),
        help="physical KV cache layout (vLLM NHD, head-major HND "
        "(pages, 2, h_k, PS, d), vLLM-main interleaved K|V, "
        f"FlashInfer NHD); default {HARNESS_KV_LAYOUT}",
    )
    parser.add_argument(
        "--kv_format",
        default="auto",
        choices=("auto", *KV_FORMATS),
        help="KV cache format (fp8-Q path for non-fp8): e4m3; fp6 "
        "e3m2/e2m3 or fp4 e2m1 read directly by the MMA (per "
        "kv-head descale); nvfp4 (e2m1 + e4m3 per-16 scales, "
        "converted to e4m3 in-kernel); default auto = "
        f"{HARNESS_KV_FORMAT_FP8Q} for fp8 Q, fp8 for bf16 Q",
    )
    parser.add_argument(
        "--kv_from_float",
        action="store_true",
        help="fp8: quantize KV from float data with per-head "
        "descales and report the quantization error (always "
        "on for the other formats) — the accuracy A/B baseline",
    )
    parser.add_argument(
        "--non_causal",
        action="store_true",
        help="every query token sees all seq_len keys (q_len > 1): the block attention of "
        "parallel drafters (DSpark / DFlash)",
    )
    parser.add_argument(
        "--strided_qo", action="store_true", help="Q/O rows inside wider rows (like vLLM's fused-QKV view)"
    )
    parser.add_argument(
        "--pdl",
        action="store_true",
        help="programmatic dependent launch (decode waits on the previous "
        "kernel, reduction launches early); time with --iterations/nsys, "
        "ncu serializes kernels and hides the effect",
    )
    parser.add_argument(
        "--l2_prefetch", type=int, default=0, help="TMA-prefetch K/V this many 128-row tiles ahead into L2 (0 = off)"
    )
    parser.add_argument(
        "--kv_evict_first",
        type=int,
        nargs="?",
        const=1,
        default=-1,
        choices=(-1, 0, 1),
        help="TMA K/V loads with an L2 EVICT_FIRST cache policy; "
        "default auto = on for fp8 Q, off for bf16 Q (the vLLM "
        "defaults); 0/1 force it",
    )
    parser.add_argument(
        "--descale",
        default="per_seq_head",
        choices=("per_seq_head", "per_head", "per_tensor"),
        help="K/V descale granularity (vLLM default: per_tensor)",
    )
    parser.add_argument("--q_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--kv_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--tolerance", type=float, default=0.05, help="atol vs the bit-faithful emulation")
    parser.add_argument("--scale_s", "--scale", type=float, default=0.0)
    parser.add_argument("--scale_o", type=float, default=1.0)
    parser.add_argument("--warmup_iterations", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument(
        "--no_l2_flush",
        action="store_true",
        help="don't flush L2 between timed launches (use under ncu --cache-control all)",
    )
    # Accepted for old command lines; paging is always on now.
    parser.add_argument("--paged", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--fused_merge",
        action="store_true",
        help="E1: the last CTA per (seq, q tile, kv head) merges the split partials in-kernel",
    )

    kwargs = vars(parser.parse_args())
    if kwargs["kv_format"] == "auto":  # sub-8-bit formats need fp8 Q
        kwargs["kv_format"] = HARNESS_KV_FORMAT_FP8Q if kwargs["q_dtype"] is cutlass.Float8E4M3FN else "fp8"
    run(**kwargs)

    print("PASS")


if __name__ == "__main__":
    main()
