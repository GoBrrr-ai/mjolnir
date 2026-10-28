"""Packed fp6 (e2m3) KV cache for the vLLM path: write, and expand for the fallback.

Format (the Mjolnir kernel's, see ``mjolnir.quant.pack_subbyte``): one e2m3 code
per element (sign | 2-bit exponent | 3-bit mantissa, bias 1, no inf/nan,
values 0 .. ±7.5), packed little-endian, 4 codes in 3 bytes. A K (or V) row
of d elements is ``d * 3 / 4`` bytes; vLLM's page cell is ``K row | V row``
(``(B, H, N, 2 * d * 3 / 4)`` uint8, any physical stride order).

Values are ``code_value * scale`` with a static scale per (layer, KV head):
``scale = amax / 7.5`` from a calibration run (``mjolnir.calibrate``).

* :func:`write_kv` — the cache write (vLLM's ``do_kv_cache_update``): scale,
  round to nearest (saturating), pack, scatter by slot.
* :func:`expand_pages_to_fp8` — prefill / mixed steps run on FlashAttention 4,
  which reads e4m3 only: the pages a step touches are expanded to e4m3 bytes
  in a scratch cache. Every e2m3 value is exactly representable in e4m3, so
  with per-head scales the expansion is lossless and the same scales are the
  descales; with per-channel scales s_c each value becomes e4m3(code * s_c / S)
  for a per-head descale S = max_c(s_c) * 7.5 / 448 (one e4m3 rounding).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

E2M3_MAX = 7.5
BITS = 6


def row_bytes(head_size: int) -> int:
    assert head_size % 4 == 0
    return head_size * BITS // 8


_LUT: dict = {}
_LUTF: dict = {}


def e2m3_values(device) -> torch.Tensor:
    """(64,) fp32: the value of every e2m3 code."""
    key = str(device)
    if key not in _LUTF:
        from .quant import minifloat_table

        _LUTF[key] = minifloat_table("e2m3").to(device)
    return _LUTF[key]


def e2m3_to_e4m3_lut(device) -> torch.Tensor:
    """(64,) uint8: e4m3fn byte of every e2m3 code (exact)."""
    key = str(device)
    if key not in _LUT:
        from .quant import minifloat_table

        vals = minifloat_table("e2m3")  # 64 float values, code order
        e4 = vals.to(torch.float8_e4m3fn)
        assert torch.equal(e4.float(), vals), "e2m3 values must be exact in e4m3"
        _LUT[key] = e4.view(torch.uint8).to(device)
    return _LUT[key]


if triton is not None:

    @triton.jit
    def _e2m3_codes(x):
        """fp32 (already divided by the scale) -> e2m3 codes (int32 0..63):
        round to nearest, saturating at 7.5, ties to the smaller magnitude
        (as ``mjolnir.quant.minifloat_quantize``)."""
        ax = tl.minimum(tl.abs(x), 7.5)
        # quantum: 0.125 below 2 (subnormals and [1, 2)), 0.25 in [2, 4), 0.5 in [4, 7.5]
        q = tl.where(ax < 2.0, 0.125, tl.where(ax < 4.0, 0.25, 0.5))
        v = tl.minimum(tl.ceil(ax / q - 0.5) * q, 7.5)
        # value -> code: [0, 2): 8 v; [2, 4): 16 + 4 (v - 2); [4, 7.5]: 24 + 2 (v - 4)
        code = tl.where(v < 2.0, v * 8.0, tl.where(v < 4.0, 16.0 + (v - 2.0) * 4.0, 24.0 + (v - 4.0) * 2.0))
        code = code.to(tl.int32)
        sign = ((x < 0) & (code > 0)).to(tl.int32)
        return code | (sign << 5)

    @triton.jit
    def _group_scales(sp, h, g, gm, D: tl.constexpr, MODE: tl.constexpr):
        """The scales of the 4 elements of each packed group: per channel
        (MODE 2), per KV head (1) or per tensor (0)."""
        if MODE == 2:
            s0 = tl.load(sp + h * D + 4 * g + 0, mask=gm, other=1.0)
            s1 = tl.load(sp + h * D + 4 * g + 1, mask=gm, other=1.0)
            s2 = tl.load(sp + h * D + 4 * g + 2, mask=gm, other=1.0)
            s3 = tl.load(sp + h * D + 4 * g + 3, mask=gm, other=1.0)
        else:
            if MODE == 1:
                sc = tl.load(sp + h)
            else:
                sc = tl.load(sp)
            s0 = tl.full(g.shape, 1.0, tl.float32) * sc
            s1 = s0
            s2 = s0
            s3 = s0
        return s0, s1, s2, s3

    @triton.jit
    def _write_kv_kernel(
        key_ptr,
        value_ptr,
        cache_ptr,
        slot_ptr,
        k_scale_ptr,
        v_scale_ptr,
        stride_kt,
        stride_kh,
        stride_vt,
        stride_vh,
        stride_cb,
        stride_ch,
        stride_cn,
        page_size,
        K_SCALE_MODE: tl.constexpr,  # 0 per tensor, 1 per KV head, 2 per (KV head, channel)
        V_SCALE_MODE: tl.constexpr,
        D: tl.constexpr,
        G_BLOCK: tl.constexpr,
    ):
        t = tl.program_id(0)
        h = tl.program_id(1)
        slot = tl.load(slot_ptr + t).to(tl.int64)
        if slot < 0:  # padding token
            return
        blk = slot // page_size
        off = slot % page_size
        G: tl.constexpr = D // 4
        RB: tl.constexpr = D * 3 // 4
        g = tl.arange(0, G_BLOCK)
        gm = g < G
        dst = cache_ptr + blk * stride_cb + h * stride_ch + off * stride_cn
        for side in tl.static_range(2):
            if side == 0:
                src = key_ptr + t * stride_kt + h * stride_kh
                s0, s1, s2, s3 = _group_scales(k_scale_ptr, h, g, gm, D, K_SCALE_MODE)
            else:
                src = value_ptr + t * stride_vt + h * stride_vh
                s0, s1, s2, s3 = _group_scales(v_scale_ptr, h, g, gm, D, V_SCALE_MODE)
            # x / scale, correctly rounded (div_rn): the same codes as the reference quantizer
            c0 = _e2m3_codes(tl.math.div_rn(tl.load(src + 4 * g + 0, mask=gm, other=0.0).to(tl.float32), s0))
            c1 = _e2m3_codes(tl.math.div_rn(tl.load(src + 4 * g + 1, mask=gm, other=0.0).to(tl.float32), s1))
            c2 = _e2m3_codes(tl.math.div_rn(tl.load(src + 4 * g + 2, mask=gm, other=0.0).to(tl.float32), s2))
            c3 = _e2m3_codes(tl.math.div_rn(tl.load(src + 4 * g + 3, mask=gm, other=0.0).to(tl.float32), s3))
            w = c0 | (c1 << 6) | (c2 << 12) | (c3 << 18)
            base = dst + side * RB + 3 * g
            tl.store(base + 0, (w & 0xFF).to(tl.uint8), mask=gm)
            tl.store(base + 1, ((w >> 8) & 0xFF).to(tl.uint8), mask=gm)
            tl.store(base + 2, ((w >> 16) & 0xFF).to(tl.uint8), mask=gm)

    @triton.jit
    def _expand_kernel(
        cache_ptr,
        src_block_ptr,
        out_ptr,
        lut_ptr,
        stride_cb,
        stride_ch,
        stride_cn,
        stride_ob,
        stride_oh,
        stride_on,
        k_ratio_ptr,
        v_ratio_ptr,
        lutf_ptr,
        RATIO: tl.constexpr,  # apply per-(head, channel) ratios: e4m3(value * ratio)
        D: tl.constexpr,
        G_BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0)  # scratch page
        h = tl.program_id(1)
        n = tl.program_id(2)  # row in the page
        src_blk = tl.load(src_block_ptr + i).to(tl.int64)
        G: tl.constexpr = D // 4
        RB: tl.constexpr = D * 3 // 4
        g = tl.arange(0, G_BLOCK)
        gm = g < G
        src = cache_ptr + src_blk * stride_cb + h * stride_ch + n * stride_cn
        dst = out_ptr + i.to(tl.int64) * stride_ob + h * stride_oh + n * stride_on
        for side in tl.static_range(2):
            b = src + side * RB + 3 * g
            w = (
                tl.load(b + 0, mask=gm, other=0).to(tl.int32)
                | (tl.load(b + 1, mask=gm, other=0).to(tl.int32) << 8)
                | (tl.load(b + 2, mask=gm, other=0).to(tl.int32) << 16)
            )
            o = dst + side * D + 4 * g
            if side == 0:
                rp = k_ratio_ptr
            else:
                rp = v_ratio_ptr
            for k in tl.static_range(4):
                code = (w >> (6 * k)) & 63
                if RATIO:
                    r = tl.load(rp + h * D + 4 * g + k, mask=gm, other=0.0)
                    val = tl.load(lutf_ptr + code) * r
                    tl.store(o + k, val.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=gm)
                else:
                    tl.store(o + k, tl.load(lut_ptr + code), mask=gm)


def _scale_mode(scale: torch.Tensor, h_k: int, d: int) -> int:
    """0: per tensor, 1: per KV head, 2: per (KV head, channel) — contiguous."""
    n = scale.numel()
    if n == h_k * d and d > 1 and (h_k > 1 or n > 1):
        assert scale.is_contiguous(), "per-channel KV scales must be contiguous (h_k, d)"
        return 2
    assert n in (1, h_k), f"KV scale must be per tensor, per KV head or (h_k, d), got {tuple(scale.shape)}"
    return 1 if n == h_k and h_k > 1 else 0


def write_kv(
    key: torch.Tensor,  # (>= T, h_k, d) bf16/fp16, d contiguous
    value: torch.Tensor,
    kv_cache: torch.Tensor,  # (B, h_k, N, 2 * row_bytes) uint8 layer view, any strides, unit last stride
    slot_mapping: torch.Tensor,  # (T,) int, -1 = padding
    k_scale: torch.Tensor,  # fp32: 1 elem, (h_k,) or (h_k, d)
    v_scale: torch.Tensor,
) -> None:
    """Quantize + pack + scatter K/V rows of T tokens into the e2m3 cache."""
    T = slot_mapping.shape[0]
    if T == 0:
        return
    _, h_k, d = key.shape
    _, H, N, C = kv_cache.shape
    assert h_k == H and 2 * row_bytes(d) == C and kv_cache.stride(3) == 1 and kv_cache.dtype == torch.uint8
    assert key.stride(2) == 1 and value.stride(2) == 1
    _write_kv_kernel[(T, h_k)](
        key,
        value,
        kv_cache,
        slot_mapping,
        k_scale,
        v_scale,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        N,
        K_SCALE_MODE=_scale_mode(k_scale, h_k, d),
        V_SCALE_MODE=_scale_mode(v_scale, h_k, d),
        D=d,
        G_BLOCK=triton.next_power_of_2(d // 4),
        num_warps=2,
    )


def expand_pages_to_fp8(
    kv_cache: torch.Tensor,  # (B, h_k, N, 2 * row_bytes) uint8 e2m3 layer view
    src_blocks: torch.Tensor,  # (P,) int32: cache page of each scratch page
    out: torch.Tensor,  # (>= P, h_k, N, 2 * d) uint8 (e4m3 bytes), unit last stride
    head_size: int,
    k_ratio: torch.Tensor | None = None,  # (h_k, d) fp32: per-channel scale / per-head descale
    v_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand P pages (K and V) to e4m3 bytes in ``out[:P]``; returns it.

    Without ratios the e4m3 bytes are the e2m3 values exactly (the cache's
    scales are the descales). With per-channel scales s_c and a per-head
    descale S, pass ratio = s_c / S: each value becomes e4m3(code * s_c / S),
    so the expanded page carries the channel scales and S stays the descale."""
    P = src_blocks.shape[0]
    _, H, N, C = kv_cache.shape
    d = head_size
    assert 2 * row_bytes(d) == C and out.shape[1:] == (H, N, 2 * d) and out.stride(3) == 1
    if P == 0:
        return out[:0]
    has_ratio = k_ratio is not None or v_ratio is not None
    mixed = (k_ratio is None) != (v_ratio is None)
    # Triton's fp8e4nv needs sm_89+; sm_87 has no e4m3 cvt, so the continuous
    # requant falls back to torch there. sm_89+ keeps the fused RATIO kernel.
    if has_ratio and (mixed or torch.cuda.get_device_capability(kv_cache.device) < (8, 9)):
        return _expand_ratio_torch(kv_cache, src_blocks, out, d, k_ratio, v_ratio)
    _expand_kernel[(P, H, N)](
        kv_cache,
        src_blocks,
        out,
        e2m3_to_e4m3_lut(kv_cache.device),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        k_ratio if k_ratio is not None else out,
        v_ratio if v_ratio is not None else out,
        e2m3_values(kv_cache.device),
        RATIO=has_ratio,
        D=d,
        G_BLOCK=triton.next_power_of_2(d // 4),
        num_warps=1,
    )
    return out[:P]


def _expand_ratio_torch(
    kv_cache: torch.Tensor,  # (B, h_k, N, 2 * row_bytes) uint8 e2m3 layer view
    src_blocks: torch.Tensor,  # (P,) int32
    out: torch.Tensor,  # (>= P, h_k, N, 2 * d) uint8 (e4m3 bytes), unit last stride
    d: int,
    k_ratio: torch.Tensor | None,  # (h_k, d) fp32
    v_ratio: torch.Tensor | None,
) -> torch.Tensor:
    """The ratio (per-channel M2) expansion in torch: Triton's ``fp8e4nv``
    needs sm_89+ and sm_87 has no e4m3 cvt, so the continuous requant runs
    on torch's software cast. The no-ratio path (the pure LUT bytes) stays
    in the Triton kernel."""
    P = src_blocks.shape[0]
    H = kv_cache.shape[1]
    rb = row_bytes(d)
    dev = kv_cache.device
    pages = kv_cache[src_blocks.to(torch.int64)]  # (P, H, N, 2*rb) uint8
    idx = torch.arange(d, device=dev)
    bb, oo = (idx * 6) // 8, (idx * 6) % 8
    hi = (bb + 1).clamp(max=rb - 1)
    lut = e2m3_values(dev)
    halves = []
    for side, ratio in enumerate((k_ratio, v_ratio)):
        half = pages[..., side * rb : (side + 1) * rb]
        codes = ((half[..., bb].int() >> oo) | (half[..., hi].int() << (8 - oo))) & 0x3F
        vals = lut[codes.reshape(-1)].reshape(*codes.shape)  # (P, H, N, d) fp32
        if ratio is not None:
            vals = vals * ratio.view(1, H, 1, d).to(torch.float32)
        halves.append(vals.to(torch.float8_e4m3fn).view(torch.uint8))
    out[:P] = torch.cat(halves, dim=-1)
    return out[:P]


# ---------------------------------------------------------------------------
# References (tests): the same math in PyTorch, via the kernel package's
# own quantizer / packer, so the Triton kernels are checked against them.
# ---------------------------------------------------------------------------
def quantize_rows_reference(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """(..., h_k, d) float -> packed bytes (..., h_k, row_bytes(d)) uint8;
    scale per tensor, per KV head or (h_k, d) per channel."""
    from .quant import minifloat_quantize, pack_subbyte

    h_k, d = x.shape[-2], x.shape[-1]
    s = scale.float().cpu()
    if s.numel() == h_k * d and d > 1:
        s = s.reshape(h_k, d)
    else:
        s = s.reshape(-1)
        s = (s.expand(h_k) if s.numel() == 1 else s).reshape(h_k, 1)
    codes, _ = minifloat_quantize(x.float().cpu() / s, "e2m3")
    return pack_subbyte(codes, BITS)


def decode_rows_reference(packed: torch.Tensor, d: int) -> torch.Tensor:
    """packed (..., row_bytes) uint8 -> e2m3 code values (..., d) float."""
    from .quant import minifloat_table

    b = packed.to(torch.int64).cpu().reshape(*packed.shape[:-1], d // 4, 3)
    w = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    codes = torch.stack([(w >> (6 * k)) & 63 for k in range(4)], dim=-1).reshape(*packed.shape[:-1], d)
    return minifloat_table("e2m3")[codes]
