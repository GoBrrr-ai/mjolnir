"""Narrow KV/Q format helpers: value tables, quantization and bit packing.

Split out of mjolnir/t5_decode.py (file-length ratchet); used by the kernel
package's fp6 path (mjolnir.kvfp6) and the standalone harness.
"""

import torch

NVFP4_GROUP = 16
NVFP4_SHIFT = 0.125

# Narrow KV formats: value tables, quantization and bit packing
# ----------------------------------------------------------------------------
_MINIFLOAT = {  # format: (exponent bits, mantissa bits, bias); no inf / nan
    "e3m2": (3, 2, 3),
    "e2m3": (2, 3, 1),
    "e2m1": (2, 1, 1),
}


def minifloat_table(fmt):
    """Values of every code of an fp6 / fp4 format (code = sign|exp|mant)."""
    eb, mb, bias = _MINIFLOAT[fmt]
    vals = []
    for code in range(1 << (1 + eb + mb)):
        sign = -1.0 if code >> (eb + mb) else 1.0
        e = (code >> mb) & ((1 << eb) - 1)
        m = code & ((1 << mb) - 1)
        v = (m / (1 << mb)) * 2.0 ** (1 - bias) if e == 0 else (1 + m / (1 << mb)) * 2.0 ** (e - bias)
        vals.append(sign * v)
    return torch.tensor(vals, dtype=torch.float32)


def minifloat_quantize(x, fmt):
    """Round-to-nearest (saturating) codes of x in an fp6 / fp4 format.
    Returns (codes uint8, decoded float values)."""
    table = minifloat_table(fmt)
    pos = table[: len(table) // 2]  # non-negative half, ascending
    ax = x.abs().clamp(max=float(pos[-1]))
    idx = torch.bucketize(ax, pos)  # first pos >= ax
    lo = (idx - 1).clamp(min=0)
    hi = idx.clamp(max=len(pos) - 1)
    pick_lo = (ax - pos[lo]) <= (pos[hi] - ax)
    mag = torch.where(pick_lo, lo, hi)
    neg = (x < 0) & (pos[mag] > 0)
    codes = (mag + neg.to(mag.dtype) * (len(table) // 2)).to(torch.uint8)
    return codes, table[codes.long()]


def pack_subbyte(codes, bits):
    """Pack codes (..., d) into bytes (..., d * bits / 8): little-endian
    bitstream, element i at bits [bits*i, bits*i + bits) (fp4: element 2i in
    the low nibble)."""
    c = codes.to(torch.int64)
    d = c.shape[-1]
    if bits == 8:
        return codes.to(torch.uint8)
    if bits == 4:
        c = c.reshape(*c.shape[:-1], d // 2, 2)
        return (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)
    assert bits == 6 and d % 4 == 0
    c = c.reshape(*c.shape[:-1], d // 4, 4)
    w = c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)
    out = torch.stack([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF], dim=-1)
    return out.reshape(*c.shape[:-2], d // 4 * 3).to(torch.uint8)


def nvfp4_quantize(x, group=NVFP4_GROUP):
    """NVFP4: e2m1 codes, one e4m3 scale per `group` elements of the last dim
    and one fp32 global scale. Returns (codes uint8, scales e4m3, global
    scale, values the T5 kernel feeds its fp8 MMA, descale to pass).
    Kernel value = e4m3(e2m1 * scale * 2^-3); descale = global * 2^3."""
    *lead, d = x.shape
    xg = x.reshape(*lead, d // group, group)
    amax_g = xg.abs().amax(dim=-1)
    gs = float(x.abs().amax().clamp(min=1e-12)) / (6.0 * 448.0)
    sc = (amax_g / 6.0 / gs).clamp(max=448.0).to(torch.float8_e4m3fn)
    scf = sc.float()
    denom = (scf * gs).unsqueeze(-1)
    q = torch.where(denom > 0, xg / denom.clamp(min=1e-30), torch.zeros_like(xg))
    codes, _ = minifloat_quantize(q, "e2m1")
    lut = minifloat_table("e2m1")
    kern = (lut[codes.long()] * scf.unsqueeze(-1) * NVFP4_SHIFT).to(torch.float8_e4m3fn)
    return (codes.reshape(*lead, d), sc, gs, kern.float().reshape(*lead, d), gs / NVFP4_SHIFT)


def pack_q(q, num_seqs, q_len, heads_k):
    """vLLM q (num_seqs * q_len, h_q, d) -> kernel q (b, h_k, q_len * h_r, d).
    A free view for q_len == 1."""
    _, h_q, d = q.shape
    h_r = h_q // heads_k
    if q_len == 1:
        return q.view(num_seqs, heads_k, h_r, d)
    return (
        q.view(num_seqs, q_len, heads_k, h_r, d)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_seqs, heads_k, q_len * h_r, d)
        .contiguous()
    )


def unpack_o(o, num_seqs, q_len, heads_k):
    """kernel o (b, h_k, q_len * h_r, d) -> vLLM (num_seqs * q_len, h_q, d)."""
    b, h_k, n_q, d = o.shape
    h_r = n_q // q_len
    if q_len == 1:
        return o.view(num_seqs, h_k * h_r, d)
    return o.view(b, h_k, q_len, h_r, d).permute(0, 2, 1, 3, 4).reshape(b * q_len, h_k * h_r, d)


def quantize_q_fp8(q, num_seqs, q_len, heads_k):
    """Per-(seq, kv head) e4m3 quantization of q (num_seqs * q_len, h_q, d).
    Returns (q_fp8 same shape, q_descale (num_seqs, h_k) fp32)."""
    _, h_q, d = q.shape
    h_r = h_q // heads_k
    qg = q.float().view(num_seqs, q_len, heads_k, h_r, d)
    amax = qg.abs().amax(dim=(1, 3, 4)).clamp(min=1e-6)  # (b, h_k)
    q_descale = (amax / 448.0).contiguous()
    q8 = (qg / q_descale[:, None, :, None, None]).to(torch.float8_e4m3fn)
    return q8.view(num_seqs * q_len, h_q, d), q_descale
