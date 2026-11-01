"""The nvfp4 (e2m1 + per-16 e4m3 scales) KV format — the fp4 campaign's P2.

The row layout (per token-head, hd256): [128 B data | 16 B e4m3 scales] =
144 B — the scale plane rides the kernel's cp.async staging row verbatim
(the g2s chunks it like the data; the dequant splits by the offset).

The code semantics: the IEEE e2m1 {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}; the
codes 2/byte LSB-first (the code k at the bits [4k, 4k+4)); the scale for
the channels [16g, 16g+16) = the e4m3 byte at the scale-plane offset g =
the group amax / 6 (the e4m3-encodable; the amax/6 <= 448 for the amax <=
2688 — the KV range). The dequant: the marlin kFE2M1f bf16 recipe (the
denormal-bias construction, the two-multiply scale — the P1 reference,
fp4_reference.py).

The reference functions are the torch (the correctness anchors + the
harness's make_case arm); the write path mirrors kvfp6.write_kv's contract
for the serving phase (P5).
"""

import torch

E2M1_MAX = 6.0
BITS = 4
GROUP = 16


def row_bytes(head_size: int) -> int:
    """The staged row: the packed data + the e4m3 scale plane."""
    assert head_size % (2 * GROUP) == 0
    return head_size // 2 + head_size // GROUP


def data_bytes(head_size: int) -> int:
    return head_size // 2


_E2M1: dict = {}


def e2m1_values(device) -> torch.Tensor:
    """(16,) fp32: the value of every e2m1 code (the sign-magnitude layout
    s<<3 | e<<1 | m — the dequant = the table[code])."""
    key = str(device)
    if key not in _E2M1:
        vals = []
        for code in range(16):
            s = (code >> 3) & 1
            e = (code >> 1) & 3
            m = code & 1
            v = (m / 2) * 2.0 ** (1 - 1) if e == 0 else (1 + m / 2) * 2.0 ** (e - 1)
            vals.append(-v if s else v)
        _E2M1[key] = torch.tensor(vals, dtype=torch.float32)
    return _E2M1[key]


def quantize_rows_reference(x: torch.Tensor, group: int = GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference quantize: x [.., d] fp32 -> (the codes int32 [.., d]
    (the sign-magnitude: s<<3 | e<<1 | m), the scales [.., d/group] fp32).
    The nearest-code; the scale = the group amax / 6 (the fp32 here — the
    e4m3 rounding happens at the pack)."""
    g = x.float().reshape(*x.shape[:-1], x.shape[-1] // group, group)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    # the scale is e4m3 BEFORE the code pick — the decode only ever sees the
    # e4m3 scale, so the encode must too (else a code lands one step off)
    s = (amax / E2M1_MAX).to(torch.float8_e4m3fn).float().reshape(*x.shape[:-1], x.shape[-1] // group)
    tab = e2m1_values(x.device)
    mids = (tab[1:] + tab[:-1]) / 2
    codes = torch.zeros(g.shape, dtype=torch.int32)
    for gi in range(g.shape[-2]):
        col = g[..., gi, :] / s[..., gi, None]
        idx = torch.bucketize(col.abs().clamp(max=E2M1_MAX), mids).clamp_max(7)
        codes[..., gi, :] = ((torch.sign(col) < 0).long() << 3) | idx
    codes = codes.reshape(*x.shape)
    return codes, s.reshape(*x.shape[:-1], x.shape[-1] // group)


def pack_rows(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The codes [.., d] (the int per element) + the scales [.., d/16] ->
    the packed uint8 rows [.., row_bytes(d)]: the data 2 codes/byte LSB-first
    (the byte i = the code 2i | the code 2i+1 << 4) + the e4m3 scale plane."""
    d = q.shape[-1]
    codes = (q.reshape(-1, d).to(torch.int8)) & 0xF
    data = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    data = data.reshape(*q.shape[:-1], d // 2)
    sc = scales.float().to(torch.float8_e4m3fn).view(torch.uint8)
    return torch.cat([data, sc.reshape(*q.shape[:-1], d // GROUP)], dim=-1)


def dequant_lut() -> torch.Tensor:
    """The 16-entry Int32 LUT for the kernel's smem dequant table: the TRUE
    bf16 e2m1 value of every code (the sign-magnitude layout s<<3|e<<1|m),
    stored as the int32 bit patterns — the u32 slots the kernel's PRMT
    pack consumes directly.

    The values are all normals (min 0.5) — no denormal ever enters the
    data path, so the sm_87 FTZ question only touches the scale (the
    verified fp8 recipe, same caveat as the fp8 path). The table is the
    bit-exact marlin construction of the P1 reference (fp4_reference.py
    decode_marlin) — the smem LUT just precomputes it, which cuts the
    dequant to 2 LDS.32 + 1 PRMT + 1 HMUL2 per pair. (The gmem-LUT
    variant measured 28% DRAM SOL: the L1 gather throughput walls the
    dequant; the 64-B smem table restores the byte cut. The 1-KB
    pair-LUT variant would break the 2-CTA smem fit.)
    """
    tab = e2m1_values("cpu").to(torch.bfloat16).view(torch.uint16)
    return tab.to(torch.int32)


def decode_rows_reference(packed: torch.Tensor, d: int) -> torch.Tensor:
    """The packed rows [.., row_bytes(d)] -> the dequantized fp32 [.., d].
    The marlin semantics: the code -> the bf16 denormal construction x 2^126,
    then the scale. The reference: the table lookup (the same values)."""
    data = packed[..., : d // 2].to(torch.int16)
    scales = packed[..., d // 2 :].view(torch.float8_e4m3fn).float()
    lo = (data & 0xF).long()
    hi = ((data >> 4) & 0xF).long()
    tab = e2m1_values(packed.device)
    out = torch.empty(*packed.shape[:-1], d, dtype=torch.float32)
    for i in range(d):
        code = lo[..., i // 2] if i % 2 == 0 else hi[..., i // 2]
        out[..., i] = tab[code] * scales[..., i // GROUP]
    return out


def expand_pages_to_fp8(
    kv_cache: torch.Tensor,  # (B, h_k, N, row_bytes) uint8 nvfp4 layer view
    src_blocks: torch.Tensor,
    out: torch.Tensor,  # (>= P, h_k, N, 2, d) fp8 bytes: the e4m3-expanded K/V
) -> None:
    """The fallback expansion for the non-kernel consumers (the reference
    decode into the e4m3 bytes; the scale folded via the per-group cast)."""
    raise NotImplementedError("P5: the serving-phase expansion")
