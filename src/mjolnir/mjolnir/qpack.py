"""Fused dynamic fp8 Q quantization + packing for the Mjolnir decode kernel.

The kernel reads Q as ``(b, h_k, n_q, d)`` e4m3 with one descale per
``(sequence, KV head)``, where row ``n = qi * h_r + r`` holds query token
``qi`` of KV-head group member ``r``. vLLM hands Q over as ``(tokens, h_q, d)``
bf16/fp16, any token and head stride (often a view of the fused QKV buffer).

Doing that with PyTorch ops costs ~10 launches per layer (cast, abs, amax,
clamp, divide, copy, divide, cast, permute-copy, copy) — at short contexts
more GPU time than the attention kernel saves. ``pack_q_fp8`` is one Triton
launch: one program per (sequence, KV head) loads its ``q_len * h_r * d``
values once, takes the amax, writes the descale and the packed e4m3 rows.
"""

from __future__ import annotations

import torch

try:  # Triton ships with vLLM / torch on CUDA; the harness may run without it
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - no Triton: the torch path is used
    triton = None
    tl = None

FP8_MAX = 448.0


def available() -> bool:
    return triton is not None


if triton is not None:

    @triton.jit
    def _pack_q_fp8_kernel(
        q_ptr,
        out_ptr,
        qd_ptr,
        mul_ptr,
        stride_t,
        stride_h,
        q_len: tl.constexpr,
        h_k: tl.constexpr,
        h_r: tl.constexpr,
        d: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FP8_MAX_C: tl.constexpr,
        HAS_MUL: tl.constexpr,
    ):
        b = tl.program_id(0)
        k = tl.program_id(1)
        n_q = q_len * h_r
        rows = tl.arange(0, BLOCK_N)  # n = qi * h_r + r
        cols = tl.arange(0, BLOCK_D)
        qi = rows // h_r
        r = rows % h_r
        token = b * q_len + qi
        head = k * h_r + r
        mask = (rows[:, None] < n_q) & (cols[None, :] < d)
        src = q_ptr + token[:, None] * stride_t + head[:, None] * stride_h + cols[None, :]
        x = tl.load(src, mask=mask, other=0.0).to(tl.float32)
        if HAS_MUL:  # per-(q head, channel) factor: fp6 K channel scales folded into Q
            x = x * tl.load(mul_ptr + head[:, None] * d + cols[None, :], mask=mask, other=0.0)
        amax = tl.maximum(tl.max(tl.max(tl.abs(x), axis=1), axis=0), 1e-6)
        scale = amax / FP8_MAX_C
        tl.store(qd_ptr + b * h_k + k, scale)
        y = tl.minimum(tl.maximum(x / scale, -FP8_MAX_C), FP8_MAX_C)
        dst = out_ptr + ((b * h_k + k) * n_q + rows[:, None]) * d + cols[None, :]
        tl.store(dst, y.to(out_ptr.dtype.element_ty), mask=mask)


def pack_q_fp8(
    q3: torch.Tensor,
    b: int,
    q_len: int,
    h_k: int,
    h_r: int,
    out: torch.Tensor,
    qd: torch.Tensor,
    mul: torch.Tensor | None = None,
) -> None:
    """Quantize ``q3`` (``(b * q_len, h_k * h_r, d)``, unit stride on ``d``)
    into ``out`` (contiguous ``(b, h_k, q_len * h_r, d)`` e4m3) and ``qd``
    (contiguous ``(b, h_k)`` fp32 descales: amax / 448). ``mul`` (contiguous
    ``(h_k * h_r, d)`` fp32) multiplies Q before quantization. One launch."""
    d = q3.shape[-1]
    assert q3.stride(-1) == 1, "Q needs a unit stride on head_dim"
    assert out.is_contiguous() and qd.is_contiguous()
    if mul is not None:
        assert mul.is_contiguous() and mul.dtype == torch.float32 and mul.numel() == h_k * h_r * d
    n_q = q_len * h_r
    block_n = triton.next_power_of_2(n_q)
    block_d = triton.next_power_of_2(d)
    warps = 8 if block_n * block_d >= 4096 else 4
    _pack_q_fp8_kernel[(b, h_k)](
        q3,
        out,
        qd,
        mul if mul is not None else qd,
        q3.stride(0),
        q3.stride(1),
        q_len=q_len,
        h_k=h_k,
        h_r=h_r,
        d=d,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        FP8_MAX_C=FP8_MAX,
        HAS_MUL=mul is not None,
        num_warps=warps,
    )


def pack_q_fp8_reference(q3: torch.Tensor, b: int, q_len: int, h_k: int, h_r: int, mul: torch.Tensor | None = None):
    """The PyTorch formulation (the runner's fallback, and the test oracle)."""
    d = q3.shape[-1]
    qg = q3[: b * q_len].float()
    if mul is not None:
        qg = qg * mul.view(h_k * h_r, d)
    qg = qg.reshape(b, q_len, h_k, h_r, d)
    amax = qg.abs().amax(dim=(1, 3, 4)).clamp_min(1e-6)  # (b, h_k)
    scale = amax / FP8_MAX
    packed = (
        (qg / scale[:, None, :, None, None])
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .permute(0, 2, 1, 3, 4)
        .reshape(b, h_k, q_len * h_r, d)
    )
    return packed, scale
