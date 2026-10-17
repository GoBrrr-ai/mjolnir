"""Opt-in lm_head experiments for decode (vLLM GPU model runner).

On Qwen3.8-27B NVFP4 the bf16 lm_head (248,320 x 5,120, 2.54 GB) is read by
the target's verify and by every MTP draft step — ~40% of a decode step's
bytes. Two experiments, both off unless their env var is set (a config opts
in through the ``.env`` sidecar inside its config dir, see
``mjolnir_cli.dockerctl``):

* ``MJOLNIR_LM_HEAD=fp8`` — the lm_head weight quantized to e4m3 per row at
  load (scale = row amax / 448) and multiplied by a Triton fp8-weight GEMM
  (bf16 activations, fp32 accumulate): half the bytes for the verify and the
  drafts. The fp8 head makes the bf16 copy dead weight, so the bf16 weight is
  freed automatically (an untied head only), returning 2.5 GB to the KV cache.
* ``MJOLNIR_DRAFT_VOCAB=<json>`` (+ ``MJOLNIR_DRAFT_VOCAB_SIZE``) — VocabTrim
  for the drafter: its logits are computed only over the most frequent
  target tokens (``mjolnir calibrate --draft-vocab``), the rest are -inf.
  Verification still uses the full vocabulary, so the output distribution is
  unchanged (rejection sampling stays exact); only acceptance can drop. The
  kept head stays in the drafter's *native* quant (NVFP4 4-bit in-kernel, FP8
  row-slice) with no dequant->bf16 up-scale; the one exception is a bf16 head,
  which goes to fp8 only when ``MJOLNIR_LM_HEAD=fp8`` is set.

The hooks wrap ``GPUModelRunner.load_model`` (installed by
``mjolnir.register`` when either variable is set) and run after the target
and the drafter are loaded, before CUDA-graph capture.
"""

from __future__ import annotations

import json
import logging
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

_log = logging.getLogger("vllm.mjolnir")

ENV_LM_HEAD = "MJOLNIR_LM_HEAD"
ENV_DRAFT_VOCAB = "MJOLNIR_DRAFT_VOCAB"
ENV_DRAFT_VOCAB_SIZE = "MJOLNIR_DRAFT_VOCAB_SIZE"
# Native VT is automatic while the VT env is set: a quantized head keeps its
# native format (NVFP4 4-bit in-kernel, FP8 row-slice) instead of dequating to
# bf16/fp8. Set to 0/off to force the dequant->fp8 fallback (the A/B control).
# The NVFP4 in-kernel GEMM is CUDA-only and self-checked; A/B on the Thor.
ENV_DRAFT_VOCAB_NATIVE = "MJOLNIR_DRAFT_VOCAB_NATIVE"
FP8_MAX = 448.0
SMALL_M = 64  # beyond this many rows (prompt logprobs) the dequantized chunks win


def wanted() -> bool:
    return bool(os.environ.get(ENV_LM_HEAD, "").strip() or os.environ.get(ENV_DRAFT_VOCAB, "").strip())


# ── fp8-weight GEMM ─────────────────────────────────────────────────────────

if triton is not None:

    @triton.jit
    def _fp8_head_kernel(
        x_ptr,
        w_ptr,
        s_ptr,
        o_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_om,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + rk
            x = tl.load(
                x_ptr + rm[:, None] * stride_xm + kk[None, :],
                mask=(rm[:, None] < M) & (kk[None, :] < K),
                other=0.0,
            )
            w = tl.load(
                w_ptr + rn[:, None] * K + kk[None, :],
                mask=(rn[:, None] < N) & (kk[None, :] < K),
                other=0.0,
            )
            acc += tl.dot(x.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
        s = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
        acc = acc * s[None, :]
        tl.store(
            o_ptr + rm[:, None] * stride_om + rn[None, :],
            acc.to(o_ptr.dtype.element_ty),
            mask=(rm[:, None] < M) & (rn[None, :] < N),
        )


ROW_CHUNK = 4096  # rows per step when quantizing / testing a head (bounds the temporaries)


def quantize_rows(w: torch.Tensor, chunk: int = ROW_CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, K) float -> (N, K) e4m3 and (N,) fp32 row scales (amax / 448).

    Row chunks of ``chunk``: a whole-head fp32 copy of the 248K x 5120 lm_head
    is 5 GiB, and the abs / divide / clamp temporaries of one expression are
    several more. Freed or not, the caching allocator keeps them reserved,
    and vLLM's startup memory profile (on Thor: one unified pool) counts them
    as used — they came straight out of the KV cache. Chunked, the peak is a
    few hundred MiB."""
    N = w.shape[0]
    q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty(N, dtype=torch.float32, device=w.device)
    for r0 in range(0, N, chunk):
        w32 = w[r0 : r0 + chunk].float()
        sc = (w32.abs().amax(dim=1) / FP8_MAX).clamp_min(1e-12)
        q[r0 : r0 + chunk] = (w32.div_(sc[:, None])).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        scale[r0 : r0 + chunk] = sc
        del w32
    return q, scale


def fp8_linear_reference(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x.float() @ (w8.float() * scale[:, None]).t()).to(x.dtype)


def fp8_linear(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """x (..., K) bf16/fp16 @ (w8 * scale)^T -> (..., N), x's dtype."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    M, K = x2.shape
    N = w8.shape[0]
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    if M <= SMALL_M and triton is not None:
        BM, BN, BK = 16, 64, 128
        grid = (triton.cdiv(N, BN), triton.cdiv(M, BM))
        _fp8_head_kernel[grid](
            x2, w8, scale, out, M, N, K, x2.stride(0), out.stride(0), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            num_warps=4, num_stages=4,
        )  # fmt: skip
    else:  # many rows: dequantize a slab of the head at a time
        for n0 in range(0, N, chunk):
            w = w8[n0 : n0 + chunk].to(x.dtype) * scale[n0 : n0 + chunk, None].to(x.dtype)
            out[:, n0 : n0 + chunk] = x2 @ w.t()
    return out.reshape(*shape[:-1], N)


class Fp8HeadMethod:
    """Stands in for the lm_head's quant method: ``apply`` runs the fp8
    GEMM; everything else (``embedding`` for a tied head, attributes) goes
    to the original method."""

    def __init__(self, base, w8: torch.Tensor, scale: torch.Tensor):
        self._base = base
        self.w8 = w8
        self.scale = scale

    def apply(self, layer, x, bias=None):
        out = fp8_linear(x, self.w8, self.scale)
        return out + bias if bias is not None else out

    def __getattr__(self, name):
        return getattr(self._base, name)


def _self_test(w: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor) -> float:
    """Relative error of the fp8 head vs bf16 on random activations."""
    g = torch.Generator(device=w.device).manual_seed(0)
    x = torch.randn(4, w.shape[1], device=w.device, dtype=w.dtype, generator=g)
    rows = torch.arange(0, w.shape[0], max(1, w.shape[0] // ROW_CHUNK), device=w.device)[:ROW_CHUNK]
    ref = x.float() @ w.index_select(0, rows).float().t()  # a strided row sample: no full fp32 copy
    w8s = w8.view(torch.uint8).index_select(0, rows).view(torch.float8_e4m3fn)  # fp8 has no index_select kernel
    got = fp8_linear(x, w8s, scale.index_select(0, rows)).float()
    return float((got - ref).norm() / ref.norm().clamp_min(1e-12))


def _is_tied(lm_head, model) -> bool:
    ptr = lm_head.weight.data_ptr()
    for name, mod in model.named_modules():
        if mod is not lm_head and name.endswith("embed_tokens") and hasattr(mod, "weight"):
            if mod.weight.data_ptr() == ptr:
                return True
    return False


def install_fp8_head(lm_head, model=None) -> bool:
    if getattr(lm_head, "_mjolnir_fp8", False):
        return True
    qm = getattr(lm_head, "quant_method", None)
    # Only an unquantized (bf16/fp16) head benefits: this halves a 2-byte weight
    # to 1-byte FP8. A head that is already quantized (ModelOpt FP8 / NVFP4 /
    # MXFP8 — any non-Unquantized method) would be dequantized and re-packed at
    # >= the same width (FP8 is 1 byte vs NVFP4's 0.5) — a pessimization, so a
    # warning and a skip, never a re-quantization.
    if qm is None or "Unquantized" not in type(qm).__name__:
        _log.warning(
            "Mjolnir lm_head fp8: %s is not an unquantized head (already FP8/NVFP4?); left as is",
            type(qm).__name__,
        )
        return False
    w = lm_head.weight.data
    if w.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        _log.warning(
            "Mjolnir lm_head fp8: %s weight dtype %s is not full precision; left as is",
            type(qm).__name__,
            w.dtype,
        )
        return False
    w8, scale = quantize_rows(w)
    err = _self_test(w, w8, scale)
    if not err < 0.05:
        _log.warning("Mjolnir lm_head fp8: self-test relative error %.4f; left in bf16", err)
        return False
    lm_head.quant_method = Fp8HeadMethod(qm, w8, scale)
    lm_head._mjolnir_fp8 = True
    freed = ""
    if model is not None:  # the fp8 head is now the only consumer: the bf16 copy is dead weight
        if _is_tied(lm_head, model):
            freed = " (tied to the embedding: bf16 weight kept)"
        else:
            lm_head.weight.data = torch.empty(0, dtype=w.dtype, device=w.device)
            torch.cuda.empty_cache()
            freed = f", bf16 weight freed ({w.numel() * w.element_size() / 2**30:.2f} GiB)"
    _log.info(
        "Mjolnir lm_head fp8: %s x %s e4m3 per-row, self-test rel. error %.4f%s",
        w.shape[0],
        w.shape[1],
        err,
        freed,
    )
    return True


# ── VocabTrim for the drafter ───────────────────────────────────────────────


def load_vocab(path: str, size: int | None = None) -> list[int]:
    """Token ids, most frequent first, from a ``mjolnir.draft_vocab/1`` file
    (``{"format": ..., "tokens": [ids], ...}``) or a plain JSON list."""
    with open(path) as f:
        doc = json.load(f)
    ids = doc["tokens"] if isinstance(doc, dict) else doc
    ids = [int(t) for t in ids]
    return ids[:size] if size else ids


def pad_vocab(ids: list[int], size: int | None, vocab: int) -> list[int]:
    """Fill a short id list up to ``size`` with the lowest unused ids.

    A calibration that saw fewer distinct tokens than the requested size
    (6K distinct from ~50K generated tokens is typical) would otherwise trim
    the drafter far below the size the config names. Low ids come first
    because BPE vocabularies are ordered by merge rank, a fair frequency
    prior for tokens the calibration never generated."""
    seen = {i for i in ids if 0 <= i < vocab}
    out = [i for i in dict.fromkeys(ids) if 0 <= i < vocab]
    if size is None or len(out) >= size:
        return out[:size] if size else out
    need = min(size, vocab) - len(out)
    for i in range(vocab):
        if need <= 0:
            break
        if i not in seen:
            out.append(i)
            need -= 1
    return out


def _dequant_weight(lm_head, out_dtype) -> torch.Tensor:
    """A dequantized (vocab, hidden) [out, in] weight for a quantized head.

    vLLM's own helper normalizes the layout for any method: an unquantized
    weight passes through, per-tensor FP8 dequantizes and transposes its
    [in, out] storage back to [out, in], and packed methods (NVFP4 / MXFP8 /
    Marlin) dequantize via an identity-matrix matmul.
    """
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_and_maybe_dequant_weights,
    )

    return get_and_maybe_dequant_weights(lm_head, out_dtype=out_dtype)


# ── native VT: keep the trimmed rows in the source quant format ─────────────
#
# Prototype. A dequanted 4-bit head kept in bf16/fp8 wastes the calibration
# (the per-16 block scales) and stores 8/16 bits. Native VT instead slices the
# packed 4-bit rows + their block scales and dequants the kept subset in the
# GEMM, so the kept head stays 4-bit (~0.19 GB for 64K x 5120) and is
# bit-exact to the source head. The in-kernel GEMM is CUDA-only and UNTESTED
# here; it self-checks against the torch reference below on first use and
# falls back to the fp8 path if it disagrees, so it never serves wrong.


# OCP fp4 e2m1 code -> value. VERIFY against the model's head on the Thor
# (nibble order / LUT / global-scale application): if any is off the whole
# native path is off.
_NVFP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _nvfp4_dequant_rows(w, bs, gs, idx):
    """Reference dequant of the kept NVFP4 rows -> (kept, in) bf16.

    w   (out, in//2) uint8   2 fp4 nibbles/byte, low nibble = even input dim
    bs  (out, in//16) e4m3   the calibrated per-16 block scales
    gs  scalar              weight_global_scale
    """
    w = w.index_select(0, idx)
    bs = bs.index_select(0, idx)
    lo = w & 0xF
    hi = (w >> 4) & 0xF
    codes = torch.stack((lo, hi), dim=-1).reshape(w.shape[0], w.shape[1] * 2).to(torch.int32)
    lut = _NVFP4_LUT.to(codes.device)
    vals = lut[codes].to(torch.float32)
    kept, inp = w.shape[0], w.shape[1] * 2
    vals = (vals.reshape(kept, inp // 16, 16) * bs.float()[:, :, None]).reshape(kept, inp)
    return (vals * float(gs.item())).to(torch.bfloat16)


def _nvfp4_native_gemm(x, w, bs, gs):
    """CUDA in-kernel GEMM: x [M,K] bf16 @ dequant(w,bs,gs).T -> [M,kept] bf16.

    Dequants the 4-bit B operand in-kernel (nibbles -> e2m1, x per-16 block
    scale, x global) so the kept head stays 4-bit. Modeled on vLLM's
    nvfp4_gathered_bias. UNTESTED — see the self-check in _nvfp4_native_head.
    Returns None if triton is unavailable (caller falls back).
    """
    if triton is None:
        return None
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    BM = 64 if M < 64 else 128
    BN, BK = 128, 64

    @triton.jit
    def _k(X, W, BS, OUT, gs, M, N, K, sxm, swn, sbsn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pm, pn = tl.program_id(0), tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            x = tl.load(X + rm[:, None] * sxm + (k0 + tl.arange(0, BK))[None, :], mask=(rm[:, None] < M), other=0.0)
            wb = tl.load(
                W + rn[:, None] * swn + (k0 // 2 + tl.arange(0, BK // 2))[None, :], mask=(rn[:, None] < N), other=0
            )
            code = tl.reshape(tl.join(wb & 0xF, (wb >> 4) & 0xF), (BN, BK))
            e, m, s = (code >> 1) & 3, code & 1, (code >> 3) & 1
            mf = m.to(tl.float32)
            mag = tl.where(e == 0, 0.5 * mf, (1.0 + 0.5 * mf) * tl.exp2((e.to(tl.int32) - 1).to(tl.float32)))
            val = tl.where(s == 1, -mag, mag)
            bsc = tl.load(
                BS + rn[:, None] * sbsn + (k0 // 16 + tl.arange(0, BK // 16))[None, :], mask=(rn[:, None] < N), other=0
            ).to(tl.float32)
            b = val * tl.reshape(bsc[:, :, None], (BN, BK)) * gs
            acc += tl.dot(x.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)))
        tl.store(
            OUT + rm[:, None] * N + rn[None, :],
            acc.to(OUT.dtype.element_ty),
            mask=(rm[:, None] < M) & (rn[None, :] < N),
        )

    _k[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        x, w, bs, out, float(gs.item()), M, N, K, K, w.shape[1], bs.shape[1], BM=BM, BN=BN, BK=BK, num_warps=4
    )
    return out


def _nvfp4_native_head(lm_head, idx):
    """Build a native 4-bit kept head for the kept token ids. Returns a
    `head(h) -> [M,kept]` callable, or None if the head isn't a W4A16 NVFP4
    ModelOpt head / the self-check fails (caller uses the fp8 path).

    Two gates before the path is trusted (both run on the Thor): the torch
    reference is validated against vLLM's own dequant (the source of truth — a
    shared LUT/nibble-order error in the reference would otherwise pass the
    kernel-vs-reference self-check), and the in-kernel GEMM is self-checked
    against the reference at serving shapes (multi-tile, not just M=1). The
    reference validation does one full dequant (freed immediately) — the same
    transient the fp8 path pays; the steady-state kept head stays 4-bit.
    """
    packed = getattr(lm_head, "weight", None)
    bs = getattr(lm_head, "weight_scale", None)
    gs = getattr(lm_head, "weight_global_scale", None)
    if packed is None or bs is None or gs is None or packed.dtype != torch.uint8:
        return None
    inp = packed.shape[1] * 2
    # the K loop reads full 64-wide blocks (BK=64), so the input dim must divide
    # 64 (the per-16 block scale divides 16); anything else would read OOB
    if inp % 64 or (
        getattr(lm_head, "input_size_per_partition", None) is not None and inp != lm_head.input_size_per_partition
    ):
        return None  # not the [out, in/2] packed NVFP4 layout
    # A1: validate the torch reference against vLLM's own dequant (source of
    # truth) on sample rows. One full dequant, freed right after.
    sample = torch.randint(0, packed.shape[0], (min(8, packed.shape[0]),), device=packed.device)
    wfull = _dequant_weight(lm_head, torch.bfloat16)
    try:
        if not torch.allclose(
            _nvfp4_dequant_rows(packed, bs, gs, sample).float(),
            wfull.index_select(0, sample).float(),
            rtol=2e-2,
            atol=2e-2,
        ):
            return None
    finally:
        del wfull
    # A2: self-check the in-kernel GEMM vs the reference at serving-shaped M —
    # the BM=64/128 switch, a tiling boundary, and the multi-tile loop — not just M=1.
    w_k = packed.index_select(0, idx)
    bs_k = bs.index_select(0, idx)
    deq = _nvfp4_dequant_rows(packed, bs, gs, idx)
    for m in (1, 64, 65, 130):
        x = torch.randn(m, inp, dtype=torch.bfloat16, device=packed.device)
        ref = (x.float() @ deq.float().t()).to(x.dtype)
        got = _nvfp4_native_gemm(x, w_k, bs_k, gs)
        if got is None or not torch.allclose(got, ref, rtol=5e-2, atol=5e-2):
            return None

    def head(h):
        return _nvfp4_native_gemm(h.reshape(-1, h.shape[-1]), w_k, bs_k, gs).reshape(*h.shape[:-1], w_k.shape[0])

    return head


def _fp8_native_slice(lm_head, idx):
    """Native row-slice of an already-fp8 head: keep the kept rows' e4m3 bytes +
    their scale (no dequant->bf16->re-quant round trip). Returns
    ``(w8[kept,in], scale[kept])`` for ``fp8_linear``, or None when the scale is
    not row-aligned (a block-scale MXFP8 / unknown layout) so the caller dequants.
    """
    w = getattr(lm_head, "weight", None)
    if w is None or w.dtype != torch.float8_e4m3fn or w.ndim != 2:
        return None
    scale = getattr(lm_head, "weight_scale", None)
    if scale is None:
        return None
    kept = w.index_select(0, idx)
    # per-row (N,) scale: row-aligned with the weights -> slice it too
    if scale.ndim == 1 and scale.shape[0] == w.shape[0]:
        return kept.contiguous(), scale.index_select(0, idx).float().contiguous()
    # per-tensor scalar: broadcast to a per-row scale (still the native bytes)
    if scale.numel() == 1:
        return kept.contiguous(), scale.float().expand(kept.shape[0]).contiguous()
    return None  # block-scale / 2-D: not row-aligned; dequant is the safe path


def _kept_fp8_head(lm_head, qm, idx):
    """The bf16/fp8 kept head: a fresh full dequant, slice the kept rows,
    re-quantize to fp8 when the source was quantized. Shared by the fp8 path
    and the native path's serving-time fallback (built only on first failure)."""
    quantized = qm is not None and "Unquantized" not in type(qm).__name__
    w = _dequant_weight(lm_head, torch.bfloat16) if quantized else lm_head.weight.data
    fp8 = quantized or getattr(lm_head, "_mjolnir_fp8", False) or os.environ.get(ENV_LM_HEAD, "").strip() == "fp8"
    sub = w.index_select(0, idx).contiguous()
    del w
    if fp8:
        sub8, sub_scale = quantize_rows(sub)
        return lambda h: fp8_linear(h, sub8, sub_scale)
    return lambda h: torch.nn.functional.linear(h, sub)


def install_draft_vocab(draft_model, ids: list[int], size: int | None = None) -> bool:
    lm_head = getattr(draft_model, "lm_head", None)
    lp = getattr(draft_model, "logits_processor", None)
    if lm_head is None or lp is None or lm_head.weight.numel() == 0:
        _log.warning("Mjolnir draft vocab: the drafter has no usable lm_head; left as is")
        return False
    if getattr(lp, "scale", 1.0) != 1.0 or getattr(lp, "soft_cap", None) is not None:
        _log.warning("Mjolnir draft vocab: logit scale / soft cap not supported; left as is")
        return False
    qm = getattr(lm_head, "quant_method", None)
    quantized = qm is not None and "Unquantized" not in type(qm).__name__
    device = lm_head.weight.device
    vocab = int(getattr(lp, "org_vocab_size", 0) or 0)
    # a quantized head's row count (the vocab) needs a dequant when org_vocab_size
    # is unset; recover it then, reusing the dequant the fp8 path needs. A
    # successful native install never reaches here (B1: no forced dequant).
    w = None
    if not vocab:
        if quantized:
            try:
                w = _dequant_weight(lm_head, torch.bfloat16)
            except Exception as e:  # noqa: BLE001 - an experiment never breaks serving
                _log.warning(
                    "Mjolnir draft vocab: cannot dequantize the %s head (%r); left as is", type(qm).__name__, e
                )
                return False
        else:
            w = lm_head.weight.data
        vocab = int(w.shape[0])
    calibrated = len({i for i in ids if 0 <= i < vocab})
    ids = pad_vocab(ids, size, vocab)
    if size and calibrated < size:
        _log.warning(
            "Mjolnir draft vocab: the file has %d tokens, padded to %d with the lowest unused ids "
            "(re-run `mjolnir calibrate --draft-vocab` with more prompts for a measured ranking)",
            calibrated,
            len(ids),
        )
    idx = torch.tensor(sorted(set(ids)), dtype=torch.long, device=device)
    # ── build the kept head, staying in the source's native quant ────────────
    # VT never up-scales to bf16 first: the kept rows come straight from the
    # head's own storage. Dispatch on the original precision. The single
    # precision-changing exception is bf16 -> fp8, and only when the fp8 env is
    # set (a 4-bit/8-bit head is never widened):
    #   NVFP4 (packed 4-bit + block/global scales) -> 4-bit in-kernel GEMM,
    #       self-checked; automatic while the VT env is set (0/off -> fp8 fallback)
    #   FP8 (e4m3 weight + row-aligned scale)       -> native row-slice of the 8-bit
    #       bytes + scales (the fp8 GEMM is unchanged)
    #   bf16 (unquantized)                          -> fp8 when the fp8 env is set, else bf16
    #   any other quant (MXFP8 block scales, ...)   -> safe dequant->fp8 fallback
    native_off = os.environ.get(ENV_DRAFT_VOCAB_NATIVE, "").strip().lower() in ("0", "off", "false", "no")
    packed = getattr(lm_head, "weight", None)
    head = None
    label = None
    if not native_off and packed is not None and packed.dtype == torch.uint8:  # NVFP4 packed 4-bit
        try:
            native = _nvfp4_native_head(lm_head, idx)
        except Exception as e:  # noqa: BLE001 - an experiment never breaks serving
            _log.warning("Mjolnir draft vocab: native 4-bit self-check failed (%r); fp8 fallback", e)
            native = None
        if native is not None:
            if w is not None:
                del w  # the vocab-recovery dequant is unused by a native path
            fb = {}

            def head(h):
                try:
                    return native(h)
                except Exception as e:  # noqa: BLE001 - a serving failure switches to fp8 for good
                    if not fb:
                        _log.warning("Mjolnir draft vocab: native 4-bit serving failure (%r); fp8 kept head", e)
                        fb["h"] = _kept_fp8_head(lm_head, qm, idx)
                    return fb["h"](h)

            label = "4-bit (nvfp4, native)"
    if head is None and packed is not None and packed.dtype == torch.float8_e4m3fn:  # FP8 / MXFP8
        sliced = _fp8_native_slice(lm_head, idx)
        if sliced is not None:
            w8, sc = sliced

            def head(h):
                return fp8_linear(h, w8, sc)

            label = "fp8 (native)"
    if head is None:
        # fallback: dequant a quant source to bf16, then fp8 or bf16. A bf16 source
        # is the one precision exception (-> fp8 only when the fp8 env is set); a
        # quant source we could not slice natively dequants and re-quants to fp8.
        if w is None:
            if quantized:
                try:
                    w = _dequant_weight(lm_head, torch.bfloat16)
                except Exception as e:  # noqa: BLE001 - an experiment never breaks serving
                    _log.warning(
                        "Mjolnir draft vocab: cannot dequantize the %s head (%r); left as is", type(qm).__name__, e
                    )
                    return False
            else:
                w = lm_head.weight.data
        if w.ndim != 2 or w.shape[0] < vocab:
            _log.warning(
                "Mjolnir draft vocab: head weight %s has no token-per-row axis (rows %d < vocab %d); left as is",
                tuple(w.shape),
                w.shape[0],
                vocab,
            )
            return False
        sub = w.index_select(0, idx).contiguous()
        del w  # the full dequant / bf16 weight is dead weight now; sub holds the kept rows
        if quantized or getattr(lm_head, "_mjolnir_fp8", False) or os.environ.get(ENV_LM_HEAD, "").strip() == "fp8":
            sub8, sub_scale = quantize_rows(sub)

            def head(h):
                return fp8_linear(h, sub8, sub_scale)

            label = "fp8 (dequant)" if quantized else "fp8"
        else:

            def head(h):
                return torch.nn.functional.linear(h, sub)

            label = "bf16"

    # row chunks bound the temporaries when a caller (vLLM's startup memory
    # profile) passes thousands of rows at once
    def compute_logits(hidden_states, *args, **kwargs):
        h = hidden_states.reshape(-1, hidden_states.shape[-1])
        out = torch.full((h.shape[0], vocab), float("-inf"), dtype=h.dtype, device=h.device)
        for r0 in range(0, h.shape[0], ROW_CHUNK):
            out[r0 : r0 + ROW_CHUNK].index_copy_(1, idx, head(h[r0 : r0 + ROW_CHUNK]).to(out.dtype))
        return out.reshape(*hidden_states.shape[:-1], vocab)

    def get_top_tokens(hidden_states, *args, **kwargs):
        h = hidden_states.reshape(-1, hidden_states.shape[-1])
        top = torch.cat([head(h[r0 : r0 + ROW_CHUNK]).argmax(dim=-1) for r0 in range(0, max(1, h.shape[0]), ROW_CHUNK)])
        return idx[top].reshape(hidden_states.shape[:-1])

    draft_model.compute_logits = compute_logits
    if hasattr(draft_model, "get_top_tokens"):
        draft_model.get_top_tokens = get_top_tokens
    draft_model._mjolnir_draft_vocab = int(idx.numel())
    _log.info("Mjolnir draft vocab: %d of %d tokens, %s head", idx.numel(), vocab, label)
    return True


# ── the hook ────────────────────────────────────────────────────────────────


def _lm_heads(model) -> list:
    seen, out = set(), []
    for name, mod in model.named_modules():
        if name.split(".")[-1] == "lm_head" and hasattr(mod, "quant_method") and id(mod) not in seen:
            seen.add(id(mod))
            out.append(mod)
    return out


def after_load(runner) -> None:
    """Apply the opted-in experiments to a loaded GPU model runner."""
    target = getattr(runner, "model", None)
    spec = getattr(runner, "speculator", None)
    draft = getattr(spec, "model", None) if spec is not None else None
    vocab_path = os.environ.get(ENV_DRAFT_VOCAB, "").strip()
    if vocab_path and draft is not None:  # before fp8 may free the bf16 weight
        try:
            size = int(os.environ.get(ENV_DRAFT_VOCAB_SIZE, "0") or 0) or None
            install_draft_vocab(draft, load_vocab(vocab_path), size)
        except Exception as e:  # noqa: BLE001 - an experiment never breaks serving
            _log.warning("Mjolnir draft vocab: not installed (%r)", e)
    elif vocab_path:
        _log.warning("Mjolnir draft vocab: %s set but this config has no drafter", ENV_DRAFT_VOCAB)
    if os.environ.get(ENV_LM_HEAD, "").strip().lower() == "fp8":
        for model in (target, draft):
            if model is None:
                continue
            for head in _lm_heads(model):
                try:
                    install_fp8_head(head, target)
                except Exception as e:  # noqa: BLE001
                    _log.warning("Mjolnir lm_head fp8: not installed (%r)", e)
    if torch.cuda.is_available():
        # hand the quantization temporaries back before vLLM profiles memory
        # for the KV cache (a cached-but-free block still counts as used)
        torch.cuda.empty_cache()


def install_hooks() -> None:
    """Wrap GPUModelRunner.load_model (idempotent; no-op unless opted in)."""
    if not wanted():
        return
    try:
        from vllm.v1.worker.gpu import model_runner as mr
    except Exception as e:  # noqa: BLE001
        _log.warning("Mjolnir lm_head hooks: vLLM's GPU model runner not importable (%r)", e)
        return
    cls = mr.GPUModelRunner
    if getattr(cls, "_mjolnir_lmhead_hooked", False):
        return
    orig = cls.load_model

    def load_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        after_load(self)
        return out

    cls.load_model = load_model
    cls._mjolnir_lmhead_hooked = True
