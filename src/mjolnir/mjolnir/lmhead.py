"""Opt-in lm_head experiments for decode (vLLM GPU model runner).

On Qwen3.8-27B NVFP4 the bf16 lm_head (248,320 x 5,120, 2.54 GB) is read by
the target's verify and by every MTP draft step — ~40% of a decode step's
bytes. Two experiments, both off unless their env var is set (a config opts
in through its ``<config>.env`` sidecar, see ``mjolnir_cli.dockerctl``):

* ``MJOLNIR_LM_HEAD=fp8`` — the lm_head weight quantized to e4m3 per row at
  load (scale = row amax / 448) and multiplied by a Triton fp8-weight GEMM
  (bf16 activations, fp32 accumulate): half the bytes for the verify and the
  drafts. ``MJOLNIR_LM_HEAD_DROP_BF16=1`` also frees the bf16 weight (an
  untied head only), returning 2.5 GB to the KV cache.
* ``MJOLNIR_DRAFT_VOCAB=<json>`` (+ ``MJOLNIR_DRAFT_VOCAB_SIZE``) — VocabTrim
  for the drafter: its logits are computed only over the most frequent
  target tokens (``mjolnir calibrate --draft-vocab``), the rest are -inf.
  Verification still uses the full vocabulary, so the output distribution is
  unchanged (rejection sampling stays exact); only acceptance can drop.

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
ENV_DROP_BF16 = "MJOLNIR_LM_HEAD_DROP_BF16"
ENV_DRAFT_VOCAB = "MJOLNIR_DRAFT_VOCAB"
ENV_DRAFT_VOCAB_SIZE = "MJOLNIR_DRAFT_VOCAB_SIZE"
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


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, K) float -> (N, K) e4m3 and (N,) fp32 row scales (amax / 448)."""
    w32 = w.float()
    scale = (w32.abs().amax(dim=1) / FP8_MAX).clamp_min(1e-12)
    q = (w32 / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.contiguous()


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
    ref = x.float() @ w.float().t()
    got = fp8_linear(x, w8, scale).float()
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
    if qm is None or "Unquantized" not in type(qm).__name__:
        _log.warning("Mjolnir lm_head fp8: %s is not an unquantized head; left as is", type(qm).__name__)
        return False
    w = lm_head.weight.data
    w8, scale = quantize_rows(w)
    err = _self_test(w, w8, scale)
    if not err < 0.05:
        _log.warning("Mjolnir lm_head fp8: self-test relative error %.4f; left in bf16", err)
        return False
    lm_head.quant_method = Fp8HeadMethod(qm, w8, scale)
    lm_head._mjolnir_fp8 = True
    freed = ""
    if os.environ.get(ENV_DROP_BF16, "").strip() in ("1", "true", "yes") and model is not None:
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


def install_draft_vocab(draft_model, ids: list[int], size: int | None = None) -> bool:
    lm_head = getattr(draft_model, "lm_head", None)
    lp = getattr(draft_model, "logits_processor", None)
    if lm_head is None or lp is None or lm_head.weight.numel() == 0:
        _log.warning("Mjolnir draft vocab: the drafter has no usable lm_head; left as is")
        return False
    if getattr(lp, "scale", 1.0) != 1.0 or getattr(lp, "soft_cap", None) is not None:
        _log.warning("Mjolnir draft vocab: logit scale / soft cap not supported; left as is")
        return False
    w = lm_head.weight.data
    vocab = int(getattr(lp, "org_vocab_size", w.shape[0]))
    calibrated = len({i for i in ids if 0 <= i < vocab})
    ids = pad_vocab(ids, size, vocab)
    if size and calibrated < size:
        _log.warning(
            "Mjolnir draft vocab: the file has %d tokens, padded to %d with the lowest unused ids "
            "(re-run `mjolnir calibrate --draft-vocab` with more prompts for a measured ranking)",
            calibrated,
            len(ids),
        )
    idx = torch.tensor(sorted(set(ids)), dtype=torch.long, device=w.device)
    fp8 = getattr(lm_head, "_mjolnir_fp8", False) or os.environ.get(ENV_LM_HEAD, "").strip() == "fp8"
    sub = w.index_select(0, idx).contiguous()
    if fp8:
        sub8, sub_scale = quantize_rows(sub)

        def head(h):
            return fp8_linear(h, sub8, sub_scale)
    else:

        def head(h):
            return torch.nn.functional.linear(h, sub)

    def compute_logits(hidden_states, *args, **kwargs):
        part = head(hidden_states)
        out = torch.full((*part.shape[:-1], vocab), float("-inf"), dtype=part.dtype, device=part.device)
        out.index_copy_(out.dim() - 1, idx, part)
        return out

    def get_top_tokens(hidden_states, *args, **kwargs):
        return idx[head(hidden_states).argmax(dim=-1)]

    draft_model.compute_logits = compute_logits
    if hasattr(draft_model, "get_top_tokens"):
        draft_model.get_top_tokens = get_top_tokens
    draft_model._mjolnir_draft_vocab = int(idx.numel())
    _log.info(
        "Mjolnir draft vocab: drafter logits over %d of %d tokens (%s head, %.0f MB)",
        idx.numel(),
        vocab,
        "fp8" if fp8 else "bf16",
        sub.numel() * (1 if fp8 else sub.element_size()) / 2**20,
    )
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
