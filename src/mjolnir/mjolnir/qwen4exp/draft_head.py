"""[draft-head] Opt-in byte reductions for the Qwen4Exp MTP drafter.

All switches default to off, which leaves the stock behaviour unchanged.

``MJOLNIR_DRAFT_VOCAB=<draft_vocab.json>`` (+ ``MJOLNIR_DRAFT_VOCAB_SIZE``)
    The MTP draft step scores only these token ids (a reduced copy of the
    lm_head rows, built and loaded with the drafter). The target still
    verifies with the full vocabulary, so the output is unchanged and only the
    acceptance length can drop. Same file and switches as ``mjolnir.lmhead``'s
    VocabTrim (``mjolnir calibrate --draft-vocab``); for qwen4_exp the drafter
    carries its own head, so ``mjolnir.lmhead`` leaves it alone.
``QWEN4EXP_DRAFT_HEAD_FP8=1``
    Online 8-bit (QWEN4EXP_FP8_FORMAT) for that reduced draft head only; the
    target head stays BF16.
``QWEN4EXP_MTP_DENSE_FP8=1``
    Online 8-bit (QWEN4EXP_FP8_FORMAT, default blockwise FP8) for the BF16
    dense projections of the MTP drafter (fc_embedding, fc_hidden, QSA qkv/o,
    shared expert). Drafter only.
``QWEN4EXP_FP8_FORMAT`` picks the 8-bit format of all of the above.
"""

import os

import numpy as np
import torch

from vllm.config.quantization import QuantizationConfigArgs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger("vllm." + __name__)  # under vLLM's handler, like vllm.mjolnir

# The reduced draft vocabulary: the same file and switches as mjolnir.lmhead.
ENV_DRAFT_VOCAB = "MJOLNIR_DRAFT_VOCAB"
ENV_DRAFT_VOCAB_SIZE = "MJOLNIR_DRAFT_VOCAB_SIZE"

# The reduced head gets a multiple of this many rows: no ParallelLMHead vocab
# padding and aligned GEMM output rows.
DRAFT_VOCAB_ALIGN = 128

# Drafter layers that the checkpoint leaves in BF16 and that run through the
# generic linear path (the HC mixers, router and shared_expert_gate do not).
# The QSA indexer projection is listed so that vLLM 0.34 keeps it fused into the
# qkv_proj GEMM (it fuses only same-method projections), as on 0.33.
MTP_DENSE_TARGET = (
    r"re:(.*\.)?mtp\.(fc_embedding|fc_hidden"
    r"|layers\.\d+\.(self_attn\.(qkv_proj|o_proj|indexer\.index_qk_proj)"
    r"|mlp\.shared_expert\.(gate_up_proj|down_proj)))$"
)


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0") not in ("", "0")


# QWEN4EXP_FP8_FORMAT picks the 8-bit format for the draft head, the MTP dense
# projections and the target side layers: "fp8_per_block" (128x128 weight
# blocks, per-token 1x128 dynamic activations; the format the earlier
# Flash-Next recipe used) or "mxfp8".
_FP8_METHODS = {
    "fp8_per_block": Fp8PerBlockOnlineLinearMethod,
    "mxfp8": Mxfp8OnlineLinearMethod,
}


def _fp8_format() -> str:
    fmt = os.environ.get("QWEN4EXP_FP8_FORMAT", "fp8_per_block")
    if fmt not in _FP8_METHODS:
        raise ValueError(f"QWEN4EXP_FP8_FORMAT must be one of {sorted(_FP8_METHODS)}, got {fmt!r}")
    return fmt


def make_draft_head(num_embeddings: int, hidden_size: int, prefix: str) -> ParallelLMHead:
    """The reduced draft head: a ParallelLMHead, online-quantized to
    QWEN4EXP_FP8_FORMAT when QWEN4EXP_DRAFT_HEAD_FP8 is set. The online method
    loads the BF16 rows onto a meta weight and quantizes them as soon as they
    arrive, so no BF16 copy stays resident."""
    quant_method = _FP8_METHODS[_fp8_format()]() if _enabled("QWEN4EXP_DRAFT_HEAD_FP8") else None
    lm_head = ParallelLMHead(num_embeddings, hidden_size, prefix=prefix, quant_method=quant_method)
    if quant_method is not None:
        # vLLM's online loader times the quantization through the layer's
        # quant_config and asserts it is set (model_loader/utils.py,
        # online_quantization_timing_context); a bare quant_method leaves it None.
        lm_head.quant_config = OnlineQuantizationConfig(QuantizationConfigArgs(targets={prefix: _fp8_format()}))
        logger.info(
            "[draft-head] %s: online %s (%d x %d) via %s",
            prefix,
            _fp8_format(),
            num_embeddings,
            hidden_size,
            type(getattr(quant_method, "kernel", None) or getattr(quant_method, "fp8_linear", None)).__name__,
        )
    return lm_head


def pad_draft_vocab(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """Sorted unique ids, topped up with the lowest unused ids (the most frequent
    BPE merges) to a multiple of DRAFT_VOCAB_ALIGN."""
    ids = np.unique(ids.astype(np.int64))
    num_pad = -len(ids) % DRAFT_VOCAB_ALIGN
    if num_pad:
        unused = np.setdiff1d(np.arange(vocab_size), ids, assume_unique=True)
        ids = np.union1d(ids, unused[:num_pad])
    return ids


def load_draft_vocab(vocab_size: int) -> torch.Tensor | None:
    """The reduced draft vocabulary from MJOLNIR_DRAFT_VOCAB, or None: the
    ``mjolnir.draft_vocab/1`` file read by ``mjolnir.lmhead.load_vocab`` (the
    first MJOLNIR_DRAFT_VOCAB_SIZE ids when set), then pad_draft_vocab."""
    path = os.environ.get(ENV_DRAFT_VOCAB, "").strip()
    if not path:
        return None
    from mjolnir.lmhead import load_vocab

    size = int(os.environ.get(ENV_DRAFT_VOCAB_SIZE, "0") or 0) or None
    ids = pad_draft_vocab(np.asarray(load_vocab(path, size)), vocab_size)
    logger.info(
        "[draft-head] MTP draft vocabulary: %d of %d ids from %s",
        len(ids),
        vocab_size,
        path,
    )
    return torch.as_tensor(ids, device=torch.get_default_device())


def maybe_quantize_mtp_dense(quant_config: QuantizationConfig) -> None:
    """Compose stock online 8-bit (QWEN4EXP_FP8_FORMAT) onto the drafter's
    checkpoint quant config for the BF16 dense MTP projections
    (QWEN4EXP_MTP_DENSE_FP8)."""
    if not _enabled("QWEN4EXP_MTP_DENSE_FP8"):
        return
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={MTP_DENSE_TARGET: _fp8_format()})
    )
    logger.info(
        "[draft-head] MTP dense projections: online %s (%s)",
        _fp8_format(),
        MTP_DENSE_TARGET,
    )


def log_mtp_dense(quant_config: QuantizationConfig | None) -> None:
    """Log the drafter layers that maybe_quantize_mtp_dense matched."""
    online = getattr(quant_config, "online_quantization_config", None)
    if online is not None:
        logger.info(
            "[draft-head] MTP online %s layers: %s",
            _fp8_format(),
            "; ".join(online.quantized_layer_summaries),
        )
