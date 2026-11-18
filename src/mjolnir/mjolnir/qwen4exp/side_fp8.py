"""[side-fp8] Online FP8 for the Qwen4Exp target's BF16 dense side layers.

The switch defaults to off, which leaves the stock behaviour unchanged.

``QWEN4EXP_SIDE_FP8=1`` ([side-fp8])
    Online FP8 for the target model's dense side layers that a ModelOpt
    MIXED_PRECISION checkpoint leaves in BF16 (GDN in/out, QSA qkv/o and
    indexer, shared expert; side_fp8_target()). A no-op when the checkpoint
    already quantizes them (FP4-Local ships them as MXFP8). Changes target
    outputs slightly.
``QWEN4EXP_SIDE_FP8_SKIP`` ([side-fp8])
    Kinds to keep in BF16. vLLM 0.34's Qwen4ExpQSAAttention fuses the indexer
    projection into the qkv_proj GEMM only when both resolve to the same quant
    method type (nvidia/qsa.py), so skipping only
    ``self_attn.indexer.index_qk_proj`` keeps just the indexer BF16 and
    unfused (C1).

``QWEN4EXP_FP8_FORMAT`` (``mjolnir.qwen4exp.draft_head``) picks the 8-bit
format.
"""

import math
import os
import re

import torch

from vllm.config.quantization import QuantizationConfigArgs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)

from .draft_head import _FP8_METHODS, _enabled, _fp8_format

logger = init_logger("vllm." + __name__)  # under vLLM's handler, like vllm.mjolnir


# [side-fp8] Target dense side layers that the NVIDIA ModelOpt checkpoint leaves
# in BF16: the class the earlier RadixArk FP8 hybrid converted (GDN in_proj_qkv,
# in_proj_z, out_proj; QSA q/k/v/o; shared-expert gate/up/down), named as the
# runtime (fused) modules, plus the QSA indexer's index_qk_proj, which shares
# the qkv_proj GEMM whenever both get the same method (as on vLLM 0.33, where
# it was always fused). Not matched: the GDN in_proj_ba (N=96), router mlp.gate,
# shared_expert_gate, HC mixers, PLE kv_proj, experts (already NVFP4),
# embed_tokens, lm_head, and the drafter ("mtp.layers.N", see MTP_DENSE_TARGET).
SIDE_FP8_KINDS = (
    "linear_attn.in_proj_qkvz",
    "linear_attn.out_proj",
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "mlp.shared_expert.gate_up_proj",
    "mlp.shared_expert.down_proj",
    "self_attn.indexer.index_qk_proj",
)
QSA_QKV_KIND = "self_attn.qkv_proj"
QSA_INDEXER_KIND = "self_attn.indexer.index_qk_proj"


def side_fp8_target() -> str:
    """[side-fp8] The online-FP8 target regex: SIDE_FP8_KINDS minus
    QWEN4EXP_SIDE_FP8_SKIP, a comma-separated list of kinds to keep in BF16.
    "self_attn.qkv_proj" keeps the QSA projections with the indexer BF16 (one
    fused GEMM, as on 0.33); "self_attn.indexer.index_qk_proj" keeps only the
    indexer BF16, which vLLM then leaves unfused (C1)."""
    kinds = SIDE_FP8_KINDS
    skip = {k.strip() for k in os.environ.get("QWEN4EXP_SIDE_FP8_SKIP", "").split(",") if k.strip()}
    if skip - set(kinds):
        raise ValueError(
            f"QWEN4EXP_SIDE_FP8_SKIP has unknown kinds {sorted(skip - set(kinds))}; valid: {', '.join(kinds)}"
        )
    if QSA_QKV_KIND in skip:
        skip.add(QSA_INDEXER_KIND)  # the indexer shared the BF16 qkv GEMM on 0.33
    return r"re:(.*\.)?model\.layers\.\d+\.(" + "|".join(re.escape(k) for k in kinds if k not in skip) + r")$"


# The same class as checkpoint (or fused) names, outside the drafter. If the
# checkpoint already quantizes any of them, QWEN4EXP_SIDE_FP8 is a no-op:
# an online target on a quantized layer would requantize it.
_SIDE_CHECKPOINT_NAME = re.compile(
    r"(?:^|(?<!mtp)\.)layers\.\d+\."
    r"(linear_attn\.(in_proj_qkvz?|in_proj_z|out_proj)"
    r"|self_attn\.(qkv_proj|q_proj|k_proj|v_proj|o_proj|indexer\.index_qk_proj)"
    r"|mlp\.shared_expert\.(gate_up_proj|gate_proj|up_proj|down_proj))$"
)


def maybe_quantize_side_layers(quant_config: QuantizationConfig | None) -> bool:
    """[side-fp8] Compose stock online FP8 onto the target's checkpoint quant
    config for the BF16 dense side layers (QWEN4EXP_SIDE_FP8). Returns whether
    it did; call before the decoder layers are built."""
    if not _enabled("QWEN4EXP_SIDE_FP8"):
        return False
    name = None if quant_config is None else quant_config.get_name()
    if name != "modelopt_mixed":
        logger.info(
            "[side-fp8] QWEN4EXP_SIDE_FP8 is a no-op: it needs a ModelOpt MIXED_PRECISION checkpoint, this one is %s.",
            name,
        )
        return False
    quantized_layers = getattr(quant_config, "quantized_layers", None) or {}
    already = sorted(k for k in quantized_layers if _SIDE_CHECKPOINT_NAME.search(k))
    if already:
        algos = sorted({str(quantized_layers[k].get("quant_algo")) for k in already})
        logger.info(
            "[side-fp8] QWEN4EXP_SIDE_FP8 is a no-op: the checkpoint already "
            "quantizes %d side-layer matrices (%s, e.g. %s).",
            len(already),
            "/".join(algos),
            already[0],
        )
        return False
    if quant_config.online_quantization_config is not None:
        raise ValueError(
            "QWEN4EXP_SIDE_FP8 cannot be combined with --quantization-config: "
            "the target already has an online quantization config."
        )
    target = side_fp8_target()
    quant_config.online_quantization_config = OnlineQuantizationConfig(  # ty: ignore[invalid-assignment]
        QuantizationConfigArgs(targets={target: _fp8_format()})
    )
    logger.info(
        "[side-fp8] Target side layers: online %s (%s)",
        _fp8_format(),
        target,
    )
    return True


def _layer_kind(name: str) -> str:
    """The module name after its layer index, e.g. linear_attn.out_proj."""
    parts = name.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            return ".".join(parts[i + 1 :])
    return name


def _fp8_bytes(fmt: str, n: int, k: int) -> int:
    """Weight bytes read per GEMM: e4m3 values plus the format's scales."""
    if fmt == "fp8_per_block":
        return n * k + math.ceil(n / 128) * math.ceil(k / 128) * 4  # fp32 128x128
    return n * k + n * math.ceil(k / 32)  # mxfp8: one e8m0 per 32 inputs


def log_side_layers(model: torch.nn.Module) -> None:
    """[side-fp8] One line: the converted matrices per kind, the weight bytes
    one forward step reads before and after, and the GEMM kernel."""
    fmt = _fp8_format()
    method_cls = _FP8_METHODS[fmt]
    counts: dict[str, int] = {}
    kernels: set[str] = set()
    bf16 = fp8 = 0
    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if type(quant_method) is not method_cls:
            continue
        if "hyper_connection" in name:  # [hc-fp8] counted by log_hc_fp8
            continue
        n = module.output_size_per_partition
        k = module.input_size_per_partition
        kind = _layer_kind(name)
        counts[kind] = counts.get(kind, 0) + 1
        bf16 += 2 * n * k
        fp8 += _fp8_bytes(fmt, n, k)
        kernel = getattr(quant_method, "kernel", None) or getattr(quant_method, "fp8_linear", None)
        kernels.add(type(kernel).__name__)
    logger.info(
        "[side-fp8] Target side layers: %d matrices to online %s (%s): %.3f GB BF16 -> %.3f GB per step via %s",
        sum(counts.values()),
        fmt,
        ", ".join(f"{kind} {num}" for kind, num in sorted(counts.items())),
        bf16 / 1e9,
        fp8 / 1e9,
        "/".join(sorted(kernels)) or "-",
    )
