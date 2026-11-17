"""Opt-in: serve W4A16-labelled NVFP4 routed experts as W4A4.

``MJOLNIR_NVFP4_MOE_W4A4=1`` makes vLLM's ModelOpt MIXED_PRECISION loader treat
routed-expert layers declared ``quant_algo: W4A16_NVFP4`` as ``NVFP4`` (W4A4):
the activations are quantized to FP4 with the checkpoint's own per-expert
``input_scale`` tensors, so the NVFP4 MoE oracle can pick an activation-
quantized backend (on sm_110: ``FLASHINFER_CUTEDSL``, enabled by
``thor-sm110-cutedsl-moe-gate.patch`` + ``fi-cutedsl-moe-sm110.patch``)
instead of MARLIN (W4A16).

Only routed experts change. Linear layers (shared experts, attention, the
lm_head) keep their declared algorithm. The checkpoint must ship calibrated
expert ``input_scale`` tensors (ModelOpt exports them even for W4A16 experts,
e.g. ``nvidia/Qwen3.6-35B-A3B-NVFP4``): the input scales are created as NaN
and the load refuses if any is still NaN, so a checkpoint without them fails
at startup instead of serving with uninitialized scales.

Unset (the default), nothing is installed.
"""

from __future__ import annotations

import logging
import os

ENV_MOE_W4A4 = "MJOLNIR_NVFP4_MOE_W4A4"

_log = logging.getLogger("vllm.mjolnir")


def wanted() -> bool:
    return os.environ.get(ENV_MOE_W4A4, "").strip().lower() in ("1", "true", "yes")


def _require_input_scales(layer, prefix: str) -> None:
    import torch

    for name in ("w13_input_scale", "w2_input_scale"):
        scale = getattr(layer, name, None)
        if scale is None or bool(torch.isnan(scale.data).any()):
            raise RuntimeError(
                f"{ENV_MOE_W4A4}=1: routed experts at {prefix!r} have no calibrated "
                f"{name} in the checkpoint, so they cannot be served W4A4. "
                f"Unset {ENV_MOE_W4A4} for this model."
            )


def install_hooks() -> None:
    """Wrap ModelOptMixedPrecisionConfig.get_quant_method (idempotent; no-op unless opted in)."""
    if not wanted():
        value = os.environ.get(ENV_MOE_W4A4, "").strip()
        if value:
            _log.warning("Mjolnir: %s=%r is not one of 1/true/yes, W4A4 MoE hook not installed", ENV_MOE_W4A4, value)
        return
    try:
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.quantization import modelopt as mo
    except Exception as e:  # noqa: BLE001
        _log.warning("Mjolnir W4A4 MoE hook: vLLM's ModelOpt config not importable (%r)", e)
        return
    cls = mo.ModelOptMixedPrecisionConfig
    if getattr(cls, "_mjolnir_moe_w4a4_hooked", False):
        return
    orig = cls.get_quant_method

    def get_quant_method(self, layer, prefix):
        if (
            isinstance(layer, RoutedExperts)
            and not self.is_layer_excluded(prefix)
            and self._resolve_quant_algo(prefix) == "W4A16_NVFP4"
        ):
            method = mo.ModelOptNvFp4FusedMoE(quant_config=self.nvfp4_config, moe_config=layer.moe_config)
            create, process = method.create_weights, method.process_weights_after_loading

            def create_weights(layer, *args, **kwargs):
                create(layer, *args, **kwargs)
                layer.w13_input_scale.data.fill_(float("nan"))
                layer.w2_input_scale.data.fill_(float("nan"))

            def process_weights_after_loading(layer):
                _require_input_scales(layer, prefix)
                process(layer)

            method.create_weights = create_weights
            method.process_weights_after_loading = process_weights_after_loading
            return method
        return orig(self, layer, prefix)

    cls.get_quant_method = get_quant_method
    cls._mjolnir_moe_w4a4_hooked = True
    _log.info("Mjolnir: %s=1, W4A16 NVFP4 routed experts are served W4A4", ENV_MOE_W4A4)
