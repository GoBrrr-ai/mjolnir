#!/usr/bin/env python3
"""Thor sm_110 gate-probe canary: T6 (GDN prefill), T9 (FA4 FP8-KV),
T10 (draft-CG gate decision, pure python — runs without a GPU),
T11 (FA4 hd256 FP8-KV — the dedicated head_dim=256 kernel gate),
T12 (CuTe DSL NVFP4 MoE — the device gate + one synthetic W4A4 layer
against a dequantized float reference).

A single manual GPU test for the Thor sm_110 gate probes (patches
`thor-gdn-prefill-sm110`, `thor-fa4-fp8kv-sm110`,
`thor-fa4-hd256-fp8-sm110`, `thor-sm110-cutedsl-moe-gate` +
`fi-cutedsl-moe-sm110`). It imports the installed (patched) vllm and
triggers each probe's cached path with a small synthetic input — it does
NOT spin up a server. T9 additionally asserts the passive-by-default
invariant: with no explicit FA4 attention-config, the FP8-KV gate stays
closed and auto backend selection still lands on FlashInfer on sm_110.

Each section reports one of:

  PASS      the probe found working kernel support, and the consuming
            gate / kernel selection agrees with the verdict;
  INERT-OK  the probe correctly resolved to "disabled" and the legacy
            path is intact. This is the *expected* state on Thor today
            for the FA4 probe (passive by default — inert until the
            config selects FA4; the GDN probe now PASSes there) — a
            disabled probe is not a failure;
  SKIP      no CUDA device visible (self-skip, exit 0);
  FAIL      an internal inconsistency (verdict disagrees with its gate or
            selection) or an unexpected exception — the only case that
            makes the script exit non-zero.

Run it on a Thor (or any CUDA box) against the patched image:

  docker run --rm --gpus all --entrypoint python3 \
    -v <repo>/docker/vllm-thor/kernel_test_sm110_gates.py:\
/tmp/kernel_test_sm110_gates.py:ro \
    mjolnir/vllm-thor:qwen38-sm110-v6 /tmp/kernel_test_sm110_gates.py

T12 JIT-compiles FlashInfer's `moe_utils` on first use (~9 min on a Thor).
To reuse the serving build, add the serving container's FlashInfer cache
mount and env (see the T12 docstring):

    -e FLASHINFER_WORKSPACE_BASE=/data/flashinfer-cache \
    -e FLASHINFER_NVCC_THREADS=2 \
    -v <data-dir>/flashinfer-cache:/data/flashinfer-cache
"""
from __future__ import annotations

import logging
import sys
import time
import types
import traceback

import torch


def _t6_gdn_prefill() -> None:
    """T6 — GDN prefill probe verdict + correctness A/B + resolved backend."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        _resolve_gdn_prefill_backend,
        _thor_gdn_prefill_probe,
        _thor_gdn_prefill_probe_run,
    )
    from vllm.platforms import current_platform

    # The Qwen3.8-27B GDN geometry: head_k_dim 128 (the arch clause value).
    cfg = types.SimpleNamespace(
        additional_config={},
        model_config=types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(linear_key_head_dim=128)),
    )

    # First call performs the one-shot FI-vs-FLA A/B; later calls are a
    # cached bool read.
    verdict = _thor_gdn_prefill_probe()

    # Correctness A/B: when the probe passed, re-run the raw A/B comparison
    # (deterministic, seeded) — a second run must agree.
    if verdict:
        assert _thor_gdn_prefill_probe_run(), (
            "GDN prefill A/B re-run disagrees with the probe verdict"
        )
        print("T6 A/B re-check: FI vs FLA output+state still match")

    resolved = _resolve_gdn_prefill_backend(cfg)
    arch_eligible = (
        current_platform.is_device_capability_family(110)
        and current_platform.get_cuda_runtime_major() >= 13
    )
    if arch_eligible:
        # On sm_110 the grant is exactly "probe passed" (head_k_dim 128 is
        # set in cfg above).
        expected = "flashinfer" if verdict else "triton"
        assert resolved == ("auto", expected), (
            f"GDN prefill resolution {resolved} != expected "
            f"('auto', {expected!r}) for probe verdict {verdict}"
        )
    if verdict:
        print(f"PASS T6 GDN prefill: FI kernel verified on this device; "
              f"resolved {resolved}")
    else:
        print(f"INERT-OK T6 GDN prefill: probe disabled (expected on "
              f"Thor today); resolved {resolved} (Triton path intact)")


def _t9_fa4_fp8kv() -> None:
    """T9 — FA4 FP8-KV probe verdict + gate consistency + passive-by-default guard."""
    import vllm.v1.attention.backends.fa_utils as FA
    from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import get_attn_backend

    family110 = current_platform.is_device_capability_family(110)

    # PASSIVE-BY-DEFAULT GUARD (runs first, before anything may have
    # triggered the probe): under the DEFAULT config there is no
    # flash_attn_version override, so on sm_110 the FA version resolves to
    # 2 and the probe-gated OR-term of the gate short-circuits on
    # `fa_version == 4`. The probe must never run, the FP8-KV gate must
    # stay closed, and auto backend selection must still land on
    # FlashInfer — the patch alone changes zero behavior.
    with set_current_vllm_config(VllmConfig()):
        assert FA._THOR_FA4_FP8KV_PROBE_RESULT is None, (
            "the FA4 FP8-KV probe ran although no explicit FA4 "
            "attention-config is set — passive-by-default is broken"
        )
        assert not FA.flash_attn_supports_kv_cache_dtype("fp8_e4m3"), (
            "FP8-KV gate open under the DEFAULT config — passive-by-"
            "default is broken"
        )
        default_backend = get_attn_backend(
            head_size=128,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_e4m3",
            num_heads=8,
        )
        assert default_backend.__name__ == "FlashInferBackend", (
            f"auto backend selection with the default config chose "
            f"{default_backend.__name__}, expected FlashInferBackend — "
            f"passive-by-default regression"
        )
    print(f"T9 passive-by-default: default config leaves the probe "
          f"untriggered, FP8-KV gate closed, auto-selection -> "
          f"{default_backend.__name__}")

    # Probe verdict + gate consistency under the explicit FA4 config
    # (the required opt-in: backend=FLASH_ATTN + flash_attn_version=4).
    verdict = FA._thor_fa4_fp8kv_probe()
    with set_current_vllm_config(
        VllmConfig(
            attention_config=AttentionConfig(
                backend=AttentionBackendEnum.FLASH_ATTN,
                flash_attn_version=4,
            )
        )
    ):
        gate = FA.flash_attn_supports_kv_cache_dtype("fp8_e4m3")
    assert gate == verdict, (
        f"flash_attn_supports_kv_cache_dtype (explicit FA4 config) = "
        f"{gate} disagrees with the probe verdict {verdict}"
    )

    if family110:
        if verdict:
            print(f"PASS T9 FA4 FP8-KV: probe enabled on sm_110; gate "
                  f"open for the explicit FA4 config; default config "
                  f"stays passive ({default_backend.__name__})")
        else:
            print(f"INERT-OK T9 FA4 FP8-KV: probe disabled on sm_110 "
                  f"(reason in the 'THOR FA4 FP8-KV probe: disabled' log "
                  f"line); gate closed for FA4; default config stays "
                  f"passive ({default_backend.__name__})")
    else:
        assert verdict is False, (
            f"probe must be a no-op off sm_110, got verdict={verdict}"
        )
        assert not gate
        print(f"INERT-OK T9 FA4 FP8-KV: not an sm_110 device; probe is a "
              f"no-op (False); gate {gate}, default selection "
              f"{default_backend.__name__} (unchanged by the patch)")


def _t11_fa4_hd256_fp8kv() -> None:
    """T11 — FA4 hd256 FP8-KV probe verdict + gate consistency + passive guard."""
    import vllm.v1.attention.backends.fa_utils as FA
    from vllm.config import AttentionConfig, CacheConfig, VllmConfig, set_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import get_attn_backend

    family110 = current_platform.is_device_capability_family(110)

    # PASSIVE-BY-DEFAULT GUARD (runs first, before anything may have
    # triggered the hd256 probe): under the DEFAULT config there is no
    # flash_attn_version override, so on sm_110 the FA version resolves
    # to 2, the hd256 fallback-reject grant is never reached, and the
    # gate's OR-term short-circuits on `fa_version == 4`. The hd256
    # probe must never run, the FP8-KV gate must stay closed for a
    # head_size=256 + fp8-e4m3 + block-128 request, and auto backend
    # selection must still land on FlashInfer — the patch alone changes
    # zero default behavior.
    with set_current_vllm_config(VllmConfig()):
        assert FA._THOR_FA4_HD256_FP8KV_PROBE_RESULT is None, (
            "the FA4 hd256 FP8-KV probe ran although no explicit FA4 "
            "attention-config is set — passive-by-default is broken"
        )
        assert not FA.flash_attn_supports_kv_cache_dtype(
            "fp8_e4m3",
            head_size=256,
            head_size_v=256,
            kv_cache_block_size=128,
            supports_fa4_hd256=True,
        ), (
            "FP8-KV gate open for the hd256 model under the DEFAULT "
            "config — passive-by-default is broken"
        )
        default_backend = get_attn_backend(
            head_size=256,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_e4m3",
            num_heads=24,
        )
        assert default_backend.__name__ == "FlashInferBackend", (
            f"auto backend selection with the default config chose "
            f"{default_backend.__name__}, expected FlashInferBackend — "
            f"passive-by-default regression"
        )
    print(f"T11 passive-by-default: default config leaves the hd256 "
          f"probe untriggered, FP8-KV gate closed for the hd256 model, "
          f"auto-selection -> {default_backend.__name__}")

    # Probe verdict + gate consistency under the explicit FA4 config
    # (the required opt-in: backend=FLASH_ATTN +
    # flash_attn_version=4), an FP8 KV cache, and the hd256 kernel's
    # required block size (128).
    verdict = FA._thor_fa4_hd256_fp8kv_probe()
    fa4_config = VllmConfig(
        attention_config=AttentionConfig(
            backend=AttentionBackendEnum.FLASH_ATTN,
            flash_attn_version=4,
        ),
        cache_config=CacheConfig(cache_dtype="fp8_e4m3"),
    )
    with set_current_vllm_config(fa4_config):
        gate = FA.flash_attn_supports_kv_cache_dtype(
            "fp8_e4m3",
            head_size=256,
            head_size_v=256,
            kv_cache_block_size=128,
            supports_fa4_hd256=True,
        )
        # block_size=128 clears the `% 128` reject, so the only reject
        # that can still fire for this config is the quantized-KV one —
        # which the Thor grant lifts exactly when the probe passed.
        fallback = FA._fa4_hd256_fallback_reason(False, False, 128, fa4_config)
    assert gate == verdict, (
        f"flash_attn_supports_kv_cache_dtype (explicit FA4, hd256) = "
        f"{gate} disagrees with the hd256 probe verdict {verdict}"
    )
    expected_reason = (
        None if verdict else "quantized KV cache dtype fp8_e4m3"
    )
    assert fallback == expected_reason, (
        f"_fa4_hd256_fallback_reason (block_size=128, FP8 KV) = "
        f"{fallback!r} disagrees with the hd256 probe verdict "
        f"{verdict} (expected {expected_reason!r}) — the hd256 FP8 "
        f"path would {'be downgraded to FA2' if fallback else 'be admitted'}"
    )

    if family110:
        if verdict:
            print(f"PASS T11 FA4 hd256 FP8-KV: probe enabled on sm_110; "
                  f"gate open + no FA2 downgrade for the explicit FA4 "
                  f"config; default config stays passive "
                  f"({default_backend.__name__})")
        else:
            print(f"INERT-OK T11 FA4 hd256 FP8-KV: probe disabled on "
                  f"sm_110 (reason in the 'THOR FA4 hd256 FP8-KV probe: "
                  f"disabled' log line); gate closed; the quantized-KV "
                  f"reject is intact ({fallback!r}); default config "
                  f"stays passive ({default_backend.__name__})")
    else:
        assert verdict is False, (
            f"probe must be a no-op off sm_110, got verdict={verdict}"
        )
        assert not gate
        assert fallback == "quantized KV cache dtype fp8_e4m3", (
            f"the quantized-KV reject must stay stock off sm_110, got "
            f"{fallback!r}"
        )
        print(f"INERT-OK T11 FA4 hd256 FP8-KV: not an sm_110 device; "
              f"probe is a no-op (False); gate {gate}; default "
              f"selection {default_backend.__name__} (unchanged by the "
              f"patch)")


# T12 synthetic routed-expert layer: 8 experts, top-2, 48 tokens. H 2560 /
# I 640 is a routed-expert shape this kernel has run on a Thor; H % 256 == 0
# and 2I % 128 == 0, so vLLM's CuTe DSL weight prep pads nothing.
_T12_E, _T12_TOPK, _T12_T, _T12_H, _T12_I = 8, 2, 48, 2560, 640
# Bounds against the float reference. For this seed a torch emulation of the
# kernel's own math lands at rel L2 0.002 (BF16 output rounding); leaving out
# the FC2 activation quantization moves it to 0.078 / cosine 0.996, swapped
# gate/up to 1.0 / 0.36. The bounds sit between: room for the kernel's
# rounding differences, none for a W4A16 or mis-laid-out result.
_T12_REL_L2_MAX, _T12_COS_MIN = 0.05, 0.998


def _t12_dequant(codes: torch.Tensor, block_scale: torch.Tensor, scale_2: torch.Tensor) -> torch.Tensor:
    """NVFP4 weight rows (packed u8 codes, E4M3 block scales, global scale) -> fp32."""
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import break_fp4_bytes

    values = break_fp4_bytes(codes, torch.float32).reshape(codes.shape[0], -1, 16)
    return (values * block_scale.float().unsqueeze(-1)).reshape(codes.shape[0], -1) * scale_2


def _t12_qdq(t: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """NVFP4 quantize-dequantize of activations (global scale = 1 / input_scale)."""
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant

    gscale = (1.0 / input_scale).reshape(1).float()
    q, block_scale = ref_nvfp4_quant(t.float(), gscale, 16)
    return (q.reshape(t.shape[0], -1, 16) * (block_scale.unsqueeze(-1) / gscale)).reshape(t.shape)


def _t12_cutedsl_nvfp4_moe() -> None:
    """T12 — CuTe DSL NVFP4 MoE: the vLLM device gate + one synthetic W4A4 layer.

    Patches `thor-sm110-cutedsl-moe-gate` (vLLM) and `fi-cutedsl-moe-sm110`
    (FlashInfer). On sm_110 the gate must be open, and a small random NVFP4
    routed-expert layer — prepared by the functions the server runs for
    FLASHINFER_CUTEDSL and launched with the arguments of
    `FlashInferCuteDSLExperts.apply` — must match a torch fp32 reference of
    the same checkpoint tensors (dequantized weights, NVFP4
    quantize-dequantize of the FC1 and FC2 inputs).

    FlashInfer's `moe_utils` module (moe_sort / unpermute) JIT-compiles on
    first use: ~9 min on a Thor, unless the FlashInfer cache already holds
    it. The build is keyed by its ninja file, which carries the cache path
    and `--threads`: it is reused only with the serving container's
    FLASHINFER_WORKSPACE_BASE (same mount path) and FLASHINFER_NVCC_THREADS.
    The CuTe DSL kernels themselves compile in seconds.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import FlashInferCuteDSLExperts
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        convert_to_nvfp4_moe_kernel_format,
        make_nvfp4_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import activation_to_flashinfer_int
    from vllm.platforms import current_platform
    from vllm.utils.flashinfer import flashinfer_cute_dsl_fused_moe_nvfp4, has_flashinfer_cutedsl_moe_nvfp4

    gate = FlashInferCuteDSLExperts._supports_current_device()
    if not current_platform.is_device_capability_family(110):
        stock = current_platform.is_device_capability_family(100) and has_flashinfer_cutedsl_moe_nvfp4()
        assert gate == stock, f"the CuTe DSL MoE gate must stay stock off sm_110: got {gate}, stock {stock}"
        print(f"INERT-OK T12 CuTe DSL NVFP4 MoE: not an sm_110 device; gate {gate} (unchanged by the patch)")
        return
    assert gate, (
        "FlashInferCuteDSLExperts._supports_current_device() is False on sm_110 — "
        "thor-sm110-cutedsl-moe-gate.patch missing, or FlashInfer has no cute_dsl_fused_moe_nvfp4"
    )

    E, K, T, H, I = _T12_E, _T12_TOPK, _T12_T, _T12_H, _T12_I  # noqa: E741
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(12)

    # Checkpoint-format tensors, as ModelOptNvFp4FusedMoE holds them before
    # process_weights_after_loading (w13 rows = [gate; up]): random FP4 codes,
    # E4M3 block scales in 100..448, global scales of realistic magnitude.
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, generator=gen).to(dev)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=gen).to(dev)
    w13_sf = (torch.rand(E, 2 * I, H // 16, generator=gen) * 348 + 100).to(torch.float8_e4m3fn).to(dev)
    w2_sf = (torch.rand(E, H, I // 16, generator=gen) * 348 + 100).to(torch.float8_e4m3fn).to(dev)
    w13_s2 = (torch.rand(E, generator=gen) * 2.2e-4 + 2.5e-5).to(dev)
    w2_s2 = (torch.rand(E, generator=gen) * 1.1e-4 + 2.3e-5).to(dev)
    a13 = torch.full((E, 2), 0.006045, device=dev)  # FC1 input scale: amax = scale * 448 * 6
    x = (torch.randn(T, H, generator=gen) * (0.006045 * 448.0)).to(torch.bfloat16).to(dev)  # 6 sigma = amax
    ids = torch.rand(T, E, generator=gen).topk(K, dim=-1).indices.to(torch.int32).to(dev)
    wts = torch.softmax(torch.randn(T, K, generator=gen), dim=-1).to(dev)

    # Reference FC1: quantized input -> per expert dequantized W13 -> silu(gate) * up.
    xq_ref = _t12_qdq(x, a13.max())
    acts = {}
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = xq_ref[tok] @ _t12_dequant(w13[e], w13_sf[e], w13_s2[e]).t()
        acts[e] = (tok, slot, torch.nn.functional.silu(h[:, :I]) * h[:, I:])
    # FC2 input scale: calibrated on these activations, as a checkpoint's is
    # (25 % headroom keeps every block scale below the E4M3 maximum).
    act_amax = max(float(act.abs().max()) for _, _, act in acts.values())
    a2 = torch.full((E,), 1.25 * act_amax / (448.0 * 6.0), device=dev)
    # Reference FC2: quantized activation -> dequantized W2 -> router weight -> sum over top-k.
    ref = torch.zeros(T, H, dtype=torch.float32, device=dev)
    for e, (tok, slot, act) in acts.items():
        y = _t12_qdq(act, a2.max()) @ _t12_dequant(w2[e], w2_sf[e], w2_s2[e]).t()
        ref.index_add_(0, tok, y * wts[tok, slot].unsqueeze(-1))

    # The server's FLASHINFER_CUTEDSL preparation (modelopt.py,
    # ModelOptNvFp4FusedMoE.process_weights_after_loading): kernel weight
    # format, quant config, then the experts class's in-place alpha fold.
    layer = types.SimpleNamespace(
        activation=MoEActivation.SILU,
        moe_config=types.SimpleNamespace(
            moe_parallel_config=types.SimpleNamespace(enable_eplb=False),
            intermediate_size_per_partition=I,
            hidden_dim=H,
            hidden_dim_unpadded=None,
            num_experts=E,
        ),
    )
    backend = NvFp4MoeBackend.FLASHINFER_CUTEDSL
    w13_k, w13_sf_k, w13_s2_k, a13_k, w2_k, w2_sf_k, w2_s2_k, a2_k = convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=backend,
        layer=layer,
        w13=w13,
        w13_scale=w13_sf,
        w13_scale_2=w13_s2.clone(),
        a13_scale=a13,
        w2=w2,
        w2_scale=w2_sf,
        w2_scale_2=w2_s2.clone(),
        a2_scale=a2,
        is_act_and_mul=True,
    )
    assert (layer.moe_config.hidden_dim, layer.moe_config.intermediate_size_per_partition) == (H, I), (
        f"unexpected padding: {layer.moe_config}"
    )
    qc = make_nvfp4_moe_quant_config(
        backend=backend,
        w13_scale=w13_sf_k,
        w2_scale=w2_sf_k,
        w13_scale_2=w13_s2_k,
        w2_scale_2=w2_s2_k,
        a13_scale=a13_k,
        a2_scale=a2_k,
    )
    w13_s2_k.mul_(a13_k)  # FlashInferCuteDSLExperts.process_weights_after_loading
    w2_s2_k.mul_(a2_k)

    # The server's step: quantize the input (prepare/finalize), then the
    # kernel with the arguments of FlashInferCuteDSLExperts.apply (SILU).
    xq, x_sf = moe_kernel_quantize_input(
        x, qc.a1_gscale, quant_dtype=qc.quant_dtype, per_act_token_quant=False, is_scale_swizzled=qc.is_scale_swizzled
    )
    out = torch.full((T, H), float("nan"), dtype=torch.bfloat16, device=dev)
    print("T12 first kernel call (JIT-compiles moe_utils, ~9 min on a Thor, unless the FlashInfer cache holds it) ...")
    t0 = time.time()
    flashinfer_cute_dsl_fused_moe_nvfp4(
        x=xq,
        x_sf=x_sf.unsqueeze(-1),
        token_selected_experts=ids,
        token_final_scales=wts.float(),
        w1_weight=w13_k,
        w1_weight_sf=qc.w1_scale,
        w1_alpha=qc.g1_alphas,
        fc2_input_scale=qc.a2_gscale,
        w2_weight=w2_k,
        w2_weight_sf=qc.w2_scale,
        w2_alpha=qc.g2_alphas,
        num_experts=E,
        top_k=K,
        num_local_experts=E,
        local_expert_offset=0,
        moe_output=out,
        activation_type=activation_to_flashinfer_int(MoEActivation.SILU),
    )
    torch.cuda.synchronize()
    seconds = time.time() - t0

    y = out.float()
    nonfinite = int((~torch.isfinite(y)).sum())
    rel_l2 = float(torch.linalg.vector_norm(y - ref) / torch.linalg.vector_norm(ref))
    cos_min = float(torch.nn.functional.cosine_similarity(y, ref, dim=-1).min())
    numbers = (
        f"rel L2 {rel_l2:.4g} (max {_T12_REL_L2_MAX}), min token cosine {cos_min:.5f} "
        f"(min {_T12_COS_MIN}), {nonfinite} non-finite; first call {seconds:.1f} s"
    )
    assert nonfinite == 0 and rel_l2 <= _T12_REL_L2_MAX and cos_min >= _T12_COS_MIN, (
        f"CuTe DSL NVFP4 MoE output disagrees with the float reference: {numbers}"
    )
    print(
        f"PASS T12 CuTe DSL NVFP4 MoE: gate open on sm_110; {E} experts top-{K}, {T} tokens, "
        f"H {H} / I {I} vs the float reference: {numbers}"
    )


def _t10_draft_cg_gate() -> None:
    """T10 — the draft-CG gate's decision logic (pure python, no GPU).

    `_thor_draft_cg_use_cuda_graph` takes the capability as an argument,
    so it is importable and testable without CUDA — the canary asserts
    the four spec cases: sm_110 below/above the threshold, the off-arch
    no-op, and the min_batch=1 knob (= today's behavior).
    """
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        _thor_draft_cg_use_cuda_graph,
    )

    cases = [
        # (capability, num_reqs, min_batch, expected use_cg)
        ((11, 0), 1, 2, False),  # sm_110, N=1 < min_batch=2 -> eager
        ((11, 0), 2, 2, True),  # sm_110, N=2 >= min_batch=2 -> CG
        ((9, 0), 1, 2, True),  # off-sm_110 arch no-op -> CG
        ((11, 0), 1, 1, True),  # knob: min_batch=1 -> CG for all N
    ]
    for capability, num_reqs, min_batch, expected in cases:
        got = _thor_draft_cg_use_cuda_graph(num_reqs, capability, min_batch)
        assert got is expected, (
            f"_thor_draft_cg_use_cuda_graph(num_reqs={num_reqs}, "
            f"capability={capability}, min_batch={min_batch}) = {got}, "
            f"expected {expected}"
        )
    print("PASS T10 draft-CG gate: pure decision logic (4/4 cases)")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("vllm").setLevel(logging.INFO)

    failures = 0
    if not torch.cuda.is_available():
        for name in (
            "T6 GDN prefill",
            "T9 FA4 FP8-KV",
            "T11 FA4 hd256 FP8-KV",
            "T12 CuTe DSL NVFP4 MoE",
        ):
            print(f"SKIP {name}: no CUDA device visible (re-run with "
                  f"--gpus all on a Thor)")
    else:
        print(f"device capability: {torch.cuda.get_device_capability()}")
        for name, section in (
            ("T6 GDN prefill", _t6_gdn_prefill),
            ("T9 FA4 FP8-KV", _t9_fa4_fp8kv),
            ("T11 FA4 hd256 FP8-KV", _t11_fa4_hd256_fp8kv),
            ("T12 CuTe DSL NVFP4 MoE", _t12_cutedsl_nvfp4_moe),
        ):
            try:
                section()
            except Exception:
                failures += 1
                print(f"FAIL {name}:", file=sys.stderr)
                traceback.print_exc()

    # T10 is pure python (the helper takes the capability as an argument)
    # — it runs even without a GPU.
    try:
        _t10_draft_cg_gate()
    except Exception:
        failures += 1
        print("FAIL T10 draft-CG gate:", file=sys.stderr)
        traceback.print_exc()

    if failures:
        print(f"kernel test sm110 gates: {failures} section(s) FAILED")
        return 1
    print("kernel test sm110 gates: DONE (exit 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
