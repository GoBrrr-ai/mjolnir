#!/usr/bin/env python3
"""Verify the vLLM PR fixes are present in the installed vllm package (and
the FlashInfer-side sm_110 hunk in the installed flashinfer package).

Run standalone (resolves the installed vllm and flashinfer themselves) or
with explicit package roots (``python3 verify_patches.py
/path/to/site-packages/vllm /path/to/site-packages/flashinfer``), or
imported as ``verify(root)`` / ``verify_flashinfer(root)`` from
``apply_patches.py``. Exits non-zero if any fix is missing, so the Dockerfile
build fails fast when a base nightly no longer matches the assumptions the
hand-adapted patches were written against.
An explicit root argument is always honored: the verdict is about that tree,
never silently about the installed package.

Each fix is detected by a sentinel that is present only after the fix is in
place. The sentinels are intentionally chosen to be stable across vllm
releases (a function/attribute name that the fix introduces) so the check
keeps working as the surrounding code churns.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# fix id -> list of (relative path from the vllm package root, sentinel text).
# A leading "!" on a sentinel asserts the text is ABSENT (rename guards).
FIXES: dict[str, list[tuple[str, str]]] = {
    "50885 flashinfer native FULL decode cudagraphs for spec-decode": [
        (
            "v1/attention/backends/flashinfer.py",
            "def flashinfer_supports_uniform_multi_token_decode",
        ),
    ],
    "49652 draft-decode capture under dynamic SD (base manager guard)": [
        # Rebased onto the Dynamic-SD candidate expansion that upstream added
        # to the BASE CudaGraphManager: draft-decode managers run at
        # decode_query_len == 1, and the expansion must be skipped for them
        # (it would produce non-positive decode_query_lens). The guard line is
        # absent in the pristine tree and present only after this patch.
        (
            "v1/worker/gpu/cudagraph_utils.py",
            "and self.decode_query_len > 1",
        ),
    ],
    "54165 hybrid mamba align cache hits under spec decode (KV connector)": [
        (
            "config/speculative.py",
            "def use_eagle_preserves_target_kv_cache",
        ),
    ],
    "55519 no draft-group warning when eagle block drop is off": [
        (
            "v1/core/kv_cache_utils.py",
            "if spec_config is None or not spec_config.use_eagle_block_drop():\n        return\n    if any(group.is_eagle_group for group in kv_cache_groups):",
        ),
        (
            "v1/core/sched/scheduler.py",
            "so the trailing block stays eligible for cross-request",
        ),
    ],
    "thor fused draft-decode: in-place FlashInfer native decode plan advance": [
        (
            "v1/attention/backends/flashinfer.py",
            "def _advance_fi_decode_plan_kernel",
        ),
        (
            "v1/attention/backends/flashinfer.py",
            "def update_draft_decode_metadata",
        ),
    ],
    "thor fused draft-decode: T5 work-plan refresh on the FULL-graph replay path": [
        # The FULL branch of _fused_multi_step_decode must rebuild the draft
        # metadata before run_fullgraph so host-side per-cycle plan state
        # (Mjolnir's T5 split-KV work distribution) sees this cycle's
        # seq_lens. Assert the build call itself, not just a comment.
        (
            "v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py",
            "if not skip_attn:\n                self._build_uniform_attn_metadata(\n                    num_reqs=num_reqs,\n                    batch_desc=batch_desc,\n                    num_query_per_req=1,\n                    seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,\n                    step=1,\n                )",
        ),
    ],
    "thor dspark non-causal draft: FULL cudagraph via the FI prefill wrapper": [
        # Sentinels assert the non-causal draft actually *routes* to the
        # cudagraph-mode prefill wrapper, not just that the symbols exist (the
        # 52244 lesson — a partial apply must fail the build, so a defined-but-
        # unused wrapper must not pass).
        #
        # The wrapper factory (definition)
        (
            "v1/attention/backends/flashinfer.py",
            "def _get_noncausal_prefill_wrapper_cudagraph",
        ),
        # The per-batch-size wrapper dict + qo_indptr buffer allocation in
        # __init__ (where the wrapper state lives)
        (
            "v1/attention/backends/flashinfer.py",
            "self._noncausal_prefill_wrappers_cudagraph: dict[",
        ),
        # The new non-causal UNIFORM_BATCH grant inside get_cudagraph_support.
        # Gated on a multi-token drafter being configured; this line is the
        # patch-introduced gate (the *first* gate condition is the bare
        # vllm_config.attention_config.use_non_causal, which pre-dates the
        # patch).
        (
            "v1/attention/backends/flashinfer.py",
            "and vllm_config.speculative_config is not None",
        ),
        # The routing CALL SITE in build() — the load-bearing "used, not just
        # defined" sentinel. The prefill pathway must actually dispatch the
        # non-causal batch to the cudagraph wrapper.
        (
            "v1/attention/backends/flashinfer.py",
            "self._get_noncausal_prefill_wrapper_cudagraph(num_prefills)",
        ),
        # The routing condition variable that selects that call site
        (
            "v1/attention/backends/flashinfer.py",
            "use_nc_cudagraph =",
        ),
    ],
    "thor sm110 GDN prefill: FI prefill grant gated by a one-shot runtime probe": [
        # The probe-gated grant clause in _resolve_gdn_prefill_backend — the
        # only file the patch touches.
        (
            "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
            "# THOR_SM110_GDN_PREFILL_PROBE",
        ),
    ],
    "thor sm110 FA4 FP8-KV: probe-gated kv-cache-dtype grant": [
        # The probe definition + its probe-gated OR-term in the FP8-KV
        # eligibility gate (the only file the patch's fa_utils.py hunks
        # touch). The marker occurs exactly once in that file: a partial
        # apply that lands the probe without the gate (or vice versa)
        # must fail the build, since either half alone is behaviorally
        # inert or unsafe.
        (
            "v1/attention/backends/fa_utils.py",
            "# THOR_SM110_FA4_PROBE",
        ),
    ],
    "thor sm110 FA4 hd256 FP8-KV: probe-gated hd256 kernel policy grant": [
        # Kernel-file sentinel (N-1): the descale-load helper is absent from
        # the pristine kernel and present post-patch, so a partial apply that
        # lands the fa_utils policy hunks but misses the kernel fold hunks
        # must fail the build (the probe sentinels alone cannot catch that).
        # Path is relative to the vllm package root (like every other
        # sentinel here), so no leading ``vllm/``.
        (
            "vllm_flash_attn/cute/sm100_hd256_2cta_fmha_forward.py",
            "_load_effective_descales",
        ),
        # The hd256 probe definition — present only after this patch.
        # The marker occurs exactly once in fa_utils.py (the comment
        # block above the probe definition).
        (
            "v1/attention/backends/fa_utils.py",
            "# THOR_SM110_FA4_HD256_FP8_PROBE",
        ),
        (
            "v1/attention/backends/fa_utils.py",
            "def _thor_fa4_hd256_fp8kv_probe",
        ),
        # The quantized-KV reject relaxation in
        # _fa4_hd256_fallback_reason — the "used, not just defined"
        # sentinel: the grant must be wired into the reject site, not
        # merely defined in the probe (a partial apply that lands the
        # probe without the relaxation would leave an explicit-FA4
        # block-128 FP8-KV hd256 config downgraded to FA2, so it must
        # fail the build).
        (
            "v1/attention/backends/fa_utils.py",
            "# THOR_SM110_FA4_HD256_FP8:",
        ),
        # The gate term's consumer site — a head_dim=256 request must
        # be backed by the hd256 probe (a partial apply that leaves the
        # family(110) term backed by the hd128 probe only would admit
        # an hd256 model without validating the hd256 kernel, so it
        # must fail the build).
        (
            "v1/attention/backends/fa_utils.py",
            "_thor_fa4_hd256_fp8kv_probe()\n            if uses_fa4_hd256_kernel",
        ),
    ],
    "thor fp6 e2m3 KV cache dtype (Mjolnir)": [
        # the dtype name, its storage dtype, and the quant mode the backend keys on
        ("config/cache.py", '"fp6_e2m3",'),
        ("utils/torch_utils.py", '"fp6_e2m3": torch.uint8'),
        ("v1/kv_cache_interface.py", "FP6_E2M3 = 12"),
        ("v1/kv_cache_interface.py", 'if kv_cache_dtype == "fp6_e2m3":\n        return KVQuantMode.FP6_E2M3'),
    ],
    "thor Mjolnir attention backend: AttentionBackendEnum.MJOLNIR": [
        # The enum member that makes `--attention-backend MJOLNIR` parse; its
        # value is the mjolnir-kernel backend class (installed by the
        # Dockerfile right after the patch stack).
        (
            "v1/attention/backends/registry.py",
            'MJOLNIR = "mjolnir.backend.MjolnirAttentionBackend"',
        ),
        # The FA4 fallback default: the helper the backend reads its default
        # fallback from, and the call that resolves FA4 for it on sm_110 (the
        # "used, not just defined" sentinel).
        (
            "v1/attention/backends/fa_utils.py",
            "def mjolnir_fallback_name() -> str:",
        ),
        (
            "v1/attention/backends/fa_utils.py",
            "elif vllm_config is not None and _mjolnir_wants_fa4(vllm_config):\n            fa_version = 4",
        ),
    ],
    "thor sm110 draft-CG gate: shape-gated draft cudagraph replay (c1 fix)": [
        # The pure decision helper (THOR_DRAFT_CG_GATE sentinel lives in
        # its docstring) — present only after this patch.
        (
            "v1/worker/gpu/spec_decode/dflash/speculator.py",
            "def _thor_draft_cg_use_cuda_graph",
        ),
        # The dispatch-site call — the "used, not just defined" sentinel
        # (a partial apply that lands the helper but never calls it would
        # leave the c1 regression in place, so it must fail the build).
        (
            "v1/worker/gpu/spec_decode/dflash/speculator.py",
            "need_eager = is_profile or _thor_draft_cg_gate_wants_eager(num_reqs)",
        ),
    ],
    "thor sm110 MXFP4 MoE: OAI Triton selector keeps MARLIN default on SM110": [
        # The widened GPT-OSS MXFP4 selector condition in
        # _get_priority_backends_for_gpt_oss: SM110 must not take the OAI
        # Triton path by default (matmul_ogs compile failure at engine
        # init, #61374). The marker rides on the changed line, so it is
        # present only after the patch.
        (
            "model_executor/layers/fused_moe/oracle/mxfp4.py",
            "current_platform.is_device_capability_family(110)  # THOR_SM110_MXFP4_GATE",
        ),
        # The _swizzle_mxfp4 constraints widening: whenever the (opt-in)
        # TRITON path runs on SM110 it needs is_persistent +
        # epilogue_subtile=1 (the SWAP_XW subtiled-epilogue codegen bug).
        (
            "model_executor/layers/quantization/utils/mxfp4_utils.py",
            "for family in (100, 110, 120)",
        ),
    ],
    "thor sm110 CUTLASS FP4 MoE: backend not auto-selected on SM110": [
        # The demoted _supports_current_device in cutlass_moe.py: sm_110
        # removed from the family allowlist (profile_run TMA-descriptor
        # fault on Thor). The marker comment is present only after the
        # patch; the collapsed single-line condition only exists in the
        # patched shape; the absence guard fails the build if the
        # family(110) term ever returns (re-adapt signal).
        (
            "model_executor/layers/fused_moe/experts/cutlass_moe.py",
            "# THOR_SM110_CUTLASS_FP4_GUARD",
        ),
        (
            "model_executor/layers/fused_moe/experts/cutlass_moe.py",
            "p.is_device_capability_family(100) or p.is_device_capability_family(120)",
        ),
        (
            "model_executor/layers/fused_moe/experts/cutlass_moe.py",
            "!is_device_capability_family(110)",
        ),
    ],
    "thor spec-decode split top-p: probability-space pivot for huge logits": [
        # The probability-space mask predicate in _topp_sb_mask_kernel —
        # the logit-space pivot (log(pivot)+logZ+M) rounds to M for huge
        # forced logits (thinking_token_budget writes 1e9; fp32 spacing
        # there is 64), masking the whole row so sampling emits token 0
        # and spec decode never terminates. The marker rides on the
        # changed line, present only after the patch.
        (
            "v1/sample/ops/topk_topp_triton.py",
            "keep = (prob > pivot) | (x == M)  # THOR_SPEC_SPLIT_TOPP",
        ),
    ],
}


# fix id -> list of (relative path from the flashinfer package root, sentinel
# text). Same sentinel conventions as FIXES; a leading "!" asserts absence.
# The flashinfer-side hunk is verified by verify_flashinfer() (separate root:
# the installed flashinfer site-packages, not the vllm package).
FLASHINFER_FIXES: dict[str, list[tuple[str, str]]] = {
    "thor sm110 GDN prefill: FI non-CP dispatch allowlist widened to major 11": [
        # The single widened condition in chunk_gated_delta_rule's non-CP
        # dispatch — the comment sentinel rides on the changed line, so it
        # is present only when the hunk applied (a partial apply is
        # impossible for a one-line hunk).
        (
            "gdn_prefill.py",
            "if _arch_major in (10, 11):  # THOR_SM110_GDN_ALLOWLIST",
        ),
    ],
    "fp6 prefill direct-read on Ampere (Mjolnir packed e2m3 KV, no scratchpad)": [
        # The fp6 element type + the vec_cast delegation to the e4m3
        # conversion (vec_dtypes.cuh) — the type everything else keys on.
        (
            "data/include/flashinfer/vec_dtypes.cuh",
            "struct alignas(1) mjolnir_fp6_e2m3_sv",
        ),
        # The partial-src cp.async loader (12-byte fp6 slots).
        (
            "data/include/flashinfer/cp_async.cuh",
            "pred_load_128b_src",
        ),
        # The "used, not just defined" sentinel: the expand call site wired
        # into the K compute chain (a partial apply that lands the helper
        # without the call would compile and silently no-op).
        (
            "data/include/flashinfer/attention/prefill.cuh",
            "expand_fp6_tile_to_fp8<KTraits, KTraits::HEAD_DIM_QK>",
        ),
        # The V-side call site.
        (
            "data/include/flashinfer/attention/prefill.cuh",
            "expand_fp6_tile_to_fp8<KTraits, KTraits::HEAD_DIM_VO>",
        ),
        # The scale params threading through the JIT signature.
        (
            "jit/attention/modules.py",
            '"maybe_k_fp6_scales"',
        ),
    ],
}


def package_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise SystemExit("could not locate the installed vllm package")
    return Path(spec.origin).parent


def flashinfer_package_root() -> Path:
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or spec.origin is None:
        raise SystemExit("could not locate the installed flashinfer package")
    return Path(spec.origin).parent


def _check_root(root: Path, fixes: dict[str, list[tuple[str, str]]]) -> list[str]:
    """Return the list of missing fixes under ``root`` (empty == all present)."""
    missing: list[str] = []
    for fix, sentinels in fixes.items():
        present = True
        for rel, sentinel in sentinels:
            path = root / rel
            text = path.read_text() if path.exists() else ""
            if sentinel.startswith("!"):
                # absence sentinel: the renamed-away name must be gone
                ok = sentinel[1:] not in text
            else:
                ok = sentinel in text
            if not ok:
                present = False
                break
        if not present:
            missing.append(fix)
    return missing


def check_root(root: Path) -> list[str]:
    return _check_root(root, FIXES)


def check_flashinfer_root(root: Path) -> list[str]:
    return _check_root(root, FLASHINFER_FIXES)


def verify(root: str | Path | None = None) -> list[str]:
    # Normalize at the contract boundary: a str root used to crash inside
    # check_root (``root / rel`` -> TypeError) instead of verifying.
    root = Path(root) if root is not None else package_root()
    return check_root(root)


def verify_flashinfer(root: str | Path | None = None) -> list[str]:
    root = Path(root) if root is not None else flashinfer_package_root()
    return check_flashinfer_root(root)


if __name__ == "__main__":
    # An explicit root argument checks that tree. (Previously the argument
    # was silently ignored and the *installed* vllm was always checked, so
    # ``verify_patches.py <pristine-tree>`` could exit 0 with "all fixes
    # present" while reporting on a different, already-patched tree.)
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else package_root()
    fi_root = Path(sys.argv[2]) if len(sys.argv) > 2 else flashinfer_package_root()
    problems = verify(root)
    fi_problems = verify_flashinfer(fi_root)
    if problems or fi_problems:
        print("PATCH VERIFICATION FAILED — missing fixes:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        for p in fi_problems:
            print(f"  - [flashinfer] {p}", file=sys.stderr)
        sys.exit(1)
    print(f"all {len(FIXES)} fixes present in {root}")
    print(f"all {len(FLASHINFER_FIXES)} flashinfer fixes present in {fi_root}")
