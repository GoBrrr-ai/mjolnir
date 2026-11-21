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
        # The two gates that consume the probe (a partial apply that lands
        # the probe without wiring it would silently keep PIECEWISE spec
        # decode on sm_110).
        (
            "v1/attention/backends/flashinfer.py",
            "and flashinfer_supports_uniform_multi_token_decode()",
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
        # The FULL branch of _run_fused_multi_step_decode must rebuild the
        # draft metadata before run_fullgraph so host-side per-cycle plan
        # state (Mjolnir's T5 split-KV work distribution) sees this cycle's
        # seq_lens. 0.34: the fused machinery lives in
        # spec_decode/autoregressive/speculator.py. Assert the build call
        # itself, not just a comment.
        (
            "v1/worker/gpu/spec_decode/autoregressive/speculator.py",
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
    # STAGGER FIX: "thor sm110 GDN prefill: FI prefill grant gated by a one-shot runtime probe": [
    # STAGGER FIX:         # The probe-gated grant clause in _resolve_gdn_prefill_backend — the
    # STAGGER FIX:         # only file the patch touches.
    # STAGGER FIX:         (
    # STAGGER FIX:             "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    # STAGGER FIX:             "# THOR_SM110_GDN_PREFILL_PROBE",
    # STAGGER FIX:         ),
    # STAGGER FIX:     ],
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
    "thor qwen4 dense decoder: mlp_only_layers hybrid rule (THOR_QWEN4_DENSE)": [
        # v0.32.0's Qwen4ExpDecoderLayer builds a sparse MoE block for every
        # layer and transformers rejects num_experts <= 0, so a dense Qwen4
        # checkpoint cannot load. The patch mirrors qwen3_next's own hybrid
        # rule (mlp_only_layers + (layer_idx+1) % decoder_sparse_step): a
        # dense checkpoint lists EVERY layer in mlp_only_layers and gets
        # Qwen3NextMLP. Inert for every existing qwen4_exp checkpoint (none
        # set mlp_only_layers) — and inert for all other model families.
        (
            "models/qwen4_exp/nvidia/model.py",
            "THOR_QWEN4_DENSE",
        ),
    ],
    "thor qwen4 qsa mjolnir: probe-gated Mjolnir decode impl (THOR_QWEN4_QSA_MJOLNIR)": [
        # MJOLNIR_ATTN_QSA swaps Qwen4ExpQSAAttention's impl for the kernel
        # package's scratch-gather decode (mjolnir.qsa_impl) — pure-decode
        # steps only, everything else falls back to the stock Triton path
        # inside the impl. INERT without the env var (the try/except keeps
        # stock even when the kernel package is absent).
        (
            "models/qwen4_exp/nvidia/qsa.py",
            "THOR_QWEN4_QSA_MJOLNIR",
        ),
    ],
    "thor qwen4 qsa fp6: the packed-e2m3 main cache (THOR_QWEN4_QSA_FP6)": [
        # kv-cache-dtype: fp6_e2m3 on the QSA pools — the dtype handler
        # fallback in the layer's torch-dtype resolution and the packed-cell
        # pool spec; the swap impl (mjolnir.qsa_impl) writes/reads it, stock
        # fallbacks expand to e4m3. Inert for every other dtype.
        (
            "models/qwen4_exp/nvidia/qsa.py",
            "THOR_QWEN4_QSA_FP6",
        ),
        (
            "models/qwen4_exp/nvidia/qsa.py",
            "handler.torch_dtype()",
        ),
    ],
    "thor sm110 CuTe DSL MoE: FlashInferCuteDSLExperts admits family 110 (THOR_SM110_CUTEDSL_MOE)": [
        # The sentinel rides on the changed line of the one-line hunk.
        (
            "model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py",
            "p.is_device_capability_family(110))  # THOR_SM110_CUTEDSL_MOE",
        ),
    ],
    "64739 safetensors shards listed in an index subdirectory (tensors/)": [
        (
            "model_executor/model_loader/weight_utils.py",
            "def get_safetensors_files_in_index(",
        ),
        # used by the loader's glob fallback, not just defined
        (
            "model_executor/model_loader/default_loader.py",
            "index_files = get_safetensors_files_in_index(hf_folder, index_file)",
        ),
    ],
    "thor qwen4 PLE disk: slot-pool cache in front of the disk table (THOR_QWEN4_PLE_DISK)": [
        (
            "models/qwen4_exp/nvidia/ngram_embedding.py",
            "# [ple-disk] THOR_QWEN4_PLE_DISK: with disk_offload and a slot-pool",
        ),
        # used, not just imported: the embedding class swap
        (
            "models/qwen4_exp/nvidia/ngram_embedding.py",
            "embedding_cls = Qwen4ExpPLEDiskEmbedding",
        ),
    ],
    "thor qwen4 draft head: reduced draft vocabulary + 8-bit drafter (THOR_QWEN4_DRAFT_HEAD)": [
        (
            "models/qwen4_exp/nvidia/mtp.py",
            "THOR_QWEN4_DRAFT_HEAD",
        ),
        # used: the drafter scores the reduced head and scatters it
        (
            "models/qwen4_exp/nvidia/mtp.py",
            "return logits.index_copy_(1, self.draft_vocab_ids, draft_logits)",
        ),
        (
            "models/qwen4_exp/nvidia/mtp.py",
            "maybe_quantize_mtp_dense(draft_quant_config)",
        ),
    ],
    "thor qwen4 screened head: exact top-k logits from an FP8 screen (THOR_QWEN4_SCREENED_HEAD)": [
        (
            "models/qwen4_exp/nvidia/model.py",
            "THOR_QWEN4_SCREENED_HEAD",
        ),
        # used: compute_logits takes the screen, load_weights builds it
        (
            "models/qwen4_exp/nvidia/model.py",
            "logits = head(hidden_states, head.top_k)",
        ),
        (
            "models/qwen4_exp/nvidia/model.py",
            "self.screened_head = ScreenedLMHead.build(",
        ),
    ],
    "thor qwen4 side FP8: online FP8 side layers (THOR_QWEN4_SIDE_FP8)": [
        (
            "models/qwen4_exp/nvidia/model.py",
            "THOR_QWEN4_SIDE_FP8",
        ),
        # used: composed before the decoder layers are built
        (
            "models/qwen4_exp/nvidia/model.py",
            "side_fp8 = maybe_quantize_side_layers(self.quant_config)",
        ),
    ],
    "thor qwen4 HC FP8: online 8-bit hyperconnection GEMMs (THOR_QWEN4_HC_FP8)": [
        (
            "models/qwen4_exp/nvidia/hyperconnection.py",
            "THOR_QWEN4_HC_FP8",
        ),
        # used: the down/up Linears take the online config, the fused BF16
        # down+SiLU (and 0.34's SM100 up+gate+mix) kernel is skipped for a
        # converted layer
        (
            "models/qwen4_exp/nvidia/hyperconnection.py",
            "quant_config=down_quant,  # [hc-fp8] stock: None",
        ),
        (
            "models/qwen4_exp/nvidia/hyperconnection.py",
            "down_quant is None  # [hc-fp8] the fused kernel needs a BF16 weight",
        ),
        (
            "models/qwen4_exp/nvidia/hyperconnection.py",
            "up_quant is None  # [hc-fp8] the fused kernel reads a BF16 weight",
        ),
        (
            "models/qwen4_exp/nvidia/model.py",
            "log_hc_fp8(self.model)",
        ),
    ],
    "thor draft FULL CG under a PIECEWISE target (THOR_DRAFT_FULL_CG)": [
        (
            "v1/worker/gpu/spec_decode/autoregressive/speculator.py",
            "THOR_DRAFT_FULL_CG",
        ),
        # used: the gate feeds the draft decode manager's mode
        (
            "v1/worker/gpu/spec_decode/autoregressive/speculator.py",
            "if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL or draft_full_cg",
        ),
    ],
    "thor MXFP8 CUTLASS tactic pin by M bucket (THOR_MXFP8_TACTIC)": [
        (
            "model_executor/kernels/linear/mxfp8/flashinfer.py",
            "THOR_MXFP8_TACTIC",
        ),
        # used: apply_weights routes through the pinned-tactic op
        (
            "model_executor/kernels/linear/mxfp8/flashinfer.py",
            "output = torch.ops.vllm.thor_mm_mxfp8_tactic(",
        ),
    ],
}


# fix id -> list of (relative path from the flashinfer package root, sentinel
# text). Same sentinel conventions as FIXES; a leading "!" asserts absence.
# The flashinfer-side hunk is verified by verify_flashinfer() (separate root:
# the installed flashinfer site-packages, not the vllm package).
FLASHINFER_FIXES: dict[str, list[tuple[str, str]]] = {
    "thor sm110 CuTe DSL MoE: the 0.34 wrappers admit family 11 (THOR_SM110_CUTEDSL_MOE)": [
        # The 0.34 port (fi-cutedsl-moe-sm110-0.34.patch): one sentinel per
        # changed gate line. The moe_utils majors include 11 upstream.
        (
            "fused_moe/cute_dsl/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py",
            "if major not in (10, 11):  # THOR_SM110_CUTEDSL_MOE",
        ),
        (
            "fused_moe/cute_dsl/blockscaled_contiguous_grouped_gemm_finalize_fusion.py",
            "if major not in (10, 11):  # THOR_SM110_CUTEDSL_MOE",
        ),
    ],
    # STAGGER FIX: "thor sm110 GDN prefill: FI non-CP dispatch allowlist widened to major 11": [
    # STAGGER FIX:         # The single widened condition in chunk_gated_delta_rule's non-CP
    # STAGGER FIX:         # dispatch — the comment sentinel rides on the changed line, so it
    # STAGGER FIX:         # is present only when the hunk applied (a partial apply is
    # STAGGER FIX:         # impossible for a one-line hunk).
    # STAGGER FIX:         (
    # STAGGER FIX:             "gdn_prefill.py",
    # STAGGER FIX:             "if _arch_major in (10, 11):  # THOR_SM110_GDN_ALLOWLIST",
    # STAGGER FIX:         ),
    # STAGGER FIX:     ],
    "fp6 prefill direct-read on Ampere (Mjolnir packed e2m3 KV, no scratchpad)": [
        # The fp6 element type + the vec_cast delegation to the e4m3
        # conversion (vec_dtypes.cuh) — the type everything else keys on.
        (
            "data/include/flashinfer/vec_dtypes.cuh",
            "struct alignas(1) mjolnir_fp6_e2m3_sv",
        ),
        # The fp6 slot loader: three 4-byte cp.asyncs (the slot's gmem source
        # is only 4B-aligned at a 12-byte stride; a cp-size=16 copy from it is
        # undefined on sm_87 -- wrong partial lengths, skipped zero-fill).
        (
            "data/include/flashinfer/cp_async.cuh",
            "pred_load_fp6_12b",
        ),
        # Rename guard: the old 16B-cp.async loader must be gone.
        (
            "data/include/flashinfer/cp_async.cuh",
            "!pred_load_128b_src",
        ),
        # The "used, not just defined" loader call site in the smem wrapper.
        (
            "data/include/flashinfer/permuted_smem.cuh",
            "Three 4-byte cp.asyncs are always aligned.",
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
        # The paged/ragged kernels' K-side branch (the expand used to exist
        # only in the single-request kernel — a partial apply that lands the
        # single-kernel call but not the batched ones would compile and
        # silently read packed bytes as fp8).
        (
            "data/include/flashinfer/attention/prefill.cuh",
            "stock 1-byte per-fragment path reads them.",
        ),
        # The paged/ragged kernels' V-side branch.
        (
            "data/include/flashinfer/attention/prefill.cuh",
            "expand the packed V tile in place before compute_sfm_v reads it.",
        ),
        # use_kv_repack must exclude fp6: the repack path would otherwise
        # hijack CTA_TILE_Q>16 fp6 tiles and interpret packed bytes as fp8.
        (
            "data/include/flashinfer/attention/prefill.cuh",
            "at any CTA_TILE_Q -- never repack.",
        ),
        # The gmem byte-density fix: a slot holds 16 codes in 12 bytes, so
        # the per-thread column offset scales by 3/4 (all three sites).
        (
            "data/include/flashinfer/attention/prefill.cuh",
            "3/4 of the fp8",
        ),
        # The scale params threading through the JIT signature.
        (
            "jit/attention/modules.py",
            '"maybe_k_fp6_scales"',
        ),
        # The dtype-string pass-through in canonicalize_torch_dtype (no
        # torch dtype exists for 6-bit float; the string keys the map).
        (
            "utils.py",
            'if dtype == "mjolnir_fp6_e2m3":',
        ),
        # The NVFP4-heuristic exemption in the paged prefill run: the
        # packed-fp6 cache is uint8 too, without scale factors.
        (
            "prefill.py",
            'and self._cached_kv_data_type.startswith("mjolnir_fp6")',
        ),
        # The output-width heuristic keys the kernel's head_dim_vo: fp6
        # packs 8 values per 6 bytes (4/3 of the packed width).
        (
            "prefill.py",
            "# MJOLNIR_FP6_KV: 8 fp6 values per 6 packed bytes.",
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
