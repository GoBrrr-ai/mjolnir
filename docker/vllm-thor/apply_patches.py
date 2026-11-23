#!/usr/bin/env python3
"""Apply the hand-adapted vLLM PR patches to the installed vllm package
(and the FlashInfer-side sm_110 hunk to the installed flashinfer package).

Designed to be safe against nightly bumps:

* Locates the installed vllm package (``importlib``) and applies each patch
  from ``patches/`` with ``patch -p1`` (paths in the files are ``vllm/...``,
  so the package root is the working directory's child ``vllm/``).
* The flashinfer-side patches (``FLASHINFER_PATCH_ORDER``) patch the
  installed flashinfer site-packages the same way (paths are
  ``flashinfer/...``); its root is located the same way or given as an
  explicit second argument.
* Uses ``--forward`` (``-N``): if a patch's hunks are already present because
  the base nightly already merged that fix, it is skipped rather than failing.
* After applying, it runs ``verify_patches.verify`` +
  ``verify_patches.verify_flashinfer``. The build fails (exit 1) if any fix
  is missing — which is exactly the signal that the base nightly has moved
  and the affected patch needs re-adapting.

Run standalone (resolves the installed vllm and flashinfer themselves) or
with explicit package roots:
``python apply_patches.py /path/to/site-packages/vllm /path/to/site-packages/flashinfer``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

# Order: re-based onto upstream vLLM 0.34.0 (2026-11 bump, tag 62d4bc3c7).
# Retired at this bump (all four in the base now — see PATCHES.md): 49652
# (the draft-decode capture shapes are explicit in the 0.34 speculator
# managers), the two sm_110 MoE gates (#62133, #62275) and the qwen4 topk
# gate (#56102). The fa4 sm_110 gate patches touch files disjoint from every
# other patch (v1/attention/backends/fa_utils.py +
# vllm_flash_attn/cute/interface.py; the hd256 patch additionally touches
# vllm_flash_attn/cute/sm100_hd256_2cta_fmha_forward.py), so their position
# among the middle patches is immaterial — except that the two fa4 sm_110
# patches both touch v1/attention/backends/fa_utils.py, so the hd256 patch
# (whose fa_utils hunks are diffed against the hd128-patched file) must
# land after the hd128 one. The three Thor v1/attention/backends/flashinfer.py
# patches land in sequence: 50885 (the FI-native uniform-multi-token FULL
# grant — the probe + the two gates; the wrapper plumbing itself is
# upstream), the fused-draft-decode patch (its hunks
# sit on top of 50885's), then the DSpark non-causal cudagraph patch
# (its get_cudagraph_support hunk is diffed against 50885's grant).
PATCH_ORDER = [
    "50885-flashinfer-full-cudagraph.patch",
    "thor-fused-draft-decode-fi-native-update.patch",
    # Refreshes the T5 split-KV work plan when the fused draft loop replays
    # its FULL cudagraph: the _run_fused_multi_step_decode FULL branch returns
    # before building the draft metadata, so a host-side per-cycle plan
    # (ours) would stay baked at capture-time seq_lens. Same feature
    # family as the patch above; the speculator file is disjoint from it.
    # 0.34: the fused machinery lives in spec_decode/autoregressive/
    # speculator.py (moved out of target_dependent_ar).
    "thor-fused-draft-t5-plan-refresh.patch",
    "thor-dspark-draft-fi-noncausal-cudagraph.patch",
    # Disjoint from every patch except the hd256 one below (fa_utils.py) —
    # must land before it.
    "thor-fa4-fp8kv-sm110.patch",
    # Retired at the 0.35.0rc0.dev nightly bump (2026-11-22), both
    # base-covered:
    #  - thor-fa4-hd256-fp8-sm110: the hd256 fwd grew a native
    #    DescaleTensors path (flash_fwd_sm100.py) and lost the
    #    _fa4_hd256_fallback_reason reject site entirely, so the hand
    #    descale port and the fa_utils grant have nothing to anchor on;
    #    the head-size-blind FP8 grant above already covers hd256.
    #  - thor-fa4-hd256-1cta-decode-sm110: the hd256_use_2cta block is
    #    gone from interface.py; the heuristics moved to config.py
    #    (default_pack_gqa prefer_hd256_2cta) and already gate on
    #    device_capacity in (10, 11). Re-tune on-device at the 0.35 bump
    #    if the decode-shaped 1CTA win reproduces there.
    # 11th (c1 fix): dispatch-side shape gate for the DSpark draft's FULL
    # cudagraph replay on sm_110. It edits v1/worker/gpu/spec_decode/
    # dflash/speculator.py, which no other patch touches, but its context
    # is the code the dspark patch's ecosystem introduced (the FULL CG
    # dispatch), so it lands after thor-dspark.
    "thor-draft-cg-gate-sm110.patch",
    # Inkling 2 Mini on sm_110: the nightly wheel's prebuilt inkling2
    # kernels carry no sm_110 cubins (upstream #69167 open — the arch
    # list is 9.0a/10.0f/10.3f), so the first native conv_mix launch dies
    # at the memory-profiling run. Route the TP1 conv/mix through the
    # stock SM12x torch-compose fallback on Thor (MJOLNIR_INKLING2_
    # CONV_MIX_NATIVE=1 forces the native path once the base carries
    # sm_110 cubins). Touches only models/inkling2/nvidia/ops/conv_mix.py
    # — order-independent. The TP>1 all-reduce/seq-conv kernels stay
    # native-only: TP1 configs only until #69167 lands in a pin.
    "thor-inkling2-sm110-conv-mix.patch",
    # ...and the FA4 sheared-bias rel-attention on sm_110 (the stock gate is
    # family(100); the FA2/FA3 ext wheels are sm_80/sm_90a-only so the
    # Triton-sm90 fallback is the only stock path on Thor). Env-off via
    # MJOLNIR_INKLING2_FA4_REL=0. Same file family, order-independent.
    "thor-inkling2-fa4-rel-sm110.patch",
    # ...and the unit-scale E4M3 KV cache on sm_110 (the model's fp8 path is
    # complete and arch-generic; only the dtype gate is SM120-ary). Same file
    # as the fa4-rel patch, disjoint hunks. Env-off MJOLNIR_INKLING2_FP8KV=0.
    "thor-inkling2-fp8kv-sm110.patch",
    # ...and the decode split-budget on Thor (the /8 denominator is tuned for
    # 100+-SM parts; the ncu repro shows 15% occupancy on the verify shape).
    "thor-inkling2-splitcap-sm110.patch",
    # the sm100 family gate widen: upstream tcgen05/tml kernels on sm_110
    "thor-inkling2-sm100gate-sm110.patch",
    # E5: sm110 tuning overrides for the sm90-path launch constants
    "thor-inkling2-tune-sm110.patch",
    # The Mjolnir attention backend name: adds AttentionBackendEnum.MJOLNIR
    # (-> mjolnir.backend.MjolnirAttentionBackend, the mjolnir-kernel
    # package) to v1/attention/backends/registry.py, which no other patch
    # touches — order-independent EXCEPT that its fa_utils.py hunk assumes
    # the fa4 patches' version of that file: it lands AFTER the fa4 pair
    # (region-disjoint from them, but a reorder must keep this order).
    # Makes `--attention-backend MJOLNIR` parse.
    "thor-mjolnir-attention-backend.patch",
    # CuTe DSL NVFP4 MoE on sm_110: vLLM's FlashInferCuteDSLExperts gate admits
    # family 110 (the FlashInfer-side capability checks are in
    # FLASHINFER_PATCH_ORDER). Only selected for W4A4 (activation-quantized)
    # NVFP4 experts; W4A16 checkpoints keep MARLIN. Touches only
    # fused_moe/experts/flashinfer_cutedsl_moe.py - order-independent. Still
    # NOT base-covered in 0.34 (the gate is family(100)-only there); the two
    # sm_110 MoE gate fixes above it were retired as 0.34-base-covered.
    "thor-sm110-cutedsl-moe-gate.patch",
    # Dense-decoder support for the qwen4_exp family (the confirmed Qwen4-27B
    # shape IS dense — recovered Intel fp8_dense.cpp): v0.32.0 builds a sparse
    # MoE block for every layer and transformers rejects num_experts <= 0, so
    # the patch mirrors qwen3_next's own hybrid rule — a checkpoint that lists
    # every layer in mlp_only_layers gets Qwen3NextMLP. INERT for every
    # existing checkpoint (none set mlp_only_layers on qwen4_exp) and for all
    # other model families; config-driven, so no sm_110 gate. Touches only
    # models/qwen4_exp/nvidia/model.py — order-independent. If the released
    # 27B arrives under a NEW model_type instead, this is necessary but not
    # sufficient (a new architectures registration is also needed) — see
    # task/research/qwen4-27b-upstream-prep/ (release-day checklist).
    "thor-qwen4-dense-model.patch",
    # The Mjolnir QSA decode: probe-gated (MJOLNIR_ATTN_QSA) impl swap inside
    # Qwen4ExpQSAAttention — wraps the stock impl and routes pure-decode steps
    # through the kernel package's scratch-gather (mjolnir.qsa_impl, which
    # ships in mjolnir-kernel and degrades to the stock Triton path whenever
    # a guard fails). INERT unless the env var is set.
    "thor-qwen4-qsa-mjolnir.patch",
    # Retired at the 0.35.0rc0.dev nightly bump: #64739 (safetensors shards
    # listed under an index subdirectory) is base-covered there — both
    # hunks skip as already applied on the nightly wheel.
    # Qwen4Exp PLE slot-pool cache in front of vLLM's disk table
    # (disk_offload + MJOLNIR_PLE_DISK_CACHE_GB): one hunk in
    # models/qwen4_exp/nvidia/ngram_embedding.py, which no other patch
    # touches; the layer itself is mjolnir.qwen4exp.ple_disk.
    "thor-qwen4-ple-disk.patch",
    # Reduced MTP draft vocabulary (MJOLNIR_DRAFT_VOCAB) + 8-bit draft head /
    # MTP dense layers (QWEN4EXP_DRAFT_HEAD_FP8, QWEN4EXP_MTP_DENSE_FP8):
    # models/qwen4_exp/nvidia/mtp.py, which no other patch touches. Logic in
    # mjolnir.qwen4exp.draft_head.
    "thor-qwen4-draft-head.patch",
    # Screened target lm_head (QWEN4EXP_LM_HEAD_SCREENED, backport of vLLM
    # #60566 into mjolnir.qwen4exp.screened_head): nvidia/model.py hunks
    # diffed against thor-qwen4-draft-head.
    "thor-qwen4-screened-head.patch",
    # Online FP8 for the target's BF16 dense side layers (QWEN4EXP_SIDE_FP8):
    # one nvidia/model.py hunk diffed against the screened-head patch;
    # mjolnir.qwen4exp.side_fp8.
    "thor-qwen4-side-fp8.patch",
    # Online 8-bit hyperconnection GEMMs (QWEN4EXP_HC_FP8):
    # nvidia/hyperconnection.py (no other patch) + one nvidia/model.py log
    # hunk diffed against the side-fp8 patch; mjolnir.qwen4exp.hc_fp8.
    "thor-qwen4-hc-fp8.patch",
    # FULL CUDA graphs for the MTP draft decode steps under a PIECEWISE
    # target (MJOLNIR_SPEC_STEP_DRAFT_FULL_CG): one gate in
    # autoregressive/speculator.py AutoRegressiveSpeculator.init_cudagraph_manager
    # (vLLM #60989 moved it to the AR base class, so it covers every
    # autoregressive drafter); diffed on top of
    # thor-fused-draft-t5-plan-refresh (same file, disjoint hunks). Env-gated
    # only, any model.
    "thor-draft-full-cg-piecewise.patch",
    # The QSA fp6 (e2m3) main cache (kv-cache-dtype: fp6_e2m3): the dense
    # scheme (mjolnir.kvfp6) on qwen4_exp QSA pools. Four hunks, all
    # models/qwen4_exp/nvidia/qsa.py — the same file the qsa-mjolnir patch
    # touches but DISJOINT regions (the backend dtype list, the impl ctor
    # check, the layer's torch-dtype resolution, the pool spec); lands after
    # it. The pluggable-dtype registration itself ships in mjolnir.kvdtype
    # (entry point, no patch); stock-fallback reads expand to e4m3 (kvfp6).
    "thor-qwen4-qsa-fp6.patch",
    # FlashInfer CUTLASS MXFP8 tactic pin by M bucket (MJOLNIR_MXFP8_TACTIC):
    # model_executor/kernels/linear/mxfp8/flashinfer.py, no other patch.
    "thor-mxfp8-tactic-sm110.patch",
]

# Patches that land in the installed *flashinfer* package (a separate
# site-packages tree, not the vllm package). The retired GDN prefill
# enablement hunk (STAGGER FIX below) widened the non-CP dispatch
# allowlist at gdn_prefill.py:460 from major 10 to (10, 11) so the SM100
# CuTe-DSL kernel (JIT-compiled for the device) is reachable on sm_110 —
# its vLLM-side grant is retired with it, so nothing today changes
# behavior on any device but the fp6 hunk.
FLASHINFER_PATCH_ORDER = [
    # STAGGER FIX: "thor-gdn-prefill-fi-sm110.patch",
    "fi-fp6-kv-ampere-prefill.patch",
    # The FlashInfer half of the CuTe DSL MoE enablement, the 0.34 port: the
    # two top-level wrappers' hard major!=10 gate admits 11 (the moe_utils
    # majors include 11 upstream -- no hunk). The 0.33-era
    # fi-cutedsl-moe-sm110.patch is retired (its paths are gone).
    "fi-cutedsl-moe-sm110-0.34.patch",
]


def package_root(explicit: str | None = None) -> Path:
    if explicit:
        root = Path(explicit).resolve()
        if not (root / "v1" / "core" / "scheduler.py").exists() and not root.exists():
            raise SystemExit(f"vllm package root not found: {root}")
        return root
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise SystemExit("could not locate the installed vllm package")
    return Path(spec.origin).parent


def flashinfer_package_root(explicit: str | None = None) -> Path:
    if explicit:
        root = Path(explicit).resolve()
        if not (root / "gdn_prefill.py").exists():
            raise SystemExit(f"flashinfer package root not found: {root}")
        return root
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or spec.origin is None:
        raise SystemExit("could not locate the installed flashinfer package")
    return Path(spec.origin).parent


def apply_one(root: Path, patch_dir: Path, name: str) -> None:
    patch_file = patch_dir / name
    if not patch_file.exists():
        raise SystemExit(f"missing patch file: {patch_file}")
    # -p1 strips the leading a/vllm/ so files land under <cwd>/vllm/...
    # --forward (-N): skip hunks already applied, keep going on the rest.
    result = subprocess.run(
        ["patch", "-p1", "--forward", "--fuzz=3", "-i", str(patch_file)],
        cwd=root.parent,
        capture_output=True,
        text=True,
    )
    status = result.stdout.strip().splitlines()
    for line in status:
        print(f"[{name}] {line}", file=sys.stderr)
    # patch(1) exits 0 (applied) or 1 (some hunks skipped/already applied).
    # Non-zero > 1 means a real failure (rejected hunk, bad patch, etc.).
    if result.returncode > 1:
        print(f"[{name}] REJECTED: {result.stderr.strip()}", file=sys.stderr)
        raise SystemExit(
            f"patch {name} failed to apply (rc={result.returncode}); "
            f"the base nightly likely diverged — re-adapt {name}.\n{result.stderr}"
        )
    # `--forward` writes .rej/.orig for hunks that are already present (a fix
    # the base nightly already merged). Remove them so the tree stays clean.
    for junk in ["*.rej", "*.orig"]:
        for f in root.glob(f"**/{junk}"):
            f.unlink(missing_ok=True)


CHANGE_NOTICE = "# Modified by the Mjolnir image (https://github.com/GoBrrr-ai/mjolnir):"


def patched_files(patch_file: Path) -> list[str]:
    """Package-relative paths a patch changes (``+++ b/vllm/x.py`` -> ``x.py``)."""
    out = []
    for line in patch_file.read_text().splitlines():
        if line.startswith("+++ b/"):
            out.append(line[len("+++ b/") :].split("/", 1)[1])
    return out


def _comment_prefix(rel: str) -> str:
    """``#`` for python/shell, ``//`` for C/C++ — a ``#`` line in a header is
    an invalid preprocessing directive, not a comment."""
    return "//" if Path(rel).suffix in {".cuh", ".cu", ".c", ".cc", ".cpp", ".h", ".hpp"} else "#"


def mark_changed_files(root: Path, patches_dir: Path, names: list[str]) -> None:
    """Apache-2.0 section 4(b): every modified file carries a prominent notice
    that it was changed. One trailing comment per file (appended, so no line
    numbers shift), naming the patches and where their text is shipped."""
    by_file: dict[str, list[str]] = {}
    for name in names:
        for rel in patched_files(patches_dir / name):
            by_file.setdefault(rel, []).append(name)
    for rel, pnames in by_file.items():
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text()
        if CHANGE_NOTICE in text:
            continue
        c = _comment_prefix(rel)
        sep = "" if text.endswith("\n") else "\n"
        path.write_text(
            text + f"{sep}\n{c} {CHANGE_NOTICE}\n{c} {', '.join(pnames)} (see /usr/share/doc/mjolnir/patches/).\n"
        )


def main() -> int:
    here = Path(__file__).resolve().parent
    patches_dir = here / "patches"
    explicit = sys.argv[1] if len(sys.argv) > 1 else None
    fi_explicit = sys.argv[2] if len(sys.argv) > 2 else None
    root = package_root(explicit)
    print(f"vllm package root: {root}", file=sys.stderr)

    for name in PATCH_ORDER:
        apply_one(root, patches_dir, name)

    fi_root = flashinfer_package_root(fi_explicit)
    print(f"flashinfer package root: {fi_root}", file=sys.stderr)
    for name in FLASHINFER_PATCH_ORDER:
        apply_one(fi_root, patches_dir, name)

    mark_changed_files(root, patches_dir, PATCH_ORDER)
    mark_changed_files(fi_root, patches_dir, FLASHINFER_PATCH_ORDER)

    # Authoritative gate: every fix must be present after application.
    sys.path.insert(0, str(here))
    import verify_patches

    missing = verify_patches.verify(root)
    fi_missing = verify_patches.verify_flashinfer(fi_root)
    if missing or fi_missing:
        for m in missing:
            print(f"MISSING FIX: {m}", file=sys.stderr)
        for m in fi_missing:
            print(f"MISSING FLASHINFER FIX: {m}", file=sys.stderr)
        return 1
    print(f"all {len(verify_patches.FIXES)} fixes verified in {root}", file=sys.stderr)
    print(
        f"all {len(verify_patches.FLASHINFER_FIXES)} flashinfer fixes verified in {fi_root}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
