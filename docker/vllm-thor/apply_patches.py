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

# Order: re-based onto upstream vLLM 0.33.0 (2026-11 bump). 54165, 55519,
# the fp6 e2m3 KV dtype and the split top-p sampler fix retired at this
# bump — all four are in the base now (see PATCHES.md); the gdn-prefill
# pair is retired too (the c4 stagger, commented STAGGER FIX below). The
# fa4 sm_110 gate patches touch files disjoint from every other patch
# (v1/attention/backends/fa_utils.py +
# vllm_flash_attn/cute/interface.py; the hd256 patch additionally touches
# vllm_flash_attn/cute/sm100_hd256_2cta_fmha_forward.py), so their position
# among the middle patches is immaterial — except that the two fa4 sm_110
# patches both touch v1/attention/backends/fa_utils.py, so the hd256 patch
# (whose fa_utils hunks are diffed against the hd128-patched file) must
# land after the hd128 one. The three Thor v1/attention/backends/flashinfer.py
# patches land in sequence: 50885 (the FI-native uniform-multi-token FULL
# grant — on 0.33.0 slimmed to the probe + the two gates; the wrapper
# plumbing itself is upstream), the fused-draft-decode patch (its hunks
# sit on top of 50885's), then the DSpark non-causal cudagraph patch
# (its get_cudagraph_support hunk is diffed against 50885's grant).
PATCH_ORDER = [
    "49652-draft-decode-capture.patch",
    "50885-flashinfer-full-cudagraph.patch",
    # STAGGER FIX (2026-11-04): the GDN
    # prefill pair carries the c4 aggregate stagger — removed. Stock GDN
    # prefill path serves correctly on 0.33 (kernelonly A/B, acc 3.59).
    "thor-fused-draft-decode-fi-native-update.patch",
    # Refreshes the T5 split-KV work plan when the fused draft loop replays
    # its FULL cudagraph: the _fused_multi_step_decode FULL branch returns
    # before building the draft metadata, so a host-side per-cycle plan
    # (ours) would stay baked at capture-time seq_lens. Same feature
    # family as the patch above; the speculator file is disjoint from it.
    "thor-fused-draft-t5-plan-refresh.patch",
    "thor-dspark-draft-fi-noncausal-cudagraph.patch",
    # Disjoint from every patch except the hd256 one below (fa_utils.py) —
    # must land before it.
    "thor-fa4-fp8kv-sm110.patch",
    # Both touch v1/attention/backends/fa_utils.py; its fa_utils hunks are
    # diffed against the hd128-patched file, so it must land on top.
    "thor-fa4-hd256-fp8-sm110.patch",
    # 1CTA decode carve-out in vllm_flash_attn/cute/interface.py — disjoint
    # from every other patch's files/regions; placed here to keep the fa4
    # group together (order-independent, lands after hd256).
    "thor-fa4-hd256-1cta-decode-sm110.patch",
    # 11th (c1 fix): dispatch-side shape gate for the DSpark draft's FULL
    # cudagraph replay on sm_110. It edits v1/worker/gpu/spec_decode/
    # dflash/speculator.py, which no other patch touches, but its context
    # is the code the dspark patch's ecosystem introduced (the FULL CG
    # dispatch), so it lands after thor-dspark.
    "thor-draft-cg-gate-sm110.patch",
    # The Mjolnir attention backend name: adds AttentionBackendEnum.MJOLNIR
    # (-> mjolnir.backend.MjolnirAttentionBackend, the mjolnir-kernel
    # package) to v1/attention/backends/registry.py, which no other patch
    # touches — order-independent. Makes `--attention-backend MJOLNIR` parse.
    "thor-mjolnir-attention-backend.patch",
    # Two sm_110 MoE-backend gate fixes (drop each when the PR lands in a
    # release we track): the OAI Triton MXFP4 MoE selector keeps MARLIN as
    # the default on SM110 (the OAI Triton path fails to compile matmul_ogs
    # at engine init, #61374) and the CUTLASS FP4 MoE backend must not
    # auto-select on sm_110 (faults in profile_run, TMA descriptor init).
    # #62133 is merged upstream (2026-10-27) but NOT in the 0.33.0 release
    # branch — re-check at every bump. All five touched files are disjoint
    # from every other patch — order-independent.
    "thor-sm110-mxfp4-moe-gate.patch",
    "thor-sm110-cutlass-fp4-moe-guard.patch",
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
    # Sparse-indexer decode top-k on sm_110: v0.32.0's auto chain leads with
    # cooperative_topk, whose thread-block-cluster launch is REJECTED on
    # Thor (device-proven 2026-10-31: "cluster misconfiguration" ->
    # EngineDeadError at the first decode top-k; the upstream SM120 report
    # vllm-project/vllm#47266 Issue 8 is the same failure class and its fix
    # excludes the family). The patch extends the exclusion to family 110 —
    # the gate falls through to persistent_topk. Probe-gated: inert on every
    # other arch.
    "thor-qwen4-topk-sm110.patch",
    # The Mjolnir QSA decode: probe-gated (MJOLNIR_ATTN_QSA) impl swap inside
    # Qwen4ExpQSAAttention — wraps the stock impl and routes pure-decode steps
    # through the kernel package's scratch-gather (mjolnir.qsa_impl, which
    # ships in mjolnir-kernel and degrades to the stock Triton path whenever
    # a guard fails). INERT unless the env var is set.
    "thor-qwen4-qsa-mjolnir.patch",
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
        ["patch", "-p1", "--forward", "-i", str(patch_file)],
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
