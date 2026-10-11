# syntax=docker/dockerfile:1.7
#
# vLLM overlay for Jetson AGX Thor (aarch64, SM 11.0a) + the Mjolnir kernel.
#
# BUILD CONTEXT = THE REPO ROOT (the image installs src/mjolnir):
#   docker build -f docker/vllm-thor/Dockerfile -t narandill/mjolnir-vllm:v0.31.0 .
#   (or: mjolnir image build)
#
# Image tags follow the upstream vLLM version this overlays (BASE_IMAGE below):
# vllm/vllm-openai:v0.31.0 -> narandill/mjolnir-vllm:v0.31.0 (+ :v0.31.0-<sha>).
# .github/workflows/docker-image.yml derives the tag from BASE_IMAGE.
#
# This is a SMALL OVERLAY, not a from-source build. It takes the official
# vLLM docker image (release tag or nightly; the image ships a prebuilt
# aarch64 wheel with the SM 11.0a CUDA kernels), then hand-adapts and applies
# a stack of vLLM PR fixes (NOT yet merged upstream, required for Qwen3.8-27B
# hybrid GDN/Mamba spec-decode correctness) plus Thor-authored vLLM and
# FlashInfer patches.
# The fixes are hand-adapted (NOT a raw `git apply`) because the base code
# has diverged from each PR's base.
#
# The authoritative patch list — every patch, what it fixes, wall analysis,
# drop rationale, and the sm_110 gate probes — is in PATCHES.md (the single
# source of truth). The patch files are in patches/, applied by
# apply_patches.py and verified by verify_patches.py; the build fails if a
# patch no longer applies (the signal that the base moved).
#
# ────────────────────────────────────────────────────────────────────────────
# BUMPING TO A NEWER vLLM
#
# The base is a release tag (vllm/vllm-openai:v0.X.Y, aarch64 variant) or the
# nightly (vllm/vllm-openai:nightly-aarch64).
#
#   1. `docker pull --platform linux/arm64 vllm/vllm-openai:<tag>`
#   2. `docker image inspect vllm/vllm-openai:<tag> --format '{{index .RepoDigests 0}}'`
#   3. Update the default BASE_IMAGE / BASE_IMAGE_DIGEST below.
#   4. `docker build -t <tag> .`
#
# The build will FAIL if a patch no longer applies or a fix is missing after
# application — that is the signal that the base moved past the assumptions
# one of the patches was written against, and that patch needs re-adapting.
# Patch application is idempotent (patch -N / --forward): a fix the base image
# already merged is skipped, not re-applied.
#
# Verified against base:
#   vllm 0.31.0  (vllm/vllm-openai:v0.31.0-ubuntu2404, released 2026-10-05)
#   flashinfer-python / flashinfer-cubin 0.7.0.post1
#   torch 2.13.0+cu130
#   (#55390 is upstream since 0.31.0: its patch is gone, its sentinels stay)
#
# Build (on Thor or any arm64 builder; repo-root context):
#   docker build -f docker/vllm-thor/Dockerfile -t narandill/mjolnir-vllm:v0.31.0 .
#
# Base image lineage: vllm/vllm-openai (aarch64, CUDA 13.0, Ubuntu 24.04,
# Python 3.12). JetPack 7.x on Thor ships CUDA 13.0, so the runtime driver
# matches the image's CUDA toolkit.

ARG BASE_IMAGE=vllm/vllm-openai:v0.31.0-ubuntu2404
# Pinned digest so the default build is reproducible. Override with
# --build-arg BASE_IMAGE=vllm/vllm-openai@<digest> (or the floating tag,
# though the floating tag is the recommended BUMP path, see above).
ARG BASE_IMAGE_DIGEST=vllm/vllm-openai@sha256:3fb023cc987cc7c2bca7ca14c1e0aff5ae06bd9431bf07773731ce7e8a682460

FROM ${BASE_IMAGE_DIGEST:-${BASE_IMAGE}}

ARG MJOLNIR_REVISION=unknown
LABEL org.opencontainers.image.title="Mjolnir — vLLM for Jetson AGX Thor (sm_110)" \
        org.opencontainers.image.description="vllm-openai v0.31.0 aarch64 overlay: the Mjolnir decode-attention kernel (--attention-backend MJOLNIR) + hand-adapted vLLM PR fixes + Thor sm_110 patches (docker/vllm-thor/PATCHES.md)" \
        org.opencontainers.image.source="https://github.com/thor-arena/mjolnir" \
        org.opencontainers.image.revision="${MJOLNIR_REVISION}" \
        org.opencontainers.image.licenses="Apache-2.0 AND BSD-3-Clause"

# patch(1) for applying the hand-adapted patches.
RUN apt-get update && \
    apt-get install -y --no-install-recommends patch && \
    rm -rf /var/lib/apt/lists

# ── SM 11.0a platform env (Jetson AGX Thor) ────────────────────────────────
# The prebuilt wheel already carries SM 11.0a gencode, so no CUDA source
# compilation happens at build time. These env vars cover the RUNTIME JIT
# paths so they target the right arch:
#   TORCH_CUDA_ARCH_LIST       — Triton / torch.compile (Inductor) codegen
#   TRITON_PTXAS_PATH          — Triton finds CUDA 13.0 ptxas (rejects sm_110a
#                                otherwise: "ptxas fatal: Value 'sm_110a' ...")
#   FLASHINFER_CUDA_ARCH_LIST  — FlashInfer JIT cache prebuilds
#   LD_LIBRARY_PATH            — JIT-compiled kernels link the CUDA 13.0 libs
ENV TORCH_CUDA_ARCH_LIST=11.0a \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    FLASHINFER_CUDA_ARCH_LIST=11.0a \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# ── Apply the hand-adapted PR fixes to the installed vllm package ───────────
# Copy the tooling + patches in, then apply against the *installed* vllm
# (apply_patches.py resolves it via importlib). --forward makes the build
# idempotent across nightly bumps; verify_patches.verify is the authoritative
# gate and fails the build if any fix is missing.
WORKDIR /opt/vllm-patches
COPY docker/vllm-thor/patches/ patches/
COPY docker/vllm-thor/apply_patches.py docker/vllm-thor/verify_patches.py \
     docker/vllm-thor/functional_check_55390_55519.py \
     docker/vllm-thor/kernel_test_fi_update.py \
     docker/vllm-thor/kernel_test_dspark_draft_nonncausal.py ./

# The patch stack includes thor-mjolnir-attention-backend.patch: it adds the
# MJOLNIR member to vLLM's AttentionBackendEnum, so
# `vllm serve <model> --attention-backend MJOLNIR` parses and resolves.
RUN python3 apply_patches.py \
    && python3 verify_patches.py

# ── Install the Mjolnir kernel (mjolnir-kernel, from this checkout) ─────────
# A pure-Python package (CuTe DSL kernel + vLLM backend + runner). --no-deps:
# torch / vllm / nvidia-cutlass-dsl / flashinfer are the base image's own.
# Its vllm.general_plugins entry point registers the backend in every vLLM
# process (MJOLNIR, and CUSTOM for stock-vLLM compatibility).
COPY src/mjolnir /opt/mjolnir-kernel
RUN python3 -m pip install --no-cache-dir --no-deps --break-system-packages /opt/mjolnir-kernel \
    && rm -rf /opt/mjolnir-kernel \
    && python3 - <<'PY'
import importlib.metadata as md
import mjolnir
from vllm.v1.attention.backends.registry import AttentionBackendEnum

assert AttentionBackendEnum["MJOLNIR"].get_path() == mjolnir.BACKEND_PATH, "MJOLNIR enum member missing"
eps = [e for e in md.entry_points(group="vllm.general_plugins") if e.name == "mjolnir"]
assert eps and eps[0].value == "mjolnir:register", "mjolnir vllm.general_plugins entry point missing"
print("mjolnir-kernel", mjolnir.__version__, "->", mjolnir.BACKEND_PATH)
PY

# ── Build-time smoke test (no GPU/driver required) ──────────────────────────
# Byte-compile every patched file and import the pure-python modules that
# touch the changed code paths. This catches a bad edit that compiles per-file
# but breaks at import. (The GPU C extensions load for real only at runtime on
# Thor, where the driver is present.)
RUN python3 - <<'PY'
import importlib, importlib.util, pathlib, py_compile

root = pathlib.Path(importlib.util.find_spec("vllm").origin).parent
touched = [
    "v1/attention/backends/flashinfer.py",
    "v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py",
    "config/speculative.py",
    "v1/core/sched/scheduler.py",
    "v1/core/kv_cache_utils.py",
    "v1/core/single_type_kv_cache_manager.py",
    "v1/core/kv_cache_coordinator.py",
]
for rel in touched:
    py_compile.compile(str(root / rel), doraise=True)
print("compileall:", len(touched), "patched files OK")

# Pure-python import smoke (no GPU needed).
import vllm
import vllm.config.speculative
import vllm.v1.core.kv_cache_utils
import vllm.v1.core.kv_cache_coordinator
import vllm.v1.core.single_type_kv_cache_manager
print("imports OK, vllm", vllm.__version__)
PY

# ── Behavioral gate for 55390/55519 (draft-group annotation + warning) ───────
# The sentinel check above proves the code landed; this one proves it behaves:
# a Qwen3.5-shaped hybrid (GDN + full-attn + MTP layer) must annotate exactly
# the MTP group as the draft group, never the Mamba groups, and the warning
# path must be gated on use_eagle_block_drop(). Mirrors the PRs' unit tests.
RUN python3 /opt/vllm-patches/functional_check_55390_55519.py

# ── GPU gate for the Thor fused-draft-decode kernel (needs a CUDA device) ────
# The five PR fixes are upstream code; the Thor fused-draft-decode kernel is
# ours, so it needs more than a sentinel check. This runs
# _advance_fi_decode_plan_kernel against a numpy reference (random, page
# boundary, empty/padded, multi-page, and a 20x determinism repeat). It
# self-skips (exit 0) when no CUDA device is visible or the GPU is saturated,
# so a plain no-GPU build still passes; build with --gpus all on a Thor to
# actually exercise the kernel.
RUN python3 /opt/vllm-patches/kernel_test_fi_update.py

# ── GPU gate for the DSpark non-causal cudagraph (needs a CUDA device) ──────
# The DSpark non-causal draft patch (thor-dspark-draft-fi-noncausal-cudagraph)
# routes a uniform non-causal draft batch through a cuda-graph-mode FI *prefill*
# wrapper with persistent KV buffers, so the captured run() replays safely
# across ragged re-plans. This drives the real patched FlashInferMetadataBuilder
# build()/run() (gate, routing, non-causal run vs. reference, capture-replay,
# determinism). Like the fused-draft-decode gate it self-skips (exit 0) when no
# CUDA device is visible or the GPU is saturated, so a plain no-GPU build still
# passes; build with --gpus all on a Thor to actually exercise the graph.
RUN python3 /opt/vllm-patches/kernel_test_dspark_draft_nonncausal.py

# Licensing: the project LICENSE + NOTICE and the exact patch texts ship in
# the image (the patched vLLM / FlashInfer files point here; see NOTICE). The
# base image's own components keep their license files where they are.
COPY LICENSE NOTICE /usr/share/doc/mjolnir/
RUN cp -r /opt/vllm-patches/patches /usr/share/doc/mjolnir/patches

# Keep the tree tidy; the applied changes live in the site-packages vllm.
RUN rm -rf /opt/vllm-patches && true

# ── Build ──────────────────────────────────────────────────────────────────
#   docker build -f docker/vllm-thor/Dockerfile -t narandill/mjolnir-vllm:v0.31.0 .
#
# This is a thin overlay: the heavy lifting (CUDA kernels for SM 11.0a) comes
# from the prebuilt wheel in the base image. The first server launch pays a
# one-time FlashInfer JIT cost; keep that cache on a persistent volume
# (see docker/vllm.yaml) so restarts are fast.
