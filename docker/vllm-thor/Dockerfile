# syntax=docker/dockerfile:1.7
#
# vLLM overlay for the Mjolnir kernel on Jetson Tegra (aarch64).
# ONE image serves every shipped Tegra platform: Thor (sm_110) and Orin
# (sm_87) are both aarch64, the kernel family resolves at runtime
# (mjolnir.platforms), and mjolnir_cli.dockerctl exports the per-arch JIT
# envs at serve time (the Thor pins in ENV below are the build default).
#
# BUILD CONTEXT = THE REPO ROOT (the image installs src/mjolnir):
#   docker build -f docker/vllm-thor/Dockerfile -t narandill/mjolnir-vllm:v0.34.0 .
#   (or: mjolnir image build)
#
# Image tags follow the upstream vLLM version this overlays (BASE_IMAGE below):
# vllm/vllm-openai:v0.34.0 -> narandill/mjolnir-vllm:v0.34.0 (+ :v0.34.0-<sha>).
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
#   vllm 0.34.0  (vllm/vllm-openai:v0.34.0, tag 62d4bc3c7, released 2026-11)
#   flashinfer-python / flashinfer-cubin 0.7.3
#   torch 2.14.0+cu130 (Triton 3.8.0); vllm-flash-attn pin d8dca8f06c74
#
# Build (on Thor or any arm64 builder; repo-root context):
#   docker build -f docker/vllm-thor/Dockerfile -t narandill/mjolnir-vllm:v0.34.0 .
#
# Base image lineage: vllm/vllm-openai (aarch64, CUDA 13.0, Ubuntu 24.04,
# Python 3.12). JetPack 7.x on Thor ships CUDA 13.0, so the runtime driver
# matches the image's CUDA toolkit. (From 0.33.0 the release images are
# CUDA 13.0 only and the `-ubuntu2404` suffix tag is gone: the plain
# version tag IS the ubuntu2404 aarch64 build.)

ARG BASE_IMAGE=vllm/vllm-openai:v0.34.0
# Pinned digest so the default build is reproducible: the release
# multi-arch INDEX (arm64 leaf e40eb98263fa…, published 2026-11-16
# 22:23 UTC; tag sha 62d4bc3c7). While this ARG holds a value it WINS
# over BASE_IMAGE — override THIS one (or empty it) to float.
ARG BASE_IMAGE_DIGEST=vllm/vllm-openai@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1

FROM ${BASE_IMAGE_DIGEST:-${BASE_IMAGE}}

ARG MJOLNIR_REVISION=unknown
LABEL org.opencontainers.image.title="Mjolnir — vLLM for Jetson AGX Thor (sm_110)" \
        org.opencontainers.image.description="vllm-openai v0.34.0 aarch64 overlay: the Mjolnir decode-attention kernel (--attention-backend MJOLNIR) + hand-adapted vLLM PR fixes + Thor sm_110 patches (docker/vllm-thor/PATCHES.md)" \
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
     docker/vllm-thor/kernel_test_fi_update.py \
     docker/vllm-thor/kernel_test_dspark_draft_nonncausal.py ./

# The patch stack includes thor-mjolnir-attention-backend.patch: it adds the
# MJOLNIR member to vLLM's AttentionBackendEnum, so
# `vllm serve <model> --attention-backend MJOLNIR` parses and resolves, and
# makes FlashAttention 4 the backend's default fallback (prefill) on sm_110.
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
# fp6_e2m3 registers through vLLM's pluggable KV-dtype API (#52670, in the
# base since 0.33.0) from the plugin entry point — no image patch anymore.
import mjolnir.kvdtype
assert mjolnir.kvdtype.register(), "fp6_e2m3 KV dtype registration failed"
from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode
assert get_kv_quant_mode("fp6_e2m3") == KVQuantMode.CUSTOM, "fp6_e2m3 KV dtype not registered"
import mjolnir.calibrate, mjolnir.kvfp6, mjolnir.kvscales, mjolnir.qpack  # noqa: F401 (pure-python, no GPU)
import mjolnir.qwen4exp.ple_disk_cache, mjolnir.qwen4exp.ple_disk_table  # noqa: F401 (NumPy only)
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
    "v1/worker/gpu/spec_decode/autoregressive/speculator.py",
    "v1/worker/gpu/spec_decode/dflash/speculator.py",
    "v1/attention/backends/fa_utils.py",
    "v1/attention/backends/registry.py",
    "vllm_flash_attn/cute/interface.py",
    "vllm_flash_attn/cute/sm100_hd256_2cta_fmha_forward.py",
    "models/qwen4_exp/nvidia/model.py",
    "models/qwen4_exp/nvidia/qsa.py",
    # the Qwen3.8-Flash-Next stack
    "model_executor/model_loader/default_loader.py",
    "model_executor/model_loader/weight_utils.py",
    "models/qwen4_exp/nvidia/ngram_embedding.py",
]
for rel in touched:
    py_compile.compile(str(root / rel), doraise=True)
print("compileall:", len(touched), "patched files OK")

# Pure-python import smoke (no GPU needed).
import vllm
import vllm.v1.core.kv_cache_utils
import vllm.v1.core.kv_cache_coordinator
import vllm.v1.core.single_type_kv_cache_manager
print("imports OK, vllm", vllm.__version__)

# The QSA impl-swap contract, checked at build time (the build host has
# no GPU — this is an import-level signature/subclass assert only; the
# kernel itself always runs on the device, see qsa_impl.py): the kernel
# package must subclass
# THIS base's stock impl under its 0.34 name and expose the kw-only forward
# the layer's _run_qsa call site uses. Catches the rename/arity class of
# drift at build time instead of at first serving.
import inspect

import mjolnir.qsa_impl as _qsa_impl

# qsa <-> model is order-sensitive upstream (qsa does `from . import model`
# mid-file): model must be imported first, as every vLLM runtime path does.
import vllm.models.qwen4_exp.nvidia.model as _qsa_model_mod  # noqa: F401
import vllm.models.qwen4_exp.nvidia.qsa as _qsa_mod

_cls = _qsa_impl.build_impl_class()
assert issubclass(_cls, _qsa_mod.Qwen4ExpQSAImpl), (
    f"mjolnir QSA impl does not subclass this base's Qwen4ExpQSAImpl "
    f"(mro[1]={_cls.__mro__[1].__module__}.{_cls.__mro__[1].__name__})"
)
_params = inspect.signature(_cls.forward).parameters
_positional = [p.name for p in _params.values()
               if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
assert _positional == ["self", "layer", "query", "kv_cache", "attn_metadata", "output"], (
    f"QSA impl forward positional surface drifted: {_positional}"
)
for _kw in ("token_to_req", "use_prefill_config", "output_gate"):
    assert _kw in _params and _params[_kw].kind is inspect.Parameter.KEYWORD_ONLY, (
        f"QSA impl forward is missing the kw-only {_kw!r} the 0.34 _run_qsa passes"
    )
print("qsa swap contract: subclass + kw-only forward OK")
PY

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
#   docker build -f docker/vllm-thor/Dockerfile -t narandill/mjolnir-vllm:v0.34.0 .
#
# This is a thin overlay: the heavy lifting (CUDA kernels for SM 11.0a) comes
# from the prebuilt wheel in the base image. The first server launch pays a
# one-time FlashInfer JIT cost; keep that cache on a persistent volume
# (see docker/vllm.yaml) so restarts are fast.
