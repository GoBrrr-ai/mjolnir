---
name: update-base-image
description: Bumps the pinned vllm/vllm-openai base of the docker/vllm-thor image (narandill/mjolnir-vllm) — new version + digest, patch-stack re-verification (including the MJOLNIR backend patch), canaries, kernel verification, image tag and default switch. Use when a new vLLM release is out, or when the patch stack must be re-validated against a newer base.
compatibility: opencode
---

# Update the base image (narandill/mjolnir-vllm)

The image in `docker/vllm-thor/` is a thin layer on
`vllm/vllm-openai:v<version>-ubuntu2404` plus the Mjolnir kernel package
(`src/mjolnir`). Bumping the base is a **gated process**: the image build itself
is the gate — it applies the patch stack (`apply_patches.py`, which includes
`thor-mjolnir-attention-backend.patch`, the `MJOLNIR` enum member) and verifies
it (`verify_patches.py`), then smoke-imports the backend; it **fails** if a patch
stops applying. A build failure means the base moved past that patch's
assumptions; the fix is to hand re-adapt the patch, not to weaken the gate.

**The image tag is the upstream vLLM version** (`v0.31.0`), independent of the
PyPI package versions.

Background: `docs/faq/base-image-bump.md`. The Dockerfile's "BUMPING TO A NEWER
vLLM" block is the short form.

## Procedure

### 1. Fetch the new base + digest (on the aarch64 box)
```bash
docker pull vllm/vllm-openai:v<X.Y.Z>-ubuntu2404
docker image inspect vllm/vllm-openai:v<X.Y.Z>-ubuntu2404 --format '{{index .RepoDigests 0}}'
```

### 2. Point the build at it
- `docker/vllm-thor/Dockerfile`: `ARG BASE_IMAGE=vllm/vllm-openai:v<X.Y.Z>-ubuntu2404`,
  `ARG BASE_IMAGE_DIGEST=vllm/vllm-openai@sha256:<digest>`, and the "Verified
  against base" header (vLLM, flashinfer, torch versions).
- `src/mjolnir-cli/mjolnir_cli/config.py`: `VLLM_VERSION = "<X.Y.Z>"` — this
  moves `DEFAULT_IMAGE` to `narandill/mjolnir-vllm:v<X.Y.Z>`. CI (`docker-image.yml`)
  refuses to publish if the two disagree.

### 3. Build — the patch gate
```bash
mjolnir image build --tag narandill/mjolnir-vllm:v<X.Y.Z>
# = docker build -f docker/vllm-thor/Dockerfile -t … .   (repo root is the context)
```
- Exit 0 → every patch applied + verified, `AttentionBackendEnum["MJOLNIR"]`
  resolves to `mjolnir.backend.MjolnirAttentionBackend`. Continue.
- **Failure → a patch drifted.** Read the build log, hand re-adapt that patch
  in `docker/vllm-thor/patches/` (never a raw `git apply` of upstream diffs),
  keep its `apply_patches.py` / `verify_patches.py` / `PATCHES.md` entries
  current, and rebuild. The MJOLNIR patch only anchors on the last enum member
  in `vllm/v1/attention/backends/registry.py`; if upstream added members, move
  the anchor.

### 4. Canaries + the kernel (fresh containers — no server needed)
```bash
mjolnir image gates --image narandill/mjolnir-vllm:v<X.Y.Z>   # sm_110 gate probes
mjolnir verify correctness --image narandill/mjolnir-vllm:v<X.Y.Z>
mjolnir verify vllm --image narandill/mjolnir-vllm:v<X.Y.Z>   # MJOLNIR backend inside vLLM
```
Everything must PASS before the tag is considered real.

### 5. Serve + gated e2e smoke (needs the GPU — clean-window rules apply)
- **Never** stop/restart the live vLLM server to make room (the GPU is shared —
  `AGENTS.md`, critical rules). Schedule in a clean window via `mjolnir gate`,
  or inspect with `--dry-run` first:
```bash
mjolnir serve up --image narandill/mjolnir-vllm:v<X.Y.Z> --dry-run
mjolnir gate --once
mjolnir verify bench --image narandill/mjolnir-vllm:v<X.Y.Z>     # kernel headline data
mjolnir bench ab --image narandill/mjolnir-vllm:v<X.Y.Z>         # Mjolnir vs FI vs FA2, end to end
```
- A bump that shifts numbers starts a **new comparison epoch**: rows keep their
  image tag; the README picks the latest headline runs.

### 6. Switch the default
- Check the state file: a remembered image (`~/.mjolnir-state.json`, or
  `mjolnir image list` → the ● row) beats `DEFAULT_IMAGE` — `mjolnir image use
  narandill/mjolnir-vllm:v<X.Y.Z>` (or drop the `"image"` key).
- `uv run mjolnir plot --readme` so the README badges and tables follow.

### 7. Record + commit
- If any patch was re-adapted: record the old→new base, per-patch deltas and
  gate outputs on the patch's `PATCHES.md` note and in the commit message.
- Commits grouped by concern: patch re-adaptation(s) / Dockerfile + `VLLM_VERSION` /
  data + re-rendered docs. Merging to `main` publishes the image.

## Hard rules
- The gate is `verify_patches.py` inside the build — never bypass it.
- Patch re-adaptation is hand-edited inside `docker/vllm-thor/patches/`,
  registered in `apply_patches.py` `PATCH_ORDER`, with a `verify_patches.py`
  sentinel and a `PATCHES.md` entry.
- Any GPU measurement goes through the clean-window gate
  (`src/mjolnir-cli/mjolnir_cli/gate.py`); wall-clock comparisons only in the
  same gated window (see `docs/methodology/benchmarking.md`).
