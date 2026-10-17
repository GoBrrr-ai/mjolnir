<!-- Use for new Thor patches or a base-image bump. Conventions:
     CONTRIBUTING.md "Add a Thor patch", docker/vllm-thor/PATCHES.md, and for
     base bumps .opencode/skills/update-base-image/SKILL.md -->

## The patch

- file: `docker/vllm-thor/patches/<name>.patch`
- [ ] registered in `apply_patches.py` `PATCH_ORDER` (order matters — dependencies)
- [ ] sentinel added to `verify_patches.py`
- [ ] `PATCHES.md` row added (what it fixes + why hand-adapted if applicable)
- probe-gated (active only on sm_110, passive elsewhere): yes / no — reason if not

## The gate

- [ ] `mjolnir image build` passes — the build applies and verifies the whole stack; a patch that stops applying fails the build
- [ ] if `PATCHES.md` changed: re-rendered with `uv run mjolnir plot --readme` (the Docker Hub description carries the patch table)

## Base bump (if any)

- [ ] `BASE_IMAGE` (Dockerfile) and `VLLM_VERSION` (`config.py`) bumped together
- [ ] image tag follows the new vLLM version (`narandill/mjolnir-vllm:v<new>`)
