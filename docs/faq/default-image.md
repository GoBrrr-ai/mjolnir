# Changing the default vLLM image

**Story:** "I want `mjolnir serve up` / `mjolnir bench …` to use *this*
image from now on (not the baked-in default), and to see what's available
locally."

## Pick / remember the active image

```bash
mjolnir image             # arrow-key picker over every local docker image
mjolnir image list        # list with ● on the active one
mjolnir image use narandill/mjolnir-vllm:v0.30.0   # non-interactive
```

The choice is written to the state file (`~/.mjolnir-state.json` by default;
see [settings-precedence.md](settings-precedence.md)). Every later
`serve` / `bench` / `verify` / `autoresearch` picks it up automatically —
no flags needed.

## One-off (state untouched)

```bash
mjolnir serve up --image my/mjolnir-vllm:dev
MJOLNIR_IMAGE=narandill/mjolnir-vllm:v0.30.0-<sha> mjolnir verify sweep
```

Precedence is always: **CLI flag > state file > `$MJOLNIR_IMAGE` > baked-in
default** (`DEFAULT_IMAGE` in `src/mjolnir-cli/mjolnir_cli/config.py`): `narandill/mjolnir-vllm:v<vLLM version>`.

## Build a new one

```bash
mjolnir image build --tag my/mjolnir-vllm:dev
mjolnir image gates                    # sm_110 gate-probe canaries, fresh container
```

- `build` = `docker build -f docker/vllm-thor/Dockerfile -t <tag> .` (the repo root
  is the context: the image installs `src/mjolnir`); the build applies and
  verifies the patch stack — including the `MJOLNIR` backend enum patch — and **fails** if a patch stops applying
  (that's your "the base moved" signal —
  [base-image-bump.md](base-image-bump.md)).
- `gates` runs the sm_110 gate-probe canaries (GDN prefill, FA4 FP8-KV,
  draft-CG, hd256) against the image in a **fresh container — no server
  needed**, no impact on the live server.
- After a green build + canary: `mjolnir image use <new-tag>` makes it
  active. Keep the old tag around for A/B reference.

## Published images

[`narandill/mjolnir-vllm`](https://hub.docker.com/r/narandill/mjolnir-vllm) is
built by `.github/workflows/docker-image.yml`. Tags follow **the upstream vLLM
version** the image is built on — not the PyPI package versions:

| tag | what |
|---|---|
| `v0.30.0` | latest build on vLLM 0.30.0 |
| `v0.30.0-<sha>` | an immutable build from one commit |
| `latest` | the newest build from `main` |
