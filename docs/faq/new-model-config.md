# Adding a new model config

**Story:** "I want to serve/bench a model that isn't in the repo's
`configs/` — without forking the repo. The config should live on *my*
machine, under the project's home path."

## Where configs live

Two roots, both scanned by every `mjolnir` command that lists or resolves
configs:

| Root | Path | Use |
|---|---|---|
| **repo** (canonical) | `<repo>/configs/<vendor>/<model>/<quant>/config.yaml` | committed, reviewed, what the project ships |
| **user-local** (persisted at `~/`) | `~/.local/share/mjolnir/configs/<vendor>/<model>/<quant>/config.yaml` | your experiments; override with `$MJOLNIR_CONFIGS_DIR` |

Rules:

- Same layout in both: `<vendor>/<model>/<quant>/config.yaml`
  (e.g. `Meta/Llama-3.3-70B/NVFP4/config.yaml`).
- On a `(model, quant)` clash the **repo copy wins** — local never shadows a
  committed config (the repo one is canonical).
- Local configs are your personal, uncommitted state; the state file
  (`~/.mjolnir-state.json`) remembers which config is *active*, not the
  config contents.

## Create one

```bash
mkdir -p ~/.local/share/mjolnir/configs/Meta/Llama-3.3-70B/NVFP4
$EDITOR ~/.local/share/mjolnir/configs/Meta/Llama-3.3-70B/NVFP4/config.yaml
```

Minimal working config (copy a repo one — e.g.
`configs/Qwen/Qwen3.8-27B/NVFP4/config.yaml` — and change the model block; the
attention-backend / KV-cache / GDN sections are Thor-specific and usually
stay):

```yaml
model: Meta-Llama-3.3-70B-Instruct      # HF checkpoint to load
served-model-name: meta-llama/Llama-3.3-70B   # what the API answers to

dtype: bfloat16
attention-backend: flashinfer
kv-cache-dtype: fp8
max-model-len: 32K
gpu-memory-utilization: 0.88
# …same serving params as the closest existing config…
```

The two fields that matter to the CLI:

- `model` — the HF checkpoint vLLM loads (downloads into
  `~/.local/share/mjolnir/models/` on first launch).
- `served-model-name` — the name `mjolnir bench perf` targets in the
  OpenAI API. If you omit it, the `model:` field is used (that's vLLM's
  default served name).

## Use it — the whole flow

```bash
mjolnir model list          # it appears now, under the 'local' column
mjolnir model use Meta/Llama-3.3-70B NVFP4     # make it active (state file)
mjolnir serve up            # serves from it: the config dir is mounted into
                             # the container at /configs/Meta/Llama-3.3-70B/NVFP4
mjolnir bench perf          # benchmarks it (targets the served-model-name)
```

`serve up --dry-run` is the no-risk way to check the wiring: you should see
`-v …/configs/Meta/Llama-3.3-70B:/configs-local/Meta/Llama-3.3-70B:ro` and
`--config /configs-local/Meta/Llama-3.3-70B/NVFP4/config.yaml` in the printed
command. A local config's whole model dir is mounted at
`/configs-local/<vendor>/<model>` (beside the read-only `/configs`, which keeps
the repo's chat templates reachable), so its model-level files —
`kv_scales.json`, `draft_vocab.json` — reach the container the same way as a
repo config's; `{configs}` in its `.env` expands to that dir.

Promote a good local config to the repo when it's worth sharing: copy the
config dir to `<repo>/configs/<vendor>/<model>/<quant>/` and commit it — the
local copy stops shadowing anything and the repo one becomes canonical.

## Env switches: the `.env` sidecar

Some switches are environment variables, not `vllm serve` flags — the fp8
lm_head, the drafter's trimmed vocabulary, Mjolnir kernel knobs. Put them in a
`.env` file inside the config dir:

```bash
# configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/.env
MJOLNIR_LM_HEAD=fp8
MJOLNIR_DRAFT_VOCAB={configs}/draft_vocab.json   # {configs} = the model's config directory inside the container
MJOLNIR_DRAFT_VOCAB_SIZE=65536
```

`mjolnir serve`, `bench` and `verify` pass every `KEY=VALUE` line to the
container (`#` starts a comment). A config without a sidecar gets nothing
extra, so an experiment is one new config dir — the `config.yaml` can be a
copy of its parent's.
