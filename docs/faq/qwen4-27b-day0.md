# Qwen4-27B on Thor — the 0-day runbook

Everything needed to go from "the checkpoint just dropped" to a served,
calibrated, benchmarked Mjolnir config — in one page per phase, with exact
commands. The background (architecture, evidence, per-item state) lives in
[qwen4-27b.md](qwen4-27b.md); this file is the do-it list. Nothing here
requires judgement that isn't written down.

**Where everything lives** (local scratch docs are on this machine only —
everything needed on the day is committed here or cited by hash):

- Code: the QSA host math
  [`src/mjolnir/mjolnir/arch/thor/qsa.py`](../../src/mjolnir/mjolnir/arch/thor/qsa.py)
  (IndexerSpec, scratch_plan, effective_seq_lens), the serving impl
  [`src/mjolnir/mjolnir/qsa_impl.py`](../../src/mjolnir/mjolnir/qsa_impl.py)
  (`build_impl_class`, the guard chain), the fused gather
  [`src/mjolnir/mjolnir/qsa_gather.py`](../../src/mjolnir/mjolnir/qsa_gather.py),
  the runner [`src/mjolnir/mjolnir/arch/thor/runner.py`](../../src/mjolnir/mjolnir/arch/thor/runner.py)
  (`T5DecodeRunner`: plan→run adjacency, `ctas_per_sm=2` default).
- Patches: [`docker/vllm-thor/PATCHES.md`](../../docker/vllm-thor/PATCHES.md)
  (20 vLLM-side + 2 FlashInfer-side; the three Qwen4 ones are
  `thor-qwen4-dense-model`, `thor-qwen4-topk-sm110`, `thor-qwen4-qsa-mjolnir`)
  — registered in `apply_patches.py`, sentineled in `verify_patches.py`.
- Tooling: [`scripts/qwen4_shape_lock.py`](../../scripts/qwen4_shape_lock.py)
  (the five-unknowns check), [`scripts/qwen4_dense_fixture.py`](../../scripts/qwen4_dense_fixture.py)
  (the fixture generator — the sm_110 canary).
- Local research (this machine/worktree only): `task/research/qwen4-27b-upstream-prep/` —
  `summary.md` (architecture + the recovered-PR story), `qwen4-27b-sm110-paper-tuning.md`
  (shape lock, bandwidth model, §3.8 integration point, §6 per-item checklist),
  `gpu-window-rooster.md` (every device-window result), `code-hunt.md`
  (the code recovery), `dumps/` (the recovered Intel tree + configs),
  `window-20261031/` (the evidence copies).
- Upstream: the SM120 cooperative_topk report
  [vllm#47266](https://github.com/vllm-project/vllm/issues/47266) (Issue 8 —
  same failure class, family-exclusion fix), the Thor warmup failure
  [vllm#56083](https://github.com/vllm-project/vllm/issues/56083) fixed by
  [#56084](https://github.com/vllm-project/vllm/pull/56084) and
  [#56102](https://github.com/vllm-project/vllm/pull/56102) (the mechanism:
  a 16-block cluster vs Thor's max 8); FA4 hd256+FP8 status
  [flash-attention#2456](https://github.com/Dao-AILab/flash-attention/issues/2456);
  SM110 FA4 auto-selection [vllm#55366](https://github.com/vllm-project/vllm/issues/55366);
  the MTP quantized-layer remap [vllm#55513](https://github.com/vllm-project/vllm/pull/55513);
  the FlashInfer XQA sm_110 gap
  [flashinfer#2522](https://github.com/flashinfer-ai/flashinfer/issues/2522).

## Proven state (all device-evidenced 2026-10-31, branch `feat/thor-kernel-improvements`)

| Capability | Evidence | Number |
|---|---|---|
| Dense qwen4_exp loads + prefills on sm_110 | fixture serve-smoke | 0 errors |
| Stock QSA decode on sm_110 | the `thor-qwen4-topk-sm110.patch` + smoke | byte-deterministic |
| Mjolnir QSA decode (scratch-gather) in real serving | `MJOLNIR_ATTN_QSA=1` smoke | `scratch-gather active (PS=800…)` |
| Gather vs dense (the QSA win) | `verify qsagain`, 16/16 | **13.3× / 11.7× @ 262K**, crossover ~16K |
| Indexer geometry (fp8 + e2m3) | `verify qsa`, 16/16 | 16→240 µs over L |
| Dense kernel vs references | default `verify sweep` | **1.24× vs FlashInfer** (baseline 1.2485 — no regression), 2.21× vs FA4 |
| Backend in real vLLM | `verify vllm`, 39/39 | \|Δlp\| ≤ 0.04 mean |
| Occupancy axis | `verify timing` | 2 CTAs/SM ≥ 1 at every L (the shipped default) |

Image: `narandill/mjolnir-vllm:v0.32.0` — 20 vLLM patches + 2 FlashInfer
patches, gates 20/20. Ratios are **reference ÷ ours** (>1 = Mjolnir faster,
±3% same-run noise — [methodology](../methodology/benchmarking.md)).

## Phase 0 — the config lands (30 min, no GPU)

```bash
# the five unknowns: layer count, GDN/full schedule, PLE, MTP depth, model_type
python scripts/qwen4_shape_lock.py /path/to/Qwen4-27B/config.json
```

Decision tree:

- **`model_type: qwen4_exp`** → everything below applies as-is.
- **A new model type** → the dense patch is necessary but NOT sufficient:
  a config-class mapping + an architectures registration are needed first
  (pattern: `docker/vllm-thor/patches/thor-qwen4-dense-model.patch`).
- The checkpoint **must list every layer in `mlp_only_layers`** (dense) and
  carry `intermediate_size` — if it doesn't, STOP: it's an MoE release and
  the dense campaign doesn't apply (the kernel work does; the QSA geometry
  probably still does).
- `indexer_budget != 2048` or `indexer_compress_ratio != 4` → update
  `IndexerSpec` in `mjolnir/arch/thor/qsa.py` (the constants are asserted,
  the change is mechanical) and re-run `verify qsa` + `qsagain`.
- If `ple_layer_ids` is non-empty → PLE is present: budget ~95 GiB
  **host-resident** (vLLM's pinned-host UVA offload, already in 0.32.0 —
  [#55513](https://github.com/vllm-project/vllm/pull/55513) is in too) and
  plan `--gpu-memory-utilization` around it — the smoke showed how tight a
  122 GiB unified board gets when a build runs alongside.
- `heads_per_ngram` / `ngram_vocab_size_base` → the 64K draft vocab: the
  `NVFP4_MJOLNIR_E2M3_LMH8_VT64K` pattern applies (see the
  [Qwen3.6 configs](../../configs/nvidia/Qwen3.6-35B-A3B/)), including the
  fp8 lm_head if the head isn't pre-quantized ([fp6-kv.md](fp6-kv.md)).

## Phase 1 — serving config + first serve (1 h)

```bash
# 1. the config dir (per [new-model-config.md](new-model-config.md)), then the .env switches:
#    MJOLNIR_ATTN_QSA=1                 # the Mjolnir QSA decode (impl-swap)
#    (the topk guard needs no switch — the patch makes `auto` safe on sm_110)
# 2. serve, stock backend first (the honest baseline):
docker run --gpus all ... narandill/mjolnir-vllm:v0.32.0 \
  vllm serve /path/to/Qwen4-27B --attention-backend FLASHINFER \
  --kv-cache-dtype fp8 --enforce-eager --gpu-memory-utilization 0.90
# 3. greedy sanity: finite tokens, deterministic ×2
# 4. re-serve with the MJOLNIR backend + the impl swap:
#    --attention-backend MJOLNIR (dense/full layers) + MJOLNIR_ATTN_QSA=1 (QSA layers)
# 5. only then: drop --enforce-eager and enable the speculative config
```

Known-shape facts to check against the log:

- **The attention page size will be forced to the mamba page size** (800 on
  the fixture: "Setting attention block size to 800 tokens…"). The Mjolnir
  QSA impl handles ANY pool page size (ratio-divisible) — the scratch stays
  at 128. If the impl announces a fallback, its reason prints once.
- The QSA layers print `MJOLNIR QSA decode: scratch-gather active …` on
  their first decode step — its absence means a fallback fired (reason
  printed once) or the env var didn't reach the engine.
- Prefill/mixed/verify steps run the stock Triton path **by design**
  (per-row selections); pure decode is the Mjolnir path.

## Phase 2 — calibration (1–2 h)

```bash
mjolnir calibrate --config <cfg>            # main-KV e2m3 scales (per layer/KV-head/channel)
mjolnir calibrate --config <cfg> --draft-vocab   # if the 64K draft vocab path applies
```

- Commit the scales next to the config (`kv_scales/2` format — per
  (layer, KV head, channel); a `/1` file is per-head and stale — see
  [fp6-kv.md](fp6-kv.md)).
- **Indexer-pool scales are phase 2**: fp8 per-tensor first, fp6 later —
  the calibration extension lands in `mjolnir calibrate` (checklist item 5,
  code pending at the time of writing).

## Phase 3 — the device ladder (2–3 h)

```bash
mjolnir verify kv-cache-size                 # tokens of KV per config (restarts per leg)
mjolnir verify qsa                           # indexer geometry at the REAL shapes
mjolnir verify qsagain --q-len 1 && mjolnir verify qsagain --q-len 5
mjolnir verify spec --drafts "1 2 3 4"       # acceptance curve — QSA caps drafts at K≤4
uv run mjolnir verify timing                 # re-confirm 2 CTAs/SM on the release unit
```

- `qsagain` compares against the **same-day dense baseline** — the release
  verdict is the RATIO, always same-run (±3% noise band).
- If the real geometry differs from the fixture's (more KV heads, another
  budget), `verify qsa` re-proves it in minutes.
- These stages refuse to run while a server is up —
  [clean-window-gate.md](clean-window-gate.md) is the discipline.

## Phase 4 — the A/B and any default flips (half a day)

```bash
mjolnir bench ab          # legs from src/mjolnir-cli/mjolnir_cli/legs.py
mjolnir bench e2e         # the README infographic
uv run mjolnir plot --readme
```

- **Both `bench ab` and `bench e2e` restart the server per leg** — schedule
  deliberately in a window you own.
- **A default flip (KV dtype, draft count, impl-swap on) requires a
  same-run A/B win across L** — never a single-cell win
  ([methodology](../methodology/benchmarking.md)).
- The infographic takes the highest decode-gain 1-request cell.

## If something fails on the day

| Symptom | Meaning | Action |
|---|---|---|
| `cooperative_topk … cluster misconfiguration` | the topk patch isn't in the image | rebuild, or `--sparse-indexer-topk-backend per_row` (fallback: `torch`) |
| `There is no module … mlp.down_proj` | the checkpoint is MoE (no `mlp_only_layers`) | not the dense release — reassess |
| The impl announces `verify/mixed batch` forever | never a pure-decode step | check the scheduler flags; verify steps are stock by design |
| `pool page size … not a multiple of the ratio` | a cache layout the gather can't index | report — the hybrid page size is expected to be ratio-divisible |
| Engine dies at startup, free memory swings | something else holds the board's RAM | `mjolnir drop-caches`, then isolate the window |
| Everything else | — | `MJOLNIR_ATTN_QSA` unset = pure stock behavior (the patch is inert) |

## Not on day 0

- CUDA graphs for the QSA path (eager-only v1 — the selection step syncs;
  the P2 in-loader gather is the fix).
- fp6-e2m3 QSA **main** KV (the impl reads fp8; the packed-cache layout
  needs the real checkpoint — note the GDN layers carry most KV bytes).
- Verify-batch (MTP) Mjolnir decode — per-row selections need either
  per-row scratch plans or a shared superset (kernel masking).
- The upstream `cooperative_topk` fix ([#56084](https://github.com/vllm-project/vllm/pull/56084)
  / [#56102](https://github.com/vllm-project/vllm/pull/56102)) — retire our
  patch at the bump that lands either.
