# AGENTS.md — working on the Mjolnir repo

This repo is **the Mjolnir decode-attention kernel for the NVIDIA Jetson AGX
Thor** (`sm_110`, 20 SMs) and everything around it: the vLLM image it ships in
(`narandill/mjolnir-vllm`), the `MJOLNIR` vLLM attention backend, and the
`mjolnir` CLI that serves, verifies and benchmarks it. Every result in the
README and `docs/kernel/` is rendered from committed raw data. This file is the
operational doc: critical rules, layout, how to work. Read it before doing any
work here.

## Critical rules (read first)

### The GPU is shared — never kill the server, wait for idle
The Thor GPU is shared with a **live vLLM server that may be serving the LLM
the agents run on**. When doing anything on the GPU:
- **NEVER** stop / restart / kill / rebuild the vLLM server (or any container
  you did not start) to "clean" the GPU.
- **Wait for a clean window** — vLLM running + waiting requests = 0 for
  several consecutive samples — before measuring. The gate is
  `src/mjolnir-cli/mjolnir_cli/gate.py` (`CleanWindow`, `wait_for_idle`,
  `server_load`); **use it, don't re-implement it.** Every `mjolnir verify`
  stage and `mjolnir bench` gates itself (`--gate`, default on);
  `mjolnir gate --once` shows the state.
- **ncu** (base clocks, cold L2, kernel replay) is the instrument for kernel
  claims; wall clock is only for **ratios measured in the same run**.
- On this box a single result moves about **±7%** between runs (`NOISE_PCT`):
  compare legs measured in the same run, and require a win across L.
- **Native vLLM baselines (FlashInfer, FA2) are measured only on the official
  `vllm/vllm-openai` image**, never on ours — the Mjolnir image patches
  FlashInfer paths too. `mjolnir verify` runs reference legs there
  (`MJOLNIR_REF_IMAGE` overrides); e2e charts ignore native configs served
  from a Mjolnir image.

### Every number ships with its raw artifact
- Kernel results: `mjolnir verify <stage>` → one `mjolnir.verify/1` JSON per run
  in `benchmarks/raw/verify/<stage>/<run_id>.json` (`"default": true` only when
  every parameter is the stage default — only those feed headline charts).
- End-to-end results: `mjolnir bench perf|ab` → `benchmarks/raw/perf-<ts>/` +
  one row per run in `benchmarks/history.jsonl`.
- `README.md`, `docs/kernel/*.md` and `assets/benchmarks/` are **generated**
  (`uv run mjolnir plot --readme`) from `README.md.j2` / `docs/kernel/*.md.j2` and
  the raw data. Never hand-edit them; CI fails if they don't reproduce.

### The page-cache clear
`mjolnir_cli/cachectl.py` clears the host page cache (`sysctl -w
vm.drop_caches=3`) before `serve up` and after `serve down`, gated on a
*targeted* NOPASSWD grant (`mjolnir grant-permissions` installs only that
rule; `mjolnir drop-caches` clears on demand). Without the grant the clear is
skipped with a hint — nothing blocks on a password.

## Current state

- **The kernel** (`src/mjolnir/`, PyPI `mjolnir-kernel`): CuTe-DSL paged decode
  on `tcgen05` tensor cores — occupancy-pinned split-KV plan (one wave of
  2 CTAs × 20 SMs), dynamic fp8 Q, TMA KV ring with L2 evict-first, TMEM
  accumulators, softmax warpgroups, LSE merge kernel; q_len ≤ 1 + spec tokens;
  full CUDA graphs. Harness default: **e2m3 (fp6) KV on head-major (HND) pages,
  2-stage ring for fp6 at 2 CTAs/SM**; vLLM serves **fp8 KV on HND pages** (no
  vLLM fp6 cache-write path yet). Results: README + `docs/kernel/`.
- **The backend**: `vllm serve <model> --attention-backend MJOLNIR` on the image
  (enum patch `thor-mjolnir-attention-backend.patch`); `--attention-backend
  CUSTOM` on stock vLLM 0.31.0 (the `vllm.general_plugins` entry point
  `mjolnir = "mjolnir:register"` fills both slots). Prefill / mixed steps run on
  the fallback (`MJOLNIR_ATTN_FALLBACK`, FlashInfer). Knobs: `MJOLNIR_ATTN_*`
  (legacy `THOR_T5_*` honored) — `src/mjolnir/README.md`.
- **The image**: `narandill/mjolnir-vllm:v0.31.0` on
  `vllm/vllm-openai:v0.31.0-ubuntu2404`; 12 vLLM patches + 1 FlashInfer patch
  (#55390 went upstream in 0.31.0),
  applied and verified at build time; the kernel package installed from
  `src/mjolnir`. Tags follow the **upstream vLLM version**
  (`v<vllm>`, `v<vllm>-<sha>`, `latest`), built by `docker-image.yml`.
  Inventory: `docker/vllm-thor/PATCHES.md`.
- **Configs** (`configs/Qwen/Qwen3.8-27B/`): `NVFP4_MJOLNIR` (default, Mjolnir),
  `NVFP4` (FlashInfer), `NVFP4_FA2` (FlashAttention 2, bf16 KV — stock vLLM's
  choice on Thor without FlashInfer), `NVFP4_FA4hd256` (patched FA4).
- **Autoresearch**: `mjolnir autoresearch init|run|evaluate|ledger|status|export`
  + OpenCode agents in `.opencode/agent/` (orchestrator `autoresearch`;
  `arxiv-researcher`, `web-researcher`, `python-engineer`, `debugger`). The loop
  is `research/program.md`; every experiment is a ledger row.
- **Retired**: the FA4 GEMV decode kernel and its tooling were removed; the
  last snapshot is in git history at
  [`gemv-archive:deprecated/gemv/`](https://github.com/thor-arena/mjolnir/tree/gemv-archive/deprecated/gemv).
- **Side stacks**: `mjolnir litellm …` (LiteLLM proxy, `docker/litellm/`) and
  `mjolnir perses …` (Perses + Prometheus dashboards, `docker/perses/`), both
  independent of the vLLM server; user files under `~/.local/share/mjolnir/`.
  `mjolnir hw setup` provisions a fresh Thor (`scripts/hw/setup-thor.sh`).

## Repo layout

- `src/mjolnir/` — **the kernel package** (`mjolnir-kernel`, import `mjolnir`):
  - `mjolnir/t5_decode.py` — the CuTe-DSL kernel, split-KV merge, standalone
    harness (`python -m mjolnir.t5_decode --help`) and `emulate_decode`, the
    bit-faithful correctness reference. The single source of truth for the
    kernel; `mjolnir verify` mounts it over the image's copy.
  - `mjolnir/runner.py` — plan buffers, compile cache, launch, KV views (no vLLM).
  - `mjolnir/backend.py` — the vLLM backend / builder / impl / metadata.
  - `mjolnir/__init__.py` — `register()` (the plugin entry point), `BACKEND_NAME`.
- `src/mjolnir-cli/` — **the CLI** (`mjolnir-cli`, import `mjolnir_cli`, command
  `mjolnir`; typer):
  - `cli.py` (commands), `config.py` (defaults: `DEFAULT_QUANT`, `IMAGE_REPO`,
    `VLLM_VERSION`, `BACKEND_LABELS`; `RepoLayout`; bundled resources),
    `dockerctl.py` (serve), `gate.py`, `cachectl.py`, `benchy.py` +
    `history.py` (e2e), `tasks.py` (canary tasks for `image gates`),
    `stack.py` / `litellm.py` / `perses.py`, `thor.py` (hw), `picker.py`.
  - `verify/` — `mjolnir verify`: `stages.py` (every stage, `ALL_LS`, the
    correctness matrix), `legs.py` (Mjolnir / FlashInfer / FA2 legs and ncu
    filters), `docker.py` (one container per leg), `ncu.py` (summary parsing),
    `results.py` (the `mjolnir.verify/1` schema), `report.py`, `cli.py`.
  - `drivers/` — the FlashInfer (NHD/HND) and FA2 benchmark drivers run in the
    image.
  - `charts.py` + `theme.py` (every chart; the design system), `readme.py`
    (README/docs rendering: tables, findings, `chart()`), `interactive.py`
    (the Plotly site), `autoresearch.py`.
  - `hatch_build.py` bundles `configs/`, `docker/litellm`, `docker/perses`,
    `scripts/hw` into the wheel so the CLI works outside a checkout.
- `tests/cli/` (pytest, runs in CI) and `tests/kernel/` (needs CUDA + CUTLASS
  DSL; runs on a Thor via `mjolnir verify vllm`).
- `docker/vllm-thor/` — the image: `Dockerfile` (repo-root build context),
  `patches/`, `apply_patches.py` + `verify_patches.py` (the build gate),
  `PATCHES.md`, canary scripts.
- `configs/<vendor>/<model>/<quant>.yaml` — serving configs; user-local configs
  in `~/.local/share/mjolnir/configs/` (`$MJOLNIR_CONFIGS_DIR`).
- `benchmarks/` — `raw/verify/` (committed kernel raws), `raw/perf-*` (e2e raws),
  `history.jsonl`, `logs/` (untracked).
- `assets/benchmarks/` — generated charts (`*.png`, `*-dark.png`, `manifest.json`).
- `README.md.j2`, `docs/kernel/*.md.j2` — the templates.
- `docs/` — case study (`kernel/`), methodology, FAQ, stack research; index
  `docs/README.md`.
- `research/` — autoresearch program, ledger (untracked while live), exported runs.
- `.opencode/` — agents (`agent/`), the `/autoresearch` command, skills
  (`kernel-iteration`, `update-base-image`).
- `.github/workflows/` — `ci.yml`, `docker-image.yml`, `release-kernel.yml`,
  `release-cli.yml`, `pages.yml`.

## How to work

### Kernel: verify and benchmark — use `mjolnir verify`, don't hand-roll `docker run`
```
mjolnir verify list                       # stages + saved runs
mjolnir verify correctness [--quick]      # vs the emulator
mjolnir verify sweep                      # Mjolnir vs FlashInfer vs FA2 (ncu)
mjolnir verify hndsweep|kvfmt|quant|ring|prefetch|pdl|ab|fp6prof|timing
mjolnir verify vllm                       # the backend inside vLLM
mjolnir verify bench | all                # every stage once (+ correctness)
mjolnir verify <stage> --dry-run          # the docker commands, nothing run
mjolnir verify show <stage>               # the latest saved run
```
Default L for every stage and chart: **128 4096 8192 16384 32768 65536 131072
262144**. `--ls` narrows a run (saved as non-default). Logs:
`benchmarks/logs/verify-<run_id>/`. Workflow:
`.opencode/skills/kernel-iteration/SKILL.md`.

### Charts, README, site
```
uv run mjolnir plot --readme              # charts + README.md + docs/kernel/*.md
uv run mjolnir plot --site _site          # the interactive page (pages.yml publishes it)
```
New chart → a function in `charts.py` using `theme.py` (series colors: Mjolnir
orange, FlashInfer blue, FA2 aqua — never ad-hoc colors; light + dark variants),
registered in `render_all`; new table → `readme.py` `TABLES`; reference both from
the template. Sentences that state a result come from `readme.py` findings,
not hand-typed numbers.

### End to end
`mjolnir bench e2e` (official vLLM image vs the Mjolnir image: decode, prefill,
TTFT, ITL — feeds the README infographic; `bench compare` re-prints it);
`mjolnir bench perf` (gated llama-benchy, raw JSON + server log + a history row);
`mjolnir bench ab` (default legs: `NVFP4` on the official image vs `NVFP4_MJOLNIR`; legs
are `<config>` or `<image>:<config>`) — **it restarts the server per leg**: only on
purpose, in a window you own.

### Building the image
```
mjolnir image build --tag <tag>   # docker build -f docker/vllm-thor/Dockerfile -t <tag> .
mjolnir image gates               # sm_110 gate-probe canaries (fresh container)
```
The build fails if a patch stops applying — the signal the base moved. Bumping:
`.opencode/skills/update-base-image/SKILL.md` (bump `BASE_IMAGE` **and**
`VLLM_VERSION` in `config.py` together).

### Autoresearch
```
mjolnir autoresearch init <tag>   # branch autoresearch/<tag> + baseline row
mjolnir autoresearch run          # OpenCode orchestrator; never stops by itself
mjolnir autoresearch ledger | status | export
```
Agents may change only `src/mjolnir/mjolnir/t5_decode.py`; the keep/discard rule
is in the harness (> 2% geomean gain over the best kept, correctness required).

### Releases
Three independent versions: `mjolnir-kernel` (tag `mjolnir-kernel-v<ver>`,
version in `src/mjolnir/mjolnir/__init__.py`), `mjolnir-cli` (tag
`mjolnir-cli-v<ver>`, `src/mjolnir-cli/mjolnir_cli/__init__.py`) — both via PyPI
trusted publishing — and the image (vLLM version, on push to `main`).

## Open work
- [ ] **Headline kernel data**: `mjolnir verify bench` on the current kernel,
      commit `benchmarks/raw/verify/**`, re-render (the README shows "pending"
      for stages without a default run).
- [ ] **End-to-end**: `mjolnir bench e2e` (native FlashInfer on the official
      vLLM image vs Mjolnir) — restarts the server per leg; schedule deliberately.
- [ ] **fp6 KV in vLLM**: a cache-write path for e2m3 so serving gets the
      harness's fp6 win (`docs/kernel/kvfmt.md`, `quant.md`).
- [ ] **fp6 byte-vs-time gap**: 64-byte L2 requests (`docs/kernel/fp6prof.md`).
- [ ] Fuse the split-KV merge into the last CTA per sequence.
- [ ] fp4 with finer scales (per-channel K folded into Q, per-token V into P).

### FA4 + hd256 + FP8 KV (prefill) — blocked upstream, plan to port later
FA4's dedicated hd256 kernel is bf16-only upstream (asserts
`descale_tensors is None`), and vLLM downgrades FA4→FA2 for hd256 + any
quantized KV — so **FlashInfer remains the only FP8-KV path for the hd256
layers** (FA4+FP8 for hd≤128 *is* proven on Thor —
`docs/thor-stack/fa4-fp8kv-sm110.md`). Depth:
`docs/thor-stack/fa4-hd256-route-trace.md` (kernel/policy trace),
`docs/research/fa4-hd256-upstream-status.md` (upstream status).

Plan (as probe-gated patches, same pattern as the shipped ones):
1. Wait for (or request early access to) the upstream FP8-descale branch for
   the hd256 kernel + the paged-kv (TMA, page-128) branch; varlen is the long
   pole — check RFC #2456 status before investing.
2. Port the FP8-descale plumbing into the vendored
   `sm100_hd256_2cta_fmha_forward.py` (transcribable from the generic kernel
   `flash_fwd_sm100.py` `DescaleTensors` path, ~200–400 LOC) + relax the
   `descale_tensors is None` assert + widen the arch assert.
3. Widen vLLM's policy gates (`fa_utils.py` "quantized KV" reason, the
   `family(110)` term in `flash_attn_supports_kv_cache_dtype`).
4. sm_110 bring-up: TMEM table entry, register/tile tuning on real Thor
   silicon (all upstream tuning is B200-derived), fp8 numerics validation.
5. Config side: `block_size=128` (TMA path) for the hd256 kernel; verify the
   kernel's other forbids (no pack_gqa, no SplitKV>1, no score_mod) against
   GQA 24:4 + paged.
6. Probe-gated patch + end-to-end A/B vs the FlashInfer baseline before any
   workload switch.

## Upstream tracking

- Dao-AILab/flash-attention RFC [#2456](https://github.com/Dao-AILab/flash-attention/issues/2456)
  (owners @Johnsonms, @tzadouri):
  - FA4 hd256 fwd+bwd merged ([#2412](https://github.com/Dao-AILab/flash-attention/pull/2412), Apr 2026).
  - **FP8 input (descaled KV) for hd256: "🔨 code complete, perf to be improved"** —
    written, unmerged, not public on any branch/PR (verified 2026-09-21).
    Est. merge: Q4 2026.
   - Merged since the RFC: exp2-emu (#2488), paged-kv TMA (#2489),
     seqused_k/q (#2810), sm_110 arch gating (#2590 — the 2CTA kernels *do* run
     on sm_110; only a software postprocess bug, #2491).
   - **hd256 SplitKV fwd merged** ([#2916](https://github.com/Dao-AILab/flash-attention/pull/2916)
     + refactor [#2917](https://github.com/Dao-AILab/flash-attention/pull/2917),
     2026-09-25, squash `e9cf2c1`): static splits on the dedicated 2CTA fwd
     kernel (LSE partials, empty-split safe, varlen scheduler, paged+split
     tests; 2.0–3.9× decode on GB300). The new CTA-count heuristic is gated
     `arch // 10 in [10, 11]` (sm_110 in scope) but auto-disables on the
     20-SM Thor for our decode shapes (48 clusters > 20 SMs → 0 splits) —
     corroborates our in-tree NO-WIN. NOT the FP8-descale work (still private
     per #2456), and NOT in vLLM 0.30.0's or 0.31.0's vendored FA4 (pre-#2916), so the fa4
     patch group is unaffected. Depth:
     `docs/research/fa4-hd256-upstream-status.md` (2026-09-26 update).
   - Open: persistent-cluster scheduler (the 20-SM small-batch lever),
     sliding-window (#2749), hd256 backward + seqused (#2891), hd512 (#2877).
- FlashInfer: no hd256+FP8 work; XQA sm_110 gap open, unassigned
  ([#2522](https://github.com/flashinfer-ai/flashinfer/issues/2522)).
- vLLM: [#55366](https://github.com/vllm-project/vllm/pull/55366) (include SM110
  in FA4 auto-selection — we currently force `flash_attn_version=4`),
  [#54705](https://github.com/vllm-project/vllm/pull/54705) (fp8 e4m3 KV cache
  with per-tensor scales on SM100), [#55196](https://github.com/vllm-project/vllm/issues/55196)
  (open RFC questioning the FP8-KV memory win on GDN hybrids — our exact model
  class).
- Draft-decode cudagraphs:
  - vLLM [PR #53383](https://github.com/vllm-project/vllm/pull/53383) (open):
    triton verify builders bake `seq_lens=1` at cudagraph capture — same bug
    class as our c1 regression, different kernel. Watch for merge.
  - vLLM [#55581](https://github.com/vllm-project/vllm/issues/55581) +
    [PR #56289](https://github.com/vllm-project/vllm/pull/56289) (in progress):
    `get_cudagraph_support()` head-count mismatch can kill/limit draft
    cudagraphs — directly relevant to the DSpark draft CG grant.
  - vLLM [#49547](https://github.com/vllm-project/vllm/issues/49547): FlashInfer
    native decode caps at UNIFORM_SINGLE_TOKEN_DECODE → PIECEWISE downgrade
    under spec-decode.
  - FlashInfer [PR #3871](https://github.com/flashinfer-ai/flashinfer/pull/3871)
    (merged Jul 2026): graph-safe uniform multi-token decode; design statement
    confirms frozen plan_info depends only on (batch, total_rows, uniform_q_len)
    — the property our c1 regression study relies on
    (`docs/draft-cudagraph/c1-regression-study.md`).

## Conventions
- **New Thor patches** go in `docker/vllm-thor/patches/` (hand-adapted, **not**
  a raw `git apply`), registered in `apply_patches.py` `PATCH_ORDER`, with a
  `verify_patches.py` sentinel and a `PATCHES.md` entry. The **probe-gated
  pattern** (active only on sm_110, passive until a config opts in) is the norm.
- **Kernel changes** go behind a harness flag first; a default flips only with a
  same-run A/B win across L. Never change `emulate_decode` to make a test pass.
- **Bench discipline:** gate via `gate.py`, raw JSON first, charts second; no
  prose-only results.
- **Lint:** `uvx ruff@0.14.14 check src/ tests/` + `format` (pre-commit runs it).
- **Commit style:** short, imperative, grouped by concern (see `git log`).

## Pointers
- The kernel case study: `docs/kernel/README.md`.
- Kernel package + env knobs: `src/mjolnir/README.md`.
- Patch inventory: `docker/vllm-thor/PATCHES.md`.
- Measurement discipline: `docs/methodology/benchmarking.md`.
- Docs index: `docs/README.md`; FAQ: `docs/faq/index.md`.
- The retired GEMV kernel (git history): [`gemv-archive:deprecated/gemv/`](https://github.com/thor-arena/mjolnir/tree/gemv-archive/deprecated/gemv).
