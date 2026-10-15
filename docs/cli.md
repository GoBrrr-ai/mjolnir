# The `mjolnir` CLI reference

> Generated from the commands themselves by `mjolnir plot --readme` (`mjolnir_cli/clidoc.py`) — do not edit by hand. `mjolnir <command> --help` prints the same.

| command | what it does |
|---|---|
| [`mjolnir gate`](#mjolnir-gate) | The clean-window gate (vLLM request queue 0/0 × N consecutive samples). |
| [`mjolnir calibrate`](#mjolnir-calibrate) | Calibrate per-(layer, KV head, channel) K/V scales for the fp6 (e2m3) KV cache (or, with --draft-vocab, the drafter's vocabulary for VocabTrim). |
| [`mjolnir grant-permissions`](#mjolnir-grant-permissions) | Check for — and, if missing, install — the NOPASSWD sudo grant the kernel benches use to clear the page cache. |
| [`mjolnir drop-caches`](#mjolnir-drop-caches) | Clear the host page cache (`sysctl -w vm.drop_caches=3`) for a clean kernel-bench baseline. Uses the NOPASSWD grant (run `mjolnir grant-permissions` if it's missing); never prompts. |
| [`mjolnir plot`](#mjolnir-plot) | Render every chart (light + dark PNGs in assets/benchmarks/, from the latest default `mjolnir verify` run of each stage + the e2e history). --readme also renders README.md and docs/**/*.md from their .md.j2 templates; --site builds the interactive GitHub Pages site. Everything is computed from the committed raw data. |
| [`mjolnir history`](#mjolnir-history) | Bench results at a glance: decode throughput (t/s) per recent run. |
| [`mjolnir version`](#mjolnir-version) | Print the version. |
| [`mjolnir serve`](#mjolnir-serve) | Drive the vLLM container (the patched image). |
| [`mjolnir serve up`](#mjolnir-serve-up) | Start the vLLM server (custom image, config from --model/--config). |
| [`mjolnir serve down`](#mjolnir-serve-down) | Stop the vLLM container. |
| [`mjolnir serve status`](#mjolnir-serve-status) | Container state + /health + current queue load. |
| [`mjolnir serve logs`](#mjolnir-serve-logs) | Tail the vLLM container logs. |
| [`mjolnir bench`](#mjolnir-bench) | Benchmarks — gated, raw data, chartable. |
| [`mjolnir bench perf`](#mjolnir-bench-perf) | End-to-end perf bench (gated clean window, raw JSON, history log). |
| [`mjolnir bench ab`](#mjolnir-bench-ab) | Headline A/B: restart the server per leg, gated perf bench for each, one history row-set per leg. |
| [`mjolnir bench accuracy`](#mjolnir-bench-accuracy) | Model quality per serving config with tool-eval-bench: does fp6 KV / the fp8 lm_head keep the model's answers? |
| [`mjolnir bench e2e`](#mjolnir-bench-e2e) | Official vLLM image vs the Mjolnir image, end to end: decode and prefill throughput, time to first token and inter-token latency per context x concurrency. Serves each image in turn (the server is restarted per leg), benchmarks it with llama-benchy, then prints the comparison and re-renders the README (infographic + table). |
| [`mjolnir bench compare`](#mjolnir-bench-compare) | Official vLLM image vs Mjolnir image from the history — no new run. `--official-config NVFP4_DSPARK --config NVFP4_MJOLNIR_E2M3_DSPARK` compares the same drafter on both images. |
| [`mjolnir model`](#mjolnir-model) | Pick the active model / config. |
| [`mjolnir model use`](#mjolnir-model-use) | Remember the active model/config (bare `mjolnir model` or no args here = arrow-key picker over configs: repo configs/ + user-local ~/.local/share/mjolnir/configs). |
| [`mjolnir model list`](#mjolnir-model-list) | List every config (repo configs/ + the user-local dir; ● = the active one; the state file is printed below the table). |
| [`mjolnir image`](#mjolnir-image) | Pick the active vLLM image / build the patched one. |
| [`mjolnir image use`](#mjolnir-image-use) | Remember the active vLLM image (bare `mjolnir image` or no args here = arrow-key picker over local docker images). |
| [`mjolnir image list`](#mjolnir-image-list) | List local docker images (● = the active one). |
| [`mjolnir image build`](#mjolnir-image-build) | Build the Mjolnir vLLM image: the patch stack + the mjolnir-kernel package from this checkout (repo-root build context). |
| [`mjolnir image gates`](#mjolnir-image-gates) | Run the sm_110 gate-probe canaries against the image (fresh container, no server needed). |
| [`mjolnir verify`](#mjolnir-verify) | Kernel correctness + benchmarks (Mjolnir vs FlashInfer vs FA2), saved as raw data for the charts. |
| [`mjolnir verify correctness`](#mjolnir-verify-correctness) | Kernel vs the bit-faithful emulator: every layout, KV format, page size, q_len and Q dtype. |
| [`mjolnir verify sweep`](#mjolnir-verify-sweep) | The headline: Mjolnir vs FlashInfer vs FlashAttention 2 at every L (ncu, cold L2, base clocks). |
| [`mjolnir verify hndsweep`](#mjolnir-verify-hndsweep) | FlashInfer on NHD and HND pages vs Mjolnir (fp8 KV) on HND: the fair layout comparison. |
| [`mjolnir verify kvfmt`](#mjolnir-verify-kvfmt) | KV layouts x formats: fp8 / fp6 (e2m3, e3m2) / fp4 (e2m1) / nvfp4 on NHD / interleaved / HND. |
| [`mjolnir verify quant`](#mjolnir-verify-quant) | Accuracy per KV format: relative L2 error vs attention over unquantized KV. |
| [`mjolnir verify ring`](#mjolnir-verify-ring) | KV ring depth (TMA → MMA stages) at every L. |
| [`mjolnir verify prefetch`](#mjolnir-verify-prefetch) | TMA L2-prefetch distance sweep. |
| [`mjolnir verify pdl`](#mjolnir-verify-pdl) | Programmatic dependent launch on/off (wall clock). |
| [`mjolnir verify ab`](#mjolnir-verify-ab) | A/B any harness flag sets (ncu + wall clock). Default: evict-first x PDL. |
| [`mjolnir verify fp6prof`](#mjolnir-verify-fp6prof) | Full ncu profiles per KV format (bytes vs requests vs time, warp stalls). |
| [`mjolnir verify timing`](#mjolnir-verify-timing) | Harness-timed medians, 1 vs 2 CTAs/SM, fp8 and bf16 Q (no ncu). |
| [`mjolnir verify profile`](#mjolnir-verify-profile) | Profile a decode workload per config (torch profiler): graph replays, eager kernels, GPU idle, top kernels. |
| [`mjolnir verify spec`](#mjolnir-verify-spec) | MTP per config and draft count: tokens/step, per-position acceptance, decode t/s (restarts the server per point). |
| [`mjolnir verify kv-cache-size`](#mjolnir-verify-kv-cache-size) | KV cache capacity per config: serves each leg (restarting the server), reads vLLM's KV sizing, stops it. |
| [`mjolnir verify vllm`](#mjolnir-verify-vllm) | The vLLM plugin: runner tests + greedy MJOLNIR-vs-FLASHINFER smoke runs (checkout only). |
| [`mjolnir verify bench`](#mjolnir-verify-bench) | Every benchmark stage with default parameters (one headline run each), then re-render the charts. |
| [`mjolnir verify all`](#mjolnir-verify-all) | Correctness + every benchmark stage, then re-render the charts. |
| [`mjolnir verify show`](#mjolnir-verify-show) | Re-print the latest saved run of a stage (or every stage). |
| [`mjolnir verify list`](#mjolnir-verify-list) | Stages and their saved runs. |
| [`mjolnir autoresearch`](#mjolnir-autoresearch) | Autonomous kernel research: OpenCode agents propose, implement and test kernel changes; every experiment is scored by a fixed harness and recorded on the ledger. |
| [`mjolnir autoresearch init`](#mjolnir-autoresearch-init) | Create the research branch and score the starting kernel (the baseline row). |
| [`mjolnir autoresearch evaluate`](#mjolnir-autoresearch-evaluate) | Score the committed kernel: quick correctness + FlashInfer vs Mjolnir sweep; append the result and the fixed keep/discard decision to the ledger. |
| [`mjolnir autoresearch ledger`](#mjolnir-autoresearch-ledger) | Show the ledger (newest last). |
| [`mjolnir autoresearch status`](#mjolnir-autoresearch-status) | Best kept score, experiment counts, the current branch. |
| [`mjolnir autoresearch chart`](#mjolnir-autoresearch-chart) | Re-render research/progress.png from the live ledger (or the newest exported run). |
| [`mjolnir autoresearch run`](#mjolnir-autoresearch-run) | Launch the loop: an OpenCode session with the `autoresearch` agent. |
| [`mjolnir autoresearch export`](#mjolnir-autoresearch-export) | Snapshot the ledger into research/runs/&lt;tag&gt;/ (ledger.jsonl, ledger.tsv, SUMMARY.md) — commit that to keep the run's notebook with the repo. |
| [`mjolnir litellm`](#mjolnir-litellm) | The LiteLLM proxy stack (proxy + postgres + redis) — independent of the vLLM server: 'litellm down' leaves vLLM running, and 'serve down' leaves the proxy up. Reads your template from the litellm data dir, ~/.local/share/mjolnir/litellm/ (see docker/litellm/). |
| [`mjolnir litellm up`](#mjolnir-litellm-up) | Render the config from your template (in the litellm data dir, ~/.local/share/mjolnir/litellm/) with the currently selected model/config, and start the stack (proxy + postgres + redis). The vLLM server is reached via the host — it may be up or down. |
| [`mjolnir litellm down`](#mjolnir-litellm-down) | Stop the stack (the vLLM server keeps running; data kept by default, --purge wipes postgres + redis state). |
| [`mjolnir litellm status`](#mjolnir-litellm-status) | Container state + proxy liveliness + the rendered config path. |
| [`mjolnir litellm logs`](#mjolnir-litellm-logs) | Tail the proxy container logs. |
| [`mjolnir litellm config`](#mjolnir-litellm-config) | Render the template with the current model/config (no containers) — validates the variables and the YAML. |
| [`mjolnir perses`](#mjolnir-perses) | The Perses + Prometheus observability stack for the vLLM server — independent of the vLLM server: 'perses down' leaves the server running, and 'serve down' leaves the stack up. Prometheus scrapes the server's /metrics (host.docker.internal + the resolved port); Perses serves the two vLLM dashboards (vendored from vllm-project/vllm, docker/perses/). Data: ~/.local/share/mjolnir/perses/. |
| [`mjolnir perses up`](#mjolnir-perses-up) | Start the observability stack (prometheus + perses) with the two vLLM dashboards provisioned. The vLLM scrape target = the resolved server port (state/$MJOLNIR_PORT — the same one `mjolnir serve` uses; override with PERSER_VLLM_TARGET in the perses env file). The server may be up or down. |
| [`mjolnir perses down`](#mjolnir-perses-down) | Stop the stack (the vLLM server keeps running; data kept by default, --purge wipes the prometheus TSDB + the perses db). |
| [`mjolnir perses status`](#mjolnir-perses-status) | Container state + API health + the vLLM scrape target + the data dir. |
| [`mjolnir perses logs`](#mjolnir-perses-logs) | Tail a stack service's logs (default: perses). |
| [`mjolnir hw`](#mjolnir-hw) | Thor host pre-configuration — one-time, idempotent provisioning (sudo where needed): headless boot target, apt upgrade, Docker + NVIDIA default runtime, pip, jtop, swap, fan profiles (recommended + max installed, default selected), service cleanup, locked clocks, MAXN power mode. Run 'hw setup' before serving or benchmarking. |
| [`mjolnir hw setup`](#mjolnir-hw-setup) | Pre-configure the Thor host (port of the ~/thor HW setup script). |
| [`mjolnir hw status`](#mjolnir-hw-status) | Installed fan profiles and the selected default (read-only, no sudo). |

<a id="mjolnir-gate"></a>

## `mjolnir gate`

The clean-window gate (vLLM request queue 0/0 × N consecutive samples).

| option | description | default |
|---|---|---|
| `--port` | the server port (default: the active one) |  |
| `--once` | check now and exit | off |
| `--confirm` |  | `6` |
| `--timeout` | seconds before a step is given up | `3600.0` |


<a id="mjolnir-calibrate"></a>

## `mjolnir calibrate`

Calibrate per-(layer, KV head, channel) K/V scales for the fp6 (e2m3) KV cache
(or, with --draft-vocab, the drafter's vocabulary for VocabTrim).

Runs the config's model on fp8 KV in the Mjolnir image, records the
per-channel amax of every K/V written to the cache over a varied prompt set,
and writes `kv_scales.json` next to the config — what an
`kv-cache-dtype: fp6_e2m3` config is served with. Needs the GPU to
itself: refuses while the vLLM server is up (stop it with
`mjolnir serve down` in a window you own).

| option | description | default |
|---|---|---|
| `--model` | vendor/Model (default: the active one) |  |
| `--config` | the config to calibrate (default: the active one) |  |
| `--image` | the Mjolnir image |  |
| `--out` | default: the model's kv_scales.json next to the config dirs — kv_scales.json inside the config dir for a config whose own drafter (DSpark) runs on Mjolnir |  |
| `--prompts` | short prompts | `48` |
| `--long` | long-context prompts | `6` |
| `--max-model-len` |  | `32768` |
| `--draft-vocab` | instead: count the target's generated tokens -&gt; draft_vocab.json (VocabTrim, MJOLNIR_DRAFT_VOCAB) | off |
| `--dry-run` | print the docker command, run nothing | off |


<a id="mjolnir-grant-permissions"></a>

## `mjolnir grant-permissions`

Check for — and, if missing, install — the NOPASSWD sudo grant the
kernel benches use to clear the page cache.

Idempotent: a present grant is left untouched. The install adds only the
cache-clear command (`sysctl -w vm.drop_caches=3`) — never full sudo —
and asks for the sudo password exactly once.

| option | description | default |
|---|---|---|
| `--yes` / `-y` | don't prompt to confirm the install | off |


<a id="mjolnir-drop-caches"></a>

## `mjolnir drop-caches`

Clear the host page cache (`sysctl -w vm.drop_caches=3`) for a clean
kernel-bench baseline. Uses the NOPASSWD grant (run
`mjolnir grant-permissions` if it's missing); never prompts.


<a id="mjolnir-plot"></a>

## `mjolnir plot`

Render every chart (light + dark PNGs in assets/benchmarks/, from the
latest default `mjolnir verify` run of each stage + the e2e history).
--readme also renders README.md and docs/**/*.md from their .md.j2
templates; --site builds the interactive GitHub Pages site. Everything is
computed from the committed raw data.

| option | description | default |
|---|---|---|
| `--readme` | also render README.md from the README.md.j2 template (every table + number computed from the committed raw data) | off |
| `--readme-template` | template override (default: &lt;repo&gt;/README.md.j2) |  |
| `--readme-out` | output override (default: &lt;repo&gt;/README.md) |  |
| `--site` | also build the interactive benchmark site (index.html) into this directory |  |


<a id="mjolnir-history"></a>

## `mjolnir history`

Bench results at a glance: decode throughput (t/s) per recent run.

The raw substrate (every sweep, every cell) is committed in
benchmarks/history.jsonl — `--json` dumps the records.

| option | description | default |
|---|---|---|
| `--limit` | how many recent runs to show | `5` |
| `--json` / `-j` | emit the raw JSON records (for scripts) | off |


<a id="mjolnir-version"></a>

## `mjolnir version`

Print the version.


<a id="mjolnir-serve"></a>

## `mjolnir serve`

Drive the vLLM container (the patched image).

Sub-commands: [`mjolnir serve up`](#mjolnir-serve-up), [`mjolnir serve down`](#mjolnir-serve-down), [`mjolnir serve status`](#mjolnir-serve-status), [`mjolnir serve logs`](#mjolnir-serve-logs).


<a id="mjolnir-serve-up"></a>

## `mjolnir serve up`

Start the vLLM server (custom image, config from --model/--config).

| option | description | default |
|---|---|---|
| `--model` | the model (`vendor/Model`; default: the active one, `mjolnir model`) |  |
| `--config` | config quant name, e.g. NVFP4_FA4hd256 |  |
| `--image` | the vLLM image (default: the active one, `mjolnir image`) |  |
| `--port` | the server port (default: the active one) |  |
| `--wait` / `--no-wait` | wait for /health before returning | --wait |
| `--wait-timeout` |  | `1800.0` |
| `--dry-run` | print what would run, run nothing | off |


<a id="mjolnir-serve-down"></a>

## `mjolnir serve down`

Stop the vLLM container.

| option | description | default |
|---|---|---|
| `--model` | the model (`vendor/Model`; default: the active one, `mjolnir model`) |  |
| `--config` | the model config (default: the active one) |  |
| `--image` | the vLLM image (default: the active one, `mjolnir image`) |  |
| `--port` | the server port (default: the active one) |  |
| `--dry-run` | print what would run, run nothing | off |


<a id="mjolnir-serve-status"></a>

## `mjolnir serve status`

Container state + /health + current queue load.

| option | description | default |
|---|---|---|
| `--port` | the server port (default: the active one) |  |


<a id="mjolnir-serve-logs"></a>

## `mjolnir serve logs`

Tail the vLLM container logs.

| option | description | default |
|---|---|---|
| `--port` | the server port (default: the active one) |  |
| `--follow` / `-f` |  | off |
| `--tail` |  | `100` |


<a id="mjolnir-bench"></a>

## `mjolnir bench`

Benchmarks — gated, raw data, chartable.

Sub-commands: [`mjolnir bench perf`](#mjolnir-bench-perf), [`mjolnir bench ab`](#mjolnir-bench-ab), [`mjolnir bench accuracy`](#mjolnir-bench-accuracy), [`mjolnir bench e2e`](#mjolnir-bench-e2e), [`mjolnir bench compare`](#mjolnir-bench-compare).


<a id="mjolnir-bench-perf"></a>

## `mjolnir bench perf`

End-to-end perf bench (gated clean window, raw JSON, history log).

| option | description | default |
|---|---|---|
| `--runs` | measured runs per cell (llama-benchy --runs) | `6` |
| `--warmup-runs` |  | `2` |
| `--repeat` | independent gated sweeps (default: one sweep, 6 runs per cell) | `1` |
| `--depths` | context depths | `0,4096,8192,16384,32768,65536,131072` |
| `--concurrency` |  | `1,2,4` |
| `--pp` |  | `2048` |
| `--tg` |  | `128` |
| `--exact-tg` / `--no-exact-tg` | pin output length to --tg (kills EOS variance) | --exact-tg |
| `--gate` / `--no-gate` | wait for a clean window (vLLM queue 0/0) before measuring | --gate |
| `--confirm` | clean-window confirm samples | `3` |
| `--gate-timeout` |  | `3600.0` |
| `--label` | backend label for the history log |  |
| `--api-key` |  |  |
| `--model` | the model (`vendor/Model`; default: the active one, `mjolnir model`) |  |
| `--config` | the model config (default: the active one) |  |
| `--image` | the vLLM image (default: the active one, `mjolnir image`) |  |
| `--port` | the server port (default: the active one) |  |
| `--plot` / `--no-plot` | re-render the charts / README afterwards | --plot |


<a id="mjolnir-bench-ab"></a>

## `mjolnir bench ab`

Headline A/B: restart the server per leg, gated perf bench for each,
one history row-set per leg.

Each leg is a model config (quant), optionally model-qualified and/or
image-pinned: '&lt;config&gt;', '&lt;image&gt;:&lt;config&gt;', '&lt;model&gt;/&lt;config&gt;' or
'&lt;image&gt;:&lt;model&gt;/&lt;config&gt;'. The image carries the backend (kernel stack
is baked in); the config (configs/&lt;model&gt;/&lt;quant&gt;/config.yaml, repo or local)
carries the serving parameters. Bare configs use --model/--image.
Compare images on one model and/or models on one image. Charts
re-render at the end.

| option | description | default |
|---|---|---|
| `--backends` | comma-separated legs: each is '&lt;config&gt;' (the --model/state model, served from --image), '&lt;image&gt;:&lt;config&gt;', '&lt;model&gt;/&lt;config&gt;' or '&lt;image&gt;:&lt;model&gt;/&lt;config&gt;' — available: mjolnir model list | `vllm/vllm-openai:v0.31.0-ubuntu2404:NVFP4,NVFP4_MJOLNIR_E2M3_LMH8_VT64K` |
| `--depths` | context depths (tokens already in the KV) | `0,4096,8192,16384,32768,65536,131072` |
| `--runs` |  | `6` |
| `--repeat` |  | `1` |
| `--gate` / `--no-gate` | wait for a clean window (vLLM queue 0/0) before measuring | --gate |
| `--model` | the model (`vendor/Model`; default: the active one, `mjolnir model`) |  |
| `--image` | image for bare-config legs (default: $MJOLNIR_IMAGE / built-in default) |  |
| `--port` | the server port (default: the active one) |  |


<a id="mjolnir-bench-accuracy"></a>

## `mjolnir bench accuracy`

Model quality per serving config with tool-eval-bench: does fp6 KV /
the fp8 lm_head keep the model's answers?

Default per leg: the 69 standard tool-calling scenarios + the 28 Hard Mode
ones (`tool-eval-bench run --hardmode --seed 42 --timeout 600`). Anything
after `--` replaces that and is forwarded verbatim (the leg's --base-url is
added), e.g. `-- run --hardmode --no-think --trials 3` or
`-- --gsm8k --mmlu --ifeval --skip-tool-eval --gsm8k-limit 200`.
Results: benchmarks/raw/accuracy/&lt;run&gt;/accuracy.json (committed; the
tool's SQLite, reports and the server log stay next to it, untracked);
docs/kernel/accuracy.md is rendered from the newest result per config and
suite.

| option | description | default |
|---|---|---|
| `--backends` | comma-separated legs, as in 'bench ab': '&lt;config&gt;', '&lt;image&gt;:&lt;config&gt;', '&lt;model&gt;/&lt;config&gt;' or '&lt;image&gt;:&lt;model&gt;/&lt;config&gt;' (default: mjolnir_cli.legs) | `vllm/vllm-openai:v0.31.0-ubuntu2404:NVFP4,NVFP4_MJOLNIR,NVFP4_MJOLNIR_E2M3,NVFP4_MJOLNIR_E2M3_LMH8_VT64K` |
| `--restart` / `--no-restart` | serve each leg (restarts the server; the GPU must be yours) — --no-restart measures the running server as one leg (pass its config with --backends) | --restart |
| `--model` | the model (`vendor/Model`; default: the active one, `mjolnir model`) |  |
| `--image` | image for bare-config legs |  |
| `--port` | the server port (default: the active one) |  |
| `--dry-run` | print the legs and the tool-eval-bench command | off |
| `--plot` / `--no-plot` | re-render the charts / README afterwards | --plot |


<a id="mjolnir-bench-e2e"></a>

## `mjolnir bench e2e`

Official vLLM image vs the Mjolnir image, end to end: decode and prefill
throughput, time to first token and inter-token latency per context x
concurrency. Serves each image in turn (the server is restarted per leg),
benchmarks it with llama-benchy, then prints the comparison and re-renders
the README (infographic + table).

| option | description | default |
|---|---|---|
| `--official-image` | the upstream vLLM image (baseline) | `vllm/vllm-openai:v0.31.0-ubuntu2404` |
| `--official-config` | config served on the official image (stock backends only) | `NVFP4` |
| `--image` | the Mjolnir image | `narandill/mjolnir-vllm:v0.31.0` |
| `--config` | config served on the Mjolnir image | `NVFP4_MJOLNIR_E2M3_LMH8_VT64K` |
| `--model` | the model (`vendor/Model`; default: the active one, `mjolnir model`) |  |
| `--runs` | measured runs per cell | `6` |
| `--repeat` | independent gated sweeps per leg | `1` |
| `--depths` | context depths (tokens already in the KV); the comparison shows up to 32K | `0,4096,8192,16384,32768` |
| `--concurrency` |  | `1,2,4` |
| `--pp` | prompt tokens per request (prefill) | `2048` |
| `--tg` | generated tokens per request (decode) | `128` |
| `--gate` / `--no-gate` | wait for a clean window (vLLM queue 0/0) before measuring | --gate |
| `--port` | the server port (default: the active one) |  |
| `--only` | run one leg: 'mjolnir' or 'official'; the comparison pairs it with the other side's latest record |  |
| `--also` | more legs to measure after the pair, comma-separated: a config (Mjolnir image) or official:&lt;config&gt; / mjolnir:&lt;config&gt;; '' = none. They land in the history (bench compare --config &lt;cfg&gt;); the README table, charts and infographic stay on --config | `` |
| `--optional` | also measure the opt-in legs (legs.OPTIONAL: DSpark and DFlash2 on the official and the Mjolnir image) — hours per leg | off |
| `--dry-run` | print the serve commands, run nothing | off |
| `--allow-stale-image` | benchmark a Mjolnir image older than the checkout's kernel | off |


<a id="mjolnir-bench-compare"></a>

## `mjolnir bench compare`

Official vLLM image vs Mjolnir image from the history — no new run.
`--official-config NVFP4_DSPARK --config NVFP4_MJOLNIR_E2M3_DSPARK` compares
the same drafter on both images.

| option | description | default |
|---|---|---|
| `--config` | Mjolnir-side config (default: the Mjolnir backend) |  |
| `--official-config` | official-side config (default: the stock baseline, NVFP4) |  |
| `--json` / `-j` |  | off |


<a id="mjolnir-model"></a>

## `mjolnir model`

Pick the active model / config.

Sub-commands: [`mjolnir model use`](#mjolnir-model-use), [`mjolnir model list`](#mjolnir-model-list).


<a id="mjolnir-model-use"></a>

## `mjolnir model use`

Remember the active model/config (bare `mjolnir model` or no args
here = arrow-key picker over configs: repo configs/ + user-local
~/.local/share/mjolnir/configs).

Persisted to a small state file ($MJOLNIR_STATE, default
~/.mjolnir-state.json) — not an env file.

| option | description | default |
|---|---|---|
| `MODEL` | argument |  |
| `QUANT` | argument |  |


<a id="mjolnir-model-list"></a>

## `mjolnir model list`

List every config (repo configs/ + the user-local dir; ● = the active
one; the state file is printed below the table).

| option | description | default |
|---|---|---|
| `--json` / `-j` | emit JSON (for scripts) | off |


<a id="mjolnir-image"></a>

## `mjolnir image`

Pick the active vLLM image / build the patched one.

Sub-commands: [`mjolnir image use`](#mjolnir-image-use), [`mjolnir image list`](#mjolnir-image-list), [`mjolnir image build`](#mjolnir-image-build), [`mjolnir image gates`](#mjolnir-image-gates).


<a id="mjolnir-image-use"></a>

## `mjolnir image use`

Remember the active vLLM image (bare `mjolnir image` or no args
here = arrow-key picker over local docker images).

Persisted to the state file ($MJOLNIR_STATE, default
~/.mjolnir-state.json) — serve/bench pick it up from there.

| option | description | default |
|---|---|---|
| `IMAGE` | argument |  |


<a id="mjolnir-image-list"></a>

## `mjolnir image list`

List local docker images (● = the active one).


<a id="mjolnir-image-build"></a>

## `mjolnir image build`

Build the Mjolnir vLLM image: the patch stack + the mjolnir-kernel
package from this checkout (repo-root build context).

The default tag is always the canonical one (IMAGE_REPO:v&lt;vLLM version&gt;),
never the remembered active image — that may be an older build.

| option | description | default |
|---|---|---|
| `--tag` | image tag (default: narandill/mjolnir-vllm:v0.31.0 — the repo + the vLLM version the Dockerfile builds on) |  |
| `--dry-run` | print what would run, run nothing | off |


<a id="mjolnir-image-gates"></a>

## `mjolnir image gates`

Run the sm_110 gate-probe canaries against the image (fresh container,
no server needed).

| option | description | default |
|---|---|---|
| `--image` | the vLLM image (default: the active one, `mjolnir image`) |  |
| `--dry-run` | print what would run, run nothing | off |


<a id="mjolnir-verify"></a>

## `mjolnir verify`

Kernel correctness + benchmarks (Mjolnir vs FlashInfer vs FA2), saved as raw data for the charts.

Sub-commands: [`mjolnir verify correctness`](#mjolnir-verify-correctness), [`mjolnir verify sweep`](#mjolnir-verify-sweep), [`mjolnir verify hndsweep`](#mjolnir-verify-hndsweep), [`mjolnir verify kvfmt`](#mjolnir-verify-kvfmt), [`mjolnir verify quant`](#mjolnir-verify-quant), [`mjolnir verify ring`](#mjolnir-verify-ring), [`mjolnir verify prefetch`](#mjolnir-verify-prefetch), [`mjolnir verify pdl`](#mjolnir-verify-pdl), [`mjolnir verify ab`](#mjolnir-verify-ab), [`mjolnir verify fp6prof`](#mjolnir-verify-fp6prof), [`mjolnir verify timing`](#mjolnir-verify-timing), [`mjolnir verify profile`](#mjolnir-verify-profile), [`mjolnir verify spec`](#mjolnir-verify-spec), [`mjolnir verify kv-cache-size`](#mjolnir-verify-kv-cache-size), [`mjolnir verify vllm`](#mjolnir-verify-vllm), [`mjolnir verify bench`](#mjolnir-verify-bench), [`mjolnir verify all`](#mjolnir-verify-all), [`mjolnir verify show`](#mjolnir-verify-show), [`mjolnir verify list`](#mjolnir-verify-list).


<a id="mjolnir-verify-correctness"></a>

## `mjolnir verify correctness`

Kernel vs the bit-faithful emulator: every layout, KV format, page size, q_len and Q dtype.

| option | description | default |
|---|---|---|
| `--dtypes` | Q dtypes to test | `fp8 bf16` |
| `--match` | only cases whose label contains this | `` |
| `--quick` | the quick subset (one case per group; what autoresearch runs) | off |
| `--fail-fast` |  | off |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | per-container timeout (s); a hung kernel's container is killed | `180.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |


<a id="mjolnir-verify-sweep"></a>

## `mjolnir verify sweep`

The headline: Mjolnir vs FlashInfer vs FlashAttention 2 at every L (ncu, cold L2, base clocks).

`--batch 4 --q-len 4 --legs fi,mjolnir-fp8` measures the shape vLLM serves
with MTP=3 at 4 concurrent requests (saved as a non-default run).

| option | description | default |
|---|---|---|
| `--legs` | legs: fi, fi-hnd, fa2, mjolnir-fp8, mjolnir | `fi,fa2,mjolnir-fp8,mjolnir` |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--ls` | L values (default: 128 4096 8192 16384 32768 65536 131072 262144) |  |
| `--iters` | profiled iterations per point (ncu stages) | `50` |
| `--gate` / `--no-gate` | wait for a clean window when the live vLLM server is up | --gate |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | per-container timeout (s); a hung kernel's container is killed | `180.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |
| `--plot` / `--no-plot` | re-render charts + README after saving (checkout only) | --no-plot |
| `--page-size` | KV page size for every leg | `128` |
| `--batch` | sequences per step, each of L tokens (serving: 1-4) | `1` |
| `--q-len` | query tokens per sequence (MTP=3 verification: 4) | `1` |


<a id="mjolnir-verify-hndsweep"></a>

## `mjolnir verify hndsweep`

FlashInfer on NHD and HND pages vs Mjolnir (fp8 KV) on HND: the fair layout comparison.

| option | description | default |
|---|---|---|
| `--legs` |  | `fi,fi-hnd,mjolnir-fp8` |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--ls` | L values (default: 128 4096 8192 16384 32768 65536 131072 262144) |  |
| `--iters` | profiled iterations per point (ncu stages) | `50` |
| `--gate` / `--no-gate` | wait for a clean window when the live vLLM server is up | --gate |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | per-container timeout (s); a hung kernel's container is killed | `180.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |
| `--plot` / `--no-plot` | re-render charts + README after saving (checkout only) | --no-plot |
| `--page-size` | KV page size for every leg | `128` |


<a id="mjolnir-verify-kvfmt"></a>

## `mjolnir verify kvfmt`

KV layouts x formats: fp8 / fp6 (e2m3, e3m2) / fp4 (e2m1) / nvfp4 on NHD / interleaved / HND.

| option | description | default |
|---|---|---|
| `--variants` | name:flags\|name:flags | `fp8-nhd:--kv_format fp8 --kv_layout nhd|fp8-interleaved:--kv_format fp8 --kv_layout interleaved|fp8-hnd:--kv_format fp8 --kv_layout hnd|e2m3-hnd:--kv_format e2m3 --kv_layout hnd|e3m2-hnd:--kv_format e3m2 --kv_layout hnd|e2m1-hnd:--kv_format e2m1 --kv_layout hnd|nvfp4-hnd:--kv_format nvfp4 --kv_layout hnd` |
| `--refs` | baseline legs | `fi,fi-hnd` |
| `--wall-iters` | harness wall-clock iterations (0 = skip) | `200` |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--ls` | L values (default: 128 4096 8192 16384 32768 65536 131072 262144) |  |
| `--iters` | profiled iterations per point (ncu stages) | `50` |
| `--gate` / `--no-gate` | wait for a clean window when the live vLLM server is up | --gate |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | per-container timeout (s); a hung kernel's container is killed | `180.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |
| `--plot` / `--no-plot` | re-render charts + README after saving (checkout only) | --no-plot |


<a id="mjolnir-verify-quant"></a>

## `mjolnir verify quant`

Accuracy per KV format: relative L2 error vs attention over unquantized KV.

| option | description | default |
|---|---|---|
| `--formats` |  | `fp8 e2m3 e3m2 e2m1 nvfp4` |
| `--batches` |  | `2` |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--ls` | L values (default: 128 4096 8192 16384 32768 65536 131072 262144) |  |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | per-container timeout (s); a hung kernel's container is killed | `180.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |
| `--plot` / `--no-plot` | re-render charts + README after saving (checkout only) | --no-plot |


<a id="mjolnir-verify-ring"></a>

## `mjolnir verify ring`

KV ring depth (TMA → MMA stages) at every L.

| option | description | default |
|---|---|---|
| `--stages` | --kv_stages values | `2 3 4 5 6` |
| `--base` | harness flags every variant gets | `` |
| `--refs` |  | `fi` |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--ls` | L values (default: 128 4096 8192 16384 32768 65536 131072 262144) |  |
| `--iters` | profiled iterations per point (ncu stages) | `50` |
| `--gate` / `--no-gate` | wait for a clean window when the live vLLM server is up | --gate |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | per-container timeout (s); a hung kernel's container is killed | `180.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |
| `--plot` / `--no-plot` | re-render charts + README after saving (checkout only) | --no-plot |


<a id="mjolnir-verify-prefetch"></a>

## `mjolnir verify prefetch`

TMA L2-prefetch distance sweep.

| option | description | default |
|---|---|---|
| `--distances` | --l2_prefetch tiles ahead | `0 1 2 4` |
| `--base` |  | `--kv_format fp8` |
| `--refs` |  | `fi` |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--ls` | L values (default: 128 4096 8192 16384 32768 65536 131072 262144) |  |
| `--iters` | profiled iterations per point (ncu stages) | `50` |
| `--gate` / `--no-gate` | wait for a clean window when the live vLLM server is up | --gate |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | per-container timeout (s); a hung kernel's container is killed | `180.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |
| `--plot` / `--no-plot` | re-render charts + README after saving (checkout only) | --no-plot |


<a id="mjolnir-verify-pdl"></a>

## `mjolnir verify pdl`

Programmatic dependent launch on/off (wall clock).

| option | description | default |
|---|---|---|
| `--batches` |  | `1 8` |
| `--dtypes` |  | `fp8` |
| `--wall-iters` |  | `200` |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--ls` | L values (default: 128 4096 8192 16384 32768 65536 131072 262144) |  |
| `--gate` / `--no-gate` | wait for a clean window when the live vLLM server is up | --gate |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | per-container timeout (s); a hung kernel's container is killed | `180.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |
| `--plot` / `--no-plot` | re-render charts + README after saving (checkout only) | --no-plot |


<a id="mjolnir-verify-ab"></a>

## `mjolnir verify ab`

A/B any harness flag sets (ncu + wall clock). Default: evict-first x PDL.

| option | description | default |
|---|---|---|
| `--variants` | name:flags\|name:flags | `base:--kv_evict_first 0|evict:--kv_evict_first 1|pdl:--kv_evict_first 0 --pdl|both:--kv_evict_first 1 --pdl` |
| `--refs` |  | `fi` |
| `--wall-iters` |  | `200` |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--ls` | L values (default: 128 4096 8192 16384 32768 65536 131072 262144) |  |
| `--iters` | profiled iterations per point (ncu stages) | `50` |
| `--gate` / `--no-gate` | wait for a clean window when the live vLLM server is up | --gate |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | per-container timeout (s); a hung kernel's container is killed | `180.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |
| `--plot` / `--no-plot` | re-render charts + README after saving (checkout only) | --no-plot |


<a id="mjolnir-verify-fp6prof"></a>

## `mjolnir verify fp6prof`

Full ncu profiles per KV format (bytes vs requests vs time, warp stalls).

| option | description | default |
|---|---|---|
| `--variants` |  | `fp8-hnd:--kv_format fp8 --kv_layout hnd|e2m3-hnd:--kv_format e2m3 --kv_layout hnd|e2m1-hnd:--kv_format e2m1 --kv_layout hnd` |
| `--count` | profiled launches per point | `3` |
| `--set` |  | `full` |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--ls` | L values (default: 128 4096 8192 16384 32768 65536 131072 262144) |  |
| `--gate` / `--no-gate` | wait for a clean window when the live vLLM server is up | --gate |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | per-container timeout (s); a hung kernel's container is killed | `180.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |
| `--plot` / `--no-plot` | re-render charts + README after saving (checkout only) | --no-plot |


<a id="mjolnir-verify-timing"></a>

## `mjolnir verify timing`

Harness-timed medians, 1 vs 2 CTAs/SM, fp8 and bf16 Q (no ncu).

| option | description | default |
|---|---|---|
| `--dtypes` |  | `fp8 bf16` |
| `--ctas` |  | `1 2` |
| `--wall-iters` |  | `50` |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--ls` | L values (default: 128 4096 8192 16384 32768 65536 131072 262144) |  |
| `--gate` / `--no-gate` | wait for a clean window when the live vLLM server is up | --gate |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | per-container timeout (s); a hung kernel's container is killed | `180.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |


<a id="mjolnir-verify-profile"></a>

## `mjolnir verify profile`

Profile a decode workload per config (torch profiler): graph replays, eager kernels, GPU idle, top kernels.

| option | description | default |
|---|---|---|
| `--legs` | side:config, comma-separated (mjolnir:NVFP4 = FlashInfer on the Mjolnir image) | `official:NVFP4,mjolnir:NVFP4_MJOLNIR_E2M3,mjolnir:NVFP4_MJOLNIR_E2M3_LMH8_VT64K` |
| `--drafts` | num_speculative_tokens (0 = the config's own) | `3` |
| `--context-words` | prompt size (~1.3 tokens/word; 0 = short) | `6000` |
| `--concurrency` |  | `1` |
| `--requests` | request rounds inside the profiled window | `2` |
| `--max-tokens` |  | `128` |
| `--model` | the model (`vendor/Model`; default: the active one, `mjolnir model`) |  |
| `--restart` | stop a running vLLM server first | off |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | seconds before a step is given up | `1800.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |


<a id="mjolnir-verify-spec"></a>

## `mjolnir verify spec`

MTP per config and draft count: tokens/step, per-position acceptance, decode t/s (restarts the server per point).

| option | description | default |
|---|---|---|
| `--legs` | side:config, comma-separated; official = the stock vLLM image, mjolnir = --image | `official:NVFP4,mjolnir:NVFP4_MJOLNIR,mjolnir:NVFP4_MJOLNIR_E2M3,mjolnir:NVFP4_MJOLNIR_E2M3_LMH8_VT64K` |
| `--drafts` | num_speculative_tokens values to serve | `1 2 3 4 5 6 7` |
| `--concurrency` | requests in flight per block | `1 4` |
| `--max-tokens` | generated tokens per request | `256` |
| `--long-words` | the long-context prompt's size (0 = none) | `6000` |
| `--model` | model (default: the active one) |  |
| `--restart` | stop a running vLLM server (default: refuse — the GPU must be yours) | off |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | per-point server start timeout (s) | `1800.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |
| `--plot` / `--no-plot` | re-render charts + README after saving (checkout only) | --no-plot |


<a id="mjolnir-verify-kv-cache-size"></a>

## `mjolnir verify kv-cache-size`

KV cache capacity per config: serves each leg (restarting the server), reads vLLM's KV sizing, stops it.

| option | description | default |
|---|---|---|
| `--legs` | side:config, comma-separated; official = the stock vLLM image, mjolnir = --image | `official:NVFP4,mjolnir:NVFP4_MJOLNIR,mjolnir:NVFP4_MJOLNIR_E2M3,mjolnir:NVFP4_MJOLNIR_E2M3_LMH8_VT64K` |
| `--model` | model (default: the active one) |  |
| `--restart` | stop a running vLLM server (default: refuse — the GPU must be yours) | off |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | per-leg wait for the KV sizing (s) | `1800.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |
| `--plot` / `--no-plot` | re-render charts + README after saving (checkout only) | --no-plot |


<a id="mjolnir-verify-vllm"></a>

## `mjolnir verify vllm`

The vLLM plugin: runner tests + greedy MJOLNIR-vs-FLASHINFER smoke runs (checkout only).

| option | description | default |
|---|---|---|
| `--smoke-modes` | graph \| eager \| e2m3 (fp6 KV) \| lmh8 (fp8 lm_head) \| spec&lt;N&gt;; '' = none | `graph eager e2m3 lmh8` |
| `--model` | the model (`vendor/Model`; default: the active one, `mjolnir model`) | `Qwen/Qwen3-0.6B` |
| `--tokens` |  | `128` |
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--dry-run` | print the docker commands, run nothing | off |
| `--timeout` | seconds before a step is given up | `1800.0` |
| `--save` / `--no-save` | write the raw result JSON | --save |


<a id="mjolnir-verify-bench"></a>

## `mjolnir verify bench`

Every benchmark stage with default parameters (one headline run each), then re-render the charts.

| option | description | default |
|---|---|---|
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--gate` / `--no-gate` | wait for a clean window when the live vLLM server is up | --gate |
| `--dry-run` | print the docker commands, run nothing | off |
| `--plot` / `--no-plot` | re-render the charts / README afterwards | --plot |
| `--from` | resume the sequence at this stage |  |


<a id="mjolnir-verify-all"></a>

## `mjolnir verify all`

Correctness + every benchmark stage, then re-render the charts.

| option | description | default |
|---|---|---|
| `--image` | vLLM image (default: the active one, see `mjolnir image`) |  |
| `--gate` / `--no-gate` | wait for a clean window when the live vLLM server is up | --gate |
| `--dry-run` | print the docker commands, run nothing | off |
| `--plot` / `--no-plot` | re-render the charts / README afterwards | --plot |
| `--from` | resume the sequence at this stage |  |


<a id="mjolnir-verify-show"></a>

## `mjolnir verify show`

Re-print the latest saved run of a stage (or every stage).

| option | description | default |
|---|---|---|
| `STAGE` | argument |  |
| `--any` | latest run of any parameters, not just default runs | off |


<a id="mjolnir-verify-list"></a>

## `mjolnir verify list`

Stages and their saved runs.


<a id="mjolnir-autoresearch"></a>

## `mjolnir autoresearch`

Autonomous kernel research: OpenCode agents propose, implement and test kernel changes; every experiment is scored by a fixed harness and recorded on the ledger.

Sub-commands: [`mjolnir autoresearch init`](#mjolnir-autoresearch-init), [`mjolnir autoresearch evaluate`](#mjolnir-autoresearch-evaluate), [`mjolnir autoresearch ledger`](#mjolnir-autoresearch-ledger), [`mjolnir autoresearch status`](#mjolnir-autoresearch-status), [`mjolnir autoresearch chart`](#mjolnir-autoresearch-chart), [`mjolnir autoresearch run`](#mjolnir-autoresearch-run), [`mjolnir autoresearch export`](#mjolnir-autoresearch-export).


<a id="mjolnir-autoresearch-init"></a>

## `mjolnir autoresearch init`

Create the research branch and score the starting kernel (the baseline row).

| option | description | default |
|---|---|---|
| `TAG` | argument |  |
| `--base` | branch to start from | `main` |
| `--baseline` / `--no-baseline` | score the starting kernel now | --baseline |
| `--image` | the vLLM image (default: the active one, `mjolnir image`) |  |


<a id="mjolnir-autoresearch-evaluate"></a>

## `mjolnir autoresearch evaluate`

Score the committed kernel: quick correctness + FlashInfer vs Mjolnir sweep;
append the result and the fixed keep/discard decision to the ledger.

| option | description | default |
|---|---|---|
| `--hypothesis` / `-m` | what this experiment tries, one line |  |
| `--agent` | who proposed/implemented it | `autoresearch` |
| `--sources` | comma-separated URLs / arXiv ids behind the idea | `` |
| `--min-gain` | required geomean gain over the best kept score | `0.02` |
| `--ls` | L values (default: all 8; a subset is never a keep) |  |
| `--iters` |  | `50` |
| `--gate` / `--no-gate` | wait for a clean window (vLLM queue 0/0) before measuring | --gate |
| `--image` | the vLLM image (default: the active one, `mjolnir image`) |  |


<a id="mjolnir-autoresearch-ledger"></a>

## `mjolnir autoresearch ledger`

Show the ledger (newest last).

| option | description | default |
|---|---|---|
| `--tail` |  | `20` |
| `--json` / `-j` |  | off |


<a id="mjolnir-autoresearch-status"></a>

## `mjolnir autoresearch status`

Best kept score, experiment counts, the current branch.


<a id="mjolnir-autoresearch-chart"></a>

## `mjolnir autoresearch chart`

Re-render research/progress.png from the live ledger (or the newest exported run).


<a id="mjolnir-autoresearch-run"></a>

## `mjolnir autoresearch run`

Launch the loop: an OpenCode session with the `autoresearch` agent.

| option | description | default |
|---|---|---|
| `--tag` | start a new run first (init &lt;tag&gt;) |  |
| `--model` | OpenCode model (provider/model) for the orchestrator |  |
| `--prompt` |  | `Read research/program.md and run the loop: setup is done when the ledger has a baseline row. Never stop to ask; record every experiment with `mjolnir autoresearch evaluate`.` |


<a id="mjolnir-autoresearch-export"></a>

## `mjolnir autoresearch export`

Snapshot the ledger into research/runs/&lt;tag&gt;/ (ledger.jsonl, ledger.tsv,
SUMMARY.md) — commit that to keep the run's notebook with the repo.

| option | description | default |
|---|---|---|
| `--tag` | run tag (default: from the autoresearch/&lt;tag&gt; branch) |  |


<a id="mjolnir-litellm"></a>

## `mjolnir litellm`

The LiteLLM proxy stack (proxy + postgres + redis) — independent of the vLLM server: 'litellm down' leaves vLLM running, and 'serve down' leaves the proxy up. Reads your template from the litellm data dir, ~/.local/share/mjolnir/litellm/ (see docker/litellm/).

Sub-commands: [`mjolnir litellm up`](#mjolnir-litellm-up), [`mjolnir litellm down`](#mjolnir-litellm-down), [`mjolnir litellm status`](#mjolnir-litellm-status), [`mjolnir litellm logs`](#mjolnir-litellm-logs), [`mjolnir litellm config`](#mjolnir-litellm-config).


<a id="mjolnir-litellm-up"></a>

## `mjolnir litellm up`

Render the config from your template (in the litellm data dir,
~/.local/share/mjolnir/litellm/) with the currently selected
model/config, and start the stack (proxy + postgres + redis). The vLLM
server is reached via the host — it may be up or down.

| option | description | default |
|---|---|---|
| `--model` | the model (`vendor/Model`; default: the active one, `mjolnir model`) |  |
| `--config` | config quant name, e.g. NVFP4_FA4hd256 |  |
| `--port` | host port of the proxy (default 8000) |  |
| `--wait` / `--no-wait` | wait for the proxy's liveliness before returning | --wait |
| `--wait-timeout` |  | `600.0` |
| `--dry-run` | print what would run, run nothing | off |


<a id="mjolnir-litellm-down"></a>

## `mjolnir litellm down`

Stop the stack (the vLLM server keeps running; data kept by
default, --purge wipes postgres + redis state).

| option | description | default |
|---|---|---|
| `--purge` | also wipe the stack data (postgres + redis) | off |
| `--dry-run` | print what would run, run nothing | off |


<a id="mjolnir-litellm-status"></a>

## `mjolnir litellm status`

Container state + proxy liveliness + the rendered config path.

| option | description | default |
|---|---|---|
| `--port` | the server port (default: the active one) |  |


<a id="mjolnir-litellm-logs"></a>

## `mjolnir litellm logs`

Tail the proxy container logs.

| option | description | default |
|---|---|---|
| `--follow` / `-f` |  | off |
| `--tail` |  | `100` |


<a id="mjolnir-litellm-config"></a>

## `mjolnir litellm config`

Render the template with the current model/config (no containers) —
validates the variables and the YAML.

| option | description | default |
|---|---|---|
| `--model` | the model (`vendor/Model`; default: the active one, `mjolnir model`) |  |
| `--config` | the model config (default: the active one) |  |
| `--show` | print the rendered config instead of just validating it | off |


<a id="mjolnir-perses"></a>

## `mjolnir perses`

The Perses + Prometheus observability stack for the vLLM server — independent of the vLLM server: 'perses down' leaves the server running, and 'serve down' leaves the stack up. Prometheus scrapes the server's /metrics (host.docker.internal + the resolved port); Perses serves the two vLLM dashboards (vendored from vllm-project/vllm, docker/perses/). Data: ~/.local/share/mjolnir/perses/.

Sub-commands: [`mjolnir perses up`](#mjolnir-perses-up), [`mjolnir perses down`](#mjolnir-perses-down), [`mjolnir perses status`](#mjolnir-perses-status), [`mjolnir perses logs`](#mjolnir-perses-logs).


<a id="mjolnir-perses-up"></a>

## `mjolnir perses up`

Start the observability stack (prometheus + perses) with the two vLLM
dashboards provisioned. The vLLM scrape target = the resolved server port
(state/$MJOLNIR_PORT — the same one `mjolnir serve` uses; override with
PERSER_VLLM_TARGET in the perses env file). The server may be up or down.

| option | description | default |
|---|---|---|
| `--port` | host port of the Perses UI (default 8080) |  |
| `--wait` / `--no-wait` | wait for the Perses API before returning | --wait |
| `--wait-timeout` |  | `300.0` |
| `--dry-run` | print what would run, run nothing | off |


<a id="mjolnir-perses-down"></a>

## `mjolnir perses down`

Stop the stack (the vLLM server keeps running; data kept by default,
--purge wipes the prometheus TSDB + the perses db).

| option | description | default |
|---|---|---|
| `--purge` | also wipe the stack data (prometheus TSDB + perses db) | off |
| `--dry-run` | print what would run, run nothing | off |


<a id="mjolnir-perses-status"></a>

## `mjolnir perses status`

Container state + API health + the vLLM scrape target + the data dir.

| option | description | default |
|---|---|---|
| `--port` | the server port (default: the active one) |  |


<a id="mjolnir-perses-logs"></a>

## `mjolnir perses logs`

Tail a stack service's logs (default: perses).

| option | description | default |
|---|---|---|
| `--follow` / `-f` |  | off |
| `--tail` |  | `100` |
| `--service` | perses (default) \| prometheus |  |


<a id="mjolnir-hw"></a>

## `mjolnir hw`

Thor host pre-configuration — one-time, idempotent provisioning (sudo where needed): headless boot target, apt upgrade, Docker + NVIDIA default runtime, pip, jtop, swap, fan profiles (recommended + max installed, default selected), service cleanup, locked clocks, MAXN power mode. Run 'hw setup' before serving or benchmarking.

Sub-commands: [`mjolnir hw setup`](#mjolnir-hw-setup), [`mjolnir hw status`](#mjolnir-hw-status).


<a id="mjolnir-hw-setup"></a>

## `mjolnir hw setup`

Pre-configure the Thor host (port of the ~/thor HW setup script).

Runs `scripts/hw/setup-thor.sh`: headless boot target, apt upgrade,
Docker + NVIDIA default runtime, pip, jtop, 32 GB swap (zRAM off),
the fan profiles (`recommended` + `max` installed per-profile
gated, with the .bck rollback backup; default selected —
`recommended` by default), service cleanup, locked max clocks,
MAXN power mode. Idempotent — safe to re-run; the script asks for
sudo.

| option | description | default |
|---|---|---|
| `--yes` / `-y` | don't prompt to continue | off |
| `--upgrade` / `--no-upgrade` | apt update + full-upgrade (default on — the original script's NVIDIA-carrier path) | --upgrade |
| `--keep-gui` | keep the graphical boot target (default: headless multi-user) | off |
| `--fan-profile` | recommended (default) \| max \| cool \| quiet — recommended/max are installed by this step if missing | `recommended` |
| `--swap-size` | swap file size in GB (default 32 — the 128 GB Thor SoM) | `32` |
| `--skip` | comma-separated steps to skip: gui,upgrade,pip,docker,jtop,memory,fan,host,clocks,power | `` |
| `--reboot` | reboot at the end if a step required one (non-interactive) | off |
| `--dry-run` | print the plan, change nothing | off |


<a id="mjolnir-hw-status"></a>

## `mjolnir hw status`

Installed fan profiles and the selected default (read-only, no sudo).

The daemon's live state (which profile it is actually running) needs
sudo — check it with `sudo nvfancontrol -q`.
