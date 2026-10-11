# Benchmarking Methodology — Clean Windows, Achieved-BW, and Provenance

> **Status:** Normative · **Applies to:** every number published in this repo (README,
> `docs/`, `benchmarks/history.jsonl`) · **Implementation:** `mjolnir_cli/gate.py`,
> `mjolnir_cli/verify/` (kernel), `mjolnir_cli/benchy.py` + `mjolnir_cli/history.py` (serving)

These are the rules every benchmark in this repo follows. A number that cannot be
traced to a raw artifact produced under these rules does not belong in this repo.

## TL;DR

- The Thor GPU is **shared**: a live vLLM server (which also serves the AI tooling
  on the box) co-tenants it with any benchmark. We **never** stop or restart that
  server to get a clean GPU.
- Instead, measurements are gated: a bench only runs inside a **clean window** —
  the live server's request queue reads `running=0, waiting=0` for N consecutive
  samples — and any window broken by activity is discarded.
- **Wall-clock is only for relative ratios measured within the same gated window.**
  Absolute numbers come from **NCU achieved memory bandwidth** (`achieved_BW`),
  which is robust to co-tenancy.
- Every published number links to a **raw artifact** (JSON/CSV) in
  [`benchmarks/raw/`](../../benchmarks/raw/) or a NCU table in the workstream
  report, with the image tag, config, and gate status recorded.

## 1. The shared-GPU constraint

Jetson AGX Thor has a single 20-SM (sm_110a) GPU. Two permanent co-tenants:

1. **The live vLLM server** — the patched image serving Qwen3.8-27B; the LLM that
   drives the subagents also runs on this GPU.
2. **Desktop Xorg** — compositing and display workloads, ungated, always present.

Consequences:

- **Never** `docker stop` / restart / kill the live server to "clean" the GPU.
  (That also takes the AI tooling down — the benchmark's operator would die.)
- Any benchmark that touches the GPU goes through the launcher
  (`src/mjolnir-cli/mjolnir_cli/tasks.py`), which **preflights the metrics endpoint** and fails
  fast if the server is down — it never restarts anything.

## 2. The clean-window gate

**Definition.** A *clean window* is a span in which the live server's vLLM request
queue reads **`num_requests_running == 0` AND `num_requests_waiting == 0`** for
**N consecutive samples** (N = 6 is the default, `DEFAULT_CONFIRM` in
`src/mjolnir-cli/mjolnir_cli/gate.py`; individual benches can raise or lower it).

**Source of truth.** The gate polls the live server's Prometheus metrics
(`vllm:num_requests_running`, `vllm:num_requests_waiting`) at
`http://127.0.0.1:6001/metrics`. Implementation: `src/mjolnir-cli/mjolnir_cli/gate.py`
(`CleanWindow`, `wait_for_idle`, `server_load`); CLI: `mjolnir gate --once` (check
now) or `mjolnir gate` (block until a window opens). Gated bench tasks call the
gate; a new bench **must** import it, never re-implement it.

**Protocol**

1. Poll the two counters at a fixed cadence.
2. Count consecutive zero-zero samples; N in a row → window OPEN, start
   measurement.
3. If a non-zero sample appears mid-measurement → the window is DIRTY: stop,
   discard the run, re-wait.
4. The gate status (`clean: true/false`, sample count, timestamps) is recorded in
   every raw result file.

**Why queue counters and not GPU utilization.** Utilization includes the Xorg
co-tenant and our own kernel, so it can't distinguish "clean" from "noisy". The
vLLM queue counters measure exactly the one co-tenant we can reason about (the
live server); Xorg noise is handled statistically (medians over many iters) and,
for absolute numbers, by NCU (below).

## 3. Wall-clock vs NCU achieved-BW

| Question | Instrument | Rule |
|---|---|---|
| "Is kernel A faster than kernel B?" | Wall-clock (µs), median over ≥ 100–300 iters, same gated window | **Relative ratios only**, same window, same shapes, both kernels back-to-back. |
| "How much of the DRAM bandwidth is this kernel using?" | **NCU** `achieved_BW` (achieved memory throughput) | Absolute GB/s vs the roofline (Thor: 273 GB/s nominal). Robust to Xorg co-tenancy because NCU replays the kernel in isolation. |

Rules that keep the numbers honest:

- Wall-clock numbers are never published as absolute "this kernel is X µs"
  against a machine state, only as **ratios in a gated window** (plus the gate
  status). Absolute kernel performance claims use NCU achieved-BW.
- NCU runs go through `mjolnir verify <stage>` (one container per leg and L);
  the per-kernel summaries are parsed into one `mjolnir.verify/1` JSON per run
  in `benchmarks/raw/verify/<stage>/`, logs in `benchmarks/logs/`.
- **Achieved-BW convention on CC 11.0:** `dram__bytes.sum` is not available on
  sm_110, so achieved BW is measured at the L2-fabric level:
  `achieved_BW = lts__t_sectors.sum × 32 B / gpu__time_duration.sum`
  (median of 4 profiled launches). The 273 GB/s DRAM roofline is shown for
  reference only; a kernel at 100% of it is already DRAM-saturated.
- The two instruments are read **together**, and disagreements are documented:
  in the retired GEMV kernel's ring-depth study, NCU showed a ring fix buying
  +12.9% kernel BW while wall clock moved only 0.6% — merge/launch overhead and
  non-KV L2 traffic swallowed the kernel-level gain, which a wall-only bench
  (or an NCU-only bench) would have mis-attributed
  ([gemv-ring-fix-bench.md, git history](https://github.com/thor-arena/mjolnir/blob/06a9ff4/deprecated/gemv/docs/gemv-ring-fix-bench.md)).
  The Mjolnir stages that touch launch behaviour (`pdl`, `ab`, `kvfmt`) report
  both.

## 4. Kernel micro-bench protocol (`mjolnir verify`)

- **Shape:** one sequence, decode (q_len 1), GQA 24:4 (24 query / 4 KV heads),
  head_dim 256, page size 128 — Qwen3.8-27B's attention layer.
- **Context lengths:** every stage defaults to the same eight —
  L ∈ {128, 4K, 8K, 16K, 32K, 64K, 128K, 256K} — so short-context launch
  overhead and long-context bandwidth are both always on the chart. A run with
  any other L list is saved as non-default and never feeds a headline chart.
- **Legs:** Mjolnir (its harness defaults, on the Mjolnir image), FlashInfer
  (vLLM's decode kernel on Thor; NHD and HND drivers) and FlashAttention 2
  (`vllm_flash_attn`, bf16 KV — what stock vLLM 0.30.0 picks when FlashInfer is
  off). **The references run on the official `vllm/vllm-openai` image** — the
  Mjolnir image patches FlashInfer paths, so a reference measured there would
  not be native vLLM (`MJOLNIR_REF_IMAGE` overrides, recorded as `ref_image`
  in every result). All legs of a stage run in the **same run**, so a ratio
  never mixes machine states.
- **NCU settings:** `--clock-control base --cache-control all` (base clocks,
  cold L2 per launch), `--print-summary per-kernel`, 50 profiled iterations
  per point. A leg's time is its main kernel **plus** its merge / reduction
  kernel — a split-KV kernel does not get to hide its second launch.
- **Bandwidth:** effective GB/s = the leg's real KV bytes (format- and
  layout-aware: an fp6 cache is 0.75 B/element) ÷ total time.
- **Noise:** on the shared Thor a single result moves about ±7% between runs;
  tables mark |Δ| < 7% as within noise (`NOISE_PCT`), and a claimed win must
  hold across L and across two runs.
- **Correctness first:** every format/layout the benchmarks use is in the
  `mjolnir verify correctness` matrix against the bit-faithful emulator.

## 5. End-to-end (serving) benchmark protocol

- **Harness:** `llama-benchy` (directly, not via a wrapper that discards raw
  data — see [research/tool-eval-bench-perf-runs.md](../research/tool-eval-bench-perf-runs.md)).
  The mjolnir wrapper is `mjolnir_cli/benchy.py`; raw JSON lands in
  `benchmarks/raw/perf-<timestamp>/`.
- **Protocol flags:** `--runs` = measured runs per cell (default 6),
  `--exact-tg` (pin output tokens — kills EOS early-stop variance), `--no-cache`
  (no server-side prefix caching between measurements).
- **Independent repeats:** `--repeat` re-runs the whole sweep R times
  (default 1), each repeat inside its **own clean window**; per-cell values
  are pooled across repeats and reported as mean ± std. A DIRTY repeat is
  dropped, not averaged in.
- **Server log:** the run saves the vLLM container's whole log
  (`vllm-server.log` next to the raw JSONs) — startup kernel dispatch is in
  there for debugging what the served image actually ran.
- **A/B mode** (`mjolnir bench ab --backends …`) restarts the server per leg
  (different image/config). This is deliberately destructive and must be
  scheduled — it takes the live service down between legs.
- **Provenance:** every sweep appends exactly one row to
  [`benchmarks/history.jsonl`](../../benchmarks/history.jsonl)
  (timestamp, image tag, backend label, config, gate status, runs, raw-path).
  The README charts and `mjolnir plot` render from that log — charts are always
  reproducible from committed data.

## 6. Provenance rules

1. **No prose-only results.** Every number in a report or the README resolves to:
   a raw JSON/CSV in `benchmarks/raw/`, an NCU table in the report, or an upstream
   anchor. The report names the artifact.
2. **Gate status is data.** Raw result files carry the gate verdict; results
   with `clean: false` are excluded from any claim.
3. **Image tags are pinned** in the history log and in report meta blocks —
   results from `mjolnir/vllm-thor:qwen38-sm110-vN` are only comparable to other
   vN-same results.
4. **Ratios, not absolutes, for wall-clock.** See §3.
5. **Bumps invalidate comparisons.** Bumping the vLLM nightly base
   ([thor-stack/nightly-bump-process.md](../thor-stack/nightly-bump-process.md))
   starts a new comparison epoch; old numbers stay cited with their tag.
