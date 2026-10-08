# T5 decode — 4-way ncu L-sweep (FI-tc / GEMV / `_spark_tc` / T5)

**2026-10-08.** The full decode-kernel comparison across the compute-bound →
DRAM-bound transition: the **T5** (tcgen05+TMEM+CVT, the new kernel), the
shipped **GEMV** (pure-FMA, the current default), the **`_spark_tc`** (our
mma.sync TC, the 20× L2-re-read bug), and the **FI-tc** (FlashInfer FA2-tc, the
reference). Per-L ncu profile, `--clock-control base --cache-control none`,
per-kernel `gpu__time_duration.sum` (robust to clock skew) +
`lts__t_sectors.sum` (the memory-access check). Driver:
`task/fa4-hd256-fp8/ncu-4way-lsweep.sh`; raw:
`benchmarks/raw/ncu-4way-lsweep/`.

Shape: Qwen3.8-27B hd256, GQA 24:4, fp8 KV, M=1 (single-token decode).

## Decode time (µs, base clock)

| L | FI-tc | GEMV | `_spark_tc` | **T5** | fastest |
|---|---|---|---|---|---|
| 8K | 67.5 | 181 | 719 | **32.1** | T5 |
| 16K | **129** | 351 | 1190 | 156 | FI-tc |
| 32K | 377 | 690 | 2170 | 381 | ~tied |
| 64K | **747** | 1360 | 4420 | 807 | FI-tc |
| 128K | **1400** | 2740 | 8050 | 1460 | FI-tc |

## L2 sectors (M) — the memory-access check

| L | FI-tc | GEMV | `_spark_tc` | T5 |
|---|---|---|---|---|
| 8K | 0.54 | 0.68 | **11.0** | 0.54 |
| 16K | 1.06 | 1.19 | **21.7** | 1.06 |
| 32K | 2.11 | 2.25 | **43.1** | 2.12 |
| 64K | 4.21 | 4.34 | **85.8** | 4.24 |
| 128K | 8.41 | 8.56 | **171.4** | 8.47 |

FI-tc / GEMV / T5 all read the KV ~once (comparable sectors). The `_spark_tc`
re-reads the KV **~20×** (11 M→171 M sectors) — the known single-stage-pipeline
L2-thrash bug (no ring → the 80 CTAs' KV working set thrashes the 32 MiB L2).

## Launched CTAs (the one-wave check)

| kernel | CTAs | config |
|---|---|---|
| FI-tc | 40 | 2/SM |
| GEMV | 80 | 4/SM |
| `_spark_tc` | 80 | 4/SM |
| **T5** | **20** | 1/SM (one wave) |

## Findings

1. **T5 beats the shipped GEMV everywhere** — 5.6× at 8K (T5 is L2-resident:
   the bf16 KV fits the 32 MiB L2) and ~1.7–1.9× at the DRAM-bound 32K–128K.
2. **T5 ≈ FI-tc at 32K; FI-tc edges ahead at 64K/128K.** The reason: T5 reads
   the **bf16** KV (2× the bytes) from the in-mma fp8→bf16 cvt, while FI-tc
   reads the **fp8** KV (1× bytes) directly. T5 is ~1.8× more DRAM-efficient
   *per byte* (184 vs 90 GB/s ≈ 67% vs 33% of the 273 GB/s peak — T5 is
   ~DRAM-bound, FI-tc is latency-bound), but the 2× bytes cost it the wall-clock
   edge. T5 wins at 8K (L2-resident).
3. **`_spark_tc` is the worst everywhere** — its 20× L2 re-read (the single-stage
   `cp_async_wait_group(0)` pipeline) makes it ~3× slower than the GEMV at every
   L. Confirms the open bug (`Fix the _spark TC's 20× L2-sector bug` in AGENTS.md).
4. **GEMV is latency-bound at the DRAM-bound lengths** (49 GB/s ≈ 18% of peak) —
   its 80-CTA 4/SM config + fine per-stage commits don't hide the DRAM latency;
   T5's one-wave 1/SM tcgen05 config does.

## Caveats
- **Co-tenancy:** the box wasn't perfectly idle the whole sweep; lean on the
  **relative** ratios + sectors, not the absolute µs.
- **The cvt is one-time:** the fp8→bf16 cvt is a workspace setup (amortized),
  not per-iter; the T5 decode kernel row is the per-iter number.
- **T5 is dense-only** in this sweep (the paged verify passes bit-exact; the
  paged gather overhead is not yet profiled).

## Next
- Wire T5 into `interface.py` dispatch behind `VLLM_FA4_HD256_T5=1` + a `t5-bench`
  gated task; re-profile paged.
- If T5 read the fp8 KV directly (no cvt) it would be ~2× faster than FI-tc at the
  DRAM-bound lengths — the fp8-mma blocker on sm_110 is what forces the bf16 cvt.
- varlen-M (MTP-verify) is the remaining functional gap.
