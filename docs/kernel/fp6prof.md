# Full profiles per KV format

> full ncu profiles: L2 sectors vs requests vs time, warp stalls, per KV format · `mjolnir verify fp6prof` · run `20261011-095706`

## The question

fp6 reads 25% fewer bytes than fp8 — so why is it not 25% faster? A full Nsight Compute profile per KV format answers where the time goes.

## How it is measured

`ncu --set full` on 3 launches of the decode kernel per format and L (cold L2, base clocks). Recorded: kernel time, L2 sectors and requests read, throughput %, achieved occupancy, warp-stall reasons. The `.ncu-rep` files stay in the run's log directory for Nsight Compute's *Add Baseline* view.

## Result

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../assets/benchmarks/fp6prof-bytes-vs-time-dark.png">
  <img src="../../assets/benchmarks/fp6prof-bytes-vs-time.png" alt="Bytes vs time per KV format" width="100%">
</picture>

- `e2m3-hnd`: kernel time **0.96×** the baseline for 0.75× the bytes — 14% of the byte saving becomes speed (mean over L).
- `e2m1-hnd`: kernel time **0.59×** the baseline for 0.50× the bytes — 82% of the byte saving becomes speed (mean over L).

| L | variant | bits | kernel time | time ratio | L2 bytes ratio | ideal | bytes / L2 request | top stall (cycles/instr) |
|---|---|--:|--:|--:|--:|--:|--:|---|
| **128** | `fp8-hnd` | 8 | 13.8 µs | 1.00 | 1.00 | 1.00 | 238 B | long_scoreboard 3.9 |
| **128** | `e2m3-hnd` | 6 | 15.1 µs | 1.09 | 0.87 | 0.75 | 142 B | long_scoreboard 3.3 |
| **128** | `e2m1-hnd` | 4 | 14.3 µs | 1.04 | 0.75 | 0.50 | 179 B | long_scoreboard 2.9 |
| **4K** | `fp8-hnd` | 8 | 46.8 µs | 1.00 | 1.00 | 1.00 | 135 B | long_scoreboard 14.7 |
| **4K** | `e2m3-hnd` | 6 | 44.1 µs | 0.94 | 0.77 | 0.75 | 69 B | long_scoreboard 13.6 |
| **4K** | `e2m1-hnd` | 4 | 31.6 µs | 0.67 | 0.53 | 0.50 | 72 B | long_scoreboard 7.8 |
| **8K** | `fp8-hnd` | 8 | 97.6 µs | 1.00 | 1.00 | 1.00 | 132 B | long_scoreboard 22.4 |
| **8K** | `e2m3-hnd` | 6 | 97.1 µs | 0.99 | 0.76 | 0.75 | 67 B | long_scoreboard 20.3 |
| **8K** | `e2m1-hnd` | 4 | 48.9 µs | 0.50 | 0.52 | 0.50 | 68 B | long_scoreboard 9.6 |
| **16K** | `fp8-hnd` | 8 | 179.8 µs | 1.00 | 1.00 | 1.00 | 130 B | long_scoreboard 29.8 |
| **16K** | `e2m3-hnd` | 6 | 211.4 µs | 1.18 | 0.75 | 0.75 | 66 B | long_scoreboard 31.8 |
| **16K** | `e2m1-hnd` | 4 | 106.8 µs | 0.59 | 0.51 | 0.50 | 66 B | long_scoreboard 13.8 |
| **32K** | `fp8-hnd` | 8 | 438.0 µs | 1.00 | 1.00 | 1.00 | 129 B | long_scoreboard 39.7 |
| **32K** | `e2m3-hnd` | 6 | 424.6 µs | 0.97 | 0.75 | 0.75 | 65 B | long_scoreboard 39.7 |
| **32K** | `e2m1-hnd` | 4 | 210.8 µs | 0.48 | 0.50 | 0.50 | 65 B | long_scoreboard 18.4 |
| **64K** | `fp8-hnd` | 8 | 861.8 µs | 1.00 | 1.00 | 1.00 | 129 B | long_scoreboard 47.4 |
| **64K** | `e2m3-hnd` | 6 | 754.6 µs | 0.88 | 0.75 | 0.75 | 64 B | long_scoreboard 45.8 |
| **64K** | `e2m1-hnd` | 4 | 436.8 µs | 0.51 | 0.50 | 0.50 | 65 B | long_scoreboard 22.5 |
| **128K** | `fp8-hnd` | 8 | 1.57 ms | 1.00 | 1.00 | 1.00 | 128 B | long_scoreboard 48.8 |
| **128K** | `e2m3-hnd` | 6 | 1.27 ms | 0.81 | 0.75 | 0.75 | 64 B | long_scoreboard 39.0 |
| **128K** | `e2m1-hnd` | 4 | 734.6 µs | 0.47 | 0.50 | 0.50 | 64 B | long_scoreboard 21.7 |
| **256K** | `fp8-hnd` | 8 | 2.72 ms | 1.00 | 1.00 | 1.00 | 128 B | long_scoreboard 45.6 |
| **256K** | `e2m3-hnd` | 6 | 2.31 ms | 0.85 | 0.75 | 0.75 | 64 B | long_scoreboard 39.9 |
| **256K** | `e2m1-hnd` | 4 | 1.28 ms | 0.47 | 0.50 | 0.50 | 64 B | long_scoreboard 20.1 |

<sub>Rendered from [`benchmarks/raw/verify/fp6prof/20261011-095706.json`](../../benchmarks/raw/verify/fp6prof/20261011-095706.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

## How to read it

- **L2 bytes ratio ≈ the ideal (bits) ratio** → no padding or over-fetch: the TMA reads exactly the packed bytes;
- **kernel time well above the bytes ratio** → the limit is not bandwidth. The *bytes per L2 request* column shows fp6 rows arriving as more, smaller requests (the TMA's sub-byte unpack path);
- the stall column shows what warps wait on — long scoreboard = waiting for memory.

## Reproduce

```bash
mjolnir verify fp6prof
mjolnir verify fp6prof --variants "fp8-hnd:--kv_format fp8 --kv_layout hnd|e2m3-hnd:--kv_format e2m3 --kv_layout hnd" --ls 65536
```

## Ideas to try

- a TMA box shape that keeps fp6 requests at 128 B;
- loading fp6 as raw bytes and unpacking in registers (costs ALU, saves requests).

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../.github/CONTRIBUTING.md).

---

[← KV format accuracy](quant.md) · [case study](README.md) · [KV ring depth →](ring.md)
