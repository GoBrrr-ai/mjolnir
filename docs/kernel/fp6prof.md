# Full profiles per KV format

> full ncu profiles: L2 sectors vs requests vs time, warp stalls, per KV format · `mjolnir verify fp6prof` · run `20261017-212447`

## The question

fp6 reads 25% fewer bytes than fp8 — so why is it not 25% faster? A full Nsight Compute profile per KV format answers where the time goes.

## How it is measured

`ncu --set full` on 3 launches of the decode kernel per format and L (cold L2, base clocks). Reported: the **fastest launch's** metrics — time, stall mix and bytes from one self-consistent replay. Identical launches of this kernel flip up to ~25% in kernel time under replay (CTA-residency modes, at every L and format), so a mean over a few launches mixes the modes and invents per-L holes that wall clock does not show; the fastest launch is the one the wall-clock harness medians see. The `.ncu-rep` files stay in the run's log directory for Nsight Compute's *Add Baseline* view.

## Result

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../assets/benchmarks/fp6prof-bytes-vs-time-dark.png">
  <img src="../../assets/benchmarks/fp6prof-bytes-vs-time.png" alt="Bytes vs time per KV format" width="100%">
</picture>

- `e2m3-hnd`: kernel time **0.90×** the baseline for 0.75× the bytes — 40% of the byte saving becomes speed (mean over L).
- `e2m1-hnd`: kernel time **0.61×** the baseline for 0.50× the bytes — 78% of the byte saving becomes speed (mean over L).

| L | variant | bits | kernel time | time ratio | L2 bytes ratio | ideal | bytes / L2 request | top stall (cycles/instr) |
|---|---|--:|--:|--:|--:|--:|--:|---|
| **128** | `fp8-hnd` | 8 | 13.0&nbsp;µs | 1.00 | 1.00 | 1.00 | 239 B | long_scoreboard 3.7 |
| **128** | `e2m3-hnd` | 6 | 13.9&nbsp;µs | 1.07 | 0.87 | 0.75 | 143 B | long_scoreboard 3.3 |
| **128** | `e2m1-hnd` | 4 | 14.8&nbsp;µs | 1.14 | 0.74 | 0.50 | 178 B | long_scoreboard 3.0 |
| **4K** | `fp8-hnd` | 8 | 50.7&nbsp;µs | 1.00 | 1.00 | 1.00 | 135 B | long_scoreboard 17.8 |
| **4K** | `e2m3-hnd` | 6 | 43.9&nbsp;µs | 0.87 | 0.77 | 0.75 | 69 B | long_scoreboard 18.3 |
| **4K** | `e2m1-hnd` | 4 | 30.0&nbsp;µs | 0.59 | 0.53 | 0.50 | 72 B | long_scoreboard 9.1 |
| **8K** | `fp8-hnd` | 8 | 93.4&nbsp;µs | 1.00 | 1.00 | 1.00 | 132 B | long_scoreboard 15.4 |
| **8K** | `e2m3-hnd` | 6 | 76.3&nbsp;µs | 0.82 | 0.76 | 0.75 | 67 B | long_scoreboard 26.9 |
| **8K** | `e2m1-hnd` | 4 | 46.4&nbsp;µs | 0.50 | 0.52 | 0.50 | 68 B | long_scoreboard 12.5 |
| **16K** | `fp8-hnd` | 8 | 167.3&nbsp;µs | 1.00 | 1.00 | 1.00 | 130 B | long_scoreboard 20.6 |
| **16K** | `e2m3-hnd` | 6 | 155.2&nbsp;µs | 0.93 | 0.76 | 0.75 | 66 B | long_scoreboard 39.6 |
| **16K** | `e2m1-hnd` | 4 | 84.4&nbsp;µs | 0.50 | 0.51 | 0.50 | 66 B | long_scoreboard 7.7 |
| **32K** | `fp8-hnd` | 8 | 331.8&nbsp;µs | 1.00 | 1.00 | 1.00 | 129 B | long_scoreboard 21.1 |
| **32K** | `e2m3-hnd` | 6 | 302.3&nbsp;µs | 0.91 | 0.75 | 0.75 | 65 B | long_scoreboard 20.1 |
| **32K** | `e2m1-hnd` | 4 | 192.1&nbsp;µs | 0.58 | 0.51 | 0.50 | 65 B | long_scoreboard 10.3 |
| **64K** | `fp8-hnd` | 8 | 642.3&nbsp;µs | 1.00 | 1.00 | 1.00 | 129 B | long_scoreboard 21.8 |
| **64K** | `e2m3-hnd` | 6 | 533.0&nbsp;µs | 0.83 | 0.75 | 0.75 | 65 B | long_scoreboard 22.9 |
| **64K** | `e2m1-hnd` | 4 | 341.2&nbsp;µs | 0.53 | 0.50 | 0.50 | 65 B | long_scoreboard 11.5 |
| **128K** | `fp8-hnd` | 8 | 1.18&nbsp;ms | 1.00 | 1.00 | 1.00 | 128 B | long_scoreboard 23.6 |
| **128K** | `e2m3-hnd` | 6 | 1.04&nbsp;ms | 0.88 | 0.75 | 0.75 | 64 B | long_scoreboard 19.8 |
| **128K** | `e2m1-hnd` | 4 | 635.5&nbsp;µs | 0.54 | 0.50 | 0.50 | 64 B | long_scoreboard 10.8 |
| **256K** | `fp8-hnd` | 8 | 2.36&nbsp;ms | 1.00 | 1.00 | 1.00 | 128 B | long_scoreboard 22.1 |
| **256K** | `e2m3-hnd` | 6 | 2.12&nbsp;ms | 0.90 | 0.75 | 0.75 | 64 B | long_scoreboard 20.2 |
| **256K** | `e2m1-hnd` | 4 | 1.19&nbsp;ms | 0.51 | 0.50 | 0.50 | 64 B | long_scoreboard 10.3 |

<details>
<summary>How the % and × here are calculated</summary>

- **time ratio** = this row's Nsight Compute kernel time ÷ the first row's at the same L; **L2 bytes ratio** = L2 read sectors ÷ the first row's; **ideal** = this row's KV bits ÷ the first row's — the ratio a purely bandwidth-bound kernel would show.
- **bytes / L2 request** = 32 B × L2 read sectors ÷ L2 read requests.

</details><br />

<sub>Rendered from [`benchmarks/raw/verify/fp6prof/20261017-212447.json`](../../benchmarks/raw/verify/fp6prof/20261017-212447.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

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

- **Tried:** a TMA box that keeps fp6 requests at 128 B (SW128 smem atoms, autoresearch run 1) — ~5% faster at 64K–256K, slower at short L; gated at ≥ 64K it is +1.7% overall. At long L e2m3 is already close to its byte ratio, so request size is not the main limit there.
- **Tried:** TMA-warp block-table readahead (a rolling page window refilled N tiles ahead of the TMA issue, `--bt_lookahead`, task/flashinfer-0.7.2-sm110-analysis.md E2a) — correct (97/97) but no win: wall ratios move ±2–4% around zero at 16K–128K. The `long_scoreboard` stall is not the block table.
- **Open:** the uniform ~10–15% shortfall vs the byte ratio at every L (the latency side of the whole kernel: L2 at 17–25% of peak, `long_scoreboard` the top stall); ring depth chosen per L at runtime. The ranked list with commands is [`research/program.md`](../../research/program.md).
- **Not worth it:** unpacking fp6 in registers instead of the TMA — the nvfp4 convert warpgroup shows what a conversion pass costs (0.48–0.69× FlashInfer).

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../.github/CONTRIBUTING.md).

---

[← Speculative decoding](spec.md) · [case study](README.md) · [KV ring depth →](ring.md)
