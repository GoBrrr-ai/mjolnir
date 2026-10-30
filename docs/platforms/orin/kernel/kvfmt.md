# KV layouts and formats

> fp8 / fp6 / fp4 / nvfp4 KV on NHD / interleaved / HND pages (ncu + wall clock) · `mjolnir verify kvfmt` · run `20261030-151506`

## The question

How many bits does a KV element need, and which page layout streams them best? The `sm_87` mma kernel reads fp8 (e4m3) and fp6 (e2m3) KV — the fp8 rows dequantize in shared memory on the `mma.m16n8k16` path, the fp6 rows ride the packed 6-bit LUT — so its sweep is fp8 vs e2m3 on NHD and HND pages.

## How it is measured

Every variant runs at every L with the harness default otherwise; FlashInfer on NHD and HND are the references. A wall-clock row (harness median, cold L2) adds what ncu cannot see — launch gaps. The tradeoff chart pairs each format's speed with its accuracy from [quant](quant.md).

## Result

<img src="../../../../assets/benchmarks/orin/Qwen3.8-27B/kvfmt-heatmap.png" alt="KV layouts and formats" width="100%">

<img src="../../../../assets/benchmarks/orin/Qwen3.8-27B/kvfmt-tradeoff.png" alt="Speed vs accuracy per KV format" width="100%">

- Fastest variant at the most context lengths: **`fp8-hnd`** (7 of 8).
- Best geomean vs FlashInfer: **`fp8-hnd`** (1.26×); worst: `e2m3-hnd` (0.77×).

| variant | flags | 128 | 4K | 8K | 16K | 32K | 64K | 128K | 256K | geomean |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `fp8-nhd` | `--kv_format fp8 --kv_layout nhd` | $\color{#16a34a}{2.17\times}$ | $\color{#16a34a}{1.45\times}$ | $\color{#16a34a}{1.28\times}$ | $\color{#16a34a}{1.16\times}$ | $\color{#16a34a}{1.09\times}$ | $\color{#16a34a}{1.07\times}$ | **$\color{#16a34a}{1.05\times}$** | $\color{grey}{1.03\times}$ | $\color{#16a34a}{\mathbf{1.25\times}}$ |
| `fp8-hnd` | `--kv_format fp8 --kv_layout hnd` | **$\color{#16a34a}{2.29\times}$** | **$\color{#16a34a}{1.45\times}$** | **$\color{#16a34a}{1.28\times}$** | **$\color{#16a34a}{1.16\times}$** | **$\color{#16a34a}{1.10\times}$** | **$\color{#16a34a}{1.07\times}$** | **$\color{#16a34a}{1.05\times}$** | **$\color{#16a34a}{1.03\times}$** | $\color{#16a34a}{\mathbf{1.26\times}}$ |
| `e2m3-hnd` | `--kv_format e2m3 --kv_layout hnd` | $\color{#16a34a}{1.91\times}$ | $\color{#dc2626}{0.90\times}$ | $\color{#dc2626}{0.76\times}$ | $\color{#dc2626}{0.68\times}$ | $\color{#dc2626}{0.63\times}$ | $\color{#dc2626}{0.61\times}$ | $\color{#dc2626}{0.60\times}$ | $\color{#dc2626}{0.59\times}$ | $\color{#dc2626}{0.77\times}$ |

<details>
<summary>How the % and × here are calculated</summary>

- Each cell = **reference time ÷ variant time** (Nsight Compute kernel time, decode + merge, base clocks, cold L2, same run, same L). `1.10×` = the variant needs 1 ÷ 1.10 = 91% of the reference's time; below 1 it is slower.
- **geomean** = exp(mean of ln(ratio)) over the context lengths; **bold** = the fastest variant at that L.
- Green ≥ 1.03, grey within ±3% (`NOISE_PCT`, the run-to-run spread of a same-run ratio), red ≤ 0.97.

</details><br />


**Wall clock** (harness median, cold L2, b=1; Δ vs the first variant):

| L | fp8-nhd | fp8-hnd | e2m3-hnd |
|---|--:|--:|--:|
| **128** | 68.0&nbsp;µs | 65.4&nbsp;µs $\color{#16a34a}{\text{−3.8\\%}}$ | 64.7&nbsp;µs $\color{#16a34a}{\text{−4.9\\%}}$ |
| **4K** | 89.1&nbsp;µs | 87.0&nbsp;µs $\color{grey}{\text{−2.4\\%}}$ | 120.9&nbsp;µs $\color{#dc2626}{\text{+35.7\\%}}$ |
| **8K** | 135.7&nbsp;µs | 135.2&nbsp;µs $\color{grey}{\text{−0.4\\%}}$ | 202.9&nbsp;µs $\color{#dc2626}{\text{+49.6\\%}}$ |
| **16K** | 228.6&nbsp;µs | 230.4&nbsp;µs $\color{grey}{\text{+0.8\\%}}$ | 366.4&nbsp;µs $\color{#dc2626}{\text{+60.3\\%}}$ |
| **32K** | 417.4&nbsp;µs | 415.6&nbsp;µs $\color{grey}{\text{−0.4\\%}}$ | 691.5&nbsp;µs $\color{#dc2626}{\text{+65.7\\%}}$ |
| **64K** | 792.3&nbsp;µs | 796.0&nbsp;µs $\color{grey}{\text{+0.5\\%}}$ | 1.34&nbsp;ms $\color{#dc2626}{\text{+69.2\\%}}$ |
| **128K** | 1.54&nbsp;ms | 1.55&nbsp;ms $\color{grey}{\text{+0.6\\%}}$ | 2.64&nbsp;ms $\color{#dc2626}{\text{+71.2\\%}}$ |
| **256K** | 3.02&nbsp;ms | 3.02&nbsp;ms $\color{grey}{\text{+0.2\\%}}$ | 5.23&nbsp;ms $\color{#dc2626}{\text{+73.4\\%}}$ |

<details>
<summary>How the % and × here are calculated</summary>

- Wall-clock median per launch over the stage's iterations (CUDA events), every variant in the same run.
- **%** = (variant − first column) ÷ first column × 100 — the change in **time**: negative = faster. `−6.0%` = 6% less time than the first column.
- Green = at least 3% less time, red = at least 3% more, grey = within ±3% (`NOISE_PCT`).

</details><br />


<sub>Rendered from [`benchmarks/raw/verify/kvfmt/20261030-151506.json`](../../../../benchmarks/raw/verify/kvfmt/20261030-151506.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

## How to read it

- **layout**: fp8 on NHD vs interleaved vs HND isolates the page layout;
- **format**: on HND, e2m3 reads 25% fewer bytes than fp8 — compare the measured speedup with that ideal (and see [fp6prof](fp6prof.md) for why fp6 realizes less of it);
- the tradeoff chart is the decision: up-and-left wins; e2m3 is the harness default because it sits next to fp8 on accuracy.

## Reproduce

```bash
mjolnir verify kvfmt
mjolnir verify kvfmt --variants "e2m3:--kv_format e2m3"
```

## Ideas to try

- close the hd128 gap (the top integration item): the mma serves head_size 256 only, and fp6 stays hd256-only — a 96-byte row splits into non-integral 4-byte lane slices.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../../../.github/CONTRIBUTING.md).

---

[← Head-major pages](hndsweep.md) · [case study](../README.md#-kernel) · [KV format accuracy →](quant.md)
