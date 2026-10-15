# KV layouts and formats

> fp8 / fp6 / fp4 / nvfp4 KV on NHD / interleaved / HND pages (ncu + wall clock) · `mjolnir verify kvfmt` · run `20261012-203801`

## The question

How many bits does a KV element need, and which page layout streams them best? Mjolnir can read fp8 (e4m3), fp6 (e2m3, e3m2) and fp4 (e2m1) KV **directly** in the tensor-core MMA (`tcgen05 kind::f8f6f4`, TMA unpacks sub-byte data to byte containers), and NVFP4 (e2m1 + one e4m3 scale per 16 elements) through a conversion warpgroup.

## How it is measured

Every variant runs at every L with the harness default otherwise; FlashInfer on NHD and HND are the references. A wall-clock row (harness median, cold L2) adds what ncu cannot see — launch gaps. The tradeoff chart pairs each format's speed with its accuracy from [quant](quant.md).

## Result

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../assets/benchmarks/kvfmt-heatmap-dark.png">
  <img src="../../assets/benchmarks/kvfmt-heatmap.png" alt="KV layouts and formats" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../assets/benchmarks/kvfmt-tradeoff-dark.png">
  <img src="../../assets/benchmarks/kvfmt-tradeoff.png" alt="Speed vs accuracy per KV format" width="100%">
</picture>

- Fastest variant at the most context lengths: **`e2m1-hnd`** (7 of 8).
- Best geomean vs FlashInfer: **`e2m1-hnd`** (1.92×); worst: `nvfp4-hnd` (0.57×).

| variant | flags | 128 | 4K | 8K | 16K | 32K | 64K | 128K | 256K | geomean |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `fp8-nhd` | `--kv_format fp8 --kv_layout nhd` | **$\color{#16a34a}{1.81\times}$** | $\color{grey}{1.02\times}$ | $\color{grey}{0.99\times}$ | $\color{#dc2626}{0.87\times}$ | $\color{#dc2626}{0.93\times}$ | $\color{grey}{1.02\times}$ | $\color{#dc2626}{0.97\times}$ | $\color{grey}{0.98\times}$ | $\color{#16a34a}{\mathbf{1.04\times}}$ |
| `fp8-interleaved` | `--kv_format fp8 --kv_layout interleaved` | $\color{#16a34a}{1.74\times}$ | $\color{grey}{0.97\times}$ | $\color{#dc2626}{0.89\times}$ | $\color{#dc2626}{0.88\times}$ | $\color{#dc2626}{0.91\times}$ | $\color{#dc2626}{0.95\times}$ | $\color{#dc2626}{0.93\times}$ | $\color{grey}{0.97\times}$ | $\color{grey}{1.00\times}$ |
| `fp8-hnd` | `--kv_format fp8 --kv_layout hnd` | $\color{#16a34a}{1.67\times}$ | $\color{#16a34a}{1.07\times}$ | $\color{#16a34a}{1.05\times}$ | $\color{grey}{0.98\times}$ | $\color{#16a34a}{1.07\times}$ | $\color{#16a34a}{1.17\times}$ | $\color{#16a34a}{1.09\times}$ | $\color{#16a34a}{1.12\times}$ | $\color{#16a34a}{\mathbf{1.14\times}}$ |
| `e2m3-hnd` | `--kv_format e2m3 --kv_layout hnd` | $\color{#16a34a}{1.77\times}$ | $\color{#16a34a}{1.17\times}$ | $\color{#16a34a}{1.09\times}$ | $\color{grey}{1.02\times}$ | $\color{#16a34a}{1.11\times}$ | $\color{#16a34a}{1.31\times}$ | $\color{#16a34a}{1.32\times}$ | $\color{#16a34a}{1.32\times}$ | $\color{#16a34a}{\mathbf{1.25\times}}$ |
| `e3m2-hnd` | `--kv_format e3m2 --kv_layout hnd` | $\color{#16a34a}{1.79\times}$ | $\color{#16a34a}{1.16\times}$ | $\color{#16a34a}{1.12\times}$ | $\color{grey}{1.02\times}$ | $\color{#16a34a}{1.09\times}$ | $\color{#16a34a}{1.32\times}$ | $\color{#16a34a}{1.31\times}$ | $\color{#16a34a}{1.31\times}$ | $\color{#16a34a}{\mathbf{1.25\times}}$ |
| `e2m1-hnd` | `--kv_format e2m1 --kv_layout hnd` | $\color{#16a34a}{1.78\times}$ | **$\color{#16a34a}{1.63\times}$** | **$\color{#16a34a}{1.75\times}$** | **$\color{#16a34a}{1.87\times}$** | **$\color{#16a34a}{1.95\times}$** | **$\color{#16a34a}{2.11\times}$** | **$\color{#16a34a}{2.09\times}$** | **$\color{#16a34a}{2.24\times}$** | $\color{#16a34a}{\mathbf{1.92\times}}$ |
| `nvfp4-hnd` | `--kv_format nvfp4 --kv_layout hnd` | $\color{#dc2626}{0.69\times}$ | $\color{#dc2626}{0.48\times}$ | $\color{#dc2626}{0.48\times}$ | $\color{#dc2626}{0.51\times}$ | $\color{#dc2626}{0.58\times}$ | $\color{#dc2626}{0.64\times}$ | $\color{#dc2626}{0.60\times}$ | $\color{#dc2626}{0.59\times}$ | $\color{#dc2626}{0.57\times}$ |

<details>
<summary>How the % and × here are calculated</summary>

- Each cell = **reference time ÷ variant time** (Nsight Compute kernel time, decode + merge, base clocks, cold L2, same run, same L). `1.10×` = the variant needs 1 ÷ 1.10 = 91% of the reference's time; below 1 it is slower.
- **geomean** = exp(mean of ln(ratio)) over the context lengths; **bold** = the fastest variant at that L.
- Green ≥ 1.03, grey within ±3% (`NOISE_PCT`, the run-to-run spread of a same-run ratio), red ≤ 0.97.

</details><br />


**Wall clock** (harness median, cold L2, b=1; Δ vs the first variant):

| L | fp8-nhd | fp8-interleaved | fp8-hnd | e2m3-hnd | e3m2-hnd | e2m1-hnd | nvfp4-hnd |
|---|--:|--:|--:|--:|--:|--:|--:|
| **128** | 14.2&nbsp;µs | 12.4&nbsp;µs $\color{#16a34a}{\text{−12.4\\%}}$ | 14.1&nbsp;µs $\color{grey}{\text{−0.3\\%}}$ | 12.2&nbsp;µs $\color{#16a34a}{\text{−13.8\\%}}$ | 12.2&nbsp;µs $\color{#16a34a}{\text{−14.0\\%}}$ | 12.1&nbsp;µs $\color{#16a34a}{\text{−14.5\\%}}$ | 22.3&nbsp;µs $\color{#dc2626}{\text{+57.5\\%}}$ |
| **4K** | 92.0&nbsp;µs | 92.2&nbsp;µs $\color{grey}{\text{+0.2\\%}}$ | 86.8&nbsp;µs $\color{#16a34a}{\text{−5.6\\%}}$ | 76.6&nbsp;µs $\color{#16a34a}{\text{−16.7\\%}}$ | 76.6&nbsp;µs $\color{#16a34a}{\text{−16.7\\%}}$ | 62.3&nbsp;µs $\color{#16a34a}{\text{−32.3\\%}}$ | 126.8&nbsp;µs $\color{#dc2626}{\text{+37.8\\%}}$ |
| **8K** | 130.8&nbsp;µs | 135.0&nbsp;µs $\color{#dc2626}{\text{+3.2\\%}}$ | 122.5&nbsp;µs $\color{#16a34a}{\text{−6.3\\%}}$ | 107.2&nbsp;µs $\color{#16a34a}{\text{−18.1\\%}}$ | 107.4&nbsp;µs $\color{#16a34a}{\text{−17.9\\%}}$ | 88.3&nbsp;µs $\color{#16a34a}{\text{−32.5\\%}}$ | 192.3&nbsp;µs $\color{#dc2626}{\text{+47.0\\%}}$ |
| **16K** | 220.0&nbsp;µs | 217.0&nbsp;µs $\color{grey}{\text{−1.4\\%}}$ | 193.2&nbsp;µs $\color{#16a34a}{\text{−12.2\\%}}$ | 169.7&nbsp;µs $\color{#16a34a}{\text{−22.9\\%}}$ | 167.7&nbsp;µs $\color{#16a34a}{\text{−23.8\\%}}$ | 119.5&nbsp;µs $\color{#16a34a}{\text{−45.7\\%}}$ | 327.5&nbsp;µs $\color{#dc2626}{\text{+48.9\\%}}$ |
| **32K** | 360.6&nbsp;µs | 373.6&nbsp;µs $\color{#dc2626}{\text{+3.6\\%}}$ | 332.7&nbsp;µs $\color{#16a34a}{\text{−7.7\\%}}$ | 285.5&nbsp;µs $\color{#16a34a}{\text{−20.8\\%}}$ | 287.6&nbsp;µs $\color{#16a34a}{\text{−20.3\\%}}$ | 186.2&nbsp;µs $\color{#16a34a}{\text{−48.4\\%}}$ | 621.4&nbsp;µs $\color{#dc2626}{\text{+72.3\\%}}$ |
| **64K** | 674.7&nbsp;µs | 692.5&nbsp;µs $\color{grey}{\text{+2.6\\%}}$ | 610.8&nbsp;µs $\color{#16a34a}{\text{−9.5\\%}}$ | 517.0&nbsp;µs $\color{#16a34a}{\text{−23.4\\%}}$ | 518.0&nbsp;µs $\color{#16a34a}{\text{−23.2\\%}}$ | 320.3&nbsp;µs $\color{#16a34a}{\text{−52.5\\%}}$ | 1.19&nbsp;ms $\color{#dc2626}{\text{+76.8\\%}}$ |
| **128K** | 1.31&nbsp;ms | 1.36&nbsp;ms $\color{#dc2626}{\text{+4.1\\%}}$ | 1.17&nbsp;ms $\color{#16a34a}{\text{−10.7\\%}}$ | 994.2&nbsp;µs $\color{#16a34a}{\text{−24.1\\%}}$ | 990.0&nbsp;µs $\color{#16a34a}{\text{−24.4\\%}}$ | 595.8&nbsp;µs $\color{#16a34a}{\text{−54.5\\%}}$ | 2.32&nbsp;ms $\color{#dc2626}{\text{+77.2\\%}}$ |
| **256K** | 2.62&nbsp;ms | 2.73&nbsp;ms $\color{#dc2626}{\text{+4.4\\%}}$ | 2.34&nbsp;ms $\color{#16a34a}{\text{−10.7\\%}}$ | 1.93&nbsp;ms $\color{#16a34a}{\text{−26.4\\%}}$ | 1.95&nbsp;ms $\color{#16a34a}{\text{−25.6\\%}}$ | 1.14&nbsp;ms $\color{#16a34a}{\text{−56.5\\%}}$ | 4.62&nbsp;ms $\color{#dc2626}{\text{+76.4\\%}}$ |

<details>
<summary>How the % and × here are calculated</summary>

- Wall-clock median per launch over the stage's iterations (CUDA events), every variant in the same run.
- **%** = (variant − first column) ÷ first column × 100 — the change in **time**: negative = faster. `−6.0%` = 6% less time than the first column.
- Green = at least 3% less time, red = at least 3% more, grey = within ±3% (`NOISE_PCT`).

</details><br />


<sub>Rendered from [`benchmarks/raw/verify/kvfmt/20261012-203801.json`](../../benchmarks/raw/verify/kvfmt/20261012-203801.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

## How to read it

- **layout**: fp8 on NHD vs interleaved vs HND isolates the page layout;
- **format**: on HND, fp6 reads 25% fewer bytes and fp4 50% fewer — compare the measured speedup with that ideal (and see [fp6prof](fp6prof.md) for why fp6 realizes less of it);
- **nvfp4** is limited by its conversion warpgroup, not by bytes;
- the tradeoff chart is the decision: up-and-left wins; e2m3 is the harness default because it sits next to fp8 on accuracy.

## Reproduce

```bash
mjolnir verify kvfmt
mjolnir verify kvfmt --variants "e2m3:--kv_format e2m3|e2m3-s3:--kv_format e2m3 --kv_stages 3"
```

## Ideas to try

- fp4 with per-channel K scales folded into Q and per-token V scales folded into P (KIVI-style), to bring e2m1's accuracy toward fp6.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../.github/CONTRIBUTING.md).

---

[← Head-major pages](hndsweep.md) · [case study](README.md) · [KV format accuracy →](quant.md)
