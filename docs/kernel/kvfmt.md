# KV layouts and formats

> fp8 / fp6 / fp4 / nvfp4 KV on NHD / interleaved / HND pages (ncu + wall clock) · `mjolnir verify kvfmt` · run `20261011-115251`

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
- Best geomean vs FlashInfer: **`e2m1-hnd`** (1.94×); worst: `nvfp4-hnd` (0.58×).

| variant | flags | 128 | 4K | 8K | 16K | 32K | 64K | 128K | 256K | geomean |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `fp8-nhd` | `--kv_format fp8 --kv_layout nhd` | $\color{#16a34a}{1.78\times}$ | $\color{#16a34a}{1.07\times}$ | $\color{#dc2626}{0.92\times}$ | $\color{#dc2626}{0.97\times}$ | $\color{#dc2626}{0.92\times}$ | $\color{grey}{0.99\times}$ | $\color{#dc2626}{0.97\times}$ | $\color{grey}{0.98\times}$ | $\color{#16a34a}{\mathbf{1.05\times}}$ |
| `fp8-interleaved` | `--kv_format fp8 --kv_layout interleaved` | $\color{#16a34a}{1.75\times}$ | $\color{#dc2626}{0.95\times}$ | $\color{#dc2626}{0.85\times}$ | $\color{grey}{0.97\times}$ | $\color{#dc2626}{0.88\times}$ | $\color{#dc2626}{0.97\times}$ | $\color{#dc2626}{0.90\times}$ | $\color{#dc2626}{0.97\times}$ | $\color{grey}{1.00\times}$ |
| `fp8-hnd` | `--kv_format fp8 --kv_layout hnd` | $\color{#16a34a}{1.71\times}$ | $\color{#16a34a}{1.07\times}$ | $\color{#16a34a}{1.10\times}$ | $\color{#16a34a}{1.09\times}$ | $\color{#16a34a}{1.05\times}$ | $\color{#16a34a}{1.18\times}$ | $\color{#16a34a}{1.11\times}$ | $\color{#16a34a}{1.19\times}$ | $\color{#16a34a}{\mathbf{1.17\times}}$ |
| `e2m3-hnd` | `--kv_format e2m3 --kv_layout hnd` | **$\color{#16a34a}{1.79\times}$** | $\color{#16a34a}{1.18\times}$ | $\color{#16a34a}{1.16\times}$ | $\color{#16a34a}{1.08\times}$ | $\color{#16a34a}{1.16\times}$ | $\color{#16a34a}{1.35\times}$ | $\color{#16a34a}{1.35\times}$ | $\color{#16a34a}{1.38\times}$ | $\color{#16a34a}{\mathbf{1.29\times}}$ |
| `e3m2-hnd` | `--kv_format e3m2 --kv_layout hnd` | $\color{#16a34a}{1.76\times}$ | $\color{#16a34a}{1.19\times}$ | $\color{#16a34a}{1.04\times}$ | $\color{#16a34a}{1.03\times}$ | $\color{#16a34a}{1.14\times}$ | $\color{#16a34a}{1.36\times}$ | $\color{#16a34a}{1.34\times}$ | $\color{#16a34a}{1.39\times}$ | $\color{#16a34a}{\mathbf{1.26\times}}$ |
| `e2m1-hnd` | `--kv_format e2m1 --kv_layout hnd` | $\color{#16a34a}{1.79\times}$ | **$\color{#16a34a}{1.51\times}$** | **$\color{#16a34a}{1.69\times}$** | **$\color{#16a34a}{1.89\times}$** | **$\color{#16a34a}{2.02\times}$** | **$\color{#16a34a}{2.25\times}$** | **$\color{#16a34a}{2.22\times}$** | **$\color{#16a34a}{2.34\times}$** | $\color{#16a34a}{\mathbf{1.94\times}}$ |
| `nvfp4-hnd` | `--kv_format nvfp4 --kv_layout hnd` | $\color{#dc2626}{0.64\times}$ | $\color{#dc2626}{0.49\times}$ | $\color{#dc2626}{0.48\times}$ | $\color{#dc2626}{0.55\times}$ | $\color{#dc2626}{0.61\times}$ | $\color{#dc2626}{0.67\times}$ | $\color{#dc2626}{0.63\times}$ | $\color{#dc2626}{0.62\times}$ | $\color{#dc2626}{0.58\times}$ |


**Wall clock** (harness median, cold L2, b=1; Δ vs the first variant):

| L | fp8-nhd | fp8-interleaved | fp8-hnd | e2m3-hnd | e3m2-hnd | e2m1-hnd | nvfp4-hnd |
|---|--:|--:|--:|--:|--:|--:|--:|
| **128** | 14.2 µs | 12.3 µs $\color{#16a34a}{\text{−13.3\%}}$ | 13.7 µs $\color{#16a34a}{\text{−3.9\%}}$ | 14.2 µs $\color{grey}{\text{−0.2\%}}$ | 14.1 µs $\color{grey}{\text{−0.5\%}}$ | 12.1 µs $\color{#16a34a}{\text{−14.6\%}}$ | 22.3 µs $\color{#dc2626}{\text{+57.2\%}}$ |
| **4K** | 107.3 µs | 132.3 µs $\color{#dc2626}{\text{+23.3\%}}$ | 107.3 µs $\color{grey}{\text{−0.0\%}}$ | 92.9 µs $\color{#16a34a}{\text{−13.4\%}}$ | 97.9 µs $\color{#16a34a}{\text{−8.8\%}}$ | 76.6 µs $\color{#16a34a}{\text{−28.6\%}}$ | 136.0 µs $\color{#dc2626}{\text{+26.7\%}}$ |
| **8K** | 168.8 µs | 161.6 µs $\color{#16a34a}{\text{−4.2\%}}$ | 166.7 µs $\color{grey}{\text{−1.2\%}}$ | 141.2 µs $\color{#16a34a}{\text{−16.4\%}}$ | 143.1 µs $\color{#16a34a}{\text{−15.2\%}}$ | 102.2 µs $\color{#16a34a}{\text{−39.4\%}}$ | 201.5 µs $\color{#dc2626}{\text{+19.4\%}}$ |
| **16K** | 248.6 µs | 277.3 µs $\color{#dc2626}{\text{+11.5\%}}$ | 254.6 µs $\color{grey}{\text{+2.4\%}}$ | 195.4 µs $\color{#16a34a}{\text{−21.4\%}}$ | 194.4 µs $\color{#16a34a}{\text{−21.8\%}}$ | 157.8 µs $\color{#16a34a}{\text{−36.6\%}}$ | 346.9 µs $\color{#dc2626}{\text{+39.5\%}}$ |
| **32K** | 416.2 µs | 449.4 µs $\color{#dc2626}{\text{+8.0\%}}$ | 393.0 µs $\color{#16a34a}{\text{−5.6\%}}$ | 314.2 µs $\color{#16a34a}{\text{−24.5\%}}$ | 314.3 µs $\color{#16a34a}{\text{−24.5\%}}$ | 215.9 µs $\color{#16a34a}{\text{−48.1\%}}$ | 634.7 µs $\color{#dc2626}{\text{+52.5\%}}$ |
| **64K** | 749.6 µs | 792.8 µs $\color{#dc2626}{\text{+5.8\%}}$ | 657.5 µs $\color{#16a34a}{\text{−12.3\%}}$ | 552.8 µs $\color{#16a34a}{\text{−26.3\%}}$ | 553.8 µs $\color{#16a34a}{\text{−26.1\%}}$ | 360.3 µs $\color{#16a34a}{\text{−51.9\%}}$ | 1.21 ms $\color{#dc2626}{\text{+60.8\%}}$ |
| **128K** | 1.57 ms | 1.58 ms $\color{grey}{\text{+0.8\%}}$ | 1.22 ms $\color{#16a34a}{\text{−22.3\%}}$ | 1.03 ms $\color{#16a34a}{\text{−34.3\%}}$ | 1.03 ms $\color{#16a34a}{\text{−34.5\%}}$ | 631.6 µs $\color{#16a34a}{\text{−59.8\%}}$ | 2.32 ms $\color{#dc2626}{\text{+47.6\%}}$ |
| **256K** | 3.15 ms | 2.96 ms $\color{#16a34a}{\text{−6.0\%}}$ | 2.41 ms $\color{#16a34a}{\text{−23.6\%}}$ | 2.00 ms $\color{#16a34a}{\text{−36.7\%}}$ | 1.97 ms $\color{#16a34a}{\text{−37.4\%}}$ | 1.18 ms $\color{#16a34a}{\text{−62.4\%}}$ | 4.65 ms $\color{#dc2626}{\text{+47.5\%}}$ |


<sub>Rendered from [`benchmarks/raw/verify/kvfmt/20261011-115251.json`](../../benchmarks/raw/verify/kvfmt/20261011-115251.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

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

- fp4 with per-channel K scales folded into Q and per-token V scales folded into P (KIVI-style), to bring e2m1's accuracy toward fp6;
- a vLLM cache-write path for e2m3, so serving can use the 6-bit cache.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../.github/CONTRIBUTING.md).

---

[← Head-major pages](hndsweep.md) · [case study](README.md) · [KV format accuracy →](quant.md)
