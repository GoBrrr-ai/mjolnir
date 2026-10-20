# KV ring depth

> TMA→MMA ring depth (--kv_stages) sweep at every L · `mjolnir verify ring` · run `20261012-210506`

## The question

How deep should the TMA → MMA ring be? A stage is freed only when the MMA that read it completes, so deeper rings let the loader run further ahead — but they cost shared memory, and with 2 CTAs per SM each CTA has ~113 KB.

## How it is measured

`--kv_stages 2…6` against the default (one whole KV tile at 2 CTAs/SM — 4 stages — for fp8; 2 stages for fp6, see the kernel's `_default_kv_stages`), on the harness default config, every L.

## Result

<img src="../../../../assets/benchmarks/thor/Qwen3.8-27B/ring-heatmap.png" alt="KV ring depth" width="100%">

- Fastest variant at the most context lengths: **`stages=5`** (2 of 8).
- Best geomean vs FlashInfer: **`stages=2`** (1.25×); worst: `stages=6` (1.16×).

| variant | flags | 128 | 4K | 8K | 16K | 32K | 64K | 128K | 256K | geomean |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `default` | (default) | $\color{#16a34a}{1.72\times}$ | $\color{#16a34a}{1.18\times}$ | $\color{grey}{1.03\times}$ | $\color{grey}{1.03\times}$ | **$\color{#16a34a}{1.16\times}$** | **$\color{#16a34a}{1.32\times}$** | $\color{#16a34a}{1.30\times}$ | $\color{#16a34a}{1.29\times}$ | $\color{#16a34a}{\mathbf{1.24\times}}$ |
| `stages=2` | `--kv_stages 2` | $\color{#16a34a}{1.71\times}$ | $\color{#16a34a}{1.17\times}$ | $\color{#16a34a}{1.11\times}$ | $\color{#16a34a}{1.05\times}$ | $\color{#16a34a}{1.15\times}$ | $\color{#16a34a}{1.30\times}$ | **$\color{#16a34a}{1.31\times}$** | **$\color{#16a34a}{1.30\times}$** | $\color{#16a34a}{\mathbf{1.25\times}}$ |
| `stages=3` | `--kv_stages 3` | $\color{#16a34a}{1.70\times}$ | $\color{#16a34a}{1.17\times}$ | $\color{#16a34a}{1.16\times}$ | $\color{grey}{1.01\times}$ | $\color{#16a34a}{1.05\times}$ | $\color{#16a34a}{1.28\times}$ | $\color{#16a34a}{1.18\times}$ | $\color{#16a34a}{1.22\times}$ | $\color{#16a34a}{\mathbf{1.21\times}}$ |
| `stages=4` | `--kv_stages 4` | **$\color{#16a34a}{1.73\times}$** | $\color{#16a34a}{1.17\times}$ | $\color{#16a34a}{1.16\times}$ | $\color{#16a34a}{1.08\times}$ | $\color{#16a34a}{1.09\times}$ | $\color{#16a34a}{1.23\times}$ | $\color{#16a34a}{1.16\times}$ | $\color{#16a34a}{1.18\times}$ | $\color{#16a34a}{\mathbf{1.21\times}}$ |
| `stages=5` | `--kv_stages 5` | $\color{#16a34a}{1.73\times}$ | **$\color{#16a34a}{1.20\times}$** | $\color{#16a34a}{1.16\times}$ | **$\color{#16a34a}{1.09\times}$** | $\color{#16a34a}{1.07\times}$ | $\color{#16a34a}{1.12\times}$ | $\color{#16a34a}{1.09\times}$ | $\color{#16a34a}{1.13\times}$ | $\color{#16a34a}{\mathbf{1.19\times}}$ |
| `stages=6` | `--kv_stages 6` | $\color{#16a34a}{1.70\times}$ | $\color{#16a34a}{1.18\times}$ | **$\color{#16a34a}{1.16\times}$** | $\color{#16a34a}{1.04\times}$ | $\color{grey}{1.01\times}$ | $\color{#16a34a}{1.11\times}$ | $\color{#16a34a}{1.10\times}$ | $\color{#16a34a}{1.13\times}$ | $\color{#16a34a}{\mathbf{1.16\times}}$ |

<details>
<summary>How the % and × here are calculated</summary>

- Each cell = **reference time ÷ variant time** (Nsight Compute kernel time, decode + merge, base clocks, cold L2, same run, same L). `1.10×` = the variant needs 1 ÷ 1.10 = 91% of the reference's time; below 1 it is slower.
- **geomean** = exp(mean of ln(ratio)) over the context lengths; **bold** = the fastest variant at that L.
- Green ≥ 1.03, grey within ±3% (`NOISE_PCT`, the run-to-run spread of a same-run ratio), red ≤ 0.97.

</details><br />

<sub>Rendered from [`benchmarks/raw/verify/ring/20261012-210506.json`](../../../../benchmarks/raw/verify/ring/20261012-210506.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

## How to read it

Depths that are not a multiple of a tile (K + V stages) rotate a tile's K/V across slots. Look for the depth that wins at most L by more than the noise band; a depth that only wins inside ±3% does not justify a change of default.

## Reproduce

```bash
mjolnir verify ring
mjolnir verify ring --base "--kv_format fp8" --stages "3 4 6"
```

## Ideas to try

- 1 CTA per SM with a deeper ring: `mjolnir verify ring --base "--ctas_per_sm 1"` (double the smem per CTA fits deeper rings).

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../../../.github/CONTRIBUTING.md).

---

[← Full profiles per KV format](fp6prof.md) · [case study](../README.md#kernel) · [L2 prefetch →](prefetch.md)
