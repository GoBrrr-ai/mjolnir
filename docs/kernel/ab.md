# Flag A/B

> any harness flag sets, ncu + wall clock (default: evict-first x PDL) · `mjolnir verify ab` · run `20261011-124125`

## The question

A general-purpose A/B for harness flags. The default run compares the L2 **evict-first** cache policy on K/V loads and PDL, alone and together.

## How it is measured

Any `name:flags|name:flags` list (`--variants`), ncu at every L plus a wall-clock row. Evict-first marks K/V lines as streaming so they do not displace the rest of the model's working set in the 32 MiB L2.

## Result

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../assets/benchmarks/ab-heatmap-dark.png">
  <img src="../../assets/benchmarks/ab-heatmap.png" alt="Flag A/B" width="100%">
</picture>

- Fastest variant at the most context lengths: **`pdl`** (3 of 8).
- Best geomean vs FlashInfer: **`both`** (1.27×); worst: `base` (1.26×).
- All variants sit within 1% of each other — inside the noise band.

| variant | flags | 128 | 4K | 8K | 16K | 32K | 64K | 128K | 256K | geomean |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `base` | `--kv_evict_first 0` | $\color{#16a34a}{1.69\times}$ | **$\color{#16a34a}{1.23\times}$** | $\color{#16a34a}{1.04\times}$ | $\color{grey}{1.02\times}$ | $\color{#16a34a}{1.13\times}$ | $\color{#16a34a}{1.39\times}$ | $\color{#16a34a}{1.32\times}$ | $\color{#16a34a}{1.34\times}$ | $\color{#16a34a}{\mathbf{1.26\times}}$ |
| `evict` | `--kv_evict_first 1` | $\color{#16a34a}{1.70\times}$ | $\color{#16a34a}{1.20\times}$ | **$\color{#16a34a}{1.18\times}$** | **$\color{grey}{1.03\times}$** | $\color{#16a34a}{1.10\times}$ | $\color{#16a34a}{1.38\times}$ | $\color{#16a34a}{1.31\times}$ | $\color{#16a34a}{1.36\times}$ | $\color{#16a34a}{\mathbf{1.27\times}}$ |
| `pdl` | `--kv_evict_first 0 --pdl` | $\color{#16a34a}{1.62\times}$ | $\color{#16a34a}{1.21\times}$ | $\color{#16a34a}{1.09\times}$ | $\color{grey}{1.00\times}$ | **$\color{#16a34a}{1.18\times}$** | $\color{#16a34a}{1.38\times}$ | **$\color{#16a34a}{1.35\times}$** | **$\color{#16a34a}{1.36\times}$** | $\color{#16a34a}{\mathbf{1.26\times}}$ |
| `both` | `--kv_evict_first 1 --pdl` | **$\color{#16a34a}{1.72\times}$** | $\color{#16a34a}{1.19\times}$ | $\color{#16a34a}{1.15\times}$ | $\color{grey}{1.02\times}$ | $\color{#16a34a}{1.14\times}$ | **$\color{#16a34a}{1.39\times}$** | $\color{#16a34a}{1.32\times}$ | $\color{#16a34a}{1.35\times}$ | $\color{#16a34a}{\mathbf{1.27\times}}$ |


**Wall clock** (harness median, cold L2, b=1; Δ vs the first variant):

| L | base | evict | pdl | both |
|---|--:|--:|--:|--:|
| **128** | 12.2 µs | 14.1 µs $\color{#dc2626}{\text{+16.3\\%}}$ | 14.2 µs $\color{#dc2626}{\text{+16.6\\%}}$ | 14.2 µs $\color{#dc2626}{\text{+16.6\\%}}$ |
| **4K** | 114.5 µs | 106.3 µs $\color{#16a34a}{\text{−7.2\\%}}$ | 113.5 µs $\color{grey}{\text{−0.9\\%}}$ | 94.5 µs $\color{#16a34a}{\text{−17.5\\%}}$ |
| **8K** | 200.5 µs | 141.1 µs $\color{#16a34a}{\text{−29.6\\%}}$ | 172.9 µs $\color{#16a34a}{\text{−13.8\\%}}$ | 134.0 µs $\color{#16a34a}{\text{−33.2\\%}}$ |
| **16K** | 275.4 µs | 195.4 µs $\color{#16a34a}{\text{−29.0\\%}}$ | 260.7 µs $\color{#16a34a}{\text{−5.3\\%}}$ | 208.7 µs $\color{#16a34a}{\text{−24.2\\%}}$ |
| **32K** | 393.9 µs | 331.6 µs $\color{#16a34a}{\text{−15.8\\%}}$ | 393.1 µs $\color{grey}{\text{−0.2\\%}}$ | 335.8 µs $\color{#16a34a}{\text{−14.8\\%}}$ |
| **64K** | 643.1 µs | 560.9 µs $\color{#16a34a}{\text{−12.8\\%}}$ | 630.5 µs $\color{grey}{\text{−2.0\\%}}$ | 566.1 µs $\color{#16a34a}{\text{−12.0\\%}}$ |
| **128K** | 1.10 ms | 1.05 ms $\color{#16a34a}{\text{−4.2\\%}}$ | 1.11 ms $\color{grey}{\text{+0.7\\%}}$ | 1.05 ms $\color{#16a34a}{\text{−4.8\\%}}$ |
| **256K** | 2.07 ms | 2.00 ms $\color{#16a34a}{\text{−3.1\\%}}$ | 2.03 ms $\color{grey}{\text{−1.5\\%}}$ | 1.98 ms $\color{#16a34a}{\text{−4.2\\%}}$ |


<sub>Rendered from [`benchmarks/raw/verify/ab/20261011-124125.json`](../../benchmarks/raw/verify/ab/20261011-124125.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

## How to read it

Use it for any idea: change a flag in the harness, run `mjolnir verify ab --variants ...`, and the heatmap shows where it wins. Custom runs are saved but never replace this page's default chart.

## Reproduce

```bash
mjolnir verify ab
mjolnir verify ab --variants "base:|mine:--my_flag 1" --ls "8192 65536 262144"
```

## Ideas to try

- anything — this is the experiment slot. `mjolnir autoresearch` uses it to test agent-proposed changes.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../.github/CONTRIBUTING.md).

---

[← Programmatic dependent launch](pdl.md) · [case study](README.md)
