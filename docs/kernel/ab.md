# Flag A/B

> any harness flag sets, ncu + wall clock (default: evict-first x PDL) · `mjolnir verify ab` · run `20261011-093214`

## The question

A general-purpose A/B for harness flags. The default run compares the L2 **evict-first** cache policy on K/V loads and PDL, alone and together.

## How it is measured

Any `name:flags|name:flags` list (`--variants`), ncu at every L plus a wall-clock row. Evict-first marks K/V lines as streaming so they do not displace the rest of the model's working set in the 32 MiB L2.

## Result

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../assets/benchmarks/ab-heatmap-dark.png">
  <img src="../../assets/benchmarks/ab-heatmap.png" alt="Flag A/B" width="100%">
</picture>

- Fastest variant at the most context lengths: **`pdl`** (4 of 8).
- Best geomean vs FlashInfer: **`pdl`** (1.26×); worst: `base` (1.22×).

| variant | flags | 128 | 4K | 8K | 16K | 32K | 64K | 128K | 256K | geomean |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `base` | `--kv_evict_first 0` | $\color{#16a34a}{1.90\times}$ | **$\color{#16a34a}{1.21\times}$** | $\color{#16a34a}{1.10\times}$ | $\color{grey}{0.98\times}$ | $\color{grey}{1.02\times}$ | $\color{#16a34a}{1.23\times}$ | $\color{#16a34a}{1.26\times}$ | $\color{#16a34a}{1.25\times}$ | $\color{#16a34a}{\mathbf{1.22\times}}$ |
| `evict` | `--kv_evict_first 1` | $\color{#16a34a}{1.90\times}$ | $\color{#16a34a}{1.18\times}$ | $\color{#16a34a}{1.16\times}$ | $\color{grey}{0.98\times}$ | $\color{#16a34a}{1.13\times}$ | $\color{#16a34a}{1.27\times}$ | $\color{#16a34a}{1.29\times}$ | $\color{#16a34a}{1.26\times}$ | $\color{#16a34a}{\mathbf{1.25\times}}$ |
| `pdl` | `--kv_evict_first 0 --pdl` | $\color{#16a34a}{1.90\times}$ | $\color{#16a34a}{1.17\times}$ | **$\color{#16a34a}{1.17\times}$** | **$\color{grey}{1.01\times}$** | **$\color{#16a34a}{1.15\times}$** | $\color{#16a34a}{1.27\times}$ | $\color{#16a34a}{1.31\times}$ | **$\color{#16a34a}{1.26\times}$** | $\color{#16a34a}{\mathbf{1.26\times}}$ |
| `both` | `--kv_evict_first 1 --pdl` | **$\color{#16a34a}{1.94\times}$** | $\color{#16a34a}{1.17\times}$ | $\color{#16a34a}{1.17\times}$ | $\color{grey}{0.98\times}$ | $\color{#16a34a}{1.14\times}$ | **$\color{#16a34a}{1.29\times}$** | **$\color{#16a34a}{1.32\times}$** | $\color{#16a34a}{1.25\times}$ | $\color{#16a34a}{\mathbf{1.26\times}}$ |


**Wall clock** (harness median, cold L2, b=1; Δ vs the first variant):

| L | base | evict | pdl | both |
|---|--:|--:|--:|--:|
| **128** | 14.2 µs | 14.2 µs $\color{grey}{\text{0.0\%}}$ | 14.1 µs $\color{grey}{\text{−0.5\%}}$ | 14.0 µs $\color{grey}{\text{−1.4\%}}$ |
| **4K** | 115.6 µs | 98.5 µs $\color{#16a34a}{\text{−14.8\%}}$ | 97.1 µs $\color{#16a34a}{\text{−16.0\%}}$ | 94.0 µs $\color{#16a34a}{\text{−18.7\%}}$ |
| **8K** | 173.8 µs | 134.0 µs $\color{#16a34a}{\text{−22.9\%}}$ | 182.1 µs $\color{#dc2626}{\text{+4.8\%}}$ | 131.9 µs $\color{#16a34a}{\text{−24.1\%}}$ |
| **16K** | 260.8 µs | 204.7 µs $\color{#16a34a}{\text{−21.5\%}}$ | 255.8 µs $\color{grey}{\text{−1.9\%}}$ | 197.4 µs $\color{#16a34a}{\text{−24.3\%}}$ |
| **32K** | 379.7 µs | 320.4 µs $\color{#16a34a}{\text{−15.6\%}}$ | 382.8 µs $\color{grey}{\text{+0.8\%}}$ | 317.2 µs $\color{#16a34a}{\text{−16.5\%}}$ |
| **64K** | 624.5 µs | 591.7 µs $\color{#16a34a}{\text{−5.2\%}}$ | 627.5 µs $\color{grey}{\text{+0.5\%}}$ | 565.1 µs $\color{#16a34a}{\text{−9.5\%}}$ |
| **128K** | 1.12 ms | 1.06 ms $\color{#16a34a}{\text{−5.5\%}}$ | 1.11 ms $\color{grey}{\text{−0.8\%}}$ | 1.06 ms $\color{#16a34a}{\text{−5.4\%}}$ |
| **256K** | 2.11 ms | 2.05 ms $\color{grey}{\text{−2.7\%}}$ | 2.12 ms $\color{grey}{\text{+0.7\%}}$ | 2.08 ms $\color{grey}{\text{−1.2\%}}$ |


<sub>Rendered from [`benchmarks/raw/verify/ab/20261011-093214.json`](../../benchmarks/raw/verify/ab/20261011-093214.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

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
