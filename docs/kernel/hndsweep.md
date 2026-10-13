# Head-major pages

> FlashInfer on NHD and HND vs Mjolnir on HND — is HND a kernel win or a layout win? · `mjolnir verify hndsweep` · run `20261012-203356`

## The question

Mjolnir reads K/V from **head-major (HND)** pages, where each KV head's page is one contiguous block, instead of vLLM's default token-major layout. Is that a kernel advantage, or a layout that would make FlashInfer just as fast?

## How it is measured

Same data in both layouts: the FlashInfer driver generates the cache once in NHD order `(pages, 2, PS, h_k, d)` and transposes it to HND `(pages, 2, h_k, PS, d)`; Mjolnir (fp8 KV) runs on HND. All legs are relative to FlashInfer on NHD.

## Result

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../assets/benchmarks/hnd-layout-dark.png">
  <img src="../../assets/benchmarks/hnd-layout.png" alt="Head-major KV pages" width="100%">
</picture>

- **Mjolnir (fp8 KV, HND)** vs FlashInfer (fp8 KV, NHD): geomean **1.15×**, from 1.00× (L=16K) to 1.65× (L=128); faster beyond the ±3% noise band at 7 of 8 context lengths.
- **FlashInfer (fp8 KV, HND)** vs FlashInfer (fp8 KV, NHD): geomean **1.04×**, from 0.99× (L=128) to 1.08× (L=64K); faster beyond the ±3% noise band at 4 of 8 context lengths.

| L | FlashInfer | FlashInfer HND | Mjolnir fp8 | fp8 vs FI | fp8 vs FI-HND |
|---|--:|--:|--:|--:|--:|
| **128** | 22.2&nbsp;µs | 22.4&nbsp;µs | **13.5&nbsp;µs** | $\color{#16a34a}{\mathbf{1.65\times}}$ | $\color{#16a34a}{\mathbf{1.66\times}}$ |
| **4K** | 61.1&nbsp;µs | 59.6&nbsp;µs | **54.6&nbsp;µs** | $\color{#16a34a}{\mathbf{1.12\times}}$ | $\color{#16a34a}{\mathbf{1.09\times}}$ |
| **8K** | 95.8&nbsp;µs | 93.0&nbsp;µs | **88.9&nbsp;µs** | $\color{#16a34a}{\mathbf{1.08\times}}$ | $\color{#16a34a}{\mathbf{1.05\times}}$ |
| **16K** | 175.7&nbsp;µs | **174.0&nbsp;µs** | 175.1&nbsp;µs | $\color{grey}{1.00\times}$ | $\color{grey}{0.99\times}$ |
| **32K** | 372.9&nbsp;µs | 353.5&nbsp;µs | **336.8&nbsp;µs** | $\color{#16a34a}{\mathbf{1.11\times}}$ | $\color{#16a34a}{\mathbf{1.05\times}}$ |
| **64K** | 782.3&nbsp;µs | 727.1&nbsp;µs | **670.7&nbsp;µs** | $\color{#16a34a}{\mathbf{1.17\times}}$ | $\color{#16a34a}{\mathbf{1.08\times}}$ |
| **128K** | 1.40&nbsp;ms | 1.33&nbsp;ms | **1.27&nbsp;ms** | $\color{#16a34a}{\mathbf{1.10\times}}$ | $\color{#16a34a}{\mathbf{1.05\times}}$ |
| **256K** | 2.70&nbsp;ms | 2.54&nbsp;ms | **2.42&nbsp;ms** | $\color{#16a34a}{\mathbf{1.12\times}}$ | $\color{#16a34a}{\mathbf{1.05\times}}$ |
| geomean |  |  |  | $\color{#16a34a}{\mathbf{1.15\times}}$ | $\color{#16a34a}{\mathbf{1.11\times}}$ |

<details>
<summary>How the % and × here are calculated</summary>

- **Time per decode step** = Nsight Compute kernel time (`gpu__time_duration`) of the decode kernel **plus** its split-KV merge kernel, at base clocks with a cold L2, median over the iterations. The reference (FlashInfer, FlashAttention 2) is measured the same way, at the same L, in the same run.
- **Speedup ×** = reference time ÷ Mjolnir time. `1.24×` means Mjolnir needs 1 ÷ 1.24 = 81% of the reference's time per step (19% less time); below 1 the reference is faster.
- **Geomean over L** = exp(mean of ln(speedup at each L)): every context length weighs the same, and one large ratio cannot dominate the way it would in an arithmetic mean.
- **Colors / "beyond the noise band"**: green = speedup ≥ 1.03, grey = within ±3%, red ≤ 0.97. ±3% (`NOISE_PCT`) is how far a same-run ratio moves between repeated runs on a dedicated Thor.

</details>

<sub>Rendered from [`benchmarks/raw/verify/hndsweep/20261012-203356.json`](../../benchmarks/raw/verify/hndsweep/20261012-203356.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

## How to read it

If FlashInfer-HND climbed to Mjolnir's line, HND would be a layout win only. The gap between the two HND lines is what the kernel itself adds: TMA boxes of one head's contiguous page stream in longer DRAM bursts, which Mjolnir's pipeline converts into speed more fully than FlashInfer's.

In vLLM, `VLLM_KV_CACHE_LAYOUT=HND` allocates this layout (`mjolnir serve` sets it for MJOLNIR configs). vLLM's interleaved `(B, H, N, 2D)` layout is head-major too, but K and V alternate per row, so each stream reads every other block — see [kvfmt](kvfmt.md).

## Reproduce

```bash
mjolnir verify hndsweep
```

## Ideas to try

- FlashInfer at page size 16 and 64 (`--page-size`), where its native vLLM path runs.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../.github/CONTRIBUTING.md).

---

[← Headline L-sweep](sweep.md) · [case study](README.md) · [KV layouts and formats →](kvfmt.md)
