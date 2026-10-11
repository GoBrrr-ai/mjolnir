# Head-major pages

> FlashInfer on NHD and HND vs Mjolnir on HND — is HND a kernel win or a layout win? · `mjolnir verify hndsweep` · run `20261011-114842`

## The question

Mjolnir reads K/V from **head-major (HND)** pages, where each KV head's page is one contiguous block, instead of vLLM's default token-major layout. Is that a kernel advantage, or a layout that would make FlashInfer just as fast?

## How it is measured

Same data in both layouts: the FlashInfer driver generates the cache once in NHD order `(pages, 2, PS, h_k, d)` and transposes it to HND `(pages, 2, h_k, PS, d)`; Mjolnir (fp8 KV) runs on HND. All legs are relative to FlashInfer on NHD.

## Result

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../assets/benchmarks/hnd-layout-dark.png">
  <img src="../../assets/benchmarks/hnd-layout.png" alt="Head-major KV pages" width="100%">
</picture>

- **Mjolnir (fp8 KV, HND)** vs FlashInfer (fp8 KV, NHD): geomean **1.16×**, from 0.99× (L=16K) to 1.71× (L=128); faster beyond the ±3% noise band at 7 of 8 context lengths.
- **FlashInfer (fp8 KV, HND)** vs FlashInfer (fp8 KV, NHD): geomean **1.06×**, from 1.00× (L=16K) to 1.14× (L=64K); faster beyond the ±3% noise band at 7 of 8 context lengths.

| L | FlashInfer | FlashInfer HND | Mjolnir fp8 | fp8 vs FI | fp8 vs FI-HND |
|---|--:|--:|--:|--:|--:|
| **128** | 23.2 µs | 22.0 µs | **13.6 µs** | $\color{#16a34a}{\mathbf{1.71\times}}$ | $\color{#16a34a}{\mathbf{1.62\times}}$ |
| **4K** | 61.0 µs | 59.0 µs | **56.2 µs** | $\color{#16a34a}{\mathbf{1.08\times}}$ | $\color{#16a34a}{\mathbf{1.05\times}}$ |
| **8K** | 95.3 µs | 91.5 µs | **90.6 µs** | $\color{#16a34a}{\mathbf{1.05\times}}$ | $\color{grey}{1.01\times}$ |
| **16K** | **174.4 µs** | 175.2 µs | 176.1 µs | $\color{grey}{0.99\times}$ | $\color{grey}{1.00\times}$ |
| **32K** | 372.5 µs | 358.7 µs | **346.5 µs** | $\color{#16a34a}{\mathbf{1.07\times}}$ | $\color{#16a34a}{\mathbf{1.04\times}}$ |
| **64K** | 828.7 µs | 726.0 µs | **704.2 µs** | $\color{#16a34a}{\mathbf{1.18\times}}$ | $\color{#16a34a}{\mathbf{1.03\times}}$ |
| **128K** | 1.48 ms | 1.35 ms | **1.29 ms** | $\color{#16a34a}{\mathbf{1.15\times}}$ | $\color{#16a34a}{\mathbf{1.05\times}}$ |
| **256K** | 2.85 ms | 2.56 ms | **2.41 ms** | $\color{#16a34a}{\mathbf{1.18\times}}$ | $\color{#16a34a}{\mathbf{1.06\times}}$ |
| geomean |  |  |  | $\color{#16a34a}{\mathbf{1.16\times}}$ | $\color{#16a34a}{\mathbf{1.09\times}}$ |

<sub>Rendered from [`benchmarks/raw/verify/hndsweep/20261011-114842.json`](../../benchmarks/raw/verify/hndsweep/20261011-114842.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

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
