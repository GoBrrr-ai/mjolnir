# Headline L-sweep

> Mjolnir vs FlashInfer vs FlashAttention 2, ncu kernel time at every L · `mjolnir verify sweep` · run `20261012-202029`

## The question

The headline: how fast is one decode step with Mjolnir, FlashInfer and FlashAttention 2 — at every context length from 128 to 256K tokens? Short contexts are latency-bound (launch, plan, merge), long ones bandwidth-bound; a decode kernel has to win both.

## How it is measured

- one sequence, 24 query / 4 KV heads, head dim 256, 128-token pages in random order — the hd256 attention shape of Qwen3.8-27B;
- **ncu kernel time** of the decode kernel(s) + the split-KV merge, averaged over 50 launches, `--cache-control all` (cold L2: every launch reads KV from DRAM) and `--clock-control base`;
- legs: **Mjolnir e2m3** (the kernel's fastest config), **Mjolnir fp8** (the `NVFP4_MJOLNIR` config; the default serves e2m3), **FlashInfer** tensor-core decode as vLLM calls it (fp8 KV, NHD pages), **FlashAttention 2** via `vllm_flash_attn` (bf16 KV — FA2 has no fp8 KV path);
- bandwidth = each leg's *own* KV bytes ÷ time (e2m3 = 6 bit, fp8 = 8 bit, bf16 = 16 bit).

## Result

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../assets/benchmarks/kernel-speedup-dark.png">
  <img src="../../assets/benchmarks/kernel-speedup.png" alt="Mjolnir speedup over FlashInfer" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../assets/benchmarks/kernel-latency-dark.png">
  <img src="../../assets/benchmarks/kernel-latency.png" alt="Decode latency per step" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../assets/benchmarks/kernel-bandwidth-dark.png">
  <img src="../../assets/benchmarks/kernel-bandwidth.png" alt="Effective KV bandwidth" width="100%">
</picture>

- **Mjolnir (e2m3 KV, HND)** vs FlashInfer (fp8 KV, NHD): geomean **1.24×**, from 0.99× (L=16K) to 1.64× (L=128); faster beyond the ±3% noise band at 7 of 8 context lengths.
- **Mjolnir (fp8 KV, HND)** vs FlashInfer (fp8 KV, NHD): geomean **1.15×**, from 1.04× (L=16K) to 1.59× (L=128); faster beyond the ±3% noise band at 8 of 8 context lengths.
- **FlashAttention 2 (bf16 KV)** vs FlashInfer (fp8 KV, NHD): geomean **0.62×**, from 0.49× (L=16K) to 1.09× (L=128); faster beyond the ±3% noise band at 1 of 8 context lengths.

| L | FlashInfer | FA2 | Mjolnir fp8 | Mjolnir e2m3 | fp8 vs FI | e2m3 vs FI | fp8 vs FA2 | e2m3 vs FA2 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **128** | 21.5&nbsp;µs | 19.7&nbsp;µs | 13.5&nbsp;µs | **13.1&nbsp;µs** | $\color{#16a34a}{\mathbf{1.59\times}}$ | $\color{#16a34a}{\mathbf{1.64\times}}$ | $\color{#16a34a}{\mathbf{1.45\times}}$ | $\color{#16a34a}{\mathbf{1.50\times}}$ |
| **4K** | 61.2&nbsp;µs | 94.9&nbsp;µs | 56.5&nbsp;µs | **50.2&nbsp;µs** | $\color{#16a34a}{\mathbf{1.08\times}}$ | $\color{#16a34a}{\mathbf{1.22\times}}$ | $\color{#16a34a}{\mathbf{1.68\times}}$ | $\color{#16a34a}{\mathbf{1.89\times}}$ |
| **8K** | 94.9&nbsp;µs | 180.6&nbsp;µs | **88.2&nbsp;µs** | 89.0&nbsp;µs | $\color{#16a34a}{\mathbf{1.08\times}}$ | $\color{#16a34a}{\mathbf{1.07\times}}$ | $\color{#16a34a}{\mathbf{2.05\times}}$ | $\color{#16a34a}{\mathbf{2.03\times}}$ |
| **16K** | 174.1&nbsp;µs | 355.3&nbsp;µs | **167.1&nbsp;µs** | 176.2&nbsp;µs | $\color{#16a34a}{\mathbf{1.04\times}}$ | $\color{grey}{0.99\times}$ | $\color{#16a34a}{\mathbf{2.13\times}}$ | $\color{#16a34a}{\mathbf{2.02\times}}$ |
| **32K** | 375.2&nbsp;µs | 664.3&nbsp;µs | 343.0&nbsp;µs | **324.4&nbsp;µs** | $\color{#16a34a}{\mathbf{1.09\times}}$ | $\color{#16a34a}{\mathbf{1.16\times}}$ | $\color{#16a34a}{\mathbf{1.94\times}}$ | $\color{#16a34a}{\mathbf{2.05\times}}$ |
| **64K** | 783.9&nbsp;µs | 1.28&nbsp;ms | 670.5&nbsp;µs | **599.8&nbsp;µs** | $\color{#16a34a}{\mathbf{1.17\times}}$ | $\color{#16a34a}{\mathbf{1.31\times}}$ | $\color{#16a34a}{\mathbf{1.91\times}}$ | $\color{#16a34a}{\mathbf{2.14\times}}$ |
| **128K** | 1.41&nbsp;ms | 2.41&nbsp;ms | 1.27&nbsp;ms | **1.08&nbsp;ms** | $\color{#16a34a}{\mathbf{1.11\times}}$ | $\color{#16a34a}{\mathbf{1.31\times}}$ | $\color{#16a34a}{\mathbf{1.91\times}}$ | $\color{#16a34a}{\mathbf{2.24\times}}$ |
| **256K** | 2.70&nbsp;ms | 4.63&nbsp;ms | 2.42&nbsp;ms | **2.06&nbsp;ms** | $\color{#16a34a}{\mathbf{1.12\times}}$ | $\color{#16a34a}{\mathbf{1.31\times}}$ | $\color{#16a34a}{\mathbf{1.92\times}}$ | $\color{#16a34a}{\mathbf{2.25\times}}$ |
| geomean |  |  |  |  | $\color{#16a34a}{\mathbf{1.15\times}}$ | $\color{#16a34a}{\mathbf{1.24\times}}$ | $\color{#16a34a}{\mathbf{1.86\times}}$ | $\color{#16a34a}{\mathbf{2.00\times}}$ |

<details>
<summary>How the % and × here are calculated</summary>

- **Time per decode step** = Nsight Compute kernel time (`gpu__time_duration`) of the decode kernel **plus** its split-KV merge kernel, at base clocks with a cold L2, median over the iterations. The reference (FlashInfer, FlashAttention 2) is measured the same way, at the same L, in the same run.
- **Speedup ×** = reference time ÷ Mjolnir time. `1.24×` means Mjolnir needs 1 ÷ 1.24 = 81% of the reference's time per step (19% less time); below 1 the reference is faster.
- **Geomean over L** = exp(mean of ln(speedup at each L)): every context length weighs the same, and one large ratio cannot dominate the way it would in an arithmetic mean.
- **Colors / "beyond the noise band"**: green = speedup ≥ 1.03, grey = within ±3%, red ≤ 0.97. ±3% (`NOISE_PCT`) is how far a same-run ratio moves between repeated runs on a dedicated Thor.

</details>

<sub>Rendered from [`benchmarks/raw/verify/sweep/20261012-202029.json`](../../benchmarks/raw/verify/sweep/20261012-202029.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

## How to read it

At L=128 the cache fits in one tile and time is mostly fixed cost: plan, launch, merge. From ~16K the cache no longer fits the 32 MiB L2 and every kernel converges toward the DRAM roofline — the remaining gap is how much of the peak each kernel sustains (bandwidth chart) and how many bytes it has to move (KV format). FA2 pays twice the bytes of every fp8 leg.

## Reproduce

```bash
mjolnir verify sweep
mjolnir verify sweep --legs fi,fi-hnd,mjolnir-fp8,mjolnir --ls "8192 65536"   # custom (not a headline run)
```

## Ideas to try

- batch > 1 and varlen batches (`--batches 8 --varlen` on the harness);
- page size 16 (what vLLM's FlashInfer backend uses on Thor) via `--page-size 16`.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../.github/CONTRIBUTING.md).

---

[← Correctness matrix](correctness.md) · [case study](README.md) · [Head-major pages →](hndsweep.md)
