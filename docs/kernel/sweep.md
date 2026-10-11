# Headline L-sweep

> Mjolnir vs FlashInfer vs FlashAttention 2, ncu kernel time at every L · `mjolnir verify sweep` · run `20261011-113515`

## The question

The headline: how fast is one decode step with Mjolnir, FlashInfer and FlashAttention 2 — at every context length from 128 to 256K tokens? Short contexts are latency-bound (launch, plan, merge), long ones bandwidth-bound; a decode kernel has to win both.

## How it is measured

- one sequence, 24 query / 4 KV heads, head dim 256, 128-token pages in random order — the hd256 attention shape of Qwen3.8-27B;
- **ncu kernel time** of the decode kernel(s) + the split-KV merge, averaged over 50 launches, `--cache-control all` (cold L2: every launch reads KV from DRAM) and `--clock-control base`;
- legs: **Mjolnir e2m3** (the kernel's fastest config), **Mjolnir fp8** (what `--attention-backend MJOLNIR` serves today), **FlashInfer** tensor-core decode as vLLM calls it (fp8 KV, NHD pages), **FlashAttention 2** via `vllm_flash_attn` (bf16 KV — FA2 has no fp8 KV path);
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

- **Mjolnir (e2m3 KV, HND)** vs FlashInfer (fp8 KV, NHD): geomean **1.28×**, from 1.09× (L=16K) to 1.71× (L=128); faster beyond the ±3% noise band at 8 of 8 context lengths.
- **Mjolnir (fp8 KV, HND)** vs FlashInfer (fp8 KV, NHD): geomean **1.17×**, from 1.06× (L=32K) to 1.60× (L=128); faster beyond the ±3% noise band at 8 of 8 context lengths.
- **FlashAttention 2 (bf16 KV)** vs FlashInfer (fp8 KV, NHD): geomean **0.64×**, from 0.54× (L=16K) to 1.09× (L=128); faster beyond the ±3% noise band at 1 of 8 context lengths.

| L | FlashInfer | FA2 | Mjolnir fp8 | Mjolnir e2m3 | fp8 vs FI | e2m3 vs FI | fp8 vs FA2 | e2m3 vs FA2 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **128** | 22.1 µs | 20.3 µs | 13.8 µs | **12.9 µs** | $\color{#16a34a}{\mathbf{1.60\times}}$ | $\color{#16a34a}{\mathbf{1.71\times}}$ | $\color{#16a34a}{\mathbf{1.47\times}}$ | $\color{#16a34a}{\mathbf{1.57\times}}$ |
| **4K** | 60.9 µs | 95.2 µs | 56.4 µs | **51.9 µs** | $\color{#16a34a}{\mathbf{1.08\times}}$ | $\color{#16a34a}{\mathbf{1.17\times}}$ | $\color{#16a34a}{\mathbf{1.69\times}}$ | $\color{#16a34a}{\mathbf{1.83\times}}$ |
| **8K** | 96.8 µs | 180.7 µs | 87.2 µs | **84.6 µs** | $\color{#16a34a}{\mathbf{1.11\times}}$ | $\color{#16a34a}{\mathbf{1.14\times}}$ | $\color{#16a34a}{\mathbf{2.07\times}}$ | $\color{#16a34a}{\mathbf{2.13\times}}$ |
| **16K** | 189.3 µs | 353.3 µs | **169.3 µs** | 173.2 µs | $\color{#16a34a}{\mathbf{1.12\times}}$ | $\color{#16a34a}{\mathbf{1.09\times}}$ | $\color{#16a34a}{\mathbf{2.09\times}}$ | $\color{#16a34a}{\mathbf{2.04\times}}$ |
| **32K** | 384.0 µs | 668.9 µs | 362.6 µs | **334.6 µs** | $\color{#16a34a}{\mathbf{1.06\times}}$ | $\color{#16a34a}{\mathbf{1.15\times}}$ | $\color{#16a34a}{\mathbf{1.84\times}}$ | $\color{#16a34a}{\mathbf{2.00\times}}$ |
| **64K** | 814.9 µs | 1.28 ms | 693.2 µs | **602.3 µs** | $\color{#16a34a}{\mathbf{1.18\times}}$ | $\color{#16a34a}{\mathbf{1.35\times}}$ | $\color{#16a34a}{\mathbf{1.85\times}}$ | $\color{#16a34a}{\mathbf{2.13\times}}$ |
| **128K** | 1.48 ms | 2.42 ms | 1.30 ms | **1.10 ms** | $\color{#16a34a}{\mathbf{1.14\times}}$ | $\color{#16a34a}{\mathbf{1.35\times}}$ | $\color{#16a34a}{\mathbf{1.87\times}}$ | $\color{#16a34a}{\mathbf{2.21\times}}$ |
| **256K** | 2.83 ms | 4.63 ms | 2.45 ms | **2.07 ms** | $\color{#16a34a}{\mathbf{1.16\times}}$ | $\color{#16a34a}{\mathbf{1.37\times}}$ | $\color{#16a34a}{\mathbf{1.89\times}}$ | $\color{#16a34a}{\mathbf{2.24\times}}$ |
| geomean |  |  |  |  | $\color{#16a34a}{\mathbf{1.17\times}}$ | $\color{#16a34a}{\mathbf{1.28\times}}$ | $\color{#16a34a}{\mathbf{1.84\times}}$ | $\color{#16a34a}{\mathbf{2.01\times}}$ |

<sub>Rendered from [`benchmarks/raw/verify/sweep/20261011-113515.json`](../../benchmarks/raw/verify/sweep/20261011-113515.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

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
