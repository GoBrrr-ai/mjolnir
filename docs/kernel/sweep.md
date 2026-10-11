# Headline L-sweep

> Mjolnir vs FlashInfer vs FlashAttention 2, ncu kernel time at every L · `mjolnir verify sweep` · run `20261011-082714`

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

- **Mjolnir (e2m3 KV, HND)** vs FlashInfer (fp8 KV, NHD): geomean **1.26×**, from 1.07× (L=16K) to 1.92× (L=128); faster beyond the ±7% noise band at 7 of 8 context lengths.
- **Mjolnir (fp8 KV, HND)** vs FlashInfer (fp8 KV, NHD): geomean **1.21×**, from 1.04× (L=16K) to 1.87× (L=128); faster beyond the ±7% noise band at 7 of 8 context lengths.
- **FlashAttention 2 (bf16 KV)** vs FlashInfer (fp8 KV, NHD): geomean **0.68×**, from 0.52× (L=8K) to 1.31× (L=128); faster beyond the ±7% noise band at 1 of 8 context lengths.

| L | FlashInfer | FA2 | Mjolnir fp8 | Mjolnir e2m3 | fp8 vs FI | e2m3 vs FI | fp8 vs FA2 | e2m3 vs FA2 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **128** | 25.5 µs | 19.5 µs | 13.6 µs | **13.3 µs** | $\color{#16a34a}{\mathbf{1.87\times}}$ | $\color{#16a34a}{\mathbf{1.92\times}}$ | $\color{#16a34a}{\mathbf{1.43\times}}$ | $\color{#16a34a}{\mathbf{1.47\times}}$ |
| **4K** | 60.3 µs | 88.8 µs | 54.8 µs | **49.4 µs** | $\color{#16a34a}{\mathbf{1.10\times}}$ | $\color{#16a34a}{\mathbf{1.22\times}}$ | $\color{#16a34a}{\mathbf{1.62\times}}$ | $\color{#16a34a}{\mathbf{1.80\times}}$ |
| **8K** | 93.3 µs | 179.1 µs | 84.4 µs | **80.5 µs** | $\color{#16a34a}{\mathbf{1.10\times}}$ | $\color{#16a34a}{\mathbf{1.16\times}}$ | $\color{#16a34a}{\mathbf{2.12\times}}$ | $\color{#16a34a}{\mathbf{2.22\times}}$ |
| **16K** | 184.4 µs | 340.3 µs | 177.2 µs | **172.9 µs** | $\color{grey}{1.04\times}$ | $\color{grey}{1.07\times}$ | $\color{#16a34a}{\mathbf{1.92\times}}$ | $\color{#16a34a}{\mathbf{1.97\times}}$ |
| **32K** | 405.3 µs | 668.4 µs | **366.1 µs** | 366.9 µs | $\color{#16a34a}{\mathbf{1.11\times}}$ | $\color{#16a34a}{\mathbf{1.10\times}}$ | $\color{#16a34a}{\mathbf{1.83\times}}$ | $\color{#16a34a}{\mathbf{1.82\times}}$ |
| **64K** | 840.3 µs | 1.28 ms | 711.8 µs | **667.7 µs** | $\color{#16a34a}{\mathbf{1.18\times}}$ | $\color{#16a34a}{\mathbf{1.26\times}}$ | $\color{#16a34a}{\mathbf{1.80\times}}$ | $\color{#16a34a}{\mathbf{1.92\times}}$ |
| **128K** | 1.60 ms | 2.37 ms | **1.31 ms** | 1.32 ms | $\color{#16a34a}{\mathbf{1.22\times}}$ | $\color{#16a34a}{\mathbf{1.21\times}}$ | $\color{#16a34a}{\mathbf{1.82\times}}$ | $\color{#16a34a}{\mathbf{1.80\times}}$ |
| **256K** | 3.05 ms | 4.51 ms | 2.44 ms | **2.40 ms** | $\color{#16a34a}{\mathbf{1.25\times}}$ | $\color{#16a34a}{\mathbf{1.27\times}}$ | $\color{#16a34a}{\mathbf{1.85\times}}$ | $\color{#16a34a}{\mathbf{1.88\times}}$ |
| geomean |  |  |  |  | $\color{#16a34a}{\mathbf{1.21\times}}$ | $\color{#16a34a}{\mathbf{1.26\times}}$ | $\color{#16a34a}{\mathbf{1.79\times}}$ | $\color{#16a34a}{\mathbf{1.85\times}}$ |

<sub>Rendered from [`benchmarks/raw/verify/sweep/20261011-082714.json`](../../benchmarks/raw/verify/sweep/20261011-082714.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±7% noise band, red = slower, gray = within noise.</sub>

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
