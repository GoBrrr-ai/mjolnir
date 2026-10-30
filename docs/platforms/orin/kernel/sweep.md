# Headline L-sweep

> the `sm_87` mma kernel vs the FlashInfer FA2-TC reference — events-timed at every L, same run, stock image · `mjolnir verify sweep` · run `20261030-162038`

## The question

The headline: how fast is one decode step with Mjolnir against the platform's stock reference — at every context length from 128 to 256K tokens? Short contexts are latency-bound (launch, plan, merge), long ones bandwidth-bound; a decode kernel has to win both.

## How it is measured

- one sequence, 24 query / 4 KV heads, head dim 256, 128-token pages in random order — the hd256 attention shape of Qwen3.8-27B;
- **CUDA-event wall clock**: the kernel number replays the compiled launch with prebuilt arguments; the FlashInfer reference runs in its own container on the same stock image, warmed up the same way, in the same run. No ncu replay on this stage, so no cold-L2/base-clock control here — the ncu-profiled stages (hndsweep, kvfmt, ab, fp6prof) carry it;
- legs: **Mjolnir MMA** (the `sm_87` `mma.m16n8k16` kernel, prebuilt launches), **FlashInfer FA2-TC** tensor-core decode as vLLM calls it (fp8 KV, NHD pages);
- bandwidth = each leg's *own* KV bytes ÷ time (fp8 = 8 bit);

## Result

<img src="../../../../assets/benchmarks/orin/Qwen3.8-27B/kernel-speedup.png" alt="Mjolnir speedup over FlashInfer" width="100%">

<img src="../../../../assets/benchmarks/orin/Qwen3.8-27B/kernel-latency.png" alt="Decode latency per step" width="100%">

<img src="../../../../assets/benchmarks/orin/Qwen3.8-27B/kernel-bandwidth.png" alt="Effective KV bandwidth vs the platform's memory peak" width="100%">

- **Mjolnir MMA (fp8 KV, HND)** vs FlashInfer FA2-TC (fp8 KV, NHD): geomean **1.14×**, from 1.00× (L=8K) to 1.54× (L=128); faster beyond the ±3% noise band at 6 of 8 context lengths.

| L | Mjolnir fp8 | FlashInfer | fp8 vs FI |
|---|--:|--:|--:|
| **128** | **69.2&nbsp;µs** | 106.3&nbsp;µs | $\color{#16a34a}{\mathbf{1.54\times}}$ |
| **4K** | **87.4&nbsp;µs** | 131.3&nbsp;µs | $\color{#16a34a}{\mathbf{1.50\times}}$ |
| **8K** | 136.3&nbsp;µs | **135.7&nbsp;µs** | $\color{grey}{1.00\times}$ |
| **16K** | **229.2&nbsp;µs** | 229.7&nbsp;µs | $\color{grey}{1.00\times}$ |
| **32K** | **417.6&nbsp;µs** | 434.8&nbsp;µs | $\color{#16a34a}{\mathbf{1.04\times}}$ |
| **64K** | **791.3&nbsp;µs** | 838.9&nbsp;µs | $\color{#16a34a}{\mathbf{1.06\times}}$ |
| **128K** | **1.54&nbsp;ms** | 1.64&nbsp;ms | $\color{#16a34a}{\mathbf{1.07\times}}$ |
| **256K** | **3.04&nbsp;ms** | 3.31&nbsp;ms | $\color{#16a34a}{\mathbf{1.09\times}}$ |
| geomean |  |  | $\color{#16a34a}{\mathbf{1.14\times}}$ |

<details>
<summary>How the % and × here are calculated</summary>

- **Time per decode step** = CUDA-event median over the timed launches. The kernel number replays the compiled launch with prebuilt arguments; the FlashInfer reference runs in its own container on the same stock image, warmed up the same way, in the same run.
- **Speedup ×** = reference time ÷ Mjolnir time; below 1 the reference is faster.
- **Geomean over L** = exp(mean of ln(speedup at each L)): every context length weighs the same.
- This stage is CUDA-event timed on this platform — no ncu replay, so no cold-L2 or base-clock control here (the ncu-profiled stages carry those). ±3% (`NOISE_PCT`) is the same-run spread band.

</details><br />

<sub>Rendered from [`benchmarks/raw/verify/sweep/20261030-162038.json`](../../../../benchmarks/raw/verify/sweep/20261030-162038.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

## How to read it

At L=128 the cache fits in one tile and time is mostly fixed cost: plan, launch, merge. From ~16K the cache no longer fits the L2 and every kernel converges toward the DRAM roofline — the remaining gap is how much of the peak each kernel sustains (bandwidth chart) and how many bytes it has to move (KV format).

## Reproduce

```bash
mjolnir verify sweep
```

## Ideas to try

- batch > 1 and varlen batches: `mjolnir verify sweep --batch 4 --q-len 4` (the MTP=3 verify shape at 4 concurrent requests), `--batches 8 --varlen` on the harness;
- page size 16 (what vLLM's FlashInfer backend uses on Jetson Orin): `mjolnir verify sweep --page-size 16`.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../../../.github/CONTRIBUTING.md).

---

[← Correctness matrix](correctness.md) · [case study](../README.md#-kernel) · [Head-major pages →](hndsweep.md)
