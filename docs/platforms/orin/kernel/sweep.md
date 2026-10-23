# Headline L-sweep

> GEMV kernel vs the full serving path vs the FlashInfer FA2-TC reference — events-timed at every L, same run, stock image · `mjolnir verify sweep` · run `20261023-095858`

## The question

The headline: how fast is one decode step with Mjolnir, FlashInfer and FlashAttention 4 hd256 — at every context length from 128 to 256K tokens? Short contexts are latency-bound (launch, plan, merge), long ones bandwidth-bound; a decode kernel has to win both.

## How it is measured

- one sequence, 24 query / 4 KV heads, head dim 256, 128-token pages in random order — the hd256 attention shape of Qwen3.8-27B;

- **CUDA-event wall clock** (no ncu exists on L4T): the kernel number replays the compiled launch with prebuilt arguments; the serving-path number times the full `runner.run()` call (argument packing + the split-KV LSE combine, python included); the FlashInfer reference runs in its own container on the same stock image, warmed up the same way, in the same run;
- legs: **Mjolnir GEMV fp8** (prebuilt launches), **Mjolnir serving path** (the full per-call cost), **FlashInfer** tensor-core decode as vLLM calls it (fp8 KV, 16-token pages);
- bandwidth = each leg's *own* KV bytes ÷ time (fp8 = 8 bit);


## Result

<img src="../../../../assets/benchmarks/orin/Qwen3.8-27B/kernel-speedup.png" alt="Mjolnir speedup over FlashInfer" width="100%">

<img src="../../../../assets/benchmarks/orin/Qwen3.8-27B/kernel-latency.png" alt="Decode latency per step" width="100%">

<img src="../../../../assets/benchmarks/orin/Qwen3.8-27B/kernel-bandwidth.png" alt="Effective KV bandwidth" width="100%">

- **Mjolnir GEMV (fp8 KV, HND)** vs FlashInfer FA2-TC (fp8 KV, NHD): geomean **0.24×**, from 0.13× (L=128K) to 0.64× (L=4K); faster beyond the ±3% noise band at 0 of 8 context lengths.

| L | Mjolnir e2m3 | Mjolnir serving path | FlashInfer | e2m3 vs FI | serving path vs FI |
|---|--:|--:|--:|--:|--:|
| **128** | 220.9&nbsp;µs | 805.9&nbsp;µs | **120.0&nbsp;µs** | $\color{#dc2626}{0.54\times}$ | $\color{#dc2626}{0.15\times}$ |
| **4K** | 442.1&nbsp;µs | 907.2&nbsp;µs | **281.7&nbsp;µs** | $\color{#dc2626}{0.64\times}$ | $\color{#dc2626}{0.31\times}$ |
| **8K** | 881.3&nbsp;µs | 1.23&nbsp;ms | **344.2&nbsp;µs** | $\color{#dc2626}{0.39\times}$ | $\color{#dc2626}{0.28\times}$ |
| **16K** | 1.65&nbsp;ms | 2.84&nbsp;ms | **364.9&nbsp;µs** | $\color{#dc2626}{0.22\times}$ | $\color{#dc2626}{0.13\times}$ |
| **32K** | 3.15&nbsp;ms | 4.37&nbsp;ms | **431.6&nbsp;µs** | $\color{#dc2626}{0.14\times}$ | $\color{#dc2626}{0.10\times}$ |
| **64K** | 6.16&nbsp;ms | 6.86&nbsp;ms | **834.9&nbsp;µs** | $\color{#dc2626}{0.14\times}$ | $\color{#dc2626}{0.12\times}$ |
| **128K** | 12.17&nbsp;ms | 12.91&nbsp;ms | **1.64&nbsp;ms** | $\color{#dc2626}{0.13\times}$ | $\color{#dc2626}{0.13\times}$ |
| **256K** | 24.19&nbsp;ms | 24.93&nbsp;ms | **3.34&nbsp;ms** | $\color{#dc2626}{0.14\times}$ | $\color{#dc2626}{0.13\times}$ |
| geomean |  |  |  | $\color{#dc2626}{0.24\times}$ | $\color{#dc2626}{0.16\times}$ |

<details>
<summary>How the % and × here are calculated</summary>

- **Time per decode step** = Nsight Compute kernel time (`gpu__time_duration`) of the decode kernel **plus** its split-KV merge kernel, at base clocks with a cold L2, median over the iterations. The reference (FlashInfer, FlashAttention 4 hd256) is measured the same way, at the same L, in the same run.
- **Speedup ×** = reference time ÷ Mjolnir time. `1.24×` means Mjolnir needs 1 ÷ 1.24 = 81% of the reference's time per step (19% less time); below 1 the reference is faster.
- **Geomean over L** = exp(mean of ln(speedup at each L)): every context length weighs the same, and one large ratio cannot dominate the way it would in an arithmetic mean.
- **Colors / "beyond the noise band"**: green = speedup ≥ 1.03, grey = within ±3%, red ≤ 0.97. ±3% (`NOISE_PCT`) is how far a same-run ratio moves between repeated runs on a dedicated Thor.

</details><br />

<sub>Rendered from [`benchmarks/raw/verify/sweep/20261023-095858.json`](../../../../benchmarks/raw/verify/sweep/20261023-095858.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

## How to read it

At L=128 the cache fits in one tile and time is mostly fixed cost: plan, launch, merge. From ~16K the cache no longer fits the 32 MiB L2 and every kernel converges toward the DRAM roofline — the remaining gap is how much of the peak each kernel sustains (bandwidth chart) and how many bytes it has to move (KV format). FA4 reads the same fp8 bytes as the other fp8 legs; the FA2 leg (available on demand via `--legs fa2`) pays twice the bytes — it has no fp8 KV path.

### FA4 hd256 as served: why the number is what it is

The FA4 leg times the kernel as the `NVFP4_FA4hd256` serving config runs it on the Mjolnir image (the only image with the FA4 hd256 + fp8 patch group; stock vLLM downgrades hd256 + quantized KV to FA2). That path is fixed, not tunable, on vLLM 0.32.0:

- **1-CTA decode carve-out** — the patched kernel's measured-best decode form: 2-CTA is correct but 1.56× slower at decode shapes (clean micro-bench, 2026-09-24), so decode-shaped calls (M ≤ 8) take the single-CTA cluster; prefill keeps 2-CTA.
- **No SplitKV at any layer** — the vendored hd256 kernel asserts `not is_split_kv` ("SM100 forward with head_dim=256 does not support SplitKV"), the dispatch clamps `num_splits = 1` for the dedicated hd256 kernel, and the backend pins `num_splits = 1`; the in-tree SplitKV machinery measured a **no-win on both cluster forms** (2-CTA slower in 40/40 cells; 1-CTA +18.6%/+6.4% at L=2048/4096 and at best noise-level at L=8192 — the per-CTA TMEM/MMA fixed cost replicates with the split count). Upstream #2916's hd256 SplitKV is not in this vendored FA4.
- **fp8 Q + KV with fp8 P·V** — the SM100/110 FA4 pipeline requires Q in fp8 alongside fp8 KV (verified: the kernel's output deviation vs an exact fp32 reference equals the fp8-P quantization floor, ~2.6% rel L2 at L=4096, and max abs is ~12× inside the patch's own fp8 tolerance).

With those pinned, the comparison is fair: FlashInfer remains the strongest stock baseline and Mjolnir the headline.

## Reproduce

```bash
mjolnir verify sweep
mjolnir verify sweep --legs fi,fa2,fi-hnd,mjolnir-fp8,mjolnir --ls "8192 65536"   # custom (not a headline run); fa2 = the on-demand bf16 baseline
```

## Ideas to try

- batch > 1 and varlen batches: `mjolnir verify sweep --batch 4 --q-len 4` (the MTP=3 verify shape at 4 concurrent requests), `--batches 8 --varlen` on the harness;
- page size 16 (what vLLM's FlashInfer backend uses on Jetson Orin): `mjolnir verify sweep --page-size 16`.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../../../.github/CONTRIBUTING.md).

---

[← Correctness matrix](correctness.md) · [case study](../README.md#kernel) · [Head-major pages →](hndsweep.md)
