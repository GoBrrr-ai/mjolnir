<p align="center">
  <img src="../../../assets/hero-thor.jpg" alt="Mjolnir on Jetson AGX Thor" width="100%">
</p>

# Jetson AGX Thor

> **supported** platform · `sm_110` · host requirements: JetPack 7.x · [all platforms](../README.md)



## Serving on Jetson AGX Thor

One row per model in `configs/` — the config the platform's overlays serve
by default (the model page's quickstart rule) and whether the
(platform, model) pair has end-to-end data. Per-config availability lives
on each [model page](../../../configs/).

| model | default config | measured |
|---|---|---|
| [`Qwen/Qwen3.8-27B`](../../../configs/Qwen/Qwen3.8-27B/README.md) | [`NVFP4_MJOLNIR_E2M3_LMH8_VT64K`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/README.md) | ✅ |
| [`nvidia/Qwen3.6-35B-A3B`](../../../configs/nvidia/Qwen3.6-35B-A3B/README.md) | [`NVFP4_MJOLNIR_E2M3_LMH8_VT64K`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/README.md) | ✅ |

## Kernel

Mjolnir on Jetson AGX Thor, measured chapter by chapter and pinned to the
headline model (Qwen/Qwen3.8-27B) — every chart and table rendered
from raw results in [`benchmarks/raw/verify/`](../../../benchmarks/raw/verify/)
by `mjolnir plot --readme`; every chapter ends with the command that
reproduces it.

**Where it stands:** 1.21× FlashInfer (geomean over L = 128 4K 8K 16K 32K 64K 128K 256K), 2.17× FlashAttention 4 hd256 (fp8 Q+KV, as the patched image serves it: 1-CTA decode, no SplitKV), 97 / 97 correctness checks — run `20261022-090701`.

<details>
<summary>How the % and × here are calculated</summary>

- **Time per decode step** = Nsight Compute kernel time (`gpu__time_duration`) of the decode kernel **plus** its split-KV merge kernel, at base clocks with a cold L2, median over the iterations. The reference (FlashInfer, FlashAttention 4 hd256) is measured the same way, at the same L, in the same run.
- **Speedup ×** = reference time ÷ Mjolnir time. `1.24×` means Mjolnir needs 1 ÷ 1.24 = 81% of the reference's time per step (19% less time); below 1 the reference is faster.
- **Geomean over L** = exp(mean of ln(speedup at each L)): every context length weighs the same, and one large ratio cannot dominate the way it would in an arithmetic mean.
- **Colors / "beyond the noise band"**: green = speedup ≥ 1.03, grey = within ±3%, red ≤ 0.97. ±3% (`NOISE_PCT`) is how far a same-run ratio moves between repeated runs on a dedicated Thor.

</details><br />

<img src="../../../assets/benchmarks/thor/Qwen3.8-27B/kernel-speedup.png" alt="Mjolnir decode speedup over FlashInfer" width="100%">

| # | chapter | question | status |
|---|---|---|---|
| 1 | [Correctness matrix](kernel/correctness.md) | kernel vs the bit-faithful emulator: every layout, format, page size, q_len, Q dtype | ✅ run `20261018-143212` |
| 2 | [Headline L-sweep](kernel/sweep.md) | Mjolnir vs FlashInfer vs FlashAttention 4 hd256 (fp8 Q+KV, on the Mjolnir image), ncu kernel time at every L | ✅ run `20261022-090701` |
| 3 | [Head-major pages](kernel/hndsweep.md) | FlashInfer on NHD and HND vs Mjolnir on HND — is HND a kernel win or a layout win? | ✅ run `20261012-203356` |
| 4 | [KV layouts and formats](kernel/kvfmt.md) | fp8 / fp6 / fp4 / nvfp4 KV on NHD / interleaved / HND pages (ncu + wall clock) | ✅ run `20261012-203801` |
| 5 | [KV format accuracy](kernel/quant.md) | relative L2 error of each KV format vs unquantized KV, at every L | ✅ run `20261012-205732` |
| 6 | [KV cache capacity](kernel/kv-cache-size.md) | tokens of KV each config fits (vLLM's startup sizing), and the concurrency that buys per context | ✅ run `20261013-124353` |
| 7 | [Speculative decoding](kernel/spec.md) | MTP acceptance per draft position and decode t/s for 1-N draft tokens, per config — the best draft count | ✅ run `20261014-182829` |
| 8 | [Full profiles per KV format](kernel/fp6prof.md) | full ncu profiles: L2 sectors vs requests vs time, warp stalls, per KV format | ✅ run `20261017-212447` |
| 9 | [KV ring depth](kernel/ring.md) | TMA→MMA ring depth (--kv_stages) sweep at every L | ✅ run `20261012-210506` |
| 10 | [L2 prefetch](kernel/prefetch.md) | TMA L2 prefetch distance (--l2_prefetch) sweep | ✅ run `20261012-211441` |
| 11 | [Programmatic dependent launch](kernel/pdl.md) | PDL on/off wall clock (decode overlaps the previous kernel; merge launches early) | ✅ run `20261012-212058` |
| 12 | [Flag A/B](kernel/ab.md) | any harness flag sets, ncu + wall clock (default: evict-first x PDL) | ✅ run `20261012-212748` |
| 13 | [Accuracy](kernel/accuracy.md) | Does a serving config keep the model's answers? Tool calling (standard + Hard Mode), GSM8K / MMLU / IFEval on demand — `mjolnir bench accuracy` | ✅ run `20261017-210613` |




### Mjolnir Kernel

All kernel numbers are **ncu kernel time** (decode + split-KV merge), cold L2, base clocks, one sequence with 24 query / 4 KV heads, head dim 256, 128-token pages — the attention shape of Qwen3.8-27B — at **every context length from 128 to 256K tokens**.
Measured on a dedicated Thor: a same-run ratio repeats within ±3% between runs, see [`docs/methodology/benchmarking.md`](../../../docs/methodology/benchmarking.md).

| leg | config | what runs | KV cache |
|---|---|---|---|
| **Mjolnir · e2m3 KV** | `NVFP4_MJOLNIR_E2M3_LMH8_VT64K` | Mjolnir kernel + fp8 lm_head + 64K MTP drafter | e2m3, HND pages |
| **Mjolnir · fp8 KV** | `NVFP4_MJOLNIR` | Mjolnir kernel | fp8 e4m3, HND pages |
| **FlashInfer · fp8 KV** | `NVFP4` | The fastest official vllm 0.32.0 FlashInfer for `sm_110` | fp8 e4m3, NHD pages |
| **FlashAttention&nbsp;4 · hd256 fp8** | `NVFP4_FA4hd256` | FA4 hd256 as our image serves it — the patched FA4: 1-CTA decode carve-out, no SplitKV (structural in vLLM 0.32.0's vendored kernel) | fp8 e4m3 Q+KV, NHD pages |
| **FlashAttention&nbsp;2 · bf16 KV** | `NVFP4_FA2` | FA2 — on demand (`--legs fa2`); stock vLLM 0.32.0's `sm_110` default | bf16 $\color{grey}\scriptstyle\text{(FA2 has no fp8 KV)}$ |


<img src="../../../assets/benchmarks/thor/Qwen3.8-27B/kernel-latency.png" alt="Decode latency per step vs context length" width="100%">

| L | FlashInfer | FA4 hd256 | Mjolnir fp8 | Mjolnir e2m3 | fp8 vs FI | e2m3 vs FI | fp8 vs FA4 | e2m3 vs FA4 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **128** | 21.9&nbsp;µs | 20.9&nbsp;µs | 13.6&nbsp;µs | **13.5&nbsp;µs** | $\color{#16a34a}{\mathbf{1.61\times}}$ | $\color{#16a34a}{\mathbf{1.63\times}}$ | $\color{#16a34a}{\mathbf{1.53\times}}$ | $\color{#16a34a}{\mathbf{1.55\times}}$ |
| **4K** | 60.5&nbsp;µs | 105.5&nbsp;µs | 55.0&nbsp;µs | **49.9&nbsp;µs** | $\color{#16a34a}{\mathbf{1.10\times}}$ | $\color{#16a34a}{\mathbf{1.21\times}}$ | $\color{#16a34a}{\mathbf{1.92\times}}$ | $\color{#16a34a}{\mathbf{2.11\times}}$ |
| **8K** | 98.2&nbsp;µs | 191.0&nbsp;µs | **91.0&nbsp;µs** | 91.3&nbsp;µs | $\color{#16a34a}{\mathbf{1.08\times}}$ | $\color{#16a34a}{\mathbf{1.07\times}}$ | $\color{#16a34a}{\mathbf{2.10\times}}$ | $\color{#16a34a}{\mathbf{2.09\times}}$ |
| **16K** | 182.6&nbsp;µs | 370.1&nbsp;µs | 181.4&nbsp;µs | **177.0&nbsp;µs** | $\color{grey}{1.01\times}$ | $\color{#16a34a}{\mathbf{1.03\times}}$ | $\color{#16a34a}{\mathbf{2.04\times}}$ | $\color{#16a34a}{\mathbf{2.09\times}}$ |
| **32K** | 358.0&nbsp;µs | 729.7&nbsp;µs | 341.5&nbsp;µs | **332.2&nbsp;µs** | $\color{#16a34a}{\mathbf{1.05\times}}$ | $\color{#16a34a}{\mathbf{1.08\times}}$ | $\color{#16a34a}{\mathbf{2.14\times}}$ | $\color{#16a34a}{\mathbf{2.20\times}}$ |
| **64K** | 748.9&nbsp;µs | 1.42&nbsp;ms | 660.0&nbsp;µs | **588.9&nbsp;µs** | $\color{#16a34a}{\mathbf{1.13\times}}$ | $\color{#16a34a}{\mathbf{1.27\times}}$ | $\color{#16a34a}{\mathbf{2.15\times}}$ | $\color{#16a34a}{\mathbf{2.41\times}}$ |
| **128K** | 1.34&nbsp;ms | 2.70&nbsp;ms | 1.26&nbsp;ms | **1.09&nbsp;ms** | $\color{#16a34a}{\mathbf{1.06\times}}$ | $\color{#16a34a}{\mathbf{1.23\times}}$ | $\color{#16a34a}{\mathbf{2.15\times}}$ | $\color{#16a34a}{\mathbf{2.48\times}}$ |
| **256K** | 2.51&nbsp;ms | 5.31&nbsp;ms | 2.40&nbsp;ms | **2.06&nbsp;ms** | $\color{#16a34a}{\mathbf{1.05\times}}$ | $\color{#16a34a}{\mathbf{1.22\times}}$ | $\color{#16a34a}{\mathbf{2.22\times}}$ | $\color{#16a34a}{\mathbf{2.58\times}}$ |
| geomean |  |  |  |  | $\color{#16a34a}{\mathbf{1.12\times}}$ | $\color{#16a34a}{\mathbf{1.21\times}}$ | $\color{#16a34a}{\mathbf{2.02\times}}$ | $\color{#16a34a}{\mathbf{2.17\times}}$ |

> <sub>Bold = fastest at that `L`. Ratio columns: baseline time ÷ this kernel's time (green = faster beyond noise).</sub>

<details>
<summary>How the % and × here are calculated</summary>

- **Time per decode step** = Nsight Compute kernel time (`gpu__time_duration`) of the decode kernel **plus** its split-KV merge kernel, at base clocks with a cold L2, median over the iterations. The reference (FlashInfer, FlashAttention 4 hd256) is measured the same way, at the same L, in the same run.
- **Speedup ×** = reference time ÷ Mjolnir time. `1.24×` means Mjolnir needs 1 ÷ 1.24 = 81% of the reference's time per step (19% less time); below 1 the reference is faster.
- **Geomean over L** = exp(mean of ln(speedup at each L)): every context length weighs the same, and one large ratio cannot dominate the way it would in an arithmetic mean.
- **Colors / "beyond the noise band"**: green = speedup ≥ 1.03, grey = within ±3%, red ≤ 0.97. ±3% (`NOISE_PCT`) is how far a same-run ratio moves between repeated runs on a dedicated Thor.

</details><br />

<img src="../../../assets/benchmarks/thor/Qwen3.8-27B/kernel-bandwidth.png" alt="Effective KV bandwidth vs the LPDDR5X peak" width="100%">


## End to end

<img src="../../../assets/benchmarks/thor/Qwen3.8-27B/opening-thor-infographic.jpg" alt="Mjolnir vs the official vLLM image — end-to-end serving" width="100%">

The official vLLM image vs Mjolnir on Jetson AGX Thor — the headline model's
own history (`latest_pair`), decode tokens/s per request at 1 request.

| official | Mjolnir | official t/s | Mjolnir t/s | change | at |
|---|---|--:|--:|--:|---|
| `NVFP4` | `NVFP4_MJOLNIR_E2M3_LMH8_VT64K` | 24.8 | 32.2 | +29.9% | 32K |

<img src="../../../assets/benchmarks/thor/Qwen3.8-27B/e2e-decode.png" alt="End-to-end decode throughput: official vLLM vs Mjolnir" width="100%">

<img src="../../../assets/benchmarks/thor/Qwen3.8-27B/e2e-prefill.png" alt="End-to-end prompt processing: official vLLM vs Mjolnir" width="100%">



Each cell: official → **Mjolnir** and the relative change of the value — **throughput change** for decode and prefill, **latency change** for TTFT and ITL.

| context | requests | decode (t/s, ↑) $\color{grey}\scriptstyle\text{throughput change}$ | prefill (t/s, ↑) $\color{grey}\scriptstyle\text{throughput change}$ | TTFT (ms, ↓) $\color{grey}\scriptstyle\text{latency change}$ | ITL (ms, ↓) $\color{grey}\scriptstyle\text{latency change}$ |
|--:|--:|--:|--:|--:|--:|
| no ctx | 1 | 23.9 → **39.2** $\color{#16a34a}{\text{+63.9\\%}}$ | — | 820 → **604** $\color{#16a34a}{\text{−26.3\\%}}$ | 42.0 → **26.5** $\color{#16a34a}{\text{−36.8\\%}}$ |
| 4K | 1 | 23.2 → **36.6** $\color{#16a34a}{\text{+57.8\\%}}$ | 2,811 → **10,114** $\color{#16a34a}{\text{+259.8\\%}}$ | 2,319 → **1,686** $\color{#16a34a}{\text{−27.3\\%}}$ | 43.3 → **27.7** $\color{#16a34a}{\text{−35.9\\%}}$ |
| 8K | 1 | 25.3 → **33.4** $\color{#16a34a}{\text{+32.2\\%}}$ | 2,560 → **5,240** $\color{#16a34a}{\text{+104.7\\%}}$ | 4,134 → **3,034** $\color{#16a34a}{\text{−26.6\\%}}$ | 39.7 → **30.6** $\color{#16a34a}{\text{−23.1\\%}}$ |
| 16K | 1 | 24.7 → **34.8** $\color{#16a34a}{\text{+41.0\\%}}$ | 2,196 → **4,120** $\color{#16a34a}{\text{+87.6\\%}}$ | 8,526 → **5,553** $\color{#16a34a}{\text{−34.9\\%}}$ | 40.7 → **29.1** $\color{#16a34a}{\text{−28.6\\%}}$ |
| 32K | 1 | 24.8 → **32.2** $\color{#16a34a}{\text{+29.9\\%}}$ | 1,756 → **3,433** $\color{#16a34a}{\text{+95.5\\%}}$ | 19,964 → **11,222** $\color{#16a34a}{\text{−43.8\\%}}$ | 40.6 → **31.3** $\color{#16a34a}{\text{−23.1\\%}}$ |
| no ctx | 2 | 44.5 → **67.1** $\color{#16a34a}{\text{+50.6\\%}}$ | — | 1,443 → **1,065** $\color{#16a34a}{\text{−26.2\\%}}$ | 42.5 → **27.9** $\color{#16a34a}{\text{−34.4\\%}}$ |
| 4K | 2 | 43.2 → **65.9** $\color{#16a34a}{\text{+52.4\\%}}$ | 2,707 → **3,497** $\color{#16a34a}{\text{+29.2\\%}}$ | 4,478 → **3,453** $\color{#16a34a}{\text{−22.9\\%}}$ | 44.2 → **28.1** $\color{#16a34a}{\text{−36.4\\%}}$ |
| 8K | 2 | 44.6 → **60.7** $\color{#16a34a}{\text{+35.9\\%}}$ | 2,499 → **3,422** $\color{#16a34a}{\text{+37.0\\%}}$ | 8,135 → **5,803** $\color{#16a34a}{\text{−28.7\\%}}$ | 42.6 → **30.3** $\color{#16a34a}{\text{−29.0\\%}}$ |
| 16K | 2 | 43.1 → **56.2** $\color{#16a34a}{\text{+30.5\\%}}$ | 2,180 → **3,345** $\color{#16a34a}{\text{+53.4\\%}}$ | 16,847 → **10,896** $\color{#16a34a}{\text{−35.3\\%}}$ | 43.7 → **32.2** $\color{#16a34a}{\text{−26.3\\%}}$ |
| 32K | 2 | 34.6 → **48.1** $\color{#16a34a}{\text{+39.0\\%}}$ | 1,743 → **3,108** $\color{#16a34a}{\text{+78.3\\%}}$ | 39,253 → **21,777** $\color{#16a34a}{\text{−44.5\\%}}$ | 49.8 → **35.5** $\color{#16a34a}{\text{−28.7\\%}}$ |
| no ctx | 4 | 83.3 → **100.2** $\color{#16a34a}{\text{+20.3\\%}}$ | — | 2,789 → **2,302** $\color{#16a34a}{\text{−17.5\\%}}$ | 43.3 → **32.3** $\color{#16a34a}{\text{−25.5\\%}}$ |
| 4K | 4 | 82.0 → **103.6** $\color{#16a34a}{\text{+26.4\\%}}$ | 2,710 → **3,470** $\color{#16a34a}{\text{+28.0\\%}}$ | 9,035 → **6,895** $\color{#16a34a}{\text{−23.7\\%}}$ | 45.8 → **32.8** $\color{#16a34a}{\text{−28.3\\%}}$ |
| 8K | 4 | 71.9 → **67.1** $\color{#dc2626}{\text{−6.7\\%}}$ | 2,461 → **3,372** $\color{#16a34a}{\text{+37.0\\%}}$ | 16,323 → **11,213** $\color{#16a34a}{\text{−31.3\\%}}$ | 46.7 → **39.9** $\color{#16a34a}{\text{−14.6\\%}}$ |
| 16K | 4 | 42.9 → **49.7** $\color{#16a34a}{\text{+15.8\\%}}$ | 2,158 → **3,246** $\color{#16a34a}{\text{+50.4\\%}}$ | 32,532 → **20,887** $\color{#16a34a}{\text{−35.8\\%}}$ | 58.5 → **48.9** $\color{#16a34a}{\text{−16.4\\%}}$ |
| 32K | 4 | 11.4 → **18.2** $\color{#16a34a}{\text{+58.9\\%}}$ | 1,726 → **3,030** $\color{#16a34a}{\text{+75.5\\%}}$ | 66,221 → **36,571** $\color{#16a34a}{\text{−44.8\\%}}$ | 161.7 → **109.7** $\color{#16a34a}{\text{−32.1\\%}}$ |

> <sub>official: `vllm/vllm-openai:v0.32.0-ubuntu2404 · NVFP4 (FlashInfer)` · Mjolnir: `narandill/mjolnir-vllm:v0.32.0 · NVFP4_MJOLNIR_E2M3_LMH8_VT64K (Mjolnir fp6 KV + fp8 lm_head + 64K draft vocab)` · TTFT = time to first token; ITL = inter-token latency (1000 / per-request decode rate) · from [`benchmarks/history.jsonl`](../../../benchmarks/history.jsonl).</sub>


<details>
<summary>How the % and × here are calculated</summary>

- **Throughput** (decode, prefill — tokens/s, higher is better): change = (Mjolnir − official) ÷ official × 100. `+57%` = 1.57× the tokens per second. On the infographic a change of +100% or more is written as the multiple Mjolnir ÷ official (`+200%` = `3×`).
- **Latency** (TTFT, ITL — milliseconds, lower is better): change = (Mjolnir − official) ÷ official × 100, so a negative value is a **latency reduction**: `−27%` = 27% fewer milliseconds. It is *not* the reciprocal speedup official ÷ Mjolnir − 1, which reads larger for the same pair.
- **Values**: the mean over the measured runs per cell (6 measured after 2 warm-ups, llama-benchy): decode = generated tokens ÷ generation time per request (summed over requests at higher concurrency), prefill = prompt tokens ÷ prompt-processing time, TTFT = time to the first content token, ITL = 1000 ÷ the per-request decode rate.
- **Colors**: green = better by at least 3%, grey = within ±3% (`E2E_NOISE_PCT`), red = worse by at least that. With speculative decoding a single decode cell moves more between runs (acceptance depends on the prompt), so repeat (`--repeat`) before reading a few-% difference.

</details><br />

## How the kernel works

A decode step is one query token per sequence against the whole KV cache: at 64K tokens that is 128 MiB of K and V per layer, read once. The design goal is to keep the LPDDR5X busy and everything else out of the way.

| | design choice | why on Thor |
|---|---|---|
| ⚙️ | **Warp-specialized `tcgen05` pipeline** — a TMA warp streams K/V pages through an SMEM ring, an MMA warp runs `S = QKᵀ` and `O += PV` on the tensor cores into TMEM, softmax warpgroups read S from TMEM and write P | the tensor cores and TMEM keep scores off the register file; the TMA engine walks the page table so the SMs never issue loads |
| 📐 | **Occupancy-pinned split-KV** — every step fills exactly 2 CTAs × 20 SMs; long sequences split into chunks, a small reduction kernel merges them | one full wave at every L; a fixed padded grid replays under full CUDA graphs |
| 🧊 | **fp8 everything** — Q quantized per (sequence, KV head) on the fly, fp8 KV, P stored as e4m3 | one MMA kind, no conversion warps, half the KV bytes of bf16 |
| 🗂️ | **Head-major (HND) pages** — each KV head's page is one contiguous block | longer DRAM bursts per TMA box: [hndsweep](kernel/hndsweep.md) |
| 🪶 | **6-bit KV (e2m3) read directly by the MMA** (`kind::f8f6f4`, TMA unpacks to byte containers) | 25% fewer bytes at near-fp8 accuracy: [kvfmt](kernel/kvfmt.md), [quant](kernel/quant.md), [capacity](kernel/kv-cache-size.md) |
| 🚿 | **L2 evict-first on K/V loads**, a 2-stage ring for fp6 | KV is streamed once; keep L2 for the rest of the model: [ring](kernel/ring.md), [ab](kernel/ab.md) |

The full walkthrough — one chapter per experiment, each with its chart, data table and the command that reproduces it — is the chapter index under [Kernel](#kernel) above.


## What's in the Mjolnir image

[`narandill/mjolnir-vllm:v0.32.0`](https://hub.docker.com/r/narandill/mjolnir-vllm) = the official `vllm/vllm-openai:v0.32.0` image + the `mjolnir-kernel` package + a hand-adapted [patch stack for Thor](../../../docker/vllm-thor/PATCHES.md). Tags follow the upstream vLLM version.

| patch | what it does |
|---|---|
| **`thor-mjolnir-attention-backend`** | adds `MJOLNIR` to vLLM's attention-backend enum → `--attention-backend MJOLNIR` |
| [#50885](https://github.com/vllm-project/vllm/pull/50885) | FULL cudagraph mode for speculative decode on the FlashInfer-native backend |
| [#49652](https://github.com/vllm-project/vllm/pull/49652) | draft-decode cudagraphs under dynamic speculative decoding |
| [#54165](https://github.com/vllm-project/vllm/pull/54165) | Mamba/GDN `copy_kv_cache` restore under speculative decode |
| [#55519](https://github.com/vllm-project/vllm/pull/55519) | no false-positive KV-reuse warning on the GDN hybrid |
| Thor `sm_110` patches | GDN prefill enablement, FA4 hd256 fp8-KV + 1-CTA decode, fused draft-decode plan advance, DSpark non-causal draft cudagraph, draft-cudagraph shape gate, the `MJOLNIR` backend name + FA4 fallback default, the `fp6_e2m3` KV cache dtype |



Explore every number in [the interactive charts](https://thor-arena.github.io/mjolnir/#platform=thor).
