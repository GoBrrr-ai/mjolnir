<p align="center">
  <img src="../../../assets/hero-orin.jpg" alt="Mjolnir on Jetson Orin" width="100%">
</p>

# <img src="../../../assets/emoji/mjolnir_orin_emoji.png" alt="" width="28" align="absmiddle"> Jetson Orin

> **supported** platform · `sm_87` · host requirements: JetPack 7.2.1+ · [all platforms](../README.md)


The `sm_87` mma kernel — a CuTe-DSL port of NVIDIA's Ampere FlashAttention-2
example, restructured for paged decode — targets AGX Orin 64GB (16 SMs,
204.8 GB/s of LPDDR5, 4 MB L2) and serves fp8 and fp6 (`e2m3`) KV; it is
serving-validated and measured against the FlashInfer FA2-TC reference, with
the chapters quantifying where it stands. Open: the mma serves head_size 256
only (the hd128 gap is the top integration item), and PDL does not exist on
Ampere.



## Serving on Jetson Orin

<img src="../../../assets/benchmarks/orin/Qwen3.8-27B/opening-orin-infographic.jpg" alt="Mjolnir vs the official vLLM image — end-to-end serving" width="100%">


## Measured Models

One row per model in `configs/` — the config the platform's overlays serve
by default (the model page's quickstart rule) and the pair's measured
speeds. Per-config availability lives on each [model page](../../../configs/).

<table width="100%">
  <thead>
    <tr><th align="left" width="250">model</th><th align="center" width="250">default config</th><th align="center" width="250">decode</th><th align="center" width="250">prefill</th></tr>
  </thead>
  <tbody>
    <tr><td align="left"><a href="../../../configs/Qwen/Qwen3.8-27B/README.md"><code>Qwen/Qwen3.8-27B</code></a></td><td align="center"><a href="../../../configs/Qwen/Qwen3.8-27B/AWQ_FLASHINFER/README.md"><code>AWQ_FLASHINFER</code></a></td><td align="center"><span title="AWQ_MJOLNIR_VT64K">✅ 23.0 t/s</span></td><td align="center"><span title="AWQ_MJOLNIR_VT64K">448.7 t/s</span></td></tr>
    <tr><td align="left"><a href="../../../configs/nvidia/Qwen3.6-35B-A3B/README.md"><code>nvidia/Qwen3.6-35B-A3B</code></a></td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
  </tbody>
</table>

> <sub>✅ at least one of the model's configs serves natively on the platform · — no config targets it. Speeds (when shown): decode tokens/s at 1 request / 0 context and prefill tokens/s at 1 request / 8K context — the fastest-by-decode serving config's latest merged run, its name in the cells' hover title · from [benchmarks/history.jsonl](../../../benchmarks/history.jsonl).</sub>
## <img src="../../../assets/emoji/mjolnir_orin_emoji.png" alt="" width="26" align="absmiddle"> Kernel

Mjolnir on Jetson Orin, measured chapter by chapter and pinned to the
headline model (Qwen/Qwen3.8-27B) — every chart and table rendered
from raw results in [`benchmarks/raw/verify/`](../../../benchmarks/raw/verify/)
by `mjolnir plot --readme`; every chapter ends with the command that
reproduces it.

**Where it stands:** 1.14× FlashInfer (geomean over L = 128 4K 8K 16K 32K 64K 128K 256K), 48 / 48 correctness checks — run `20261030-162038`.

<details>
<summary>How the % and × here are calculated</summary>

- **Time per decode step** = CUDA-event median over the timed launches. The kernel number replays the compiled launch with prebuilt arguments; the FlashInfer reference runs in its own container on the same stock image, warmed up the same way, in the same run.
- **Speedup ×** = reference time ÷ Mjolnir time; below 1 the reference is faster.
- **Geomean over L** = exp(mean of ln(speedup at each L)): every context length weighs the same.
- This stage is CUDA-event timed on this platform — no ncu replay, so no cold-L2 or base-clock control here (the ncu-profiled stages carry those). ±3% (`NOISE_PCT`) is the same-run spread band.

</details><br />

<img src="../../../assets/benchmarks/orin/Qwen3.8-27B/kernel-speedup.png" alt="Mjolnir decode speedup over FlashInfer" width="100%">

| # | chapter | question | status |
|---|---|---|---|
| 1 | [Correctness matrix](kernel/correctness.md) | kernel vs the bit-faithful emulator: every layout, format, page size, q_len, Q dtype | ✅ run `20261030-140528` |
| 2 | [Headline L-sweep](kernel/sweep.md) | Mjolnir vs FlashInfer, fp8 KV — same-run kernel medians | ✅ run `20261030-162038` |
| 3 | [Head-major pages](kernel/hndsweep.md) | FlashInfer on NHD and HND vs Mjolnir on HND — is HND a kernel win or a layout win? | ✅ run `20261030-141453` |
| 4 | [KV layouts and formats](kernel/kvfmt.md) | fp8 / fp6 / fp4 / nvfp4 KV on NHD / interleaved / HND pages (ncu + wall clock) | ✅ run `20261030-151506` |
| 5 | [KV format accuracy](kernel/quant.md) | relative L2 error of each KV format vs unquantized KV, at every L | ✅ run `20261030-153729` |
| 6 | [KV cache capacity](kernel/kv-cache-size.md) | tokens of KV each config fits (vLLM's startup sizing), and the concurrency that buys per context | ✅ run `20261026-203434` |
| 7 | [Speculative decoding](kernel/spec.md) | MTP acceptance per draft position and decode t/s for 1-N draft tokens, per config — the best draft count | ✅ run `20261026-175121` |
| 8 | [Full profiles per KV format](kernel/fp6prof.md) | full ncu profiles: L2 sectors vs requests vs time, warp stalls, per KV format | ✅ run `20261030-154138` |
| 9 | [KV ring depth](kernel/ring.md) | TMA→MMA ring depth (--kv_stages) sweep at every L | ⏳ pending |
| 10 | [L2 prefetch](kernel/prefetch.md) | TMA L2 prefetch distance (--l2_prefetch) sweep | ⏳ pending |
| 11 | [Programmatic dependent launch](kernel/pdl.md) | PDL on/off wall clock (decode overlaps the previous kernel; merge launches early) | ⏳ pending |
| 12 | [Flag A/B](kernel/ab.md) | any harness flag sets, ncu + wall clock (default: evict-first x PDL) | ✅ run `20261030-145834` |
| 13 | [Accuracy](kernel/accuracy.md) | Does a serving config keep the model's answers? Tool calling (standard + Hard Mode), GSM8K / MMLU / IFEval on demand — `mjolnir bench accuracy` | ✅ run `20261017-210613` |




### <img src="../../../assets/emoji/mjolnir_orin_emoji.png" alt="" width="26" align="absmiddle"> Mjolnir Kernel

All kernel numbers are **cuda-event timed kernel time**, one sequence with 24 query / 4 KV heads, head dim 256, 128-token pages — the attention shape of Qwen3.8-27B — at **every context length from 128 to 256K tokens**.
Measured on the Orin: the `sm_87` mma decode kernel against the FlashInfer reference, timed in the same run; the chapters quantify where it stands. Noise and methodology: [`docs/methodology/benchmarking.md`](../../../docs/methodology/benchmarking.md).

| leg | what runs | KV cache |
|---|---|---|
| **Mjolnir MMA** | the `sm_87` `mma.m16n8k16` decode kernel (a CuTe-DSL port of NVIDIA's Ampere FA2 example, restructured for paged decode): one CTA per (KV head, split, batch), `cp.async` staging, smem dequant, split-KV + fused LSE combine | fp8 e4m3, HND pages |
| **FlashInfer FA2-TC** | the official vLLM 0.32.0 FlashInfer tensor-core decode path for `sm_87` | fp8 e4m3, NHD pages |


<img src="../../../assets/benchmarks/orin/Qwen3.8-27B/kernel-latency.png" alt="Decode latency per step vs context length" width="100%">

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

> <sub>Bold = fastest at that `L`. Ratio columns: baseline time ÷ this kernel's time (green = faster beyond noise).</sub>

<img src="../../../assets/benchmarks/orin/Qwen3.8-27B/kernel-bandwidth.png" alt="Effective KV bandwidth vs the LPDDR5 peak" width="100%">


## End to end

The official vLLM image vs Mjolnir on Jetson Orin — the headline model's
own history (`latest_pair`), decode tokens/s per request at 1 request.

| official | Mjolnir | official t/s | Mjolnir t/s | change | at |
|---|---|--:|--:|--:|---|
| `AWQ_FLASHINFER` | `AWQ_MJOLNIR_VT64K` | 16.6 | 22.4 | +34.7% | 32K |

<img src="../../../assets/benchmarks/orin/Qwen3.8-27B/e2e-decode.png" alt="End-to-end decode throughput: official vLLM vs Mjolnir" width="100%">

<img src="../../../assets/benchmarks/orin/Qwen3.8-27B/e2e-prefill.png" alt="End-to-end prompt processing: official vLLM vs Mjolnir" width="100%">


Each cell: official → **Mjolnir** and the relative change of the value — **throughput change** for decode and prefill, **latency change** for TTFT and ITL.

| context | requests | decode (t/s, ↑) $\color{grey}\scriptstyle\text{throughput change}$ | prefill (t/s, ↑) $\color{grey}\scriptstyle\text{throughput change}$ | TTFT (ms, ↓) $\color{grey}\scriptstyle\text{latency change}$ | ITL (ms, ↓) $\color{grey}\scriptstyle\text{latency change}$ |
|--:|--:|--:|--:|--:|--:|
| no ctx | 1 | 18.4 → **23.0** $\color{#16a34a}{\text{+24.6\\%}}$ | — | 4,713 → **4,628** $\color{grey}{\text{−1.8\\%}}$ | 54.5 → **43.8** $\color{#16a34a}{\text{−19.7\\%}}$ |
| 4K | 1 | 17.7 → **21.2** $\color{#16a34a}{\text{+19.7\\%}}$ | 469 → **472** $\color{grey}{\text{+0.6\\%}}$ | 13,320 → **13,255** $\color{grey}{\text{−0.5\\%}}$ | 57.1 → **47.9** $\color{#16a34a}{\text{−16.1\\%}}$ |
| 8K | 1 | 18.6 → **23.6** $\color{#16a34a}{\text{+26.9\\%}}$ | 448 → **449** $\color{grey}{\text{+0.2\\%}}$ | 23,072 → **23,047** $\color{grey}{\text{−0.1\\%}}$ | 54.4 → **42.7** $\color{#16a34a}{\text{−21.6\\%}}$ |
| 16K | 1 | 17.3 → **22.3** $\color{#16a34a}{\text{+28.3\\%}}$ | 428 → **428** $\color{grey}{\text{−0.0\\%}}$ | 43,274 → **43,305** $\color{grey}{\text{+0.1\\%}}$ | 58.3 → **45.4** $\color{#16a34a}{\text{−22.1\\%}}$ |
| 32K | 1 | 16.6 → **22.4** $\color{#16a34a}{\text{+34.7\\%}}$ | 395 → **393** $\color{grey}{\text{−0.4\\%}}$ | 88,381 → **88,742** $\color{grey}{\text{+0.4\\%}}$ | 60.5 → **44.7** $\color{#16a34a}{\text{−26.1\\%}}$ |
| no ctx | 2 | 32.2 → **37.7** $\color{#16a34a}{\text{+17.1\\%}}$ | — | 8,981 → **8,810** $\color{grey}{\text{−1.9\\%}}$ | 59.2 → **48.3** $\color{#16a34a}{\text{−18.4\\%}}$ |
| 4K | 2 | 34.0 → **40.4** $\color{#16a34a}{\text{+18.9\\%}}$ | 462 → **459** $\color{grey}{\text{−0.6\\%}}$ | 26,513 → **26,749** $\color{grey}{\text{+0.9\\%}}$ | 56.1 → **47.0** $\color{#16a34a}{\text{−16.3\\%}}$ |
| 8K | 2 | 32.6 → **38.7** $\color{#16a34a}{\text{+18.9\\%}}$ | 448 → **447** $\color{grey}{\text{−0.2\\%}}$ | 45,596 → **45,786** $\color{grey}{\text{+0.4\\%}}$ | 60.2 → **47.5** $\color{#16a34a}{\text{−21.1\\%}}$ |
| 16K | 2 | 29.1 → **34.4** $\color{#16a34a}{\text{+18.1\\%}}$ | 428 → **428** $\color{grey}{\text{−0.1\\%}}$ | 85,848 → **85,543** $\color{grey}{\text{−0.4\\%}}$ | 63.3 → **51.4** $\color{#16a34a}{\text{−18.8\\%}}$ |
| 32K | 2 | 7.3 → **7.1** $\color{grey}{\text{−1.8\\%}}$ | 393 → **392** $\color{grey}{\text{−0.3\\%}}$ | 164,001 → **163,375** $\color{grey}{\text{−0.4\\%}}$ | 166.0 → **163.2** $\color{grey}{\text{−1.7\\%}}$ |
| no ctx | 4 | 60.3 → **73.2** $\color{#16a34a}{\text{+21.3\\%}}$ | — | 17,462 → **17,276** $\color{grey}{\text{−1.1\\%}}$ | 61.5 → **48.7** $\color{#16a34a}{\text{−20.8\\%}}$ |
| 4K | 4 | 44.2 → **57.1** $\color{#16a34a}{\text{+29.3\\%}}$ | 458 → **456** $\color{grey}{\text{−0.3\\%}}$ | 52,740 → **53,342** $\color{grey}{\text{+1.1\\%}}$ | 67.2 → **53.2** $\color{#16a34a}{\text{−20.8\\%}}$ |
| 8K | 4 | 33.4 → **31.0** $\color{#dc2626}{\text{−7.2\\%}}$ | 446 → **445** $\color{grey}{\text{−0.1\\%}}$ | 89,246 → **88,459** $\color{grey}{\text{−0.9\\%}}$ | 79.8 → **76.0** $\color{#16a34a}{\text{−4.8\\%}}$ |
| 16K | 4 | 6.6 → **6.6** $\color{grey}{\text{−0.4\\%}}$ | 428 → **427** $\color{grey}{\text{−0.3\\%}}$ | 146,771 → **145,938** $\color{grey}{\text{−0.6\\%}}$ | 262.7 → **259.0** $\color{grey}{\text{−1.4\\%}}$ |
| 32K | 4 | 2.4 → **2.4** $\color{grey}{\text{−0.4\\%}}$ | 393 → **392** $\color{grey}{\text{−0.1\\%}}$ | 264,903 → **263,463** $\color{grey}{\text{−0.5\\%}}$ | 767.3 → **767.8** $\color{grey}{\text{+0.1\\%}}$ |

> <sub>official: `vllm/vllm-openai:v0.32.0-ubuntu2404 · AWQ_FLASHINFER (AWQ int4 + FlashInfer)` · Mjolnir: `narandill/mjolnir-vllm:v0.32.0 · AWQ_MJOLNIR_VT64K` · TTFT = time to first token; ITL = inter-token latency (1000 / per-request decode rate) · from [`benchmarks/history.jsonl`](../../../benchmarks/history.jsonl).</sub>

<details>
<summary>How the % and × here are calculated</summary>

- **Throughput** (decode, prefill — tokens/s, higher is better): change = (Mjolnir − official) ÷ official × 100. `+57%` = 1.57× the tokens per second. On the infographic a change of +100% or more is written as the multiple Mjolnir ÷ official (`+200%` = `3×`).
- **Latency** (TTFT, ITL — milliseconds, lower is better): change = (Mjolnir − official) ÷ official × 100, so a negative value is a **latency reduction**: `−27%` = 27% fewer milliseconds. It is *not* the reciprocal speedup official ÷ Mjolnir − 1, which reads larger for the same pair.
- **Values**: the mean over the measured runs per cell (6 measured after 2 warm-ups, llama-benchy): decode = generated tokens ÷ generation time per request (summed over requests at higher concurrency), prefill = prompt tokens ÷ prompt-processing time, TTFT = time to the first content token, ITL = 1000 ÷ the per-request decode rate.
- **Colors**: green = better by at least 3%, grey = within ±3% (`E2E_NOISE_PCT`), red = worse by at least that. With speculative decoding a single decode cell moves more between runs (acceptance depends on the prompt), so repeat (`--repeat`) before reading a few-% difference.

</details><br />


Explore every number in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#platform=orin).
