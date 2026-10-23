# Jetson Orin

> **research** platform · `sm_87` · host requirements: JetPack 7.2.1+ (L4T 39.2.1, CUDA 13.2) · [all platforms](../README.md)



## Serving on Jetson Orin

One row per model in `configs/` — the config the platform's overlays serve
by default (the model page's quickstart rule) and whether the
(platform, model) pair has end-to-end data. Per-config availability lives
on each [model page](../../../configs/).

| model | default config | measured |
|---|---|---|
| [`Qwen/Qwen3.8-27B`](../../../configs/Qwen/Qwen3.8-27B/README.md) | [`AWQ_FLASHINFER`](../../../configs/Qwen/Qwen3.8-27B/AWQ_FLASHINFER/README.md) | — |
| [`nvidia/Qwen3.6-35B-A3B`](../../../configs/nvidia/Qwen3.6-35B-A3B/README.md) | — | — |



## Kernel

Mjolnir on Jetson Orin, measured chapter by chapter and pinned to the
headline model (Qwen/Qwen3.8-27B) — every chart and table rendered
from raw results in [`benchmarks/raw/verify/`](../../../benchmarks/raw/verify/)
by `mjolnir plot --readme`; every chapter ends with the command that
reproduces it.

**Where it stands:** 0.24× FlashInfer (geomean over L = 128 4K 8K 16K 32K 64K 128K 256K), 24 / 24 correctness checks — run `20261023-095858`.

<details>
<summary>How the % and × here are calculated</summary>

- **Time per decode step** = Nsight Compute kernel time (`gpu__time_duration`) of the decode kernel **plus** its split-KV merge kernel, at base clocks with a cold L2, median over the iterations. The reference (FlashInfer, FlashAttention 4 hd256) is measured the same way, at the same L, in the same run.
- **Speedup ×** = reference time ÷ Mjolnir time. `1.24×` means Mjolnir needs 1 ÷ 1.24 = 81% of the reference's time per step (19% less time); below 1 the reference is faster.
- **Geomean over L** = exp(mean of ln(speedup at each L)): every context length weighs the same, and one large ratio cannot dominate the way it would in an arithmetic mean.
- **Colors / "beyond the noise band"**: green = speedup ≥ 1.03, grey = within ±3%, red ≤ 0.97. ±3% (`NOISE_PCT`) is how far a same-run ratio moves between repeated runs on a dedicated Thor.

</details><br />

<img src="../../../assets/benchmarks/orin/Qwen3.8-27B/kernel-speedup.png" alt="Mjolnir decode speedup over FlashInfer" width="100%">

| # | chapter | question | status |
|---|---|---|---|
| 1 | [Correctness matrix](kernel/correctness.md) | kernel vs the bit-faithful emulator: every layout, format, page size, q_len, Q dtype | ✅ run `20261023-092606` |
| 2 | [Headline L-sweep](kernel/sweep.md) | Mjolnir vs FlashInfer vs FlashAttention 4 hd256 (fp8 Q+KV, on the Mjolnir image), ncu kernel time at every L | ✅ run `20261023-095858` |
| 3 | [Head-major pages](kernel/hndsweep.md) | FlashInfer on NHD and HND vs Mjolnir on HND — is HND a kernel win or a layout win? | ⏳ pending |
| 4 | [KV layouts and formats](kernel/kvfmt.md) | fp8 / fp6 / fp4 / nvfp4 KV on NHD / interleaved / HND pages (ncu + wall clock) | ⏳ pending |
| 5 | [KV format accuracy](kernel/quant.md) | relative L2 error of each KV format vs unquantized KV, at every L | ⏳ pending |
| 6 | [KV cache capacity](kernel/kv-cache-size.md) | tokens of KV each config fits (vLLM's startup sizing), and the concurrency that buys per context | ⏳ pending |
| 7 | [Speculative decoding](kernel/spec.md) | MTP acceptance per draft position and decode t/s for 1-N draft tokens, per config — the best draft count | ⏳ pending |
| 8 | [Full profiles per KV format](kernel/fp6prof.md) | full ncu profiles: L2 sectors vs requests vs time, warp stalls, per KV format | ⏳ pending |
| 9 | [KV ring depth](kernel/ring.md) | TMA→MMA ring depth (--kv_stages) sweep at every L | ⏳ pending |
| 10 | [L2 prefetch](kernel/prefetch.md) | TMA L2 prefetch distance (--l2_prefetch) sweep | ⏳ pending |
| 11 | [Programmatic dependent launch](kernel/pdl.md) | PDL on/off wall clock (decode overlaps the previous kernel; merge launches early) | ⏳ pending |
| 12 | [Flag A/B](kernel/ab.md) | any harness flag sets, ncu + wall clock (default: evict-first x PDL) | ⏳ pending |
| 13 | [Accuracy](kernel/accuracy.md) | Does a serving config keep the model's answers? Tool calling (standard + Hard Mode), GSM8K / MMLU / IFEval on demand — `mjolnir bench accuracy` | ✅ run `20261017-210613` |



Explore every number in [the interactive charts](https://thor-arena.github.io/mjolnir/#platform=orin).
