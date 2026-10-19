# Jetson Orin

> **research** platform · `sm_87` · host requirements: JetPack 7.2.1+ (L4T 39.2.1, CUDA 13.2) · [all platforms](../README.md)



## Serving on Jetson Orin

Every serving config in `configs/`, whether it exists on Jetson Orin (the
overlay rule), its availability there — one cell, read from the in-code
availability table — and whether the pair has end-to-end data. Availability
is not measurement: links only go where data exists (decision 9).

| config | model | on Jetson Orin | measured |
|---|---|---|---|
| [`AWQ_FLASHINFER`](../../../configs/Qwen/Qwen3.8-27B/AWQ_FLASHINFER/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/AWQ_FLASHINFER/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4`](../../../configs/Qwen/Qwen3.8-27B/NVFP4/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_DFLASH2`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_DFLASH2/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_DFLASH2/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_DSPARK`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_DSPARK/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_DSPARK/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_FA2`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_FA2/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_FA2/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_FA4hd256`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_FA4hd256/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_FA4hd256/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_DFLASH2`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_DFLASH2/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_DFLASH2/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_DSPARK`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_DSPARK/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_DSPARK/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_LMH8`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_LMH8_VT32K`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8_VT32K/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8_VT32K/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_LMH8_VT64K`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_VT32K`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_VT32K/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_VT32K/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_DFLASH`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_DFLASH/README.md) · [benchmarks](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_DFLASH/BENCHMARKS.md) | `nvidia/Qwen3.6-35B-A3B` | ✅ | measured: no |
| [`NVFP4_MJOLNIR`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR/README.md) · [benchmarks](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR/BENCHMARKS.md) | `nvidia/Qwen3.6-35B-A3B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3/README.md) · [benchmarks](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3/BENCHMARKS.md) | `nvidia/Qwen3.6-35B-A3B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_LMH8_VT64K`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/README.md) · [benchmarks](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/BENCHMARKS.md) | `nvidia/Qwen3.6-35B-A3B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_VT64K`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3_VT64K/README.md) · [benchmarks](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3_VT64K/BENCHMARKS.md) | `nvidia/Qwen3.6-35B-A3B` | ❌ | measured: no |


- `AWQ_FLASHINFER` on Jetson Orin: ✅ Marlin serves W4A16 (GPTQ/AWQ) from compute capability 7.5 up — the bandwidth-optimized classic edge format.
- `NVFP4` on Jetson Orin: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.
- `NVFP4_DFLASH2` on Jetson Orin: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.
- `NVFP4_DSPARK` on Jetson Orin: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.
- `NVFP4_FA2` on Jetson Orin: ✅ BF16 (and fp16) KV is the unquantized baseline: every backend class serves it wherever that backend itself runs.
- `NVFP4_FA4hd256` on Jetson Orin: ❌ fp8 KV is served by FlashInfer on every platform and by FA4 / the Mjolnir kernel on thor only
- `NVFP4_MJOLNIR` on Jetson Orin: ❌ fp8 KV is served by FlashInfer on every platform and by FA4 / the Mjolnir kernel on thor only
- `NVFP4_MJOLNIR_E2M3` on Jetson Orin: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_MJOLNIR_E2M3_DFLASH2` on Jetson Orin: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_MJOLNIR_E2M3_DSPARK` on Jetson Orin: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_MJOLNIR_E2M3_LMH8` on Jetson Orin: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_MJOLNIR_E2M3_LMH8_VT32K` on Jetson Orin: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_MJOLNIR_E2M3_LMH8_VT64K` on Jetson Orin: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_MJOLNIR_E2M3_VT32K` on Jetson Orin: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_DFLASH` on Jetson Orin: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.
- `NVFP4_MJOLNIR` on Jetson Orin: ❌ fp8 KV is served by FlashInfer on every platform and by FA4 / the Mjolnir kernel on thor only
- `NVFP4_MJOLNIR_E2M3` on Jetson Orin: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_MJOLNIR_E2M3_LMH8_VT64K` on Jetson Orin: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_MJOLNIR_E2M3_VT64K` on Jetson Orin: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it


## Kernel case study

No kernel results for Jetson Orin yet. Jetson Orin is a **research**
platform: native baselines (the upstream vLLM image) until a Mjolnir kernel
lands for it — this page's matrix and the config BENCHMARKS pages carry its
data as it arrives.


## End to end

The per-config A/B rows live on each [model page](../../../configs/) — one
row per platform, from that platform's own history; the per-config cells in
the [BENCHMARKS.md](../../../configs/) pages likewise. Platform-level kernel
charts: under `kernel/` above, when they exist.

Explore every number in [the interactive charts](https://thor-arena.github.io/mjolnir/#platform=orin).
