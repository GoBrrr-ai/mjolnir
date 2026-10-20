# DGX Spark

> **research** platform · `sm_121` · host requirements: DGX OS + CUDA 13 · [all platforms](../README.md)



## Serving on DGX Spark

Every serving config in `configs/`, whether it exists on DGX Spark (the
overlay rule), its availability there — one cell, read from the in-code
availability table — and whether the pair has end-to-end data. Availability
is not measurement: links only go where data exists (decision 9).

| config | model | on DGX Spark | measured |
|---|---|---|---|
| `AWQ_FLASHINFER` | `Qwen/Qwen3.8-27B` | — | measured: no |
| [`NVFP4`](../../../configs/Qwen/Qwen3.8-27B/NVFP4/README.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_DFLASH2`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_DFLASH2/README.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_DSPARK`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_DSPARK/README.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_FA2`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_FA2/README.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_FA4hd256`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_FA4hd256/README.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR/README.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3/README.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_DFLASH2`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_DFLASH2/README.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_DSPARK`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_DSPARK/README.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_LMH8_VT64K`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/README.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_DFLASH`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_DFLASH/README.md) | `nvidia/Qwen3.6-35B-A3B` | ✅ | measured: no |
| [`NVFP4_MJOLNIR`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR/README.md) | `nvidia/Qwen3.6-35B-A3B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3/README.md) | `nvidia/Qwen3.6-35B-A3B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_LMH8_VT64K`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/README.md) | `nvidia/Qwen3.6-35B-A3B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_VT64K`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3_VT64K/README.md) | `nvidia/Qwen3.6-35B-A3B` | ❌ | measured: no |


- `NVFP4` on DGX Spark: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.
- `NVFP4_DFLASH2` on DGX Spark: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.
- `NVFP4_DSPARK` on DGX Spark: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.
- `NVFP4_FA2` on DGX Spark: ✅ BF16 (and fp16) KV is the unquantized baseline: every backend class serves it wherever that backend itself runs.
- `NVFP4_FA4hd256` on DGX Spark: ❌ fp8 KV is served by FlashInfer on every platform and by FA4 / the Mjolnir kernel on thor only
- `NVFP4_MJOLNIR` on DGX Spark: ❌ fp8 KV is served by FlashInfer on every platform and by FA4 / the Mjolnir kernel on thor only
- `NVFP4_MJOLNIR_E2M3` on DGX Spark: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_MJOLNIR_E2M3_DFLASH2` on DGX Spark: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_MJOLNIR_E2M3_DSPARK` on DGX Spark: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_MJOLNIR_E2M3_LMH8_VT64K` on DGX Spark: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_DFLASH` on DGX Spark: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.
- `NVFP4_MJOLNIR` on DGX Spark: ❌ fp8 KV is served by FlashInfer on every platform and by FA4 / the Mjolnir kernel on thor only
- `NVFP4_MJOLNIR_E2M3` on DGX Spark: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_MJOLNIR_E2M3_LMH8_VT64K` on DGX Spark: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it
- `NVFP4_MJOLNIR_E2M3_VT64K` on DGX Spark: ❌ fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it


## Kernel case study

No kernel results for DGX Spark yet. DGX Spark is a **research**
platform: native baselines (the upstream vLLM image) until a Mjolnir kernel
lands for it — this page's matrix and the config pages' Benchmarks sections
carry its data as it arrives.


## End to end

The per-config A/B rows live on each [model page](../../../configs/) — one
row per platform, from that platform's own history; the per-config cells in
each config's Benchmarks section likewise. Platform-level kernel
charts: under `kernel/` above, when they exist.

Explore every number in [the interactive charts](https://thor-arena.github.io/mjolnir/#platform=spark).
