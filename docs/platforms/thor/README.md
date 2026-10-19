# Jetson AGX Thor

> **supported** platform · `sm_110` · host requirements: JetPack 7.x · [all platforms](../README.md)



## Serving on Jetson AGX Thor

Every serving config in `configs/`, whether it exists on Jetson AGX Thor (the
overlay rule), its availability there — one cell, read from the in-code
availability table — and whether the pair has end-to-end data. Availability
is not measurement: links only go where data exists (decision 9).

| config | model | on Jetson AGX Thor | measured |
|---|---|---|---|
| [`NVFP4`](../../../configs/Qwen/Qwen3.8-27B/NVFP4/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: yes |
| [`NVFP4_DFLASH2`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_DFLASH2/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_DFLASH2/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_DSPARK`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_DSPARK/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_DSPARK/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_FA2`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_FA2/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_FA2/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_FA4hd256`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_FA4hd256/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_FA4hd256/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ❌ | measured: no |
| [`NVFP4_MJOLNIR`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_MJOLNIR_E2M3`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_DFLASH2`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_DFLASH2/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_DFLASH2/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_DSPARK`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_DSPARK/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_DSPARK/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_LMH8`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_LMH8_VT32K`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8_VT32K/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8_VT32K/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_LMH8_VT64K`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: yes |
| [`NVFP4_MJOLNIR_E2M3_VT32K`](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_VT32K/README.md) · [benchmarks](../../../configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_VT32K/BENCHMARKS.md) | `Qwen/Qwen3.8-27B` | ✅ | measured: no |
| [`NVFP4_DFLASH`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_DFLASH/README.md) · [benchmarks](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_DFLASH/BENCHMARKS.md) | `nvidia/Qwen3.6-35B-A3B` | ✅ | measured: yes |
| [`NVFP4_MJOLNIR`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR/README.md) · [benchmarks](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR/BENCHMARKS.md) | `nvidia/Qwen3.6-35B-A3B` | ✅ | measured: no |
| [`NVFP4_MJOLNIR_E2M3`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3/README.md) · [benchmarks](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3/BENCHMARKS.md) | `nvidia/Qwen3.6-35B-A3B` | ✅ | measured: yes |
| [`NVFP4_MJOLNIR_E2M3_LMH8_VT64K`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/README.md) · [benchmarks](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/BENCHMARKS.md) | `nvidia/Qwen3.6-35B-A3B` | ✅ | measured: no |
| [`NVFP4_MJOLNIR_E2M3_VT64K`](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3_VT64K/README.md) · [benchmarks](../../../configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3_VT64K/BENCHMARKS.md) | `nvidia/Qwen3.6-35B-A3B` | ✅ | measured: no |


- `NVFP4` on Jetson AGX Thor: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.
- `NVFP4_DFLASH2` on Jetson AGX Thor: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.
- `NVFP4_DSPARK` on Jetson AGX Thor: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.
- `NVFP4_FA2` on Jetson AGX Thor: ✅ BF16 (and fp16) KV is the unquantized baseline: every backend class serves it wherever that backend itself runs.
- `NVFP4_FA4hd256` on Jetson AGX Thor: ❌ fp8 KV is served by FlashInfer on every platform and by FA4 / the Mjolnir kernel on thor only
- `NVFP4_MJOLNIR` on Jetson AGX Thor: ✅ The Mjolnir kernel reads e4m3 KV natively on thor as its default dtype.
- `NVFP4_MJOLNIR_E2M3` on Jetson AGX Thor: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.
- `NVFP4_MJOLNIR_E2M3_DFLASH2` on Jetson AGX Thor: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.
- `NVFP4_MJOLNIR_E2M3_DSPARK` on Jetson AGX Thor: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.
- `NVFP4_MJOLNIR_E2M3_LMH8` on Jetson AGX Thor: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.
- `NVFP4_MJOLNIR_E2M3_LMH8_VT32K` on Jetson AGX Thor: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.
- `NVFP4_MJOLNIR_E2M3_LMH8_VT64K` on Jetson AGX Thor: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.
- `NVFP4_MJOLNIR_E2M3_VT32K` on Jetson AGX Thor: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.
- `NVFP4_DFLASH` on Jetson AGX Thor: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.
- `NVFP4_MJOLNIR` on Jetson AGX Thor: ✅ The Mjolnir kernel reads e4m3 KV natively on thor as its default dtype.
- `NVFP4_MJOLNIR_E2M3` on Jetson AGX Thor: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.
- `NVFP4_MJOLNIR_E2M3_LMH8_VT64K` on Jetson AGX Thor: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.
- `NVFP4_MJOLNIR_E2M3_VT64K` on Jetson AGX Thor: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.


## Kernel case study

The [Jetson AGX Thor kernel case study](kernel/README.md) — one chapter per
experiment, each with its chart, data table and the command that reproduces
it, pinned to the platform's headline model (Qwen/Qwen3.8-27B).
[correctness](kernel/correctness.md) · [sweep](kernel/sweep.md) · [hndsweep](kernel/hndsweep.md) · [kvfmt](kernel/kvfmt.md) · [quant](kernel/quant.md) · [kv-cache-size](kernel/kv-cache-size.md) · [spec](kernel/spec.md) · [fp6prof](kernel/fp6prof.md) · [ring](kernel/ring.md) · [prefetch](kernel/prefetch.md) · [pdl](kernel/pdl.md) · [ab](kernel/ab.md) ·


## End to end

The per-config A/B rows live on each [model page](../../../configs/) — one
row per platform, from that platform's own history; the per-config cells in
the [BENCHMARKS.md](../../../configs/) pages likewise. Platform-level kernel
charts: under `kernel/` above, when they exist.

Explore every number in [the interactive charts](https://thor-arena.github.io/mjolnir/#platform=thor).
