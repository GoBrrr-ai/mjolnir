<p align="center">
  <img src="../../../assets/hero-spark.png" alt="Mjolnir on DGX Spark" width="100%">
</p>

# DGX Spark

> **research** platform · `sm_121` · host requirements: DGX OS + CUDA 13 · [all platforms](../README.md)



## Serving on DGX Spark

One row per model in `configs/` — the config the platform's overlays serve
by default (the model page's quickstart rule) and whether the
(platform, model) pair has end-to-end data. Per-config availability lives
on each [model page](../../../configs/).

| model | default config | measured |
|---|---|---|
| [`Qwen/Qwen3.8-27B`](../../../configs/Qwen/Qwen3.8-27B/README.md) | [`NVFP4`](../../../configs/Qwen/Qwen3.8-27B/NVFP4/README.md) | — |
| [`nvidia/Qwen3.6-35B-A3B`](../../../configs/nvidia/Qwen3.6-35B-A3B/README.md) | — | — |


Explore every number in [the interactive charts](https://thor-arena.github.io/mjolnir/#platform=spark).
