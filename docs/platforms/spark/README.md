<p align="center">
  <img src="../../../assets/hero-spark.jpg" alt="Mjolnir on DGX Spark" width="100%">
</p>

# DGX Spark

> <img src="../../../assets/emoji/mjolni_spark_emoji.png" alt="" width="20" align="absmiddle"> **research** platform · `sm_121` · host requirements: DGX OS + CUDA 13 · [all platforms](../README.md)


<img src="../../../assets/emoji/mjolni_spark_emoji.png" alt="" width="20" align="absmiddle"> Research platform (GB10, DGX OS + CUDA 13). No Mjolnir kernel targets
`sm_121` yet — the kernel registry ships `sm_110` and `sm_87` — so serving
here runs the stock backends and the platform's data is official-image
baselines only; the stock vLLM image stays the default until a kernel lands.



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
