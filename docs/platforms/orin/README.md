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




Explore every number in [the interactive charts](https://thor-arena.github.io/mjolnir/#platform=orin).
