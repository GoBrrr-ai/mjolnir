<p align="center">
  <img src="../../../assets/hero-spark.jpg" alt="Mjolnir on DGX Spark" width="100%">
</p>

# <img src="../../../assets/emoji/mjolni_spark_emoji.png" alt="" width="28" align="absmiddle"> DGX Spark

> **research** platform · `sm_121` · host requirements: DGX OS + CUDA 13 · [all platforms](../README.md)


Research platform (GB10, DGX OS + CUDA 13). No Mjolnir kernel targets
`sm_121` yet — the kernel registry ships `sm_110` and `sm_87` — so serving
here runs the stock backends and the platform's data is official-image
baselines only; the stock vLLM image stays the default until a kernel lands.



## Serving on DGX Spark



## Measured Models

One row per model in `configs/` — the config the platform's overlays serve
by default (the model page's quickstart rule) and the pair's measured
speeds. Per-config availability lives on each [model page](../../../configs/).

<table width="100%">
  <thead>
    <tr><th align="left" width="250">model</th><th align="center" width="250">default config</th><th align="center" width="250">decode</th><th align="center" width="250">prefill</th></tr>
  </thead>
  <tbody>
    <tr><td align="left"><a href="../../../configs/Qwen/Qwen3.8-27B/README.md"><code>Qwen/Qwen3.8-27B</code></a></td><td align="center"><a href="../../../configs/Qwen/Qwen3.8-27B/NVFP4/README.md"><code>NVFP4</code></a></td><td align="center">✅</td><td align="center">✅</td></tr>
    <tr><td align="left"><a href="../../../configs/nvidia/Qwen3.6-35B-A3B/README.md"><code>nvidia/Qwen3.6-35B-A3B</code></a></td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
  </tbody>
</table>

> <sub>✅ at least one of the model's configs serves natively on the platform · — no config targets it. Speeds (when shown): decode tokens/s at 1 request / 0 context and prefill tokens/s at 1 request / 8K context — the fastest-by-decode serving config's latest merged run, its name in the cells' hover title · from [benchmarks/history.jsonl](../../../benchmarks/history.jsonl).</sub>

Explore every number in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#platform=spark).
