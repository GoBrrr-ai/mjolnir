# <img src="../../assets/emoji/mjolnir_emoji.png" alt="" width="28" align="absmiddle"> Platforms

Mjolnir targets edge GPUs; the platform layer (detection-first, no remembered
state) resolves which one a host is. Every platform has a page here: its
status, what serves on it, its host requirements — and, once kernel results
exist for its headline model, its own kernel case study under `kernel/`.

| platform | status | compute | host requirements |
|---|---|---|---|
| <img src="../../assets/emoji/mjolnir_thor_emoji.png" alt="" width="20" align="absmiddle"> [Jetson AGX Thor](thor/README.md) | supported | `sm_110` | JetPack 7.x |
| <img src="../../assets/emoji/mjolni_spark_emoji.png" alt="" width="20" align="absmiddle"> [DGX Spark](spark/README.md) | research | `sm_121` | DGX OS + CUDA 13 |
| <img src="../../assets/emoji/mjolnir_orin_emoji.png" alt="" width="20" align="absmiddle"> [Jetson Orin](orin/README.md) | supported | `sm_87` | JetPack 7.2.1+ |


Per model and per serving config: the [config tree](../../configs/) — each
model dir has a generated `README.md`, each serving config a `README.md`
(with its Benchmarks section), all rendered from raw data by
`mjolnir plot --readme`.
