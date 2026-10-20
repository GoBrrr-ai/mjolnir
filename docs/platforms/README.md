# Platforms

Mjolnir targets edge GPUs; the platform layer (detection-first, no remembered
state) resolves which one a host is. Every platform has a page here: its
status, what serves on it, its host requirements — and, once kernel results
exist for its headline model, its own kernel case study under `kernel/`.

| platform | status | compute | host requirements |
|---|---|---|---|
| [Jetson AGX Thor](thor/README.md) | supported | `sm_110` | JetPack 7.x |
| [DGX Spark](spark/README.md) | research | `sm_121` | DGX OS + CUDA 13 |
| [Jetson Orin](orin/README.md) | research | `sm_87` | JetPack 7.2.1+ (L4T 39.2.1, CUDA 13.2) |


Per model and per serving config: the [config tree](../../configs/) — each
model dir has a generated `README.md`, each serving config a `README.md`
(with its Benchmarks section), all rendered from raw data by
`mjolnir plot --readme`.
