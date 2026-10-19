# nvidia/Qwen3.6-35B-A3B

`NVFP4_MJOLNIR_E2M3` — Mjolnir fp6 KV · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`;
the numbers in [BENCHMARKS.md](BENCHMARKS.md) and the model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Served as | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Format | Mjolnir fp6 KV |
| Max context | 256K |
| Max concurrency | 1 |
| KV cache | fp6_e2m3 |
| Attention backend | MJOLNIR |
| Spec decode | MTP ×5 |
| GPU memory util | 0.45 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --platform thor --config NVFP4_MJOLNIR_E2M3
```
Availability: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.

Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=NVFP4_MJOLNIR_E2M3).
