# Qwen/Qwen3.8-27B

`NVFP4_MJOLNIR_E2M3_DSPARK` — Mjolnir fp6 KV + DSpark · serving profile. Flags live in
[config.yaml](config.yaml);
the numbers in [BENCHMARKS.md](BENCHMARKS.md) and the model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| Format | Mjolnir fp6 KV + DSpark |
| Max context | 200K |
| Max concurrency | 8 |
| Prefill chunk | 16K |
| KV cache | fp6_e2m3 · skip: sliding_window |
| Attention backend | MJOLNIR |
| Spec decode | DSPARK ×7 |
| GPU memory util | 0.80 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --platform thor --config NVFP4_MJOLNIR_E2M3_DSPARK
```
Availability: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.

Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=NVFP4_MJOLNIR_E2M3_DSPARK).
