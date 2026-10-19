# Qwen/Qwen3.8-27B

`NVFP4_MJOLNIR` — Mjolnir · serving profile. Flags live in
[config.yaml](config.yaml);
the numbers in [BENCHMARKS.md](BENCHMARKS.md) and the model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| Format | Mjolnir |
| Max context | 200K |
| Max concurrency | 8 |
| Prefill chunk | 16K |
| KV cache | fp8 · skip: sliding_window |
| Attention backend | MJOLNIR |
| Spec decode | MTP ×3 |
| GPU memory util | 0.80 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --platform thor --config NVFP4_MJOLNIR
```
Availability: ✅ The Mjolnir kernel reads e4m3 KV natively on thor as its default dtype.

Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=NVFP4_MJOLNIR).
