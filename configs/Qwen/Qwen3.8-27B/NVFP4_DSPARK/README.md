# Qwen/Qwen3.8-27B

`NVFP4_DSPARK` — FlashInfer + DSpark · serving profile. Flags live in
[config.yaml](config.yaml);
the numbers in [BENCHMARKS.md](BENCHMARKS.md) and the model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| Format | FlashInfer + DSpark |
| Max context | 200K |
| Max concurrency | 8 |
| Prefill chunk | 16K |
| KV cache | fp8 · skip: [sliding_window, '64', '65', '66', '67', '68'] |
| Attention backend | FLASHINFER |
| Spec decode | DSPARK ×7 |
| GPU memory util | 0.80 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --platform thor --config NVFP4_DSPARK
```
Availability: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.

Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=NVFP4_DSPARK).
