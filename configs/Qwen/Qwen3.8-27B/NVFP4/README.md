# Qwen/Qwen3.8-27B

`NVFP4` — FlashInfer · serving profile. Flags live in
[config.yaml](config.yaml), platform deltas in [platforms/](platforms/);
the numbers in [BENCHMARKS.md](BENCHMARKS.md) and the model's [README](../README.md).

## Serving profile

| parameter | Thor | Spark | Orin |
|---|---|---|---|
| Checkpoint | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| Format | FlashInfer | FlashInfer | FlashInfer |
| Max context | 200K | 200K | 200K |
| Max concurrency | 8 | 8 | 8 |
| Prefill chunk | 16K | 16K | 16K |
| KV cache | fp8 · skip: sliding_window | fp8 · skip: sliding_window | fp8 · skip: sliding_window |
| Attention backend | FLASHINFER | FLASHINFER | FLASHINFER |
| Spec decode | MTP ×3 | MTP ×3 | MTP ×3 |
| GPU memory util | 0.80 | 0.80 | 0.80 |
| Parsing | qwen3_coder / qwen3 | qwen3_coder / qwen3 | qwen3_coder / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --platform thor --config NVFP4
```
Availability: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.

**DGX Spark**
```bash
mjolnir serve up --platform spark --config NVFP4
```
Availability: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.

**Jetson Orin**
```bash
mjolnir serve up --platform orin --config NVFP4
```
Availability: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.

Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=NVFP4).
