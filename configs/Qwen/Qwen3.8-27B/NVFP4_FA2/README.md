# Qwen/Qwen3.8-27B

`NVFP4_FA2` — FlashAttention 2 · serving profile. Flags live in
[config.yaml](config.yaml);
the numbers in [BENCHMARKS.md](BENCHMARKS.md) and the model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| Format | FlashAttention 2 |
| Max context | 160K |
| Max concurrency | 10 |
| Prefill chunk | 16K |
| KV cache | auto · skip: sliding_window |
| Attention backend | FLASH_ATTN |
| Spec decode | MTP ×3 |
| GPU memory util | 0.88 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --platform thor --config NVFP4_FA2
```
Availability: ✅ BF16 (and fp16) KV is the unquantized baseline: every backend class serves it wherever that backend itself runs.

Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=NVFP4_FA2).
