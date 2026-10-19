# nvidia/Qwen3.6-35B-A3B

`NVFP4_DFLASH` — FlashInfer + DFlash · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`;
the numbers in [BENCHMARKS.md](BENCHMARKS.md) and the model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Served as | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Format | FlashInfer + DFlash |
| Max context | 256K |
| Max concurrency | 1 |
| KV cache | fp8 |
| Attention backend | FLASHINFER |
| Spec decode | DFLASH ×12 |
| GPU memory util | 0.45 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --platform thor --config NVFP4_DFLASH
```
Availability: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.

Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=NVFP4_DFLASH).
