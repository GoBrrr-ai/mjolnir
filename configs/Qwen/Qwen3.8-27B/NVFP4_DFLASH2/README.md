# Qwen/Qwen3.8-27B

`NVFP4_DFLASH2` — FlashInfer + DFlash2 · serving profile. Flags live in
[config.yaml](config.yaml); every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| Format | FlashInfer + DFlash2 |
| Max context | 200K |
| Max concurrency | 8 |
| Prefill chunk | 16K |
| KV cache | fp8 · skip: [sliding_window, '64', '65', '66', '67', '68'] |
| Attention backend | FLASHINFER |
| Spec decode | DFLASH ×7 |
| GPU memory util | 0.80 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config NVFP4_DFLASH2
```
Availability: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4_DFLASH2).

## Benchmarks

**no data yet** — no end-to-end, speculative-decoding, KV-capacity or accuracy run for `Qwen/Qwen3.8-27B` / `NVFP4_DFLASH2` on Jetson AGX Thor so far.

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#kernel), one chapter per verify stage.
