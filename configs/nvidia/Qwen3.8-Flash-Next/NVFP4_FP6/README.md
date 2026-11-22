# nvidia/Qwen3.8-Flash-Next

`NVFP4_FP6` — NVFP4_FP6 · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| Format | NVFP4_FP6 |
| Max context | 256K |
| Max concurrency | 1 |
| Prefill chunk | 8K |
| KV cache | fp6_e2m3 |
| Spec decode | MTP ×3 |
| GPU memory util | 0.78 |
| Parsing | qwen3_xml / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config NVFP4_FP6
```
Availability: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4_FP6).

## Benchmarks
### End to end (NVFP4_FP6, 1 request · 32K context)


| metric | `NVFP4_FP6` |
|---|--:|
| decode (t/s) | 21.8 |
| prefill (t/s) | 266 |
| TTFT (ms) | 119,234 |
| ITL (ms) | 45.9 |

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#-kernel), one chapter per verify stage.
