# nvidia/Qwen3.8-Flash-Next

`NVFP4_FP8KV` — NVFP4_FP8KV · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| Format | NVFP4_FP8KV |
| Max context | 256K |
| Max concurrency | 1 |
| Prefill chunk | 8K |
| KV cache | fp8_e4m3 |
| Spec decode | MTP ×3 |
| GPU memory util | 0.78 |
| Parsing | qwen3_xml / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config NVFP4_FP8KV
```
Availability: ❌ kv-cache-dtype 'fp8_e4m3' is not in the availability table — known dtypes: bf16, fp6_e2m3, fp8, int4, int8, nvfp4.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4_FP8KV).

## Benchmarks
### End to end (NVFP4_FP8KV, 1 request · 32K context)


| metric | `NVFP4_FP8KV` |
|---|--:|
| decode (t/s) | 43.8 |
| prefill (t/s) | 3,089 |
| TTFT (ms) | 10,464 |
| ITL (ms) | 22.9 |

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#-kernel), one chapter per verify stage.
