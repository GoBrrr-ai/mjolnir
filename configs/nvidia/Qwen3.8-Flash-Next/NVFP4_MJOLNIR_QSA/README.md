# nvidia/Qwen3.8-Flash-Next

`NVFP4_MJOLNIR_QSA` — NVFP4_MJOLNIR_QSA · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| Format | NVFP4_MJOLNIR_QSA |
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
mjolnir serve up --config NVFP4_MJOLNIR_QSA
```
Availability: ❌ kv-cache-dtype 'fp8_e4m3' is not in the availability table — known dtypes: bf16, fp6_e2m3, fp8, int4, int8, nvfp4.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4_MJOLNIR_QSA).

## Benchmarks
### End to end (NVFP4_MJOLNIR_QSA, 1 request · 32K context)


| metric | `NVFP4_MJOLNIR_QSA` |
|---|--:|
| decode (t/s) | 42.7 |
| prefill (t/s) | 3,085 |
| TTFT (ms) | 10,457 |
| ITL (ms) | 23.7 |

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#-kernel), one chapter per verify stage.
