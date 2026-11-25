# thinkingmachines/Inkling-2-Mini

`NVFP4_FP8KV` — NVFP4_FP8KV · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `thinkingmachines/Inkling-2-Mini` |
| Format | NVFP4_FP8KV |
| Max context | 200K |
| Max concurrency | 8 |
| Prefill chunk | 8K |
| KV cache | fp8_e4m3 |
| Spec decode | DSPARK ×7 |
| GPU memory util | 0.95 |
| Parsing | inkling / inkling |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config NVFP4_FP8KV
```
Availability: ✅ The Mjolnir kernel reads e4m3 KV natively on thor as its default dtype.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4_FP8KV).

## Benchmarks
### End to end (NVFP4_FP8KV, 1 request · 128K context)


| metric | `NVFP4_FP8KV` |
|---|--:|
| decode (t/s) | 40.0 |
| prefill (t/s) | 1,617 |
| TTFT (ms) | 82,513 |
| ITL (ms) | 25.4 |

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#-kernel), one chapter per verify stage.
