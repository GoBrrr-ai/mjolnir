# thinkingmachines/Inkling-2-Mini

`NVFP4_LMH8` — NVFP4_LMH8 · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `thinkingmachines/Inkling-2-Mini` |
| Format | NVFP4_LMH8 |
| Max context | 200K |
| Max concurrency | 8 |
| Prefill chunk | 8K |
| Spec decode | DSPARK ×7 |
| GPU memory util | 0.95 |
| Parsing | inkling / inkling |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config NVFP4_LMH8
```
Availability: ✅ BF16 (and fp16) KV is the unquantized baseline: every backend class serves it wherever that backend itself runs.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4_LMH8).

## Benchmarks
### End to end (NVFP4_LMH8, 1 request · 128K context)


| metric | `NVFP4_LMH8` |
|---|--:|
| decode (t/s) | 40.1 |
| prefill (t/s) | 3,312 |
| TTFT (ms) | 40,348 |
| ITL (ms) | 25.2 |

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#-kernel), one chapter per verify stage.
