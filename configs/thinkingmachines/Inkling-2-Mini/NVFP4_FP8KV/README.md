# thinkingmachines/Inkling-2-Mini

`NVFP4_FP8KV` — NVFP4_FP8KV · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `thinkingmachines/Inkling-2-Mini` |
| Format | NVFP4_FP8KV |
| Max context | 256K |
| Max concurrency | 256 |
| Prefill chunk | 8K |
| KV cache | fp8_e4m3 |
| Spec decode | DSPARK ×7 |
| GPU memory util | 0.80 |
| Parsing | inkling / inkling |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config NVFP4_FP8KV
```
Availability: ❌ kv-cache-dtype 'fp8_e4m3' is not in the availability table — known dtypes: bf16, fp6_e2m3, fp8, int4, int8, nvfp4.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4_FP8KV).

## Benchmarks

**no data yet** — no end-to-end, speculative-decoding, KV-capacity or accuracy run for `thinkingmachines/Inkling-2-Mini` / `NVFP4_FP8KV` on Jetson AGX Thor so far.

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#-kernel), one chapter per verify stage.
