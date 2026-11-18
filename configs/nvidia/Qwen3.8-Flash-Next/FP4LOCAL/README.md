# nvidia/Qwen3.8-Flash-Next

`FP4LOCAL` — QSA + FP4-Local (MXFP8 side layers) + FP8 hyperconnections · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `nvidia/Qwen3.8-Flash-Next-FP4-Local` |
| Format | QSA + FP4-Local (MXFP8 side layers) + FP8 hyperconnections |
| Max context | 256K |
| Max concurrency | 1 |
| Prefill chunk | 8K |
| KV cache | auto |
| Spec decode | MTP ×3 |
| GPU memory util | 0.75 |
| Parsing | qwen3_xml / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config FP4LOCAL
```
Availability: ✅ BF16 (and fp16) KV is the unquantized baseline: every backend class serves it wherever that backend itself runs.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=FP4LOCAL).

## Benchmarks

**no data yet** — no end-to-end, speculative-decoding, KV-capacity or accuracy run for `nvidia/Qwen3.8-Flash-Next` / `FP4LOCAL` on Jetson AGX Thor so far.

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#-kernel), one chapter per verify stage.
