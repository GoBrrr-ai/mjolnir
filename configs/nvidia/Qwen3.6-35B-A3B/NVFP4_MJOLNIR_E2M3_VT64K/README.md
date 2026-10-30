# nvidia/Qwen3.6-35B-A3B

`NVFP4_MJOLNIR_E2M3_VT64K` — Mjolnir fp6 KV + 64K draft vocab · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Served as | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Format | Mjolnir fp6 KV + 64K draft vocab |
| Max context | 256K |
| Max concurrency | 1 |
| KV cache | fp6_e2m3 |
| Attention backend | MJOLNIR |
| Spec decode | MTP ×5 |
| GPU memory util | 0.45 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config NVFP4_MJOLNIR_E2M3_VT64K
```
Availability: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4_MJOLNIR_E2M3_VT64K).

## Benchmarks

**no data yet** — no end-to-end, speculative-decoding, KV-capacity or accuracy run for `nvidia/Qwen3.6-35B-A3B` / `NVFP4_MJOLNIR_E2M3_VT64K` on Jetson AGX Thor so far.

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#kernel), one chapter per verify stage.
