# Qwen/Qwen3.8-27B

`NVFP4_FA4hd256` — FlashAttention 4 hd256 · serving profile. Flags live in
[config.yaml](config.yaml); every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| Format | FlashAttention 4 hd256 |
| Max context | 200K |
| Max concurrency | 10 |
| Prefill chunk | 16K |
| KV cache | fp8 · skip: sliding_window |
| Attention backend | FLASH_ATTN@4 |
| Spec decode | MTP ×3 |
| GPU memory util | 0.88 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config NVFP4_FA4hd256
```
Availability: ❌ fp8 KV is served by FlashInfer on every platform and by FA4 / the Mjolnir kernel on thor only

Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=NVFP4_FA4hd256).

## Benchmarks

**no data yet** — no end-to-end, speculative-decoding, KV-capacity or accuracy run for `Qwen/Qwen3.8-27B` / `NVFP4_FA4hd256` on Jetson AGX Thor so far.

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#kernel), one chapter per verify stage.
