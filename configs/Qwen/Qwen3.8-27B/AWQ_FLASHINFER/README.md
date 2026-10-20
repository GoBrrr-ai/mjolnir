# Qwen/Qwen3.8-27B

`AWQ_FLASHINFER` — AWQ int4 + FlashInfer · serving profile. Flags live in
[config.yaml](config.yaml), platform deltas in [platforms/](platforms/); every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Orin |
|---|---|
| Checkpoint | `twolven/Qwen3.8-27B-abliterated-AWQ-MTP` |
| Format | AWQ int4 + FlashInfer |
| Max context | 190K |
| Max concurrency | 5 |
| Prefill chunk | 8K |
| KV cache | fp8 · skip: sliding_window |
| Attention backend | FLASHINFER |
| Spec decode | MTP ×3 |
| GPU memory util | 0.90 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson Orin**
```bash
mjolnir serve up --config AWQ_FLASHINFER
```
Availability: ✅ Marlin serves W4A16 (GPTQ/AWQ) from compute capability 7.5 up — the bandwidth-optimized classic edge format.

Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=AWQ_FLASHINFER).

## Benchmarks

**no data yet** — no end-to-end, speculative-decoding, KV-capacity or accuracy run for `Qwen/Qwen3.8-27B` / `AWQ_FLASHINFER` on Jetson Orin so far.
