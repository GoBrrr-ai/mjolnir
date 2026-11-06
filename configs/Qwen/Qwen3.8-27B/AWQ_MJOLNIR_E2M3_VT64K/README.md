# Qwen/Qwen3.8-27B

`AWQ_MJOLNIR_E2M3_VT64K` — Mjolnir fp6 KV + 64K draft vocab, AWQ weights · serving profile. Flags live in
[config.yaml](config.yaml), platform deltas in [platforms/](platforms/), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Orin |
|---|---|
| Checkpoint | `twolven/Qwen3.8-27B-abliterated-AWQ-MTP` |
| Format | Mjolnir fp6 KV + 64K draft vocab, AWQ weights |
| Max context | 190K |
| Max concurrency | 5 |
| Prefill chunk | 8K |
| KV cache | fp6_e2m3 · skip: sliding_window |
| Attention backend | MJOLNIR |
| Spec decode | MTP ×3 |
| GPU memory util | 0.72 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson Orin**
```bash
mjolnir serve up --config AWQ_MJOLNIR_E2M3_VT64K
```
Availability: ✅ Marlin serves W4A16 (GPTQ/AWQ) from compute capability 7.5 up — the bandwidth-optimized classic edge format.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=AWQ_MJOLNIR_E2M3_VT64K).

## Benchmarks
### End to end (AWQ_MJOLNIR_E2M3_VT64K, 1 request · 32K context)


| metric | `AWQ_MJOLNIR_E2M3_VT64K` |
|---|--:|
| decode (t/s) | 20.5 |
| prefill (t/s) | 380 |
| TTFT (ms) | 91,836 |
| ITL (ms) | 49.3 |

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson Orin kernel case study](../../../../docs/platforms/orin/README.md#-kernel), one chapter per verify stage.
