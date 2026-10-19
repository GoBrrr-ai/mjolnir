# Qwen3.6-35B-A3B

`nvidia/Qwen3.6-35B-A3B` — served as `nvidia/Qwen3.6-35B-A3B-NVFP4` — with 5
serving configs in `configs/nvidia/Qwen3.6-35B-A3B/`. Everything on this page is
derived from the config tree and the raw results; there is no hand-written
model prose (decision 7).

## Serving configs

One dir per config (`format_backend_suffix`, never a platform name — the
platform comes from the overlay). The availability cells are the in-code
table's states for that config's KV dtype × backend class (or its weights
format) on each platform.

| config | backend | thor | spark | orin |
|---|---|---|---|---|
| [`NVFP4_DFLASH`](NVFP4_DFLASH/README.md) · [BENCHMARKS](NVFP4_DFLASH/BENCHMARKS.md) | FlashInfer + DFlash | ✅ | — | — |
| [`NVFP4_MJOLNIR`](NVFP4_MJOLNIR/README.md) · [BENCHMARKS](NVFP4_MJOLNIR/BENCHMARKS.md) | Mjolnir | ✅ | — | — |
| [`NVFP4_MJOLNIR_E2M3`](NVFP4_MJOLNIR_E2M3/README.md) · [BENCHMARKS](NVFP4_MJOLNIR_E2M3/BENCHMARKS.md) | Mjolnir fp6 KV | ✅ | — | — |
| [`NVFP4_MJOLNIR_E2M3_LMH8_VT64K`](NVFP4_MJOLNIR_E2M3_LMH8_VT64K/README.md) · [BENCHMARKS](NVFP4_MJOLNIR_E2M3_LMH8_VT64K/BENCHMARKS.md) | Mjolnir fp6 KV + fp8 lm_head + 64K draft vocab | ✅ | — | — |
| [`NVFP4_MJOLNIR_E2M3_VT64K`](NVFP4_MJOLNIR_E2M3_VT64K/README.md) · [BENCHMARKS](NVFP4_MJOLNIR_E2M3_VT64K/BENCHMARKS.md) | NVFP4_MJOLNIR_E2M3_VT64K | ✅ | — | — |


- `NVFP4_DFLASH` on thor: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.
- `NVFP4_MJOLNIR` on thor: ✅ The Mjolnir kernel reads e4m3 KV natively on thor as its default dtype.
- `NVFP4_MJOLNIR_E2M3` on thor: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.
- `NVFP4_MJOLNIR_E2M3_LMH8_VT64K` on thor: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.
- `NVFP4_MJOLNIR_E2M3_VT64K` on thor: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.


## Quickstart

```bash
mjolnir serve up --platform thor --config NVFP4_MJOLNIR_E2M3_LMH8_VT64K
```

Any config the platform's overlays allow; the full list is the table above.
The per-platform flags (KV layout, the config's `.env` switches) come from
the config itself — [each config's README](NVFP4_DFLASH/README.md) names them.

## End to end: the official vLLM image vs Mjolnir

One row per platform — each from that platform's own history
(`latest_pair(platform, model)`), decode tokens/s per request at 1 request,
longest shared context.

| platform | official | Mjolnir | official t/s | Mjolnir t/s | change | at |
|---|---|---|--:|--:|--:|---|
| Jetson AGX Thor | `NVFP4_DFLASH` | `NVFP4_MJOLNIR_E2M3` | 70.4 | 104.2 | +48.0% | 128K |


<details>
<summary>How the % and × here are calculated</summary>

- **Throughput** (decode, prefill — tokens/s, higher is better): change = (Mjolnir − official) ÷ official × 100. `+57%` = 1.57× the tokens per second. On the infographic a change of +100% or more is written as the multiple Mjolnir ÷ official (`+200%` = `3×`).
- **Latency** (TTFT, ITL — milliseconds, lower is better): change = (Mjolnir − official) ÷ official × 100, so a negative value is a **latency reduction**: `−27%` = 27% fewer milliseconds. It is *not* the reciprocal speedup official ÷ Mjolnir − 1, which reads larger for the same pair.
- **Values**: the mean over the measured runs per cell (6 measured after 2 warm-ups, llama-benchy): decode = generated tokens ÷ generation time per request (summed over requests at higher concurrency), prefill = prompt tokens ÷ prompt-processing time, TTFT = time to the first content token, ITL = 1000 ÷ the per-request decode rate.
- **Colors**: green = better by at least 3%, grey = within ±3% (`E2E_NOISE_PCT`), red = worse by at least that. With speculative decoding a single decode cell moves more between runs (acceptance depends on the prompt), so repeat (`--repeat`) before reading a few-% difference.

</details><br />

> <sub>From [`benchmarks/history.jsonl`](../../../benchmarks/history.jsonl). Platform pages: [Jetson AGX Thor](../../../docs/platforms/thor/README.md).</sub>
