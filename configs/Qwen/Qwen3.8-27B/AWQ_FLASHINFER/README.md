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
| GPU memory util | 0.72 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson Orin**
```bash
mjolnir serve up --config AWQ_FLASHINFER
```
Availability: ✅ Marlin serves W4A16 (GPTQ/AWQ) from compute capability 7.5 up — the bandwidth-optimized classic edge format.

Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=AWQ_FLASHINFER).

## Benchmarks
### End to end (AWQ_FLASHINFER vs AWQ_MJOLNIR_E2M3_VT64K, 1 request · 32K context)

| metric | official `AWQ_FLASHINFER` | Mjolnir `AWQ_MJOLNIR_E2M3_VT64K` | change |
|---|--:|--:|--:|
| decode (t/s) | 16.6 | 19.8 | +18.8% |
| prefill (t/s) | 395 | 377 | −4.6% |
| TTFT (ms) | 88,381 | 92,647 | +4.8% |
| ITL (ms) | 60.5 | 50.7 | −16.1% |

<img src="../../../../assets/benchmarks/orin/Qwen3.8-27B/e2e-decode.png" alt="End-to-end decode throughput" width="100%">

<details>
<summary>How the % and × here are calculated</summary>

- **Throughput** (decode, prefill — tokens/s, higher is better): change = (Mjolnir − official) ÷ official × 100. `+57%` = 1.57× the tokens per second. On the infographic a change of +100% or more is written as the multiple Mjolnir ÷ official (`+200%` = `3×`).
- **Latency** (TTFT, ITL — milliseconds, lower is better): change = (Mjolnir − official) ÷ official × 100, so a negative value is a **latency reduction**: `−27%` = 27% fewer milliseconds. It is *not* the reciprocal speedup official ÷ Mjolnir − 1, which reads larger for the same pair.
- **Values**: the mean over the measured runs per cell (6 measured after 2 warm-ups, llama-benchy): decode = generated tokens ÷ generation time per request (summed over requests at higher concurrency), prefill = prompt tokens ÷ prompt-processing time, TTFT = time to the first content token, ITL = 1000 ÷ the per-request decode rate.
- **Colors**: green = better by at least 3%, grey = within ±3% (`E2E_NOISE_PCT`), red = worse by at least that. With speculative decoding a single decode cell moves more between runs (acceptance depends on the prompt), so repeat (`--repeat`) before reading a few-% difference.

</details><br />

<details>
<summary>Speculative decoding</summary>

AWQ_FLASHINFER, 1 request: best with **3 drafts** — 18.9 t/s per request, 3.11 tokens per step. AWQ_FLASHINFER, 4 requests: best with **3 drafts** — 17.7 t/s per request, 3.12 tokens per step.

<details>
<summary>How the % and × here are calculated</summary>

- **tokens/step** = 1 + accepted draft tokens ÷ verify steps, from vLLM's speculative-decoding counters read before and after the block; **accept @i** = the share of verify steps whose i-th draft was accepted.
- **decode t/s per request** = (generated tokens − 1) ÷ (time of the last token − time of the first token); **aggregate t/s** = all generated tokens ÷ the block's wall time.
- ★ = the draft count with the highest decode t/s for that config and concurrency.

</details><br />
</details>

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson Orin kernel case study](../../../../docs/platforms/orin/README.md#kernel), one chapter per verify stage.
