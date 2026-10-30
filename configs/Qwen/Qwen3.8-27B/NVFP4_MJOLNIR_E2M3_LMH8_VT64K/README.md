# Qwen/Qwen3.8-27B

`NVFP4_MJOLNIR_E2M3_LMH8_VT64K` — Mjolnir fp6 KV + fp8 lm_head + 64K draft vocab · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| Format | Mjolnir fp6 KV + fp8 lm_head + 64K draft vocab |
| Max context | 200K |
| Max concurrency | 8 |
| Prefill chunk | 16K |
| KV cache | fp6_e2m3 · skip: sliding_window |
| Attention backend | MJOLNIR |
| Spec decode | MTP ×5 |
| GPU memory util | 0.80 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config NVFP4_MJOLNIR_E2M3_LMH8_VT64K
```
Availability: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4_MJOLNIR_E2M3_LMH8_VT64K).

## Benchmarks
### End to end (NVFP4 vs NVFP4_MJOLNIR_E2M3_LMH8_VT64K, 1 request · 32K context)

| metric | official `NVFP4` | Mjolnir `NVFP4_MJOLNIR_E2M3_LMH8_VT64K` | change |
|---|--:|--:|--:|
| decode (t/s) | 24.8 | 32.2 | +29.9% |
| prefill (t/s) | 1,756 | 3,433 | +95.5% |
| TTFT (ms) | 19,964 | 11,222 | −43.8% |
| ITL (ms) | 40.6 | 31.3 | −23.1% |

<img src="../../../../assets/benchmarks/thor/Qwen3.8-27B/e2e-decode.png" alt="End-to-end decode throughput" width="100%">

<details>
<summary>How the % and × here are calculated</summary>

- **Throughput** (decode, prefill — tokens/s, higher is better): change = (Mjolnir − official) ÷ official × 100. `+57%` = 1.57× the tokens per second. On the infographic a change of +100% or more is written as the multiple Mjolnir ÷ official (`+200%` = `3×`).
- **Latency** (TTFT, ITL — milliseconds, lower is better): change = (Mjolnir − official) ÷ official × 100, so a negative value is a **latency reduction**: `−27%` = 27% fewer milliseconds. It is *not* the reciprocal speedup official ÷ Mjolnir − 1, which reads larger for the same pair.
- **Values**: the mean over the measured runs per cell (6 measured after 2 warm-ups, llama-benchy): decode = generated tokens ÷ generation time per request (summed over requests at higher concurrency), prefill = prompt tokens ÷ prompt-processing time, TTFT = time to the first content token, ITL = 1000 ÷ the per-request decode rate.
- **Colors**: green = better by at least 3%, grey = within ±3% (`E2E_NOISE_PCT`), red = worse by at least that. With speculative decoding a single decode cell moves more between runs (acceptance depends on the prompt), so repeat (`--repeat`) before reading a few-% difference.

</details><br />

<details>
<summary>Speculative decoding</summary>

Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab, 1 request: best with **6 drafts** — 39.8 t/s per request, 3.94 tokens per step. Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab, 4 requests: best with **5 drafts** — 34.0 t/s per request, 3.70 tokens per step.

<details>
<summary>How the % and × here are calculated</summary>

- **tokens/step** = 1 + accepted draft tokens ÷ verify steps, from vLLM's speculative-decoding counters read before and after the block; **accept @i** = the share of verify steps whose i-th draft was accepted.
- **decode t/s per request** = (generated tokens − 1) ÷ (time of the last token − time of the first token); **aggregate t/s** = all generated tokens ÷ the block's wall time.
- ★ = the draft count with the highest decode t/s for that config and concurrency.

</details><br />
</details>

<details>
<summary>KV capacity</summary>

**Mjolnir · fp6 e2m3 KV + fp8 lm_head + 64K draft vocab**: **2,608,065 tokens**, 12.7 requests at the full 200K context (vLLM's own estimate).

<details>
<summary>How the % and × here are calculated</summary>

- **KV tokens** and KV memory come from vLLM's own startup log ("GPU KV cache size", "Available KV cache memory") for the same model and memory budget.
- **vs first** = this row's KV tokens ÷ the first row's (`1.33×` = 33% more tokens).
- **fits @ L** = ⌊KV tokens ÷ (L + 2,048 prompt + 128 generated)⌋ — whole requests of the e2e bench's shape that fit at once (`max-num-seqs` ignored).

</details><br />
</details>

<details>
<summary>Model quality (per suite)</summary>

| suite | score |
|---|--:|
| Tool calls | 96.4 |
| Hard Mode | 85.7 |

<details>
<summary>How the % and × here are calculated</summary>

- **Score** = points ÷ maximum points × 100, each scenario 2 (pass), 1 (partial) or 0 (fail); GSM8K / MMLU = accuracy in %; IFEval = prompt-level accuracy in %.
- **(±)** = this row's score − the baseline row's score, in percentage **points**, not a relative change: 86.5 vs 88.0 is −1.5 points (a relative change would be −1.7%).
- **n** = runs averaged (`--trials`); **completion** = the share of scenarios the endpoint answered — compare rows only at equal completion.

</details><br />
</details>

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#-kernel), one chapter per verify stage.
