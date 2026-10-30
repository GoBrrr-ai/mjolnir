# nvidia/Qwen3.6-35B-A3B

`NVFP4_MJOLNIR_E2M3_LMH8_VT64K` — Mjolnir fp6 KV + fp8 lm_head + 64K draft vocab · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Served as | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Format | Mjolnir fp6 KV + fp8 lm_head + 64K draft vocab |
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
mjolnir serve up --config NVFP4_MJOLNIR_E2M3_LMH8_VT64K
```
Availability: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4_MJOLNIR_E2M3_LMH8_VT64K).

## Benchmarks

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
| Tool calls | 91.3 |
| Hard Mode | 73.2 |

<details>
<summary>How the % and × here are calculated</summary>

- **Score** = points ÷ maximum points × 100, each scenario 2 (pass), 1 (partial) or 0 (fail); GSM8K / MMLU = accuracy in %; IFEval = prompt-level accuracy in %.
- **(±)** = this row's score − the baseline row's score, in percentage **points**, not a relative change: 86.5 vs 88.0 is −1.5 points (a relative change would be −1.7%).
- **n** = runs averaged (`--trials`); **completion** = the share of scenarios the endpoint answered — compare rows only at equal completion.

</details><br />
</details>

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#-kernel), one chapter per verify stage.
