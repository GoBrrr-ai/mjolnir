# Qwen/Qwen3.8-27B

`NVFP4_MJOLNIR_E2M3` — Mjolnir fp6 KV · serving profile. Flags live in
[config.yaml](config.yaml); every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| Format | Mjolnir fp6 KV |
| Max context | 200K |
| Max concurrency | 8 |
| Prefill chunk | 16K |
| KV cache | fp6_e2m3 · skip: sliding_window |
| Attention backend | MJOLNIR |
| Spec decode | MTP ×3 |
| GPU memory util | 0.80 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config NVFP4_MJOLNIR_E2M3
```
Availability: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4_MJOLNIR_E2M3).

## Benchmarks

<details>
<summary>Speculative decoding</summary>

Mjolnir · fp6 KV, 1 request: best with **3 drafts** — 25.7 t/s per request, 3.04 tokens per step. Mjolnir · fp6 KV, 4 requests: best with **4 drafts** — 24.5 t/s per request, 3.47 tokens per step.

<details>
<summary>How the % and × here are calculated</summary>

- **tokens/step** = 1 + accepted draft tokens ÷ verify steps, from vLLM's speculative-decoding counters read before and after the block; **accept @i** = the share of verify steps whose i-th draft was accepted.
- **decode t/s per request** = (generated tokens − 1) ÷ (time of the last token − time of the first token); **aggregate t/s** = all generated tokens ÷ the block's wall time.
- ★ = the draft count with the highest decode t/s for that config and concurrency.

</details><br />
</details>

<details>
<summary>KV capacity</summary>

**Mjolnir · fp6 e2m3 KV**: **2,645,421 tokens**, 12.9 requests at the full 200K context (vLLM's own estimate).

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
| Tool calls | 98.5 |
| Hard Mode | 83.9 |

<details>
<summary>How the % and × here are calculated</summary>

- **Score** = points ÷ maximum points × 100, each scenario 2 (pass), 1 (partial) or 0 (fail); GSM8K / MMLU = accuracy in %; IFEval = prompt-level accuracy in %.
- **(±)** = this row's score − the baseline row's score, in percentage **points**, not a relative change: 86.5 vs 88.0 is −1.5 points (a relative change would be −1.7%).
- **n** = runs averaged (`--trials`); **completion** = the share of scenarios the endpoint answered — compare rows only at equal completion.

</details><br />
</details>

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#kernel), one chapter per verify stage.
