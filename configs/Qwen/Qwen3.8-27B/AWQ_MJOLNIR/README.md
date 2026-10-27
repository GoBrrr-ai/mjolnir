# Qwen/Qwen3.8-27B

`AWQ_MJOLNIR` — compressed-tensors · serving profile. Flags live in
[config.yaml](config.yaml), platform deltas in [platforms/](platforms/); every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Orin |
|---|---|
| Checkpoint | `twolven/Qwen3.8-27B-abliterated-AWQ-MTP` |
| Format | compressed-tensors |
| Max context | 190K |
| Max concurrency | 5 |
| Prefill chunk | 8K |
| KV cache | fp8 · skip: sliding_window |
| Attention backend | MJOLNIR |
| Spec decode | MTP ×3 |
| GPU memory util | 0.72 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson Orin**
```bash
mjolnir serve up --config AWQ_MJOLNIR
```
Availability: ✅ Marlin serves W4A16 (GPTQ/AWQ) from compute capability 7.5 up — the bandwidth-optimized classic edge format.

Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=AWQ_MJOLNIR).

## Benchmarks

<details>
<summary>Speculative decoding</summary>

AWQ_MJOLNIR, 1 request: best with **3 drafts** — 20.0 t/s per request, 3.12 tokens per step. AWQ_MJOLNIR, 4 requests: best with **3 drafts** — 18.6 t/s per request, 3.13 tokens per step.

<details>
<summary>How the % and × here are calculated</summary>

- **tokens/step** = 1 + accepted draft tokens ÷ verify steps, from vLLM's speculative-decoding counters read before and after the block; **accept @i** = the share of verify steps whose i-th draft was accepted.
- **decode t/s per request** = (generated tokens − 1) ÷ (time of the last token − time of the first token); **aggregate t/s** = all generated tokens ÷ the block's wall time.
- ★ = the draft count with the highest decode t/s for that config and concurrency.

</details><br />
</details>

<details>
<summary>KV capacity</summary>

**Mjolnir · fp8 KV**: **575,866 tokens**, 3.0 requests at the full 190K context (vLLM's own estimate).

<details>
<summary>How the % and × here are calculated</summary>

- **KV tokens** and KV memory come from vLLM's own startup log ("GPU KV cache size", "Available KV cache memory") for the same model and memory budget.
- **vs first** = this row's KV tokens ÷ the first row's (`1.33×` = 33% more tokens).
- **fits @ L** = ⌊KV tokens ÷ (L + 2,048 prompt + 128 generated)⌋ — whole requests of the e2e bench's shape that fit at once (`max-num-seqs` ignored).

</details><br />
</details>

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson Orin kernel case study](../../../../docs/platforms/orin/README.md#kernel), one chapter per verify stage.
