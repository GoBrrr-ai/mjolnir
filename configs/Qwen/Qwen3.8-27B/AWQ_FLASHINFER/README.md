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

<details>
<summary>Speculative decoding</summary>

AWQ_FLASHINFER, 1 request: best with **3 drafts** — 18.8 t/s per request, 3.08 tokens per step. AWQ_FLASHINFER, 4 requests: best with **4 drafts** — 17.4 t/s per request, 3.45 tokens per step.

<details>
<summary>How the % and × here are calculated</summary>

- **tokens/step** = 1 + accepted draft tokens ÷ verify steps, from vLLM's speculative-decoding counters read before and after the block; **accept @i** = the share of verify steps whose i-th draft was accepted.
- **decode t/s per request** = (generated tokens − 1) ÷ (time of the last token − time of the first token); **aggregate t/s** = all generated tokens ÷ the block's wall time.
- ★ = the draft count with the highest decode t/s for that config and concurrency.

</details><br />
</details>

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson Orin kernel case study](../../../../docs/platforms/orin/README.md#kernel), one chapter per verify stage.
