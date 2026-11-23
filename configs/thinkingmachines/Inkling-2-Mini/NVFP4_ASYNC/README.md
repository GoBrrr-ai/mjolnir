# thinkingmachines/Inkling-2-Mini

`NVFP4` — FlashInfer · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `thinkingmachines/Inkling-2-Mini` |
| Format | FlashInfer |
| Max context | 256K |
| Max concurrency | 256 |
| Prefill chunk | 8K |
| Spec decode | DSPARK ×7 |
| GPU memory util | 0.80 |
| Parsing | inkling / inkling |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config NVFP4
```
Availability: ✅ BF16 (and fp16) KV is the unquantized baseline: every backend class serves it wherever that backend itself runs.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4).

## Benchmarks
### End to end (NVFP4, 1 request · 128K context)


| metric | `NVFP4` |
|---|--:|
| decode (t/s) | 32.9 |
| prefill (t/s) | 3,394 |
| TTFT (ms) | 40,368 |
| ITL (ms) | 30.8 |

<details>
<summary>Speculative decoding</summary>

official vLLM · FlashInfer fp8, 1 request: best with **4 drafts** — 25.8 t/s per request, 3.46 tokens per step. official vLLM · FlashInfer fp8, 4 requests: best with **4 drafts** — 24.8 t/s per request, 3.50 tokens per step.

<details>
<summary>How the % and × here are calculated</summary>

- **tokens/step** = 1 + accepted draft tokens ÷ verify steps, from vLLM's speculative-decoding counters read before and after the block; **accept @i** = the share of verify steps whose i-th draft was accepted.
- **decode t/s per request** = (generated tokens − 1) ÷ (time of the last token − time of the first token); **aggregate t/s** = all generated tokens ÷ the block's wall time.
- ★ = the draft count with the highest decode t/s for that config and concurrency.

</details><br />
</details>

<details>
<summary>KV capacity</summary>

**official vLLM · FlashInfer · fp8 KV**: **1,715,683 tokens**, 8.4 requests at the full 200K context (vLLM's own estimate).

<details>
<summary>How the % and × here are calculated</summary>

- **KV tokens** and KV memory come from vLLM's own startup log ("GPU KV cache size", "Available KV cache memory") for the same model and memory budget.
- **vs first** = this row's KV tokens ÷ the first row's (`1.33×` = 33% more tokens).
- **fits @ L** = ⌊KV tokens ÷ (L + 2,048 prompt + 128 generated)⌋ — whole requests of the e2e bench's shape that fit at once (`max-num-seqs` ignored).

</details><br />
</details>

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#-kernel), one chapter per verify stage.
