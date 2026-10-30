# Speculative decoding

> MTP acceptance per draft position and decode t/s for 1-N draft tokens, per config — the best draft count · `mjolnir verify spec` · run `20261026-175121`

## The question

Every config serves Qwen3.8-27B with its MTP head proposing draft tokens that one forward pass of the full model then verifies. Two numbers decide the decode rate: how many tokens a verify step yields (1 + the drafts accepted) and how long the step takes. Each extra draft adds one forward of the draft head — on this model a bf16 MTP layer plus the bf16 lm_head, ~3 GB of reads — so more drafts are only worth it while the last one is still accepted often enough. And does the Mjolnir kernel (fp8 Q, fp6 KV) change what gets accepted?

## How it is measured

For every config and every draft count k the server is started with `--speculative-config` overridden to k (the rest of the config unchanged; the server restarts per point). After two warm-up requests, a fixed prompt set — code, maths, prose, translation, tables, one ~8K-token context — is sent at **temperature 0**, so every point decodes the same text, once per concurrency level. Around each block the stage reads vLLM's speculative-decoding counters:

- **tokens per step** = 1 + accepted ÷ drafts (what one verify forward yields);
- **acceptance per position** = how often draft #1, #2, … was accepted (each needs all earlier ones accepted);
- **decode t/s per request** from the stream: tokens after the first ÷ time from the first to the last token; **aggregate t/s** = all generated tokens ÷ the block's wall time.

## Result

<img src="../../../../assets/benchmarks/orin/Qwen3.8-27B/spec-drafts.png" alt="Decode rate vs draft tokens" width="100%">

- AWQ_FLASHINFER, 1 request: best with **3 drafts** — 18.9 t/s per request, 3.11 tokens per step.
- AWQ_FLASHINFER, 4 requests: best with **3 drafts** — 17.7 t/s per request, 3.12 tokens per step.
- AWQ_MJOLNIR, 1 request: best with **3 drafts** — 20.0 t/s per request, 3.12 tokens per step.
- AWQ_MJOLNIR, 4 requests: best with **3 drafts** — 18.6 t/s per request, 3.13 tokens per step.
- AWQ_MJOLNIR_E2M3, 1 request: best with **3 drafts** — 20.1 t/s per request, 3.16 tokens per step.
- AWQ_MJOLNIR_E2M3, 4 requests: best with **4 drafts** — 18.3 t/s per request, 3.52 tokens per step, +0.0% vs 3 drafts.
- AWQ_MJOLNIR_E2M3_VT64K, 1 request: best with **5 drafts** — 25.8 t/s per request, 3.71 tokens per step, +3.6% vs 3 drafts.
- AWQ_MJOLNIR_E2M3_VT64K, 4 requests: best with **5 drafts** — 23.0 t/s per request, 3.73 tokens per step, +4.7% vs 3 drafts.

| config | c | drafts | tokens/step | accept @1 | accept @2 | accept @3 | accept @4 | accept @5 | decode t/s/req | aggregate t/s | TTFT |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| AWQ_FLASHINFER | 1 | 1 | 1.89 | 0.89 |  |  |  |  | 15.4 | 14.2 | 1.50 s |
| AWQ_FLASHINFER | 1 | 2 | 2.59 | 0.87 | 0.72 |  |  |  | 18.0 | 16.2 | 1.57 s |
| AWQ_FLASHINFER | 1 | 3 **★** | 3.11 | 0.86 | 0.70 | 0.55 |  |  | 18.9 | 16.9 | 1.58 s |
| AWQ_FLASHINFER | 1 | 4 | 3.41 | 0.85 | 0.66 | 0.51 | 0.39 |  | 18.4 | 16.3 | 1.63 s |
| AWQ_FLASHINFER | 1 | 5 | 3.80 | 0.84 | 0.68 | 0.54 | 0.42 | 0.33 | 18.4 | 16.3 | 1.67 s |
| AWQ_FLASHINFER | 4 | 1 | 1.89 | 0.89 |  |  |  |  | 14.6 | 47.2 | 0.55 s |
| AWQ_FLASHINFER | 4 | 2 | 2.60 | 0.87 | 0.73 |  |  |  | 16.8 | 54.4 | 0.63 s |
| AWQ_FLASHINFER | 4 | 3 **★** | 3.12 | 0.86 | 0.71 | 0.55 |  |  | 17.7 | 57.8 | 0.65 s |
| AWQ_FLASHINFER | 4 | 4 | 3.47 | 0.84 | 0.69 | 0.53 | 0.41 |  | 17.3 | 57.9 | 0.72 s |
| AWQ_FLASHINFER | 4 | 5 | 3.87 | 0.86 | 0.70 | 0.55 | 0.43 | 0.34 | 17.6 | 58.9 | 0.77 s |
| AWQ_MJOLNIR | 1 | 1 | 1.88 | 0.88 |  |  |  |  | 16.0 | 14.7 | 1.47 s |
| AWQ_MJOLNIR | 1 | 2 | 2.59 | 0.87 | 0.71 |  |  |  | 18.8 | 16.9 | 1.51 s |
| AWQ_MJOLNIR | 1 | 3 **★** | 3.12 | 0.87 | 0.70 | 0.55 |  |  | 20.0 | 17.8 | 1.53 s |
| AWQ_MJOLNIR | 1 | 4 | 3.37 | 0.85 | 0.65 | 0.50 | 0.37 |  | 19.1 | 17.0 | 1.52 s |
| AWQ_MJOLNIR | 1 | 5 | 3.71 | 0.83 | 0.66 | 0.52 | 0.40 | 0.30 | 19.1 | 16.8 | 1.61 s |
| AWQ_MJOLNIR | 4 | 1 | 1.88 | 0.88 |  |  |  |  | 15.1 | 48.8 | 0.55 s |
| AWQ_MJOLNIR | 4 | 2 | 2.58 | 0.87 | 0.71 |  |  |  | 17.4 | 56.3 | 0.65 s |
| AWQ_MJOLNIR | 4 | 3 **★** | 3.13 | 0.88 | 0.70 | 0.55 |  |  | 18.6 | 61.1 | 0.70 s |
| AWQ_MJOLNIR | 4 | 4 | 3.44 | 0.85 | 0.67 | 0.52 | 0.40 |  | 18.0 | 59.8 | 0.65 s |
| AWQ_MJOLNIR | 4 | 5 | 3.85 | 0.85 | 0.70 | 0.55 | 0.42 | 0.33 | 18.2 | 60.7 | 0.73 s |
| AWQ_MJOLNIR_E2M3 | 1 | 1 | 1.88 | 0.88 |  |  |  |  | 16.0 | 14.6 | 1.50 s |
| AWQ_MJOLNIR_E2M3 | 1 | 2 | 2.61 | 0.89 | 0.72 |  |  |  | 18.9 | 17.0 | 1.54 s |
| AWQ_MJOLNIR_E2M3 | 1 | 3 **★** | 3.16 | 0.87 | 0.72 | 0.57 |  |  | 20.1 | 17.8 | 1.57 s |
| AWQ_MJOLNIR_E2M3 | 1 | 4 | 3.52 | 0.85 | 0.69 | 0.55 | 0.43 |  | 19.9 | 17.5 | 1.62 s |
| AWQ_MJOLNIR_E2M3 | 1 | 5 | 3.78 | 0.84 | 0.67 | 0.54 | 0.41 | 0.32 | 19.3 | 17.0 | 1.63 s |
| AWQ_MJOLNIR_E2M3 | 4 | 1 | 1.88 | 0.88 |  |  |  |  | 15.1 | 48.7 | 0.61 s |
| AWQ_MJOLNIR_E2M3 | 4 | 2 | 2.59 | 0.88 | 0.72 |  |  |  | 17.2 | 56.9 | 0.58 s |
| AWQ_MJOLNIR_E2M3 | 4 | 3 | 3.12 | 0.87 | 0.69 | 0.55 |  |  | 18.3 | 60.9 | 0.65 s |
| AWQ_MJOLNIR_E2M3 | 4 | 4 **★** | 3.52 | 0.84 | 0.69 | 0.55 | 0.43 |  | 18.3 | 59.5 | 0.72 s |
| AWQ_MJOLNIR_E2M3 | 4 | 5 | 3.74 | 0.85 | 0.66 | 0.53 | 0.39 | 0.30 | 17.5 | 58.1 | 0.72 s |
| AWQ_MJOLNIR_E2M3_VT64K | 1 | 1 | 1.88 | 0.88 |  |  |  |  | 17.5 | 15.9 | 1.50 s |
| AWQ_MJOLNIR_E2M3_VT64K | 1 | 2 | 2.57 | 0.87 | 0.70 |  |  |  | 22.0 | 19.5 | 1.49 s |
| AWQ_MJOLNIR_E2M3_VT64K | 1 | 3 | 3.14 | 0.86 | 0.71 | 0.57 |  |  | 24.9 | 21.6 | 1.50 s |
| AWQ_MJOLNIR_E2M3_VT64K | 1 | 4 | 3.49 | 0.84 | 0.68 | 0.54 | 0.43 |  | 25.7 | 22.2 | 1.53 s |
| AWQ_MJOLNIR_E2M3_VT64K | 1 | 5 **★** | 3.71 | 0.83 | 0.65 | 0.52 | 0.40 | 0.31 | 25.8 | 22.0 | 1.54 s |
| AWQ_MJOLNIR_E2M3_VT64K | 4 | 1 | 1.88 | 0.88 |  |  |  |  | 16.5 | 52.8 | 0.61 s |
| AWQ_MJOLNIR_E2M3_VT64K | 4 | 2 | 2.60 | 0.88 | 0.73 |  |  |  | 20.0 | 65.5 | 0.56 s |
| AWQ_MJOLNIR_E2M3_VT64K | 4 | 3 | 3.08 | 0.86 | 0.69 | 0.54 |  |  | 22.0 | 73.7 | 0.57 s |
| AWQ_MJOLNIR_E2M3_VT64K | 4 | 4 | 3.48 | 0.84 | 0.68 | 0.54 | 0.42 |  | 23.0 | 75.5 | 0.60 s |
| AWQ_MJOLNIR_E2M3_VT64K | 4 | 5 **★** | 3.73 | 0.85 | 0.66 | 0.52 | 0.40 | 0.30 | 23.0 | 78.2 | 0.60 s |

<details>
<summary>How the % and × here are calculated</summary>

- **tokens/step** = 1 + accepted draft tokens ÷ verify steps, from vLLM's speculative-decoding counters read before and after the block; **accept @i** = the share of verify steps whose i-th draft was accepted.
- **decode t/s per request** = (generated tokens − 1) ÷ (time of the last token − time of the first token); **aggregate t/s** = all generated tokens ÷ the block's wall time.
- ★ = the draft count with the highest decode t/s for that config and concurrency.

</details><br />

<sub>Rendered from [`benchmarks/raw/verify/spec/20261026-175121.json`](../../../../benchmarks/raw/verify/spec/20261026-175121.json). ★ = the draft count with the highest decode t/s per request for that config and concurrency.</sub>

## How to read it

Tokens per step rises with k but flattens as later positions are accepted less; the step cost rises linearly with k. The best k is where the next draft's acceptance no longer pays for its forward. If the Mjolnir configs accept fewer tokens per step than the official one on the same prompts, the kernel's numerics (fp8 Q, fp6 KV) shift the target model's distribution — that slows decode no matter how fast the kernel is. Cheaper drafts (an fp8 lm_head, an fp8 MTP layer) move the optimum to more drafts: re-run after such changes.

## Reproduce

```bash
mjolnir serve down                                         # restarts per point — the GPU must be yours
mjolnir verify spec                                        # 4 configs × k = 1..4, 1 and 4 requests
mjolnir verify spec --legs mjolnir:NVFP4_MJOLNIR_E2M3 --drafts "2 3"
```

## Ideas to try

- an fp8 MTP layer (the draft head is still bf16 — the lm_head is already fp8 via `MJOLNIR_LM_HEAD`), then `mjolnir verify spec`;
- per-request adaptive draft length (stop drafting when the draft's confidence drops).

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../../../.github/CONTRIBUTING.md).

---

[← KV cache capacity](kv-cache-size.md) · [case study](../README.md#-kernel) · [Full profiles per KV format →](fp6prof.md)
