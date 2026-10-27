# Speculative decoding

> MTP acceptance per draft position and decode t/s for 1-N draft tokens, per config — the best draft count · `mjolnir verify spec` · run `20261027-193846`

## The question

Every config serves Qwen3.8-27B with its MTP head proposing draft tokens that one forward pass of the full model then verifies. Two numbers decide the decode rate: how many tokens a verify step yields (1 + the drafts accepted) and how long the step takes. Each extra draft adds one forward of the draft head — on this model a bf16 MTP layer plus the bf16 lm_head, ~3 GB of reads — so more drafts are only worth it while the last one is still accepted often enough. And does the Mjolnir kernel (fp8 Q, fp6 KV) change what gets accepted?

## How it is measured

For every config and every draft count k the server is started with `--speculative-config` overridden to k (the rest of the config unchanged; the server restarts per point). After two warm-up requests, a fixed prompt set — code, maths, prose, translation, tables, one ~8K-token context — is sent at **temperature 0**, so every point decodes the same text, once per concurrency level. Around each block the stage reads vLLM's speculative-decoding counters:

- **tokens per step** = 1 + accepted ÷ drafts (what one verify forward yields);
- **acceptance per position** = how often draft #1, #2, … was accepted (each needs all earlier ones accepted);
- **decode t/s per request** from the stream: tokens after the first ÷ time from the first to the last token; **aggregate t/s** = all generated tokens ÷ the block's wall time.

## Result

<img src="../../../../assets/benchmarks/orin/Qwen3.8-27B/spec-drafts.png" alt="Decode rate vs draft tokens" width="100%">

- AWQ_FLASHINFER, 1 request: best with **3 drafts** — 18.8 t/s per request, 3.08 tokens per step.
- AWQ_FLASHINFER, 4 requests: best with **4 drafts** — 17.4 t/s per request, 3.45 tokens per step, +0.1% vs 3 drafts.
- AWQ_MJOLNIR_E2M3_VT64K, 1 request: best with **4 drafts** — 25.8 t/s per request, 3.50 tokens per step, +4.4% vs 3 drafts.
- AWQ_MJOLNIR_E2M3_VT64K, 4 requests: best with **4 drafts** — 23.3 t/s per request, 3.53 tokens per step, +4.8% vs 3 drafts.

| config | c | drafts | tokens/step | accept @1 | accept @2 | accept @3 | accept @4 | accept @5 | decode t/s/req | aggregate t/s | TTFT |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| AWQ_FLASHINFER | 1 | 1 | 1.88 | 0.88 |  |  |  |  | 15.4 | 14.2 | 1.49 s |
| AWQ_FLASHINFER | 1 | 2 | 2.59 | 0.87 | 0.72 |  |  |  | 18.0 | 16.2 | 1.58 s |
| AWQ_FLASHINFER | 1 | 3 **★** | 3.08 | 0.85 | 0.68 | 0.55 |  |  | 18.8 | 16.8 | 1.57 s |
| AWQ_FLASHINFER | 1 | 4 | 3.48 | 0.86 | 0.69 | 0.52 | 0.41 |  | 18.7 | 16.7 | 1.63 s |
| AWQ_FLASHINFER | 1 | 5 | 3.82 | 0.84 | 0.69 | 0.54 | 0.42 | 0.33 | 18.7 | 16.5 | 1.68 s |
| AWQ_FLASHINFER | 4 | 1 | 1.89 | 0.89 |  |  |  |  | 14.6 | 46.8 | 0.70 s |
| AWQ_FLASHINFER | 4 | 2 | 2.59 | 0.87 | 0.72 |  |  |  | 16.6 | 54.4 | 0.62 s |
| AWQ_FLASHINFER | 4 | 3 | 3.07 | 0.85 | 0.68 | 0.53 |  |  | 17.4 | 57.3 | 0.66 s |
| AWQ_FLASHINFER | 4 | 4 **★** | 3.45 | 0.85 | 0.67 | 0.53 | 0.40 |  | 17.4 | 57.6 | 0.86 s |
| AWQ_FLASHINFER | 4 | 5 | 3.72 | 0.84 | 0.66 | 0.52 | 0.39 | 0.31 | 16.9 | 57.4 | 0.76 s |
| AWQ_MJOLNIR_E2M3_VT64K | 1 | 1 | 1.88 | 0.88 |  |  |  |  | 17.5 | 15.9 | 1.51 s |
| AWQ_MJOLNIR_E2M3_VT64K | 1 | 2 | 2.55 | 0.86 | 0.69 |  |  |  | 21.8 | 19.4 | 1.50 s |
| AWQ_MJOLNIR_E2M3_VT64K | 1 | 3 | 3.10 | 0.86 | 0.69 | 0.55 |  |  | 24.7 | 21.5 | 1.47 s |
| AWQ_MJOLNIR_E2M3_VT64K | 1 | 4 **★** | 3.50 | 0.84 | 0.68 | 0.54 | 0.43 |  | 25.8 | 22.3 | 1.52 s |
| AWQ_MJOLNIR_E2M3_VT64K | 1 | 5 | 3.69 | 0.83 | 0.65 | 0.52 | 0.40 | 0.30 | 25.7 | 22.0 | 1.54 s |
| AWQ_MJOLNIR_E2M3_VT64K | 4 | 1 | 1.89 | 0.89 |  |  |  |  | 16.4 | 53.2 | 0.54 s |
| AWQ_MJOLNIR_E2M3_VT64K | 4 | 2 | 2.54 | 0.84 | 0.70 |  |  |  | 19.6 | 64.5 | 0.52 s |
| AWQ_MJOLNIR_E2M3_VT64K | 4 | 3 | 3.10 | 0.85 | 0.69 | 0.55 |  |  | 22.3 | 72.6 | 0.66 s |
| AWQ_MJOLNIR_E2M3_VT64K | 4 | 4 **★** | 3.53 | 0.85 | 0.69 | 0.55 | 0.44 |  | 23.3 | 76.3 | 0.73 s |
| AWQ_MJOLNIR_E2M3_VT64K | 4 | 5 | 3.78 | 0.83 | 0.67 | 0.54 | 0.41 | 0.33 | 23.2 | 77.9 | 0.58 s |

<details>
<summary>How the % and × here are calculated</summary>

- **tokens/step** = 1 + accepted draft tokens ÷ verify steps, from vLLM's speculative-decoding counters read before and after the block; **accept @i** = the share of verify steps whose i-th draft was accepted.
- **decode t/s per request** = (generated tokens − 1) ÷ (time of the last token − time of the first token); **aggregate t/s** = all generated tokens ÷ the block's wall time.
- ★ = the draft count with the highest decode t/s for that config and concurrency.

</details><br />

<sub>Rendered from [`benchmarks/raw/verify/spec/20261027-193846.json`](../../../../benchmarks/raw/verify/spec/20261027-193846.json). ★ = the draft count with the highest decode t/s per request for that config and concurrency.</sub>

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

[← KV cache capacity](kv-cache-size.md) · [case study](../README.md#kernel) · [Full profiles per KV format →](fp6prof.md)
