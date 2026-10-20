# Speculative decoding

> MTP acceptance per draft position and decode t/s for 1-N draft tokens, per config — the best draft count · `mjolnir verify spec` · run `20261014-182829`

## The question

Every config serves Qwen3.8-27B with its MTP head proposing draft tokens that one forward pass of the full model then verifies. Two numbers decide the decode rate: how many tokens a verify step yields (1 + the drafts accepted) and how long the step takes. Each extra draft adds one forward of the draft head — on this model a bf16 MTP layer plus the bf16 lm_head, ~3 GB of reads — so more drafts are only worth it while the last one is still accepted often enough. And does the Mjolnir kernel (fp8 Q, fp6 KV) change what gets accepted?

## How it is measured

For every config and every draft count k the server is started with `--speculative-config` overridden to k (the rest of the config unchanged; the server restarts per point). After two warm-up requests, a fixed prompt set — code, maths, prose, translation, tables, one ~8K-token context — is sent at **temperature 0**, so every point decodes the same text, once per concurrency level. Around each block the stage reads vLLM's speculative-decoding counters:

- **tokens per step** = 1 + accepted ÷ drafts (what one verify forward yields);
- **acceptance per position** = how often draft #1, #2, … was accepted (each needs all earlier ones accepted);
- **decode t/s per request** from the stream: tokens after the first ÷ time from the first to the last token; **aggregate t/s** = all generated tokens ÷ the block's wall time.

## Result

<img src="../../../../assets/benchmarks/thor/Qwen3.8-27B/spec-drafts.png" alt="Decode rate vs draft tokens" width="100%">

- official vLLM · FlashInfer fp8, 1 request: best with **4 drafts** — 25.8 t/s per request, 3.46 tokens per step, +4.9% vs 3 drafts.
- official vLLM · FlashInfer fp8, 4 requests: best with **4 drafts** — 24.8 t/s per request, 3.50 tokens per step, +0.9% vs 3 drafts.
- Mjolnir · fp8 KV, 1 request: best with **3 drafts** — 25.9 t/s per request, 3.02 tokens per step.
- Mjolnir · fp8 KV, 4 requests: best with **4 drafts** — 24.3 t/s per request, 3.41 tokens per step, +4.5% vs 3 drafts.
- Mjolnir · fp6 KV, 1 request: best with **3 drafts** — 25.7 t/s per request, 3.04 tokens per step.
- Mjolnir · fp6 KV, 4 requests: best with **4 drafts** — 24.5 t/s per request, 3.47 tokens per step, +5.8% vs 3 drafts.
- Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab, 1 request: best with **6 drafts** — 39.8 t/s per request, 3.94 tokens per step, +6.4% vs 3 drafts.
- Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab, 4 requests: best with **5 drafts** — 34.0 t/s per request, 3.70 tokens per step, +7.1% vs 3 drafts.
- Acceptance, 3 drafts, 1 request: Mjolnir · fp8 KV 3.02 vs official 3.08 tokens/step (-2.0%) — same prompts, temperature 0.
- Acceptance, 3 drafts, 1 request: Mjolnir · fp6 KV 3.04 vs official 3.08 tokens/step (-1.3%) — same prompts, temperature 0.
- Acceptance, 3 drafts, 1 request: Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab 3.06 vs official 3.08 tokens/step (-0.7%) — same prompts, temperature 0.

| config | c | drafts | tokens/step | accept @1 | accept @2 | accept @3 | accept @4 | accept @5 | accept @6 | accept @7 | decode t/s/req | aggregate t/s | TTFT |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| official vLLM · FlashInfer fp8 | 1 | 1 | 1.87 | 0.87 |  |  |  |  |  |  | 20.9 | 20.4 | 0.34 s |
| official vLLM · FlashInfer fp8 | 1 | 2 | 2.54 | 0.86 | 0.68 |  |  |  |  |  | 24.3 | 23.4 | 0.40 s |
| official vLLM · FlashInfer fp8 | 1 | 3 | 3.08 | 0.87 | 0.69 | 0.53 |  |  |  |  | 24.6 | 23.6 | 0.44 s |
| official vLLM · FlashInfer fp8 | 1 | 4 **★** | 3.46 | 0.85 | 0.67 | 0.53 | 0.41 |  |  |  | 25.8 | 24.5 | 0.47 s |
| official vLLM · FlashInfer fp8 | 1 | 5 | 3.68 | 0.83 | 0.66 | 0.50 | 0.39 | 0.30 |  |  | 24.4 | 23.2 | 0.50 s |
| official vLLM · FlashInfer fp8 | 1 | 6 | 3.90 | 0.82 | 0.65 | 0.51 | 0.39 | 0.31 | 0.22 |  | 23.6 | 22.1 | 0.53 s |
| official vLLM · FlashInfer fp8 | 1 | 7 | 3.95 | 0.83 | 0.63 | 0.49 | 0.36 | 0.27 | 0.21 | 0.16 | 21.9 | 20.4 | 0.56 s |
| official vLLM · FlashInfer fp8 | 4 | 1 | 1.87 | 0.87 |  |  |  |  |  |  | 20.1 | 65.2 | 0.39 s |
| official vLLM · FlashInfer fp8 | 4 | 2 | 2.54 | 0.86 | 0.68 |  |  |  |  |  | 23.3 | 75.7 | 0.41 s |
| official vLLM · FlashInfer fp8 | 4 | 3 | 3.11 | 0.88 | 0.70 | 0.53 |  |  |  |  | 24.6 | 80.5 | 0.43 s |
| official vLLM · FlashInfer fp8 | 4 | 4 **★** | 3.50 | 0.86 | 0.68 | 0.55 | 0.42 |  |  |  | 24.8 | 84.4 | 0.47 s |
| official vLLM · FlashInfer fp8 | 4 | 5 | 3.83 | 0.86 | 0.70 | 0.54 | 0.42 | 0.32 |  |  | 23.7 | 78.3 | 0.51 s |
| official vLLM · FlashInfer fp8 | 4 | 6 | 3.94 | 0.85 | 0.66 | 0.52 | 0.40 | 0.30 | 0.22 |  | 22.2 | 70.3 | 0.56 s |
| official vLLM · FlashInfer fp8 | 4 | 7 | 3.97 | 0.84 | 0.62 | 0.49 | 0.36 | 0.28 | 0.21 | 0.16 | 20.7 | 67.0 | 0.62 s |
| Mjolnir · fp8 KV | 1 | 1 | 1.88 | 0.88 |  |  |  |  |  |  | 21.7 | 20.8 | 0.54 s |
| Mjolnir · fp8 KV | 1 | 2 | 2.60 | 0.88 | 0.72 |  |  |  |  |  | 25.5 | 24.2 | 0.54 s |
| Mjolnir · fp8 KV | 1 | 3 **★** | 3.02 | 0.83 | 0.67 | 0.51 |  |  |  |  | 25.9 | 24.3 | 0.59 s |
| Mjolnir · fp8 KV | 1 | 4 | 3.31 | 0.83 | 0.63 | 0.49 | 0.36 |  |  |  | 24.9 | 23.2 | 0.63 s |
| Mjolnir · fp8 KV | 1 | 5 | 3.72 | 0.85 | 0.68 | 0.52 | 0.39 | 0.29 |  |  | 25.2 | 23.5 | 0.65 s |
| Mjolnir · fp8 KV | 1 | 6 | 3.91 | 0.85 | 0.67 | 0.52 | 0.37 | 0.29 | 0.22 |  | 24.1 | 22.3 | 0.69 s |
| Mjolnir · fp8 KV | 1 | 7 | 3.95 | 0.82 | 0.65 | 0.49 | 0.36 | 0.26 | 0.20 | 0.16 | 22.2 | 20.6 | 0.75 s |
| Mjolnir · fp8 KV | 4 | 1 | 1.88 | 0.88 |  |  |  |  |  |  | 19.9 | 60.2 | 1.44 s |
| Mjolnir · fp8 KV | 4 | 2 | 2.57 | 0.87 | 0.70 |  |  |  |  |  | 23.8 | 72.9 | 1.27 s |
| Mjolnir · fp8 KV | 4 | 3 | 3.05 | 0.84 | 0.68 | 0.53 |  |  |  |  | 23.3 | 71.6 | 1.54 s |
| Mjolnir · fp8 KV | 4 | 4 **★** | 3.41 | 0.84 | 0.67 | 0.52 | 0.38 |  |  |  | 24.3 | 76.5 | 1.36 s |
| Mjolnir · fp8 KV | 4 | 5 | 3.64 | 0.84 | 0.65 | 0.51 | 0.36 | 0.28 |  |  | 23.1 | 72.8 | 1.44 s |
| Mjolnir · fp8 KV | 4 | 6 | 4.09 | 0.85 | 0.68 | 0.56 | 0.42 | 0.34 | 0.25 |  | 23.4 | 72.1 | 1.52 s |
| Mjolnir · fp8 KV | 4 | 7 | 4.00 | 0.82 | 0.64 | 0.50 | 0.37 | 0.28 | 0.21 | 0.17 | 21.0 | 66.2 | 1.54 s |
| Mjolnir · fp6 KV | 1 | 1 | 1.88 | 0.88 |  |  |  |  |  |  | 20.7 | 19.9 | 0.54 s |
| Mjolnir · fp6 KV | 1 | 2 | 2.53 | 0.85 | 0.68 |  |  |  |  |  | 25.1 | 23.8 | 0.56 s |
| Mjolnir · fp6 KV | 1 | 3 **★** | 3.04 | 0.85 | 0.67 | 0.52 |  |  |  |  | 25.7 | 24.1 | 0.59 s |
| Mjolnir · fp6 KV | 1 | 4 | 3.40 | 0.85 | 0.67 | 0.50 | 0.39 |  |  |  | 25.0 | 23.4 | 0.62 s |
| Mjolnir · fp6 KV | 1 | 5 | 3.66 | 0.82 | 0.66 | 0.51 | 0.39 | 0.28 |  |  | 24.7 | 22.9 | 0.69 s |
| Mjolnir · fp6 KV | 1 | 6 | 3.92 | 0.84 | 0.66 | 0.52 | 0.38 | 0.30 | 0.22 |  | 24.1 | 22.3 | 0.72 s |
| Mjolnir · fp6 KV | 1 | 7 | 4.14 | 0.84 | 0.67 | 0.52 | 0.41 | 0.30 | 0.23 | 0.18 | 23.4 | 21.5 | 0.74 s |
| Mjolnir · fp6 KV | 4 | 1 | 1.88 | 0.88 |  |  |  |  |  |  | 20.9 | 62.8 | 1.21 s |
| Mjolnir · fp6 KV | 4 | 2 | 2.55 | 0.86 | 0.69 |  |  |  |  |  | 22.8 | 69.3 | 1.50 s |
| Mjolnir · fp6 KV | 4 | 3 | 3.04 | 0.85 | 0.68 | 0.51 |  |  |  |  | 23.2 | 70.6 | 1.53 s |
| Mjolnir · fp6 KV | 4 | 4 **★** | 3.47 | 0.85 | 0.68 | 0.54 | 0.40 |  |  |  | 24.5 | 77.9 | 1.33 s |
| Mjolnir · fp6 KV | 4 | 5 | 3.68 | 0.84 | 0.65 | 0.50 | 0.40 | 0.30 |  |  | 23.6 | 73.3 | 1.44 s |
| Mjolnir · fp6 KV | 4 | 6 | 3.99 | 0.84 | 0.67 | 0.53 | 0.40 | 0.31 | 0.23 |  | 22.0 | 68.0 | 1.48 s |
| Mjolnir · fp6 KV | 4 | 7 | 4.10 | 0.85 | 0.67 | 0.51 | 0.39 | 0.29 | 0.22 | 0.17 | 21.4 | 68.6 | 1.59 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 1 | 1 | 1.88 | 0.88 |  |  |  |  |  |  | 25.3 | 24.2 | 0.49 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 1 | 2 | 2.57 | 0.86 | 0.70 |  |  |  |  |  | 33.6 | 31.4 | 0.52 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 1 | 3 | 3.06 | 0.85 | 0.68 | 0.52 |  |  |  |  | 37.4 | 34.7 | 0.52 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 1 | 4 | 3.38 | 0.83 | 0.65 | 0.51 | 0.38 |  |  |  | 38.2 | 35.2 | 0.53 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 1 | 5 | 3.68 | 0.82 | 0.65 | 0.51 | 0.40 | 0.30 |  |  | 39.4 | 35.9 | 0.57 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 1 | 6 **★** | 3.94 | 0.85 | 0.67 | 0.53 | 0.38 | 0.29 | 0.21 |  | 39.8 | 36.0 | 0.57 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 1 | 7 | 4.08 | 0.84 | 0.68 | 0.51 | 0.39 | 0.29 | 0.22 | 0.16 | 39.4 | 35.3 | 0.60 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 4 | 1 | 1.87 | 0.87 |  |  |  |  |  |  | 23.3 | 69.5 | 1.38 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 4 | 2 | 2.58 | 0.87 | 0.71 |  |  |  |  |  | 30.2 | 88.5 | 1.61 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 4 | 3 | 3.03 | 0.85 | 0.67 | 0.51 |  |  |  |  | 31.8 | 95.0 | 1.43 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 4 | 4 | 3.35 | 0.81 | 0.65 | 0.51 | 0.38 |  |  |  | 33.4 | 102.0 | 1.22 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 4 | 5 **★** | 3.70 | 0.84 | 0.67 | 0.51 | 0.39 | 0.30 |  |  | 34.0 | 102.7 | 1.28 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 4 | 6 | 3.83 | 0.83 | 0.65 | 0.49 | 0.37 | 0.28 | 0.21 |  | 33.8 | 102.6 | 1.29 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 4 | 7 | 3.99 | 0.82 | 0.65 | 0.50 | 0.37 | 0.28 | 0.21 | 0.15 | 32.4 | 98.0 | 1.33 s |

<details>
<summary>How the % and × here are calculated</summary>

- **tokens/step** = 1 + accepted draft tokens ÷ verify steps, from vLLM's speculative-decoding counters read before and after the block; **accept @i** = the share of verify steps whose i-th draft was accepted.
- **decode t/s per request** = (generated tokens − 1) ÷ (time of the last token − time of the first token); **aggregate t/s** = all generated tokens ÷ the block's wall time.
- ★ = the draft count with the highest decode t/s for that config and concurrency.

</details><br />

<sub>Rendered from [`benchmarks/raw/verify/spec/20261014-182829.json`](../../../../benchmarks/raw/verify/spec/20261014-182829.json). ★ = the draft count with the highest decode t/s per request for that config and concurrency.</sub>

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

[← KV cache capacity](kv-cache-size.md) · [case study](README.md) · [Full profiles per KV format →](fp6prof.md)
