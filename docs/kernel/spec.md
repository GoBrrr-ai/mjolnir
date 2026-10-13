# Speculative decoding

> MTP acceptance per draft position and decode t/s for 1-4 draft tokens, per config — the best draft count · `mjolnir verify spec` · run `20261012-220110`

## The question

Every config serves Qwen3.8-27B with its MTP head proposing draft tokens that one forward pass of the full model then verifies. Two numbers decide the decode rate: how many tokens a verify step yields (1 + the drafts accepted) and how long the step takes. Each extra draft adds one forward of the draft head — on this model a bf16 MTP layer plus the bf16 lm_head, ~3 GB of reads — so more drafts are only worth it while the last one is still accepted often enough. And does the Mjolnir kernel (fp8 Q, fp6 KV) change what gets accepted?

## How it is measured

For every config and every draft count k the server is started with `--speculative-config` overridden to k (the rest of the config unchanged; the server restarts per point). After two warm-up requests, a fixed prompt set — code, maths, prose, translation, tables, one ~8K-token context — is sent at **temperature 0**, so every point decodes the same text, once per concurrency level. Around each block the stage reads vLLM's speculative-decoding counters:

- **tokens per step** = 1 + accepted ÷ drafts (what one verify forward yields);
- **acceptance per position** = how often draft #1, #2, … was accepted (each needs all earlier ones accepted);
- **decode t/s per request** from the stream: tokens after the first ÷ time from the first to the last token; **aggregate t/s** = all generated tokens ÷ the block's wall time.

## Result

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../assets/benchmarks/spec-drafts-dark.png">
  <img src="../../assets/benchmarks/spec-drafts.png" alt="Decode rate vs draft tokens" width="100%">
</picture>

- official vLLM · FlashInfer fp8, 1 request: best with **4 drafts** — 25.8 t/s per request, 3.46 tokens per step, +0.3% vs 3 drafts.
- official vLLM · FlashInfer fp8, 4 requests: best with **3 drafts** — 24.6 t/s per request, 3.11 tokens per step.
- Mjolnir · fp8 KV, 1 request: best with **3 drafts** — 26.0 t/s per request, 3.04 tokens per step.
- Mjolnir · fp8 KV, 4 requests: best with **3 drafts** — 24.6 t/s per request, 3.06 tokens per step.
- Mjolnir · fp6 KV, 1 request: best with **3 drafts** — 26.5 t/s per request, 3.12 tokens per step.
- Mjolnir · fp6 KV, 4 requests: best with **4 drafts** — 24.7 t/s per request, 3.46 tokens per step, +4.4% vs 3 drafts.
- Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab, 1 request: best with **4 drafts** — 38.9 t/s per request, 3.43 tokens per step, +3.7% vs 3 drafts.
- Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab, 4 requests: best with **4 drafts** — 33.7 t/s per request, 3.38 tokens per step, +3.3% vs 3 drafts.
- Acceptance, 3 drafts, 1 request: Mjolnir · fp8 KV 3.04 vs official 3.08 tokens/step (-1.3%) — same prompts, temperature 0.
- Acceptance, 3 drafts, 1 request: Mjolnir · fp6 KV 3.12 vs official 3.08 tokens/step (+1.3%) — same prompts, temperature 0.
- Acceptance, 3 drafts, 1 request: Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab 3.10 vs official 3.08 tokens/step (+0.5%) — same prompts, temperature 0.

| config | c | drafts | tokens/step | accept @1 | accept @2 | accept @3 | accept @4 | decode t/s/req | aggregate t/s | TTFT |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| official vLLM · FlashInfer fp8 | 1 | 1 | 1.87 | 0.87 |  |  |  | 20.3 | 19.9 | 0.34 s |
| official vLLM · FlashInfer fp8 | 1 | 2 | 2.54 | 0.86 | 0.68 |  |  | 24.5 | 23.6 | 0.40 s |
| official vLLM · FlashInfer fp8 | 1 | 3 | 3.08 | 0.87 | 0.69 | 0.53 |  | 25.7 | 24.6 | 0.43 s |
| official vLLM · FlashInfer fp8 | 1 | 4 **★** | 3.46 | 0.85 | 0.67 | 0.53 | 0.41 | 25.8 | 24.4 | 0.46 s |
| official vLLM · FlashInfer fp8 | 4 | 1 | 1.87 | 0.87 |  |  |  | 20.2 | 65.2 | 0.39 s |
| official vLLM · FlashInfer fp8 | 4 | 2 | 2.56 | 0.87 | 0.70 |  |  | 23.4 | 75.5 | 0.41 s |
| official vLLM · FlashInfer fp8 | 4 | 3 **★** | 3.11 | 0.88 | 0.70 | 0.53 |  | 24.6 | 80.9 | 0.44 s |
| official vLLM · FlashInfer fp8 | 4 | 4 | 3.48 | 0.86 | 0.68 | 0.53 | 0.41 | 24.5 | 82.3 | 0.48 s |
| Mjolnir · fp8 KV | 1 | 1 | 1.88 | 0.88 |  |  |  | 21.7 | 20.8 | 0.54 s |
| Mjolnir · fp8 KV | 1 | 2 | 2.61 | 0.88 | 0.73 |  |  | 25.2 | 23.9 | 0.56 s |
| Mjolnir · fp8 KV | 1 | 3 **★** | 3.04 | 0.84 | 0.68 | 0.52 |  | 26.0 | 24.3 | 0.59 s |
| Mjolnir · fp8 KV | 1 | 4 | 3.31 | 0.83 | 0.63 | 0.49 | 0.36 | 24.3 | 22.7 | 0.63 s |
| Mjolnir · fp8 KV | 4 | 1 | 1.87 | 0.87 |  |  |  | 19.9 | 60.1 | 1.69 s |
| Mjolnir · fp8 KV | 4 | 2 | 2.58 | 0.87 | 0.70 |  |  | 22.8 | 68.8 | 1.51 s |
| Mjolnir · fp8 KV | 4 | 3 **★** | 3.06 | 0.85 | 0.68 | 0.52 |  | 24.6 | 78.0 | 1.30 s |
| Mjolnir · fp8 KV | 4 | 4 | 3.42 | 0.85 | 0.67 | 0.52 | 0.39 | 24.3 | 76.8 | 1.37 s |
| Mjolnir · fp6 KV | 1 | 1 | 1.88 | 0.88 |  |  |  | 21.7 | 20.8 | 0.53 s |
| Mjolnir · fp6 KV | 1 | 2 | 2.59 | 0.86 | 0.72 |  |  | 25.7 | 24.3 | 0.55 s |
| Mjolnir · fp6 KV | 1 | 3 **★** | 3.12 | 0.87 | 0.70 | 0.55 |  | 26.5 | 24.9 | 0.59 s |
| Mjolnir · fp6 KV | 1 | 4 | 3.40 | 0.85 | 0.66 | 0.52 | 0.38 | 25.3 | 23.6 | 0.66 s |
| Mjolnir · fp6 KV | 4 | 1 | 1.87 | 0.87 |  |  |  | 20.4 | 62.5 | 1.26 s |
| Mjolnir · fp6 KV | 4 | 2 | 2.56 | 0.85 | 0.70 |  |  | 23.8 | 70.0 | 1.93 s |
| Mjolnir · fp6 KV | 4 | 3 | 3.09 | 0.86 | 0.69 | 0.54 |  | 23.7 | 74.0 | 1.54 s |
| Mjolnir · fp6 KV | 4 | 4 **★** | 3.46 | 0.84 | 0.67 | 0.54 | 0.41 | 24.7 | 77.2 | 1.36 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 1 | 1 | 1.87 | 0.87 |  |  |  | 26.2 | 24.9 | 0.52 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 1 | 2 | 2.56 | 0.86 | 0.70 |  |  | 33.4 | 31.3 | 0.53 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 1 | 3 | 3.10 | 0.87 | 0.69 | 0.54 |  | 37.5 | 34.8 | 0.52 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 1 | 4 **★** | 3.43 | 0.84 | 0.67 | 0.52 | 0.40 | 38.9 | 35.7 | 0.54 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 4 | 1 | 1.86 | 0.86 |  |  |  | 23.0 | 69.1 | 1.40 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 4 | 2 | 2.55 | 0.86 | 0.69 |  |  | 31.1 | 94.3 | 1.17 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 4 | 3 | 3.07 | 0.86 | 0.68 | 0.52 |  | 32.6 | 95.0 | 1.66 s |
| Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab | 4 | 4 **★** | 3.38 | 0.85 | 0.66 | 0.50 | 0.38 | 33.7 | 102.8 | 1.22 s |

<details>
<summary>How the % and × here are calculated</summary>

- **tokens/step** = 1 + accepted draft tokens ÷ verify steps, from vLLM's speculative-decoding counters read before and after the block; **accept @i** = the share of verify steps whose i-th draft was accepted.
- **decode t/s per request** = (generated tokens − 1) ÷ (time of the last token − time of the first token); **aggregate t/s** = all generated tokens ÷ the block's wall time.
- ★ = the draft count with the highest decode t/s for that config and concurrency.

</details><br />

<sub>Rendered from [`benchmarks/raw/verify/spec/20261012-220110.json`](../../benchmarks/raw/verify/spec/20261012-220110.json). ★ = the draft count with the highest decode t/s per request for that config and concurrency.</sub>

## How to read it

Tokens per step rises with k but flattens as later positions are accepted less; the step cost rises linearly with k. The best k is where the next draft's acceptance no longer pays for its forward. If the Mjolnir configs accept fewer tokens per step than the official one on the same prompts, the kernel's numerics (fp8 Q, fp6 KV) shift the target model's distribution — that slows decode no matter how fast the kernel is. Cheaper drafts (an fp8 lm_head, an fp8 MTP layer) move the optimum to more drafts: re-run after such changes.

## Reproduce

```bash
mjolnir serve down                                         # restarts per point — the GPU must be yours
mjolnir verify spec                                        # 4 configs × k = 1..4, 1 and 4 requests
mjolnir verify spec --legs mjolnir:NVFP4_MJOLNIR_E2M3 --drafts "2 3"
```

## Ideas to try

- an fp8 lm_head and MTP layer (cheaper drafts), then re-run;
- a reduced draft vocabulary for the MTP head;
- per-request adaptive draft length (stop drafting when the draft's confidence drops).

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../.github/CONTRIBUTING.md).

---

[← KV cache capacity](kv-cache-size.md) · [case study](README.md) · [Full profiles per KV format →](fp6prof.md)
