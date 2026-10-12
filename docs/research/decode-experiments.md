# Decode-path experiments — ranked by expected decode impact

Status: plan, 2026-10-12. Inputs: `mjolnir verify spec` run `20261011-220331`,
`verify sweep --batch 4 --q-len 4` run `20261011-211511`, `verify kv-cache-size`
run `20261011-213435`, e2e records of 2026-10-12 (`benchmarks/history.jsonl`).

## Where a decode step goes (Qwen3.8-27B NVFP4, MTP, measured)

`verify spec` serves each config at 1–4 draft tokens on the same prompts at
temperature 0. Step time = tokens per step ÷ decode t/s; a straight-line fit
over k gives the verify forward and the cost of each draft forward:

| config (1 request) | verify forward | per draft | step at k=3 | tokens/step at k=3 |
|---|--:|--:|--:|--:|
| official vLLM, FlashInfer fp8 | 82.6 ms | 12.7 ms | 120.5 ms | 3.08 |
| Mjolnir fp8 KV | 71.4 ms | 15.4 ms | 115.8 ms | 3.04 |
| Mjolnir fp6 KV | 72.4 ms | 15.0 ms | 117.3 ms | 3.12 |

- The Mjolnir image's **verify forward is ~10 ms (13%) faster**; each **draft
  forward is ~2.3 ms (18%) slower**. With 3 drafts the two nearly cancel at
  4 requests (77.9 + 3×16.4 vs 82.7 + 3×14.6 ms) — the c=4 e2e deficit.
- **Acceptance is the same** on both images (k=3: 3.08 / 3.04 / 3.12 tokens per
  step; e2e: 2.976 vs 2.972) — fp8 Q and fp6 KV do not cost acceptance.
- **k=3 is the right draft count today** (k=4 adds ~0.3 tokens/step for ~13%
  more step time).
- From the checkpoint config: the bf16 `lm_head` (248,320 × 5,120, 2.54 GB)
  is read by the verify and by every draft (4× per step ≈ 10 GB of ~26 GB);
  the bf16 MTP layer (~0.7 GB) 3×. Attention at ≤32K is 1–4% of the step.

## Diagnosing the draft step (experiment 1)

Facts so far (2026-10-12):

- Both images **capture** the drafter: the Mjolnir server log has
  `Capturing prefill CUDA graphs (FULL)` and `Capturing decode CUDA graphs
  (FULL)` for the speculator, and no "fused multi-step draft decode is not
  supported" line — Mjolnir's draft decode runs as **one fused graph** for all
  draft steps. The official image has no `thor-fused-draft-decode` patch, so
  its FlashInfer drafter cannot advance metadata in place: it rebuilds the
  metadata and replays **one graph per draft step**. Fewer launches on our side,
  yet each draft costs ~2.3 ms more — so the extra time is GPU work (or a stall)
  inside the draft step, not launch overhead.
- The two images differ in more than attention (the patch stack), so the first
  split is image vs backend.

Tools (each restarts the server per leg — the GPU must be yours):

```bash
# 1) image vs backend: FlashInfer on the Mjolnir image is the middle leg
mjolnir verify spec --legs official:NVFP4,mjolnir:NVFP4,mjolnir:NVFP4_MJOLNIR_E2M3 --drafts "1 3" --concurrency "1 4"

# 2) where the GPU time goes: torch-profiler trace per leg (graph replays and
#    their kernels, eager kernels per step, GPU idle, top kernels)
mjolnir verify profile --legs official:NVFP4,mjolnir:NVFP4_MJOLNIR_E2M3 --drafts 3
mjolnir verify profile --legs official:NVFP4,mjolnir:NVFP4_MJOLNIR_E2M3 --drafts 1   # no draft decode graph: the diff is the draft step

# 3) isolate the drafter's attention / the fused loop on the Mjolnir image
MJOLNIR_ATTN_DRAFT_T5=0     mjolnir verify spec --legs mjolnir:NVFP4_MJOLNIR_E2M3 --drafts "1 3"
MJOLNIR_ATTN_FUSED_DRAFT=0  mjolnir verify spec --legs mjolnir:NVFP4_MJOLNIR_E2M3 --drafts "1 3"
```

Reading `verify profile`: the graph with the most kernels is the target's
verify step; with k=3 the drafter adds a prefill graph and a fused decode graph
(2 draft steps). Compare the decode graph's GPU time per replay ÷ 2 between
legs; its top kernels name the extra work. "eager kernels/step" and "GPU idle"
show host-side cost.

Also found in `verify spec` (per-request TTFT): one-off **~2.7–3 s stalls** on
the Mjolnir image — the first 4 requests at 4 in flight, one request at 1 in
flight — and none on the official image. They look like FlashAttention 4's
CuTe-DSL compile on the first use of a prefill shape after each server start
(the `cannot be converted to a JitArgument` warning is printed per compile):

```bash
grep -c "cannot be converted to a JitArgument" benchmarks/logs/verify-<run>/spec/*.log
```

Fix candidate: persist FA4's compiled kernels (the CuTe DSL export/load the
Mjolnir kernel cache uses) or warm the prefill shapes at startup.

## lm_head experiments (2, 3, 4): how to run them

Opt-in configs (YAML identical to `NVFP4_MJOLNIR_E2M3`; the switch is the
`<config>.env` sidecar the CLI passes to the container):

| config | env | what |
|---|---|---|
| `NVFP4_MJOLNIR_E2M3_LMH8` | `MJOLNIR_LM_HEAD=fp8` | lm_head e4m3 per row + Triton fp8-weight GEMM (verify + drafts) |
| `NVFP4_MJOLNIR_E2M3_VT32K` | `MJOLNIR_DRAFT_VOCAB={configs}/draft_vocab.json`, `…_SIZE=32768` | drafter logits over the 32K most frequent target tokens |
| `NVFP4_MJOLNIR_E2M3_LMH8_VT32K` | both | both |

```bash
mjolnir image build                                        # the image needs mjolnir.lmhead
mjolnir verify vllm --smoke-modes "graph lmh8"            # kernel tests (fp8 head vs reference, faster than bf16,
                                                           # vocab trim) + logprob smoke with the fp8 head
mjolnir calibrate --config NVFP4_MJOLNIR_E2M3 --draft-vocab   # -> configs/<model>/draft_vocab.json + coverage per K
mjolnir verify spec --legs mjolnir:NVFP4_MJOLNIR_E2M3,mjolnir:NVFP4_MJOLNIR_E2M3_LMH8,mjolnir:NVFP4_MJOLNIR_E2M3_VT32K,mjolnir:NVFP4_MJOLNIR_E2M3_LMH8_VT32K --drafts "3 4"
mjolnir bench ab --backends NVFP4_MJOLNIR_E2M3,NVFP4_MJOLNIR_E2M3_LMH8_VT32K --repeat 2
```

Pass criteria: smoke `lmh8` passes; `verify spec` acceptance (tokens/step) within
~2% of the baseline for LMH8 and within ~5% for VT32K, decode t/s up; the
server log shows `Mjolnir lm_head fp8: … self-test rel. error` and `Mjolnir
draft vocab: drafter logits over 32768 of 248320 tokens`.

## The list

| # | experiment | expected decode gain | effort | how to A/B |
|---|---|---|---|---|
| 1 | **Find the draft-step overhead** (+2.3 ms per draft vs official): is the drafter's attention CUDA-graphed on MJOLNIR, does the first draft step (variable accepted length) fall back to FA4 + fp6 page expansion, Python plan cost per draft | +5–6% at k=3, closes the c=4 gap | low (diagnosis) | `mjolnir serve logs \| grep -iE "cudagraph\|Capturing\|draft"`; nsys on one decode window; then `mjolnir verify spec --legs official:NVFP4,mjolnir:NVFP4_MJOLNIR_E2M3 --drafts "1 3" --concurrency "1 4"` (the per-draft slope must drop to ≤12.7 ms) |
| 2 | **fp8 lm_head** (per-row scales, quantized at load; shared by verify and drafts) — Thor patch, opt-in config | ~+20% (−5 GB/step) | medium | after the patch, config `NVFP4_MJOLNIR_E2M3_LMH8`: `mjolnir verify vllm` (smoke) → `mjolnir verify spec --legs mjolnir:NVFP4_MJOLNIR_E2M3,mjolnir:NVFP4_MJOLNIR_E2M3_LMH8 --drafts "3 4"` (acceptance must hold) → `mjolnir bench ab --backends NVFP4_MJOLNIR_E2M3,NVFP4_MJOLNIR_E2M3_LMH8 --repeat 2` |
| 3 | **VocabTrim for the MTP draft head**: draft logits over the ~32K most frequent target tokens (counted on the target's own generations — the `mjolnir calibrate` prompt set), mapped back for verification | ~+15–20% (−6.6 GB/step bf16; −3.3 GB on top of #2); paper: 2–5% acceptance drop | medium-high | config `…_VT32K`: `mjolnir verify spec --legs mjolnir:NVFP4_MJOLNIR_E2M3,mjolnir:NVFP4_MJOLNIR_E2M3_VT32K --drafts "2 3 4"` (cheaper drafts may move the best k to 4) → `mjolnir bench ab --backends …` |
| 4 | **fp8 MTP layer** (the drafter's transformer layer is bf16 in the checkpoint) | ~+3–4% (−1 GB/step) | low on top of #2 | same as #2, config `…_MTP8` |
| 5 | **Re-tune / adapt the draft length**: after #2–#4 drafts get cheap, so re-run the sweep; then adaptive length (stop drafting when the draft's top-1 probability drops — SpecDec++ / PACER style) | +3–5% | low (re-tune) / medium (adaptive) | `mjolnir verify spec --drafts "2 3 4 5"`; adaptive behind a flag, same command |
| 6 | **Long-context decode** — measure where the kernel matters: at 128K, attention is ~25% of a 1-request step and the kernel is 1.3–1.5× FlashInfer | +5–10% at 64K–128K (estimate) | none (bench) | `mjolnir bench e2e --depths 65536,131072 --concurrency 1 --runs 3` (cells merge into the table) |
| 7 | **fp6 bandwidth gap**: fp6 streams at ~200 GB/s vs fp8's ~230 (64-byte L2 requests, `fp6prof`) — closing it is +15% kernel at long L | +3–4% e2e at 128K | medium-high (kernel) | `mjolnir verify fp6prof`, `mjolnir verify sweep --batch 4 --q-len 4` |
| 8 | **NVFP4 KV with a Hadamard rotation** (BitDecoding / KVLinC): rotate K's head dim (fold the rotation into Q like the per-channel scales), 4-bit KV | ≤1% at ≤32K; ~+5% at 128K; +33% KV capacity over fp6 | high + accuracy risk | `mjolnir verify quant`, `mjolnir verify kvfmt`; later a `NVFP4_MJOLNIR_FP4KV` config through `verify vllm` (smoke) and `verify spec` |
| 9 | **Short-context route** (PersistentKV-style): below ~1–2K tokens skip split-KV + merge (Mjolnir is 0.5× FlashInfer at L=128: +16 µs × 16 layers) | <0.5% | low | `mjolnir verify sweep --ls "128 1024 2048 4096" --batch 4 --q-len 4` |

Not decode, but found on the way: **TTFT on short prompts** is ~0.2 s slower
on the Mjolnir image in `verify spec` (c=1 mean 534–653 ms vs 341–464 ms;
c=4 1.2–1.5 s vs 0.33 s), while 2K+ prompts are much faster (e2e). `verify
spec` now stores per-request TTFT (`ttft_list_ms`, `ttft_p50_ms`): a few
outliers point at a one-off compile per prompt shape (FA4), a uniform shift
at the fallback's per-step overhead.

## References

- [VocabTrim (arXiv 2506.22694)](https://arxiv.org/abs/2506.22694) — trimmed drafter LM head, +16–25% memory-bound speedup.
- [SpecDec++ (arXiv 2405.19715)](https://arxiv.org/abs/2405.19715), [PACER (arXiv 2602.01274)](https://arxiv.org/abs/2602.01274) — adaptive draft length.
- [BitDecoding (arXiv 2503.18773)](https://arxiv.org/abs/2503.18773) — low-bit / NVFP4 KV decode on tensor cores, Blackwell.
- [KVLinC (arXiv 2510.05373)](https://arxiv.org/abs/2510.05373) — Hadamard rotation + linear correction for 2–4-bit KV.
- [PersistentKV (arXiv 2606.26666)](https://arxiv.org/abs/2606.26666) — per-step route selection on a small-SM GPU.
