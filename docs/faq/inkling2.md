# Inkling 2 Mini on Thor (sm_110)

**Story:** "Thinking Machines released Inkling 2 Mini. What runs on the
Thor, which Mjolnir stack features apply, and what do we serve?"

## The model

`thinkingmachines/Inkling-2-Mini` — 26.4B total / ~3.5B active **looped
MoE** (`Inkling2LoopForConditionalGeneration`, `inkling2_loop`): 24 physical
layers split prelude (0–5) / loop core (6–17, **three passes with shared
weights** = 48 layer visits per token) / coda (18–23); 128 routed experts
top-8 (NVFP4 layers 0–19, MXFP8 20–23); attention = 20 sliding-window + 4
full-attention layers, 24Q/8KV hd128, relative-bias + per-token sinks +
gated value norm; four short convolutions; block attention residuals with
rank-64 loop controllers; a 23-layer ViT; 1M context.

**The spec decider ships in the checkpoint**: `mtp.safetensors` (0.6 GB) is
a 3-layer DSpark drafter (inkling-shaped layers + a recurrent rank-256
Markov head, draft logits from a BF16 GEMM) served through vLLM's
`method: dspark` at the trained block, K=7, `probabilistic`. There is **no
MTP head** — the repo's MTP/VT64K machinery does not apply as-is.

## The base: the 0.35.0rc0.dev nightly

Upstream merged Inkling 2 Mini on 2026-11-21 (#69028); v0.35.0 is
unreleased, so the branch pins the **2026-11-22 nightly**
(`nightly-647a7686…`, arm64 `135d65d5…`) as
`narandill/mjolnir-vllm:v0.35.0rc0.dev`. The patch stack re-anchored clean
(19 vLLM + 2 FlashInfer fixes verified); three patches retired as
base-covered — see `docker/vllm-thor/PATCHES.md` ("Retired at the
v0.35.0rc0.dev nightly bump"): #64739 (shards in an index subdir), the
FA4-hd256 FP8 port (the nightly's fwd grew native `DescaleTensors`), and
the hd256 1CTA decode carve-out (the heuristics moved to `config.py` and
now gate on `device_capacity in (10, 11)` themselves).

**The one Thor gap**: the nightly wheel's prebuilt inkling2 kernels
(`csrc/inkling2`: `conv_mix`/`allreduce_conv_mix`/`seq_parallel`) carry no
sm_110 cubins — CMake's `INKLING2_KERNEL_ARCHS` intersects
`9.0a;10.0f;10.3f` only (upstream #69167, one line, open). The image
carries `thor-inkling2-sm110-conv-mix.patch`: the per-layer conv/mix takes
the stock SM12x torch-compose fallback on Thor (both halves Triton →
sm_110 JIT); `MJOLNIR_INKLING2_CONV_MIX_NATIVE=1` flips to the native path
once a base with sm_110 cubins is pinned. Everything else in the model —
the fused rel-decode attention, the sconv/prologue stack, gate/sink — is
Triton/gluon or FlashInfer-JIT and compiles for sm_110 at runtime.
TP>1 also waits for #69167 (the all-reduce kernels): TP1 configs only.

## Which Mjolnir stack features apply

| Feature | Verdict |
|---|---|
| `--attention-backend MJOLNIR` (the decode kernel) | **Does not plug.** The model owns its attention: `Inkling2AttentionBackend` (a `FlashAttentionBackend` subclass) with a fused custom decode kernel (`rel_decode_attention`: SDPA + rel-bias + gate/sink in one) and FA4-with-sheared-bias / the sm90 kernel for prefill. The global backend switch is moot for these layers. |
| fp6 KV (`fp6_e2m3`) | **Phase 2.** The cache is written by the model's own `attention_prologue` and read by its own kernels; the dtype gate allows `auto`/`bfloat16` only off SM12x (`fp8` is SM120+, and fp16 conflicts with spec decode). fp6 needs inkling-side kernel/loader work — the Mjolnir pluggable dtype registers, but nothing consumes it on this path yet. |
| fp8 KV | Same story, SM120-gated upstream. The unit-scale e4m3 cache path exists in the kernels; untesting on sm_110 is a patch, not a config. |
| fp8 lm_head (`MJOLNIR_LM_HEAD=fp8`) | **Plausible, verify at first serve** — config `NVFP4_LMH8`. The head is a stock `ParallelLMHead` behind the model's `_apply_head` (low-M fp32 path on SM12x only) and the `final_logit_softcapping 30.0` is a `LogitsProcessor` post-op, so the wrapper compose; the smoke is a NaN check + `bench accuracy`. |
| MTP 5/6 | **N/A as MTP; the K sweep is the honest version.** The drafter is DSpark-shaped with a recurrent Markov head — any K runs; the trained block is 7. `verify spec` needs inkling legs before it can sweep. |
| VT64K (`MJOLNIR_DRAFT_VOCAB`) | **Phase 2.** The hook targets the qwen4 MTP draft head; the inkling drafter's `compute_base_logits` BF16 GEMM is a separate surface (a small patch, real upside — it runs once per draft step over the full 200052-token vocab). |
| MoE W4A4 (`MJOLNIR_NVFP4_MOE_W4A4`) | **No.** The experts run the model's own per-arch ops (TRT-LLM-gen / Humming / Triton), not `ModelOptNvFp4FusedMoE`. |
| The image/patch stack itself | **The whole value**: the nightly base, the sm_110 conv/mix bridge, the FlashInfer-JIT/Triton env pins, and the re-anchored PR fixes. |

## The community quant cut

| Checkpoint | What it is | vLLM/Thor verdict |
|---|---|---|
| `thinkingmachines/Inkling-2-Mini` | the native NVFP4/MXFP8 mix | **the main config** (`NVFP4`) — the reference every other arm validates against |
| `capyctl/Inkling-2-Mini-NVFP4-FP8` | the 144 attention projections (q/k/v/r/gate/o ×24) → E4M3 per-channel; experts/head/drafter byte-identical; −0.64 GB | the **right idea** — the looped arc re-reads those weights 3×, and sm_110 decode is bandwidth-bound (~39.6 tok/s ≈ 267 GB/s on the PR's numbers) — but the artifact is SGLang-shaped (repacked qkvr, a split sink projection, a loader patch). Reproduce the transform vLLM-shape (ModelOpt mixed groups via `hf_quant_config`, which is already reader-shaped) rather than adopting the file. Their GB10 numbers: geomean 1.043 vs the community base, −2.6% single-stream. Phase 2. |
| `aday777/Inkling-2-Mini-Abliterated-NVFP4` | the same weights, two edited BF16 tensors (embed/unembed refusal-direction shrink t=0.5) | not a quant — a behavior variant, zero serving risk. Optional arm, not the main. |
| `Enntity/Inkling-2-Mini-Spark-NVFP4` | a 459-tensor **overlay**: GPTQ NVFP4 for the attention linears + lm_head + the coda experts (all-NVFP4); KL 0.0081 / top-1 97.0% vs RTN 0.0170 — properly calibrated | the strongest quality data, but all-NVFP4 dense GEMMs on sm_110 are uncharted (the house 27B keeps attention BF16 deliberately) and the overlay must be materialized onto the base checkpoint first. Phase 2. |

## Serving configs

- `configs/thinkingmachines/Inkling-2-Mini/NVFP4/` — the reference
  (bf16 KV, native DSpark K=7 probabilistic, `inkling` parsers).
- `configs/thinkingmachines/Inkling-2-Mini/NVFP4_LMH8/` — the fp8-head
  experiment (`MJOLNIR_LM_HEAD=fp8` via the user-managed `.env` sidecar).

## Open after this branch

1. `#69167` upstream → re-pin a nightly with sm_110 inkling cubins, flip
   `MJOLNIR_INKLING2_CONV_MIX_NATIVE=1` (or retire the patch), unblock TP>1.
2. `verify spec` / `bench e2e` legs for the model (the stages are
   model-table-driven; inkling entries pending) — then the K sweep
   (5/6/7) and the DSpark acceptance baseline on Thor.
3. The capyctl-style fp8-attention reproduction (vLLM-shape) — the biggest
   bandwidth lever the looped arc offers.
4. The VT64K-for-DSpark hook and the fp6-KV inkling path.
5. The device canaries (`mjolnir image gates`) on the day the GPU window
   opens; the build-time gates are green.
