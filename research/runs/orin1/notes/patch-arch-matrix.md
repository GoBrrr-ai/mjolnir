# vLLM image patch × arch matrix (2026-10-25)

Every patch in `docker/vllm-thor/patches/` classified per arch. "Applies" =
the patch's mechanism is reachable and useful on that arch; "inert" =
ships but is a structural no-op there; "impossible" = the underlying
hardware/kernel class doesn't exist on that arch.

Verdicts from reading each patch's gates (grep for arch terms + the
PATCHES.md descriptions), not from memory. The three arch-agnostic
families: (a) upstream backports with no arch term, (b) probe-gated
sm_110 grants, (c) Thor-authored fixes with no arch term.

| Patch | Upstream | Thor sm_110 | Orin sm_87 | Spark sm_121 |
|---|---|---|---|---|
| 49652 draft-decode capture | #49652 | applies | **applies** (0 arch refs) | applies |
| 50885 FI full-cudagraph decode | #50885 | applies (native path) | **applies — native path is the ONLY path on sm_87** (no trtllm/XQA); version-probe passes on FI 0.7.0.post1 | applies via trtllm/XQA |
| 54165 mamba spec-kv restore | #54165 | applies | **applies** (Qwen3.8 is a GDN hybrid; scheduler-level) | applies |
| 55519 no-warn block-drop-off | #55519 | applies | **applies** | applies |
| fused-draft-decode FI-native update | Thor | applies (K≥3 payoff) | **applies** (native-FI builder gate includes sm_87; MTP K=3 needs it) | applies (native path) |
| fused-draft T5 plan refresh | Thor | applies (the c1-class fix) | **applies — the Orin runner's plan buffers are host-computed too** | n/a unless Mjolnir |
| dspark draft FI non-causal CG | Thor | applies | **applies** (gate is non-SM12x) | inert (SM12x excluded — XQA covers it) |
| draft-CG shape gate (c1) | Thor | applies (default N=1→eager) | **inert by design** (structural no-op off sm_110 → draft always CG) | inert |
| fa4-fp8kv probe | Thor | applies (probe-gated) | **impossible** (FA4 = SM100+; sm_87 has no FA4) | inert (arch assert (10,11) excludes family 12) |
| fa4-hd256-fp8 | Thor | applies (probe-gated) | **impossible** (FA4 hd256 = SM100+) | inert |
| fa4-hd256-1cta-decode | Thor | applies | **impossible** | inert |
| fp6-e2m3 KV dtype | Thor | applies | **inert** (dtype registration arch-agnostic; no Orin reader; availability matrix blocks any config) | inert |
| gdn-prefill grant | Thor | applies (probe-gated) | **inert** (allowlist (10,11); Orin keeps Triton/FLA) | inert |
| gdn-prefill FI allowlist | FlashInfer | applies | **inert** (same (10,11) term; the SM100 CuTe kernel won't compile for sm_87) | inert |
| MJOLNIR backend name | Thor | applies | **applies — required** (the Orin MJOLNIR serve parses through it; only its FA4-default hunk is sm_110-gated) | applies (fallback default differs) |
| MXFP4 MoE gate | #62133 | applies (MARLIN default) | **inert** (touches only the 110/120 selector branches; sm_87 keeps upstream defaults) | applies (MARLIN default) |
| CUTLASS FP4 MoE guard | #62275 | applies (fixes the fault) | **inert** (sm_87 was never in the allowlist; tcgen05-impossible anyway) | inert (120 stays allowed) |
| spec-decode split-topp | #58231 | applies | **applies** (arch-agnostic CUDA fix) | applies |

Count: **11 of 17 vLLM patches (+0 of 1 FI) are no-ops on Orin**; 10
apply (8 fully arch-agnostic + the MJOLNIR enum + the draft-CG pieces
that gate on non-SM12x). The image is safe to share; the sm_110-only
patches are all probe-gated or structurally inert off Thor.

## Can Orin have FULL cudagraph for decode and drafter? YES — mechanism-wise nothing blocks it

**Decode (spec-decode verify batches, and plain decode):**
- The 50885 gate `flashinfer_supports_uniform_multi_token_decode()` is a
  **FlashInfer version probe, not an arch probe**: verified in-image that
  FI 0.7.0.post1's `fast_decode_plan` accepts `q_len_per_req` (FI #3871,
  merged Jul 2026 — the graph-safety property: frozen plan_info depends
  only on batch/total_rows/uniform_q_len).
- On sm_87 the native fa2 tensor-core decode is the ONLY decode path (no
  trtllm-gen, no XQA) — exactly the case the patch's comment anticipates
  ("e.g. SM110"). The fa2 TC kernel family is SM80+; fp8 KV on Orin is
  NATIVE per the availability matrix. The verify shape is causal — the
  fa2 multi-token path's supported mode (the non-causal multi-token
  belongs to prims-ts, SM100a/103a only — irrelevant here).

**Drafter (MTP K=3 — the AWQ_MJOLNIR config):**
- 49652 (draft-decode capture under dynamic SD): arch-agnostic.
- The fused multi-step draft decode: the Mjolnir backend implements
  `update_draft_decode_metadata` (backend.py:636) and
  `get_cudagraph_support` returns UNIFORM_BATCH for max_q_len>1
  (backend.py:525) — both arch-agnostic; the Orin path returns the same.
- The Orin runner is capture-safe by construction: capacity-fixed
  buffers, the combine's ns is the runner-fixed ns_cap (not per-step),
  per-step state is device-buffered, and there is an explicit
  warm-up-before-capture guard (runner.py `_combine`).
- The T5-plan-refresh patch is REQUIRED on Orin too: the Orin runner's
  split plan is host-computed from seq_lens, the same stale-plan hazard
  the patch fixes for the Thor (the c1 regression's root cause).

**Drafter (DSpark, opt-in):** the non-causal CG grant gates on
non-SM12x → sm_87 included; the DSpark draft rides the FI prefill
wrapper (fa2 prefill, SM80+). Open items are arch-independent: the
DSpark legs have never run anywhere, and vLLM #56289
(get_cudagraph_support head-count vs draft CG) is still in progress.

## Caveats

1. **The c1 shape-gate is sm_110-only by design.** On Orin the draft is
   always CG at N=1. The Thor forensics found the graphed draft 7–16%
   slower than eager at N=1 (replay overhead on a tiny draft). If the
   e2e bench shows the same on Orin, port the gate: add family 8 to the
   helper's condition (one line + the env knob already exists).
2. **torch's DEVICE_REQUIREMENT warning on Orin is noise.** At CUDA init
   torch 2.13+cu130 warns "no published builds support this GPU" (its
   table excludes 8.7 from the sm_80/86 intervals) — but the entire
   session's kernel benchmarks ran on this Orin in stock-image
   containers, so the sm_80 cubins demonstrably run on sm_87. Advisory
   only; do not treat it as a blocker.
3. The `get_cudagraph_support` head-count mismatch (#55581/#56289) can
   limit draft CGs on any arch — watch for it in the smoke logs.
