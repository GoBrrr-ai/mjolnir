# Serving Qwen4-27B (dense `qwen4`) on Thor

**Story:** "I want to serve the upcoming Qwen4-27B — the dense member of the
Qwen4 family — on Jetson Thor the day its weights land, with the Mjolnir
decode kernel and the fp6 KV cache already working."

## What the architecture is

The Qwen4 family's mechanics are public today as
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(`model_type: qwen4_exp`, the "experimental preview of the architecture that
will underpin Qwen4"). Per layer, a hybrid of:

- **Gated DeltaNet (GDN)** linear-attention layers (16 QK / 48 V heads × 128,
  conv 4) — the same family Qwen3.8-27B runs, and the same layers vLLM
  already serves natively;
- **QSA ("Qwen Sparse Attention") full-attention layers**: GQA 24 Q / 4 KV,
  head_dim 256, plus a small **indexer** — an MQA (4 query heads × 128) over
  a *second, compressed* KV pool (one pooled key per 4 tokens, budget 2048
  tokens = 512 blocks) — that selects which tokens the main attention reads;
- **Hyper-connections**: a 4-branch residual stream (`hc_count: 4`, low-rank
  gate) around every block — this widens activation/drafter buffers, not the
  KV cache;
- **MTP** speculative decoding (1 drafter layer), and optionally a **PLE**
  n-gram embedding table (~95 GiB in bf16 at the family's 20M×2560; CPU-offloaded
  by vLLM).

What is **not** public is the 27B member itself: community reports (via a
since-deleted Intel kernel PR, cross-referenced in
[llama.cpp #31072](https://github.com/ggml-org/llama.cpp/issues/31072)) say
it is **dense**, hidden 5120, FFN 14336, with MTP exercised at **4 draft
tokens** — unconfirmed until the config ships. The full-attention geometry
(24:4 × 256) is reported unchanged, which matches the "KV cache per token
does not shrink" observation.

## What the image already carries

- **`thor-qwen4-dense-model.patch`** — vLLM's `qwen4_exp` model builds a
  sparse-MoE block for every layer, and the config class rejects
  `num_experts <= 0`, so a dense checkpoint cannot load upstream. The patch
  mirrors qwen3_next's own hybrid rule: a checkpoint that lists every layer
  in `mlp_only_layers` gets a dense MLP (`Qwen3NextMLP`). It is **inert**
  for every existing checkpoint — see `docker/vllm-thor/PATCHES.md`
  ("Thor qwen4 dense decoder").
- **`MJOLNIR_ATTN_QSA`** (kernel probe, default off) — the QSA indexer
  decode is the Mjolnir kernel re-parameterized (MQA 4×128 over the
  compressed pool, one KV row per 4-token slot); `mjolnir verify qsa`
  proves that geometry kernel-vs-emulator at every context length. The
  gather decode (reading only the *selected* main-KV rows) is the follow-up
  kernel change, still behind the probe.
- **`thor-qwen4-topk-sm110.patch`** — device-proven 2026-10-31: stock
  v0.32.0's sparse-indexer decode top-k leads with `cooperative_topk`,
  which **hard-fails on Thor** (16-block cluster vs max 8 → `EngineDeadError`
  at the first decode step). The patch excludes sm_110 from that gate
  (→ `persistent_topk`), same fix upstream applied for SM120. See
  `docker/vllm-thor/PATCHES.md` ("Thor qwen4 topk sm110").

One structural fact to know: **`--attention-backend` does not route the QSA
layers.** `Qwen4ExpQSAAttention` hardcodes its own backend and impl
(v0.32.0: a Triton sparse kernel over the `logical_indices` buffer, with
FlashAttention used only as interface scaffolding — flash-attn itself never
runs over the QSA cache), and the GDN layers are linear attention. So
Mjolnir's QSA integration is a **probe-gated impl swap inside
`Qwen4ExpQSAAttention`** (our tcgen05 gather + d=128 indexer decode behind
`MJOLNIR_ATTN_QSA`), not a backend-registry change. The v0.32.0 Triton
kernels stay the correctness reference.

## What QSA changes for serving

- **Main KV per token is unchanged** (same full-attention heads) — the fp6
  e2m3 path and its calibration apply as for any model. The indexer pool
  adds its own bytes: 2 rows (raw + pooled key) of 128 per 4-token slot —
  with the 16-full-layer assumption that is ≈ 7.8% on top of the main KV in
  fp8; **the exact layer count is unknown until the config ships**.
- **Speculative drafts cap at 4** on QSA layers (the pending index-key ring
  holds one compressed group — observed in SGLang's QSA MTP support);
  plan the `verify spec --drafts` sweep at K ≤ 4 and re-confirm on the
  checkpoint.
- **Prefill / mixed steps** run the QSA layers densely (selection = every
  token under the budget is bit-identical to dense attention — the
  correctness oracle), so FA4 fallback correctness is not at risk; only
  gather *speed* is a kernel project.
- **If the 27B ships PLE**, its ~95 GiB table must stay host-resident
  (vLLM's PLE offload): exclude it from `gpu-memory-utilization` planning
  and expect `mjolnir gc`'s page-cache discipline to matter around restarts.

## Fixture serve-smoke on Thor (2026-10-31) — one hard upstream fail found

Serving the tiny dense `qwen4_exp` fixture (GDN+QSA layers, random weights)
on sm_110 with stock vLLM 0.32.0 paths (Mjolnir image, `--enforce-eager`):

- **The engine loads and prefills** — the dense-MLP patch, the GDN
  CuTe-DSL prefill kernel (`gdn_blackwell_prefill_sm110a`), hyper-connection
  glue and the indexer pool allocation all work on Thor.
- **The first decode top-k kills the engine**: `cooperative_topk.cu:48 —
  launch failed: cluster misconfiguration`. vLLM's
  `--sparse-indexer-topk-backend auto` chain is cooperative → persistent →
  per_row, but a launch *error* is a fatal `EngineDeadError`, not a
  fallback: the indexer's thread-block cluster config does not fit
  sm_110's 20-SM partitions. **Stock Qwen4 QSA serving is broken on Thor
  until the topk backend is pinned or the Mjolnir impl owns it.**
- Backend knobs (`vllm serve --sparse-indexer-topk-backend …`):
  `deep_select` (SM100a/103a only), `cooperative` (the fail above),
  `persistent`, `per_row`, `flashinfer`, `torch` (debug reference).
  **`persistent` VALIDATED on Thor (2026-10-31, patched image):** the
  fixture serves end-to-end — greedy decode finite and deterministic
  through all QSA layers, zero engine errors. That run was also the
  **first execution of the stock Triton QSA decode attention on sm_110**
  (untuned but functional). The `torch` backend is the guaranteed-correct
  reference; `per_row`/`flashinfer` remain untested.
- This **upgrades the impl-swap from perf to correctness** for decode:
  whichever way release day goes, the serving config must either pin a
  working topk backend or route QSA decode through the Mjolnir impl.
- Fixture-generator notes (transformers 5.19): build the causal LM from
  `Qwen4ExpTextConfig` (the composite does not forward attributes), ship a
  composite-shaped `config.json` (`model_type: qwen4_exp` +
  `text_config` — vLLM reads `hf_text_config`), and the dense tensors are
  `mlp.gate_up_proj.weight` / `mlp.down_proj.weight` with the MoE router
  `mlp.gate.weight` dropped.
- **The Mjolnir impl-swap ships** (`thor-qwen4-qsa-mjolnir.patch` +
  `mjolnir.qsa_impl`): `MJOLNIR_ATTN_QSA=1` routes pure-decode steps
  through the tcgen05 scratch-gather (the measured 13.5×/11.7× path) and
  falls back to the stock Triton path for prefill/verify/mixed batches,
  bf16 caches, or anything unexpected — announcing both. **Device-validated
  2026-10-31**: the fp8-cache fixture smoke shows the scratch-gather live in
  real serving (`scratch-gather active (fp8, q_len=1, PS=800, …)` — note the
  GDN hybrid forces the attention page size to the mamba page size, 800;
  the gather reads it through the pool's own strides and lands the rows in
  a 128-row scratch), byte-deterministic greedy output, zero engine errors.

## Release-day checklist

> **The runnable version of this checklist — exact commands, decision
> tree, failure table — is [qwen4-27b-day0.md](qwen4-27b-day0.md).** This
> section is the rationale; that file is what you open on the day.

0. **Base audit (done 2026-10-31, v0.32.0 cut 2026-10-17):** the MTP
   quantized-layer remap (#55513) and the PLE pinned-host UVA offload are
   **already in**; the post-cut QSA-naming and PLE-disk-offload PRs are out
   but non-blocking. The qwen4_exp custom ops carry **no sm_110 hard gates**
   (HC glue, QSA sparse, the `hc_down_silu` CuTe-DSL JIT and the
   capability-keyed skinny-GEMM plans all fall back gracefully) — **with one
   device-proven exception** (see the smoke result below): the sparse-indexer
   decode top-k `auto` chain leads with `cooperative_topk`, which is a hard
   launch failure on Thor.
1. **Fetch the config** and check the five unknowns: layer count, the
   GDN/full schedule, whether PLE exists, the MTP drafter depth, and the
   `model_type`.
2. **`model_type: qwen4_exp`** → the dense patch handles it; the checkpoint
   should list every layer in `mlp_only_layers` (and `intermediate_size`).
   **A new model type** → the patch is necessary but not sufficient: a new
   architectures registration is needed first.
3. **Create the serving config** per [new-model-config.md](new-model-config.md):
   `max-model-len` per the checkpoint, `--speculative-config` with
   `num_speculative_tokens ≤ 4`, and the KV dtype from the quantized
   checkpoint (`kv-cache-dtype-skip-layers: sliding_window` if the family
   keeps SWA layers — check `layer_types`).
4. **Memory ledger**: `mjolnir verify kv-cache-size` with the new config —
   the indexer pool changes the per-token bytes (see above).
5. **`mjolnir verify qsa`** on the device, then `mjolnir verify spec
   --drafts "1 2 3 4"` for the acceptance curve.
6. **Any default flip** (KV dtype, draft count) follows the same-run A/B
   rule — see [../methodology/benchmarking.md](../methodology/benchmarking.md).

## Sources

- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
  config.json + model card (the family mechanics; accessed 2026-10-30);
- [vLLM `qwen4_exp` model code](https://github.com/vllm-project/vllm/tree/v0.32.0/vllm/models/qwen4_exp)
  (present in our pinned v0.32.0);
- [llama.cpp #31072](https://github.com/ggml-org/llama.cpp/issues/31072) and
  [SGLang issue #44099](https://github.com/sgl-project/sglang/issues/44099)
  (the 27B shapes and the MTP ring constraint — community-reported, unconfirmed);
- this repo: `docker/vllm-thor/PATCHES.md` ("Thor qwen4 dense decoder"),
  `src/mjolnir/mjolnir/arch/thor/qsa.py`.
