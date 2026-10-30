# vLLM Thor PR Patches

Seven vLLM PR fixes, **hand-adapted** onto the vLLM base (not a raw
`git apply` of the upstream PR diffs — the four originals because the base
code has diverged from each PR's base, the three newer fixes to pin them
to v0.32.0 and carry the THOR sentinels), plus nine **Thor-authored** patches: the **Mjolnir
attention-backend name** (`AttentionBackendEnum.MJOLNIR` — see "The Mjolnir
attention backend" below), two for a capability the upstream backends don't
yet have, two `sm_110` gate probes (see "Thor sm_110 gate probes" below), one
dispatch-side `sm_110` draft-CG shape gate (see "Thor sm_110 draft-CG shape
gate" below), one **flashinfer-side** sm_110 enablement hunk (see "Thor sm_110
GDN prefill enablement (flashinfer side)" below), one hd256 FP8-KV kernel +
policy patch (probe-gated, T11 canary in `kernel_test_sm110_gates.py`) and one
hd256 1CTA decode carve-out (see "Thor hd256 1CTA decode carve-out (v11,
2026-09-24)" below), plus the fp6 e2m3 KV cache dtype for the Mjolnir backend
and two **sm_110 MoE backend gate fixes** plus a **spec-decode split top-p
fix** (see "Thor MoE backend gates on sm_110" below).
Seventeen patches in total, applied by `apply_patches.py`
onto the installed vllm package **and** the installed flashinfer package;
verified by `verify_patches.py`.

The retired GEMV decode kernel patch (`thor-fa4-hd256-gemv-decode-sm110.patch`)
and its kernel were removed when the Mjolnir kernel replaced them; the last
snapshot is the git tag `gemv-archive`: [`gemv-archive:deprecated/gemv/`](https://github.com/GoBrrr-ai/mjolnir/tree/gemv-archive/deprecated/gemv).

Base this was verified against: **vllm 0.32.0**
(`vllm/vllm-openai:v0.32.0-ubuntu2404`), FlashInfer 0.7.0.post1.

| File | PR | What it fixes |
|------|----|---------------|
| `50885-flashinfer-full-cudagraph.patch` | [#50885](https://github.com/vllm-project/vllm/pull/50885) | Capture **FULL** decode cudagraphs for spec-decode on the FlashInfer **native** (non-trtllm) path — uniform multi-token verify batches. Previously only trtllm-gen (SM100) / dedicated-XQA (SM12x) could FULL-capture spec decode; elsewhere it fell back to PIECEWISE, hurting Thor (SM110) TTFT. |
| `49652-draft-decode-capture.patch` | [#49652](https://github.com/vllm-project/vllm/pull/49652) | Autoregressive (MTP) **draft-decode** cudagraph capture under **Dynamic SD**. The draft decoder always processes 1 token/request, so its graphs must stay at fixed query length 1 instead of following the target's dynamic-SD verify schedule. |
| `54165-mamba-spec-kv-restore.patch` | [#54165](https://github.com/vllm-project/vllm/pull/54165) | Restore **hybrid mamba-align cache hits under spec decode** with a KV connector. Gates the EAGLE last-block drop on drafters that actually *preserve* (pollute) the target KV cache — DFlash/DSpark don't, so they must not back off `last_cache_position`. **Bump note (2026-10-28):** upstream #54163 (merged 2026-10-23, in the next release) landed the *gating half* of this as `needs_eagle_block_drop()` (= `use_eagle()` minus DFlash/DSpark) — at the next bump re-base this patch's `use_eagle_block_drop` hunk onto it (our `use_eagle_preserves_target_kv_cache()` predicate is the same idea, property-named instead of method-listed) and keep the rest until #54165 itself merges (still open). `config/speculative.py` + `dflash/speculator.py` + `scheduler.py` all churned post-0.32.0 — expect re-adaptation. |
| `55519-no-warn-when-block-drop-off.patch` | [#55519](https://github.com/vllm-project/vllm/pull/55519) | **Don't warn "prefix reuse is disabled" when `disable_eagle_block_drop` is set.** All flag-all fallbacks are gated on `use_eagle_block_drop()`, so with the drop disabled reuse is intact — the old `use_eagle()` gate made the warning a false positive. Also rewords the scheduler's "drop disabled" warning to document that reuse works in that mode. |
| `thor-fused-draft-decode-fi-native-update.patch` | — (Thor-authored) | **Implement `update_draft_decode_metadata` for the FlashInfer backend** so the spec-decode loop can use the **fused multi-step draft decode** instead of the "rebuild attention metadata between draft steps" fallback. The upstream contract (backend.py `update_draft_decode_metadata`) is implemented for triton/triton_mla (a `pass`, they read live GPU buffers) and flash_attn (re-plan the scheduler), but the FlashInfer native decode wrapper plans from host buffers and had no in-place advance. Adds `_advance_fi_decode_plan_kernel`, which re-derives the FI decode plan (`paged_kv_indptr`/`last_page_len`/`page_indices`) **on the GPU** from the advancing `seq_lens` + block table — the same formula as `build()`, but capture-safe (no host sync, no D2D index copy) so it replays correctly inside the draft CUDA graph. Gated to full-cudagraph, single-rank, non-DCP, native-FI (non-trtllm) builders. Pays off only at **K ≥ 3** (MTP3 / DSpark K≥2); it is a deliberate no-op at K=2. |
| `thor-fused-draft-t5-plan-refresh.patch` | — (Thor-authored) | **Rebuild the draft attention metadata on the fused FULL-graph replay path** so host-side per-cycle plan state survives graph replay. 0.32.0's `_fused_multi_step_decode` returns before `_build_uniform_attn_metadata` when `cg_mode == FULL` (0.31.x built it unconditionally), so any builder that computes per-cycle host state in `build()` — Mjolnir's T5 split-KV **work plan** (`work_buf`/`nsplit_buf`, host-computed from seq_lens) — stays **baked at capture-time seq_lens**: every replayed draft step attends a truncated context and MTP acceptance collapses beyond position 0 (measured 2026-10-18: K=5 acceptance 4.1→**2.04**, per-position 0.82/0.17/0.04/…; eager or `MJOLNIR_ATTN_FUSED_DRAFT=0` healthy). The fix re-runs the build before `run_fullgraph` — the returned metadata is unused (the graph replays baked launches), but the side effect refreshes the persistent plan buffers the graph reads, restoring the 0.31.x contract at the same per-cycle host cost 0.31.x always paid. |
| `thor-dspark-draft-fi-noncausal-cudagraph.patch` | — (Thor-authored) | **Capture the DSpark non-causal multi-token draft under a FULL CUDA graph** instead of running the draft eagerly (removes the `speculator.py:124` "does not support full CUDA graphs; running the draft eagerly" warning). A non-causal drafter (DSpark/DSpark) routes its draft through the FI *prefill* wrapper — the FI *decode* wrapper cannot express non-causal multi-token (its fa2 kernel is block-causal only for `q_len_per_req > 1`). In cuda-graph mode that prefill wrapper freezes its batch size at creation and reads its KV plan from **persistent** buffers, so one wrapper per captured batch size is needed (`_noncausal_prefill_wrappers_cudagraph`, mirroring `_decode_wrappers_cudagraph`); the draft plans eagerly outside the captured graph and the captured `run()` replays safely across ragged re-plans. Adds the `_get_noncausal_prefill_wrapper_cudagraph` factory, grants a non-causal `UNIFORM_BATCH` in `get_cudagraph_support` (gated on a multi-token drafter, non-SM12x, single-rank DCP), and dispatches the prefill batch to the cudagraph wrapper in `build()`. Lands **last** in `PATCH_ORDER` (its `flashinfer.py` hunks sit on top of the fused-draft-decode patch's changes). Most bump-fragile patch — see below. |
| `thor-draft-cg-gate-sm110.patch` | — (Thor-authored) | **Shape-gate the DSpark draft's FULL cudagraph replay on sm_110 (the c1 fix).** The forensics showed the graphed draft is 7–16% *slower* than the eager draft at N=1 (replay overhead eats the launch savings on a tiny 5-layer draft). The gate routes per-step draft batches below `VLLM_THOR_DRAFT_CG_MIN_BATCH` requests (default **2**; min 1, where 1 = today's behavior) through the **existing eager fallback** via the dispatch's `need_eager` channel — no new eager path, no capture-logic changes. sm_110 only: off sm_110 the decision helper is a structural no-op (always CG). One-shot `THOR draft-CG gate: …` log lines; any gate evaluation error degrades to the CG behavior (logged once). Probe-gated, inert on non-sm_110, default N=1→eager. |
| `thor-fa4-hd256-1cta-decode-sm110.patch` | — (Thor-authored) | **The 1CTA decode carve-out for the FA4 hd256 kernel (the v11 "N-2 decode" workstream).** Decode-shaped hd256 calls — `max_seqlen_q <= 8` (decode M 1..8, incl. MTP verify; prefill is 1024+), non-local, dense or paged-128 TMA (the kernel's only legal paged mode) — now OR into the `hd256_use_2cta` carve-out in `vllm_flash_attn/cute/interface.py` (one hunk), so the kernel drops its 2CTA cluster to a single CTA (`cluster (1,1)`, `CtaGroup.ONE`): a **1.56× decode win** at L=8192 M=1 (clean micro-bench 2CTA 323 µs → 1CTA 207 µs), with 1CTA == 2CTA == fp32-ref within fp8 tolerance; the kernel file is untouched and the 2CTA prefill path is unchanged (separate compile keys coexist in one process). The in-tree hd256 SplitKV machinery was investigated and is a no-win on **both** the 2CTA and 1CTA paths — kept inert (see "Thor hd256 1CTA decode carve-out (v11, 2026-09-24)" below). |
| `thor-mjolnir-attention-backend.patch` | — (Thor-authored) | **The `MJOLNIR` attention-backend name.** Adds `MJOLNIR = "mjolnir.backend.MjolnirAttentionBackend"` to `AttentionBackendEnum` (`v1/attention/backends/registry.py`), so `vllm serve <model> --attention-backend MJOLNIR` parses (the CLI resolves the name through the enum *before* plugins could add one — an `Enum` cannot grow at runtime) and `Attention.__init__`'s `AttentionBackendEnum[backend.get_name()]` round-trip works. The class itself comes from the `mjolnir-kernel` package, installed by the `Dockerfile`; its `vllm.general_plugins` entry point also maps `CUSTOM` to the same class, so a stock vLLM can use `--attention-backend CUSTOM`. Also makes **FlashAttention 4 the Mjolnir backend's default fallback** (its prefill/mixed-step backend) on sm_110: a `fa_utils.py` hunk resolves `flash_attn_version` to 4 when the selected backend is `MJOLNIR`/`CUSTOM`, `MJOLNIR_ATTN_FALLBACK` is unset or `FLASH_ATTN`, and none is configured — see "The Mjolnir attention backend" below. |
| `thor-fp6-e2m3-kv-cache.patch` | — (Thor-authored) | **The `fp6_e2m3` KV cache dtype.** Adds `"fp6_e2m3"` to `CacheDType` (`config/cache.py`), its storage dtype (`uint8`) and `is_quantized_kv_cache` (`utils/torch_utils.py`), and `KVQuantMode.FP6_E2M3` with its `get_kv_quant_mode` mapping (`v1/kv_cache_interface.py`). Nothing else: the Mjolnir backend publishes the packed cell size (`customize_spec`: `d·6/8 bytes per K and per V row`), writes the cache, reads it in decode, and expands a step's pages to e4m3 for the FA4 fallback. Inert unless a config sets `kv-cache-dtype: fp6_e2m3`; no other backend accepts it. See "fp6 e2m3 KV cache" below. **Retire at the next base bump**: vLLM #52670 (pluggable KV cache data types, merged 2026-10-20 — first release after 0.32.0) added `register_kv_cache_dtype` + `KVQuantMode.CUSTOM`, which is exactly this patch's four edits as an API; `mjolnir.kvdtype` (probe-gated, ships in the package since 2026-10-28) registers the dtype from the plugin entry point on any base that has the API. All four files the patch touches were also churned by #52670 itself (`is_quantized_kv_cache` moved to `v1/kv_cache_interface.py`) — the patch will not apply. |
| `fi-fp6-kv-ampere-prefill.patch` | — (Mjolnir-authored, **FlashInfer-side**) | **Packed-fp6 (e2m3) KV direct-read for FlashInfer's Ampere (sm_80-family) FA2 prefill** — the scratchpad-free fp6 fallback: the produce stage stages 16 codes (12 bytes, partial-src cp.async) per smem slot and an in-place pass rewrites each slot to e4m3 (`code · s_channel`, per-(kv-head, channel) float scales via two new optional `maybe_{k,v}_fp6_scales` params), after which the tile is byte-identical to fp8 and **every** downstream path (per-fragment `vec_cast`, both tile shapes) runs unchanged. Inert unless the fp6 dtype string is requested; see "fp6 prefill direct-read (flashinfer side, sm_80/87)" below. |
| `thor-gdn-prefill-sm110.patch` | — (Thor-authored) | GDN **prefill** grant in `_resolve_gdn_prefill_backend()` (the fast FlashInfer kernel for the model's GDN layers) — probe-gated at import; see "Thor sm_110 gate probes". |
| `thor-fa4-fp8kv-sm110.patch` | — (Thor-authored) | **FA4 + FP8-KV** eligibility gate (probe-gated fam(110) OR-term in `flash_attn_supports_kv_cache_dtype`) + the FA4 arch assert widened to `(10, 11)` — **inert until a config selects FA4** (default stays FlashInfer); see "Thor sm_110 gate probes". |
| `thor-fa4-hd256-fp8-sm110.patch` | — (Thor-authored) | hd256 **FP8-KV kernel + policy** patch (the "N-1" workstream — enable the FA4 hd256 kernel on sm_110, probe-gated, T11 canary in `kernel_test_sm110_gates.py`); the decode path is the 1CTA carve-out above. |
| `thor-gdn-prefill-fi-sm110.patch` | — (FlashInfer) | **FlashInfer-side** GDN prefill non-CP dispatch allowlist widened major 10 → (10, 11) so the SM100 CuTe-DSL kernel is reachable on sm_110 (behavior unchanged on other devices); see "Thor sm_110 GDN prefill enablement (flashinfer side)". |
| `thor-sm110-mxfp4-moe-gate.patch` | [#62133](https://github.com/vllm-project/vllm/pull/62133) | **Fix OAI Triton MXFP4 MoE on SM110** (supersedes closed #61387). #58877 (in v0.32.0) widened the OAI-Triton device gate to `< (13, 0)`, admitting SM110 → `matmul_ogs` compile failure at engine init (upstream issue #61374). The GPT-OSS MXFP4 **selector** keeps **MARLIN** the default on SM110 (TRITON opt-in) and `_swizzle_mxfp4` applies the persistent + `epilogue_subtile=1` constraints on SM110 too. Drop when upstream merges #62133 — **merged 2026-10-27 (f03b704), first release after 0.32.0: retire this patch at the next base bump** (upstream rewrote the same hunks; its Thor numbers — MARLIN 13:22 vs TRITON 18:24 eval — confirm the MARLIN default our gate chose). |
| `thor-sm110-cutlass-fp4-moe-guard.patch` | [#62275](https://github.com/vllm-project/vllm/pull/62275) | **Disable CUTLASS FP4 MoE backend on sm_110.** NVFP4 MoE auto-select picks `VLLM_CUTLASS` on Thor and faults in `profile_run` (TMA descriptor init); sm_110 is removed from `_supports_current_device`'s family allowlist so the fallback backend is chosen. Drop when upstream merges #62275. |
| `thor-spec-decode-split-topp.patch` | [#58231](https://github.com/vllm-project/vllm/pull/58231) | **Split top-p masks every token for huge forced logits → token 0.** `_topp_sb_mask_kernel` (the split top-p pipeline: CUDA, batches ≤ 64, rows with `top_p < 1` and no active top-k) mapped its probability-space pivot back to logit space as `log(pivot) + log(Z) + M`, which fp32 rounds to `M` when a forced logit makes `|M|` huge (thinking-token-budget writes 1e9) — the whole row masks, spec decode emits token 0 forever (≈3× throughput loss). The fix masks in probability space (`exp(x − M)/Z > pivot`, bitwise the step kernels' predicate; `x == M` kept as a backstop). Trigger needs a **huge forced logit**, not merely `top_p < 1` + spec decode. Drop when upstream merges #58231 — it is merged 2026-10-19, first release after 0.32.0. |

## Why "hand-adapted" and not the raw diff

The four PRs were written against vllm bases that predate the current
nightly. Two kinds of drift force manual adaptation:

1. **The nightly already contains *newer* concepts.** The scheduler/coordinator
   now use `use_eagle_block_drop()` / `use_eagle_block_drop()` gating, which the
   PRs (written against `use_eagle`) don't know about. For **54165** this means
   its "introduce `drop_last_prefix_cache_block`" hunk is *superseded* — the
   correct adaptation is to make the existing `use_eagle_block_drop()` return
   `use_eagle_preserves_target_kv_cache()` instead of `use_eagle()`. For
   **52244**, the new `state_position` stop is gated on `use_eagle_block_drop`
   (the nightly's equivalent of the PR's `use_eagle`).

2. **Signatures grew a trailing parameter.** The nightly's
   `reachable_block_mask(...)` added `dcp_world_size`/`final_segment_end_block`
   params after `reachable_boundaries`; the PRs insert
   `unreachable_boundaries` at that same slot, so the whole signature (and the
   `SlidingWindowManager` override) must be extended, not just the base.

3. **A PR hunk is already done by the nightly.** 54165's
    "SlidingWindow `find_longest_cache_hit` assert → block-aligned fallback"
    hunk is already present in the nightly (it has the dense-fallback comment),
    so that hunk is omitted from the adapted patch.

Note: **55390/55519** (opened 2026-09-04/06, i.e. ~2 days after this nightly)
were adapted with zero context drift — only the `tests/` hunks were dropped
(the wheel ships no test dir). Their source hunks applied verbatim at offset
only, so these two should survive nightly bumps longest.

## Files touched (per patch)

- **50885**: `v1/attention/backends/flashinfer.py`
  - new `flashinfer_supports_uniform_multi_token_decode()` (@cache) probing
    FlashInfer's `fast_decode_plan` for `q_len_per_req`;
  - `_decode_wrappers_cudagraph` keyed by `(batch_size, q_len_per_req)`;
  - `supports_spec_as_decode` / `get_cudagraph_support` extended to return
    `UNIFORM_BATCH` when the FI native multi-token path is available;
  - `build()` computes `decode_q_len = num_decode_tokens // num_decodes`,
    per-request indptr/last_page_len, passes `q_len_per_req` through to
    `_get_decode_wrapper` and `fast_plan_decode`.
- **49652**: base `CudaGraphManager._init_candidates` (rebased 2026-09-18)
  - one-line guard: skip Dynamic-SD candidate expansion when
    `decode_query_len <= 1` — replaces the old
    `SpeculatorCudaGraphManager.__init__` workaround in
    `v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py`.
- **54165**: `config/speculative.py`
  - new `use_eagle_preserves_target_kv_cache()` (eagle/eagle3/mtp only);
  - `use_eagle_block_drop()` now returns
    `use_eagle_preserves_target_kv_cache() and not disable_eagle_block_drop`.
- **55519**: `v1/core/kv_cache_utils.py`, `v1/core/sched/scheduler.py`
  - `_warn_if_unannotated_eagle_mamba()` gate changed from `use_eagle()` to
    `use_eagle_block_drop()` — when `disable_eagle_block_drop` is set, every
    flag-all consumer fallback is already off, so the "reuse disabled" warning
    was a false positive;
  - scheduler's "drop disabled" warning reworded to document that the trailing
    block stays cache-eligible in that mode.
- **thor fused draft-decode** (Thor-authored, no upstream PR):
  `v1/attention/backends/flashinfer.py`
  - `FlashInferMetadataBuilder.__init__` sets
    `supports_draft_decode_metadata_update = enable_cuda_graph and
    dcp_world_size==1 and not use_trtllm_decode_attention` (the trtllm flag
    identifies the *non*-native path) and allocates the live references
    `_draft_seq_lens_gpu` / `_draft_block_table`;
  - `build()` captures `common_attn_metadata.seq_lens` / `.block_table_tensor`
    (the advancing GPU buffers) into those fields;
  - new `update_draft_decode_metadata()` — returns for the trtllm/XQA
    `decode` (already stable), guards to the pure uniform single-token decode,
    and launches `_advance_fi_decode_plan_kernel`;
  - new module-level Triton `_advance_fi_decode_plan_kernel` — one program per
    request re-derives `paged_kv_indptr` (prefix block count, recomputed from
    the read-only `seq_lens` so programs are independent), `last_page_len`
    (promoted to `page_size` when a non-empty request lands on a boundary), and
    `page_indices` (copied from the live block table). Same formula as
    `_compute_flashinfer_kv_metadata`.
- **thor draft-CG gate** (Thor-authored, no upstream PR):
  `v1/worker/gpu/spec_decode/dflash/speculator.py`
  - pure decision helper `_thor_draft_cg_use_cuda_graph(num_reqs,
    capability, min_batch)` — `True` = replay the FULL draft graph; a
    structural no-op (always `True`) off sm_110;
  - lazy once-parsed `VLLM_THOR_DRAFT_CG_MIN_BATCH` env knob (default 2,
    minimum 1; invalid → 2 + one-time warning) and lazy
    `torch.cuda.get_device_capability()` read;
  - the per-step dispatch (`dispatch_cg_and_sync_dp` call in `propose`) now
    passes `need_eager = is_profile or _thor_draft_cg_gate_wants_eager(num_reqs)`
    — below the threshold the existing `NONE`-descriptor eager path runs;
  - four one-shot `logger.info` lines (state / eager / cudagraph / error).

## Behavioral gate (`functional_check_55519.py`)

The sentinel check in `verify_patches.py` proves the code *landed*; the Dockerfile
additionally runs `functional_check_55519.py`: the "prefix reuse is disabled"
warning path must be gated on `use_eagle_block_drop()`, and the scheduler's
"drop disabled" warning must document that reuse still works. (#55390, the MTP
draft-group annotation this used to check too, is upstream since vLLM 0.31.0.)
Run standalone inside any container with a patched vllm:
`docker run --rm --entrypoint python3 <image> /opt/vllm-patches/functional_check_55519.py`.

## Verifying a new nightly before re-adapting

```
docker pull vllm/vllm-openai:nightly-aarch64
# quick check: which fixes are already merged upstream?
docker run --rm --entrypoint sh vllm/vllm-openai:nightly-aarch64 \
  python3 - <<'PY'
import importlib.util, inspect, pathlib
r = pathlib.Path(importlib.util.find_spec("vllm").origin).parent
def has(rel, s): return s in (r/rel).read_text()
print("50885", has("v1/attention/backends/flashinfer.py",
                    "flashinfer_supports_uniform_multi_token_decode"))
print("54165", has("config/speculative.py",
                    "use_eagle_preserves_target_kv_cache"))
print("52244", has("v1/core/kv_cache_utils.py",
                    "mamba_state_cache_position"))
print("55519", has("v1/core/kv_cache_utils.py",
                    'not spec_config.use_eagle_block_drop():'))
print("fi-update", has("v1/attention/backends/flashinfer.py",
                       "def _advance_fi_decode_plan_kernel"))
PY
```
If a fix is already `True`, delete that `.patch` file and its entry in
`apply_patches.py::PATCH_ORDER` + `verify_patches.py::FIXES` (the build's
`--forward` would skip it anyway, but keeping the list accurate keeps
verification meaningful). If a patch's `--forward` run leaves a missing fix,
`verify_patches.py` names it — re-adapt that one patch against the new source.

## The Thor fused-draft-decode patch — verification & fragility

This patch is **not** an upstream PR; it is the first of the six-plus-one where
the overlay *adds a capability* to vLLM rather than backporting a merged fix.
Consequences:

- **Most fragile under bumps.** It inserts into the FI native-decode branch of
  `build()` and the `__init__` buffer block — both churned by #50885 and any
  future FlashInfer-backend work. If it fails to apply after a nightly bump,
  re-adapt it against the new `flashinfer.py` (the other six are unaffected).
- **Kernel is GPU-verified, not just present.** `verify_patches.py` only proves
  the symbols landed. The actual Triton kernel is checked by
  `kernel_test_fi_update.py`, which runs `_advance_fi_decode_plan_kernel`
  against a numpy reference over random seq_lens, page-boundary, empty
  (padded) requests, multi-page, and a 20× determinism repeat. The test
  **self-skips (exit 0)** when no CUDA device is visible or the GPU is
  saturated (a live engine holds most of the 128GB and can fragment the free
  pool) — so a plain no-`--gpus` build still passes. Build with
  `docker build --gpus all …` on a Thor (or run the test in a `--gpus all`
  container) to actually exercise the kernel.
- **Correctness guardrails baked in.** The `enable_cuda_graph` gate is load-
  bearing: without full-cudagraph the decode wrapper is built with `paged_kv_*
  = None` (`_get_decode_wrapper`), so the kernel would have nothing valid to
  advance — enabling the fused path in that case would be a real bug. The
  `not use_trtllm_decode_attention` gate keeps the trtllm/XQA path (which is
  already stable and needs a `pass`) out of the GPU-advance path. The method
  also refuses to advance anything that is not a pure uniform single-token
  decode, so a future non-uniform draft query degrades to the fallback rather
  than reading a stale plan.
- **No K=2 regression.** At `num_speculative_tokens=2` the fused loop runs a
  single draft step and the inter-step update is guarded off by
  `step < num_speculative_steps - 1`, so the K=2 MTP config is byte-for-byte
  unchanged in behavior; the patch only activates the fused path for K ≥ 3
  (MTP3 / DSpark K≥2), which is where the per-step metadata rebuild is avoided.

## The DSpark non-causal cudagraph patch — fragility

`thor-dspark-draft-fi-noncausal-cudagraph.patch` is the **most bump-fragile**
patch in the set (more than the fused-draft-decode patch). It inserts into two
of the most churned regions of `v1/attention/backends/flashinfer.py` at once:
the `get_cudagraph_support` tail (where the non-causal `UNIFORM_BATCH` grant
lands) and the prefill branch of `build()` (where the routing call site
replaces the `_get_prefill_wrapper` call). Both are rewritten by #50885 and
further by the fused-draft-decode patch, so a nightly that moves either region
breaks this patch's context — if it fails to apply after a bump, re-adapt it
against the new `flashinfer.py` (the other six are unaffected). It is kept
**last** in `PATCH_ORDER` for exactly this reason.

Because both hunks target hot code, the `verify_patches.py` entry asserts the
*used* lines, not just the definitions (the 52244 lesson): the wrapper factory
def, the `_noncausal_prefill_wrappers_cudagraph` dict in `__init__`, the gate
line in `get_cudagraph_support`, and — load-bearing — the `build()` routing call
site `self._get_noncausal_prefill_wrapper_cudagraph(num_prefills)` plus the
`use_nc_cudagraph =` condition. A partial apply that lands the definition but
not the routing must fail the gate. The captured behavior itself (gate, routing,
non-causal run vs. reference, capture-replay across ragged re-plans,
determinism) is checked by `kernel_test_dspark_draft_nonncausal.py` (Dockerfile
gate), which **self-skips (exit 0)** when no CUDA device is visible or the GPU
is saturated, so a plain no-`--gpus` build still passes. Build with
`docker build --gpus all …` on a Thor (or run the test in a `--gpus all`
container) to actually exercise the graph.

## Thor sm_110 gate probes (2026-09-18)

Two Thor-authored patches that turn *static upstream arch gates* into
**one-shot runtime probes** for Thor (SM 110 / sm_110a). Each adds an SM110
grant to an existing gate, but the grant fires only if a small
deterministic kernel A/B test passes on the current device. The verdict is
process-cached (zero per-forward overhead), any failure degrades to
today's exact behavior, and — the point of the probes — each grant
**auto-activates with zero vLLM changes** when the underlying kernel
support lands in a future CuTe-DSL / flashinfer release.

| Patch | What it gates | Probe behavior | Current Thor status |
|-------|---------------|----------------|---------------------|
| `thor-gdn-prefill-sm110.patch` | The FlashInfer **GDN prefill** grant in `_resolve_gdn_prefill_backend()` (`qwen_gdn_linear_attn.py`) — the fast prefill kernel for the model's GDN layers | FlashInfer's `fi_chunk_gated_delta_rule` vs. the Triton/FLA reference on a deterministic, state-carrying 2-chunk input (B=1, T=256, H=2, K=V=128, seeded; pre-L2-normalized q/k); output **and** final state must match within tight bf16 tolerances | **Enablement (probe-gated, live-verified 2026-09-21 on this Thor)**: the flashinfer-side hunk (`thor-gdn-prefill-fi-sm110.patch`) widens the FI GDN prefill dispatch to sm_110, so the probe now *exercises the real FI kernel path*; live T6 canary: probe **PASS** (`THOR GDN prefill probe: FI kernel enabled on sm_110`, `resolved ('auto', 'flashinfer')`); the grant fires only if the probe passes, otherwise Triton prefill, byte-identical to today |
| `thor-fa4-fp8kv-sm110.patch` | The **FA4 + FP8-KV** eligibility gate: probe-gated fam(110) OR-term in `flash_attn_supports_kv_cache_dtype` (`fa_utils.py`) — the *only* policy gate that rejects FA4+FP8 on sm_110 — plus the FA4 kernel's arch assert widened `arch//10 == 10` → `in (10, 11)` (`cute/interface.py`) | One Qwen3.8-shaped full-FP8 paged GQA decode (B=2, 8q/4kv, head_dim 128, page 16, causal, seqused 96/64; Q+K+V all e4m3, mirroring vLLM's quantize-Q-for-FP8-KV production path) through the real `fa_version=4` dispatcher vs. an fp32 reference (rtol 2e-2 / atol 1e-2; live evidence on this shape class: maxerr ≈ 0.015) | **INERT today until config explicitly selects FA4** (passive-by-default): sm_110's default FA version is 2, so the probe-gated term is unreachable and the patch changes nothing for every existing config; enabling requires `--attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}'` + fp8 KV, and even then the grant fires only while the probe passes on the device |

> **REMOVED 2026-09-21:** the other two probe-gated sm_110 grants —
> `thor-b12x-nvfp4-sm110.patch` (NVFP4 B12x GEMM) and
> `thor-trtllm-xqa-decode-sm110.patch` (XQA/TRTLLM decode) — are out of the
> stack: probe-gated sm_110 grants whose underlying kernels are
> arch-incompatible (B12x targets sm_120a–121f; XQA whitelist
> majors [9, 10, 12]) — measured INERT on Thor, no improvement over stock;
> per the no-improvement rule. Re-add trigger: a flashinfer release adding
> sm_110 B12x/XQA support (code preserved in git history +
> `task/nvfp4-thor/patchB-b12x-nvfp4.md` / `patchC-trtllm-xqa.md`).

`thor-gdn-prefill` and `thor-fa4-fp8kv` are diffed against the pristine
base (they touch files that no other patch touches), so their position in
`PATCH_ORDER` is immaterial.

Runtime check on Thor: `kernel_test_sm110_gates.py` (manual, `--gpus all`)
exercises each probe against the installed patched vllm and reports
**PASS** (kernel support found and the consuming gate/selection agrees) /
**INERT-OK** (probe correctly disabled — the *expected* state on the
current stack, not a failure) / **SKIP** (no CUDA device); it exits 0
unless a probe verdict disagrees with the gate or kernel selection that
consumes it. **T10** (the draft-CG gate's pure decision logic) needs no
GPU and runs unconditionally — without a device it is the only section
that executes (T6/T9 print SKIP).

## Thor sm_110 draft-CG shape gate (2026-09-19)

`thor-draft-cg-gate-sm110.patch` — purpose: fix the **c1 regression**
(V4 draft-cudagraph measured 7–16% *slower* at N=1, neutral at N=2, −5%
at N=4; study: `task/dspark-nc-cudagraph/cudagraph-c1-fix-study.md` Option
B). It does not touch capture: the draft graphs are still captured for
every bucket (the 49652 `_init_candidates` guard is undisturbed); only the
per-step *dispatch* decision changes.

- **Files touched:** `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`
  (only file — no other patch touches it, so its position in `PATCH_ORDER`
  is safe as long as it lands after `thor-dspark`).
- **Env knob:** `VLLM_THOR_DRAFT_CG_MIN_BATCH` — int, default **2**,
  minimum 1. `1` = CG for every batch size (today's behavior, for A/B).
  Invalid values → default 2 + one-time warning. Parsed lazily once.
- **Gate site:** the per-step draft dispatch in `propose`
  (`dispatch_cg_and_sync_dp(..., need_eager=...)`, dflash/speculator.py):
  `need_eager = is_profile or _thor_draft_cg_gate_wants_eager(num_reqs)`.
  Below the threshold the existing `NONE`-descriptor path runs (the same
  eager fallback the draft already uses for profile runs / no-CG-support
  — no new eager code). `is_profile` short-circuits, so the gate never
  runs during capture.
- **THOR log lines** (one-shot each, `logger.info`):
  `THOR draft-CG gate: active (sm_110, min_batch=<t>)` /
  `THOR draft-CG gate: inactive (arch <maj>.<min>)` (first evaluation);
  `THOR draft-CG gate: N=<n> -> eager (<n> < min_batch)` (first eager
  dispatch); `THOR draft-CG gate: N=<n> -> cudagraph` (first CG dispatch);
  `THOR draft-CG gate: error -> defaulting to cudagraph (<exc>)` (on any
  gate evaluation error — logged once, then today's CG behavior).
- **Sentinel:** `THOR_DRAFT_CG_GATE` (helper docstring + dispatch-site
  comment); `verify_patches.py` additionally asserts the dispatch call
  site (a partial apply that lands the helper but never routes through it
  must fail the build).
- **Status:** probe-gated, inert on non-sm_110 (the helper is a structural
  no-op there), default N=1→eager.

## Thor sm_110 GDN prefill enablement (flashinfer side) (2026-09-21)

`thor-gdn-prefill-fi-sm110.patch` — a **one-line hunk in the installed
flashinfer package** (a different site-packages tree from vllm) that
widens the GDN prefill dispatch allowlist so the SM100 CuTe-DSL kernel is
reachable on sm_110. This is the *enablement* half of the GDN prefill
probe; the vLLM-side grant (`thor-gdn-prefill-sm110`) stays exactly as
before (probe-gated), so the hunk alone changes no behavior on any
device.

- **The wall it opens** (`flashinfer/gdn_prefill.py`, flashinfer
  0.6.18.post1): `chunk_gated_delta_rule`'s non-CP dispatch is
  `if _arch_major == 10 / elif 12 / elif 9 / else` (L460/L519/L546) —
  major 11 falls into the final `else` and raises
  `NotImplementedError("GDN prefill DSL kernel is unavailable")` (L576).
  The kernel behind the SM100 branch (`gdn_kernels/blackwell/
  gated_delta_net_chunked.py`) is **device-JIT-compiled CuTe-DSL**
  (FA4-type wall, not the B12x type): its `cute.compile` carries no
  `GPUArch` option, so the DSL targets the current device; its single
  sm_100-specific construct is the `arch = "sm_100"` class attribute,
  used only as a TMEM-column sizing lookup
  (`cute.arch.get_max_tmem_alloc_cols`, L193/L280). The CP kernels by
  contrast are arch-pinned (the SM100 CP prefill hard-rejects
  `major != 10`; the SM90/SM120 delta-rule DSL kernels pin
  `GPUArch("sm_90a")` / `sm_12xa`), which is why the CP heuristic
  (`_arch_major in (9, 10, 12)`, L359) is deliberately **not** widened —
  sm_110 simply never routes to CP and falls through to the non-CP
  branch.
- **The hunk** (`gdn_prefill.py:460`, one line, sentinel comment on the
  changed line):
  `if _arch_major == 10:` →
  `if _arch_major in (10, 11):  # THOR_SM110_GDN_ALLOWLIST`.
- **Mechanism change** (`apply_patches.py` / `verify_patches.py`):
  `FLASHINFER_PATCH_ORDER` applies the hunk to the installed flashinfer
  package (`flashinfer_package_root()`, `importlib`-resolved, same
  `patch -p1 --forward` semantics as the vllm root);
  `verify_patches.py` gains a parallel `FLASHINFER_FIXES` table checked
  by `verify_flashinfer()`. Both scripts take an optional explicit root
  (vllm root arg 1, flashinfer root arg 2) — the verdict is about the
  given trees, matching the existing explicit-root convention. The
  Dockerfile is unchanged: it already runs both scripts against the
  installed packages.
- **Verification:** no-GPU fresh-tree gate (apply 12+1, verify 12+1
  fixes, py_compile, import smoke, canary T10) and the live T6 canary —
  see `task/nvfp4-thor/gdn-prefill-fi-enable.md`.

## Thor hd256 1CTA decode carve-out (v11, 2026-09-24)

`thor-fa4-hd256-1cta-decode-sm110.patch` — the "N-2 decode" workstream: a
decode carve-out in the FA4 hd256 kernel's interface that drops the 2CTA
cluster to a single CTA for decode-shaped calls. One hunk in
`vllm/vllm_flash_attn/cute/interface.py` (`_flash_attn_fwd`, the
`hd256_use_2cta` site); the kernel file is untouched. v11 = v10 + this patch
only — the live tree's inert SplitKV kernel plumbing is deliberately not
shipped (see below).

- **What it does.** The kernel's `use_2cta` flag (a compile-key slot, so the
  two cluster forms are separate cached binaries) is now driven by an OR'd
  decode condition:

      # decode-shaped hd256 calls (small M) take the 1CTA form
      hd256_decode_1cta = (
          max_seqlen_q is not None
          and max_seqlen_q <= 8                       # decode M (1..8); prefill is 1024+
          and not local
          and (page_size in (None, tile_n))           # dense or paged-128 TMA (only legal hd256 paged mode)
      )

  i.e. decode M (1..8 — covers the MTP verify batch; prefill is 1024+),
  non-local, and dense or paged-128 TMA (the hd256 kernel's only legal
  paged mode — its own `paged_kv_non_tma` assert makes page-128-or-dense
  the only legal KV layout, so the last term is a defensive tautology).
  The 1CTA form (cluster (1,1), `CtaGroup.ONE`, shallower k/v rings
  2/3 vs 4/4) has roughly half the 2CTA per-KV-token scan slope (0.016 vs
  0.032 µs/tok on the 20-SM Thor), so the win grows with L. Prefill
  (2CTA) and decode (1CTA) coexist in one process under separate compile
  keys; a serving-like prefill→decode compile sequence was verified to
  route decode to the 1CTA binary (compile key `(varlen_b1, l2_swizzle,
  mask_residual, use_2cta) = (True, False, True, False)`). Like the rest
  of the fa4 group this only runs on the FA4 hd256 path, which is still
  passive-by-default on sm_110 (default FA version 2), so no existing
  config changes behavior.
- **Evidence (clean micro-bench, server 0/0-gated, 2026-09-24).** L=8192,
  M=1: 2CTA 323.0 µs → 1CTA 207.1 µs median — a **1.56× decode win**
  (M=4: 1.55×); the ratio grows 1.20× (L=256) → 1.56× (L=8192) for M=1.
  The clean 2CTA number reproduces the prior clean 2CTA (324.1 µs) to
  0.3%. Correctness: full-FP8 A/B 12/12 (probe2a-fp8 — all prefill shapes
  stay on the unchanged 2CTA path, all decode shapes run the 1CTA binary);
  1CTA == 2CTA == fp32-ref within fp8 tolerance (1CTA-vs-2CTA max diff
  9.77e-4 @ L=2048 / 4.88e-4 @ L=8192 vs tolerances 1.52e-2 / 1.35e-2 —
  both cluster forms equally correct; the residual is fp8/softmax
  numerics, not a cluster-form difference).
- **Status.** Built into the v11 image (`mjolnir/vllm-thor:
  qwen38-sm110-v11`, same pinned base digest, no Dockerfile change). Full
  v11 flow simulation on the expanded v10 site-packages: all v10 patches
  idempotently skipped, this patch applied cleanly, gate **`all 11 fixes
  verified`** (+ `all 1 flashinfer fixes verified`), and the resulting
  `interface.py` is **byte-identical** to the live 1CTA tree the clean
  micro-bench measured (`cmp` clean). No `verify_patches.py` sentinel was
  added — the hunk re-wraps an existing expression (adding only the
  `hd256_decode_1cta` local), and the gate verifies the existing fixes,
  which all remain present.
- **Residual gap.** 1CTA is still ~2.9× FlashInfer at L=8192, M=1 (207.1
  µs vs the isolated-clean FI reference 72.5 µs; 2.25× in-window) — the
  remaining lever is a FlashInfer-inspired GEMV-style decode kernel (the
  active next task in `AGENTS.md`), not more cluster/split tuning.

### hd256 SplitKV — investigated, no win (kept inert)

The hd256 kernel's SplitKV machinery was benchmarked on **both** cluster
forms and found **not** to be a performance win, so it is kept inert in
the tree: the v11 patch ships the 1CTA carve-out only (the live tree's
A1a/A1b kernel plumbing — `num_splits` param, per-split gO/LSE staging,
FP8-descale-correct LSE, test-only `A1B_TEST_NUM_SPLITS` knob — is
deliberately excluded from v11, and the shipped interface never enables
`num_splits > 1` on hd256; the machinery stays in the live tree as a
correctness-verified harness for future kernel work):

- **2CTA path** (2026-09-23): ns>1
  is slower at every L — 40/40 cells (L∈{256…8192} × M∈{1,4} ×
  ns∈{2,4,8,16}) above serial; the best case (ns=2, L=8192) is still
  1.24× *slower*, and the per-split penalty shows no amortization knee as
  L grows.
- **1CTA path** (2026-09-24):
  real SplitKV was enabled on the 1CTA path (num_splits clamp lifted,
  per-ns compile key, `num_splits` as a constructor constant — a trailing
  FFI argument breaks CuTe DSL IR verification) and verified correct
  (every leg within fp8 tolerance; per-split LSE + combine match the fp32
  logsumexp to ~1e-6; empty-split paths exercised) — but still no win:
  clear loss at L=2048/4096 (ns=2 +18.6% / +6.4%), and at L=8192 at most a
  marginal ns=2 "win" inside the inter-run noise band (−1.5…−2.4% across
  four measurements), with ns=4/ns=8 clear losses.
- **Why (confirmed on both paths):** the per-CTA 512-tmem/MMA machinery
  cost is replicated ns× over shrinking KV chunks — splitting trades
  per-KV-token work for ns× the fixed per-CTA cost, so it never pays on
  either cluster form. The shipped decode optimization is the 1CTA
  carve-out (this section); the real long-context SplitKV win is the
  FlashInfer path (1.78–2.8× faster than FA4 1CTA serial at L=8192),
  which is also the reference architecture for the next GEMV-style kernel.
  — *superseded as the decode answer first by the (now archived) GEMV kernel,
  then by the Mjolnir kernel (`--attention-backend MJOLNIR`).*

## fp6 e2m3 KV cache — `patches/thor-fp6-e2m3-kv-cache.patch`

**What:** the dtype name and quant mode vLLM needs to allocate and validate a
6-bit KV cache; everything that touches the bytes lives in `mjolnir-kernel`
(`mjolnir/kvfp6.py`, `mjolnir/kvscales.py`, `mjolnir/backend.py`):

- **page size** — `MjolnirAttentionBackend.customize_spec` sets
  `state_content_bytes = 2 · d · 6/8` (384 B per (head, token) at d=256, vs
  512 for fp8), so the allocator, the hybrid (GDN) page alignment and the
  memory budget see the real size;
- **write** — `do_kv_cache_update` → one Triton kernel: `x / scale`, round to
  nearest e2m3 (saturating), pack 4 codes into 3 bytes, scatter by slot;
- **decode** — T5 reads the packed codes (`kv_format="e2m3"`);
- **prefill / mixed** — FA4 reads e4m3 only, so the step's pages are expanded
  to an e4m3 scratch (exact) and FA4 runs on a copy of its metadata pointing
  at it; cascade attention is off for fp6;
- **scales** — static per (layer, KV head), `amax / 7.5`, from a
  `mjolnir calibrate` file (`MJOLNIR_ATTN_KV_SCALES`), installed as the
  layers' `_k_scale` / `_v_scale` (per head), which the write, T5 and FA4
  all read. No file → the server refuses to start with fp6.

**Sentinel:** `verify_patches.py` checks the dtype literal, the storage map,
`FP6_E2M3 = 11` and the mapping. **Build smoke:** the `Dockerfile` asserts
`get_kv_quant_mode("fp6_e2m3") == KVQuantMode.FP6_E2M3` and imports the
kernel package's fp6 modules. **Tests:** `tests/kernel/test_runner.py` (the
write vs the kernel package's quantizer and packer, the expansion vs a
decode, T5 on an fp6 cache vs the emulator) and `smoke_vllm.py --kv fp6_e2m3`
(`mjolnir verify vllm`, mode `e2m3`).

## fp6 prefill direct-read — `patches/fi-fp6-kv-ampere-prefill.patch` (2026-10-27)

**What:** FlashInfer's Ampere FA2 prefill learns to read the packed-fp6
(e2m3) KV cache **directly — no fp8 scratch expansion**. Research verdict:
sm_87 has no fp8 tensor cores,
so FlashInfer already dequantizes quantized KV in-kernel on Ampere ("essentially
free", its own PR #4746) — and NVFP4 KV (software unpack) runs there at ≈fp8
speed after #4769. The fp6 path is the same machinery with a different unpack.

**Mechanics (all `if constexpr`-guarded, inert for every existing dtype):**

- `vec_dtypes.cuh`: the 1-byte `mjolnir_fp6_e2m3_sv` element + `vec_cast`
  specializations for `half`/`bf16` that **delegate to the e4m3 conversion** —
  post-expansion the smem tile *is* fp8, so `compute_qk`/`compute_sfm_v` are
  untouched (no per-fragment changes, no new compute paths).
- `cp_async.cuh` / `permuted_smem.cuh`: `pred_load_128b_src<12>` — a 16-byte
  slot cp.async reading only the first 12 bytes (PTX zero-fills the rest; the
  runtime src-size operand already exists in `pred_load_128b`). A slot holds
  16 codes = 12 packed bytes; gmem rows advance 0.75× the fp8 bytes.
- `prefill.cuh`: fp6 branches in `produce_kv` / `page_produce_kv` /
  `page_produce_kv_on_the_fly` (both swizzle modes; the per-j gmem scalar is
  `6·upcast_size` — 96 bytes per 128 codes, the fp4 `4·16` pattern's 0.75×
  sibling), and `expand_fp6_tile_to_fp8` — a flat thread-strided pass at the
  fp4-repack hook: each slot's 12 bytes (4 codes per 3 bytes, LSB-first — the
  kvfp6 row layout) expand to 16 e4m3 bytes `code · s_channel` via the native
  `cuda_fp6.h` conversion and the per-(kv_head, channel) float scales from two
  new optional params (`maybe_{k,v}_fp6_scales`, threaded through
  `modules.py`'s `additional_tensor_names` like the NVFP4 `maybe_*_cache_sf`).
  The shared-K/V time-slicing and both tile shapes are covered by the
  in-place expansion (the slot bytes are rewritten before any fragment read).
- `jit/utils.py`: the `mjolnir_fp6_e2m3` → `mjolnir_fp6_e2m3_sv` dtype string
  (no torch dtype exists for 6-bit float; the packed pool rides uint8 bytes).

**Numerics:** identical to the validated fp8-scratch path — the scratch holds
`e4m3(code · s_c)` with scales at 1; the kernel now computes the same bytes
per tile. **Fragility:** the per-j gmem scalars and the j-loop bound are the
bump-fragile parts — re-derive `upcast_size` and the fp4/fp8 scalars from the
new base's `produce_kv` before re-adapting. **Validation status: patch
applies to the pinned tree, the headers + the fp6 expand instantiation
compile for sm_87 (nvcc, CPU-side); runtime canary + the vLLM-side
`MJOLNIR_ATTN_FP6_DIRECT` wiring are pending.**

**Sentinel:** `verify_patches.py` checks the fp6 struct in `vec_dtypes.cuh`,
the partial-src loader in `cp_async.cuh`, and — the "used, not just defined"
sentinel — the `expand_fp6_tile_to_fp8<KTraits, KTraits::HEAD_DIM_QK>` call
site in `prefill.cuh` (a partial apply that lands the helper without the
compute-side call would compile and silently no-op).

## The Mjolnir attention backend — `patches/thor-mjolnir-attention-backend.patch`

**What:** one enum member. `vllm serve` parses `--attention-backend` with
`AttentionBackendEnum[value.upper()]` in the API-server process, before any
plugin code runs, and `Attention.__init__` maps the selected class back with
`AttentionBackendEnum[attn_backend.get_name()]` — so a third-party backend can
only use the reserved `CUSTOM` name unless the enum has a member for it. The
patch adds `MJOLNIR`, whose value is the class path in the `mjolnir-kernel`
package; `MjolnirAttentionBackend.get_name()` returns `"MJOLNIR"` when the
member exists and `"CUSTOM"` otherwise.

**And the FA4 fallback default** (`v1/attention/backends/fa_utils.py`). The
backend runs prefill, mixed and cascade steps on a stock backend it subclasses.
FlashInfer's prefill on Thor is ~40% slower than FlashAttention 4's (the GEMV
image ran FA4 prefill: 3,557 vs 2,559 t/s at 8K), but on sm_110
`get_flash_attn_version()` resolves to FA2 unless the config says
`flash_attn_version: 4`, and FA2 cannot read an fp8 KV cache. The hunk adds
`mjolnir_fallback_name()` (`MJOLNIR_ATTN_FALLBACK`, default `FLASH_ATTN`) — the
backend reads its default from it — and one `elif` in step 2 of
`get_flash_attn_version()`: no explicit `flash_attn_version`, the selected
backend is `MJOLNIR` (or `CUSTOM`, the plugin's slot), the fallback is
`FLASH_ATTN`, and the device is sm_110 → FA4. Decided from the config and the
environment, so the API server, engine core and every worker agree. The FA4
FP8-KV probes (`thor-fa4-fp8kv-sm110`, `thor-fa4-hd256-fp8-sm110`) still gate
the layer; if they fail, the backend logs it and falls back to FlashInfer.
An explicit `flash_attn_version` always wins; nothing changes off sm_110 or
for any other backend.

**Apply to a vLLM 0.32.0 checkout** (after the FA4 patches, which also touch
`fa_utils.py`): `git apply docker/vllm-thor/patches/thor-mjolnir-attention-backend.patch`
(paths are `vllm/...`; `apply_patches.py` applies it to the installed package
with `patch -p1 --forward`). **Sentinel:** `verify_patches.py` checks the
member line, `mjolnir_fallback_name` and the `elif` call site. **Build smoke:** the `Dockerfile` asserts
`AttentionBackendEnum["MJOLNIR"].get_path() == mjolnir.BACKEND_PATH` and that
the `mjolnir` `vllm.general_plugins` entry point is installed.

**Bump fragility:** low — the registry hunk's context is the last three
members before `CUSTOM` (a new upstream member there moves one line); the
`fa_utils.py` hunk sits on the FA4 patches' version of that file, next to the
`flash_attn_version` override in `get_flash_attn_version()`, so it is
re-generated whenever those patches are.

## Thor MoE backend gates on sm_110 (2026-10-22)

Three upstream fixes, hand-adapted onto v0.32.0 — the two below prevent
faults when an NVFP4/MXFP4 MoE model is served on Thor; the third (a sampler
fix, not MoE-related) closes the batch out. Files are disjoint from every
other patch, so their `PATCH_ORDER` position is immaterial.

- **`thor-sm110-mxfp4-moe-gate.patch`** —
  [#62133](https://github.com/vllm-project/vllm/pull/62133) "Fix OAI Triton
  MXFP4 MoE on SM110" (ready-labeled; supersedes the closed #61387). #58877
  (merged 2026-10-06, in v0.32.0) widened the OAI-Triton device gate to
  `< (13, 0)`, admitting SM110 → `matmul_ogs` compile failure at engine init
  (upstream issue #61374). The final state keeps the device gate (TRITON
  stays explicitly selectable) but: the GPT-OSS MXFP4 **selector**
  (`_get_priority_backends_for_gpt_oss` in `oracle/mxfp4.py`) keeps **MARLIN**
  as the default on SM110 (TRITON opt-in), and `_swizzle_mxfp4`
  (`quantization/utils/mxfp4_utils.py`) applies the `is_persistent` +
  `epilogue_subtile=1` constraints on SM110 too (the SWAP_XW
  subtiled-epilogue codegen bug); the `_triton_kernel_moe_supports_current_device`
  comment in `experts/gpt_oss_triton_kernels_moe.py` is updated to match
  (comment-only hunk). **Drop when upstream merges #62133.**
- **`thor-sm110-cutlass-fp4-moe-guard.patch`** —
  [#62275](https://github.com/vllm-project/vllm/pull/62275) "Disable CUTLASS
  FP4 MoE backend on sm_110". NVFP4 MoE auto-select picks `VLLM_CUTLASS` on
  Thor and faults in `profile_run` (TMA descriptor init), validated upstream
  on Thor with Qwen3.8-Flash-Next-NVFP4. The fix removes sm_110 from
  `CutlassMoEMethod._supports_current_device`'s family allowlist
  (`experts/cutlass_moe.py`) so the fallback backend is chosen. **Drop when
  upstream merges #62275.**

**Sentinels:** `THOR_SM110_MXFP4_GATE` (selector condition in `mxfp4.py`) +
the `for family in (100, 110, 120)` widening (`mxfp4_utils.py`);
`THOR_SM110_CUTLASS_FP4_GUARD` + the collapsed single-line condition +
an absence guard on `is_device_capability_family(110)` in `cutlass_moe.py`.

**Adaptation drift:** none of substance — all six hunks apply to the
v0.32.0 text verbatim; only line offsets differ (`mxfp4.py` −11,
`mxfp4_utils.py` −17, `topk_topp_triton.py` −9; `gpt_oss_triton_kernels_moe.py`
and `cutlass_moe.py` land at their exact upstream lines). The `tests/` hunks
are dropped (the wheel ships no test dir). Verified on a pristine
`vllm/vllm-openai:v0.32.0-ubuntu2404` tree: full stack applies, all fixes
verify (no GPU).

**`thor-spec-decode-split-topp.patch`** —
[#58231](https://github.com/vllm-project/vllm/pull/58231) "[Bugfix][Sampler]
Fix split top-p masking every token for huge logits (thinking_token_budget +
spec decode emits token 0)" (merged 2026-10-19 — first release after 0.32.0,
so 0.32.0 still carries the bug). The split top-p mask kernel
(`_topp_sb_mask_kernel` in `v1/sample/ops/topk_topp_triton.py` — CUDA,
batches ≤ 64 rows, rows with `top_p < 1` and no active top-k) found its
pivot in probability space but masked in logit space:
`pivot_logit = log(pivot) + log(Z) + M` rounds to `M` in fp32 once a forced
logit makes `|M|` huge (fp32 spacing at 1e9 is 64), every token fails the
strict `x > pivot_logit`, the row goes all `-inf`, and sampling emits token
0 forever (V1 rejection sampler: softmax NaN → draft rejected →
`recovered_id = 0`; MRV2 verifier: argmax of an all `-inf` row = 0), with a
≈3× throughput loss on top. **Trigger: a huge forced logit —
thinking-token-budget forcing (writes 1e9 to the end-of-thinking token) is
the known producer; plain `top_p < 1` + spec decode does NOT suffice**
(ordinary logits keep the log-space round-trip exact to ~1e-6), and greedy /
`top_p = 1.0` are unaffected. Our serving configs (`top_p: 0.95`, MTP=5) sit
exactly on the vulnerable path — spec decode routes its verify rows through
`apply_top_k_top_p` (non-spec stays on the fused FlashInfer sampler unless
processed logprobs is on or FlashInfer is unavailable) — and
`thinking_token_budget` is a **request-level** param in 0.32.0 (validated in
`sampling_params.py`, no serving-config key needed), so any client can arm
it against our server. The fix masks in probability space
(`exp(x − M)/Z > pivot`, bitwise the step kernels' predicate; `x == M` kept
as a backstop) and changes nothing for ordinary logits (upstream: 24/24
randomized rows identical before/after). Sentinel: `THOR_SPEC_SPLIT_TOPP`.
**Drop when upstream merges #58231 — it is merged 2026-10-19, first release
after 0.32.0.**

## Open upstream watch list (assessed 2026-10-28, base v0.32.0)

Status of every open PR relevant to Thor after the #62133 merge — what to
pre-patch (nothing), what to take at the next bump (both images inherit it,
so the e2e attention delta stays the only variable), and what to keep out of
the image on purpose:

- **Already carried here:** #62275 (the cutlass-fp4-moe-guard patch —
  essential for the Qwen3.6-35B-A3B NVFP4 MoE) and #54165 (the
  mamba-spec-kv-restore patch; #54163 took its gating half — see its row).
- **Take at the next bump, not before:** #53337/#44590 (Thor-tuned Triton
  `selective_state_update` — our image serves the hybrid model on the fused
  CUDA GDN decode kernel, so one-sided adoption would change nothing for our
  configs while contaminating the e2e comparison) and #60566 (screened
  lm_head — MRV2, explicitly targets Thor, +7.7% greedy on Thor-class hw,
  but a moving target, mutually exclusive with `MJOLNIR_LM_HEAD=fp8`
  (a quantized head raises at load), and default-on would inflate our e2e;
  benchmark it against LMH8 at bump time instead).
- **Keep out on purpose:** #55366 (SM110 FA4 auto-selection — changes the
  stock default our official-image baselines must reflect; add a new
  official leg when it lands), #56084/#56102 (cooperative-topk — QSA-model
  path, not our models), #53520 (prompt_logprobs corruption — V1-runner
  only; our image runs MRV2 and no bench reads prompt_logprobs),
  #54705 (fp8-KV scales — SM100-scoped; our kernel owns sm_110 fp8-KV),
  #53383 (ROCm), #58868 (mmap fault-in — startup-time only).
- **Closed unmerged — our patches stay:** #50885, #49652, #55519, #56289.
