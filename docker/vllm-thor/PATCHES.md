# vLLM Thor PR Patches

Twenty-two vLLM-side patches (**22** sentinels green on the pinned base
`vllm/vllm-openai@sha256:135d65d5…`; the leaned production set — the run
record: `docs/inkling2-thor-run.md`), **hand-adapted** onto
more are retired-on-file keepers for the GDN retest), **hand-adapted** onto
the vLLM base (not a raw `git apply` of the upstream PR diffs — the two
originals because the base code has diverged from each PR's base, the
Thor-authored ones to add capabilities the stock backends don't have), plus
the **Mjolnir attention-backend name** (`AttentionBackendEnum.MJOLNIR` — see
"The Mjolnir attention backend" below) and one **flashinfer-side** fp6 hunk
(see "fp6 prefill direct-read" below). In total: **22 vLLM-side
patches** (incl. the CuTe DSL NVFP4 MoE sm_110 pair below and the
**Qwen3.8-Flash-Next (qwen4_exp) stack**: env-gated hooks into the
`mjolnir.qwen4exp` modules — see "Qwen3.8-Flash-Next
(qwen4_exp) serving stack" below), applied by `apply_patches.py` to the installed vllm package
**and** the installed flashinfer package (two more patches there, plus two
retired-and-commented); verified by `verify_patches.py`.

The retired GEMV decode kernel patch (`thor-fa4-hd256-gemv-decode-sm110.patch`)
and its kernel were removed when the Mjolnir kernel replaced them; the last
snapshot is the git tag `gemv-archive`: [`gemv-archive:deprecated/gemv/`](https://github.com/GoBrrr-ai/mjolnir/tree/gemv-archive/deprecated/gemv).

**Retired at the v0.33.0 bump (2026-11-03)** — all four are in the base now:

- `54165-mamba-spec-kv-restore.patch` — #54163 (the gating half) is in the
  release AND the 0.33.0 scheduler keys the mamba-align backoff and the
  trailing-block drop on the gated `use_eagle_block_drop()`
  (= `needs_eagle_block_drop() and not disable_eagle_block_drop`), which is
  this patch's exact semantics for DFlash/DSpark; #61589's lookahead block
  hashes (on by default) remove the MTP last-block drop altogether.
- `55519-no-warn-when-block-drop-off.patch` — the false-positive warning is
  gone from the base (`_warn_if_unannotated_eagle_mamba` moved to
  `kv_cache_planning.py` and no longer claims reuse is disabled), and the
  scheduler's "drop disabled" warning is already gated on
  `needs_eagle_block_drop()` (no false positive).
- `thor-fp6-e2m3-kv-cache.patch` — vLLM #52670 (pluggable KV cache data
  types) is in the base: `register_kv_cache_dtype` + `KVQuantMode.CUSTOM`
  are exactly this patch's four edits as an API. `mjolnir.kvdtype`
  registers the dtype from the plugin entry point on any base that has the
  API (the Dockerfile smoke asserts the registration).
- `thor-spec-decode-split-topp.patch` — #58231 merged 2026-10-19 is in the
  base (the probability-space pivot, byte-identical predicate).

**Retired at the v0.34.0 bump (2026-11-16)** — all four are in the base now:

- `49652-draft-decode-capture.patch` — the 0.34 speculator rework makes the
  draft-decode graph shapes explicit: `AutoregressiveSpecimator.init_cudagraph_manager`
  constructs the draft decode manager with `decode_query_lens={1:
  max_num_reqs}` (fixed query length 1 — this patch's exact semantics) plus
  a draft-specific `get_draft_decode_cudagraph_request_sizes()`; the
  Dynamic-SD candidate expansion moved out of `CudaGraphManager
  ._init_candidates` into the opt-in `specialize_spec_tokens` capture path.
  The old guard's hunk target no longer exists.
- `thor-sm110-mxfp4-moe-gate.patch` — #62133 is in the release: the GPT-OSS
  MXFP4 selector carries the SM110 clause ("Jetson Thor (SM110): MARLIN
  stays the default", `oracle/mxfp4.py`) and `_swizzle_mxfp4` applies
  `is_persistent` + `epilogue_subtile=1` for families (100, 110, 120).
- `thor-sm110-cutlass-fp4-moe-guard.patch` — #62275 is in the release:
  `CutlassMoEMethod._supports_current_device` allows families (100, 120)
  with the "SM110 excluded: grouped NVFP4 kernel is sm100-only" comment.
- `thor-qwen4-topk-sm110.patch` — #56102 is in the release:
  `qsa_indexer.py`'s `use_cooperative_topk` carries
  `not is_device_capability_family(110)` (the #47266-class family guard).

Base this was verified against: **vllm 0.34.0** (`vllm/vllm-openai:v0.34.0`,
tag `62d4bc3c7`), FlashInfer 0.7.3, torch 2.14.0+cu130, vllm-flash-attn
pin `d8dca8f06c74`. The full stack applies on the pristine tag and every
`verify_patches` sentinel passes (no-GPU gate, 2026-11-16); the image build
awaits the release image on the hub.

**Retired at the v0.35.0rc0.dev nightly bump (2026-11-22)** — all three are
in the nightly now (image self-version `0.34.0rc1.dev1133+g647a76866`;
`647a7686`):

- `64739-safetensors-index-subdir.patch` — base-covered: both hunks skip as
  already applied on the nightly wheel (the loader finds index-listed
  subdirectory shards stock).
- `thor-fa4-hd256-fp8-sm110.patch` — the hd256 forward grew a **native
  `DescaleTensors` path** (`flash_fwd_sm100.py`: `q/k/v_descale`,
  `_load_effective_descales`, per-(batch, kv-head) descales in the softmax —
  the #2454-class descale work, landed) and lost the dedicated
  `sm100_hd256_2cta_fmha_forward.py` entirely; the hand descale port and the
  `_fa4_hd256_fallback_reason` grant have nothing to anchor on. The
  head-size-blind FP8 grant in `thor-fa4-fp8kv-sm110` covers hd256 too (its
  probe exercises the hd128 shape — the hd256+FP8 combination is validated
  by the e2e legs, not a one-shot probe).
- `thor-fa4-hd256-1cta-decode-sm110.patch` — `hd256_use_2cta` is gone from
  `interface.py`; the heuristics moved to `config.py`
  (`default_pack_gqa`/`prefer_hd256_2cta`) and already gate on
  `device_capacity in (10, 11)`. Re-tune on-device at the 0.35 bump if the
  1.56× decode-shaped 1CTA win reproduces on the new kernel generation.

Base this was verified against: the **2026-11-22 nightly**
(`vllm/vllm-openai:nightly-647a7686…`, arm64 leaf `135d65d5db702…`; vLLM
`0.34.0rc1.dev1133+g647a76866`, FlashInfer **0.7.4**, torch 2.14.0+cu130).
The full stack applies and every `verify_patches` sentinel passes
(no-GPU gate, 2026-11-23 post-lean; 22 vLLM + 2 FlashInfer fixes).

| File | PR | What it fixes |
|------|----|---------------|
| `50885-flashinfer-full-cudagraph.patch` | [#50885](https://github.com/vllm-project/vllm/pull/50885) (closed unmerged) | Capture **FULL** decode cudagraphs for spec-decode on the FlashInfer **native** (non-trtllm) path — uniform multi-token verify batches. **0.33.0 slim-down:** the base now carries the wrapper plumbing itself (`_get_decode_wrapper(batch_size, use_cudagraph, q_len_per_req)`, the tuple-keyed `_decode_wrappers_cudagraph`, `fast_plan_decode` threading `q_len_per_req` into FlashInfer's `fast_decode_plan` — which since 0.7.2 plans uniform multi-token queries on the fa2 tensor-core backend). What remains Thor-side is the *policy*: the `flashinfer_supports_uniform_multi_token_decode()` probe (does the installed FI accept `q_len_per_req`), the `supports_spec_as_decode` OR-term and the `get_cudagraph_support` UNIFORM_BATCH grant (both gated on the probe + non-trtllm + causal) and the `build()` per-request `num_input_tokens`/`q_len_per_req` computation (previously fa2-native-only, pre-SM90). Without it sm_110 spec decode still falls to PIECEWISE graphs. Unchanged on 0.34 (`get_cudagraph_support` is byte-identical 0.33→0.34). |
| `thor-fused-draft-decode-fi-native-update.patch` | — (Thor-authored) | **Implement `update_draft_decode_metadata` for the FlashInfer backend** so the spec-decode loop can use the **fused multi-step draft decode** instead of the "rebuild attention metadata between draft steps" fallback. The upstream contract is a `pass` stub for trtllm-gen (0.33.0: one trtllm-gen decode build serves every step); the patch ORs `supports_draft_decode_metadata_update` with the FI-native full-cudagraph, single-rank, non-DCP, non-trtllm case and replaces the stub with `_advance_fi_decode_plan_kernel`, which re-derives the FI decode plan (`paged_kv_indptr`/`last_page_len`/`page_indices`) **on the GPU** from the advancing `seq_lens` + block table — the same formula as `build()`, but capture-safe (no host sync, no D2D index copy) so it replays correctly inside the draft CUDA graph. Pays off only at **K ≥ 3** (MTP3 / DSpark K≥2); a deliberate no-op at K=2. |
| `thor-fused-draft-t5-plan-refresh.patch` | — (Thor-authored) | **Rebuild the draft attention metadata on the fused FULL-graph replay path** so host-side per-cycle plan state survives graph replay. `_fused_multi_step_decode` returns before `_build_uniform_attn_metadata` when `cg_mode == FULL`, so any builder that computes per-cycle host state in `build()` — Mjolnir's T5 split-KV **work plan** (`work_buf`/`nsplit_buf`, host-computed from seq_lens) — stays **baked at capture-time seq_lens**: every replayed draft step attends a truncated context and MTP acceptance collapses beyond position 0 (measured 2026-10-18: K=5 acceptance 4.1→**2.04**; eager or `MJOLNIR_ATTN_FUSED_DRAFT=0` healthy). The fix re-runs the build before `run_fullgraph` — the returned metadata is unused (the graph replays baked launches), but the side effect refreshes the persistent plan buffers the graph reads. |
| `thor-dspark-draft-fi-noncausal-cudagraph.patch` | — (Thor-authored) | **Capture the DSpark non-causal multi-token draft under a FULL CUDA graph** instead of running the draft eagerly. A non-causal drafter (DSpark) routes its draft through the FI *prefill* wrapper — the FI *decode* wrapper cannot express non-causal multi-token (its fa2 kernel is block-causal only for `q_len_per_req > 1`). In cuda-graph mode that prefill wrapper freezes its batch size at creation and reads its KV plan from **persistent** buffers, so one wrapper per captured batch size is needed (`_noncausal_prefill_wrappers_cudagraph`, mirroring `_decode_wrappers_cudagraph`); the draft plans eagerly outside the captured graph and the captured `run()` replays safely across ragged re-plans. Adds the `_get_noncausal_prefill_wrapper_cudagraph` factory, grants a non-causal `UNIFORM_BATCH` in `get_cudagraph_support` (gated on a multi-token drafter, non-SM12x, single-rank DCP), and dispatches the prefill batch to the cudagraph wrapper in `build()`. Lands **after** 50885 (its `get_cudagraph_support` hunk is diffed against 50885's grant). Most bump-fragile patch — see below. |
| `thor-draft-cg-gate-sm110.patch` | — (Thor-authored) | **Shape-gate the DSpark draft's FULL cudagraph replay on sm_110 (the c1 fix).** The forensics showed the graphed draft is 7–16% *slower* than the eager draft at N=1 (replay overhead eats the launch savings on a tiny 5-layer draft). The gate routes per-step draft batches below `VLLM_THOR_DRAFT_CG_MIN_BATCH` requests (default **2**; min 1, where 1 = today's behavior) through the **existing eager fallback** via the dispatch's `need_eager` channel — no new eager path, no capture-logic changes. sm_110 only: off sm_110 the decision helper is a structural no-op (always CG). One-shot `THOR draft-CG gate: …` log lines; any gate evaluation error degrades to the CG behavior (logged once). Probe-gated, inert on non-sm_110, default N=1→eager. |
| `thor-fa4-hd256-1cta-decode-sm110.patch` | — (Thor-authored) | **RETIRED at the 0.35.0rc0.dev nightly (see above).** The 1CTA decode carve-out for the FA4 hd256 kernel (the v11 "N-2 decode" workstream).** Decode-shaped hd256 calls — `max_seqlen_q <= 8` (decode M 1..8, incl. MTP verify; prefill is 1024+), non-local, dense or paged-128 TMA (the kernel's only legal paged mode) — now OR into the `hd256_use_2cta` carve-out in `vllm_flash_attn/cute/interface.py` (one hunk), so the kernel drops its 2CTA cluster to a single CTA (`cluster (1,1)`, `CtaGroup.ONE`): a **1.56× decode win** at L=8192 M=1 (clean micro-bench 2CTA 323 µs → 1CTA 207 µs), with 1CTA == 2CTA == fp32-ref within fp8 tolerance; the kernel file is untouched and the 2CTA prefill path is unchanged (separate compile keys coexist in one process). |
| `thor-inkling2-sm110-conv-mix.patch` | — (Thor-authored) | **Inkling 2 Mini on sm_110.** The nightly wheel's prebuilt inkling2 kernels (`csrc/inkling2`: `conv_mix`, `allreduce_conv_mix`, `seq_parallel`) carry no sm_110 cubins — CMake's `INKLING2_KERNEL_ARCHS` is `9.0a;10.0f;10.3f` only (upstream #69167 open) — so the first `torch.ops._C.inkling2_conv_mix` launch dies with "no kernel image … for the device" at the memory-profiling run. The patch routes the TP1 conv/mix through the stock SM12x torch-compose fallback (`short_conv` + `attn_res`, both Triton → sm_110 JIT) on Thor, env-reversible: `MJOLNIR_INKLING2_CONV_MIX_NATIVE=1` forces the native kernels once the base carries sm_110 cubins. The TP>1 all-reduce/seq-conv kernels stay native-only — TP1 configs until #69167 lands in a pin. Everything else in the model is Triton/gluon or FlashInfer-JIT and compiles for sm_110 at runtime. |
| `thor-inkling2-fa4-rel-sm110.patch` | — (Thor-authored) | **Inkling 2's own attention + ViT on sm_110 FA4.** The model owns attention (`Inkling2AttentionBackend`; per-token sinks, sheared rel-bias) and gates its FA4 path on `use_sm100_kernels()` — the else-branch is the Triton-sm90 pair, fine on sm90, dead-slow assumption on Thor. Patched to run the tml-FA4 sheared-bias kernel on sm_110 too (runtime-compiled `sm_110a`, probed OK dense+paged), env-off `MJOLNIR_INKLING2_FA4_REL=0`. The **vision encoder** (`vision.py`) hardcoded `fa_version = 2 if family(120) else 3` — and the wheel's `_vllm_fa2_C`/`_vllm_fa3_C` carry **sm_80/sm_90a cubins only** (FA2's PTX also needs a newer driver JIT than the 580/CUDA-13.0 stack accepts), so the encoder-cache profiling run died with `no kernel image` (`hopper/flash_fwd_launch_template.h:172`) right after weight load — the crash that looked like a target/draft backend bug. Routed to the same tml gate. The **plain `FLASH_ATTN` legs** (DSpark draft, pass-cache attn) get a stock `get_flash_attn_version` sm_110 branch in the backend patch's fa_utils hunk (`major == 11 → FA4`, env-off `MJOLNIR_FA2_ON_SM110=1`; stock majors 9/10 only → FA2). The `_vllm_fa*` sm_110 rebuild was probed and parked (scratch `.fa110/`, not shipped): the fork pins FA2 gencode `8.0+PTX` / FA3 `9.0a` per-file, so an ext rebuild is a fork-level change — the cutedsl/tml runtime paths need no ext. |
| `thor-inkling2-ll120gate.patch` | [#69168](https://github.com/vllm-project/vllm/pull/69168) (merged-in-pin, family(120)-gated) | **E10 KEPT: the low-M GEMM trio gates → 110, default-ON.** `use_sm12x_bf16_ll_gemm`/`use_qkv_bmm` admit sm_110: the low-M linear leaves plain `F.linear` for the Triton split-K/blend path (the M-shape checks self-gate). The run's best step reads (19.78 env-arm / 19.70 plain confirm) and the eager tail 52.7→52.0. Escape `MJOLNIR_INKLING2_NO_LL120`. |

| `thor-async-sched-depth.patch` | [#64145](https://github.com/vllm-project/vllm/pull/64145) (post-pin 2026-11-19, cherry) | **`--async-scheduling-depth N`** — keeps N−1 scheduler steps in flight so the host round-trip (output processing, scheduling, input prep) hides under the GPU step; upstream +53% c1 tok/s with DSpark-7 (885→1359 across depth 2–6, TP8 GB300 — expect the low end at TP1). Runtime hunks only (+82/−22, tests stripped); flag-gated, default 2 = prior behavior. The inkling run's async-scheduling wash was the depth-2 arm — the untested region is 4–6. |
| `thor-inkling2-fp8kv-sm110.patch` | — (Thor-authored) | **Unit-scale fp8 (E4M3) KV on sm_110.** The model's fp8 path is complete and arch-generic — the storage view (`_split_kv_cache`), the triton decode's `cvt.rn.f16x2.e4m3x2` PTX arm — but the dtype gate accepts `auto/bfloat16` off SM12x. Patch adds the family(110) arm, env-off `MJOLNIR_INKLING2_FP8KV=0`; config `NVFP4_FP8KV`. Motivation: the M0 profile — the decode window is attention-bound at depth (23.5% of the step at 128K, `_rel_attention` the top kernel) and fp8 halves the KV read. The tml FA4 path is same-dtype-asserted: the wrapper converts the fp8 cache on entry (no-op bf16). **Revived 2026-11-23**: the capacity regime — pool-bound at 2.12×/262K — flips the E4 step trade (~1.8× tokens for +15.7 %/step); the variant rebuilt on the 0.95-capacity base. |
| `thor-inkling2-splitcap-sm110.patch` | — (Thor-authored) | **Decode split-budget on Thor.** `plan_decode_attention`'s `/8` denominator hands the 20-SM Thor 2 splits where an H100 gets 16; the ncu repro (`task/thor/inkling2-phase2/`) shows the kernel at 14.9% occupancy / 23.7% mem SOL on the verify shape. `/4` doubles the budget. A-candidate: kept only on a same-run-style profile win. |






| `thor-mjolnir-attention-backend.patch` | — (Thor-authored) | **The `MJOLNIR` attention-backend name.** Adds `MJOLNIR = "mjolnir.backend.MjolnirAttentionBackend"` to `AttentionBackendEnum` (`v1/attention/backends/registry.py`), so `vllm serve <model> --attention-backend MJOLNIR` parses (the CLI resolves the name through the enum *before* plugins could add one — an `Enum` cannot grow at runtime) and `Attention.__init__`'s `AttentionBackendEnum[backend.get_name()]` round-trip works. The class itself comes from the `mjolnir-kernel` package, installed by the `Dockerfile`; its `vllm.general_plugins` entry point also maps `CUSTOM` to the same class, so a stock vLLM can use `--attention-backend CUSTOM`. Also makes **FlashAttention 4 the Mjolnir backend's default fallback** (its prefill/mixed-step backend) on sm_110 — see "The Mjolnir attention backend" below. The patch's fa_utils hunk also carries the stock sm_110 → FA4 `flash_attn_version` default for non-Mjolnir backends (see the inkling2 row). |
| `thor-gdn-prefill-sm110.patch` | — (Thor-authored) | **RETIRED on 0.33.0 (2026-11-04, the c4 aggregate stagger — a Thor patch bisect isolated this pair; the spec-graph family and the kernel package were exonerated):** GDN **prefill** grant in `_resolve_gdn_prefill_backend()` (the fast FlashInfer kernel for the model's GDN layers) — probe-gated at import; see "Thor sm_110 gate probes". **Carried the 0.33 c4 stagger**: with the pair installed, requests 2–4's prefills slide ~5–7 s behind request 1 in mixed steps (benchy aggregate −30% at c4/8K); removing just this pair restores the 0.32 ordering (headfix3: aggregate 64.5–68.5 vs 0.32's 66.5, TTFT spread ~3.6 s = healthy). Stock GDN prefill serves correctly on 0.33 (kernelonly A/B: acc 3.59–3.63). Re-visit when the underlying mixed-step interaction is understood. |
| `thor-fa4-fp8kv-sm110.patch` | — (Thor-authored) | **FA4 + FP8-KV** eligibility gate (probe-gated fam(110) OR-term in `flash_attn_supports_kv_cache_dtype`) + the FA4 arch assert widened to `(10, 11)` — **head-size-blind since the 0.35.0rc0.dev nightly, where it also carries hd256+FP8** (the native `DescaleTensors` fwd landed; the hd256-specific patch retired) — **inert until a config selects FA4** (default stays FlashInfer); see "Thor sm_110 gate probes". |
| `thor-fa4-hd256-fp8-sm110.patch` | — (Thor-authored) | **RETIRED at the 0.35.0rc0.dev nightly (native descales; see above).** hd256 **FP8-KV kernel + policy** patch (the "N-1" workstream — enable the FA4 hd256 kernel on sm_110, probe-gated, T11 canary in `kernel_test_sm110_gates.py`); the decode path is the 1CTA carve-out above. |
| `thor-gdn-prefill-fi-sm110.patch` | — (FlashInfer) | **RETIRED on 0.33.0 together with `thor-gdn-prefill-sm110`** (the c4 aggregate stagger — the pair is the carrier; see that row). **FlashInfer-side** GDN prefill non-CP dispatch allowlist widened major 10 → (10, 11) so the SM100 CuTe-DSL kernel is reachable on sm_110 (behavior unchanged on other devices); see "Thor sm_110 GDN prefill enablement (flashinfer side)". |
| `fi-fp6-kv-ampere-prefill.patch` | — (Mjolnir-authored, **FlashInfer-side**) | **Packed-fp6 (e2m3) KV direct-read for FlashInfer's Ampere (sm_80-family) FA2 prefill** — the scratchpad-free fp6 fallback: the produce stage stages 16 codes (12 bytes, partial-src cp.async) per smem slot and an in-place pass rewrites each slot to e4m3 (`code · s_channel`, per-(kv-head, channel) float scales via two new optional `maybe_{k,v}_fp6_scales` params), after which the tile is byte-identical to fp8 and **every** downstream path (per-fragment `vec_cast`, both tile shapes) runs unchanged. Inert unless the fp6 dtype string is requested; see "fp6 prefill direct-read (flashinfer side, sm_80/87)" below. **0.33.0 bump:** applies to FlashInfer 0.7.2.post1 (22/23 hunks verbatim; the one dropped hunk was a pure line-collapse in `jit/attention/modules.py`). |
| `thor-sm110-cutedsl-moe-gate.patch` | — (Thor-authored) | **Admit sm_110 to the CuTe DSL NVFP4 MoE.** `FlashInferCuteDSLExperts._supports_current_device` accepts device family 110 next to 100, so the NVFP4 MoE oracle can pick `FLASHINFER_CUTEDSL` for **activation-quantized (W4A4)** experts on Thor. W4A16 experts keep MARLIN (the CuTe DSL kernel needs FP4 activations). Pairs with the FlashInfer-side `fi-cutedsl-moe-sm110.patch`. Probe-gated: a no-op off sm_110. |
| `fi-cutedsl-moe-sm110.patch` | — (Thor-authored, **FlashInfer-side**) | **RETIRED on 0.34 (the FI cute-dsl MoE restructure).** 0.34's moe_utils majors already include 11 and the grouped-GEMM wrappers moved to `fused_moe/cute_dsl/rubin/` with a hard `major != 10` gate — the patch's paths no longer exist; out of the 0.34 apply order. The W4A4-on-0.34 FI rework is a follow-up. Original: **FlashInfer half of the CuTe DSL MoE enablement.** The blockscaled contiguous grouped-GEMM wrappers (`…gather_grouped_gemm_act_fusion.py`, `…grouped_gemm_finalize_fusion.py`) accept major 11 (CuTe DSL compiles them for `sm_110a` natively), and the `moe_utils` JIT module (`moe_sort`/`unpermute`) builds for major 11. No behavior change on any other arch. |
| `thor-qwen4-dense-model.patch` | Thor-authored (basis: the recovered Intel `fp8_dense.cpp` — Qwen4-27B is **dense**) | **Dense-decoder support for the qwen4_exp family.** v0.32.0 builds `Qwen4ExpSparseMoeBlock` for every layer and transformers rejects `num_experts <= 0`, so no dense Qwen4 checkpoint can load. The patch mirrors qwen3_next's own hybrid rule (`mlp_only_layers` + `(layer_idx + 1) % decoder_sparse_step`): a checkpoint that lists every layer in `mlp_only_layers` gets `Qwen3NextMLP` (`Qwen2MoeMLP`) with `config.intermediate_size`. The MTP drafter goes dense through the same layer class; `Qwen4ExpMixtureOfExperts` already handles zero MoE layers. **Inert** for every existing checkpoint (none set `mlp_only_layers` on qwen4_exp). See "Thor qwen4 dense decoder" below. |
| `thor-qwen4-qsa-mjolnir.patch` | Thor-authored (kernel: `mjolnir.qsa_impl`) | **Probe-gated Mjolnir decode for the QSA layers** (`MJOLNIR_ATTN_QSA`). Wraps the stock impl — routes pure-decode steps through the tcgen05 scratch-gather (the measured 13.5×/11.7× path) and falls back to the stock Triton kernel for prefill/verify/mixed steps, bf16 caches, or anything else it does not own. The heavy logic ships in `mjolnir-kernel` (`src/mjolnir/mjolnir/qsa_impl.py`), not in the patch. INERT without the env var. |
| `64739-safetensors-index-subdir.patch` | [#64739](https://github.com/vllm-project/vllm/pull/64739) (open) | **RETIRED at the 0.35.0rc0.dev nightly (base-covered; see above).** **Shards listed in an index subdirectory.** The default loader only globs the model root; FP4-Local keeps its 21 shards under `tensors/` (the index names `tensors/model-000NN-of-00021.safetensors`), so stock vLLM (0.33 and 0.34) finds no weights. With the index present, `_prepare_weights` and `filter_duplicate_safetensors_files` take the index's file list (paths outside the model folder raise); root-level checkpoints are unchanged. Lets the FP4-Local config load from the Hub id. Qwen3.8-Flash-Next stack, see below. |
| `thor-qwen4-ple-disk.patch` | Thor-authored (layer: `mjolnir.qwen4exp.ple_disk`) | **GPU slot-pool cache + prefetch in front of vLLM's PLE disk table** (`--engram-config '{"disk_offload": true}'` + `MJOLNIR_PLE_DISK_CACHE_GB` > 0). `Qwen4ExpNGramEmbedding` builds `Qwen4ExpPLEDiskEmbedding` instead of upstream's `Qwen4ExpPLEDiskOffloadEmbedding`: zero-row placeholders, the weight loader records each table shard's file and offset from the loader's view (as upstream does), `bind_file_shards` opens a GPU CLOCK slot pool, each lookup reads only missed rows (parallel `pread`) on a prefetch thread, no per-step stream sync. One hunk in `nvidia/ngram_embedding.py` (the modules are imported only on that branch); INERT without `disk_offload` or with the cache at 0 (upstream's reader then runs unchanged). |
| `thor-qwen4-draft-head.patch` | Thor-authored (logic: `mjolnir.qwen4exp.draft_head`) | **Reduced MTP draft vocabulary and 8-bit drafter.** With `MJOLNIR_DRAFT_VOCAB` (the same `draft_vocab.json` and loader as `mjolnir.lmhead`) `Qwen4ExpMTP` builds its own `draft_head` from those `lm_head` rows during the weight load and scores only them (`-inf` elsewhere; verification keeps the full vocabulary, so output is unchanged). `QWEN4EXP_DRAFT_HEAD_FP8` quantizes that head online, `QWEN4EXP_MTP_DENSE_FP8` the drafter's BF16 dense projections (stock online methods, `QWEN4EXP_FP8_FORMAT`, default 128x128 block FP8). `nvidia/mtp.py`; INERT without the env vars. |
| `thor-qwen4-screened-head.patch` | backport of [#60566](https://github.com/vllm-project/vllm/pull/60566) (open; logic: `mjolnir.qwen4exp.screened_head`) | **Screened target lm_head** (`QWEN4EXP_LM_HEAD_SCREENED=1`): an FP8 copy of the head with a rigorous per-row error bound selects the rows that can be in the top k; only those get exact BF16-weight logits, the rest `-inf`. Used for greedy / `top_k <= QWEN4EXP_LM_HEAD_SCREEN_K` (32) batches without logprobs, penalties or other logits processors; every other batch takes the full head. `nvidia/model.py` (`compute_logits`, `load_weights`) + wrappers on `GPUModelRunner.sample` / `Sampler.add_request` installed at load; INERT without the env var. |
| `thor-qwen4-qsa-fp6.patch` | Thor-authored (the packed-e2m3 KV path; sentinel `THOR_QWEN4_QSA_FP6`) | **The packed-e2m3 main KV cache for the QSA layers** (`kv_cache_dtype == "fp6_e2m3"` branch in the QSA attention's cache write/read): the fp6 kernel-family storage on the qwen4-exp sparse-attention stacks. Registered in `apply_patches.py`, two sentinels in `verify_patches.py`; documented here per the final-CR completeness pass (I4). The qwen4_exp-specific inventory lives in `PATCHES-qwen38-flash-next.md`. |
| `thor-qwen4-side-fp8.patch` | Thor-authored (logic: `mjolnir.qwen4exp.side_fp8`) | **Online FP8 side layers** for `nvidia/Qwen3.8-Flash-Next-NVFP4` (ModelOpt MIXED_PRECISION, side layers BF16). `QWEN4EXP_SIDE_FP8=1` composes vLLM's stock `OnlineQuantizationConfig` onto the target's quant config for GDN in/out, QSA qkv/o/indexer and shared-expert projections (a no-op when the checkpoint already quantizes them, e.g. FP4-Local's MXFP8); `QWEN4EXP_SIDE_FP8_SKIP=self_attn.indexer.index_qk_proj` keeps only the QSA indexer BF16, which vLLM 0.34's own fusion rule then leaves unfused. `nvidia/model.py`; INERT without the env var. |
| `thor-qwen4-hc-fp8.patch` | Thor-authored (logic: `mjolnir.qwen4exp.hc_fp8`) | **Online 8-bit hyperconnection GEMMs** (`QWEN4EXP_HC_FP8=1`, C2): the target's `GatedResidual` down(+inject) / up projections and the final mixer (194 BF16 matrices in both checkpoints, built with `quant_config=None`) take stock online methods (`QWEN4EXP_HC_FP8_FORMAT`, default `fp8_per_block,mxfp8`, the only pair whose kernels accept both shapes); the pad rows load as zeros, a converted down layer runs Linear + `hc_silu` instead of the fused BF16 CuTe DSL kernel, a converted up layer skips 0.34's fused SM100 `hc_up_gate_mix`. Drafter mixers stay BF16. `nvidia/hyperconnection.py` + one `nvidia/model.py` log hunk; INERT without the env var. |
| `thor-draft-full-cg-piecewise.patch` | Thor-authored | **FULL CUDA graphs for draft decodes under a PIECEWISE target** (`MJOLNIR_SPEC_STEP_DRAFT_FULL_CG=1`). vLLM gives the autoregressive draft decode manager `NONE` unless the target mode has a FULL decode mode, so with a PIECEWISE target every draft pass after the first runs eagerly op by op. The gate grants `FULL_DECODE_ONLY` when the draft's attention supports at least `UNIFORM_SINGLE_TOKEN_DECODE`; capture, dispatch and replay are the stock FULL path. `autoregressive/speculator.py` (the 0.34 AR base class: every autoregressive drafter, MTP included); INERT without the env var. |
| `thor-mxfp8-tactic-sm110.patch` | Thor-authored | **FlashInfer CUTLASS MXFP8 tactic pin by M bucket** (`MJOLNIR_MXFP8_TACTIC="<max_m>:<tactic>[,…]"`, C3). With autotune off, `mm_mxfp8(backend="cutlass")` runs tactic 0 (tile 128×128×256, cluster 4×2×1), which leaves Thor's bandwidth unused at decode M; the switch registers a custom op that runs FlashInfer's own CUTLASS MXFP8 runner with the pinned tactic for M up to the bucket and the stock call above it. Every tactic is bit-identical. `kernels/linear/mxfp8/flashinfer.py`; INERT (op not even registered) without the env var. |

## Why "hand-adapted" and not the raw diff

The PRs were written against vllm bases that predate the current one. The
drift that forced manual adaptation (54165's `drop_last_prefix_cache_block`
supersession, the `reachable_block_mask` signature growth, hunks the
nightly already had) is now history: **at the 0.33.0 bump the remaining two
open-PR patches (50885, 49652) apply with zero or near-zero context drift**,
and 50885 shrank to its policy half — 0.33.0 adopted the wrapper plumbing
(`_get_decode_wrapper(..., q_len_per_req)`, the tuple-keyed CG wrapper dict,
`fast_plan_decode` threading `q_len_per_req` into FlashInfer 0.7.2's
`fast_decode_plan`, which plans uniform multi-token queries on the fa2
tensor-core backend). What stayed Thor-side is the probe + the two gates +
the `build()` per-request computation, because 0.33.0's own multi-token
decode grant covers only trtllm/XQA (`has_uniform_batch_support`) and the
pre-SM90 `use_fa2_native_decode` path — not the FI-native tensor-core path
sm_110 actually uses.

## Files touched (per patch)

- **50885**: `v1/attention/backends/flashinfer.py`
  - new `flashinfer_supports_uniform_multi_token_decode()` (@cache) probing
    FlashInfer's `fast_decode_plan` for `q_len_per_req`;
  - `supports_spec_as_decode` extended with the FI-native multi-token
    OR-term, `get_cudagraph_support` returns `UNIFORM_BATCH` when the probe
    passes (causal, non-trtllm);
  - `build()` computes the per-request `num_input_tokens`/`q_len_per_req`
    for the FI-native path (previously fa2-native-only, pre-SM90).
- **49652**: base `CudaGraphManager._init_candidates`
  - one-line guard: skip Dynamic-SD candidate expansion when
    `decode_query_len <= 1` — replaces the old
    `SpeculatorCudaGraphManager.__init__` workaround in
    `v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py`.
- **thor fused draft-decode** (Thor-authored, no upstream PR):
  `v1/attention/backends/flashinfer.py`
  - `FlashInferMetadataBuilder.__init__` ORs
    `supports_draft_decode_metadata_update` with `enable_cuda_graph and
    dcp_world_size==1 and not use_trtllm_decode_attention` (the trtllm flag
    identifies the *non*-native path) and allocates the live references
    `_draft_seq_lens_gpu` / `_draft_block_table`;
  - `build()` captures `common_attn_metadata.seq_lens` / `.block_table_tensor`
    (the advancing GPU buffers) into those fields;
  - replaces the upstream `update_draft_decode_metadata()` `pass` stub
    (0.33.0 added it for trtllm-gen) — returns for the trtllm/XQA
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
- **thor CuTe DSL MoE** (Thor-authored): vLLM
  `model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py`
  (`_supports_current_device`: family 100 **or** 110); FlashInfer
  `fused_moe/cute_dsl/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py`
  and `…grouped_gemm_finalize_fusion.py` (capability check `major not in
  (10, 11)`), `jit/moe_utils.py` (`supported_major_versions=[9, 10, 11]`).
  One line each, sentinel `THOR_SM110_CUTEDSL_MOE` on every changed line.

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
print("fi-update", has("v1/attention/backends/flashinfer.py",
                       "def _advance_fi_decode_plan_kernel"))
# the retired-at-0.33.0 fixes, if you want to confirm they stay in the base:
print("54163/54165", has("config/speculative.py",
                         "def needs_eagle_block_drop"))
print("58231", has("v1/sample/ops/topk_topp_triton.py",
                   "keep = (prob > pivot) | (x == M)"))
print("52670", has("config/cache.py", "def register_kv_cache_dtype"))
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

## fp6 e2m3 KV cache — RETIRED at the 0.33.0 bump (was `patches/thor-fp6-e2m3-kv-cache.patch`)

**The dtype now registers through vLLM's pluggable KV-cache-dtype API**
(vLLM #52670, in the base since 0.33.0): `mjolnir.kvdtype.register()` — called
from the plugin entry point (`mjolnir:register`) — decorates a handler whose
`torch_dtype()` is the packed-uint8 cell view and whose `quant_mode()` is
`KVQuantMode.CUSTOM` ("backend self-manages kernel dispatch"). The old image
patch's four edits (the `CacheDType` literal, `STR_DTYPE_TO_TORCH_DTYPE`,
`get_kv_quant_mode`, `is_quantized_kv_cache`) are exactly that API.

Everything that touches the bytes still lives in `mjolnir-kernel`
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

**Verification:** the `Dockerfile` build smoke asserts
`mjolnir.kvdtype.register()` returns True and
`get_kv_quant_mode("fp6_e2m3") == KVQuantMode.CUSTOM` after registration,
then imports the kernel package's fp6 modules. **Tests:**
`tests/kernel/test_runner.py` (the write vs the kernel package's quantizer
and packer, the expansion vs a decode, T5 on an fp6 cache vs the emulator),
`tests/cli/test_kvdtype.py` (the registration no-ops) and
`smoke_vllm.py --kv fp6_e2m3` (`mjolnir verify vllm`, mode `e2m3`).

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
applies to FlashInfer 0.7.3 (0.34.0 base, 2026-11-16 —
`vec_dtypes/cp_async/permuted_smem/jit utils` are byte-identical
0.7.2.post1→0.7.3; `prefill.cuh` churned but every hunk lands) — the headers
+ the fp6 expand instantiation compile for sm_87 (nvcc, CPU-side; the
`fp6-expand` image canary is bit-exact on Thor); runtime prefill canary +
the vLLM-side `MJOLNIR_ATTN_FP6_DIRECT` wiring are pending. The Python
side needed a re-anchor for 0.7.3 (2026-11-17): the 0.32-era
`_validate_workspace`/single-prefill-list hunks sat on pre-token-head-sf
text, so the whole `jit/attention/modules.py` section is regenerated against
the wheel (net diff; the dropped hunks were line-collapses) and
`apply_patches.py` patches with `--fuzz=3` (one `prefill.py` context needs
it). Clean-room: fresh-wheel apply 0 FAILED, full sentinel set green, all
four Python files parse. The first device gate then caught a real one the
sentinels cannot see: 0.7.3 grew `*_scale_smem_ptr` trailing args on the
plain `compute_qk`/`compute_sfm_v` calls, which broke the V-hunk's context
match and let `--fuzz` slide the insertion into the `USE_SINGLE_PREFILL…`
branch (dead def, `warp_vo_base` unqualified). The whole `prefill.cuh`
section is likewise regenerated against the wheel: fp6 arms mirrored onto
the 0.7.3 else-shape (scale args included, per the "stock path unchanged"
intent), the misplaced `vosplit` back in its own branch, fp6 last in the
chain — fp6-expand canary bit-exact 4/4 heads on Thor.**

**Sentinel:** `verify_patches.py` checks the fp6 struct in `vec_dtypes.cuh`,
the partial-src loader in `cp_async.cuh`, and — the "used, not just defined"
sentinel — the `expand_fp6_tile_to_fp8<KTraits, KTraits::HEAD_DIM_QK>` call
site in `prefill.cuh` (a partial apply lands the helper without the
compute-side call would compile and silently no-op).

## The 0.34 fp6 fallback regression — root-caused: vLLM #63232 (2026-11-17)

**The change: vLLM [`23b898419` — "FA4: pass no FP8 descale for a unit
per-tensor scale" (#63232)]**, new in 0.34. It makes
`FlashAttentionImpl._descale` return **None** instead of a ones tensor when
the layer scale is scalar-1 — and the cute interface's `DescaleTensors`
handling of that None is incomplete on the path we ride:

- the interface **packs the present descales positionally** (its own comment:
  values "cannot shift … and silently corrupt QK/PV scaling");
- the missing-slot materialization exists **only in the SM90
  `fp8_kv_dequant` branch** — the `is_fp8` path (sm_100/**110**, every FA4
  fallback call) passes Nones straight through;
- our fp6 fallback hands it `q_descale=None` (scalar-1 `_q_scale`) plus
  per-head K/V tensors ⇒ **the K/V descales land in the wrong slots** ⇒
  corrupted attention ⇒ the smoke's period-1 token loops
  (`authority authority…`, `.def.def.def…`) on every page size, graphs and
  eager alike. 0.33 always passed real tensors: no shift, green.

**Proven by staged revert on the 0.34 image (FA4 fallback forced):**
the whole-`vllm_flash_attn` 0.33 package over 0.34 → still red (package
exonerated; the CuTe kernels are unchanged 0.33→0.34); the `_descale` hunk
alone reverted to `scale.expand(shape)` → **8/8 near-ties**; the
`canonicalize_singleton_dim_strides` rider kept with only the None branch
dropped → **8/8** — the None is the bug, the stride walk innocent.

**The fix (ours): `MjolnirImpl._descale` overrides the base with the 0.33
semantics (always a real expanded tensor) — FA4 stays the fallback, no image
env, the fp8 legs untouched. The upstream-shaped fix (materialize missing
descale slots in the `is_fp8` branch too) is the candidate vLLM PR.**

**The upstream candidate, concretely** (`vllm/vllm_flash_attn/cute/
interface.py`, extend the existing materialization to both paths):

```python
-if fp8_kv_dequant and any(t is None for t in (q_descale, k_descale, v_descale)):
+if any(t is None for t in (q_descale, k_descale, v_descale)) and (
+        fp8_kv_dequant or is_fp8):
     # DescaleTensors' MLIR reconstruction packs present values positionally.
     # ... (unchanged)
```

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

**Apply to a vLLM 0.34.0 checkout** (after the FA4 patches, which also touch
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

## Thor MoE backend gates on sm_110 — RETIRED at the 0.34.0 bump (2026-11-16)

Both fixes are in the v0.34.0 base (verified on the pristine tag): the
GPT-OSS MXFP4 selector carries the SM110 clause (`oracle/mxfp4.py`, "Jetson
Thor (SM110): MARLIN stays the default") with the `_swizzle_mxfp4`
persistent + `epilogue_subtile=1` constraints on families (100, 110, 120)
(#62133), and `CutlassMoEMethod._supports_current_device` allows exactly
families (100, 120) with the "SM110 excluded: grouped NVFP4 kernel is
sm100-only" comment (#62275). The patch files and their `PATCH_ORDER` /
`verify_patches` entries are gone; git history has the last applied state
(`v0.33.0`-era).

- **`thor-sm110-mxfp4-moe-gate.patch`** —
  [#62133](https://github.com/vllm-project/vllm/pull/62133) "Fix OAI Triton
  MXFP4 MoE on SM110" (supersedes the closed #61387). #58877
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

**0.34.0 bump status:** both patches are RETIRED — #62133 and #62275 are
in the v0.34.0 tag (the selector's SM110 clause and the
`(100, 120)`-only `_supports_current_device`, both verified on the pristine
tag). The 0.33.0-era record: #62133 merged upstream 2026-10-27 but was not
in the 0.33.0 release branch; #62275 was still open. The split top-p fix
#58231 retired earlier (0.33.0 base). Verified on a pristine
`vllm/vllm-openai:v0.33.0` tree back then: full stack applied, all fixes
verified (no GPU); the 0.34 equivalent gate is the 2026-11-16 no-GPU run
recorded at the top.

## Thor CuTe DSL NVFP4 MoE on sm_110 — `patches/thor-sm110-cutedsl-moe-gate.patch` + `patches/fi-cutedsl-moe-sm110.patch` (2026-11-16)

**0.34 status: the vLLM-side gate below is in the apply order and verified; the
FlashInfer half is RETIRED (0.34 restructured the cute-dsl MoE — see the row)
and the W4A4-on-0.34 FI rework is a follow-up. The 0.33-era record:**
FlashInfer 0.7.2.post1's CuTe DSL NVFP4 MoE (fused gather/act FC1, fused
finalize FC2) is written for SM100 and gated on `major == 10`, but CuTe DSL
4.8 targets `sm_110a` natively: with the four one-line gates lifted it
compiles and runs on Thor, matching a float reference (T12 below).
The kernel is selected only for **W4A4** (activation-quantized) NVFP4
experts; W4A16 experts stay on MARLIN. Note: with the gate open, a natively
W4A4 NVFP4 MoE checkpoint on sm_110 now auto-selects FLASHINFER_CUTEDSL
(second in vLLM 0.33's NVFP4 MoE order) without any switch, and pays the
first-use `moe_utils` JIT below.

**Opt-in for W4A16 checkpoints: `MJOLNIR_NVFP4_MOE_W4A4=1`** (`mjolnir/moew4a4.py`,
installed from the plugin entry point; no vLLM file patch). ModelOpt exports
calibrated per-expert `input_scale` tensors even for experts it declares
`W4A16_NVFP4` (`nvidia/Qwen3.6-35B-A3B-NVFP4`: its `config_groups` record 4-bit
input activations for them). The switch serves those routed experts as NVFP4
(W4A4) with the checkpoint's own scales; linear layers, shared experts and the
lm_head keep their declared algorithm. The input scales are created NaN and
the load refuses if any is still NaN, so a checkpoint without them fails at
startup. Config: `configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_W4A4` (YAML
identical to `NVFP4_MJOLNIR`, the switch in its `.env`).

**Measured** (Thor, 2026-11-16, `mjolnir bench perf`, c1, pp2048/tg128,
6 runs + 1 warmup per cell, gated; two runs 2 h apart: `NVFP4_MJOLNIR` on the
image without the patches, `NVFP4_MJOLNIR_W4A4` on the image with them). The
two `history.jsonl` records are held for a follow-up PR (they would pair with
the v0.31 official 35B baseline in the e2e tables); raw in
`staging/perf-20261116T180418Z` and `staging/perf-20261116T200607Z`.

| depth | TTFT ms, W4A16 (MARLIN) | TTFT ms, W4A4 (CuTe DSL) | decode t/s, W4A16 | decode t/s, W4A4 |
|---|---|---|---|---|
| 0 | 510 | 534 | 127.5 ± 8.0 | 136.8 ± 16.7 |
| 8K | 1937 | **1223** (−37 %) | 145.1 ± 7.1 | 142.6 ± 8.3 |
| 32K | 6743 | **4377** (−35 %) | 142.4 ± 5.4 | 141.3 ± 11.7 |
| 128K | 31096 | **22371** (−28 %) | 101.9 ± 5.6 | 110.0 ± 9.3 |

TTFT −37 % / −35 % / −28 % at 8K / 32K / 128K (1.58 / 1.54 / 1.39× time
ratio); decode unchanged within the MTP spread.
`mjolnir bench accuracy` (tool-eval-bench b60e2f00, `--trials 2`, both legs on the
patched image, run `20261116-205937`): tool calls 90.2 (W4A16) vs 90.9 (W4A4),
Hard Mode 74.1 vs 78.6 — level within the trial spread (the W4A16 leg had
TC-45 excluded in one trial).

**Canary:** `kernel_test_sm110_gates.py` T12 asserts the gate on sm_110 and runs a
small synthetic NVFP4 MoE layer (8 experts, top-2, 48 tokens, H 2560 / I 640)
through the server's FLASHINFER_CUTEDSL preparation path against a float
reference: on Thor rel L2 0.0024 (bound 0.05), min token cosine 1.00000.
`verify vllm` on the patched image: 39/39 (the graph smoke needed one retry for
the known 0.33 startup memory assert, on its FLASHINFER leg).

**First use:** `moe_utils` has no prebuilt `sm_110a` build in
`flashinfer_jit_cache`; it JIT-compiles once (~9 min on 3 CPUs) into the
mounted FlashInfer cache and is reused afterwards. The CuTe DSL kernels
themselves JIT in ~1 s each.

**Drop when** FlashInfer's CuTe DSL MoE wrappers and `moe_utils` admit SM110
and vLLM's `FlashInferCuteDSLExperts` gate includes family 110. FlashInfer
v0.7.4 / main already list major 11 in `jit/moe_utils.py`
(`supported_major_versions=[8, 9, 10, 11, 12]`): that hunk retires at the next
FlashInfer bump; the two wrapper checks (`major != 10`) and vLLM main's
family-100 gate still need it.

## Thor qwen4 dense decoder — `patches/thor-qwen4-dense-model.patch` (2026-10-31)

**What:** dense-decoder support for the `qwen4_exp` family, so the confirmed
**Qwen4-27B dense** shape can serve on the image the day weights land. Basis:
the recovered Intel `fp8_dense.cpp` ("Dense Qwen4 TP2 decode") and the
`qwen4_exp` family code in v0.32.0 — full evidence chain in
`task/research/qwen4-27b-upstream-prep/` (private).

- **The blocker:** `Qwen4ExpDecoderLayer` builds `Qwen4ExpSparseMoeBlock`
  unconditionally, and transformers' `Qwen4ExpTextConfig.__post_init__`
  rejects `num_experts <= 0` — no dense checkpoint can load. Upstream has no
  dense decoder for the family (checked 2026-10-30).
- **The fix:** `mlp_only_layers` + `(layer_idx + 1) % decoder_sparse_step` —
  qwen3_next's own hybrid rule, verbatim. A dense checkpoint ships
  `num_experts > 0` (satisfying the validation) and lists **every layer** in
  `mlp_only_layers`; every `mlp` becomes `Qwen3NextMLP` (the `Qwen2MoeMLP`
  alias — `gate_up_proj`/`down_proj`, the tensor names a dense checkpoint
  carries) with `config.intermediate_size` (leak: 14336). Asserts fire with a
  clear message if a "dense" config forgets either field.
- **What rides along for free:** the MTP drafter builds its layer through the
  same `Qwen4ExpDecoderLayer` (dense drafter); `Qwen4ExpMixtureOfExperts.
  set_moe_parameters` already returns zero counts when no layer matches;
  `is_model_fused_shared_expert_compatible` returns False; `load_weights`'
  `maybe_fuse_shared_experts` is `getattr`-defaulted.
- **Probe-gated, not sm_110-gated:** the branch is config-driven on every
  platform, and **inert** for every existing checkpoint — none set
  `mlp_only_layers` on a `qwen4_exp` config, so `is_moe_layer` stays True
  everywhere and the built model is byte-identical to stock. Other model
  families never import this file.
- **Not sufficient if** the released 27B arrives under a **new model_type**
  (a dense `qwen4` config class + architectures string): that also needs a
  config-class mapping and a registry entry — release-day work, checklist in
  the research dir. The QSA layers (indexer cache, budget 2048) come from the
  family code unchanged; the Mjolnir kernel side is `MJOLNIR_ATTN_QSA`
  (`mjolnir/arch/thor/qsa.py`, `mjolnir verify qsa`).

## Thor qwen4 topk sm110 — `patches/thor-qwen4-topk-sm110.patch` — RETIRED at the 0.34.0 bump

**Retired 2026-11-16: #56102 is in the v0.34.0 base** — `qsa_indexer.py`'s
`use_cooperative_topk` carries `not
current_platform.is_device_capability_family(110)` (next to the existing
120 term), exactly this patch's one-line gate; the dispatch falls through
to `persistent_topk` upstream. The historical record below stands; the
0.34 note on `persistent_topk`-on-sm_110 validation is unchanged (still
device-untested — the first 0.34 smoke covers it).

**What (historical):** the sparse-indexer decode top-k (`qsa_indexer._topk`) must not
select `cooperative_topk` on sm_110 — the gate gains a
`not is_device_capability_family(110)` term, so the Thor dispatch falls
through to `persistent_topk`. One hunk in
`models/qwen4_exp/nvidia/ops/qsa_indexer.py`; **inert on every other arch**
(the gate's existing terms are untouched).

- **Device-proven 2026-10-31** (the fixture serve-smoke, rooster step 5):
  the dense qwen4_exp fixture **loads and prefills** on Thor, then the
  first decode top-k dies with
  `RuntimeError: launch_cooperative_cluster, /workspace/csrc/cooperative_topk.cu:48,
  ... cluster misconfiguration` → `EngineDeadError` — fatal, no fallback
  (`auto` = cooperative → persistent → per_row, but a launch *error* isn't
  a constraint probe). Stock Qwen4 QSA decode is **broken on Thor**.
- **Upstream precedent, same failure class:** SM120 report
  [#47266](https://github.com/vllm-project/vllm/issues/47266) Issue 8
  ("cluster launch rejected on SM120") — its fix excludes the family from
  `use_cooperative_topk`; the 0.32.0 gate already carries the family-120
  exclusion, just not 110. Two open upstream PRs are exactly this fix:
  [#56084](https://github.com/vllm-project/vllm/pull/56084) (fixes #56083)
  and [#56102](https://github.com/vllm-project/vllm/pull/56102) — which
  pins the **mechanism**: the cooperative kernel requests a 16-block
  cluster, Thor's max cluster dimension is 8 (structural; an attribute set
  would not help). Retire this patch at the bump that lands either.
- **Open validation (next window):** `persistent_topk` on sm_110 is
  unmeasured — the smoke re-run with the patched image is the first test
  (then `per_row`/`flashinfer`/`torch` as fallbacks if it also trips;
  `torch` is the guaranteed reference). **VALIDATED 2026-10-31:** the
  patched image serves the fixture end-to-end — greedy decode deterministic
  and finite through all QSA layers, zero errors; this was also the first
  execution of the stock Triton QSA decode attention on sm_110.

## Thor qwen4 qsa mjolnir — `patches/thor-qwen4-qsa-mjolnir.patch` (2026-10-31; re-anchored on 0.34.0)

**0.34.0 re-anchor (2026-11-16).** The qwen4_exp QSA stack was reworked
upstream and the patch + the kernel side (`mjolnir/qsa_impl.py`) were
ported to the new API:

- the stock impl is renamed `Qwen4ExpQSAFlashAttentionImpl` →
  **`Qwen4ExpQSAImpl`** (still `models/qwen4_exp/nvidia/qsa.py`) and its
  ctor slimmed to `(num_heads, head_size, scale, num_kv_heads,
  kv_cache_dtype)` — alibi/window/sink/AttentionType moved into the layer;
- **`forward_qsa` is gone**: the impl entry is now
  `forward(layer, query, kv_cache, attn_metadata, output, *,
  token_to_req, use_prefill_config, output_gate)` (keyword-only, no
  key/value, no per-call output-scale args — the fused-output quantization
  reads `layer._k_scale_float` internally). `build_impl_class`'s subclass
  override follows the same signature and every fallback routes through
  `super().forward(..., token_to_req=..., use_prefill_config=...,
  output_gate=...)`;
- **the main QSA metadata is slim** (`num_actual_tokens`, `max_query_len`,
  `block_table`, `slot_mapping`) — the per-request `seq_lens` moved onto
  the indexer's side metadata (`QSAIndexerMetadata`). The kernel's host
  side fetches it through the forward context under
  `layer.indexer.raw_key_cache.prefix` (the same key the owner layer uses
  in `_run_qsa`) via the new `qsa_impl._indexer_metadata()`; any surprise
  there degrades to the stock path;
- the patch itself shrinks to three hunks (the `init_logger` import, the
  module logger, the probe-gated impl swap after the stock
  `self.impl = Qwen4ExpQSAImpl(...)` assignment) — the 0.33-era cosmetic
  reflow hunks are dropped;
- **not yet device-run**: the 0.32-era smoke (dense fixture load, decode
  through all QSA layers) remains the device evidence; the 0.34 path needs
  the same smoke once the image exists.

**What (0.33-era record):** `MJOLNIR_ATTN_QSA` swaps `Qwen4ExpQSAAttention`'s impl for
`mjolnir.qsa_impl.Qwen4ExpQSAMjolnirImpl` — a subclass of the stock impl
whose `forward_qsa` routes **pure-decode steps** (one query token per
request) through the tcgen05 scratch-gather: packed selection → per-request
block ids → `scratch_plan` → one fused Triton gather per K/V → the dense
kernel on `effective_seq_lens` plans → sigmoid output gate applied
host-side (the stock Triton kernel folds it; the dense kernel doesn't).

- **Fallbacks are the contract:** prefill/verify/mixed batches (per-row
  selections), bf16 caches, unsupported page sizes, or any surprise falls
  back to the stock Triton path — with a one-line reason printed once per
  reason (`MJOLNIR QSA decode: stock path (...)`), and the live path
  announces itself once (`MJOLNIR QSA decode: scratch-gather active ...`).
- **Shared state:** the runner + scratch workspace are keyed by geometry
  and shared across all QSA layers of a model (plan→run stays adjacent
  within one layer's forward; layers run sequentially per step).
- **Eager only in v1:** the selection→plan work syncs; graph capture rides
  on the P2 in-loader gather. Verify-batch (MTP) support needs per-row
  scratch plans or a shared superset selection — deferred with the e2e
  tests.
- **Inert without the env var**; the try/except in the patch keeps stock
  even when the kernel package is absent.

**Sentinel:** `verify_patches.py` checks `THOR_QWEN4_QSA_MJOLNIR` in
`models/qwen4_exp/nvidia/qsa.py` (plus `THOR_QWEN4_DENSE` in `model.py` and
`THOR_QWEN4_TOPK_SM110` in `ops/qsa_indexer.py`). **Validated against the
pristine v0.33.0 tree (2026-11-11, no GPU):** all three patches apply clean
(`git apply --check`) and the patched sources compile; the two 0.33 drifts
are folded in — the dtype-check tuple gained `torch.uint8` (sub-sm_89
byte-decode; hunk re-anchored) and the fused-writer gate gained the sm_89
guard (our cosmetic one-liner hunk dropped, upstream's form kept). **Not
yet run:** a server smoke on the 0.33 image — the 0.32 one (dense fixture
load + prefill + deterministic decode through all QSA layers) remains the
device evidence.

## Qwen3.8-Flash-Next (qwen4_exp) serving stack (2026-11-17)

The sections for the Qwen3.8-Flash-Next patches (what each does, its switch, the
measured effect and when to drop it) live in
[PATCHES-qwen38-flash-next.md](PATCHES-qwen38-flash-next.md).

## Open upstream watch list (re-assessed 2026-11-16, base v0.34.0 = `62d4bc3c7`)

Status of every open PR relevant to Thor — what to pre-patch (nothing),
what to take at a future bump (both images inherit it, so the e2e attention
delta stays the only variable), and what to keep out of the image on
purpose:

- **Retired at this bump (in the 0.34.0 base):** #62133 + #62275 (both
  sm_110 MoE gates), #56102 (the cooperative-topk sm_110 exclusion — the
  `qwen4-topk` patch), #49652's substance (the 0.34 speculator rework
  pins draft-decode graph shapes explicitly), #65050 (CuteDSL GDN prefill
  on SM110 — the selector + the fewer-SMs grid clamp are upstream; our
  retired gdn pair stays retired, and the c4-stagger re-test now has a
  third path: the in-tree CuTeDSL one, still FLA by default).
- **Still open, carry here:** #50885 (the FI-native uniform-multi-token
  FULL grant — our longest-lived patch, unchanged on 0.34) and #56289
  (closed unmerged: the `get_cudagraph_support` head-count mismatch that
  kills draft cudagraphs — our dspark/cg-gate pair is the working
  alternative).
- **Take at a future bump, not before:** #53337/#44590 (Thor-tuned Triton
  `selective_state_update` — still open; our image serves the hybrid model
  on the fused CUDA GDN decode kernel, so one-sided adoption would change
  nothing for our configs while contaminating the e2e comparison) and
  #60566 (screened lm_head — MRV2, explicitly targets Thor, +7.7% greedy
  on Thor-class hw, still open; mutually exclusive with
  `MJOLNIR_LM_HEAD=fp8` (a quantized head raises at load), and default-on
  would inflate our e2e; benchmark it against LMH8 at bump time).
- **Post-0.34-cut merges to watch for the next bump:** #67571 (merged
  2026-11-16, [MRV2][Spec Decode] cap dummy-run requests by their draft
  slots — touches the MTP dummy-batch path our spec legs exercise) and the
  GDN perf pair already in 0.34 (#65874 fused-GDN-MTP DSpark K=8 idle-SMs,
  #66064 prefill state gather/scatter fuse — base-side, both images
  inherit).
- **Keep out on purpose:** #55366 (SM110 FA4 auto-selection — changes the
  stock default our official-image baselines must reflect; add a new
  official leg when it lands), #53520 (prompt_logprobs corruption — V1-runner
  only; our image runs MRV2 and no bench reads prompt_logprobs),
  #54705 (fp8-KV scales — SM100-scoped; our kernel owns sm_110 fp8-KV),
  #53383 (ROCm), #58868 (mmap fault-in — startup-time only).
- **FlashInfer side:** the GDN-prefill FI-kernel pair (flashinfer#7827 +
  the vLLM selector follow-up) remains open upstream — 0.35 at the
  earliest; our retired `thor-gdn-prefill-fi-sm110.patch` is the
  stopgap (re-verify against whatever FI ships then).
- **Post-tag drift = the 0.35 re-anchor list** (main @ 2026-11-16, 618
  commits past `62d4bc3c7`): `v1/attention/backends/flashinfer.py` +376
  lines (all three of our patches there), `dflash/speculator.py` +83 (the
  cg-gate dispatch), `qwen4_exp/nvidia/{qsa,qsa_indexer,ops/qsa_indexer}`
  churn (the qsa patch + the dense fixture), `fa_utils.py −75` (a
  refactor — re-derive the fa4/mjolnir hunks), `cudagraph_utils.py ±39`.
  Plus #67006 (online weight-only NVFP4 for DENSE linear layers, merged
  2026-11-16 post-tag — evaluate for the qwen4 dense day-0 quantization
  matrix) and #65802/#66118 (online-quant reload / FP8 indexer-wk
  loading — qwen4-adjacent loader churn).
