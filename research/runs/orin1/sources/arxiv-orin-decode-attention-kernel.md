# arXiv research brief — Orin decode-attention kernel optimization (sm_87, fp8 KV, M=1..8 verify)

**Date:** 2026-10-22 · **Requester context:** Mjolnir decode attention on Jetson AGX Orin 64GB (sm_87, 16 SMs, ~205 GB/s LPDDR5x, 100KB smem/block, fp16/bf16 mma.sync, no fp8 tensor cores). Kernel: paged decode attention, head_dim=256, GQA g=6, q_len M=1..8 (speculative verify), fp8 e4m3 KV with descale-on-load, split-KV + LSE combine, cp.async smem ring. Competitor: FlashInfer FA2-decode + GEMV path.

Papers were located via arXiv API search (title/abstract queries) and read via the local MCP cache (FlashDecoding++, LeanAttention, PAT, DeFT, P-Cast, SageAttention, FlashInfer read in depth; others verified via abstract).

---

## Topic 1 — Flash-decoding / split-KV decode attention

### 2311.01282 — FlashDecoding++: Faster Large Language Model Inference on GPUs (Hong, Dai et al., 2023, v4)
**Takeaway:** The synchronized partial-softmax update across split-KV partials costs ~18.8% of attention time (Llama2-7B, A100, 1K ctx). Replacing the per-partial max with a single **unified max φ** (chosen from the statistical range of pre-softmax scores) makes partials independent — no cross-CTA sync, no recompute of earlier partials. Fallback: if any `x_i − φ` leaves the representable window, that vector falls back to synchronized softmax (>99.99% of `x_i` stay in-range for Llama2-7B: −16.8 < x_i < 6.5).
**Transferable technique:** (a) **Asynchronized softmax**: compute partials with a declared max φ; combine with plain reweighting. (b) **Flat-GEMM pad-to-8**: tile the M dimension to the native tensor-core M (8), *not* 64 — previous engines pad M to 64 and waste >50% of mma work. (c) **Heuristic dataflow**: at M=1 a CUDA-core GEMV (FastGEMV) beats cuBLAS tensor-core by ~22% (TC = 82.15% of CUDA-core perf); at M=4 the TC path is ~2× the CUDA-core path — pick per-M via profiled inflection points, don't fix one dataflow.

### 2405.10480 — LeanAttention (Sanovar et al., Microsoft, 2024)
**Takeaway:** FlashDecoding/FlashInfer-style **fixed-split** KV partitioning strands SMs on imbalanced loads and its reduction (combine) overhead scales with split count. LeanAttention treats softmax re-scaling as an *associative reduction* and applies **stream-K scheduling** over KV tiles: every SM gets equal work regardless of context length, occupancy ≈100% for all problem sizes, combine overhead constant. Result: 1.73× avg attention speedup over FlashDecoding (2.18× @ 256K).
**Transferable technique:** (a) Equalize **KV tokens per SM-wave**, not CTAs-per-request — split counts should differ per sequence when lengths differ. (b) **LeanTile geometry**: smallest efficient KV tile = 256 tokens (head_dim 64) / 128 tokens (head_dim 128) on A100-class smem; scale tile bytes to smem budget, then fill the wave with unequal tail tiles reduced via the associative rescale. (c) Never let split count be a free knob — it trades occupancy against combine traffic; pin it to `#SMs` with work-stealing.

### 2511.22333 — PAT: Prefix-Aware Attention with Resource-Efficient Multi-Tile Kernel (Yi et al., 2025)
**Takeaway:** Decode attention wastes bandwidth on (1) repeated global loads of *shared-prefix* KV (one-query-per-CTA kernels) and (2) one-size-fits-all tiles that idle on-chip resources and add bubbles for ragged KV lengths. PAT's **pack–forward–merge**: pack queries sharing a prefix into one CTA (KV loaded once), right-size tiles per CTA, split over-long KV CTAs across streams, then a lightweight online-softmax merge. 53.5% avg attention-latency reduction vs SOTA kernels; decode attention is up to 53% of total latency at long context.
**Transferable technique:** (a) For our M=1..8 verify rows over *identical* KV: one CTA per (KV-head, KV-chunk) carrying all M rows — never one-CTA-per-query-row. (b) **Adaptive tile sizing** to smem/register budget per CTA rather than fixed tile → fewer bubbles on ragged batches. (c) Merge with online-softmax LSE composition is negligible cost; the win is all in the pack/forward stage.

### 2501.01005 — FlashInfer (Ye et al., MLSys 2025) — *the competitor, read as spec*
**Takeaway:** FlashInfer formalizes split attention as the **Attention State ⊕ operator** (output + LSE, associative, commutative) — the canonical statement of our split-KV/LSE-combine contract. Scheduling: compile-time tile-size selection + runtime load-balanced plan; persistent CTAs = k×#SMs; partial-output workspace bounded by 2×#CTA×T_q×H×(D+1). It is a *format + scheduler* engine; per-kernel mma geometry is a JIT template parameter.
**Transferable technique:** The LSE-combine math (`O = (e^{LSE_I}·O_I + e^{LSE_J}·O_J)/Σ; LSE = log Σ e^·`) is the exact bit-level reference for our merge kernel — including the j-ordered partial-sum associativity we treat as spec. For quantifying FlashInfer inefficiency: LeanAttention (imbalance/occupancy loss of fixed-split) and PAT (53.5% headroom vs SOTA decode kernels) are the citable external numbers; no paper reports FlashInfer's achieved-GB/s on Orin specifically (see Negative results).

### 2408.04093 — Tree Attention (Shyam et al., Zyphra, 2024)
**Takeaway:** Multi-GPU, but gives the cleanest log-domain statement of the ⊕ reduction and its tree form. Useful for the combine-phase: the LSE merge is a *reduction over (O, LSE) pairs* and can be organized as a balanced tree with identical numerics.
**Transferable:** Merge partials as a tree (pairwise) rather than a serial j-loop when partial count grows — same math, fewer dependent steps.

### 2303.06865 — FlexGen (Sheng et al., 2023) and 2211.05102 — Efficiently Scaling Transformer Inference (Pope et al., 2022)
**Takeaways:** The canonical **roofline framing** for decode: decode attention/linear layers are memory-bound; arithmetic intensity O(g·l_qo) for GQA (FlashInfer §2.1 states this precisely: operational intensity scales with group size g). Pope et al. give the analytical cost model: per-token latency ≈ total bytes moved / achievable BW; batching amortizes *weights* but **not** KV reads per request.
**Transferable:** For Orin roofline: per decode step per layer, KV bytes = `L · n_kv_heads · 2·d_k · bytes/el`. fp8 (1B): L=32K, n_kv=4, d=256 → 64 MB → 312 µs at 205 GB/s (≈435 µs at 80% achieved). M=1..8 reads the *same* KV bytes → verify M=8 amortizes 8 draft tokens over ~zero extra attention traffic. This is the quantitative argument that M-row tiling with KV read-once is the only shape that matters on Orin.

---

## Topic 2 — Tensor-core MMA shapes for decode (M small / GEMV-shaped)

### 2311.01282 — FlashDecoding++ (flat-GEMM section, §4–5)
**Takeaway:** cuBLAS/CUTLASS tile M to 64 to hide latency; decode's M≪64 pads >87% of mma lanes with zeros. The fix: **M-tile = 8** (Ampere mma.sync m16n8k* native M for one fragment row-pair; two fragments cover m16) with K-dim and N-dim tiling carrying the parallelism, plus double-buffered smem for the flat shapes.
**Transferable:** For M=1..8 on sm_87: build the S = Q·Kᵀ mma as **m16n8k16 with Q padded in M to 16** (one mma per K-tile per warp); the padding lanes are compute-free and cost no extra KV traffic. For M<8 the same tile serves all rows — no re-tile per M.

### 2410.02367 — SageAttention (Zhang et al., 2024, v9)
**Takeaway:** 8-bit attention that actually wins on Ampere-class GPUs (RTX 4090/3090, no fp8 TCs): **Q,K → INT8 mma(u8s8s32 / m16n8k32)** for QKᵀ with dynamic per-block scales; **PV stays fp16 with fp16 accumulator mma(f16.f16.f16)** (2× faster than fp32-acc, accuracy-safe because P∈[0,1]); online softmax in fp32. Scales are applied *outside* the mma: `C = ψ⁻¹_{δAδB}(ÂB̂)` — dequant is a post-mma scalar application per tile, not a per-fragment transform. K channel-outlier smoothing (<0.2% overhead) fixes the K-quantization accuracy cliff. 340 TOPS @ 52% of INT8 peak on 4090.
**Transferable:** (a) The **scales-outside-mma** pattern is exactly how our fp8→fp16 descale should be staged: dequantize fp8 KV fragments to fp16 *after* the `ldmatrix`/load (cheap cvt), fold per-channel K scales into Q (diagonal pre-scale, KIVI-style) and apply per-token V scales to the PV accumulator epilogue — zero multiplies inside the inner loop. (b) fp16-accumulator PV is the Ampere-friendly fast path for P·V. (c) INT8 (not fp8) QKᵀ is the proven high-throughput QK path on sm_8x if we ever want >fp8 compute density.

### 2405.10480 — LeanAttention (LeanTile §5.2)
**Takeaway:** Smallest optimal KV-row tile = 256 tokens (d=64) / 128 tokens (d=128) on A100 — the tile is set by compute-per-byte and smem, fixed per architecture. For d=256 fp8 the byte-equivalent tile is ~64–128 KV rows.
**Transferable:** Start the Orin sweep at 64/96/128 KV rows per stage per CTA: 64 rows × 256B (K) = 16 KB/stage; a 3-stage K+V ring = 96 KB — fits 100 KB smem only with 2 stages or K/V split stages; 32-row stages give 48 KB (3-stage ring + Q + stats). This byte-budgeted geometry is the concrete starting point for our cp.async ring.

**Negative result:** No dedicated arXiv paper exists on warp-level KV-row splitting across warps / `ldmatrix` staging of dequantized fp8→fp16 fragments for decode attention. That knowledge lives in code (FlashInfer decode templates, TensorRT-LLM XQA, FastGEMV, SageAttention kernels), not in papers. Closest paper-level treatments are FlashDecoding++ (pad-to-8) and SageAttention (scale-outside-mma), cited above.

---

## Topic 3 — FP8/INT4 KV-cache decode on Ampere-class (no fp8 TCs)

### 2402.02750 — KIVI: Tuning-Free Asymmetric 2bit KV Quantization (Liu et al., 2024)
**Takeaway:** The key cache has strong **channel-wise** structure → K must be quantized **per-channel** (per column of Kᵀ, i.e., along d_k), while V has token-wise structure → **per-token**. This asymmetric scheme holds at 2–4 bits and is the reason per-channel K scales are the industry norm.
**Transferable:** Our per-(layer, KV-head, channel) K scales match this. The decode-kernel consequence: a per-channel-descaled K row equals `diag(s)·k̂`; since S = Q·diag(s)·K̂ᵀ = (Q∘s)·K̂ᵀ, **fold the K channel scales into Q** once per step (M×d vector scale) — then the inner loop reads raw fp8, cvt's to fp16, and does a pure fp16 mma with no per-element descale at all. V per-token scales apply once per KV tile to the PV accumulator (`α_v` scalar per row) — again outside the mma.

### 2401.18079 — KVQuant: 10M-context KV quantization (Hooper et al., 2024)
**Takeaway:** Sub-4-bit KV needs: per-channel K, **pre-RoPE** K quantization, non-uniform datatypes, per-vector dense-and-sparse outlier handling. Custom CUDA kernels: ~1.7× vs fp16 GEMV decode. Quantifies that dequant-in-flight KV decode kernels can beat fp16 on bandwidth alone.
**Transferable:** Pre-RoPE K quantization is the accuracy lever if our e4m3 K ever shows RoPE-amplified outliers (check `mjolnir calibrate` amax per position); per-vector dense-and-sparse is the fallback for outlier channels (a few fp16 rows per tile is cheap).

### 2410.02367 — SageAttention (see Topic 2)
**Takeaway for KV:** INT8 QKᵀ with per-block dynamic scales + K smoothing; the paper's Table-3 analysis shows 8-bit QK keeps attention accuracy where naive fp8 P·V fails. Ampere-relevant (4090/3090): 8-bit mma is the right compute primitive on sm_86/87; fp8 data *storage* with fp16 *compute* (our design) is compatible — fp8 is a bandwidth optimization here, not a TC format.

### 2606.06521 — P-Cast: Precision in FP8 Attention (Lau, 2026)
**Takeaway:** For e4m3 P (P·V), the static scale S and KV iteration order are first-order precision determinants under attention-sink. Results: (1) **S = 256 = 2⁸ is optimal** (bit-exact scaling, minimum worst-case quantization step 2⁻⁴, max normal-range coverage); 448 (FlashInfer / TRT-LLM XQA default) is *not bit-exact* — slightly higher coverage but no closed-form step guarantee. (2) **Forward KV iteration + sink tokens ⇒ P-collapse**: a closed-form fraction Φ(Δ+δ_k−6.93−ln S) of non-sink P values underflows to zero; **reverse iteration removes it** (zero-underflow guarantee with S=256). (3) Keep ℓ (the softmax denominator) in fp32 *pre-cast* — normalization stays exact; error only in the numerator. (4) Failure threshold: Δ_c = 6.93 + ln S − δ_k (sink-strength Δ vs scale).
**Transferable:** Directly actionable for any fp8-P experiment (our `FP8_P_SCALE=448` pin): switch to S=256, iterate KV blocks in reverse, keep ℓ fp32. Note our e2m3 (fp6) P variant changes the envelope — recompute the min-step/coverage argument for 6-bit mantissa before reusing S=256 (the sawtooth argument is format-general: S = 2^k on the lower envelope, k set by mantissa bits).

### 2602.10718 — SnapMLA: FP8 Quantized Pipelining for MLA Decoding (Zhang et al., 2026)
**Takeaway:** FP8 *decode* kernel design: numerical heterogeneity (mixed dtypes per stage) handled by a descale pipeline scheduled across the KV loop; hardware-aware stage overlap. Hopper-oriented, but the descale-in-pipeline structure (dequant fused into the async-load consumer stage) is the pattern for Ampere too.
**Transferable:** Put the fp8→fp16 cvt on the *consumer* side of the cp.async ring (during smem→register staging), never on the producer side; this keeps the TMA/cp.async producer purely byte-streaming (our L2-request-width findings say: the producer must stream wide, contiguous bytes).

### 2606.20474 — UltraQuant: 4-bit KV for context-heavy agents (2026) & 2608.23834 — Minima-KV (2026)
**Takeaways:** Recent 4-bit KV with TurboQuant-style scales (UltraQuant) and mixed-format paged attention (FP8 anchor pages + packed older pages, Minima-KV). Both confirm 4-bit/fp8 KV decode is the mainstream endgame; Minima-KV's mixed-format paging (hot pages fp8, cold packed) is a vLLM-cache-format change we already deemed out of scope, but the *hot/cold* split could be a per-page quant policy later.

---

## Topic 4 — Online softmax numerics (log2/exp2, base-2 scaling)

### 2307.08691 — FlashAttention-2 (Dao, 2023)
**Takeaway:** The canonical rescaling online softmax: `m_new = max(m, rowmax(S)); P = exp(S − m_new); O ← α·O + P·V; ℓ ← α·ℓ + rowsum(P)` with `α = exp(m − m_new)`. FA2's implementation detail (and all FA-family code): **exp is computed in base-2** — `exp2((S − m)·log₂e)`, with the softmax scale `1/√d` folded into the log₂e multiplier, so the inner loop is a single FMA + `exp2` HW instruction. Accuracy: base-2 vs natural-log domain is *bit-comparable* when the scale is folded pre-exponent (identical real function; only fp rounding differs by one multiply position).
**Transferable:** Keep our exp2 pipeline; the multiply `scale·log₂e` belongs in the *score* FMA (Q-side), not in the exponent argument, so per-tile descaling of S never touches the exp path.

### 2311.01282 — FlashDecoding++ (unified max, §3)
**Takeaway:** The max is only needed to avoid overflow, not for correctness — softmax scale φ is arbitrary. A **declared max** (from a calibrated statistical bound) removes the rowmax dependency entirely; a fallback path recomputes synchronized softmax when `x_i − φ` exits the representable window. This is the *declared softmax* our Thor fp8-P analysis found structurally incompatible with fp8 P (P must stay ≤1 for a fixed fp8 scale) — but for **bf16/fp16 P** (no fp8 cast) it is viable and removes the tile-max chain.
**Transferable:** (a) For bf16-P paths: declared max from calibration + fallback flag. (b) Consistent with our run-2 finding: gating the per-tile max reduction to every 4th/8th tile was timing-flat — the softmax chain is not the critical path on bandwidth-bound hardware; don't spend refactor effort there. (c) For split-KV partials, the combine already needs no max (LSE composition) — the unified-max win applies *within* a partial's inner loop only.

### 2606.06521 — P-Cast (see Topic 3)
**Takeaway (numerics):** ℓ in fp32 pre-cast ⇒ normalization exact, error only in numerator; S=2^k bit-exactness; sawtooth dp(S) envelope argument generalizes to any mantissa width (e4m3 → dp_min = 2⁻⁴; for e2m3 recompute).

**Negative result:** No arXiv paper is dedicated to log2-domain exp2 softmax accuracy analysis. The authoritative sources are FA2 (§ numerics + code), FlashDecoding++ (§3) and P-Cast (quantitative). Tree Attention (2408.04093) supplies the log-domain algebra for the combine.

---

## Topic 5 — Speculative-decode verify kernels (M>1 over shared KV)

### 2305.09781 — SpecInfer (Miao et al., 2023)
**Takeaway:** Introduced **tree attention**: token-tree drafts verified in one pass with a topology-aware causal mask (bitmask per token). The M queries share all prefix KV; the tree's internal edges add tiny mask-only blocks.
**Transferable:** Represent the verify batch's causal structure as a compact bitmask; the KV read stays read-once; mask storage is 1 bit/token.

### 2401.10774 — Medusa (Cai et al., ..., Tri Dao, 2024)
**Takeaway:** Tree attention with a **static** causal mask for MTP-style multi-head verify; 2.2–3.6× end-to-end. Confirms the M-row verify shape: all draft positions attend to full KV + tree-internal mask.
**Transferable:** The MTP verify case (our M=1..5+1) is exactly "sequential M rows + fixed mask" — one tile shape, mask folded into score epilogue.

### 2404.00242 — DeFT: Decoding with Flash Tree-Attention (Yao et al., 2024, v4)
**Takeaway:** The kernel-level paper for tree/spec verify attention. Two mechanisms: **KV-Guided Grouping** — group the *prefix KV* with all queries that share it so prefix KV is loaded exactly once (Q re-loads are negligible: M rows ≪ KV length); **Flattened Tree KV Splitting** — flatten the tree-structured KV into even chunks for load balance, carrying **bit causal masks** to restore causality; two-stage kernel (partial flash pass + global LSE reduction). 73–99% KV-IO reduction, ~100% partial-result IO reduction, 2.2× decode latency for speculative decoding.
**Transferable:** (a) For our MTP verify: M rows are "tree queries over a sequence KV" — the same KV-Guided Grouping says: **one (KV-head, chunk) CTA computes all M rows per tile** (our current design), and never partition M across CTAs (it would multiply KV traffic by the partition count). (b) For DSpark-style draft-tree block steps (non-causal within tree, causal across prefix): flatten the tree KV, split into even chunks, attach bitmasks — keeps partials balanced and mask IO ~0. (c) The A.4 discussion is explicit that Flash-Decoding's Q-first grouping is what wastes IO on shared KV.

### 2401.15077 — EAGLE (Li et al., 2024)
**Takeaway:** The verify structure context (tree of feature-level drafts verified in one forward pass); not a kernel paper, but defines the M-row workload our kernel serves, incl. the draft-tree depth/width tradeoffs that set M.

### 2501.01005 — FlashInfer (§ tree decoding + scheduling)
**Takeaway:** FlashInfer supports tree/spec decoding via block-sparse composed formats + load-balanced scheduling with CUDAGraph-compatible static plans — i.e., the competitor *does* batch M correctly at the scheduler level; its decode-kernel inefficiency on Orin is per-kernel (occupancy, achieved BW), not a batching flaw.

---

## Topic 6 — Edge / Jetson LLM serving & decode on unified memory

### 2506.09554 — Understanding the Performance and Power of LLM Inferencing on Edge Accelerators (Arya & Simmhan, 2025)
**Takeaway:** Systematic characterization on **Jetson AGX Orin 64GB** (shared CPU-GPU LPDDR5x): Llama3.1/Phi2/DeepSeek-R1-Qwen 2.7–32.8B; batch/seq-length/quantization sweeps + power modes. Key findings: token throughput **decreases with sequence length** (decode attention's growing KV traffic), quantization effects are non-monotonic on small models, power-mode caps directly throttle decode. The most citable Orin-specific baseline for "decode is the wall."
**Transferable:** (a) Always benchmark at **MAXN / pinned clocks** — DVFS noise is first-order on Orin (matches our gate discipline). (b) Their seq-length→throughput curves are the system-level confirmation of the KV-bandwidth roofline: on Orin the only wins are bytes/token (fp8/fp6 KV) and achieved BW.

### 2412.15352 — LLMs on Small Resource-Constrained Systems (Seymour, Kutukcu, Baidya, 2024)
**Takeaway:** Orin-family characterization + batch-testing harness; documents software-stack (backend) deltas on decode speed for small models on Jetson Orin. Useful as a cross-check for e2e expectations, not kernel-level.

### 2505.23219 — Ghidorah: Fast LLM Inference on Edge with Speculative Decoding and Hetero-Core Parallelism (Wei et al., 2025)
**Takeaway:** On Jetson (NX-class, unified memory): speculative decoding + workload split across heterogeneous cores; up to **7.6× decode-phase speedup** vs sequential. Confirms spec-decode is the edge lever and that the unified-memory bus is the contention point during decode.
**Transferable:** (a) Speculative verify (our M-row kernel) is *the* Orin strategy — it converts decode from bandwidth-bound to bandwidth-amortized. (b) Keep CPU work off the LPDDR bus during GPU decode steps (unified memory = shared bandwidth).

**Negative result:** No arXiv paper measures decode-attention *kernel-level achieved GB/s vs roofline* on AGX Orin (or any Jetson) — the closest are the two characterization studies above (system-level) and SageAttention/LeanAttention (roofline method on desktop Ampere). Any Orin kernel-BW claim we publish would be novel; measure with the gate + ncu per repo discipline.

---

## Negative results / gaps found (queries that came up empty)

1. **Original FlashDecoding tech report** (Dao et al. 2023, "Flash-Decoding") is a blog post, not on arXiv. Its content is formalized in FlashInfer §2.2 (Attention State ⊕) and DeFT §2. Cite the blog + FlashInfer.
2. **No dedicated paper on m16n8k16 M=1..8 decode mma geometry / ldmatrix fp8→fp16 fragment staging.** Code-level only. FlashDecoding++ pad-to-8 + SageAttention scale-outside-mma are the paper-level anchors.
3. **No Orin/edge achieved-bandwidth study of decode attention kernels.**
4. **No dedicated exp2/log2-softmax numerics paper.** FA2 + FlashDecoding++ + P-Cast cover it.
5. **PipeDec:** no arXiv paper under that name found (searched `abs:PipeDec`, pipelined-decode phrasing). Possibly non-existent or non-arXiv.
6. **Fermi 4-bit decode paper (2606.24031)** surfaced in searches; excluded — wrong venue/credibility.

---

## Synthesis — Top techniques for our Orin refactor (ranked by expected impact)

**Roofline first (Orin sm_87, 205 GB/s, 16 SMs, 100 KB smem):**
- fp8 KV, d=256: 512 B/token/layer (K+V). L=32K, n_kv=4 → 64 MB/step/layer → **312 µs @ 100% BW, ~390 µs @ 80%**. M=1..8 costs the *same* bytes → M-row verify is free bandwidth-wise; the only losses are serialization and split overheads.
- Compute: fp16 mma m16n8k16 at M=8 is ~0.06% of Orin's fp16 peak for this workload → we are 100% bandwidth-bound; every mma-geometry decision should be judged *only* by whether it keeps the byte streams flowing (smem ring never stalls, requests wide/contiguous, no wave imbalance).

**Ranked techniques:**

1. **Fold per-channel K descale into Q; per-token V scale into the PV epilogue** (KIVI + SageAttention + KVQuant). Descale-on-load becomes zero inner-loop work: load fp8 bytes → `cvt` to fp16 (register cvt, no multiply) → fp16 mma. Expected impact: highest — removes per-fragment FMAs and, more importantly, keeps the fp8 byte stream the *only* thing crossing the ring; correctness matches our `kv_scales/2` format. Requires Q pre-scaling per step (M×256 multiply, trivial).
2. **One (KV-head, KV-chunk) CTA computes all M rows per tile — never partition M** (DeFT KV-Guided Grouping + Medusa/SpecInfer tree-attention shape). KV read-once across the verify batch; M=1..8 all reuse one tile shape (Q padded to m16). For DSpark non-causal block steps: DeFT's flattened-tree KV splitting + bitmasks.
3. **Stream-K / equalized split-KV across all 16 SMs** (LeanAttention): assign unequal KV-chunk sizes so the wave is full; pin total partials to ~2×16 (2 waves) rather than per-request fixed splits; merge via LSE ⊕ as a *balanced tree* (Tree Attention) to cut the launch-bound merge chain (our run-2 merge-floor finding: ~4–5 µs launch/latency floor — fewer, balanced partials is the only lever).
4. **M-tile = m16 (pad Q), double-buffered 2–3-stage cp.async ring sized to 100 KB** (FlashDecoding++ pad-to-8 + LeanTile byte-budget): 32–64 fp8 KV rows per stage (16–32 KB/stage per K or V), producer streams wide contiguous 256 B rows (HND pages keep K rows dense — on Orin fp8's 256 B rows avoid the 64 B-request wall we hit with packed fp6 on Thor), consumer cvt's fp8→fp16 in the staging step. Verify ring depth vs smem: 3-stage K+V @32 rows = 48 KB + Q + stats — start there.
5. **fp8-P numerics if/when revisited** (P-Cast): S = 2⁸ (not 448), reverse KV-block iteration, ℓ in fp32 pre-cast. For e2m3-P, redo the dp(S) envelope for 3-bit mantissa (and remember e4m3-P pins P∈[0,1] — declared softmax stays structurally out for quantized P, as our run-2 note says).
6. **Heuristic GEMV-vs-TC dataflow at M=1** (FlashDecoding++ §6): keep a CUDA-core split-KV GEMV path for the M=1 leg and the TC path for M=2..8; pick via measured crossover (A100 data: M=1 CUDA +22%, M=4 TC 2×). On a 16-SM bandwidth-bound part the M=1 gap is small but the GEMV path saves the smem ring entirely.
7. **Declared/unified max for bf16-P paths** (FlashDecoding++ §3) — low expected impact on Orin (we measured softmax-chain slack ≈ 0), but free to add via calibrated φ + fallback; do it only if it simplifies the warp-role structure.
8. **Measurement hygiene on Orin** (Arya & Simmhan + Ghidorah): MAXN power mode + pinned clocks; treat ±3% same-run ratio as noise (repo discipline); keep CPU/drafter work off the unified-memory bus during verify steps.

**Competitor inefficiency quantification (what the literature supports):**
- LeanAttention: fixed-split FlashDecoding/FlashInfer schedules leave SMs idle on imbalanced loads; combine cost grows with split factor → their decode kernels underutilize the machine at long L and ragged batches.
- PAT: 53.5% average attention-latency reduction vs SOTA decode kernels (incl. FlashInfer-class) via pack-forward-merge — an upper bound on what better scheduling alone buys.
- FlashDecoding++: TC-flat-GEMM pads waste >50% of mma work (M-padded-to-64 engines); CUDA-core GEMV beats TC at M=1.
- P-Cast: FlashInfer's S=448 P-cast is non-bit-exact (precision regression risk at strong attention sinks; closed-form threshold).
- Not found anywhere: FlashInfer achieved-GB/s vs Orin roofline. Our `mjolnir verify` ncu data is the only source for that claim — keep the headline as a measured same-run ratio.

## Artifacts
- Local paper cache: FlashDecoding++ (2311.01282v4), LeanAttention (2405.10480v2), PAT (2511.22333v3), DeFT (2404.00242v4), P-Cast (2606.06521v1), SageAttention (2410.02367v9), FlashInfer (2501.01005v2).
- BibTeX can be exported on request via `arxiv_export_citations` for any subset.
