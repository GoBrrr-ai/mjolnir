# Orin kernel family — the GEMV FMA decode kernel (sm_87)

This subpackage restores the archived GEMV FMA decode kernel as the Mjolnir
kernel for **Jetson AGX Orin 64GB (`sm_87`, GA10B Ampere)** — re-targeted at
that one architecture (Thor keeps the `tcgen05` `t5_*` family; this family
never runs on sm_100/sm_110).

## Provenance

* Source: git tag **`gemv-archive`** (commit `6989e30`),
  `deprecated/gemv/kernel/sm100_hd256_decode_gemv.py` (the retired FA4
  `head_dim=256` GEMV experiment, removed from `main` — see AGENTS.md
  "Retired"). Its design/analysis docs (`gemv-decode-design.md`,
  `gemv-ring-fix-bench.md`, `gemv-fi-gap-final.md`) and verify/bench
  scripts live in the tag and are the reference for the harness wiring.
* The restore is faithful except: the class is renamed
  (`BlackwellHd256DecodeGEMV` → `GEMVDecodeKernel`), the env knobs are
  Mjolnir's (`MJOLNIR_ATTN_TILE`, `MJOLNIR_ATTN_KV_STAGES`), `DescaleTensors`
  is defined locally (the kernel package must not import vLLM, and the FA4
  interface does not exist on Ampere), and an sm_87 shared-memory budget
  assert guards the ring (GA10B CC 8.7: **163 KiB/block** per the CUDA
  guide's CC 8.7 column — A100-class, *not* the 100 KiB of the CC 8.6/8.9
  GA10x parts — vs 227 KiB on sm_110a).

## What the kernel is

A CuTe-DSL port of FlashInfer's fast-decode GEMV kernel
(`decode.cuh`, `use_tensor_cores=False`): one CTA per
`(kv_head, kv-chunk, batch)`, 16-wide FMA dot + butterfly shuffle-reduce,
log2-domain online softmax, `cp.async` S-stage KV ring, paged or dense KV,
`M = 1..8` (MTP verify), SplitKV with chunk-normalized fp32 partials +
LSE combine. Scope gates live in the class docstring.

## The target to beat

The native sm_87 path in vLLM 0.32.0 is **FlashInfer's tensor-core FA2
decode** — confirmed at both tags: `vllm/v1/attention/backends/flashinfer.py`
constructs `BatchDecodeWithPagedKVCacheWrapper(..., use_tensor_cores=True)`
unconditionally, which routes to the FA2 prefill kernel
(`get_batch_prefill_module("fa2", ...)`); XQA/trtllm-gen are SM90/100/120-only.
Its Ampere shape for our model (packed qo = q_len·g = 6..24 ≤ 16 →
`CTA_TILE_Q = 16`, 1 Q-warp + 4 KV-warps = 128 threads):

* **NUM_MMA_KV = 2 → 128 KV rows per iteration** (fp8: 32 KB K+V smem per
  step; the launcher's 2-CTA/SM budget against the *100 KB-class* Ampere
  parts — Orin's 163 KB is headroom it never uses);
* fp8 KV is converted to fp16 **in registers before every HMMA**
  (`fast_dequant_f8f16x4`, ≈2–4 ALU ops per 4 values — no fp8 cvt or mma
  exists below sm_89) — a per-value tax our FMA leg pays once per loaded
  vector;
* Q stays **bf16** on SM87 + fp8 KV ("architectures with only fa2 cannot
  consume FP8 queries");
* split-KV is capped at 2 CTAs/SM (32 CTAs on Orin's 16 SMs), the LSE
  merge is a **second kernel launch**, and PDL does not exist on
  Ampere (`device_support_pdl` requires major ≥ 9) — a hard serial
  launch floor per decode step that Thor does not pay;
* FlashInfer's GEMV path (`use_tensor_cores=False`) gets
  `NUM_STAGES_SMEM = 2` on all CC ≥ 8 — but vLLM never selects it.

## The levers (ranked, from the deep research)

1. **Stage 2×+ the KV per iteration** in the 163 KB budget FlashInfer's
   tuning leaves on the table (fewer barriers, deeper pipeline, or 1
   CTA/SM with a much larger KV step).
2. **Kill the conversion tax**: convert fp8/fp6 → fp16 once into smem
   (the repack buffer FlashInfer skips at `CTA_TILE_Q = 16`) or run the
   fp32 FMA leg where each 16 B load converts once and the 128 FP32
   lanes/SM run unchained FMAs.
3. **Fuse the split-KV merge into the last CTA per (kv-head, request)**
   — no PDL on Orin to hide the second launch. (Caveat from the Thor
   data: the fused merge measured 0.56× at 4K in ncu — validate at the
   Ls that matter before keeping it.)
4. **L2 policy on the KV stream** (`createpolicy` / `evict_first` +
   `.L2::128B` prefetch): supported on 8.x, unused by FlashInfer decode.
5. **Sub-byte KV (fp6)** — the only lever that scales with L on a
   204.8 GB/s part (see the plan below).
6. **Dispatch M=1 vs M>1** — FlashDecoding++ (2311.01282) measured GEMV
   +22% over tensor cores at M=1 and TC 2× at M=4; keep both legs and
   pick by the measured Orin crossover.

Hardware reference (AGX Orin 64GB): 2048 cores / **16 SMs** / 64 3rd-gen
Tensor Cores / 5.3 TFLOPS fp32 / ~42.5 TFLOPS fp16 (derived) /
**204.8 GB/s** LPDDR5 / 4 MB L2 / 163 KB smem per block / 1.3 GHz.

## Why pure FMA cannot be the only path on Orin (the roofline)

Per KV row of 256 elements (fp8: 256 B), the GQA group g=6 does `6 × 256`
FMA for QK and `6 × 256` for PV — **12 FLOP/byte** across K+V. At Orin's
~205 GB/s LPDDR5x that asks ~2.5 TFLOP/s fp32 of a ~5.3 TFLOP/s peak
(~46%) for the M=1 decode shape — tight but achievable, and
FlashDecoding++ (arXiv 2311.01282) measured the GEMV path **+22% over
tensor cores at M=1** (their TC-vs-GEMV heuristic: M=1 → CUDA cores,
M=4 → tensor cores 2×). At the MTP verify shape (M=4 rows) the FMA
stream quadruples past the fp32 peak — **the mma.sync variant is required
for M > 1**. So the family keeps both legs and dispatches on the measured
Orin crossover: FMA for M=1, mma.sync for M > 1. The loader, ring,
split-KV plan and combine are shared by both.

Literature anchors for the rest of the refactor are in the deep-research
report (SageAttention 2410.02367: scales applied outside the mma;
KIVI 2402.02750: K per-channel / V per-token; DeFT 2404.00242: one CTA
per (kv-head, chunk) computes all M rows, KV read once; LeanAttention
2405.10480: stream-K balanced splits; P-Cast 2606.06521: fp8-P numerics).

## The plan (in order)

1. **Runner surface** (`GEMVDecodeRunner`): persistent plan buffers,
   graph-safe launch (the `mjolnir.runner` pattern), KV views, knob
   plumbing; then flip `mjolnir.platforms._SHIPPED` orin → implemented and
   point the backend's fallback default at `FLASHINFER` on sm_87.
2. **mma.sync compute variant for M > 1** (research-driven): Q rows
   (`M·g`, padded) on the mma M dimension, KV rows on N, D on k; fp8/fp6
   KV dequant in flight to fp16 fragments; per-channel K scales folded
   into Q (the Thor trick — ports unchanged), V scales in the epilogue.
   FMA stays the M=1 leg (see the roofline above). Loader, ring, split
   plan and combine survive unchanged.
3. **fp6 e2m3 KV reader**: ✅ **landed** — the packed pool reads in-kernel
   (see "The fp6 e2m3 KV variant" below): 25% fewer bytes than fp8,
   measured 1.35× the fp8 tokens/s at L=4K on the same issue count.
4. **Harness + verify wiring**: ✅ landed — `mjolnir verify` dispatches to
   this family (the correctness sweep; the headline `sweep` runs the
   events-timed same-run comparison vs the FlashInfer FA2-TC reference on
   the stock image — no ncu exists on L4T, so wall clock carries the
   headline data and the ncu-profiled stages refuse with a clear error).
5. **Single-image serving**: the overlay image resolves arch at runtime
   (entrypoint exports the per-arch `TORCH_CUDA_ARCH_LIST` /
   `FLASHINFER_CUDA_ARCH_LIST` / `CUTE_DSL_ARCH`), the GEMV kernel
   DSL-compiles for sm_87 on first use (cached).

## Measured status (2026-10-23, AGX Orin 64GB, vLLM 0.32.0 stock image)

The honest baseline (`mjolnir verify sweep` on this platform, same-run, prebuilt
launches vs FlashInfer FA2-TC on the same image): the GEMV streams
**~20 GB/s** of KV against FlashInfer's **~162 GB/s marginal** (79% of
the 204.8 GB/s LPDDR5 peak) — 0.52× at L=128 widening to ~0.14× at
262K. The kernel is bandwidth-starved, not latency- or issue-bound:

* the CTA is 96 threads (3 warps); at 3 CTAs/SM (the 49 KB ring) the
  machine runs 12 of 48 warp slots — 25% occupancy;
* the knob sweep at L=4K: deeper splits (ns 32/64) don't help (the
  extra CTAs queue behind the smem-limited residency); smaller rings do —
  `MJOLNIR_ATTN_KV_STAGES=2 MJOLNIR_ATTN_TILE=2` + ns=32 = 414 µs vs the
  457 µs base (**~10%**, 20 GB/s);
* the structural fixes are the mma.sync variant (the M>1 leg), wider
  CTAs (more load lanes per SM), and the cooperative row-chunk copy —
  all future work; the FMA leg alone cannot reach the DRAM peak.

## The fp6 e2m3 KV variant (landed)

The GEMV reads the `kvfp6` packed pool (d·3/4 = 192 bytes per row for
d=256, 4 codes per 3 bytes LE) directly:

* **g2s**: the 192-byte row moves as 12 × 16-byte chunks — the only
  cp.async size the DSL issues. Lane tx covers chunk `tx % 12`; lanes
  12-15 re-issue chunks 0-3 (benign: identical bytes to identical smem
  addresses, and the per-lane commit/wait group counts stay uniform).
  Issue count per row: 16 (vs fp8's 16) — the compression converts
  straight to bandwidth.
* **s2r**: each lane's 12-byte slice (its 16 codes span exactly 12
  bytes) loads as 3 × 4-byte smem reads (a 12-byte slice is only 4-byte
  aligned; the 16-byte cp.async of the fp8 path would need a padded row).
* **unpack**: code i of a lane's slice sits at bit 6i —
  `(B[b] >> o) | (B[b+1] << (8-o))`, `o = 6i % 8`, all static per i; the
  last code ends exactly at byte 11 so the hi-byte read is guarded. The
  value comes from the 64-entry e2m3 LUT passed as a kernel arg
  (`kvfp6.e2m3_values` — exact for every code, subnormals included).
* **scales**: the per-(layer, KV head) scale rides the existing
  `DescaleTensors` (the K descale folds into the score scale, the V
  descale into the epilogue) — no new plumbing.

Verified: 10/10 configs (page size 16-128 × M 1-8 × L 1K-64K, batch 8,
non-causal) at rel_err ≈ 0.0017 vs the fp32 reference; the fp8 sweep
24/24 unchanged. At L=4K the fp6 kernel runs 349.6 µs vs fp8's 472.7 µs
for the same tokens — **1.35×**, the 25% byte saving realized (the
kernel is issue/bandwidth-bound, so fewer bytes = proportionally faster).

## Not portable from Thor (deliberate)

* `tcgen05 kind::f8f6f4` (the Thor fp6 read) — Ampere has no such MMA;
  fp6 read here is unpack + `mma.sync`.
* PDL / programmatic dependent launch — sm_90+.
* fp8 lm_head (`MJOLNIR_LM_HEAD=fp8`) — no fp8 tensor cores on Ampere;
  the Triton fp8 GEMM would be compute-bound. Orin keeps a bf16 lm_head.
* The Thor-tuned ring depths / split plans — retuned for 16 SMs,
  100 KiB smem and LPDDR5x latency; the sm_110 numbers do not transfer.
