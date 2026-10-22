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
  assert guards the ring (GA10B: 100 KiB/block opt-in vs 227 KiB on
  sm_110a).

## What the kernel is

A CuTe-DSL port of FlashInfer's fast-decode GEMV kernel
(`decode.cuh`, `use_tensor_cores=False`): one CTA per
`(kv_head, kv-chunk, batch)`, 16-wide FMA dot + butterfly shuffle-reduce,
log2-domain online softmax, `cp.async` S-stage KV ring, paged or dense KV,
`M = 1..8` (MTP verify), SplitKV with chunk-normalized fp32 partials +
LSE combine. Scope gates live in the class docstring.

## The target to beat

The native sm_87 path in vLLM 0.32.0 is **FlashInfer's tensor-core decode**:
`vllm/v1/attention/backends/flashinfer.py` creates
`BatchDecodeWithPagedKVCacheWrapper(..., use_tensor_cores=True)`
unconditionally — FA2-style `mma.sync` tiles with fp8 KV dequantized to
fp16 staging. FlashInfer's own GEMV path gets `NUM_STAGES_SMEM = 2` on all
CC ≥ 8 (`DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM`).

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
3. **fp6 e2m3 KV reader**: the cache format (`mjolnir/kvfp6.py`, 4 codes
   per 3 bytes little-endian + per-(layer, h, channel) scales) is
   arch-agnostic; the Orin reader bit-assembles e2m3 → fp16 fragments
   (no `cvt` instruction exists for fp6 — ~7 int ops/element, affordable
   once the MACs live on tensor cores). 25% fewer bytes than fp8 is the
   biggest single lever on a bandwidth-bound platform. The fallback
   scratch expansion (`expand_pages_to_fp8`) is torch/Triton — works
   unchanged; the vLLM fp6 patch is already in the image's patch set
   (inert until a config requests `fp6_e2m3`).
4. **Harness + verify wiring**: correctness vs `emulate_decode`
   (tolerance-based — the GEMV/ mma reduction order is not the
   bit-faithful j-ordered spec; greedy logprob smoke as the second gate),
   then the sweep/ring/kvfmt/quant/ab stages with platform-tagged raws so
   `docs/platforms/orin/kernel/*` render like Thor's.
5. **Single-image serving**: the overlay image resolves arch at runtime
   (entrypoint exports the per-arch `TORCH_CUDA_ARCH_LIST` /
   `FLASHINFER_CUDA_ARCH_LIST` / `CUTE_DSL_ARCH`), the GEMV kernel
   DSL-compiles for sm_87 on first use (cached).

## Not portable from Thor (deliberate)

* `tcgen05 kind::f8f6f4` (the Thor fp6 read) — Ampere has no such MMA;
  fp6 read here is unpack + `mma.sync`.
* PDL / programmatic dependent launch — sm_90+.
* fp8 lm_head (`MJOLNIR_LM_HEAD=fp8`) — no fp8 tensor cores on Ampere;
  the Triton fp8 GEMM would be compute-bound. Orin keeps a bf16 lm_head.
* The Thor-tuned ring depths / split plans — retuned for 16 SMs,
  100 KiB smem and LPDDR5x latency; the sm_110 numbers do not transfer.
