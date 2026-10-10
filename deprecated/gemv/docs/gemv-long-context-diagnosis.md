# Why our hd256 decode kernel loses to FlashInfer at long context — consolidated diagnosis

**Status: banked.** This consolidates the 2026-09-29 session (DRAM-bound ncu + multi-L
gated sweep) with the three research agents (arXiv, FlashInfer-source, FI-vs-GEMV source
contrast). Every lever is marked **PROVEN / REFUTED-WITH-DATA / TO-VALIDATE** so the
knowledge is not re-derived. The verdict, the corrected FlashInfer geometry, and the
next steps (the `mma.sync` TC kernel + the arXiv-lever Thor sweep) are all here.

**TL;DR verdict:** at M=1, GQA 24:4, hd256, FP8 KV on Thor, our FMA GEMV kernel is
**compute/issue-bound**, not memory-bound. The FlashInfer kernel is a **tensor-core
memory streamer** whose per-CTA compute is "free" relative to the bytes it streams.
That is the whole gap. The **`mma.sync` TC decode kernel is the primary fix**. The
memory-side levers the arXiv pass surfaced (coalesce, L2-prefetch, deep-ring) are
**candidates to measure on Thor** (not primary — memory sits at 26% of SOL), and the
two of them that were predicted to be the biggest levers (raw in-flight depth, the
16 MB HBM threshold) are **refuted by Thor data**.

---

## 1. The question

Beat FlashInfer's decode attention at **long context** (L ≥ 32768, KV > the 32 MiB L2,
DRAM-bound) with a kernel we own, on NVIDIA Jetson AGX Thor (sm_110a, 20 SMs, ~273 GB/s
LPDDR5X, 32 MiB L2). Not "fall back to FlashInfer" — a kernel we can iterate on.

Target shape: M=1, GQA 24:4 (group 6 q-heads/kv-head), head_dim 256, Q bf16, K/V FP8
e4m3, paged NHD (page 16) or dense, SplitKV + in-tree LSE combine.

---

## 2. The decisive data

### 2.1 Multi-L gated sweep (clean window, `benchmarks/raw/gemv-geom-sweep.json`)

KV bytes = 2·L·256·4 heads·1 B. best-GEMV = the fastest (tile, ns) cell.

| L | KV | FlashInfer | fa4_1cta | best GEMV | best-GEMV / FI | GEMV scaling (×) | FI scaling (×) |
|---|---|---|---|---|---|---|---|
| 8192  | 16.8 MB (L2-resident) | 266 µs | 216 µs | tile1 229 µs | **0.86 (GEMV wins)** | — | — |
| 32768 | 67.1 MB (DRAM) | 518 µs | 654 µs | tile1 745 µs | **1.44 (GEMV loses)** | 8K→32K ×3.25 | 8K→32K ×1.95 |
| 65536 | 134.2 MB (DRAM) | 775 µs | 1224 µs | tile1 1423 µs | **1.84 (GEMV loses)** | 32K→64K ×1.91 | 32K→64K ×1.50 |

- **GEMV scales near-linearly with L** (time ∝ L; achieved-BW flat ~34% of roofline).
- **FlashInfer scales sub-linearly** (BW climbs 23%→63% of the 273 GB/s DRAM roofline as
  L grows). Crossover: GEMV wins at 8K, loses from 32K on.
- **`tile1` (48 KiB in-flight) is ALWAYS the best GEMV geometry** at every L; `tile2`
  (96 KiB) and `tile4` (192 KiB, "the FI tile") are *worse* (more in-flight = more
  multi-wave serialization on the 80-CTA grid). This single fact refutes "more in-flight
  bytes → faster" (see §6, lever R2).

### 2.2 DRAM-bound ncu of the GEMV kernel (L=32768, ns=20, `--clock-control none --cache-control none`)

GEMV kernel `sm100_hd256_decode_gemv`, grid (4,20,1), 96 threads:

| metric | value |
|---|---|
| `gpu__time_duration` | 693 µs |
| `lts__t_sectors` → achieved-BW | **~103 GB/s** (≈38% of the 273 GB/s DRAM roofline) |
| `sm__inst_executed / sm__cycles_active` | 2.91 inst/SM-cycle (**~73% of the 4-wide issue limit**) |

Speed-of-light + warp state:

| section | value | meaning |
|---|---|---|
| **Compute (SM) Throughput** | **68%** | the SM pipes are the bottleneck |
| **Memory Throughput** | **26%** | 3× headroom — *not* the limit |
| Eligible warps | 2.22 / 6.66 active | warps stall ~9.8 cyc/instr (FMA dependency chain) |

**Read:** the FMA kernel's achieved-BW is roughly **flat** from the L2-resident 8K
baseline (~118 GB/s) to the DRAM-bound 32K (~103 GB/s) — i.e. it is capped by an
**L-invariant issue/FP32-FMA ceiling**, not by DRAM latency. Memory has 3× headroom, so
no amount of memory-side depth (deeper ring / coalescing / prefetch) can raise the
ceiling past ~118 GB/s — and FI's ~173 GB/s is above it. The per-KV-row FMA + butterfly-
shuffle issue pressure (16 QK-FMA + 4 shfl + 16 PV-FMA per row, serialized on the
accumulator) is the binding constraint.

---

## 3. The diagnosis

The FMA GEMV is **compute/issue-bound** at DRAM-bound lengths (Compute SOL 68% ≫ Memory
SOL 26%; 73% issue; FMA dependency-chain stalls). FlashInfer's decode is a **tensor-core
memory streamer** (mma.sync QK+PV; compute is "free" vs the bytes). To beat FI at long L
we must **cut compute-per-byte**, which means QK/PV on `mma.sync` — turning the kernel
into a streamer like FI's. Memory-side levers (deeper ring, coalescing, L2 prefetch)
cannot close a gap that is set on the *compute* side.

This is corroborated three independent ways:
1. **ncu** (this session): Compute SOL 68% vs Memory SOL 26%.
2. **FlashInfer-source** dissection: the FI fast path is the FA2 tensor-core prefill
   kernel in split-KV mode; its per-CTA compute is ~1 mma per KV row (compute-free vs
   bytes), and its sub-linear scaling comes from occupancy-pinned single-wave scheduling
   + a deep per-CTA stream — *not* from TC-vs-CC growing with L.
3. **arXiv** pass: sub-linear KV-length scaling is a **bandwidth-saturation**
   phenomenon, not a TC-vs-CC length crossover — the kernel that wins is the one that
   reaches the DRAM BW roofline; ours doesn't because its *issue* rate, not its
   memory, caps it.

---

## 4. Corrected FlashInfer geometry (fixes a 2× misread in two prior docs)

`gemv-fi-gap-final.md` §1 and `fi-decode-gemv-analysis.md` §1.2 state FI's Thor tile is
"192 KiB in-flight / 384-row tiles / 200 KiB smem / NUM_MMA_KV=6". That is **2× the real
tile** — it either counted smem+in-flight (96+96) or was derived from the K/V-shared
2-byte-KV config that does not apply to hd256/FP8/decode. The misread traced to
`cudaDevAttrMemoryClockRate` (665 600 **kHz**) being read as 665 600 **bytes** of smem.

Measured on the actual Thor (`sizeof` probe on the byte-identical 0.6.18.post1 headers +
live device attrs), the real geometry is:

| param | doc claim (misread) | **actual on Thor** |
|---|---|---|
| NUM_MMA_KV | 6 | **3** |
| CTA_TILE_KV | 384 rows | **192 rows** (3 · 4 warps · 16) |
| smem/CTA | 200 KiB | **104 KiB** (Q 8 KiB + K 48 KiB + V 48 KiB) |
| in-flight KV/CTA (one double-buffered tile) | 192 KiB | **96 KiB** (192-row K + 192-row V) |
| threads | 128 | 128 (1 Q-warp × 4 KV-warps) |
| CTAs/SM (grid) | 2 | 2 (2 × 104 ≤ 228 KiB/SM) |
| grid | 40 (one wave) | 40 = 2/SM × 20 SMs; per-CTA chunk grows ∝ L |

GPU-wide in-flight prefetch = 40 × 96 KiB = **3.75 MiB** (constant in L). Live device
attrs: `MaxSharedMemoryPerMultiprocessor` = 233 472 B (228 KiB), `PerBlockOptin` =
232 448 B (227 KiB), 20 SMs, 32 MiB L2, CC 11.0.

> **Action:** annotate `docs/fa4-hd256-fp8/gemv-fi-gap-final.md` §1 + `fi-decode-gemv-analysis.md`
> §1.2 with this correction (the conclusions there still hold — "in-flight ~192 KB" is
> approximately right as smem+in-flight — but the per-tile numbers are 2×).

---

## 5. Consolidated research inventory (all sources, all ideas preserved)

### 5.1 arXiv pass (`task/research/arxiv-gemv-decode-long-kv-bw.md`)

Bottom line: sub-linear scaling = **bandwidth saturation**, not TC-vs-CC crossover
(arithmetic intensity is L-invariant at M=1; both TC and CC decode are memory-bound).
Papers (all read unless marked lead):

1. **PersistentKV — 2606.26666.** Page-aware decode scheduling. Cost model
   `time ≈ KV_bytes/β` once occupancy ≥ Cmin = **4 CTAs/SM**. Nsight: the split kernel
   hits Mem 74.5% / L2 0.44 vs FI 62.7% / 0.12 (more CTAs → higher mem throughput).
   B1 long-context win is the largest (1.40×); **short-context (<16K) routes to FI**
   (split/merge tax dominates). GQA caveat: only G=4 calibrated; G=6 (ours) untested.
2. **Ragged Paged Attention — 2604.15464.** TPU. The **≥16 MB in-flight HBM-saturation
   threshold** (8K ctx @ d128, ~4K @ d256); below it you're latency-bound. Up to 86%
   MBU when saturated. **Access-pattern fix**: gather the KV block contiguous, then
   strided *vector* loads with an **odd-stride offset** (full DMA, no bank conflicts).
3. **Asynchronous KV Cache Prefetching — 2504.06319.** H20. `cp.async.bulk.prefetch.L2`
   (CC≥9.0 → **Thor CC 11.0 qualifies**); prefetch the next KV tile during the current
   compute; benefit **grows with L**; 2.15× kernel. **GQA caveat**: benefit decays with
   KV-head reduction (4:1 +15%, **7:1 regressed**); our 6:1 is between → measure.
4. **Helix — 2507.07120.** NVIDIA. Confirms self-attention DRAM time scales **linearly**
   in S; sub-linear via **KV sharding across more units** (multi-GPU SplitKV generalization;
   same LSE combine we have). On one Thor we're capped at 20 units → our sub-linear
   ceiling is set by per-SM MLP.
5. **Leads (not fully read):** InfiniteHiP — 2502.08910 (1M/3M-ctx decode; *algorithmic*
   KV offload/prune — tangential). HipKittens — 2511.08083 (tile/MLP for memory-bound, AMD).

Negative results (arXiv): **no** TC-vs-CC decode crossover measured vs KV length
(closest: BaseRT 2607.19438 — decode stays on non-TC kernels by *memory behavior*);
**no** persistent-CUDA-kernel-for-decode lever (PersistentKV disclaims residency —
the lever is *work distribution*); **no** stream-K/work-stealing at M=1 (that's a GEMM
technique); most 2024–26 "long-context decode" hits are **algorithmic KV
compression/eviction** (SAGE-KV, RocketKV, ShadowKV, MosaicKV) — they reduce bytes, not
kernel scaling, and can't be applied without changing the exact attention math.

### 5.2 FlashInfer-source pass (`task/research/flashinfer-decode-sublinear-scaling/summary.md`)

Kernel identity: on Thor (CC 11.0, not an SM100 target) `determine_attention_backend`
returns **`"fa2"`** — the **FA2 tensor-core *paged-prefill* kernel in FlashDecoding
split-KV mode** + a small PDL LSE-merge kernel (NOT the SM100 CUTLASS FMHA, which gates
on sm100a/103a/107a; NOT the CUDA-core `decode.cuh` GEMV). The mechanism is
**occupancy-pinned single-wave split-KV** (grid fixed at 40 CTAs = 2/SM × 20 SMs;
per-CTA chunk grows ∝ L) **×** a 96 KiB double-buffered cp.async in-flight tile per CTA.
The fixed per-CTA prologue + per-call `plan()` tax amortize over a longer DRAM stream →
BW climbs 23%→47%→63% of roofline as L grows. The **`plan()` host tax is ~190–210 µs**
(a large fraction of the 8K wall; fully amortized by 32K/64K).

Key correction (also in §4): the tile is **192-row / 96 KiB in-flight / 104 KiB smem**,
not 384-row / 192 KiB / 200 KiB. And **total in-flight bytes are not the lever** — GEMV
`tile2` (96 KiB, = FI's) *loses* to `tile1` (48 KiB) at 64K because the 80-CTA grid drops
below one wave at 96 KiB, whereas FI **co-sizes its grid (40) to its 2/SM tile
occupancy** so it always runs exactly one wave.

### 5.3 FI-vs-GEMV source contrast (this session, `explore` agent)

Confirmed kernel identities + the 228 KiB/SM smem correction, and ranked the structural
differences: (1) per-CTA in-flight 96–144 KiB (FI) vs 48 KiB (GEMV, and tile2/4 still
lose → in-flight alone isn't it); (2) **compute class: ~1 mma/KV-row (FI, compute-free)
vs tens of FMA issue slots/KV-row (GEMV, issue-bound)** — the core differentiator; (3)
L2 residency at L ≤ 16K; (4) fixed plan/merge/launch overhead dominating FI's short-L
curve (why it looks *flat*, not just shallow). Same corrected-geometry table (§4).

---

## 6. Lever-by-lever status

| # | lever (source) | status on Thor | evidence |
|---|---|---|---|
| R1 | **`mma.sync` TC kernel (QK+PV)** — cut compute/byte (FI-source #4, session ncu) | **PROVEN-PRIMARY** | ncu: Compute SOL 68% ≫ Mem 26%; FI's ~1 mma/row is the model. In progress (§7). |
| R2 | **raw in-flight depth (deeper ring / bigger tile)** (arXiv #2 "16 MB threshold") | **REFUTED-WITH-DATA** | sweep: tile1(48 KiB) beats tile2(96)/tile4(192) at *all* L incl. DRAM 32K/64K. The 16 MB HBM number does **not** transfer to LPDDR (FI saturates at 3.75 MiB in-flight). |
| R3 | **coalesce the strided 256 B/1024 B access** (arXiv #1, RPA §3.2) | **TO-VALIDATE** | ncu says memory isn't the bottleneck (26% SOL), so unlikely to be *primary*; but not yet measured. Sweep leg pending (§7). |
| R4 | **L2-oriented prefetch `cp.async.bulk.prefetch.L2`** (arXiv #3, 2504.06319) | **TO-VALIDATE** | CC 11.0 qualifies; benefit grows with L; GQA 6:1 unproven (7:1 regressed). Sweep leg pending (§7). |
| R5 | **single-wave grid/occupancy co-sizing** (FI-source #1: grid=40, 2/SM, ns≈10) | **TO-VALIDATE** | the sweep already has ns4/8/10/16/20/32 per tile — ns=10 (40 CTAs) legs are in `gemv-geom-sweep.json`; confirm the one-wave shape. Strong candidate. |
| R6 | **coarse `cp.async` commit granularity** (FI-source #3: one 48 KiB group/tile vs per-6-row 3 KiB) | **TO-VALIDATE** | reduces stall points / sustains the prefetch stream. Kernel change; candidate. |
| R7 | **amortize / cache the `plan()`** (FI-source #5) | **PROVEN-TAX (not our kernel)** | FI's own ~190–210 µs host tax; our GEMV has no plan() tax, so this is an FI-side artifact, not a lever for us. Noted for the A/B interpretation. |
| R8 | **SplitKV / enough CTAs to hit Cmin=4/SM** (arXiv #4, PersistentKV) | **PROVEN-ALREADY-DONE** | ns=20 = 80 CTAs = 4/SM already meets Cmin; the lever is the *per-CTA stream* (R2/R3), not more CTAs. Over-splitting is a tax (PersistentKV B4 near-parity). |
| R9 | **KV layout for DRAM-channel/L2 locality** (arXiv #5, PersistentKV Table 5) | **TO-VALIDATE (secondary)** | interleave 20 CTAs' reads across banks/channels; L2 0.44 vs 0.12 signal. Secondary; needs a layout experiment. |
| R10 | **lower-bit KV (INT4/2)** (arXiv #5, Helix FP4) | **NOT-ADOPTED** | already FP8 (the low-bit win); INT4/2 halves bytes but adds dequant + changes numerics. Out of scope for the exact-math goal. |

---

## 7. Next steps

### 7.1 Primary — the `mma.sync` TC decode kernel (in progress)

New kernel `docker/vllm-thor/fa4-gemv-kernel/sm100_hd256_decode_tc_spark.py` (scaffold written
this session), following `flash_fwd.py`'s `FlashAttentionForwardSm80` mma structure
(`MmaF16BF16Op(16,8,16)`, `LdMatrix8x8x16bOp`, `ampere_helpers.gemm`/`gemm_rs`) adapted
to the decode tile shapes (GQA group of 6 → padded M=16; N = KV rows / hdim; the FMA
kernel's SplitKV/combine/descale/tail scaffolding reused). QK+PV on mma (compute-free vs
bytes) → the kernel becomes a memory streamer like FI. Target geometry: 192-row tiles,
~104 KiB smem/CTA, 128 threads, single-wave grid. Gate: `mjolnir verify` vs fp32 ref; then
the 32K/64K gated sweep vs FI.

### 7.2 Direct comparison — validate the arXiv "combines" levers on Thor (requested)

Even though the ncu says the levers in R3–R6 aren't the *primary* fix (memory isn't the
bottleneck), we **measure them anyway** so the numbers sit side-by-side with the TC
kernel and with each other — "measure, don't assume." Plan:
1. Add **env-gated** GEMV variants (do not touch the default kernel):
   - `VLLM_FA4_HD256_GEMV_COALESCE=1` — R3: rewrite the strided 256 B/1024 B per-row load
     to a contiguous KV block read (gather 4–8 tokens / 2–4 KB, RPA §3.2 odd-stride
     vector load). Numerics-neutral.
   - `VLLM_FA4_HD256_GEMV_L2PREFETCH=1` — R4: `cp.async.bulk.prefetch.L2` (or the CuTe
     equivalent) for the next KV tile during current-tile compute.
   - R5/R6 are dispatch/knob changes (ns=10 single-wave; coarser commit) — reuse existing
     knobs.
2. One gated sweep at L=32768/65536 (clean window) adding legs: `baseline`, `coalesce`,
   `l2prefetch`, `coalesce+l2prefetch`, `singlewave(ns10)`, and (when the TC kernel
   lands) `tc`. Raw JSON → `benchmarks/raw/gemv-longctx-lever-sweep.json`.
3. Table: achieved-BW + wall for each leg at 32K/64K; the gap to FI; which lever(s)
   move the needle (expected: R5/R6 and the arXiv levers each recover a slice of the
   memory side but **cannot** pass the ~118 GB/s issue ceiling; TC (R1) is the only one
   that can cross it). This is the direct comparison the user asked for.

### 7.3 Doc fixes (banking hygiene)

- Annotate `gemv-fi-gap-final.md` §1 + `fi-decode-gemv-analysis.md` §1.2 with the §4
  geometry correction (192-row/96 KiB/104 KiB, not 384-row/192/200; the 665 600 kHz
  misread).
- Update `AGENTS.md` "Open work → GEMV kernel": B3 is now the `mma.sync` TC kernel
  (primary), with the R3–R6 Thor lever sweep as the measured comparison.

---

## 8. Raw artifacts (provenance)

- `benchmarks/raw/gemv-geom-sweep.json` — multi-L (8K/32K/64K) gated clean sweep,
  `window.gated=true clean=true`, best-GEMV + FI + fa4 legs (§2.1).
- ncu (this session, host docker, `--clock-control none --cache-control none`): GEMV
  kernel at L=32768 ns=20 — duration 693 µs, lts_sectors 2.24M (~103 GB/s), 2.91
  inst/SM-cycle; SOL Compute 68% / Memory 26%, eligible 2.22/6.66 (§2.2).
- `task/research/arxiv-gemv-decode-long-kv-bw.md` (arXiv, 4 papers + leads, negative
  results). `task/research/flashinfer-decode-sublinear-scaling/summary.md` (FI source,
  corrected geometry, sweep reads). `docker/vllm-thor/fa4-gemv-kernel/sm100_hd256_decode_tc_spark.py`
  (TC kernel scaffold).
