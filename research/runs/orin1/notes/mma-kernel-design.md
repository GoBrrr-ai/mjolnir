# MMA decode kernel for Orin sm_87 — design

Status: design (2026-10-23). The evidence for WHY lives in
`gemv-vs-mma-bottleneck.md` (the measured scalar-issue roofline: the ALU
pipe 59.6% of peak, the memory 8%, the flat 19–22 GB/s at every L, the
tensor pipes at zero). This doc specifies the replacement kernel's
structure, the register/smem budgets, the interfaces, and the
milestones. Open questions are marked **OPEN** — they resolve against
FlashInfer's `prefill.cuh` (the open TC decode reference) during
implementation.

## 1. The target

| | now (GEMV) | target (MMA) |
|---|---|---|
| fp8 KV BW @128K | 22 GB/s | 100–150 GB/s (the 60–80% LPDDR5 class) |
| fp8 4K latency | 453 µs | ~100–150 µs |
| fp6 128K | 12.9 ms | ~1.2–1.5 ms (the byte win converts) |
| The math pipe | the scalar issue roofline | the memory-bound (the TC math ~0.3 ms ≪ the 1.3 ms loads) |

Shape coverage (identical to the GEMV): b=1..8, M=1..8 (the MTP verify
shape), GQA 24:4 (h_r=6), hd256 only, pages 16/32/64/128, fp8 e4m3 +
e2m3 KV, split-KV via the existing Alg.1 plan + the torch LSE combine.

## 2. The instruction choice

`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` — the sm_87 has no
fp8/fp6 tensor path, so the KV codes convert to bf16 before the MMA
(the cvt cost ~5 µs at 4K, ~30 µs at 128K — the order of magnitude
below the load time; the roofline holds at every L). The fp8 Q: the
same — the runner's qpack produces e4m3; the kernel converts in the
consumer stage. No int8-IMMA variant unless the cvt traffic shows up in
ncu (it is 4 instructions per 4 values with the `fast_dequant_f8f16x4`
trick — the FlashInfer-provided device function, reusable as-is).

## 3. The kernel structure

**Grid** `(h_kv, ns, batch)` — identical to the GEMV's; `plan.py` is
reused with an smem-aware chunk floor (the chunk ≥ 256 rows so the
stage pipeline amortizes; the default ns drops from 16 to ~4–8 at 4K —
fewer merge partials, which the FlashDecoding++/DeFT line says is the
merge-side lever anyway).

**CTA**: 128 threads = 4 warps. **The warps split the stage's KV rows**
(N-direction) for both QK^T and PV — the FlashInfer TC layout
(`FA2DetermineCtaTileQ → CTA_TILE_Q=16`, the 1×4 warp arrangement). The
6 GQA q-heads occupy the M rows 0–5 of the 16-row MMA tile; the padding
rows are zero-work (the Q smem rows zeroed, the S masked to p=0) — no
extra KV traffic, and the MMA math is ~0.3 ms of a 1.3 ms budget at
128K, so the 62.5% M-utilization is irrelevant.

**smem map** (per CTA, fp8):

| region | size | notes |
|---|---|---|
| Q (padded M=16 × 256 bf16) | 8 KB | pre-scaled by the per-channel K scales (S = (Q∘s)·K̂ᵀ) — the scales folded at the pack time |
| K ring: 3 stages × 32 rows × 256 B | 24 KB | fp8 rows staged raw; `cp.async.cg` 16B/thread; the 128B swizzle |
| V ring: 3 stages × 32 rows × 256 B | 24 KB | same |
| O cross-warp scratch (16 × 256 bf16) | 8 KB | the final warp-partial reduction |
| LSE + the reduce scratch | ~1 KB | |

**≈ 65 KB** — fits `sharedMemPerBlockOptin` (99 KB) at 1 CTA/SM with
headroom; a 2 CTAs/SM variant trims to 2 stages × 16 rows (~49 KB) —
**OPEN**: the default is 1 CTA/SM × 3×32 (the deeper ring) vs 2 × 2×16
(the latency tolerance); both are env-tunable, decided by the measured
A/B (the repo convention: the knobs first, the default flips only with
the same-run win across L).

**Bytes in flight**: 1 CTA/SM × 2 in-flight stages × 16 KB × 16 SMs =
512 KB — 4× the ~123 KB the LPDDR5 needs at 204.8 GB/s and ~600 ns.
The GEMV failed on issue rate, not the MLP; this covers the MLP with
the issue rate left to the MMA.

**The pipeline**: the producer/consumer overlap via
`cp.async.commit_group` / `wait_group<N-1>` — the 3-stage ring keeps 2
loads in flight while the warps compute the current stage. No
warp-specialization on Ampere (the producer/consumer split is a
Hopper pattern); the software pipeline is the FA2-Ampere-proven form.

## 4. The math path per stage (32 KV rows)

1. **QK^T**: each warp owns 8 rows of the stage. For the d-slice k
   (16 steps): `ldmatrix` the Q fragment (the M=16 × K=16 bf16) once per
   step, `ldmatrix` the K fragment (the warp's 8 rows × 16) — the fp8
   rows load as `.b16` and convert via `fast_dequant_f8f16x4` in the
   registers — `mma.m16n8k16` → the S fragment (16×8 fp32, 4
   regs/thread).
2. **The online softmax across the warps**: the S rows' max/sum reduce
   over the KV dimension, which is SPLIT across the 4 warps → the
   per-stage cross-warp reduction through the scratch (the m/l per
   stage, the acc rescale factor per stage — the standard FA2
   warp-split-N pattern). The causal tail: the row-wise `lim` mask
   (the existing GEMV semantics).
3. **P → bf16**: the S fragment × `exp2(scale_eff)` (the
   `scale_eff = softmax_scale·log2e·qd·kd` — the existing descale
   semantics) → bf16, the padded rows forced to 0.
4. **PV**: `ldmatrix` the V fragment (the warp's 8 rows × the 8
   d-columns, bf16) → `mma.m16n8k16` accumulating the O fragment
   (16 × 8 fp32 per d-tile). **The register budget: the O accumulator
   is the full d per warp = 32 d-tiles × 4 regs = 128 regs/thread** —
   plus the pipeline/addresses lands ~180–200, under the 255 limit,
   matching the FA2-Ampere hd256's known pressure. **OPEN**: if the
   pressure hurts the occupancy, the fallback is the d-split
   accumulation through smem (the 2× the smem, the half the regs).
5. **The V descale**: the per-(seq, head) vd multiplies the O at the
   finalize (the existing semantics; for e2m3 the vd==kd==the per-head
   scale — see §6).

## 5. The epilogue

The 4 warps' O partials → the smem reduction (the scale by
`exp(lse_i − L)`), then the LSE combine: the existing torch-op merge is
kept (the plan's partials + the `_combine` path in the runner —
family-independent, and the merge is launch-bound so fewer partials is
the win). The output rows: only the 6 real M rows × the h_q heads
written to `out` (the bf16), the padding dropped.

## 6. The e2m3 path (Phase 2)

Phase 1 ships fp8; e2m3 follows once the fp8 pipeline is proven:

- **6a (the correct-first)**: the K/V rows unpack in the consumer
  stage — the 12-byte thread slice → the 16 codes → the 64-entry smem
  LUT (128 B) → bf16 → the standard MMA. The per-element cost ~4–6 INT
  ops; at 128K that is ~30–50 µs against a 0.9 ms load budget — fine.
  The per-head scale (kd) stays OUT of the values: the LUT holds the
  raw code values, the S epilogue multiplies kd (the row-constant), the
  O epilogue vd — zero per-element scale math.
- **6b (the BitDecoding upgrade)**: pack the codes directly in the
  ldmatrix fragment layout so the LUT lookup lands in the registers
  with no reshuffle — only if the 6a's cvt/LSU pressure shows in ncu.

## 7. The interfaces (the scaffold in this commit)

- `--impl gemv|mma` on the harness (the default `gemv` until the MMA
  wins a same-run A/B across L — the repo's flag-first rule); the
  runner dispatches on it, the compile key carries it.
- The verify stages need no changes: the legs pass `--impl mma` as
  flags for the A/B; the plan/combine/kv formats ride along.
- The kernel module: `mjolnir/arch/orin/mma.py` (added to the
  file-length exclude — the traced-kernel family).

## 8. The milestones

1. **M1 — the fp8 skeleton**: the CTA/warp structure, the smem ring,
   the QK^T+PV MMA, the single-stage (no ring) — correctness vs the
   reference at b=1 M=1 L=4K. The gate: 24/24 like the GEMV.
2. **M2 — the pipeline + the splits**: the 3-stage ring, the plan's
   smem-aware chunking, the cross-warp softmax/reduction. The gate:
   the ncu memory throughput > 30% at 4K (vs the GEMV's 7.9%).
3. **M3 — the full matrix**: M=1..8, the pages, the varlen, the fp8 Q.
   The gate: the correctness stage green on both Q paths.
4. **M4 — the A/B vs the GEMV and FI**: `verify hndsweep/sweep --legs`
   with the --impl flags; the default flip only on the same-run win
   across L.
5. **M5 — e2m3** (6a, then 6b if the ncu says so).

## 9. The risks

- **The register pressure** (the 128-reg O accumulator): the fallback
  is the d-split-through-smem; the FlashInfer's hd256 config is the
  existence proof either way.
- **The cross-warp softmax cost**: the per-stage smem reduction × the
  stages — if it shows, the stage rows go 32→64 (the fewer, bigger
  reductions) at the ring-depth cost.
- **The TVM-FFI signature**: the compile-cache machinery bakes the
  dims — the new kernel reuses it unchanged (the key already carries
  the impl after this commit).
- **The 96-vs-128-thread lesson**: the FlashInfer's GEMV launches 128
  with 96 active — our CTA is 128 all-active; no dead threads.

## M1 debug log (2026-10-23, session 1)

The skeleton compiles through four real errors so far, three fixed:

1. **The staged control flow**: the variables assigned in a staged if/else
   must pre-exist (the DSL's rule) — fixed.
2. **The V smem layout**: the [d][32 rows] tile can't take the (8,64)
   swizzle atom — the 32-col blocks need swizzle(2) + the (8,32) atom
   (the reference's own rule) — fixed.
3. **The retile bridge**: the mma fragments' layouts differ from the
   smem-copy atoms'; the copies need `smem_thr.retile(fragment)` as the
   dst (the reference's tSrQ_copy_view etc.) — fixed.
4. **OPEN — the swizzled-smem manual indexing**: `sK16[r, c] = ...` on a
   composed/swizzled layout fails the coord-rank check (the layout is
   ((8,..),(64,..)) — 4 modes; the 2-elt coord mismatches), and the
   ldmatrix src from the [d][32] layout reports the 16-bit alignment.
   The fix (next iteration): the dequant goes through the register
   fragments — the fp8 flat-view loads (the GEMV's s2r idiom) -> the cvt
   -> `StMatrix8x8x16bOp` into the swizzled bf16 tiles (the warp module
   ships StMatrix; the reference never hand-indexes swizzled smem). The
   Q stage: the same (the fp8-style flat -> the fragment -> stmatrix, or
   the universal-copy into the UNSWIZZLED smem for the Q and let the
   ldmatrix read it — the swizzle is the bank-conflict optimization, not
   a correctness requirement; M1 correctness first, M2 re-swizzles).

The lesson recorded: on this DSL, swizzled smem is written and read only
through copy atoms — the manual element indexing is for flat layouts.

## The M-dimension revision: MTP=7 from day one (2026-10-23)

The MTP=7 verify shape = q_len 8 (7 drafts + the real token) — the GEMV
family's max (max_q_len <= 8). The M tile math:

- The M rows = q_len x 6 GQA heads, TOKEN-MAJOR (row = m*6 + r, the r =
  the head-in-group) — matching the reference's (m, h_k, h_r, d) view.
- n_tiles = ceil(q_len * 6 / 16): 1 tile at q_len<=2, 2 at 3..5, **3 at
  6..8** (the q_len 8: 48 rows = exactly 3 tiles, zero padding waste!).
- The Q smem: (n_tiles*16, 256) bf16 — 24 KB at the q_len 8; the total
  smem 73 KB <= 99 KB. The acc_O: 3 tiles x 32 fp32 regs/thread = 96 —
  plus the working set ~150-180 of 255, feasible; the tiles are
  constexpr so the binary is per-q_len (the compile key already carries
  seqlen_q).
- The loop order: the stage OUTER, the tile INNER — the acc_O for all
  the tiles persists across the stages (the registers); the QK gemm,
  the softmax (the per-tile row stats through the same sStats), and
  the PV all iterate the tiles per stage.
- The causal mask becomes PER-TOKEN: the row (m, r) attends to the keys
  gk <= kv_len - q_len + m (the GEMV ref_decode contract, L - M + 1 +
  m); the M=1 case degenerates to the no-op (the gk <= kv_len - 1).
  The mask needs the stage's global row (kv_start + row0 + c) and the
  row's token m — both available in the softmax helper.
- The epilogue: the row R -> (m, r) = divmod(R, 6); the O at
  o_off + m*row_stride + (kv_head*6 + r)*D; the LSE per (token, head)
  at (split, qo_head, q_row_base + m) — the GEMV's exact coords.
- The row-stat writers: the row r's quad leader is the lane (r%8)*4
  (the atom's D layout: the rows = lane//4 and lane//4+8).

The M=1 path is the n_tiles=1 special case of all of the above — the
skeleton gets reworked once, not twice.

## M1 debug log (2026-10-23, session 2 — the kernel compiles and runs)

The M1 skeleton now compiles and launches (the harness `--impl mma
--check` executes end to end). Bugs found and fixed on the way:

1. **The DSL list-comprehension ICE** — comprehensions over
   `range_constexpr` ICE ("should be preprocessed by preprocessor"):
   explicit append loops.
2. **`continue` inside `range_constexpr`** is a SyntaxError at trace
   time (it is not a Python loop): nest into guarded ifs.
3. **`_acc_mn_view` missing `cute.composition`** — the reference's
   `_make_acc_tensor_mn_view` composes the rebuilt layout with the
   fragment's actual layout; without it the view indexes the wrong
   registers AND the identity-tensor coord path ICEs
   (`'GetScalarsOp' object has no attribute 'type'`).
4. **ldsm alignment through a re-wrapped smem tensor** —
   `make_tensor(iterator, layout)` drops the 16B assumption the
   ldmatrix atom needs; use the allocator's tensors directly, and build
   the transposed V view by composition over a plain-allocated tile
   (a directly-allocated transposed layout loses the alignment too).
5. **GEMM operands must be the MMA-side views** (`tSrK`, `tOrVt`), not
   the copy-atom retiles (`*_copy`) — the reference's gemm calls.
6. **`GEMVKey` had no `impl` field** — the runner's module-level
   `_COMPILED` map handed the GEMV's binary to the MMA runner when both
   ran in one process (every early "pass" was the GEMV's kernel). The
   key now carries `impl`.
7. **The O epilogue double-counted `kv_head*G*D`** (it is already in
   the per-row `head*D` term) — alternating kv-heads landed zero or in
   other heads' slots.
8. **The quad reduction was a swap, not a reduce** —
   `shuffle_sync_bfly` must be COMBINED with the local value
   (`op(val, shuffle(val, 2))` then `op(val, shuffle(val, 1))`, the
   reference's `_threadquad_reduce`), not assigned from the shuffle.

### The warp-arrangement decision (the M-dimension revision revised)

The planned `(1,4,1)` N-split breaks the PV: with the warps splitting
the stage rows, each warp's S fragment covers only 8 of the A-atom's
K=16. The kernel now follows the reference's structure instead: **the
M tile is 64 rows and the 4 warps split the M** (`(4,1,1)`, one
16-row atom per warp). q_len*6 <= 48 pads into one tile, the row
stats are quad-local (no cross-warp smem stats at all), and the PV's
A operand sees the full K. The M-dimension revision's n_tiles/tile
loop is gone — MTP=7 fits one tile by construction.

### Open (next session's first task)

`rel_err ~1.0` on the random case; the known-answer probes show the
row_sum is now correct (8.516 vs 8 at chunk 8 — ~6% high, worth a look)
but the PV accumulator lands ~1/8 of the expected value, uniformly over
d and heads, for both the S->A paths tried (the reference's
logical_divide and a P-through-smem round-trip). The QK/softmax/
reduction side checks out (row_sum via LSE); the remaining suspect is
the P fragment's value placement relative to the A-operand layout, or
the V B-fragment's k-slice mapping. Sharpest next probe: a
PV-only standalone kernel with the identity P and the known V.
