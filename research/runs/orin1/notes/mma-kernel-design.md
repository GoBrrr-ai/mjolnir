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

## M1 debug log (2026-10-24, session 3 — the bug is localized)

The LSE now carries `row_max * scale_eff` (the GEMV's m_run is the
scaled max; mine is raw — the epilogue was missing the factor).

Probes (all with the identity-style known-answer inputs):

- **The PV gemm path is EXONERATED**: the single-key selector probe
  (Q[0]=100, K[5][0]=1, v=5 everywhere) gives O = 5 x 0.94375 — the
  exact softmax math (p_else = exp2(-9.02), 31 keys). Both k-atoms
  correct, no k-scrambling.
- **The QK path is per-column correct**: the d-sweep (Q/K hot in a
  single d column, dq in 0..255) gives the selector O for every dq.
- **The row_sum (via the split LSE) is per-row CORRECT**: the hot row
  reads 1.06 (the S seen), the cold rows 32 (the S=0) — exactly right.
- **The A-fragment's P values are WRONG per row**: every row's
  accumulator lands at the v-mean (the uniform-P signature) — the acc
  sees the S=0 distribution while the row_sum sees the true one.

The row-sweep (one hot row at a time, v[k][0]=k) shows the INVERSE
pattern: the hot row's O = 14.62 (the v-mean x 0.9437) and the cold
rows' = 0.4844 (= 15.5/32). Conclusion: the acc_S values are consumed
correctly by the softmax (row_sum) but land SCRAMBLED — row-dependently
— in the P fragment that feeds the gemm. Both S->A conversions tried
(the reference's logical_divide and the P-through-smem round-trip) fail
identically, so the corruption is upstream of the conversion: either
the acc_S values are already misplaced per thread (the QK gemm writes
S at slots whose (row, col) mapping differs from what the identity-tensor
coords claim), or the bf16 fragment's flat order does not match the
acc_S's logical order the way the epilogue assumes.

Sharpest next probe: dump the O's full d-profile for the v[k][d]=d/256
case and compare against the predicted per-slot permutation; or verify
the acc_S slot mapping directly by making S = f(row, col) injective
(Q = the identity over rows AND the K = the identity over cols with
distinct magnitudes) and reading the permutation off the O.

### The S-magnitude dependence (the final probe of session 3)

- Q=0, K=V=1, L=32:        O = 1/32   (the acc = 1, the row_sum = 32)
- Q=100*I, K=I, v[k][0]=k: O = v-mean (the acc = 15.5 for EVERY row, the
  row_sum correct per row: the hot 1.06, the cold 32)
- Q=100 (all), K=V=1:      O = 0.0    (the acc = 0)

The rP fill via the reference's layout-aware `store(load().to())`
matches the flat version exactly (not the flat-order scramble).

The acc's value is a function of the S magnitude in a way a correct
pipeline forbids: the same p=1 P and v=1 V give the acc = 1 (the S=0),
the 15.5 (the S=100 diagonal), and the 0 (the S=25600). Combined with
the row-sweep inversion (the hot row's acc = the v-mean, the cold
rows' = the v-mean/32), the prime suspect is the acc_S fragment's
slot <-> value correspondence itself: the QK gemm writes S at
per-thread slots whose logical (row, col) mapping does not match what
the identity-tensor coords and the _acc_mn_view claim, so the softmax
(reducing over cols per row) and the P->A conversion (consuming the
same fragment as (m, k)) disagree. Next: the injective-S probe
(Q and K with distinct magnitudes per row/col) to read the actual
permutation off the O, then fix the slot mapping at the source.

## Root cause (2026-10-24): two bugs, both found by oracle bisection

The oracle ladder (dbg21-dbg31, /tmp/opencode/) proved every stage correct
in isolation before the real kernel passed:

1. **n_row_slots** (dbg21-25): the softmax/epilogue loop bound was
   `cute.size(partition_shape_C[0])` = 4 (the full (2,2) mode-0 shape:
   the n0-pair AND the m0-pair), but the MN view's mode 0 is
   `shape[0][0] * shape[1]` = 2 (the reference's row_max sizing). With 4,
   iterations r=2,3 resolved through the MN view's divmod to the SAME flat
   slots as r=0,1: O was normalized twice (acc/row_sum^2) and the P writes
   landed on foreign slots. Fix: `size(shape[0][0]) * size(shape[1])`.
2. **page-table g2s** (dbg26-28): the paged addressing was
   `page*PS + (gk_row - page*PS)` — algebraically `gk_row`. The block
   table cancelled out; the kernel read the wrong physical rows whenever
   the table is not the identity. Every pre-dbg26 probe passed because it
   built an identity table. Fix: `page*PS + gk_row % PS`.

Oracle methodology that worked: standalone @cute.kernel probes with
injective codings (pow2 x odd for S; delta(d==k) V for O), dumping
per-thread fragment slots to gmem, compared against torch refs computed
over the SAME fp8-rounded values. dbg30 proved the MN view's (r,c) ->
flat map directly (fill acc with the flat index, read through the view).
Debug slots written into the dump tensors must be excluded from the
comparison (they were 100% of the residual "mismatches").

## M1 gate: PASSED (2026-10-24)

- harness --check: L 32..4096, b 1..8, q_len 1..8 (the full MTP=7
  matrix), ps 16/32/64/128, causal + non-causal: rel_err ~0.002 (tol 0.05).
- gemv parity on the same cases: ~0.0017-0.002.
- bf16 KV correctly refused (M1 scope: fp8 KV only).
- probe ladder dbg9-20: clean after the fixes.
- commit d6a9775.

## M2 pass 1 (2026-10-24, commit 30afc4b): swizzle + 4-wide dequant — 2.4x + 1.24x

**The bottleneck was the L1/TEX pipe, not memory.** ncu at L=4096 (M=1):
Memory (SpeedOfLight) 76.1% but that is the L1/TEX (76.6%); DRAM
"Max Bandwidth" 12.2%; stalls: short_scoreboard 8.87 of 13.09
cycles/instr (68%), wait 2.03, long_scoreboard only 0.33. The plain
512 B row-major smem rows put every ldmatrix 8-row group in the same
banks: an 8-way conflict per ldmatrix. The grid is (HK=4, ns=16, b) =
64 CTAs on 16 SMs (1 CTA/SM, 80 KB), num_splits is always ns_cap=16
(kv_chunk = ceil(L/16)) — the waves do not change per-SM work.

Fixes (commit 30afc4b):
1. sQ/sK16/sV16 carry the FA2-style (8, 64)-element atom with
   Swizzle<3,3,3>; sVt composes over the swizzled storage. Smem size
   unchanged (80 KB, 2 CTAs/SM). A barrier is REQUIRED between the sQ
   fill and the (now hoisted) Q ldmatrix: the flat fill scatters each
   sQ row across all 4 warps (M=8 race caught by --check, M=1 passed
   by timing luck).
2. Q fragments load once before the stage loop (the per-stage reload
   was 32 KB of ldmatrix traffic per stage).
3. Dequant: FlashInfer/Marlin fast_dequant_f8f16x4 (prmt + shift/mask +
   one packed-bf16 fma per 4 values). Verified bit-exact on sm_87 with
   a CUDA probe (normals + zero exact; subnormals -> normal-formula
   neighbors, the same documented caveat). GOTCHA: the C++ MASK1>>7 is
   an ARITHMETIC shift on a signed int; the sign extension makes the
   mask 0x7F007F00, NOT 0x01000100 (dbg32 caught it: b0's bits were
   dropped).

Results (same-run A/B, /tmp/opencode/ab_sweep.py):
- mma vs gemv, M=1: 2.21-2.44x across L (was 1.03-1.11x)
- mma vs gemv, M=4: 3.07-6.96x (was 2.02-2.26x)
- effective KV: 23 -> 74 GB/s (36% of 204.8)
- ncu: total warp-instr 4.63M -> 2.78M (-40%), ALU 50% -> 37%,
  short_scoreboard 8.87 -> 1.48, total stalls 13.1 -> 4.5 cycles/instr

vs FlashInfer (fi_driver.py, same recipe HQ24/HK4/D256/PS16/NHD,
events-timed 200 iters): mma WINS at L<=4096 (M=1: 2.26x at 1K,
1.05x at 4K; M=4: 2.24x at 1K, 1.13x at 4K), trails at 16K+
(M=1 L=262144: 7338 vs 3264 us). FI is DRAM-bound at large L
(165 GB/s = 80% of peak); the kernel is issue-bound at 74 GB/s.

## M2 pass 2 plan: the issue-bound end game

Per-warp-stage instruction budget: 1357 vs the ~1575 issue slots —
the kernel sits AT issue capacity. Remaining cuts, in order:
1. Register-path dequant (drop sK16/sV16, 48 KB smem): ldmatrix on
   fp8-as-bf16 + frag_layout_swizzle_16b_to_8b (FI's approach) —
   kills the 128 STS.16/thread/stage and halves smem traffic; makes
   SR=64 affordable (sQ 32 + fp8 32 = 64 KB) which halves all
   per-stage fixed costs. The big one.
2. 2-accumulator QK chain (wait stall 2.21: the 16-deep serial HMMA
   accumulation; split k into 2 halves, merge before softmax).
3. Vectorized STS for the dequant output (u32-pair stores through the
   swizzled layout; the 4-aligned c groups stay within one 16 B
   swizzle chunk).

## M2 pass 2 (2026-10-24, commit c3e2715): 2-accumulator QK — 1.12x

Clocks LOCKED at 1.3 GHz (the earlier ncu %s ran at the 306 MHz idle
clock and distorted the balance — the wall-clock was right). Clean
profile at L=4096 M=1 (137 us duration, matching the wall clock):
DRAM Max Bandwidth 27.7%, L1/TEX 53.5%, L2 hit 81.7%; stalls
cycles/instr: long_scoreboard 1.96 (the DRAM latency — now #1),
short_scoreboard 1.79, wait 1.82. The kernel is stall-bound on all
three fronts, NOT issue-bound (the true-clock issue demand is ~24% of
the scheduler capacity — the earlier "issue-bound" reading was the
idle-clock artifact).

Pass 2: the QK accumulation split into 2 independent chains (the k
halves), merged with 16 adds — the 16-deep serial HMMA chain was the
wait-stall source. M=1: 3.29-3.72x vs gemv (the 82 GB/s effective);
M=4: 4.43-7.82x. vs FI (locked clocks): wins at L<=4096 (the 2.8x at
1K, the 1.0-1.14x at 4K), trails 2.0x/1.54x at 16K+ (the FI is
DRAM-bound at 165 GB/s). The mma's M=4 costs +2.6% over M=1 (the
tensor cores absorb the GQA rows) vs the FI's +30% — the gap shrinks
with MTP depth (the M=8+ likely wins everywhere — worth measuring).

## M2 pass 3 (the 16K+ endgame): register path + fp8 ring

The DRAM latency (long_sb 1.96) needs a cp.async ring; the ring costs
+16 KB smem which forces 1 CTA/SM at the current 80 KB (the 96 KB ×
2 > the ~164 KB/SM) — halving the latency-hiding warps. The register
path (the dequant straight into the mma fragments from the fp8 smem,
no sK16/sV16) drops the smem to 48 KB so 48 + a 16 KB fp8 ring =
64 KB keeps 2 CTAs/SM — and kills the 128 STS.16/thread/stage and the
extraction ALU. The fragment remap (which fp8 bytes land in which
fragment register after ldmatrix on fp8-as-bf16 — the FI's
frag_layout_swizzle_16b_to_8b) needs the dbg probe ladder before the
rewrite: fill sK8 injectively, ldmatrix, dump the 4 regs/lane, derive
the lane->(d-pair, key) map, then the prmt recipe per fragment.
The 2-acc + the STS-pair stores ride along.

## Pass-3 probe state (2026-10-24 evening): the ldmatrix-on-fp8 fragment map

dbg33b/dbg33c (/tmp/opencode/): the fp8 K tile (SR=32, D=256, plain
row-major) viewed as (SR, D/2) bf16, run through the kernel's exact
smem_copy_K ldmatrix, every tSrK element's raw bytes dumped per lane,
with two fills (value = d, then value = key). Findings so far:

- Each tSrK element (one 4-byte register) carries 4 CONSECUTIVE fp8
  values of ONE key: [d, d+1, d+2, d+3] — the dequant is the x4
  recipe on the raw register, no cross-register shuffle needed.
- d0 (the element's first d) = 4*(l%4) + 16*(i%2) + (i>=8 groups of
  +32 — the exact high-index pattern unverified); key = (l//4) +
  offset(i) where offset = [0, 0, 1, 1, 9, 9, 17, 17, ...] for the
  first 8 elements — the +1 on the odd matrix indices is the one
  unjustified term left.
- The element index i (0..127 per lane) = the flat cute fragment index
  = 8 per k-step (4 n-tiles x 2 regs); the exact (k-step, n-tile, reg)
  decomposition is the remaining decode (extend dbg33c: dump all 128
  elements, fit d0/key against every (ks, nt, reg) hypothesis, or just
  build the mapping as an explicit constexpr table).

The register path then is: g2s -> sK8 (fp8, swizzled for the ldmatrix
rows) -> per element: prmt-recipe on the raw 4 bytes (already the
right consecutive-d grouping!) -> write the bf16 fragment. No sK16,
no STS, no extraction. Plus a 16 KB fp8 ring fits (48+16=64 KB, 2
CTAs/SM) to hide the DRAM latency (long_scoreboard 1.96, the #1 stall
at locked clocks).

### dbg34 (2026-10-24 night): the fragment map, decoded structure

The earlier decode had a bug: each tSrK element is ONE u16 = 2 fp8
bytes (the (SR, D/2) bf16 view's element), not a 4-byte register.
dbg34 dumps slot[tidx, k*16 + e] = tSrK[None, None, k][e].bitcast(u16)
with the d-fill and the key-fill. The clean table (lane 0, k-step 0):

  e0: d=(0,1) key=0    e1: d=(2,3)   key=0    e2: d=(16,17)  key=0   e3: d=(18,19) key=0
  e4: d=(0,1) key=1    e5: d=(2,3)   key=1    e6: d=(16,17)  key=1   e7: d=(18,19) key=1
  e8: d=(0,1) key=9    ...

- each element = 2 CONSECUTIVE fp8 of ONE key (the view-column pair
  (2dv, 2dv+1)); the d0 = 4*(l%4) + [0, 2, 16, 18][pair] (+ 32*k-step
  in the (SR, D/2)-view scaling); the key = (l//4) + offset(pair),
  offset = [0, 1, 9, 17] for pairs 0..3 — the +1 on the odd pairs is
  the last unjustified term (likely the x4 matrix row order).
- PER_K=16 in the dump bleeds the next k-step into slots 8-15 of each
  (the real per-k element count is 8 = 4 n-tiles x 2 regs); slots
  k*16+e for e>=8 belong to k-step k+1.
- The lane scaling: lane 1 = d+4 (the (l%4)*4 view-col base), lane 4 =
  key+1 (the l//4 row base) — both confirmed.

Next session: fit the (k-step, n-tile, reg) -> slot decomposition from
the full 256-slot dump (the python fit over the captured tables), then
the register-path rewrite: g2s -> sK8 (swizzled for ldmatrix rows) ->
per element: the x4 prmt recipe on the 2 consecutive fp8 -> the bf16
fragment. No sK16/sV16, no STS, no extraction; the 16 KB fp8 ring
fits (48+16=64 KB, 2 CTAs/SM) to hide the DRAM latency.

### dbg34 decode COMPLETE (same night): the register-path recipe

Corrected reading (slot = one u16 = 2 fp8; real per-k elements = 8,
the PER_K=16 dump bleeds k-step k+1 into slots 8..15 of each k row):

per (lane l, k-step ks, element e in 0..7):
  n-tile nt = e//4 ; reg r = (e//2)%2 ; half = e%2
  key(l, e) = l//4 + nt                       (verified lanes 0/4/8)
  dv(l, e)  = 8*r + (l%4)*2 + half            (the view-column index)
  the element's 2 fp8 = K[d = 2*dv + half], K[d = 2*dv + half + 1]
  -> the register (elements 2r, 2r+1 of the (nt, ks) pair) holds the
     4 CONSECUTIVE d [2*(8r + (l%4)*2), +4) — the x4 prmt recipe
     dequantizes it in place to the 2 bf16 the mma wants.

So the register path needs NO byte permutation at all: the ldmatrix
(on the (SR, D/2) bf16 view of the full-density fp8 tile) already
delivers each 4-byte register as 4 consecutive d. The pipeline:
  g2s fp8 -> sK8 (16 KB? no: SR*D bytes, swizzled for the 16 B rows)
  per k-step: ldmatrix.x4 -> 8 elements = 4 registers
  per register: prmt + shift/mask + fma (the x4 recipe) -> the bf16
  fragment registers, fed straight to cute.gemm. No sK16/sV16, no
  STS, no extraction. Smem: sQ 32 + sK8 8 + sV8 8 = 48 KB; a 16 KB
  fp8 ring (48+16=64) keeps 2 CTAs/SM and hides the DRAM latency
  (long_scoreboard 1.96, the #1 stall at locked clocks).

Caveat to verify in the implementation: the (SR, D/2) view halves the
mma's k-extent per pass (the ks spans d [32ks, +32)), so the Q side
must use the same k indexing (partition_A over the matching half-D
view) or the loop runs D/2 k-steps of the full 16-k atoms. The
correctness gate is the harness --check matrix either way.

### The M=8 data (2026-10-24 night): FI is flat in M, the fight is pure DRAM

FI at M=8 == FI at M=4 (157.8 us at 1K both; 4315.5 vs 4317.4 at 262K):
its prefill path packs all query rows into tensor-core tiles, same as
ours. The mma is also flat in M (6801 us at M=8 vs 6540 at M=1, 262K).
So M-shape does NOT change the verdict — the single remaining fight is
the 16K+ DRAM utilization: mma 82 GB/s (40%) vs FI 164 (80%).

The full mma/FI table (locked clocks, mma/FI us, ratio):
  M=1: 1K 44.8/125.9 (2.81x WIN) | 4K 125.7/126.7 (1.01x WIN)
       | 16K 429.8/227.8 (1.88x) | 262K 6540/3309 (1.98x)
  M=4: 1K 58.4/157.8 (2.70x WIN) | 4K 137.7/158.5 (1.15x WIN)
       | 16K 448.6/293.5 (1.53x) | 262K 6696/4317 (1.55x)
  M=8: 1K 69.8/157.8 (2.26x WIN) | 4K 149.1/157.9 (1.06x WIN)
       | 16K 461.6/298.3 (1.55x) | 262K 6801/4316 (1.58x)
(mma beats FI at L<=4096 in EVERY shape; the 16K+ gap is 1.5-2.0x,
shape-independent — exactly the register-path + fp8-ring target.)

For the M4 gate: GEMV is irrelevant now (24-50x slower at M=8); the
gate is FI, and the pass-3 lever is the only remaining one.

## Pass 3 measured: BOTH levers are negative (2026-10-24, locked clocks)

### The 2-stage fp8 ring (96 KB, 1 CTA/SM) — +20% WORSE at depth

Correctness first: the extracted `g2s_stage` + prologue/prefetch pipeline
passed the full extended matrix 42/42 (all ps, M 1/8, causal+non-causal,
L 32..131072 incl. non-multiples of SR, b=4, paged). The clamped-row0
prefetch is branch-free; the trailing sync after dequant protects the
buffer reuse. Patch archived: `/tmp/opencode/ring.patch`.

Timing (same-run, prebuilt launches, M=1): 16K 516.3 µs vs the
single-buffered 429.8 (+20%), 262K 7922.6 vs 6540 (+21%). Effective KV
BW 82 -> 68 GB/s. The stall math predicted it: 4 warps x 0.20 issue <
8 warps x 0.17 — the lost warp dominates the hidden g2s latency.

### The register path (K side, 2-pass even/odd) — +19% WORSE at depth

The mechanism, after three oracle probes (dbg35/36/37, all committed to
/tmp/opencode):
- The B side is CORRECT and cheap: ldmatrix on the (SR, D/2) bf16 view
  of the full-density fp8 tile delivers each u32 register pair as 4
  consecutive d; the even pass masks the LOW bytes into the recipe
  position ((u & 0x00FF00FF) << 8), the odd pass needs NO shift (the
  odd bytes already sit there) — ~160 instr/lane/stage vs ~320.
- The A side MUST match in view units (the view halves the mma k-extent:
  A k=256 vs B k=128 is unrunnable), which forces the A fragments to be
  filled directly: a generic CopyS2ROp tiled-copy does NOT match the mma
  A-fragment layout (dbg35: duplicated slots), and a composed strided
  view mis-decodes (dbg37: the j-stride collapses). The decoded m16k16
  A-fragment map (dbg36, from the known-correct LDSM load): per 8-element
  k-step, row m0 + 8*((e//2)%2), col k0 + (e%2) + 8*(e//4), m0 =
  16*warp + lane/4, k0 = 2*(lane%4). With the warp term the fill is
  correct: full matrix 42/42, rel_err 0.0020.
- BUT the two A fragments cost +32 live registers on a kernel at the
  ptxas ceiling; merging the passes into one acc (dropping acc_S2, -16
  regs) made it WORSE still (+28%) — the 2-acc HMMA chains are
  load-bearing even with the dequant between them. Net: the ~160 saved
  dequant instructions never show up; the wall at depth is NOT the ALU.

Patch archived: `/tmp/opencode/regpath.patch`. Reverted to c3e2715.

### The standing (same-run vs FI ONLY, GEMV dropped from comparisons)

fi_compare (2026-10-24): mma/FI — 1K 0.53/0.49/0.60 (M=1/4/8, WIN),
4K 1.18/1.04/1.09, 16K+ 1.56-2.05x. The mma wins L<=1024, is
parity-to-behind at 4096, and the 16K+ gap is ~1.6x (M>=4) / 2.0x (M=1).

### M4 gate: WIRED, verdict NO FLIP

`stage_orinbench` gained the opt-in `mjolnir-mma` legs (correctness gate
+ prebuilt-launch timing + series; `--legs mjolnir-mma,fi`). The gate
run (sweep/20261024-110825, raw JSON committed): mma vs FI ratio
FI/mma = 1.73x at L=128 (WIN), 0.88x at 4K, 0.53x at 8K, 0.49-0.51x at
16K+ — the mma does NOT win the geomean, so the default stays GEMV
(whose serving-path full numbers remain the shipped series). The mma is
a measured, gated opt-in.

### ns=8 vs 16 (pass 3d): a non-lever

The merge kernel is ~4-5 us of launch floor (~1% at 16K). The harness
--max_splits path for the mma bench is half-wired (mO_partial shape
mismatch — the harness fixes ns_cap=16), not worth completing: the
lever is bounded at ~1%.

### What would actually close the 16K+ gap

The physical read rate: the mma pulls KV at ~41 GB/s (2 CTAs x 8 KB in
flight per stage) vs FI's ~82. Doubling bytes-in-flight needs either
+smem (the ring: measured negative at 1 CTA), +CTAs (ns is capped by
the grid at 64 already), or a wider copy engine (TMA does not exist
below sm_90; cp.async caps at 16 B). The honest conclusion: on sm_87
the 16K+ decode band belongs to FI unless the KV layout changes
(the Thor-side conclusion — wider requests need a vLLM cache-format
change — applies here too).

## The sQ-aliased KV ring: the 16K+ serialization broken (2026-10-24, later)

The single-buffered stage loop idles the DRAM during dequant+mma. The
2-stage ring measured +20% WORSE at 1 CTA/SM (96 KB); the fix: the ring's
second buffers ALIAS sQ — the Q tile is dead once its fragments are
hoisted (registers), so its 32 KB host the prefetch buffers. Total stays
80 KB -> 2 CTAs/SM AND the g2s overlaps the compute. The 3-stage variant
(a third pair in sQ's remainder) measures PARITY with the 2-stage: the
in-flight bytes = the wait depth (2 groups), not the buffer count.

The in-flight model now fits every measurement: BW ~= (outstanding/SM ÷
~143 KB LPDDR5 latency-BW product) x 204.8 GB/s. 2-stage ring: 64 KB/SM
-> 92 GB/s measured (the harness's 74-82 was the pre-ring duty cycle).
FI-TC: 128 KB/SM -> ~162 (the recon: task/research/fi-sm80-fp8-decode-
summary — FI-TC double-buffers K and V separately with split waits,
raw fp8 in XOR-swizzled smem, dequant per fragment in registers, L2::128B
on every load).

Same-run mma/FI with the 3-stage ring (fi_compare): 1K 0.42-0.51x WIN,
4K 0.82-0.97x WIN/PARITY, 16K+ 1.21-1.61x (was 1.56-2.05). 4K is now a
win; the 16K+ gap halved.

### The remaining lever: the register path done FI-style

Our register-path attempt failed on the A side (+32 regs for the
even/odd fragments). FI's frag_layout_swizzle_16b_to_8b
(data/include/flashinfer/frag_layout_swizzle.cuh) resolves the k-extent
puzzle WITHOUT extra A fragments: 2x shfl_xor + 2x byte_perm per u32
redistribute the ldmatrix fragment's bytes ACROSS LANES so each lane
holds the 4 consecutive fp8 its mma k-elements need — the full-d A
fragment stays valid. Cost ~1 instr/fp8 + the x4 dequant 2.5/fp8, and it
frees sK16/sV16 (32 KB): the 48 KB budget enables 3 CTAs/SM (12 warps)
or 2 CTAs with deeper rings / bigger tiles. That + the L2::128B cp.async
hint are the path to the FI-class 128 KB/SM in flight.

### The config math (the recon + the measurements combined)

FI's fp8 TC path repacks fp8->bf16 INTO SMEM (repack_fp8_tile_to_bf16,
prefill.cuh) and uses the standard 16-bit ldmatrix — the shfl-based
frag_layout_swizzle_16b_to_8b is their FP4 in-loop path, NOT fp8. Our
structure is already FI-equivalent; the difference is the in-flight
bytes (theirs 128 KB/SM via 128-row tiles, ours 64).

The smem accounting at 80 KB/2 CTAs: the raw pairs + the bf16 tiles
(sK16/sV16) + sQ compete. Raising the in-flight past 64 KB/SM REQUIRES
dropping the bf16 tiles (the register path) AND shrinking sQ (the split
staging, 16 KB, 2 chunks of 32 rows):

  regpath (A-even/odd fragments only, no tSrQ — the same register count)
  + sQ split-staged 16 KB (fill chunk -> sync -> the warps owning the
    chunk ldsm their fragments via get_slice(tidx % 64) -> sync)
  + 4 raw pairs (a, d real; b, c aliased into sQ's 16 KB)
  = 80 KB, 2 CTAs/SM, 3 groups in flight = 48 KB/CTA = 96 KB/SM
  -> ~137 GB/s -> 16K ~= 245 us vs FI 228 (M=1); M>=4 WINS.

The L2::128B cp.async hint is not exposed by the DSL's CopyG2SOp
(cache_mode only) — parked unless an inline-asm path shows up; FI's
advantage is the in-flight bytes anyway. cute.arch.shuffle_sync_op
exists if the shfl path is ever needed (the FP4-style in-loop dequant).

The earlier register-path +19% was measured WITHOUT the ring: the
serialization dominated. With the ring + the split-sQ the instruction
savings and the freed smem compound — the config above is the bet.

### The L2-hint question, closed with data (2026-10-24)

Is there an equivalent of FI's cp.async `L2::128B`? Three classes, all
measured at M=1 (the probe reproduces the committed baseline exactly):
probe `/tmp/opencode/l2probe.py` (L2MODE=none|gran|window around the stock
harness; sed for the in-kernel cache_mode, reverted).

| knob | L=16K M=1 | verdict |
|---|---|---|
| baseline (GLOBAL/.cg, default fetch granularity 64) | 365.6 us / 92 GB/s | — |
| cudaLimitMaxL2FetchGranularity=128 (the device-wide L2::128B) | 363.4 | noise |
| persisting L2 window over the KV (2.9 MB carve, hitRatio=carve/window, streaming misses) | 362.5 (16K), 121.6 (4K vs 122.8) | noise |
| in-kernel LoadCacheMode.STREAMING (evict-first) | 371.7 | +1.7% worse |
| in-kernel LoadCacheMode.ALWAYS (.ca) | 371.6 | +1.7% worse |

Our rows are 256B contiguous — the 8 back-to-back 16B cp.asyncs cover each
128B granule — dbg38 (pattern vs contiguous = equal) plus gran 64->128
doing nothing close the fetch-granularity dimension: it was never our
loss. FI's 162 GB/s is the in-flight bytes (128-row tiles), not the hint.
The window result adds one more: even 17% of reads hitting L2 does not
raise throughput — the kernel is cadence-bound (the ring's wait depth
caps outstanding bytes), not latency-bound.

If the exact instruction is ever needed: the DSL exposes llvm.inline_asm
(cute.arch.cvt_i4_to_bf16_with_shuffle_impl is implemented with it —
llvm.inline_asm(Int32.mlir_type, [args], asm, constraints)); a
cp.async.cg.shared.global.L2::128B is wireable the same way (smem address
via cvta.to.shared inside the asm block). Parked — measured inert.

Point 2 CLOSED. The lever stays the 96 KB in-flight config (register path
K+V, sQ split-staging, 4-pair ring — increments 1-3).

### The ncu verdict: the kernel is ARITHMETIC-bound, not memory-bound (2026-10-24)

Profiled the 128 KB/SM build vs FI at 16K M=1 (the stall cycles per
issued instruction):

  ours:   wait .36 | math_pipe_throttle .12 | short_sb .03 | LONG_SB .027
  FI:     LONG_SB .31 | wait .155 | short_sb .08 | barrier .075 | math .047

Both run 255 regs -> 8 warps/SM. FI is DRAM-latency-bound AND STILL 1.6x
FASTER; we stall on fixed-latency ALU chains + the math pipe. The
in-flight bytes were never the lever: 2/3/4 groups in flight measured
363.8/377.6/388.5 us (the monotonic WORSE — the register paths' 2-pass
dequant costs, not the ring). The L2 window + the fetch-granularity
results now have their mechanism: nothing memory-side was ever the cap.

THE STRUCTURAL COST: the M tile is 64 rows; q_len*6 <= 48 are real. The
P/softmax/acc_O work per KV byte scales with MT — ours processes 64
P-values per KV row where FI (CTA_TILE_Q=16) processes 16 — 4x the
compute per byte. The padded-band skip (warp-uniform, compile-time
per seqlen_q) measured NEUTRAL — the softmax was not the throttle
either; the acc/softmax work is structural to MT=64.

MEASUREMENT HYGIENE: a timed-out gate run left a DEADLOCKED container
spinning on the GPU for 33 min; the FI leg read 2.2x slow until
`docker rm -f` — after any timeout, `docker ps` for leftovers before
trusting a number.

The register paths (the even/odd 2-pass dequant, /tmp/opencode/
regpath.patch + the ring tree edits) are archived; the tree reverts to
b198fbd. The next real move is the M-tile redesign (below).

### The M-tile redesign sketch (the next build)

MT=16 for the small M (compiled per seqlen_q — the runner already keys
on q_len), 4 warps re-rolled:

  QK^T: the warps K-SPLIT the d (4 x 64) — the acc_S partial per warp
    -> smem (16x32 f32 = 2 KB, the dead-sQ region) -> barrier -> the P
    computed once -> smem (1 KB bf16, the reference's P round-trip)
  PV: the warps N-SPLIT the d — the warp w owns the O columns
    [64w, 64w+64); the A = the P fragment read from the smem P by all
    warps (the m-atom is shared); the acc_O per warp (16, 64) = 32
    regs/lane (vs 128 today)

The payoff: the P/softmax/acc work per KV byte /4 (16 P-values per KV
row, FI's exact ratio); the registers ~150 -> 3 CTAs/SM = 12 warps
(the latency hiding x1.5); the Q tile 8 KB -> the smem freed for the
ring depth if the memory latency ever matters again (it does for FI:
long_scoreboard .31). MT=32 for M<=5, MT=64 (the current kernel) for
M=6-8 — the three paths coexist behind the constexpr seqlen_q.

The risk: the cross-warp S merge + the P smem round-trip = a new sync
structure (3 barriers/stage); the M=6-8 band keeps today's kernel.

### The instruction-mix profile: the ALU pipe is the wall (2026-10-24)

16K M=1, both kernels at 2 CTAs/8 warps, mma 349 us vs FI 227 us:

  pipe (% of peak)   MMA     FI      per-launch note
  ALU                47.3%   30.7%   dequant + addresses + tile logic
  fma (fp32)         17.6%   11.0%   softmax/rescale
  fma fp16 (packed)  14.8%    5.9%   dequant HFMA2 + P converts
  LSU                 9.1%   22.9%   FI pushes loads 2.5x harder
  tensor HMMA         7.4%    3.0%   ours 2.5x per byte (the 64-row tile)
  sass spills      125,504  40,768  3x FI's local traffic
  mem throughput     30.4%   57.0%   our DRAM half-idle behind the ALU

The dequant recipe is the SAME per byte as FI's (their repack) — the
DIFFERENTIAL is the 64-row tile: 4x the softmax/P/HMMA/acc work per KV
byte + the register pressure it causes. Our ALU clogs the issue slots
so the LSU (9%) never feeds the DRAM (30%) that FI's kernel drives to
57% from the same 8 warps.

THE RANKED PLAN:
1. MT=16 redesign (the k-split QK / the smem S merge / the P round-trip
   / the N-split PV) — attacks the fp16 14.8->~6, the HMMA 7.4->~3,
   the ALU tile share 47->~31, the spills 3x->1x, AND the occupancy
   (150 regs -> 3 CTAs = 12 warps). The only option touching all five.
2. The deferred-frame softmax — the O in a fixed max-frame, the P =
   exp2(S - m_ref), no per-stage acc_O rescale (128 fp32 fmul/stage/
   lane -> 0) + the warp-uniform re-anchor when the max drifts. A
   chunk of the fma 17.6%. Stackable with 1.
3. The dequant diet / the occupancy-only fixes — marginal; the dequant
   is already FI-parity per byte. Parked.

### Phase A result: the deferred-frame softmax (2026-10-24, b35c4f0)

The O accumulates in a fixed max-frame (row_max = the anchor, -inf =
unset); the P = exp2(S - anchor) may exceed 1 by the drift; the
warp-local re-anchor fires only when the logits climb >60 log2 units
(the exp2(60) P ~1e18 — orders inside the bf16/fp32 range with the V
descales). The per-stage acc_O rescale (128 fp32 fmul/stage/lane) is
GONE; the LSE = anchor + log2(sum) is frame-invariant; the normalize
cancels the frame. Gated 42/42 including the re-anchor path forced at
the threshold 2.

The sweep: the 16K+ ratios 1.56-1.59 (M=1) / 1.18-1.25 (M>=4) — a
real ~2-3% cut, the best-ever 262K rows. Modest as predicted: the
rescale was ~1/3 of the fma pipe; the ALU 47% stands — the MT=16
build attacks it.

PROCESS NOTE: the inc-3 kernel was accidentally committed inside the
notes commit c2a63ab and the "revert" that followed was a no-op
("Updated 0 paths") — the current-state table presented the inc-3
numbers. Fixed in cdf5271 (the true b198fbd baseline restored; the
inc-3 state stays in the history). Lesson: after any git checkout,
VERIFY the content (grep for the marker), and commit kernel states
explicitly — the sed+checkout cycles can silently wipe uncommitted
work (it did; re-applied).

### Phase B refined design: MT=16 via the (1,4,1) N-split arrangement

SCOPE: the q_len <= 2 path only (M_TILE=16 when seqlen_q*G <= 16, else
the current MT=64 kernel — the constexpr seqlen_q picks per compile).
The M>=3 band keeps today's kernel until this one proves out.

The (1,4,1) arrangement splits N for BOTH gemms with ONE tiled_mma:
  QK^T (C=16x32): the warp w owns the S columns [8w, 8w+8) over the
    FULL k=256 — 16 HMMAs/warp (the total 64, 4x less than the MT=64
    tile's 256 at M=1 — the padding waste is the whole win).
  PV (C=16x256): the warp w owns the d [64w, 64w+64) — 8 n-atoms x
    2 k-steps = 16 HMMAs/warp; the acc_O per warp (16, 64) = 32 f32
    regs (vs 128) -> ~150 regs/thread -> 3 CTAs = 12 warps.
The A operands are SHARED across the N-warps (the Q hoist unchanged;
the P read by every warp from smem).

The cross-warp row stats (the S columns are disjoint): the per-warp
partial max/sum per row (after the existing quad butterfly) -> the
smem slabs (4 warps x 16 rows x 2 = 512 B) -> barrier -> every lane
merges its rows' 4 partials. The P is then computed per-warp on its
own columns and stored to a (16,32) bf16 swizzled tile; barrier; the
PV's A = the ldmatrix from that tile (the smem_copy_A idiom).

The smem budget: the Q tile 8 KB (was 32) + 3 ring pairs (one buffer
aliased into the 8 KB sQ) 48 KB + sK16/sV16 32 KB = 80 KB -> 2 CTAs
(3 CTAs needs the register paths back — the inc-1/2 machinery, the
archived patches — as the follow-up). The P tile ALIASES sK16's first
1 KB: the QK's ldmatrix reads complete before the stats-merge barrier,
the P writes follow it, and the next stage's dequant waits behind the
post-PV barrier — the same hazard chain as the ring reuse.

The barriers/stage: 4 (the ring wait, the dequant-end, the stats
merge, the P-complete) — ~120 ns against a ~2-3 us stage.
