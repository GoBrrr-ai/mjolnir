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
