# The MMA kernel design log, part 2 (the M=1-2 register-path package)

Continues `mma-kernel-design.md`. The 3-CTA package: the K register path,
the sV16 alias over the dead sQ + a pad, the 2-pair ring, minctasm=3.

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

### Phase B result: MT=16 (1,4,1) gates 42/42 and cuts the 16K+ gap by a third (2026-10-24)

First pass failed 30/30 M=1 cases with rel_err ~3.0 — the row-SUM
partials were stored per-lane without the quad butterfly (the lane
holds only its 2 of the warp's 8 columns -> the row_sum = 1/4 -> the
O ~4x). One butterfly fixed it; 42/42.

The sweep (MJOLNIR_MMA16=1, mma vs FI same-run):

  M  L       mma      FI     ratio   (the MT=64 deferred-frame)
  1  4096    105.0    125.0  0.84x   (0.96x)  - now a WIN
  1  16384   296.5    227.5  1.30x   (1.59x)  -18%
  1  65536   1051.5   818.6  1.28x   (1.58x)  -20%
  1  262144  4075.1   3253.9 1.25x   (1.56x)  -21%
  2  16384   305.2    238.8  1.28x
  2  262144  4186.1   3218.0 1.30x

The BW: 92 -> 113 GB/s at 16K, -> 134 at 262K (the DRAM getting fed —
the profile's prediction confirmed: fewer instructions per byte, the
issue slots free, the LSU runs). The remaining 1.25-1.30: the next
levers are the 3-CTA occupancy (the register paths + the freed Q smem)
and the MT=32 for M=3-5.

### The MT=32 occupancy trap + the fix (2026-10-24)

The first MT=32 sweep read 1.71-1.76x — WORSE than the MT=64. The
smem: the sQ doubles to 16 KB but only the pair-b K was aliased (the
V real) -> 89 KB dynamic -> 1 CTA/SM = 4 warps. The fix: the 16 KB
sQ hosts BOTH pair-b buffers (the sV8b alias at +SR*D//2 bf16
elements; exact fit) -> 81 KB -> 2 CTAs. Gate: the ring 42/42 + the
M=3/4/5 at the ps 16/32 all OK first shot.

The M=3-5 band (the MT=32, the deferred-frame softmax):

  4K    0.76-0.78x  WIN
  16K   1.13-1.14x
  64K   1.08-1.09x
  262K  1.05-1.06x  near-parity

The band table now: the M=1-2 (MT=16) 1.25-1.30; the M=3-5 (MT=32)
1.05-1.14; the M=6-8 (MT=64) 1.18-1.41. The M=6-8 is the worst band —
the MT=48 (3 m-atoms, the (3,1,1) arrangement, the 4th warp idle) is
the open question for the MTP=5's q_len=6. The M=1-2's next lever is
the 3-CTA occupancy (the register ceiling 255 -> the cap experiment).

### Full-sweep consolidation + the 3-CTA register-cap probe (2026-10-24)

The 18-shape sweep (M in {1,4,8} x L in {1K..262K}, MJOLNIR_MMA16=1 ->
the dispatch: M=1 MT16 / M=4 MT32 / M=8 MT64): all three bands
reproduce the committed numbers -- the kernel16->kernel_small
parameterization cost nothing. New: L=1024 wins big (0.41-0.51x --
the FI launch floor 126-159 us vs our 64-75 us).

  M=1 (MT16):  0.51 0.84 1.31 1.28 1.30 1.27
  M=4 (MT32):  0.41 0.78 1.13 1.08 1.08 1.06
  M=8 (MT64):  0.48 0.89 1.26 1.21 1.20 1.18

The register-cap probe (MJOLNIR_REGCAP=1 -> nvvm.minctasm=3, the
__launch_bounds__(128,3) equivalent; the DSL LaunchConfig exposes
min_blocks_per_mp, _generate_kernel_attrs emits nvvm.minctasm):

  regs 255 -> 168 (the cap honored, under the 170 target)
  local mem 1024 ld + 1024 st sectors/launch ~ 32 KB -- negligible
  16K M=1: 298.0 -> 306.0 us (+2.7% -- the ILP loss, no spill wall)

VERDICT: GO. The live set fits <=170. The package: the register
paths (drop sK16/sV16 -> ~50 KB -> 3 CTAs at the 54.6 KB budget) +
the P-tile relocation + min_blocks_per_mp=3. 3 CTAs x 128 threads x
168 regs = 64512 <= 65536 -- fits on both axes. The expected win:
+50% warps -> the BW 113-134 toward the FI 147-165 GB/s.

MT=48 parked (the math): the (3,1,1) idles the 4th warp or needs the
96-thread plumbing; the sQ 24 KB -> 89 KB smem = 1 CTA without more
aliasing; the win bounded at -25% row-work (one garbage m-atom per
M in 6-8) -- contention-relief scale, not the MT=32 structural win.
Cheaper equivalent if ever needed: a compile-time warp skip in the
MT=64 (const_expr on the m-atom index vs seqlen_q*g; needs the
stats-slab/merge structure read first).

### The 3-CTA package build: the K register path + the sV16 alias + the 2-pair ring (2026-10-24, WIP)

The package (MJOLNIR_REGPATH=1, MT=16 only): the K dequant in the
fragment registers (no sK16; the (SR, D/2) bf16 view of the fp8 bytes;
the even/odd 2-pass gemm; the single acc_S), the sV16 aliased over the
dead-after-hoist sQ + an 8 KB pad, the 2-pair ring (a/c; the wait(1);
the refill issued after the post-PV barrier since it retargets the
consumed buffer), nvvm.minctasm=3. 48.5 KB dynamic -> 3 CTAs at the
168-register cap. Compiles and runs.

GATE: FAILS -- rel_err 0.45 (the M=1). The default path unaffected:
the ring 42/42 + the M=3/4/5 OK with the flags off (re-checked on the
same tree).

ROOT CAUSE (the dbg44-47 probes, /tmp/opencode/dbg4{4,5,6,7}.py):
the A and the B fragment k-column orders disagree. The B fragment
(the LDSM from the swizzled (SR, D/2) view) carries the view columns
in the SW3-permuted k-atom order {0,4,1,5,2,6,3,7} (the dbg45 phase
C: the u32 i holds the view cols {j0(i), j0(i)+1}, the lane-0
j0 = 8*(i%2) + 64*((i//2)%2) + 16*((i//4)%2) + 32*((i//8)%2); the
base per the 4-element k-atom group ai = (ai&1)*64 + ((ai>>1)&1)*16 +
((ai>>2)&1)*32). The A fragment (the dbg47: the real (MT, D/2) tile,
the setitem fill, the LDSM read) is LINEAR (the ks -> the view cols
16ks + 2q + (e%2) + 8*(e//4); the m = lane/4 + 8*((e//2)%2) per the
phase-A one-hot). My A fill assumed the linear order -> the gemm pairs
the wrong columns -> the uniform scramble (the rel_err 0.45, uniform
across the rows, the d parities, and the d quarters -- the dbg43).

THE FIX (the next increment): the A fill's ks->view-col map uses the
B's actual order. Note the MT=64 (4,1,1) register-path map (the
c2a63ab patch) was derived for a different arrangement and only ever
bench-checked (--skip_ref_check) -- it likely carries the same class
of bug; re-gate it before any revival.

## Session 4 (2026-10-25): the package fixed — three bugs, then the swizzle

The package (2-pair ring + sV16 alias) failed the M=1-2 gate with
rel_err 0.479-0.51. Four bugs, in the order the bisection found them:

1. **sP's swizzle base misalignment.** The package allocated sStat
   (512 B) between the 8K-aligned buffers and sP after it — sP's base
   landed at 48.5K, not 1024-aligned, so the P tile's SW3 pattern
   misapplied and EVERY stage was wrong (L=32 included). Fix: allocate
   sP before sStat.
2. **The K fragment ldmatrix was hardwired to buffer a.** `tSsK =
   partition_S(sK8v)` never rotated with the ring parity — stage >= 1
   read stale buffer a. The `srcu_k` par-selection only fed the
   non-regpath dequant arm. Fix: sK8v_c/tSsK_c twins + a runtime
   `tSsK_cur` selection (the DSL yields same-layout tensors from a
   dynamic if — the same mechanism srcu_k uses).
3. **The bisect's real sV8b was never read**: the u32 view aliases
   (`sV8bu = sV8cu = recast(sV8c)`) are correct for the 2-pair package
   but pointed the par==1 dequant at unwritten smem in the 3-pair
   bisect. Diagnostic only; the package ships.
4. **The plain-view sK8v (my session-start state) was the real
   performance bug**: a row-major (SR, D/2) bf16 view has a 256 B row
   stride — every ldmatrix 8-way bank conflict, 10.2M conflicts at
   L=65536. The ORIGINAL swizzled view (tile_to_shape bf16_atom) was
   right all along; its gate failures were bugs 1-2. With the swizzle
   restored: conflicts 3.2M, kernel 803 us (was 1030 default, 1090
   plain) at M=1 L=65536.

ncu ground truth (M=1, L=4096): default 255 regs / 2 CTAs / 0 spills /
81.7 us; regpath capped 168 regs / 3 CTAs / 113K local ops (spills) /
~120 us; regpath no-cap 255 regs / 2 CTAs / 0 spills. **The 3-CTA prize
is register-blocked**: 3 CTAs x 128 threads on sm_87 cap at 170
regs/thread and the regpath's working set (acc_O 32 + acc_S 16 +
tSrQ_even/odd 64 + tSrK 16 + tOrVt 32 + tSrP 8 + state) does not fit.
minctasm=3 on the regpath is REMOVED (it only spills); MJOLNIR_REGCAP=1
keeps the default-kernel probe.

Stall profile of the swizzled package (L=65536): wait 1.62 +
short_scoreboard 1.49 per issue-active — smem/ALU dependency chains at
8 warps/SM. Memory throughput fell 77 -> 66% while the kernel got 22%
faster: the kernel is issue/latency-bound, not bandwidth-bound.

**The M=1-2 band after the fix (mma/FI, same-run, 200 iters):**

| M | L=4K | 16K | 64K | 262K |
|---|---|---|---|---|
| 1 | 0.72 | 1.03 | 0.98 | 0.97 |
| 2 | 0.63 | 1.04 | 1.02 | 1.02 |

(was 0.86/1.31/1.28/1.24 for M=1.) Correctness: msweep 18/18 +
ring_check 42/42 on the package; flags-off regression 18/18.

The 16K cell sits at 1.03-1.04 across 3 repeats (mma side stable to
+-0.3%, FI +-1.5%) — a real ~3% loss; 64K is parity. Split count is
not the lever (ns=8/16/32 flat at 16K). The remaining levers: the V
smem round-trip (sV16 write + ldmatrix read = half the stage traffic)
would need the transposed-B fragment archaeology AND a register budget
that does not exist; 3 CTAs need <=170 regs.

## Session 5 (2026-10-25): the u32 packed dequant stores

The M=6 profile said issue-bound (wait 1.63 + short_scoreboard 0.98,
tensor pipe 30%, LSU 27%, conflicts 3.19M at 16K). The dequant loop was
storing every converted bf16 as a scalar ST.U16 -- 8 stores/iteration,
128/thread/stage, ~4-way bank conflicted (the swizzle de-conflicts the
ldmatrix, not the scalar stores).

Fix: u32 views of the swizzled bf16 tiles. Byte-identical re-index:
Swizzle(3,3,3) on an (8,64) bf16 atom == Swizzle(3,2,3) on an (8,32)
u32 atom (the granule-xor-row pattern re-expressed in 4B units; the
swizzle preserves the low 4 bits, so adjacent bf16 pairs stay
adjacent). Each fma_packed_bf16x2 output IS a bf16x2 -> one u32 store
per pair: 4 stores/iteration instead of 8, bit-exact.

Both kernels: the big kernel (sK16u/sV16u) and kernel_small (V only in
the regpath; K+V in the mt=32 path). Gates: msweep 24/24 both flag
states, ring_check 42/42, ruff clean.

Band (same-run mma/FI, package flags; was -> is):

| M | 4K | 16K | 64K | 262K |
|---|----|-----|-----|------|
| 1 | 0.72->0.705 | 1.03->0.987 | 0.98->0.925 | 0.97->0.887 |
| 2 | 0.63->0.581 | 1.04->0.959 | 1.02->0.949 | |
| 3 | | ->0.979 | ->0.915 | |
| 4 | | ->0.980 | ->0.909 | |
| 5 | | ->0.980 | ->0.912 | |
| 6 | 0.84->0.775 | 1.249->1.098 | 1.204->1.031 | 1.184->1.004 |
| 8 | 0.88->0.816 | 1.257->1.102 | 1.212->1.038 | |

The MIO pressure was throttling the whole issue stream: 343->304us
(ncu) at M=6 16K, conflicts 3.19M->0.82M, tensor pipe 30->35%. Every
band now beats FI at 16K+ except M=6/8 at 16K (~1.10) and 64K
(~1.03); M=6 262K is parity. Stalls now wait 1.80 + short_sb 1.08 --
dependency-bound at 255 regs (acc_O alone 128 = MT*D/NT).

MT=48 re-tested with the edit CONFIRMED in the file (the first attempt
silently did not stick; the "passing" gates ran MT=64): 24/24 FAIL
(rel_err 0.3-11.8, all M/L). Two structural breaks: the sV8c alias
overflows the 24KB sQ into sK8, and the (3,1,1) idle warp 3 produces
garbage tScS coords -> the epilogue's `q_rows > R` guard passes for
garbage R < q_rows -> spurious live-row writes. A retry needs sV8c as
a real 8KB alloc (24+8+8+8+16+16 = 80K, still 2 CTAs -- the offsets
stay 1024-aligned) and `tidx < (MT//16)*32` guards around the mma
sections (warp-uniform, free at MT=64). The payoff: acc_O 96 (+32
regs ILP), -25% tensor work, -25% ldmatrix redundancy (3 warps read
the B operands, not 4).

## Session 5b: the split capacity matches the MMA occupancy

The auto plan used BLOCKS_PER_SM=4 -- the GEMV CTA's residency (~100
regs, ~6 KiB smem). Every MMA shape is pinned at 2 CTAs/SM (255 regs +
50-82 KiB smem both bind), so the capacity over-split: 16 splits on 16
SMs = 64 CTAs = 2 waves. The runner now picks blocks_per_sm = 2 for
impl="mma" -> ns_cap 8 at batch 1 (32 CTAs = exactly 1 wave).

Band (same-run mma/FI, package flags, the new default):

| M | 4K | 16K | 64K |
|---|-----|-----|-----|
| 1 | 0.659 | 0.966 | 0.920 |
| 2 | 0.533 | 0.940 | 0.927 |
| 3 | 0.641 | 0.963 | 0.922 |
| 4 | 0.640 | 0.967 | 0.913 |
| 5 | 0.665 | 0.962 | 0.917 |
| 6 | 0.704 | 1.080 | 1.032 |
| 7 | 0.730 | 1.058 | 1.035 |
| 8 | 0.722 | 1.071 | 1.035 |

M=6 262K: 1.004 (session 5). The 4K band gained 5-8% everywhere (the
grid fills the machine at short L); M=6-8 16K gained ~2%.

Dead ends this session, measured: MT=48 (the tile-waste cut is
invisible -- the kernel is memory/dependency-bound; identical times at
M=6/8 16K/64K with the idle-warp guards correct and 24/24 green), the
per-element causal-mask hoist in softmax_p (a 3-9% REGRESSION -- the
duplicated bodies cost more than the saved compares; reverted), the
fragment-K 2-pass port to the big kernel (register-dead: acc_O alone
is 128 = MT*D/NT, the regpath working set ~250 at MT=16 scales to
~380 vs the 255 cap -- 3 CTAs need <=170).

The M=6-8 16K residual (~6-8%) is the design's latency floor: the
barrier-lockstep stage loop (2 syncs + the cp.async wait per stage),
the sK16/sV16 dequant round trip (32 of ~96 KB stage traffic), and the
255-reg cap. The next lever is warp-specialized producers/consumers --
a redesign, not a knob.
