# arXiv research brief — latency attacks for the Mjolnir Thor decode kernel

Date: 2026-10-21. Scope: scheduling / pipelining / softmax-chain / split-KV ideas from
2023–2026 decode-attention kernel literature, filtered against the measured state
(latency-bound, 64-B L2 request wall, ring-depth lever exhausted, merge-fusion retired).

Papers read in full for this brief: 2609.33410, 2311.01282, 2407.08608, 2508.18850,
2604.23553. Screened and rejected: 2604.00028 (useful data point only — see §5),
2511.22333 PAT (prefix-sharing across requests — single-request MTP workload),
2410.18038 POD-Attention (prefill/decode hybrid batching — N/A), 2609.00097 FFD +
other sparse-attention methods (changes results; correctness gate is a bit-faithful
emulator), 2505.21487 GLA/GTA (architecture change, out of scope), green-context /
SM-partitioning serving papers (serving-level, kernel-level metric unaffected).

---

## Idea 1 — Declared-reference softmax: delete the max-reduction + rescale from the S→P chain

**Sources:** FoldAttention, arXiv:2609.33410 (§3 "additive contract", §4 decode kernel);
FlashDecoding++, arXiv:2311.01282 (§3 asynchronized softmax with unified maximum);
FlashAttention-3, arXiv:2407.08608 (§3.1 skip-max reasoning for bounded scores).

**Mechanism.** Softmax is invariant to a common shift, so the per-row reference Z_i need
not be the discovered max — it only has to keep weights in range. Fix Z_i per row before
scanning: P = exp2(s − Z_i) is *final* when computed. FoldAttention proves the range
condition (Prop 1(iii)): with n ≤ 2^22 keys and |v| < 2^16, any Z_i within ~80 binades
of the row max keeps every weight/numerator/denominator in range in BF16/FP32, with
flush loss < 2^-24 relative. Consequences for our softmax warpgroup chain
(S load from TMEM → max-reduce → exp2 → P store to SMEM → BMM2): the cross-lane max
reduction leaves the critical path, the P store no longer waits on a reduction of its
own tile, and the TMEM O accumulator is *never* rescaled (not merely "rescale skipped"
— the update decision itself disappears). Split partials become (A, L) pairs that merge
by plain addition — no LSE shift math in the merge kernel.

**Z_i sourcing, cheapest-first (all testable behind a harness flag):**
1. Static per-(layer, kv-head) Z from the empirical score distribution of the calibrated
   fp8-Q × fp6-K path (measure once offline with the emulator; FlashDecoding++ measured
   >99.99% of scores inside a ~24-binade window on Llama2-7B). Only valid if the
   empirical range is tight — a worst-case scale-derived bound is far too loose and
   flushes everything; do NOT use the quant-scale product bound.
2. Per-row front estimate à la FoldAttention: attention-sink token + 63 most recent
   keys scored exactly + stratified sample of the rest (192 scores/row; we have ≤ 7
   rows per KV head, so this is ~1.3K coarse dot-products per step — producer-warp
   fodder). No state between steps.
3. Previous-step LSE cache per (head, row) with FoldAttention's certify-and-rerun
   fallback: after each step check L ∈ [2^-1, 2^100]; rerun violating rows with
   Z_i = ℓ_i. Their measured rate: 118 reruns per 1.8B rows.

**Why it fits THIS kernel.** The softmax chain is explicitly flagged as remaining time;
long_scoreboard is the #1 stall and the S→max→P edge is a serial dependency repeated
per 128-row tile. GQA 6:1 with q_len ≤ 1+spec means ≤ 7 rows per KV head — the Z vector
is tiny and the exp2 work is trivially overlappable once unchained. Spec/MTP rows are
just extra query columns (FoldAttention §4 handles exactly this shape, including
ancestor masking). The merge already exists as a separate one-pass kernel; (A,L)
additive partials shrink it and are the prerequisite for Idea 4. This is *not* the
already-shipped "TMEM accumulator rescale skips": those still pay the max reduction and
the branch every tile; this removes both.

**Expected per-L shape.** Scales with tiles-per-CTA (≈ L/128 / splits): near-nothing at
128–4K, visible at 8K (compounds with the known bytes-in-flight sensitivity there),
largest at 128K–256K. Flat small win at every L from the simpler merge. If the softmax
chain is ~15–20% of kernel time (FlashDecoding++ measured ~20% for softmax sync/update
in 2023-era decode), expect mid-single-digit % geomean, more at long L.

**Correctness caveat (decision needed).** Real-arithmetic identical to emulator softmax;
FP rounding differs (different reference) and summation grouping differs. The emulator
stays untouched; the kernel side needs the verify tolerance to accept a different
rounding of the same math — same class of deviation as split-order reordering, but
larger. Gate behind a harness flag; if verify is bit-exact-only, this idea dies and
only the merge-side additive form (identical grouping semantics) survives.

---

## Idea 2 — Weight-gated V fetches: attack the 16K–32K request wall from the request-count side

**Source:** FoldAttention, arXiv:2609.33410 (§4 "Weight-gated reads"; §3 depth-T dial;
§6.2 ablation: T=16 within 1.5% of lowest-BF16 error on 7 real generations; 1.14–1.30×
dense, up to 3.09× gated on H100).

**Mechanism.** With fixed Z_i (Idea 1), weights are final at exp2 time, so they can gate
memory traffic *before bytes are requested*: a key is "live" iff s_ij ≥ Z_i − T for some
row in the GQA group (verdict ORed over the group — for us ≤ 7 rows). A 128-row tile
with no live keys skips its V reads; cut keys that are read keep their weight in L
(denominator stays exact over all keys) but not in A, bounding the output error by the
cut mass. T is a per-call parameter; T = ∞ is dense decode.

**Why it fits THIS kernel.** The 16K–32K hole is request-rate-walled with 64-B requests
locked — the only remaining lever on request count is issuing fewer requests. V is
~half of KV bytes → ~half of all TMA boxes; dead-tile skipping removes them wholesale
at tile granularity (the ring's streaming unit — no sub-tile change, which is the
refuted direction). This is the *opposite* of the failed L2 prefetch (which added
requests): it removes them. Implementation on our ring: split K and V boxes into
separate ring stages so the producer can drop a V stage after the consumer's S-verdict
for that tile (the producer already does runtime pacing, so a "skip stage" branch is in
vocabulary). Weaker fallback: fetch V but skip BMM2 + the TMEM S slot (saves tensor-core
+ softmax work, not requests) — still wins at long L where BMM2 queueing shows up.

**Expected per-L shape.** ~Parity at 128–8K (few tiles, mostly live). From 16K up,
attention concentration grows and dead-tile fraction rises; this is the one idea that
can *move the 16K–32K wall* (fewer in-flight requests → shorter L2/DRAM queues → lower
long_scoreboard) and it compounds at 64K–256K. FoldAttention's own sweep: benefits grow
monotonically with context.

**Correctness caveat (harder than Idea 1).** This is a genuine approximation, not a
reordering. The emulator computes the exact quantized result; a T-gated kernel cannot
match it bit-for-bit. Requires an explicit correctness-policy decision (tolerance band
vs emulator, plus the existing `bench accuracy` quality gate). FoldAttention's evidence:
at T=16, error within 1.5% of the lowest BF16-kernel error across 7 model generations;
at T=14 fewer bytes within the same band. Recommend: dense fixed-Z first (Idea 1),
then T=∞-vs-T=16 A/B through `bench accuracy` before ever letting it near the judge.

---

## Idea 3 — CTA-pair ping-pong: phase-offset the 2 CTAs sharing each SM

**Source:** FlashAttention-3, arXiv:2407.08608 (§3.1 pingpong scheduling; §4.2 ablation:
warp-spec + 2-stage GEMM/softmax overlap 570→661 TFLOPs; §B.3: 3-stage pipelining is
*worse* than 2-stage — don't blindly deepen S/P).

**Mechanism.** FA3 overlaps one warpgroup's softmax with another's GEMM via bar.sync
phase offsets (570→620–640 TF on H100 hd128): non-matmul units (MFU/exp2) and tensor
cores should never be busy for the same warp-group at the same time. Our kernel has ONE
softmax warpgroup per CTA — but 2 CTAs per SM sharing one tcgen05 issue unit, one TMEM
port pair, and one SMEM fabric. In lockstep, both CTAs burst tcgen05.ld(S)/st(P) at the
same time and queue their BMM2s behind each other; half-a-tile phase opposition makes
CTA0's softmax chain run during CTA1's tensor-core op and vice versa. In decode shapes
this pays *more* than prefill: q-tiles ≤ 7 rows mean tensor-core work per tile is tiny
(M=64+ padding), so the softmax fraction of the critical path is larger than FA3's
prefill case. Zero-risk variant: open-loop parity flip (per-CTA lead/lag slot rotation
by `cta_rank_in_sm % 2`) — no cross-CTA sync, no deadlock; stronger variant: pair the
2 same-SM CTAs into a 2-CTA cluster and hand off via named barriers.

**Why it fits THIS kernel.** 40 CTAs = 2/SM is occupancy-pinned (likely TMEM: S slots +
hd256 accumulators), so CTA count is fixed — the only free scheduling axis on an SM is
*phase*. FA3's §B.3 negative result (3-stage worse than 2-stage) says deepen-with-stages
is the wrong move on a constrained pipeline; phase-offset gets the same overlap for
free. The measured long_scoreboard dominance is exactly the signature this attacks
(TMEM/SMEM transfer waits), and ncu can verify it directly (stall cycles/instr per L).

**Expected per-L shape.** Proportional to softmax-chain fraction: ~nothing at 128
(1–2 tiles, prologue-dominated), growing through 8K→256K as tiles-per-CTA grows.
Expect a same-run-ratio-measurable few % with a clean ncu signature (long_scoreboard
down, tensor-core idle down). Cheap enough to try in one brick.

---

## Idea 4 — Cluster-resident split-KV merge: DSMEM tree reduce inside the kernel

**Sources:** ClusterFusion, arXiv:2508.18850 (§3.1 `ClusterReduce` over DSMEM, binary
tree; §3.2 "one cluster per head, data-dependent dims inside the cluster");
ClusterFusion++, arXiv:2604.23553 (§2.2 tree reduction O(log n) vs ring; CUDA-Graph-
compatible persistent TMA descriptors; sm_120 consumer GPU).

**Mechanism.** Map the CTAs that share a (kv-head, q-tile) — the split partials of the
same output rows — into ONE thread-block cluster. After each CTA's last tile, partials
(A,L) [or (partial, m, l)] move over DSMEM (`cluster.map_shared_rank`) and reduce by
binary tree (ClusterFusion Alg. 1); rank 0 writes output. No separate merge kernel
node, no partials round-trip through L2/DRAM, no global rendezvous — the reduce starts
the moment each CTA finishes.

**Why it fits THIS kernel — and why it is NOT the retired fused merge.** The 0.56×
failure fused all merging into the *last* CTA serially: one CTA paid the whole head's
reduce after its own last tile, stalling the persistent wave. The cluster version is
collective — all CTAs of the head participate simultaneously in O(log N) DSMEM hops,
inside the same launch. Our plan pins 40 CTAs; at full wave that is 10 splits/head —
over the portable cluster cap of 8. Co-design: cap splits/head at 8 (32 CTAs, 8 spare
for spec q-tiles / uneven heads) or opt-in 16-CTA clusters (2 heads fully covered).
The planner owns the (head → cluster) mapping; correctness is unchanged (same math,
different transport — merge summation order changes, same caveat class as Idea 1).
ClusterFusion++ ships the atomic-add alternative and reports only FP16-atomic
non-determinism — atomics are off the table for us if verify is bit-exact, but the
DSMEM tree is order-controllable if the planner fixes the reduction pairing.

**Expected per-L shape.** The merge is a serial tail after the *last* chunk: at 128–4K
there are ≤ 1–2 splits/head (merge is trivial or skippable — little to win); the tail
bites at 8K–32K where the kernel is 10–50µs and merge ≈ 2–4µs + launch (~5–10% of the
cell) — exactly the mid-L band we want to un-stick; at 64K–256K it amortizes to ~1–3%.
Also removes one graph node per layer — a small but real per-step latency cut.

**Feasibility check first:** sm_110 cluster support + max cluster size on Thor, and
whether the CuTe-DSL kernel can express `cluster.map_shared_rank` in the epilogue
path (t5_decode epilogue + t5_plan split mapping). If clusters are a dead end on
sm_110, the fallback is FoldAttention's additive (A,L) merge from Idea 1 (same data,
cheaper math) with the transport unchanged.

---

## §5 Screened but demoted

- **Sequence-aware split heuristic for low-head-count decode** — arXiv:2604.00028:
  FA3's default under-splits low-head-count decode; a sequence-aware split policy gave
  +21–24% decoder-kernel efficiency with no regressions. Our `plan_split_kv` is already
  per-L and occupancy-pinned, so the headline doesn't transfer — but keep it as the
  reference when re-tuning splits/head for Idea 4's cluster cap (the optimal split
  count interacts with MTP q-tile count and with the merge transport).
- **PAT** (arXiv:2511.22333), **POD-Attention** (arXiv:2410.18038): batch/prefix-level
  scheduling — single-request MTP decode, N/A.
- **Sparse-attention decode** (FFD arXiv:2609.00097, JSSA arXiv:2610.19026, ETA
  arXiv:2609.20888, …): changes results; correctness gate is a bit-faithful emulator.
  Idea 2 is the only traffic-cutting form compatible with a tolerance-based gate.
- **GLA/GTA** (arXiv:2505.21487): architecture change, out of scope.
- **Green contexts / SM partitioning** (arXiv:2603.10342, 2511.04791, 2601.07600):
  serving-level isolation; the scored metric is kernel-level on gated clean windows.

## Cross-cutting ordering recommendation

1 (dense fixed-Z) → 3 (ping-pong) → 4 (cluster merge) are mutually compatible and
correctness-safe modulo rounding-tolerance policy; 2 (V-gating) builds on 1 and needs
the accuracy gate. 1+3 compound: fixed-Z shortens the softmax chain, ping-pong hides
what remains of it behind the sibling CTA's tensor-core work.
