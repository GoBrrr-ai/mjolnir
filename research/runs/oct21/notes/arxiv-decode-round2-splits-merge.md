# Round-2 arXiv brief — splits, merge tail, TMEM/SMEM, scheduling (and why most of it is dry)

Date: 2026-10-21. Follow-up to `task/research/arxiv-decode-latency-softmax.md` (round 1),
re-filtered against the round-2 constraint set (request-rate wall locked, ring exhausted,
softmax chain hidden, merge-fusion retired, bit-faithful emulator gate).

**Verdict up front: the well is dry for ≥2% keeps.** One sub-bar stacker survives
(split-count coarsening in the request-rate-wall regime), one conditional probe remains
(8K chunk-refinement — the arithmetic below likely kills it), and three categories the
round-2 brief asks about are closed by code evidence + arithmetic. The only realistic
round-2 *keep* is a batch of the round-1 ring brick + the coarsening brick, and that
lands at ≈ +1.1–1.6% — at or below the 2% bar.

Papers read in full this round: **2601.16032** (Sawtooth Wavefront Reordering, GB10),
**2603.05451** (FlashAttention-4). Screened against the code: `t5_plan.py`,
`t5_call.py`, `t5_decode.py`.

---

## Code facts that close candidate ideas (checked, not assumed)

| Fact | Evidence | Consequence |
|---|---|---|
| Grid is `(num_work, n_tiles, h_k)` — x fastest | `t5_call.py:482` | With 2 CTAs/SM, co-resident SM pairs are **already adjacent chunks of the same head** (HND → contiguous addresses). "Reorder work items for locality" has no slack: the pairing is optimal by construction. |
| O-rescale is already gated on max-change, bit-identical skip | `t5_decode.py:1321–1334` ("when nothing moved every correction is exactly exp2(0)=1") | The FA4 "conditional rescaling" trick (§3.1.4) is already shipped. O-rescale TMEM traffic is already minimal. |
| Reduction is compiled out when `n_splits==1`; one-pass, up-front predicated loads, j-ordered sum | `t5_call.py:536`, `t5_decode.py:1475–1541` | The merge is structurally near-floor and its **summation order is frozen** by the bit-faithful gate — any pairwise/DSMEM pre-merge reassociates the j-ordered sum and fails correctness. |
| Plan picks the smallest chunk that fills one wave; `max_splits=16`; longest-first | `t5_plan.py:140–197` | At 8K: chunk=7 tiles, 10 items × 4 heads = 40 CTAs, all resident. At 16K+: 16 chunks/seq (cap). The split count is the one free scheduling knob left. |

---

## Idea 1 (LIVE, sub-bar) — Coarsen the split count in the request-rate-wall regime (T ≥ 128 tiles)

**Source:** FlashInfer, arXiv:2501.01005 §3.3.1 — only requests whose KV length exceeds
`total_kv / #CTAs` are split, and its partial-output accounting bounds merge input by
split count; FlashDecoding++, arXiv:2311.01282 (split-count vs sequence length).
Our own `2604.00028` note (round-1 §5) is the fine-tuning reference; here we go the
*other* way, justified by the wall.

**Mechanism (2 sentences).** At 16K–32K the kernel is capped by the L2 front-end
request rate, so extra split CTAs cannot add throughput — but each extra chunk still
writes an fp32 partial (6 rows × 256 + LSE per (kv-head, chunk)) that the merge reads
back, and those writes/reads spend front-end budget inside the wall band. Halving
`max_splits` (16→8) for `T ≥ 128` tiles keeps steady-state concurrency at the
occupancy pin (each CTA's ring already holds ≫ enough in-flight 64-B requests to
saturate the wall alone: ~2 stages × 256 boxes × 2 operands ≈ 10³ per CTA vs ~10²
needed), and halves the partial round trip + merge input.

**Why it survives constraints 1–7.** No geometry change (same 96-B boxes, same 64-B
requests — constraint 2); no ring change (3); doesn't touch the softmax chain (4);
it *reduces* merge work rather than re-fusing it (5); not an approximation (7) — and
the emulator consumes the plan, so a different `n_splits` needs **no emulator math
change** (`plan_split_kv` output is an input to `emulate_decode`, `t5_harness.py:231–255`).

**Expected per-L shape.** 16K–32K: −1 to −2 µs of merge/traffic on ~150 µs cells
(≈ +0.5–1%); 64K–256K: ~+0.3% (partial traffic is a smaller fraction); 128–8K:
unchanged (gate at `T ≥ 128`). **Geomean ≈ +0.3–0.6% — below the park bar alone;
this is batch fodder.** It edits `t5_plan.py` (plan heuristic), so it stacks with the
round-1 ring brick (producer side, `t5_decode/t5_call`) but *not* with any other
plan-file edit in the same batch.

---

## Idea 2 (CONDITIONAL, arithmetic says dead) — Chunk refinement at 8K (`force_splits=16`)

**Source:** same split-sizing family (FlashDecoding++ §3; 2604.00028 sequence-aware
splits for low-head-count decode — the case for *finer* splits when heads are few).

**Mechanism + why the arithmetic kills it.** At 8K the plan already runs 40 resident
CTAs (10 chunks × 4 heads); refinement to 16 chunks × 4 heads = 64 blocks **cannot
raise resident concurrency above the 40-CTA occupancy pin** (TMEM pins
`ctas_per_sm=2`: O alone is 256 of 512 columns at d=256, per the known-negative list)
— the extra 24 blocks just queue, adding a second ring ramp + merge growth (+6
partials × 24 heads ≈ +150 KB round trip) with zero new steady-state streams. The
only way finer splits pay is if per-CTA MLP were the limit — but the 4-stage paced
ring (round-1 brick) already maxed that, and post-brick 8K sits at 164–172 GB/s
effective, *inside* the 64K–256K band (168–195).

**Per-L shape / decision rule.** Predicted ~0 or negative at 8K, worse at 4K. Run it
only if the post-brick ncu sweep shows the 8K request rate still materially below the
64K–256K band; otherwise this is closed. (This is the last untested 8K-specific lever —
see §Closed.)

---

## §Closed — the three categories the brief asked about, with arithmetic

### A. The 8K cell beyond the deep ring: dry
- KV bytes at 8K = 8192 × 4 heads × 2 (K+V) × 256 × 0.75 B = **12.6 MB** → 46 µs at
  273 GB/s, **61 µs at the fp6 byte-ratio ceiling (0.75 × 273 ≈ 205 GB/s)**.
- Post-brick 8K ≈ 84–89 µs × (1 − 0.13) ≈ **73–77 µs → 164–172 GB/s**, which is
  *inside* the 64K–256K effective band (168–195 GB/s). **8K is no longer special**;
  its residual is the same uniform ~10–17%-above-byte-ratio floor that persists at
  every L (AGENTS.md, fp6prof) — and every mechanism proposed for that floor
  (SW128, prefetch, readahead, block-table scheduling, sub-tile fill) is measured dead.
- Concurrency is pinned (TMEM ⇒ 2 CTAs/SM ⇒ 40 CTAs), per-CTA MLP is maxed (ring
  brick), request geometry is locked. **Nothing 8K-specific survives.**

### B. The merge tail (~7 µs at 8–32K): dry within scope
- Structure is already optimal: one-pass, all split loads issued up front (one
  collective latency, not a chain — `t5_decode.py:1502–1515`), grid `(2, h_q, b)`,
  skipped entirely when unsplit (`t5_call.py:536`).
- Floor estimate: graph-node dispatch (~1–2 µs) + one collective L2/DRAM latency
  (~1–2 µs) + ≤ ~400 KB of partials that are L2-resident when read (written moments
  earlier; ≪ L2) ≈ **4–5 µs vs 7 measured → ≤ 2–3 µs theoretical, ~0 practical**.
- Every transport change is dead: fused last-CTA (0.56×, run 1), PDL (measured no
  gain), cluster/DSMEM pre-merge — the last one *also* fails correctness because the
  reduction's j-ordered sum is the bit-faithful spec; reassociating it
  (`(a₀+a₁)+(a₂+a₃)` vs `((a₀+a₁)+a₂)+a₃`) changes rounding → gate fails. Multi-stream
  graph fork/join to overlap head-wise merges would work mechanically but is
  `backend.py` wiring — out of scope (7).
- The only legal merge lever is *fewer partials* = Idea 1.

### C. TMEM/SMEM traffic on the critical path: dry
- **S loads / P stores**: hidden behind BMM2 by the 2-stage S/P pipeline
  (constraint 4 — softmax warpgroup has slack at every L). Any payoff here is a
  "shorten the softmax chain" payoff — excluded.
- **O rescales**: already gated bit-identically (`t5_decode.py:1321–1327`). FA4's
  τ-slack variant (skip rescale until m moves > log2(256), arXiv:2603.05451 §3.1.4)
  changes the rounding path → not bit-faithful → dead here.
- **2-CTA MMA mode** (FA4 §3.2.3 halves SMEM traffic): requires the CTA pair to share
  MMA operands (B split across the pair). Our co-resident CTAs own **disjoint chunks**
  — no shared operands exists in b=1 decode. Dead.
- **TMEM capacity**: 2 CTAs/SM is the TMEM ceiling (O = 256 of 512 columns); 3 is a
  known-negative. No headroom for restructuring.

### D. Negative transfer worth recording: Sawtooth Wavefront Reordering (arXiv:2601.16032)
The closest published work on our exact chip family (GB10/DGX Spark: 48 SMs, 24 MiB
L2, LPDDR5X). Its mechanism — cut KV **reuse distance** across CTAs that process
different q-tiles over the same K/V — **has no target in b=1 decode**: every KV byte
is read exactly once (chunks partition the sequence; the 6 q-rows of a kv-head live in
one CTA), so KV misses are purely compulsory and no traversal order can reduce them.
What *is* worth keeping from it: (a) platform numbers for the sibling die — 24 MiB L2
means 16K KV (25.2 MB) straddles L2 capacity and 32K (50 MB) doubles it; the 16–32K
hole sits exactly at that boundary. One cheap ncu corroboration (`lts__t_sector_hit_rate`
vs L) would confirm the brief's request-rate attribution — but no legal lever falls out
(single-pass reads; request count locked). (b) Its locality lineage (CTA clustering,
HPCA'14/ASPLOS'17) is already satisfied by our grid construction (see code-facts table).

---

## Batch arithmetic & recommendation for round 2

- Ring brick (parked): +0.8–1% geomean (−13% at 8K, parity 16K+, loses at short L).
- Idea 1 (coarsening): +0.3–0.6% geomean, 16K+ weighted, **different code region
  (`t5_plan.py`) → stacks**.
- Batch total ≈ **+1.1–1.6%** — *below* the 2% keep bar. No surviving idea adds more.
- **Recommendation:** run the two cheap confirmations (post-brick ncu: 8K request rate
  vs the 64K+ band; `lts__t_sector_hit_rate` across the 16–32K boundary), then either
  (a) evaluate the brick+coarsening batch once as a lottery ticket on the noise band,
  or (b) declare saturation for round 2 and spend rows on re-measuring the two noisiest
  cells. Absent a constraint change (cache format, emulator rounding policy, backend
  graph structure), the geomean is within ~1–2% of its ceiling under the stated rules.

## Sources

| arXiv ID | Used for |
|---|---|
| 2501.01005 (FlashInfer) | §3.3.1 long-request-only splitting; partial-output bound; plan-time split sizing; merge-kernel precedent |
| 2603.05451 (FlashAttention-4) | §3.1.4 conditional rescaling (our gating is SOTA; τ-slack not bit-faithful); §3.1.2 TMEM S/P partitioning; §3.2.3 2-CTA mode operand-sharing requirement; §3.3 LPT (we already sort longest-first) |
| 2601.16032 (Sawtooth, GB10) | Platform facts (24 MiB L2, L1 pass-through, LPDDR5X); §3.4 wavefront-reuse mechanism — negative transfer for b=1 decode |
| 2311.01282 (FlashDecoding++) | Split-count vs sequence length (round-1 brief, §3) |
| 2604.00028 | Sequence-aware split policy for low-head-count decode (round-1 §5) |
| 2609.33410 / 2508.18850 / 2604.23553 | Round-1 ideas, now closed by constraints 4/7 + merge-order freeze |
