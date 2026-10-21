# Round-3 arXiv brief — the residual ceiling is locked: the well is dry

Date: 2026-10-22. Third pass, after rounds 1–2
(`arxiv-decode-latency-softmax.md`, `arxiv-decode-round2-splits-merge.md`) and the
round-2/3 measurement campaign that parked the pacing brick and closed every
category. Scope: the four surfaces round 3 was explicitly asked to consider
(Q-path, tcgen05 issue stream, mbarrier/TMA issue mechanics, scheduler-level
effects) plus a fresh 2024–2026 literature scan.

**Verdict up front: DRY for ≥2% keeps — under either reading of the judge rule.**
The residual pot under all rules is the parked pacing brick (+0.8–1.0% geomean,
8K-only) plus split-count coarsening (measured +0.2%, 16K+) ≈ **+1.0–1.2%
geomean stacked, sub-bar**. Every other leg is pinned by measured or code-proven
locks. Arithmetic in §5.

---

## 1. Per-leg ledger (post round-2 measurements)

| L | State | Binding wall | Residual pot |
|---|---|---|---|
| 128 | launch-dominated, parity w/ FlashInfer | graph-node dispatch + merge floor 4–5 µs | ≈ 0 |
| 4K | latency-bound, few tile-steps/CTA | TMA box-issue + prologue at fixed boxes/byte | ≈ 0 unbricked levers; pacing wins 8K only |
| 8K | 112 µs vs 46 µs BW floor; **post-brick 164–172 GB/s — inside the 64K–256K band (168–195)** | same uniform ~10–17% above the fp6 byte-ratio floor as every L | pacing brick = the whole pot (+13% ncu 8K, parked); deeper rings wall-negative (+1.8/+9.1% at 6/8) |
| 16K–32K | L2 front-end request-rate wall | 96-B gmem boxes → 64-B L2 requests (2× fp8 count @ 0.75× bytes); **all box shapes fail** | only legal lever = fewer partials: measured **+0.2%** |
| 64K–256K | 168–195 GB/s of 273 (~70%), near the format byte ratio | request geometry locked in the vLLM cache | 0 (wider requests = cache-format change, out of scope) |

Key round-2/3 fact: **8K is no longer special.** Post-brick it sits *inside* the
long-L effective-BW band, so the "latency-bound" pot at 8K has been collected by
the parked brick, and what remains at every L ≥ 8K is one and the same uniform
gap to the format ceiling — whose every kernel-side mechanism (SW128, L2
prefetch, block-table readahead/scheduling, sub-tile fill, box shapes) is
measured dead.

## 2. Open-surface audit — each terminates in a locked quantity

- **Q-path.** Q is ~1.5 KB/CTA (6 rows × 256, fp8), loaded once in the prologue;
  per-tile A-operand re-reads from SMEM are ~13% of SMEM bandwidth — non-binding
  (the stage stream is ~3.5 µs vs ~1 µs of tensor work per tile-step). Moving Q
  to TMEM (tcgen05 A-in-TMEM) shifts no binding resource. Pot ≈ 0 at every L.
- **tcgen05 issue stream.** Tensor cores are ≤ ~20% busy at every scored L
  (51.5 GFLOP total at 8K over 112 µs across 40 CTAs). The MMA queue is never
  the wall; issue-order/decoupling variants (BMM1(t+1) before BMM2(t)) are
  already realized by the 2-stage S/P pipeline. Pot ≈ 0.
- **mbarrier/TMA issue mechanics.** The one un-swept item was per-operand
  arrival granularity (separate K and V barriers per ring stage so BMM1(t)
  issues on K(t) without waiting V(t)). It survives constraints 2–5 *nominally*
  (same 4 stages, same SMEM, same boxes/requests/order — wait granularity
  only), but its pot is ≈ 0: post-brick, the consumer is not the binding
  resource at any L (8K sits inside the long-L band), and splitting a barrier
  neither reduces box count (locked) nor raises the request-stream rate (the
  L2 front-end wall). Worth a zero-cost A/B only if the loop needs a formal
  close-out row; it cannot clear the bar.
- **Scheduler-level.** Grid is `(num_work, n_tiles, h_k)` — co-resident SM pairs
  are already adjacent chunks of the same head (optimal locality by
  construction, round-2 code facts, `t5_call.py:482`); extra blocks just queue
  (TMEM pins `ctas_per_sm=2`, O = 256 of 512 columns); persistence/CLC variants
  collapse into the retired fusion patterns. Pot ≈ 0.

## 3. Why every mechanism class in the 2024–2026 literature is out

| Mechanism class | Papers (arXiv) | Killed by |
|---|---|---|
| KV sparsity / compression / eviction | SPLASH 2609.23816; RocketKV 2502.14051; Mustafar 2505.22913; HieraSparse 2604.16864; S²ANTA 2605.01910; RDKV 2605.08317; Self-Indexing 2609.13205; SparseSpec 2512.01278 | changes results — bit-faithful emulator gate (constraint 7) |
| Cross-request prefix reuse / packing | PAT 2511.22333 | b=1 single-sequence MTP workload |
| Layout / access-granularity reform | TileLens 2607.04031 (proves granularity is a *layout* property; even its fix is a TMA/DSL-level change) | vLLM cache format out of scope (constraint 1) |
| New memory hardware | PATTON (PIM) 2609.11392; SPLASH (HBF) | hardware change |
| Warp-spec structures / compilers / simulation | Tawa 2510.14719; AsyncSparse 2604.17834; Sim-FA 2605.00555 | structures we already have; their wins are overlap with no buyer (constraint 3) or sparsity (constraint 7) |
| Microarchitecture characterization | Hopper 2501.12084; Blackwell 2512.02189, 2605.04178 | confirm TMA/TMEM behavior; no kernel-side lever |
| Canon of prior rounds | FlashDecoding 2210.03150; FlashDecoding++ 2311.01282 (declared softmax — structurally out, e4m3-P); FA3 2407.08608; FA4 2603.05451; FoldAttention 2609.33410; Sawtooth 2601.16032; FlashInfer 2501.01005; sequence-aware splits 2604.00028 | see rounds 1–2 briefs |

## 4. The dry proof (arithmetic)

- **Residual stack** (everything legal, known, measured): pacing brick
  +0.8–1.0% geomean (its only win is 8K) + coarsening +0.2% (16K+ weighted) =
  **+1.0–1.2% full-8 geomean.** Below the 2% bar.
- **Strict drop-best-L reading:** drop 8K (the brick's only leg) → the batch's
  7-leg geomean ≈ **+0.2–0.3%**. Far below 2%.
- **What a keep would require:** with 5 legs (16K–256K) pinned at the locked
  request-geometry ceiling and 128 at parity, the second-best leg must carry
  the bar alone: 7-leg geomean > 2% ⇒ `(1+x)^(1/7) > 1.02` ⇒ **x > +14.9% at
  4K** (plus ≥ +2% at 8K for the full-8 check). The best measured mechanism on
  any leg in three rounds is **+13% ncu on one leg** (pacing, 8K). The required
  effect size is ~2× beyond anything measured, on a leg whose floor analysis
  (§1) shows no available mechanism.
- **Merge floor:** ≤ 2–3 µs theoretical above the 4–5 µs dispatch/latency
  floor, ~0 practical (PDL no gain; fusion 0.56× retired; j-ordered sum frozen
  by the bit-faithful gate; fp16 partials out of tolerance). Fewer-partials
  measured +0.2%.

## 5. What would change the verdict — all out of scope

1. **vLLM cache-format change** (wider gmem boxes / per-token fp8-class KV
   layout): unlocks the uniform 10–17% residual at every L ≥ 8K *and* the
   16K–32K request wall. This is the only lever with > 2% geomean inside it.
2. Emulator-math change (declared-reference softmax, fp4 KV, sparsity):
   constraints 3/7.
3. Scored-workload change (b > 1, spec-heavy shapes, cross-request sharing):
   the metric is b=1 q_len=1.
4. Judge-rule reinterpretation: §4 shows the batch fails even the loose reading.

## 6. Recommendation

Close the loop. Rounds 1–3 converged from independent directions on one root
cause: **bytes per 96-B box in the packed-fp6 d-major cache layout**, which
fixes boxes/byte, hence the issue-rate floor (4K–8K) and the L2 request-rate
wall (16K+). The only in-scope actions left are bookkeeping: the round-2
confirmations (post-brick 8K ncu vs the 64K–256K band; `lts__t_sector_hit_rate`
across 16–32K), the zero-cost K/V-split-barrier A/B if a formal close-out row
is wanted, and optionally shipping brick+coarsening as an opt-in batch at
≈ +1% geomean.
