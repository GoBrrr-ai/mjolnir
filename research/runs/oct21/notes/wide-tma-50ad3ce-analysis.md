# Python Engineer: wide-TMA commit 50ad3ce analysis (digest)
2026-10-21; python-engineer; analysis of run-1 commits 50ad3cec3fd… ("gated wide fp6 KV TMA", 181 insertions, t5_decode.py only) + child e2d09c2074a4… (legalization fix, producer copy sites only) — neither an ancestor of fd2a0c8.
**Verdict:** COMPLETE (analysis) — the wide ring is a second view over the SAME tAsK allocation (zero added smem) with consumer-side fragment re-points; its layout derivation is the same function fed an fp8-e4m3 A-operand tiled_mma (the premise later proven to produce nothing new — wide-sw128-ring-inert.md).
**Key facts:**
- (a) Wide ring = `cute.recast_ptr(tAsK.iterator, ...)` — zero added smem bytes; fp6 d=256 2-CTA stage = 128×128×1 B = 16 KiB, kv_stages = 2 (t5_call.py:61-62 special case) → 32 KiB ring, unchanged.
- (b) Stages are NOT byte-identical (SW128 permutes differently than SW64); consumer DOES get wide fragments: tAtK_wide/tAtV_wide re-pointed into tAtK_cvt/tAtV_cvt inside `if wide_ring:` in the mma_kq/mma_vp warps; fp6 KV on fp8-Q → no cvt warpgroup (convert_warpgroups=0).
- (c) `WIDE` compile-time = flag ∧ kv_format ∈ {e2m3,e3m2} ∧ PS == blk_tile_s; `wide_ring = seq_len >= fp6_wide_tma_min_l` runtime vs mSeqLens[b_idx]; default 65536 token threshold (__init__ default 0 = always wide; 0 disables the gate).
- (d) Wide layout = same `sm100_utils.make_smem_layout_a` fed `make_trivial_tiled_mma(Float8E4M3FN, …)` so the swizzle atom keys to SW128 (t5_call.py:403-406).
**Full report:** `git show 56caea5:research/runs/oct21/notes/wide-tma-50ad3ce-analysis.md`
