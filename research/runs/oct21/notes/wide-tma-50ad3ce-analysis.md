# Python Engineer: wide-TMA commit 50ad3ce analysis

## Plan Reference

Subagent analysis task (orchestrator session): read the gated wide fp6 KV TMA
feature commit and answer (a) smem, (b) consumer side, (c) the gate, (d) layout
derivation. Analysis only — no edits, no GPU.

## Commits

- `50ad3cec3fdce9f4fcfece0520fcea5d973082a2` — "gated wide fp6 KV TMA: stream
  SW128 ring iff seq_len >= fp6_wide_tma_min_l (default 64K)"; 181 insertions,
  1 file (`src/mjolnir/mjolnir/t5_decode.py`); parent `eefa370` (README cleanup).
- `e2d09c2074a4b6a0b8dc2cff7e3a42d9a48df499` — legalization fix (child of
  50ad3ce, +1h12m): producer copy sites only (no TMA-atom PHI).
- Neither is an ancestor of this worktree's HEAD (`fd2a0c8`).

## Key findings

- (a) The wide ring is a SECOND VIEW over the same `tAsK` allocation
  (`cute.recast_ptr(tAsK.iterator, ...)`) — **zero added smem bytes**. fp6
  d=256 2-CTA: stage = mma_tile_m × mma_tile_k × 1 B = 128×128 = 16 KiB,
  kv_stages = 2 (`_default_kv_stages` fp6/2-CTA/4-stages-per-tile special case,
  t5_call.py:61-62) → 32 KiB ring, unchanged.
- (b) Stages are NOT byte-identical (SW128 permutes the stage differently than
  SW64); the consumer dispatch DOES get wide fragments: `tAtK_wide`/`tAtV_wide`
  are built from the wide views and re-pointed into `tAtK_cvt`/`tAtV_cvt`
  inside `if wide_ring:` in the mma_kq / mma_vp warps. fp6 KV runs on fp8-Q →
  no cvt warpgroup (`convert_warpgroups=0`, A-operand SMEM source), so the two
  MMA warps are the only ring readers.
- (c) `WIDE` = compile-time (flag `fp6_wide_tma` ∧ kv_format ∈ {e2m3,e3m2} ∧
  PS == blk_tile_s); `wide_ring = seq_len >= self.fp6_wide_tma_min_l` is a
  runtime per-sequence comparison against `mSeqLens[b_idx]`.
  `fp6_wide_tma_min_l` = token threshold (run()/CLI default 65536; __init__
  default 0 = always wide; 0 disables the gate entirely).
- (d) The wide layout is the SAME derivation function as the current fp6 ring:
  `sm100_utils.make_smem_layout_a` (t5_call.py:403 and 404-406 at HEAD), fed an
  fp8-e4m3 A-operand tiled_mma (`make_trivial_tiled_mma`, cf. t5_call.py:198-222)
  so the swizzle atom keys to SW128 instead of SW64.

## Status

COMPLETE (analysis; no production files touched)
