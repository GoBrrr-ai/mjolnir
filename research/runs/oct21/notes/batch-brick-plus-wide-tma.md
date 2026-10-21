# Python Engineer: batch (brick + wide-TMA port)

## Plan Reference

Orchestrator session: compose `ar/parked/2f20c4f` (the 4-stage + pacing brick) +
the gated SW128 wide fp6 ring port (50ad3ce + e2d09c2 legalization) onto the
split family. Faithful port — same mechanism, same gates, same legalization
pattern; adapt only placement/interaction points.

## Progress

- Step 1 DONE: `ar/parked/2f20c4f` is the linear pair `2724fd3` (4-stage ring +
  pacing) → `2f20c4f` (counter gated to chunks > 13 tiles), both on fd2a0c8.
  Picking the tag tip alone conflicts (its hunks edit 2724fd3's code), so both
  were cherry-picked in order: f709399 + 68c75d1.
  `git diff ar/parked/2f20c4f HEAD -- src/mjolnir/` → empty (tree == brick).
- Step 2 IN PROGRESS: port wide TMA onto the split family (t5_call.py host
  hunks + t5_decode.py device hunks + t5_harness.py run()/CLI).

## Port map (50ad3ce monolith → split family @ 68c75d1)

| 50ad3ce hunk | Split-family home |
|---|---|
| `__init__` params + store | t5_call.py `__init__` (after ring_uncapped) |
| SMEM layouts + WIDE + wide MMA subst | t5_call.py `__call__` after smem_layout_k/v |
| wide TMA atoms + else-alias | t5_call.py after narrow atoms (PS==blk_tile_s block) |
| decode launch args | t5_call.py decode(...) call |
| decode() signature | t5_decode.py — wide args at END (per instruction) |
| WIDE const + wide_ring gate | t5_decode.py prologue |
| prefetch_descriptor | t5_decode.py init_warp |
| wide smem views + fragments | t5_decode.py after narrow tAsK/tAtV setup |
| loader wide tilings + per-site atom sel (e2d09c2) | t5_decode.py PS==blk_tile_s branch |
| consumer re-points (mma_kq, mma_vp) | t5_decode.py |
| run() params + pass-through | t5_harness.py |
| CLI flags | t5_harness.py |

## Status

COMPLETE

## Files Modified

- src/mjolnir/mjolnir/t5_call.py — WIDE const + wide SW128 SMEM layouts
  (fp8-MMA substitution), wide TMA atoms + narrow-alias else branch, the six
  wide objects passed through the decode() launch (at the END, matching the
  signature)
- src/mjolnir/mjolnir/t5_decode.py — `__init__` params/stores (signature lives
  here, not t5_call.py), decode() wide args at END, WIDE const + wide_ring
  gate + descriptor prefetch in the prologue, wide ring views, loader wide
  tilings/partitions + per-site atom selection (e2d09c2 — no atom PHI),
  consumer re-points in mma_kq/mma_vp
- src/mjolnir/mjolnir/t5_harness.py — run() params (True/65536 defaults) +
  pass-through + `--no-fp6-wide-tma` / `--fp6-wide-tma-min-l`

## Commit

ff84fff batch: ar/parked/2f20c4f (4-stage ring + pacing) + the gated SW128
wide fp6 ring port (50ad3ce+e2d09c2) — on 68c75d1 ← f709399 (the cherry-picked
brick pair) ← fd2a0c8.

## Notes

- `ar/parked/2f20c4f` = linear pair 2724fd3 → 2f20c4f; the tip alone conflicts
  (edits 2724fd3's code) — both cherry-picked; tree == brick verified.
- SIM102 noqa added on the two consumer re-points (same marker the brick uses
  for the identical const_expr+runtime nested-if pattern).
- Loader comment "point the TMA objects at the wide ring" kept verbatim from
  50ad3ce (e2d09c2 removed the rebinds but left the comment).
