# Python Engineer: batch (brick + split coarsening)

## Plan Reference

Orchestrator session: compose `ar/parked/2f20c4f` (the 4-stage ring + pacing
brick, linear pair f709399 + 68c75d1) + the split coarsening experiment
(t5_plan.py + t5_harness.py only). No GPU run — smoke comes after review.

## Approach

1. Cherry-pick the brick pair (tip alone conflicts; pair verified last time),
   verify tree == brick via `git diff ar/parked/2f20c4f HEAD -- src/mjolnir/`.
2. Split coarsening: `LONG_SPLIT_MIN_TILES = 128` in t5_plan.py +
   `long_split_cap` kwarg on `plan_split_kv` (0 = off); harness run() param
   `long_split_cap: int = 8` (after `fused_merge`) + CLI flag. Kernel/merge
   untouched — n_splits stays <= compile-time max_splits=16 capacity.
3. Gates: py_compile, ruff, file-length ratchet. Commit. No GPU.

## Status

COMPLETE

## Files Modified

- src/mjolnir/mjolnir/t5_plan.py — LONG_SPLIT_MIN_TILES=128 + long_split_cap
  kwarg on plan_split_kv (0 = off); eff_max_splits only tightens the `lo` floor
  of the existing search — binary search, work emission, assert untouched
- src/mjolnir/mjolnir/t5_harness.py — run(long_split_cap=8) (after fused_merge),
  pass-through to plan_split_kv, --long_split_cap CLI flag (after --max_splits)

## Verification (no GPU, per orchestrator)

- Brick: f709399+68c75d1 cherry-picked cleanly; `git diff ar/parked/2f20c4f
  HEAD -- src/mjolnir/` EMPTY before the coarsening commit
- Planner behavior (torch stubbed, host math only): b=1 off = 10 splits for
  every L >= 1280; cap 8 -> 8 splits for T >= 128 (chunk 16@16K, 32@32K,
  64@64K, 256@256K); L <= 8K untouched; max(n_splits) <= 16 invariant holds
  over a 128..262144 sweep x cap {0,8,16}; runner.py backend call unaffected
  (kwargs, default 0)
- Gates: py_compile PASS; ruff 0.14.14 check PASS; ruff format PASS;
  check_file_length --base origin/main PASS; pre-commit (ty typecheck incl.) PASS

## Commit

fe34442 (on bdf7e36 <- 0379e63 <- fd2a0c8)
