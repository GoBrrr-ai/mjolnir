# Python Engineer: batch (brick + split coarsening) (digest)
2026-10-21; python-engineer; assembly of batch commit fe34442 (on bdf7e36 ← 0379e63 ← fd2a0c8) — brick pair f709399 + 68c75d1 cherry-picked + split coarsening; no GPU in this step.
**Verdict:** COMPLETE — the row-6 batch assembled and gated (it later evaluated DISCARD 1.2617, resting on 8K: +0.2% without it).
**Key facts:**
- Brick: `ar/parked/2f20c4f` = linear pair 2724fd3 (4-stage ring + pacing) → 2f20c4f (counter gated to chunks > 13 tiles); tip alone conflicts (edits 2724fd3's code) so both cherry-picked; `git diff ar/parked/2f20c4f HEAD -- src/mjolnir/` EMPTY before coarsening.
- t5_plan.py: `LONG_SPLIT_MIN_TILES = 128` + `long_split_cap` kwarg on plan_split_kv (0 = off); eff_max_splits only tightens the `lo` floor — binary search, work emission, assert untouched; n_splits stays ≤ compile-time max_splits=16.
- t5_harness.py: run(long_split_cap=8) after fused_merge + `--long_split_cap` CLI flag.
- Planner stub-verified (torch stubbed, host math): b=1 off → 10 splits for every L ≥ 1280; cap 8 → 8 splits for T ≥ 128 (chunk 16@16K, 32@32K, 64@64K, 256@256K); L ≤ 8K untouched; max(n_splits) ≤ 16 over a 128..262144 sweep × cap {0,8,16}.
- Gates: py_compile, ruff 0.14.14 check + format, file-length ratchet, pre-commit (ty incl.) — all PASS.
**Full report:** `git show 56caea5:research/runs/oct21/notes/batch-brick-plus-split-coarsening.md`
