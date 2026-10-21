# Python Engineer: bt-page-reuse (block-table page load hoisting + V reuse)

## Plan Reference

Orchestrator request (autoresearch/oct21 @ fd2a0c8): Task 1 — reset to fd2a0c8
(proved kernel-identical to 43c659b, 0-byte diff on src/mjolnir/). Task 2 —
implement the bt_page_reuse experiment in t5_decode.py + t5_harness.py, no commit.

## Files In Scope

- src/mjolnir/mjolnir/t5_decode.py
- src/mjolnir/mjolnir/t5_harness.py

## Approach

The TMA-producer warp re-reads the block table on its critical path (fp6prof:
long_scoreboard top stall). K runs exactly one tile ahead of V
(`prefetch_iters = min(sp_stages - 1, iters_s) = 1` whenever the loop runs,
`sp_stages = 2`), so V can reuse the K arm's pages; in the paged path each page
index is also re-loaded inside the per-stage dk/dm loops. Load per-tile page
indices once into loop-carried rmem/scalars; V reads the shifted-out pair.

## Deviation from the literal spec (correctness, flagged for review)

The spec's placement is off by one in BOTH paths; corrected while keeping every
stated invariant ("tile s - prefetch_iters == the K arm's tile s - 1"):

- Scalar path as spec'd (K arm: `page_k_prev = page` after the load; V reads
  `page_k_prev`): within iteration s the K arm runs BEFORE the V arm, so V
  would read tile **s**, not tile s-1 — every interior V tile off by one.
- Paged path as spec'd (shift inside the K arm): the final V iteration
  (s = iters_s) has no K arm, so no shift happens and V would read tile
  iters_s-2 instead of iters_s-1 — stale tail tile.
- Fix (both paths): shift unconditionally at the TOP of the loop body, before
  the K arm; K arm writes the curr slot; V arm reads prev. Simulation
  (iters_s=3): s0 shift(garbage→never read), curr←t0; s1 prev←t0,curr←t1,V:t0 ✓;
  s2 prev←t1,curr←t2,V:t1 ✓; s3 no-K, prev←t2, V:t2 ✓. Needs a second scalar
  `page_k_curr` in the scalar path (same curr/prev pair the paged path has).
- Everything else exactly per spec: names, guard conditions, acquire/copy
  structure, `l_idx` sourcing, `--bt_page_reuse` CLI, harness plumbing.

## SOLID Analysis

- S: change is confined to the TMA KV producer's page bookkeeping; no other
  warp's contract changes.
- D: `bt_page_reuse` is a kernel knob defaulted True; harness + runner pass it
  by keyword, so the trailing __init__ kwarg is additive.

## Status

COMPLETE

## Files Modified

- src/mjolnir/mjolnir/t5_decode.py — `bt_page_reuse=True` trailing kwarg; per-tile
  page bookkeeping hoisted out of the stage loops in BOTH TMA KV paths
  (PS == blk_tile_s: `page_k_curr`/`page_k_prev` scalars; paged: `pages_curr`/
  `pages_prev` rmem tensors); V arm reads prev under `cutlass.const_expr`.
- src/mjolnir/mjolnir/t5_harness.py — `run(bt_page_reuse=True)` plumbed into the
  constructor; `--bt_page_reuse` CLI flag (int 0/1, default 1) next to `--l2_prefetch`.
- Untouched: runner.py, emulate_decode, reduction kernel, nvfp4 packed path, L2P
  prefetch blocks. Nothing committed.

## Gates

- python3 -m py_compile (both files): OK
- uvx ruff@0.14.14 check (both files): All checks passed!
- uvx ruff@0.14.14 format --check: 2 files already formatted
- scripts/check_file_length.py --base origin/main: exit 0

## Notes for Reviewer

The literal spec had an off-by-one in both paths (see "Deviation" section above):
the scalar write after the K load is read by the SAME iteration's V arm (tile s,
not s-1), and the paged shift inside the K arm never runs for the final V
iteration (s = iters_s has no K arm → stale tile iters_s-2). Corrected by
shifting unconditionally at the top of the loop body; needs one extra scalar
`page_k_curr`. All V tiles verified by hand-simulation (iters_s=3 → t0,t1,t2).
The `bt_page_reuse=0` path reloads exactly as the original code did.

## Evaluate outcome (row 3)

Committed as `287f315`; `uv run mjolnir autoresearch evaluate` ran correctness
(26/26 pass, `20261021-091624`) + ncu sweep vs FlashInfer (`20261021-091922`).
Verdict: **DISCARD** — score 1.2369 vs best kept 1.2485, geomean −0.9% ≤ +2%
(16K −5.6% is the worst L). Full stdout: `/tmp/opencode/evaluate_row3.log`.
