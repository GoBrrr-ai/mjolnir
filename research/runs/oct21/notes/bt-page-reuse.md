# Python Engineer: bt-page-reuse (block-table page load hoisting + V reuse) (digest)
2026-10-21; python-engineer; row-3 experiment implemented (no commit by this step) then committed as 287f315 and evaluated — correctness run 20261021-091624, sweep run 20261021-091922.
**Verdict:** row 3 DISCARD — score 1.2369 vs best kept 1.2485 (geomean −0.9% ≤ +2%; 16K −5.6% is the worst L) — `--bt_page_reuse 0` restores the original reloads.
**Key facts:**
- Mechanism: per-tile block-table page indices hoisted out of the dk/dm stage loops; V arm reuses the K arm's pages (K runs exactly one tile ahead: `prefetch_iters = min(sp_stages - 1, iters_s) = 1`) — removes 1–3 dependent gmem loads per tile from the TMA producer's issue stream (fp6prof: long_scoreboard top stall).
- Spec deviation (flagged for review): the literal spec was off by one in BOTH paths (V would read tile s not s−1; the paged final V iteration never shifts) — fixed by shifting unconditionally at the top of the loop body + a second scalar `page_k_curr`; hand-simulated iters_s=3 → t0,t1,t2.
- Files: t5_decode.py (both TMA KV paths; `bt_page_reuse=True` trailing kwarg), t5_harness.py (`--bt_page_reuse` int 0/1, default 1). runner.py/emulate_decode/reduction/nvfp4/L2P untouched.
- Gates at implementation: py_compile, ruff check + format, file-length ratchet — all green.
- Evaluate: correctness 26/26 (20261021-091624) + ncu sweep vs FlashInfer (20261021-091922); full stdout /tmp/opencode/evaluate_row3.log.
**Full report:** `git show 56caea5:research/runs/oct21/notes/bt-page-reuse.md`
