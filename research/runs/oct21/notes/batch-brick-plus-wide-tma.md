# Python Engineer: batch (brick + wide-TMA port) (digest)
2026-10-21; python-engineer; assembly of batch commit ff84fff (brick pair f709399 + 68c75d1 + the gated SW128 wide fp6 ring port of run-1 50ad3ce + e2d09c2 onto the split family).
**Verdict:** COMPLETE — the wide-TMA batch assembled per the placement map, but the ported wide path was later proven a semantic no-op (see wide-sw128-ring-inert.md) and the batch was never kept.
**Key facts:**
- Brick pair cherry-picked first (tip alone conflicts with 2724fd3's code); `git diff ar/parked/2f20c4f HEAD -- src/mjolnir/` → empty (tree == brick) before the port.
- Port map (12 rows): 50ad3ce monolith hunks → t5_call.py (WIDE const, wide SW128 SMEM layouts via fp8-MMA substitution, wide TMA atoms + narrow-alias else, six wide launch args), t5_decode.py (signature args at END, WIDE const + wide_ring gate + descriptor prefetch, wide ring views, per-site atom selection per e2d09c2 — no atom PHI, consumer re-points in mma_kq/mma_vp), t5_harness.py (run() params True/65536 + `--no-fp6-wide-tma` / `--fp6-wide-tma-min-l`).
- SIM102 noqa added on the two consumer re-points (same marker the brick uses).
- Commit ff84fff on 68c75d1 ← f709399 ← fd2a0c8.
**Full report:** `git show 56caea5:research/runs/oct21/notes/batch-brick-plus-wide-tma.md`
