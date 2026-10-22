# Python Engineer: ring-pacing Phase 1 (4-stage fp6 ring + runtime producer pacing) (digest)
2026-10-21; python-engineer; Phase-1 implementation of the pacing experiment (became the parked brick pair 2724fd3 → 2f20c4f).
**Verdict:** COMPLETE — the full pacing mechanism landed in the worktree with all gates green (it later hung in smoke → the deadlock fix, then parked as `ar/parked/2f20c4f`).
**Key facts:**
- t5_call.py: `_default_kv_stages` fp6/2-CTA/d=256 → 4 stages.
- t5_decode.py: DEEP_ITERS=13, `ring_uncapped` param, smem `ring_consumed` counter + warp-0 zero, producer pacing in both TMA KV paths (PS==blk_tile_s + paged), elected consumer adds at both release sites.
- t5_harness.py: `--ring_uncapped` threaded through run().
- Spec deviation (flagged): the task's inline depth formula was degenerate for the only tracing config (always kv_stages); implemented the inversion — cap = kv_stages when (uncapped or chunk ≤ 13 tiles), else 2; runtime if/else depth selection can't bind-escape staged regions (t5_call.py:242).
- Gates: py_compile OK; ruff 0.14.14 PASS; check_file_length --base origin/main EXIT 0; image import check OK; the SIM102 noqa on the zeroing if is load-bearing (verified via /tmp copy).
**Full report:** `git show 56caea5:research/runs/oct21/notes/ring-pacing-phase1.md`
