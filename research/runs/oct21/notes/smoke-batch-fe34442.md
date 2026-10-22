# Python Engineer: smoke — batch fe34442 (brick + split coarsening) (digest)
2026-10-21; python-engineer; smoke of batch commit ff84fff (brick + wide-TMA port) — correctness run 20261021-195438, ref checks smoke-ref-16k / smoke-ref-256k (ad-hoc containers, no run id), triad run 20261021-200107.
**Verdict:** smoke PASS — correctness 4/4 within class, ref checks exact (chunk 16@16K / 256@256K, max 8 chunks/seq), triad shows no kill-criterion loss (cap8 ≈ cap16) — but the wide-TMA half was later proven inert by construction (wide-sw128-ring-inert.md) and the batch was never kept.
**Key facts:**
- Correctness 20261021-195438: 4/4 PASS; coarsening active only in the fused-merge varlen case (chunk 15→16, max chunks/seq 9→8); the two L=8192 cases (T=64 < 128) bit-identical to baseline 20261021-081808.
- Ref checks PASS: |kern-emu| max 0.0001219 / mean 1.465e-05 (16K); 3.041e-05 / 3.524e-06 (256K); plan logs confirm chunk=16/256, work=8, max chunks/seq=8.
- Triad (cap8/cap16): 32K wins both instruments (ncu 0.975, wall 0.979); 16K ncu +7.2% but wall −2.4%; 64K/256K ≈ noise; identical-plan 8192 legs calibrate this run's leg drift at +2.8% ncu / +6.5% wall.
- Logs: benchmarks/logs/verify-20261021-195438/, benchmarks/logs/verify-20261021-200107/ab/wall/.
**Full report:** `git show 56caea5:research/runs/oct21/notes/smoke-batch-fe34442.md`
