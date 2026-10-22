# Python Engineer: warm-up-only L2 prefetch default — smoke review (digest)
2026-10-21; python-engineer; review + GPU smokes of the uncommitted warm-up-only L2-prefetch default (later ledger row 2, a9fd954, DISCARD 1.2402) — smoke runs 20261021-083931 (A: correctness) and 20261021-084014 (B: p0 vs p2 A/B).
**Verdict:** smoke-green — correctness 4/4 and the A/B at parity within ±3% — but the ledger later scored the default leg a discard (row 2, 1.2402, geomean −0.7%).
**Key facts:**
- Change: `l2_prefetch` default 0 → 2 (warm-up only; the steady-state variant was retired — it lost 7–27% at ≥8K); nvfp4 path forces `self.l2_prefetch = 0`; PS==128 steady-state prefetch arms removed (only `s == 0` prefetches); harness/CLI defaults updated.
- Gates: py_compile OK, ruff 0.14.14 clean, check_file_length exit 0.
- Smoke A (e2m3 correctness): all 4 pass — fp8 varlen 4.38e-04 max / bf16 default 3.91e-03 max / fused merge 9.73e-04 max.
- Smoke B (p0 vs p2, L=4096/262144): wall p0 51.6/2077.4 µs vs p2 51.4/2087.6 µs (parity); ncu 77.7/1932.1 vs 77.3/1939.2 (~0.4%); 1.18× / 1.33–1.34× vs FlashInfer.
- Nothing committed by this step.
**Full report:** `git show 56caea5:research/runs/oct21/notes/l2-prefetch-warmup-default.md`
