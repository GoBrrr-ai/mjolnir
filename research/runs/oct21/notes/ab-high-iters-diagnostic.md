# Python Engineer: ab-high-iters-diagnostic (digest)
2026-10-21; python-engineer; diagnostic run 20261021-162141 (verify ab --iters 150 --refs "" --no-save; base2/capped/uncapped at L=8192–65536).
**Verdict:** high-iteration triad confirms the gated ring wins only at 8K and the uncapped ring loses at ≥16K in both instruments.
**Key facts:**
- Same-run ratios: capped/base2 geomean ncu 0.967 / wall 0.992; uncapped/base2 ncu 1.006 / wall 1.032.
- capped wins only at 8K (ncu 0.870, wall 0.948); 16–64K all cells within the ±3% noise band.
- uncapped loses at ≥16K consistently (worst 64K wall 1.131 = +13%).
- Logs: benchmarks/logs/verify-20261021-162141/ (ncu `ab/t5_<leg>-L<L>.txt`, wall `ab/wall/*.log`); gate offline (idle GPU).
- CLI surface: `--iters` (default 50), `--refs` (default `fi`) accepted at stages.py:179.
**Full report:** `git show 56caea5:research/runs/oct21/notes/ab-high-iters-diagnostic.md`
