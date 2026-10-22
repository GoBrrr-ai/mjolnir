# Python Engineer: evaluate row 6 (batch fe34442) (digest)
2026-10-21; python-engineer; row-6 evaluate of commit fe34442 (brick + split coarsening) — correctness run 20261021-201325, sweep run 20261021-201630.
**Verdict:** row 6 DISCARD — score 1.2617 vs best kept 1.2485 (+1.1% geomean, under the +2% bar) and it rests on 8K (+7.4%; +0.2% without it).
**Key facts:**
- Correctness pass; decision line verbatim in `/tmp/opencode/evaluate_row6.log` (EXIT:0).
- Revert line: `git reset --hard 43c659b` (the batch is not kept).
- Verify runs: correctness 20261021-201325, sweep 20261021-201630 (raw JSONs since pruned — evidence lives here + the export).
- Nothing modified by the evaluate step itself; the harness appended research/ledger.{jsonl,tsv}, row-6 raws, regenerated research/progress.png (uncommitted, as before).
- Judge rule: keeps need > 2% geomean surviving drop-best-L (research/program.md); structured sub-bar gains park as `ar/parked/<sha>`.
**Full report:** `git show 56caea5:research/runs/oct21/notes/evaluate-row6-batch.md`
