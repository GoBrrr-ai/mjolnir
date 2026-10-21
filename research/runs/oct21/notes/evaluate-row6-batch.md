# Python Engineer: evaluate row 6 (batch fe34442)

## Plan Reference

`research/program.md` — judge keeps a > 2% geomean gain vs best kept (survives
dropping its best L, ≤ 1 L slower than noise); structured sub-bar gains park as
`ar/parked/<sha>`. Orchestrator directive: run the fixed harness, report verbatim.

## Files In Scope

- none (evaluation only; harness may append `research/ledger.{jsonl,tsv}`,
  `benchmarks/raw/verify/**`, `research/progress.png`)

## Approach

Run the fixed command verbatim, background, stdout →
`/tmp/opencode/evaluate_row6.log` (stderr → `evaluate_row6.stderr.log`).
It self-gates on GPU idle (up to 60 min wait) — never kill/restart containers.
Then report the decision line, reason line, per-L line, any revert:/parked:
line, git state, and the two verify run ids.

## Status

COMPLETE

## Result

Ledger row 6: **DISCARD** — score 1.2617 (best kept 1.2485), correctness pass,
commit fe34442. Geomean +1.1% under the +2% bar and rests on 8K (+7.4%);
+0.2% without it. Revert line: `git reset --hard 43c659b`.
Verify runs: correctness 20261021-201325, sweep 20261021-201630.
Full stdout: /tmp/opencode/evaluate_row6.log (EXIT:0).

## Files Modified

- none by me — harness appended research/ledger.{jsonl,tsv}, row-6 raw JSONs,
  regenerated research/progress.png (uncommitted, as before)
