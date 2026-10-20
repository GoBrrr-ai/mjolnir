<!-- Use for benchmark drops from a Thor: `mjolnir verify <stage>` kernel
     results or `mjolnir bench` end-to-end runs. Rules:
     docs/methodology/benchmarking.md -->

## The run

- command (exact, incl. any narrowing flags like `--ls` / `--batch`):
- run id(s): `benchmarks/raw/verify/<stage>/<run_id>.json` and/or `benchmarks/raw/perf-<ts>/`
- default run (all parameters at stage defaults — feeds the headline charts) or narrowed:
- host: JetPack / L4T version, power mode

## Artifacts

- [ ] raw JSON committed (`benchmarks/raw/...`)
- [ ] e2e: row appended to `benchmarks/history.jsonl`
- [ ] re-rendered: `uv run mjolnir plot --readme` → `README.md`, `docs/platforms/`, `assets/benchmarks/`
- [ ] clean window: stages gate themselves by default; if in doubt, attach the `mjolnir gate --once` output or the run log

## What the numbers say

<!-- one or two lines; link the raw paths. A run that shows a dead end is a
     contribution too. -->
