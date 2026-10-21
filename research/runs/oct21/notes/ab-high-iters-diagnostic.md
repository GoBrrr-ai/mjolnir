# Python Engineer: ab-high-iters-diagnostic

## Plan Reference

Orchestrator task (diagnostic only): high-iteration `verify ab` triad
(base2 / capped / uncapped) at L = 8192 16384 32768 65536, --refs "" , --no-save.

## Files In Scope

- none (diagnostic only — no source changes, no ledger row, no commit)

## Approach

Run `verify ab` with `--iters 150`, `--refs ""` (accepted: `stages.py:179`
`ref_legs = L_.legs_from_ids(refs) if refs else []`), `--no-save`.

## Result

- Run id: `20261021-162141`; logs: `benchmarks/logs/verify-20261021-162141/` (ncu
  `ab/t5_<leg>-L<L>.txt`, wall `ab/wall/*.log`). No errors, gate offline (idle GPU).
- CLI surface: `--iters` exists (default 50) → set 150; `--refs` exists (default `fi`).

Same-run ratios (Mjolnir-time, <1 = faster):

| ratio | ncu 8K | ncu 16K | ncu 32K | ncu 64K | ncu geo | wall 8K | wall 16K | wall 32K | wall 64K | wall geo |
|---|---|---|---|---|---|---|---|---|---|---|
| capped/base2 | 0.870 | 1.009 | 0.983 | 1.014 | 0.967 | 0.948 | 1.025 | 0.981 | 1.015 | 0.992 |
| uncapped/base2 | 0.862 | 1.054 | 1.052 | 1.071 | 1.006 | 0.939 | 1.013 | 1.056 | 1.131 | 1.032 |

Reading (±3% same-run noise): both gated/uncapped win only at L=8192 (beyond noise);
uncapped loses consistently at ≥16K in both instruments (worst 64K wall +13%);
capped ≈ parity at 16–64K (all cells within noise).

## Status

COMPLETE
