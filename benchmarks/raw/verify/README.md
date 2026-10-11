# Kernel results (`mjolnir verify`)

One JSON document per `mjolnir verify <stage>` run:

```bash
benchmarks/raw/verify/<stage>/<run_id>.json      # schema "mjolnir.verify/1"
```

Stages: `correctness`, `sweep`, `hndsweep`, `kvfmt`, `quant`, `ring`, `prefetch`,
`pdl`, `ab`, `fp6prof`, `timing`, `vllm`. The schema (header, `params`, per-leg
`series` with per-L points, harness `rows`, correctness `cases`, ncu `profiles`)
is documented in
[`mjolnir_cli/verify/results.py`](../../../src/mjolnir-cli/mjolnir_cli/verify/results.py).

The README, `docs/kernel/` and the charts read the **latest run with
`"default": true`** of each stage — a run made with the stage's default
parameters (all eight L: 128 … 256K). Runs with custom parameters, and every
autoresearch experiment, are saved with `"default": false`: kept for
comparison, never a headline.

To add yours: `mjolnir verify bench` on a Thor, then
`uv run mjolnir plot --readme`, and commit this directory together with the
re-rendered README, `docs/kernel/` and `assets/benchmarks/`
([docs/faq/verify-kernel.md](../../../docs/faq/verify-kernel.md)).
