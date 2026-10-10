---
name: "Kernel idea / experiment"
about: Propose (or report) a Mjolnir kernel experiment — positive or negative results welcome
title: "kernel: "
labels: ["kernel"]
assignees: []
---

## Hypothesis

One sentence, testable: "doing X will make the decode step faster at L ≥ … because …".

## Where it lands

The part of `src/mjolnir/mjolnir/t5_decode.py` it touches, and the harness flag
it would go behind (e.g. `--merge_fused`).

## Evidence / prior art

Papers (arXiv ids), CUTLASS / FlashInfer / FlashAttention code, ncu numbers
from `mjolnir verify fp6prof`, the kernel chapter it relates to (`docs/kernel/…`).

## Results (if you ran it)

- [ ] `mjolnir verify correctness` passes
- [ ] same-run A/B: `mjolnir verify ab --variants "base:|new:--my_flag 1"` — attach
      `benchmarks/raw/verify/ab/<run_id>.json`
- [ ] or an autoresearch ledger row (`research/runs/<tag>/`)

Negative results are useful: say what you tried and what the data showed.
