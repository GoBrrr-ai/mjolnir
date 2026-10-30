# Flag A/B

> any harness flag sets, ncu + wall clock (default: evict-first x PDL) · `mjolnir verify ab` · **no run yet**

## The question

A general-purpose A/B for harness flags. The sm_87 default run compares the KV formats themselves — fp8 vs e2m3 on HND pages — the Thor's launch knobs (evict-first, PDL) do not exist on Ampere.


## How it is measured

Any `name:flags|name:flags` list (`--variants`), ncu at every L plus a wall-clock row.

## Result


> Pending — run `mjolnir verify ab` on a Jetson Orin, then `mjolnir plot --readme`.

## How to read it

Use it for any idea: change a flag in the harness, run `mjolnir verify ab --variants ...`, and the heatmap shows where it wins. Custom runs are saved but never replace this page's default chart.

## Reproduce

```bash
mjolnir verify ab
mjolnir verify ab --variants "base:|mine:--my_flag 1" --ls "8192 65536 262144"
```

## Ideas to try

- anything — this is the experiment slot. `mjolnir autoresearch` uses it to test agent-proposed changes.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../../../.github/CONTRIBUTING.md).

---

[← Programmatic dependent launch](pdl.md) · [case study](../README.md#-kernel)
