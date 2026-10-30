# Programmatic dependent launch

> PDL on/off wall clock (decode overlaps the previous kernel; merge launches early) · `mjolnir verify pdl` · **no run yet**

> [!NOTE]
> PDL does not exist on Ampere (`device_support_pdl` requires compute capability ≥ 9.0), so the `sm_87` kernel cannot use it — the stage stays pending on the Orin by construction.

## The question

Programmatic dependent launch lets the decode kernel start while the previous kernel drains, and the split-KV merge launch before decode finishes. Does it shorten a step?

## How it is measured

Wall clock (harness median of 200 iterations, cold L2) with and without `--pdl`, batch 1 and batch 8 (varlen), every L. ncu serializes kernels and cannot see PDL, so this is a wall-clock stage.

## Result


> Pending — run `mjolnir verify pdl` on a Jetson Orin, then `mjolnir plot --readme`.

## How to read it

PDL removes launch gaps, so any win is largest where the step is short (small L). Bars inside the gray ±3% band are noise; PDL stays opt-in (`MJOLNIR_ATTN_PDL=1`) unless it wins consistently.

## Reproduce

```bash
mjolnir verify pdl
```

## Ideas to try

- measure inside a full vLLM CUDA graph, where the neighbors are real model kernels: `mjolnir verify profile` with and without `MJOLNIR_ATTN_PDL=1`.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../../../.github/CONTRIBUTING.md).

---

[← L2 prefetch](prefetch.md) · [case study](../README.md#-kernel) · [Flag A/B →](ab.md)
