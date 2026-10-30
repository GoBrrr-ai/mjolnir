# KV ring depth

> TMA→MMA ring depth (--kv_stages) sweep at every L · `mjolnir verify ring` · **no run yet**

> [!NOTE]
> The `sm_87` mma kernel stages KV through a fixed `cp.async` pipeline — it has no `--kv_stages` knob — so this stage has nothing to sweep on the Orin and the saved run is empty. The chapter stays in the index for the cross-platform comparison.

## The question

How deep should the TMA → MMA ring be? A stage is freed only when the MMA that read it completes, so deeper rings let the loader run further ahead — but they cost shared memory, and with 2 CTAs per SM each CTA has ~113 KB.

## How it is measured

`--kv_stages 2…6` against the default (one whole KV tile at 2 CTAs/SM — 4 stages — for fp8; 2 stages for fp6, see the kernel's `_default_kv_stages`), on the harness default config, every L.

## Result


> Pending — run `mjolnir verify ring` on a Jetson Orin, then `mjolnir plot --readme`.

## How to read it

Depths that are not a multiple of a tile (K + V stages) rotate a tile's K/V across slots. Look for the depth that wins at most L by more than the noise band; a depth that only wins inside ±3% does not justify a change of default.

## Reproduce

```bash
mjolnir verify ring
mjolnir verify ring --base "--kv_format fp8" --stages "3 4 6"
```

## Ideas to try

- 1 CTA per SM with a deeper ring: `mjolnir verify ring --base "--ctas_per_sm 1"` (double the smem per CTA fits deeper rings).

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../../../.github/CONTRIBUTING.md).

---

[← Full profiles per KV format](fp6prof.md) · [case study](../README.md#-kernel) · [L2 prefetch →](prefetch.md)
