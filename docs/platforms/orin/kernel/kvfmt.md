# KV layouts and formats

> fp8 / fp6 / fp4 / nvfp4 KV on NHD / interleaved / HND pages (ncu + wall clock) · `mjolnir verify kvfmt` · **no run yet**

## The question

How many bits does a KV element need, and which page layout streams them best? The `sm_87` mma kernel reads fp8 (e4m3) and fp6 (e2m3) KV — the fp8 rows dequantize in shared memory on the `mma.m16n8k16` path, the fp6 rows ride the packed 6-bit LUT — so its sweep is fp8 vs e2m3 on NHD and HND pages.

## How it is measured

Every variant runs at every L with the harness default otherwise; FlashInfer on NHD and HND are the references. A wall-clock row (harness median, cold L2) adds what ncu cannot see — launch gaps. The tradeoff chart pairs each format's speed with its accuracy from [quant](quant.md).

## Result


> Pending — run `mjolnir verify kvfmt` on a Jetson Orin, then `mjolnir plot --readme`.

## How to read it

- **layout**: fp8 on NHD vs interleaved vs HND isolates the page layout;
- **format**: on HND, e2m3 reads 25% fewer bytes than fp8 — compare the measured speedup with that ideal (and see [fp6prof](fp6prof.md) for why fp6 realizes less of it);
- the tradeoff chart is the decision: up-and-left wins; e2m3 is the harness default because it sits next to fp8 on accuracy.

## Reproduce

```bash
mjolnir verify kvfmt
mjolnir verify kvfmt --variants "e2m3:--kv_format e2m3"
```

## Ideas to try

- close the hd128 gap (the top integration item): the mma serves head_size 256 only, and fp6 stays hd256-only — a 96-byte row splits into non-integral 4-byte lane slices.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../../../.github/CONTRIBUTING.md).

---

[← Head-major pages](hndsweep.md) · [case study](../README.md#-kernel) · [KV format accuracy →](quant.md)
