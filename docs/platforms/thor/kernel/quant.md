# KV format accuracy

> relative L2 error of each KV format vs unquantized KV, at every L · `mjolnir verify quant` · run `20261012-205732`

## The question

What does each KV format cost in accuracy? Speed from fewer bits is only worth it if attention output stays close to the unquantized result.

## How it is measured

The harness generates float K/V, quantizes them to each format (per-KV-head descale; NVFP4 uses its per-16 scales + a global scale), runs the kernel, and reports the **relative L2 error of the output vs attention over the unquantized float KV** with the same queries. fp8 is quantized the same way, so every format starts from the same data. Two sequences per L; the kernel must still agree with its emulator (the stage records that too).

## Result

<img src="../../../../assets/benchmarks/thor/Qwen3.8-27B/quant-accuracy.png" alt="KV format accuracy" width="100%">

- `fp8`: mean relative L2 error **0.045** (1.0× fp8's 0.045).
- `e2m3`: mean relative L2 error **0.052** (1.1× fp8's 0.045).
- `e3m2`: mean relative L2 error **0.079** (1.8× fp8's 0.045).
- `nvfp4`: mean relative L2 error **0.141** (3.1× fp8's 0.045).
- `e2m1`: mean relative L2 error **0.203** (4.5× fp8's 0.045).

| format | bits | 128 | 4K | 8K | 16K | 32K | 64K | 128K | 256K | mean |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `fp8` | 8 | 0.0434 | 0.0447 | 0.0448 | 0.0456 | 0.0461 | 0.0461 | 0.0455 | 0.0457 | **0.0452** |
| `e2m3` | 6 | 0.0456 | 0.0510 | 0.0512 | 0.0514 | 0.0521 | 0.0538 | 0.0541 | 0.0530 | **0.0515** |
| `e3m2` | 6 | 0.0756 | 0.0802 | 0.0792 | 0.0805 | 0.0797 | 0.0808 | 0.0804 | 0.0790 | **0.0794** |
| `nvfp4` | 4.5 | 0.1356 | 0.1407 | 0.1409 | 0.1402 | 0.1440 | 0.1412 | 0.1421 | 0.1400 | **0.1406** |
| `e2m1` | 4 | 0.1891 | 0.1967 | 0.2010 | 0.2045 | 0.2070 | 0.2105 | 0.2086 | 0.2068 | **0.2030** |

<sub>Rendered from [`benchmarks/raw/verify/quant/20261012-205732.json`](../../../../benchmarks/raw/verify/quant/20261012-205732.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

## How to read it

fp8 is the floor every format is compared with (it is what vLLM serves). An error a few percent above fp8 is in the same regime; several times fp8 changes model outputs and needs an end-to-end eval before use.

## Reproduce

```bash
mjolnir verify quant
```

## Ideas to try

- end-to-end evals for fp4 KV on a real model (`mjolnir bench accuracy` — the fp6 KV rows already live in [accuracy](accuracy.md));
- finer-grained scales for e2m1.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../../../.github/CONTRIBUTING.md).

---

[← KV layouts and formats](kvfmt.md) · [case study](README.md) · [KV cache capacity →](kv-cache-size.md)
