# Correctness matrix

> kernel vs the bit-faithful emulator: every layout, format, page size, q_len, Q dtype · `mjolnir verify correctness` · run `20261014-154855`

## The question

Before any speed number counts, the kernel has to compute the same thing as attention. This stage runs the kernel harness over a matrix of shapes and options and compares the output **and** the log-sum-exp against a *bit-faithful emulator* — a PyTorch re-implementation that rounds Q, K, V and P to the same fp8/fp6/fp4 values at the same points as the kernel, so any mismatch is a kernel bug, not quantization noise.

## How it is measured

- every case runs once per Q dtype (fp8, bf16) in a fresh container;
- unused pages in the KV cache are **poisoned with NaN**: if the page-table walk ever touches a page it should not, the output turns NaN and the case fails;
- cases pin the vLLM production config (fp8 KV, NHD pages) unless they test something else, and a `default:` group runs the harness default (e2m3 KV on HND pages);
- groups cover lengths (1 token to 65K, partial tiles, ring wrap-around), batches and varlen, speculative q_len 2–4, page sizes 16–128, KV layouts (vLLM interleaved, FlashInfer NHD, HND), strided Q/O, descale granularities, the KV formats, PDL / L2 prefetch / evict-first launch options.

## Result

- **94 / 94** checks pass (kernel output and LSE vs the bit-faithful emulator, poisoned unused pages must stay NaN-free).

| case group | passed | Q dtypes |
|---|--:|---|
| shapes / batches | ✅ 28/28 | bf16, fp8 |
| scheduling / paging | ✅ 14/14 | bf16, fp8 |
| KV layouts / strides | ✅ 18/18 | bf16, fp8 |
| launch options | ✅ 14/14 | bf16, fp8 |
| KV formats | ✅ 10/10 | fp8 |
| default config | ✅ 10/10 | bf16, fp8 |

<sub>Rendered from [`benchmarks/raw/verify/correctness/20261014-154855.json`](../../benchmarks/raw/verify/correctness/20261014-154855.json). Ratios: FlashInfer time ÷ variant time; green = faster beyond the ±3% noise band, red = slower, gray = within noise.</sub>

## How to read it

A ❌ in any group blocks a change. The `check:` line in each case's log reports the max abs error of output and LSE against the emulator (tolerance 0.05).

## Reproduce

```bash
mjolnir verify correctness                  # all cases, fp8 + bf16 Q
mjolnir verify correctness --match nvfp4 --dtypes fp8   # a subset
```

## Ideas to try

- add a case for any shape your model uses (`stages.CASES` in `src/mjolnir-cli/mjolnir_cli/verify/stages.py`); GQA ratios other than 6 are already covered by the 40 query / 8 KV drafter cases.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../.github/CONTRIBUTING.md).

---

[case study](README.md) · [Headline L-sweep →](sweep.md)
