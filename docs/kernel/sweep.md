# Headline L-sweep

> Mjolnir vs FlashInfer vs FlashAttention 2, ncu kernel time at every L · `mjolnir verify sweep` · **no run yet**

## The question

The headline: how fast is one decode step with Mjolnir, FlashInfer and FlashAttention 2 — at every context length from 128 to 256K tokens? Short contexts are latency-bound (launch, plan, merge), long ones bandwidth-bound; a decode kernel has to win both.

## How it is measured

- one sequence, 24 query / 4 KV heads, head dim 256, 128-token pages in random order — the hd256 attention shape of Qwen3.8-27B;
- **ncu kernel time** of the decode kernel(s) + the split-KV merge, averaged over 50 launches, `--cache-control all` (cold L2: every launch reads KV from DRAM) and `--clock-control base`;
- legs: **Mjolnir e2m3** (the kernel's fastest config), **Mjolnir fp8** (what `--attention-backend MJOLNIR` serves today), **FlashInfer** tensor-core decode as vLLM calls it (fp8 KV, NHD pages), **FlashAttention 2** via `vllm_flash_attn` (bf16 KV — FA2 has no fp8 KV path);
- bandwidth = each leg's *own* KV bytes ÷ time (e2m3 = 6 bit, fp8 = 8 bit, bf16 = 16 bit).

## Result


> Pending — run `mjolnir verify sweep` on a Thor, then `mjolnir plot --readme`.

## How to read it

At L=128 the cache fits in one tile and time is mostly fixed cost: plan, launch, merge. From ~16K the cache no longer fits the 32 MiB L2 and every kernel converges toward the DRAM roofline — the remaining gap is how much of the peak each kernel sustains (bandwidth chart) and how many bytes it has to move (KV format). FA2 pays twice the bytes of every fp8 leg.

## Reproduce

```bash
mjolnir verify sweep
mjolnir verify sweep --legs fi,fi-hnd,mjolnir-fp8,mjolnir --ls "8192 65536"   # custom (not a headline run)
```

## Ideas to try

- batch > 1 and varlen batches (`--batches 8 --varlen` on the harness);
- page size 16 (what vLLM's FlashInfer backend uses on Thor) via `--page-size 16`.

Found something? Open a PR with the raw JSON from `benchmarks/raw/verify/` — see [CONTRIBUTING](../../.github/CONTRIBUTING.md).

---

[← Correctness matrix](correctness.md) · [case study](README.md) · [Head-major pages →](hndsweep.md)
