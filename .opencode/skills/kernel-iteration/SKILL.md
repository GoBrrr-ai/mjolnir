---
name: kernel-iteration
description: Iterate on the Mjolnir decode-attention kernel (src/mjolnir/mjolnir/t5_decode.py) — measure with `mjolnir verify`, compare against FlashInfer / FA2, keep only data-backed changes, and re-render the README from raw results.
---

# Mjolnir kernel iteration

Use this when changing the kernel or its defaults by hand (the autonomous loop is
`mjolnir autoresearch`, see `research/README.md`).

## Ground rules

- The GPU is shared with a live vLLM server: **never stop, restart or kill it**.
  Every `mjolnir verify` stage waits for a clean window (`--gate`, default on).
- Correctness first: `emulate_decode` in `t5_decode.py` is the bit-faithful
  reference; never change its math to make a test pass.
- New code paths go behind a harness flag; flip a default only with data.
- Differences under ±7% between two runs are noise (`NOISE_PCT`); compare legs
  measured *in the same run*.

## The loop

1. **Correctness** — `mjolnir verify correctness --quick` while iterating;
   the full matrix (`mjolnir verify correctness`) before a PR.
2. **Measure the change** with the stage that isolates it (all default to the 8
   headline L: 128 4K 8K 16K 32K 64K 128K 256K):

   | change | stage |
   |---|---|
   | anything | `mjolnir verify sweep` (Mjolnir vs FI vs FA2) |
   | a new flag | `mjolnir verify ab --variants "base:\|new:--my_flag 1"` |
   | KV format / layout | `mjolnir verify kvfmt`, `mjolnir verify quant` |
   | TMA ring / prefetch | `mjolnir verify ring`, `mjolnir verify prefetch` |
   | launch overhead | `mjolnir verify pdl`, `mjolnir verify timing` |
   | "why is it this fast/slow" | `mjolnir verify fp6prof` (full ncu profiles) |

   Runs with non-default parameters are saved with `default: false` and never
   replace a headline chart.
3. **Read it** — `mjolnir verify show <stage>`; raw JSON in
   `benchmarks/raw/verify/<stage>/<run_id>.json` (schema `mjolnir.verify/1`),
   container logs in `benchmarks/logs/verify-<run_id>/`.
4. **Publish** — when a change is a win: `mjolnir verify bench` (one headline run
   per stage), then `uv run mjolnir plot --readme`; commit the raw JSON, the
   re-rendered `README.md`, `docs/kernel/*.md` and `assets/benchmarks/`. CI fails
   if the README does not reproduce from the committed data.
5. **Write it up** — the chapter in `docs/kernel/<stage>.md.j2` (question /
   method / result / ideas). Text that states a number comes from the data via
   the template, never typed by hand.

## The vLLM side

After a kernel change that alters the runner contract, run `mjolnir verify vllm`
(runner tests + greedy smoke) and an end-to-end `mjolnir bench ab`
(`NVFP4_MJOLNIR,NVFP4,NVFP4_FA2`).
