# Verifying and benchmarking the kernel

**Story:** "I want to check the Mjolnir kernel is correct on my Thor, measure it
against FlashInfer and FlashAttention 2, and have the README charts update from
my numbers."

Everything runs in containers, one per leg: Mjolnir legs on the Mjolnir image
(`narandill/mjolnir-vllm:v<vllm>`) against the kernel source in your checkout
(`src/mjolnir/` is mounted over the installed package); the FlashInfer and FA2
reference legs on the **official** `vllm/vllm-openai:v<vllm>-ubuntu2404` image,
so they are native vLLM (`MJOLNIR_REF_IMAGE` overrides). Every stage waits for a
[clean window](clean-window-gate.md) when the live server is busy — and never
stops it.

## The stages

```bash
mjolnir verify list                 # stages + their saved runs
mjolnir verify correctness          # kernel vs the bit-faithful emulator (full matrix)
mjolnir verify correctness --quick  # one case per group, ~2 min
mjolnir verify sweep                # Mjolnir vs FlashInfer vs FA2 at every L (ncu)
mjolnir verify bench                # every benchmark stage once, with defaults
mjolnir verify all                  # correctness + bench
mjolnir verify show sweep           # re-print the latest run
```

| stage | answers | chart(s) |
|---|---|---|
| `sweep` | the headline: time per decode step vs FI and FA2 | `kernel-speedup`, `kernel-latency`, `kernel-bandwidth` |
| `hndsweep` | is the win just the head-major layout? | `hnd-layout` |
| `kvfmt` | KV formats × layouts, ncu + wall clock | `kvfmt-heatmap`, `kvfmt-tradeoff` |
| `quant` | accuracy per KV format | `quant-accuracy` |
| `ring` | TMA → MMA ring depth | `ring-heatmap` |
| `prefetch` | L2 prefetch distance | `prefetch-heatmap` |
| `pdl` | programmatic dependent launch (wall clock) | `pdl-delta` |
| `ab` | any harness flag sets | `ab-heatmap` |
| `fp6prof` | full ncu profiles: bytes vs requests vs time | `fp6prof-bytes-vs-time` |
| `timing` | harness medians, 1 vs 2 CTAs/SM | — |
| `vllm` | the plugin inside vLLM: runner tests + greedy smoke | — |

All benchmark stages default to the same eight context lengths —
**128, 4K, 8K, 16K, 32K, 64K, 128K, 256K** — so every chart covers short and
long context. `--ls "8192 65536"` narrows a run while iterating.

## Where results go

Each run writes one JSON file, `benchmarks/raw/verify/<stage>/<run_id>.json`
(schema `mjolnir.verify/1`: git commit, image, GPU, parameters, every leg's
per-L numbers), and container logs to `benchmarks/logs/verify-<run_id>/`.

A run with default parameters is a **headline** run (`"default": true`) and feeds
the README and `docs/kernel/`. A run with any non-default parameter (`--ls`,
`--variants`, `--iters`, `--page-size`, …) is saved with `"default": false`: it is
kept for comparison but never replaces a headline chart.

## Publishing

```bash
mjolnir verify bench                 # or the single stage you changed
uv run mjolnir plot --readme         # charts + README.md + docs/kernel/*.md
uv run mjolnir plot --site _site     # optional: the interactive page, locally
git add benchmarks/raw/verify assets/benchmarks README.md docs/kernel
```

CI re-renders the README from the committed raw data and fails on any diff, so
a number in the README is always one you can trace to a raw file.

## Improving the kernel

1. Put the new path behind a harness flag in `src/mjolnir/mjolnir/t5_decode.py`
   (`python -m mjolnir.t5_decode --help` lists the existing ones).
2. `mjolnir verify correctness --quick`, then the full matrix.
3. A/B it in one run, so drift on the shared GPU cancels out:

   ```bash
   mjolnir verify ab --variants "base:|new:--my_flag 1"
   ```

4. Differences inside ±7% are noise on this box; a win shows up across L and
   across two runs.
5. If it wins: flip the default, wire the env knob in `runner.py`/`backend.py`
   if vLLM should see it, run `mjolnir verify bench` + `mjolnir verify vllm`,
   re-render, and open a PR with the chapter in `docs/kernel/` updated.

Overnight, the [autoresearch loop](autoresearch.md) does steps 1–4 by itself.
