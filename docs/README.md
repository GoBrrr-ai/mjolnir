# Docs

Everything behind the Mjolnir decode-attention kernel and the vLLM stack it
ships in. Every number in this tree is traceable to raw data: kernel results in
[`benchmarks/raw/verify/`](../benchmarks/raw/verify/) (schema `mjolnir.verify/1`),
end-to-end results in [`benchmarks/history.jsonl`](../benchmarks/history.jsonl),
or upstream PR/issue anchors cited inline.

## How to read this tree

| Path | What it is |
|---|---|
| [`kernel/`](platforms/thor/kernel/) | **The kernel case study** — how Mjolnir works and why, one measured feature per chapter (layout, KV formats, accuracy, ring, prefetch, PDL, flag A/Bs, profiles). Rendered from the raw results; the index is the [platform page's `## Kernel` section](platforms/thor/README.md#-kernel). |
| [`methodology/`](methodology/) | **The measurement discipline** — the clean-window gate, ncu vs wall clock, noise bands, provenance. Read before publishing a number. |
| [`faq/`](faq/) | **Task guides for the `mjolnir` CLI** — serve, verify, A/B, new model configs, base-image bumps, autoresearch. Start at [`faq/index.md`](faq/index.md). |
| [`../research/`](../research/) | The **autoresearch** loop: program, agents, ledgers of every experiment. |

## Suggested reading order

1. [the platform page's kernel index](platforms/thor/README.md#-kernel) — the design in one section, then the chapters.
2. [`methodology/benchmarking.md`](methodology/benchmarking.md) — the rules every number follows.
3. [`kernel/sweep.md`](platforms/thor/kernel/sweep.md) — the headline: Mjolnir vs FlashInfer vs FlashAttention 2 at every context length.
4. [`../docker/vllm-thor/PATCHES.md`](../docker/vllm-thor/PATCHES.md) — the image, and how `--attention-backend MJOLNIR` is wired.
5. [`../research/README.md`](../research/README.md) — how to run (or contribute to) the autoresearch loop.

## Index

### kernel/ (the case study — rendered from raw data)
| Chapter | Question |
|---|---|
| [the kernel index](platforms/thor/README.md#-kernel) | Architecture: paged KV → TMA ring → `tcgen05` MMA → softmax warpgroups → split-KV merge. |
| [correctness.md](platforms/thor/kernel/correctness.md) | Does every layout / format / page size / q_len / Q dtype match the bit-faithful emulator? |
| [sweep.md](platforms/thor/kernel/sweep.md) | How fast is it vs FlashInfer and FA2 at every L? |
| [hndsweep.md](platforms/thor/kernel/hndsweep.md) | Is the win the layout? FlashInfer on NHD and HND pages vs Mjolnir. |
| [kvfmt.md](platforms/thor/kernel/kvfmt.md) | Which KV format and layout: fp8 / fp6 / fp4 / nvfp4 × NHD / interleaved / HND. |
| [quant.md](platforms/thor/kernel/quant.md) | What each KV format costs in accuracy. |
| [fp6prof.md](platforms/thor/kernel/fp6prof.md) | Why fp6 doesn't realize its full byte saving: full ncu profiles. |
| [ring.md](platforms/thor/kernel/ring.md) | How deep should the TMA → MMA ring be? |
| [prefetch.md](platforms/thor/kernel/prefetch.md) | Does an explicit L2 prefetch help a streamed read? |
| [pdl.md](platforms/thor/kernel/pdl.md) | Programmatic dependent launch: what overlap buys at the wall clock. |
| [ab.md](platforms/thor/kernel/ab.md) | A/B of harness flags (default: L2 evict-first × PDL). |

### methodology/
| File | TL;DR |
|---|---|
| [benchmarking.md](methodology/benchmarking.md) | Clean-window gate, ncu achieved-BW vs wall clock, bench protocols, provenance rules. |

### faq/ (task guides)
| File | TL;DR |
|---|---|
| [index.md](faq/index.md) | Question list → one guide per CLI task. |
| [setup-jetson.md](faq/setup-jetson.md) | First-time Jetson (Thor/Orin) setup: power mode, clocks, the CLI. |
| [serve-server.md](faq/serve-server.md) | `serve up/down/status/logs`, first launch, `--dry-run`. |
| [default-image.md](faq/default-image.md) | The active image: picker / state / env / flag; `image build` + `image gates`. |
| [default-model-quant.md](faq/default-model-quant.md) | The active model/config; config → attention backend → chart label. |
| [new-model-config.md](faq/new-model-config.md) | Add a model config: repo `configs/` vs user-local configs. |
| [base-image-bump.md](faq/base-image-bump.md) | Bump the vLLM base; the build as gate; re-adapting hunks; image tags follow vLLM. |
| [e2e-vs-vllm.md](faq/e2e-vs-vllm.md) | Official vLLM image vs Mjolnir end to end: `bench e2e` / `bench compare`, decode, prefill, TTFT, ITL. |
| [ab-image.md](faq/ab-image.md) | A/B a custom image or backend vs the default: `bench ab`. |
| [ab-models.md](faq/ab-models.md) | A/B different models: model-qualified legs, fairness rules. |
| [clean-window-gate.md](faq/clean-window-gate.md) | The shared-GPU rule, what "clean" means, `mjolnir gate`. |
| [results-history-charts.md](faq/results-history-charts.md) | Raw JSON → charts → README; `mjolnir plot --readme` / `--site`. |
| [settings-precedence.md](faq/settings-precedence.md) | Flag > state > env > default; state file; data paths. |
| [platforms.md](faq/platforms.md) | Detection-first platform resolution (no state), config overlays + the availability matrix, what spark/orin can't do yet. |
| [litellm-proxy.md](faq/litellm-proxy.md) | The LiteLLM proxy in front of the server. |
