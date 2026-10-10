# Docs

Everything behind the Mjolnir decode-attention kernel and the vLLM stack it
ships in. Every number in this tree is traceable to raw data: kernel results in
[`benchmarks/raw/verify/`](../benchmarks/raw/verify/) (schema `mjolnir.verify/1`),
end-to-end results in [`benchmarks/history.jsonl`](../benchmarks/history.jsonl),
or upstream PR/issue anchors cited inline.

## How to read this tree

| Path | What it is |
|---|---|
| [`kernel/`](kernel/) | **The kernel case study** — how Mjolnir works and why, one measured feature per chapter (layout, KV formats, accuracy, ring, prefetch, PDL, flag A/Bs, profiles). Rendered from the raw results; start at [`kernel/README.md`](kernel/README.md). |
| [`methodology/`](methodology/) | **The measurement discipline** — the clean-window gate, ncu vs wall clock, noise bands, provenance. Read before publishing a number. |
| [`faq/`](faq/) | **Task guides for the `mjolnir` CLI** — serve, verify, A/B, new model configs, base-image bumps, autoresearch. Start at [`faq/index.md`](faq/index.md). |
| [`thor-stack/`](thor-stack/) | **The serving stack** — why the image patches exist: route traces, FP8 on `sm_110`, GDN prefill, base bumps, end-to-end forensics. |
| [`fa4-hd256-fp8/`](fa4-hd256-fp8/) | **The road to Mjolnir** — the earlier FA4 head_dim=256 decode workstream (feasibility, 1-CTA patch, SplitKV) whose lessons set the kernel's design. Historical. |
| [`draft-cudagraph/`](draft-cudagraph/) | The draft-model CUDA-graph c1 regression study and its gate patch. |
| [`research/`](research/) | Research notes — upstream status, prior-art surveys, field notes. |
| [`../research/`](../research/) | The **autoresearch** loop: program, agents, ledgers of every experiment. |

The retired GEMV decode kernel and its reports are in git history:
[`06a9ff4:deprecated/gemv/`](https://github.com/thor-arena/mjolnir/tree/06a9ff4/deprecated/gemv).

## Suggested reading order

1. [`kernel/README.md`](kernel/README.md) — the design in one page, then the chapters.
2. [`methodology/benchmarking.md`](methodology/benchmarking.md) — the rules every number follows.
3. [`kernel/sweep.md`](kernel/sweep.md) — the headline: Mjolnir vs FlashInfer vs FlashAttention 2 at every context length.
4. [`../docker/vllm-thor/PATCHES.md`](../docker/vllm-thor/PATCHES.md) — the image, and how `--attention-backend MJOLNIR` is wired.
5. [`../research/README.md`](../research/README.md) — how to run (or contribute to) the autoresearch loop.

## Index

### kernel/ (the case study — rendered from raw data)
| Chapter | Question |
|---|---|
| [README.md](kernel/README.md) | Architecture: paged KV → TMA ring → `tcgen05` MMA → softmax warpgroups → split-KV merge. |
| [correctness.md](kernel/correctness.md) | Does every layout / format / page size / q_len / Q dtype match the bit-faithful emulator? |
| [sweep.md](kernel/sweep.md) | How fast is it vs FlashInfer and FA2 at every L? |
| [hndsweep.md](kernel/hndsweep.md) | Is the win the layout? FlashInfer on NHD and HND pages vs Mjolnir. |
| [kvfmt.md](kernel/kvfmt.md) | Which KV format and layout: fp8 / fp6 / fp4 / nvfp4 × NHD / interleaved / HND. |
| [quant.md](kernel/quant.md) | What each KV format costs in accuracy. |
| [fp6prof.md](kernel/fp6prof.md) | Why fp6 doesn't realize its full byte saving: full ncu profiles. |
| [ring.md](kernel/ring.md) | How deep should the TMA → MMA ring be? |
| [prefetch.md](kernel/prefetch.md) | Does an explicit L2 prefetch help a streamed read? |
| [pdl.md](kernel/pdl.md) | Programmatic dependent launch: what overlap buys at the wall clock. |
| [ab.md](kernel/ab.md) | A/B of harness flags (default: L2 evict-first × PDL). |

### methodology/
| File | TL;DR |
|---|---|
| [benchmarking.md](methodology/benchmarking.md) | Clean-window gate, ncu achieved-BW vs wall clock, bench protocols, provenance rules. |

### faq/ (task guides)
| File | TL;DR |
|---|---|
| [index.md](faq/index.md) | Question list → one guide per CLI task. |
| [setup-thor.md](faq/setup-thor.md) | First-time Thor setup: power mode, clocks, the CLI. |
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
| [litellm-proxy.md](faq/litellm-proxy.md) | The LiteLLM proxy in front of the server. |

### thor-stack/ (serving stack)
| File | TL;DR |
|---|---|
| [base-bump-2026-09-26.md](thor-stack/base-bump-2026-09-26.md) | Base bump to the vLLM 0.30.0 release image: per-patch outcomes and canaries. |
| [fa4-hd256-route-trace.md](thor-stack/fa4-hd256-route-trace.md) | The downgrade path for FA4 + FP8-KV + hd256 on `sm_110`. |
| [fa4-fp8kv-sm110.md](thor-stack/fa4-fp8kv-sm110.md) | Proof FA4 hd128 FP8 2-CTA runs on `sm_110a` (probes, PTX evidence). |
| [gdn-prefill-fi-enable.md](thor-stack/gdn-prefill-fi-enable.md) | GDN prefill enablement: the 1-line FlashInfer hunk, gates, live canary. |
| [nightly-bump-process.md](thor-stack/nightly-bump-process.md) | Bumping the vLLM base and re-verifying the patch stack. |
| [spec-decode-e2e-forensics.md](thor-stack/spec-decode-e2e-forensics.md) | End-to-end forensics: FA4 hd256 vs FlashInfer serving logs. |

### fa4-hd256-fp8/ (historical — the road to Mjolnir)
| File | TL;DR |
|---|---|
| [decode-kernel-feasibility.md](fa4-hd256-fp8/decode-kernel-feasibility.md) | Go/no-go on a custom hd256 decode kernel: Thor is bandwidth-bound at M=1. |
| [dk1a-1cta-paged.md](fa4-hd256-fp8/dk1a-1cta-paged.md) | 1-CTA + paged-128 TMA wiring: 1.60–1.62× vs 2-CTA. |
| [decode-microbench.md](fa4-hd256-fp8/decode-microbench.md) | First clean microbench: 2-CTA vs 1-CTA, FlashInfer baseline, achieved BW vs roofline. |
| [v11-1cta-patch.md](fa4-hd256-fp8/v11-1cta-patch.md) | The 1-CTA patch: design, inventory, ×1.56. |
| [splitkv-sweep.md](fa4-hd256-fp8/splitkv-sweep.md) | SplitKV split-count sweep; where the knee sits. |
| [real-splitkv-bench.md](fa4-hd256-fp8/real-splitkv-bench.md) | Paged SplitKV on the 1-CTA kernel; the stages regression. |
| [decode-tiling-design.md](fa4-hd256-fp8/decode-tiling-design.md) | Tiling notes for the decode kernels. |
| [decode-1cta-clean.md](fa4-hd256-fp8/decode-1cta-clean.md) | Clean-window 1-CTA bench: the reference numbers. |

### draft-cudagraph/
| File | TL;DR |
|---|---|
| [c1-regression-study.md](draft-cudagraph/c1-regression-study.md) | The c1 regression: why FA2 `batch_prefill` is replay-safe; the gate patch. |

### research/ (notes)
| File | TL;DR |
|---|---|
| [fa4-hd256-upstream-status.md](research/fa4-hd256-upstream-status.md) | Upstream FA4 hd256 status (2026-09-21): RFC table, policy gates. |
| [draft-decode-metadata-prior-art.md](research/draft-decode-metadata-prior-art.md) | Prior art for `update_draft_decode_metadata`: upstream vs net-new. |
| [thor-build-env-vars.md](research/thor-build-env-vars.md) | Build/runtime env-var audit for vLLM on Thor aarch64. |
| [nvfp4-sm110-field-notes.md](research/nvfp4-sm110-field-notes.md) | NVFP4/FP8 on `sm_110`/`sm_12x` field notes. |
| [tool-eval-bench-perf-runs.md](research/tool-eval-bench-perf-runs.md) | What tool-eval-bench's perf mode does, and why mjolnir wraps llama-benchy. |
