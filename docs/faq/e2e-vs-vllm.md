# Official vLLM image vs Mjolnir, end to end

**Story:** "I want to know what the Mjolnir image buys me over the official
vLLM image when serving a real model — decode speed, prefill speed, time to
first token, inter-token latency."

## The command

```bash
mjolnir bench e2e            # official image (NVFP4), then the Mjolnir image (NVFP4_MJOLNIR)
mjolnir bench compare        # re-print the comparison from history, no new run
mjolnir bench compare --json # the same, for scripts
```

`bench e2e` serves each image in turn **(it restarts the vLLM server per
leg — the GPU must be yours for the run)**, waits for a clean window, runs
llama-benchy on it, appends one record per image to
`benchmarks/history.jsonl`, prints the comparison table and re-renders the
README — the opening infographic and the "Official vLLM image vs Mjolnir"
table come from this pair.

Defaults:

| flag | default | |
|---|---|---|
| `--official-image` | `vllm/vllm-openai:v<vLLM>-ubuntu2404` | the upstream image the Mjolnir image is built on |
| `--official-config` | `NVFP4` | stock backends only (vLLM picks FlashInfer on Thor) |
| `--image` / `--config` | `narandill/mjolnir-vllm:v<vLLM>` / `NVFP4_MJOLNIR` | the Mjolnir image + backend |
| `--depths` | `0,4096,8192,16384,32768` | tokens already in the KV cache |
| `--concurrency` | `1,2,4` | simultaneous requests |
| `--pp` / `--tg` | `2048` / `128` | prompt / generated tokens per request |
| `--runs` / `--repeat` | `6` / `1` | measured runs per cell / independent gated sweeps |

`--dry-run` prints both `docker run` commands without touching the server.

## Why the baseline runs on the official image

The Mjolnir image carries a patch stack that also touches FlashInfer paths
(CUDA-graph modes, spec-decode metadata, GDN prefill). FlashInfer served from
*our* image is therefore not native vLLM — and it flatters the baseline. Native
backends are only ever measured on the untouched `vllm/vllm-openai` image;
`bench compare`, the README table, the e2e charts and the infographic ignore
`NVFP4` / `NVFP4_FA2` runs served from a Mjolnir image (they stay in
`history.jsonl`, unused).

## What is measured

| metric | from llama-benchy | better |
|---|---|---|
| **decode** t/s | `tg_throughput` (all requests together) | higher |
| **prefill** t/s | `pp_throughput` | higher |
| **TTFT** ms | `e2e_ttft` — request sent → first token | lower |
| **ITL** ms | `1000 / tg_req_throughput` — the gap between tokens of one request | lower |

Gains are always "+ = Mjolnir better": for TTFT and ITL that is
official ÷ Mjolnir − 1. The README marks a gain beyond ±7% with ✅ and
anything inside it with 🟰.

The infographic picks **one request at the longest context both runs
measured** — the decode-bound case the kernel is built for — and the table
shows every cell.

## Fair-comparison rules

- Same model, same serving config family, same host, back to back.
- The two legs are separate server launches; first-launch JIT/caches are on
  persistent volumes, and llama-benchy discards warm-up runs.
- A single cell moving a few percent is noise; trust a gain that holds across
  contexts and concurrency, and re-run with `--repeat 2` before publishing.
