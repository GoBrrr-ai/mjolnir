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
llama-benchy on it, stops that leg's server, appends one record per image to
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

`--only mjolnir` (or `--only official`) runs one leg — e.g. after rebuilding
the Mjolnir image. The comparison, the README table and the infographic then
pair it with the other side's latest record in `benchmarks/history.jsonl`, so
run both legs again when the official image or the serving parameters change.

`--optional` adds the opt-in legs after the pair: DSpark and DFlash2 drafters
instead of MTP, on both images ([spec-drafters.md](spec-drafters.md)).
`--also` takes more legs one by one: a config name runs on the Mjolnir image;
`official:<config>` runs it on the official image. Extra legs are saved to the
history (`bench compare --config <cfg>`, `--official-config <cfg>`). They never
replace the pair the README is built from.

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
official ÷ Mjolnir − 1. The README colors a gain green beyond the ±3% band,
grey inside it and red below it (as the kernel tables do).

**Both sides run MTP speculative decoding with 3 speculative tokens** (the
`speculative-config` of `NVFP4` and `NVFP4_MJOLNIR`; each history record stores
it as `spec`, and the README, charts and `bench compare` print it). Decode t/s
is therefore tokens *accepted* per second: it depends on how many of the 3
draft tokens the target accepts, which varies by prompt — single runs spread
10–15%, so a 6-run mean carries a 2–6% standard error. Decode differences
inside ~6% need `--repeat 2` or more to tell apart. The attention kernel is
also only a slice of a decode step here (16 of 64 layers are full attention;
the NVFP4 weights dominate the bytes read), so its share — and Mjolnir's
effect on decode — grows with context and concurrency.

The infographic picks **one request at the longest context both runs
measured** — the decode-bound case the kernel is built for — and the table
shows every cell.

## Fair-comparison rules

- Same model, same serving config family, same host, back to back.
- The two legs are separate server launches; first-launch JIT/caches are on
  persistent volumes, and llama-benchy discards warm-up runs.
- A single cell moving a few percent is noise; trust a gain that holds across
  contexts and concurrency, and re-run with `--repeat 2` before publishing.
