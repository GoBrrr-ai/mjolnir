<p align="center">
  <img src="assets/hero.png" alt="Mjolnir — Jetson AGX Thor vLLM serving CLI with dedicated paged fp8 decode-attention kernel" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/mjolnir-kernel/"><img src="https://img.shields.io/pypi/v/mjolnir-kernel?label=mjolnir-kernel&color=eb6834" alt="mjolnir-kernel on PyPI"></a>
  <a href="https://pypi.org/project/mjolnir-cli/"><img src="https://img.shields.io/pypi/v/mjolnir-cli?label=mjolnir-cli&color=2a78d6" alt="mjolnir-cli on PyPI"></a>
  <a href="https://hub.docker.com/r/narandill/mjolnir-vllm"><img src="https://img.shields.io/docker/v/narandill/mjolnir-vllm?label=docker&sort=semver&color=1baf7a" alt="narandill/mjolnir-vllm on Docker Hub"></a>
  <img src="https://img.shields.io/badge/vLLM-0.31.0-blue" alt="vLLM 0.31.0">
  <img src="https://img.shields.io/badge/Jetson-AGX%20Thor-76b900" alt="NVIDIA Jetson AGX Thor">
  <a href="https://github.com/thor-arena/mjolnir/actions/workflows/ci.yml"><img src="https://github.com/thor-arena/mjolnir/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0">
</p>

<p align="center">
  <b><em>Mjölnir</em></b> <sup>(MYOL-nir)</sup> — <em>a decode-attention kernel built for exactly one GPU: the <b>Jetson AGX Thor</b>.</em>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="https://thor-arena.github.io/mjolnir/">Interactive charts</a> •
  <a href="#how-the-kernel-works">How it works</a> •
  <a href="docs/kernel/README.md">Case study</a> •
  <a href="#contributing">Contributing</a>
</p>

## What is ⚡ Mjolnir?

> *Mjölnir* is Old Norse for "Thor's hammer" — pronounced **"MYOL-nir"**.

**Mjolnir is a paged fp8 decode-attention kernel for the NVIDIA Jetson AGX Thor (`sm_110`)**, written in CuTe DSL on Blackwell's `tcgen05` tensor cores, `TMEM` and `TMA`, and shipped as a vLLM attention backend:

```bash
vllm serve <model> --attention-backend MJOLNIR
```

Every generated token re-reads the whole KV cache, so on a 273 GB/s edge GPU **decode speed is a memory problem**. Stock vLLM 0.31.0 runs FlashAttention 2 on Thor, which has no fp8 KV path and reads twice the bytes; FlashInfer is the fastest stock option. Mjolnir is built for this exact chip: 20 SMs, 2 CTAs per SM, a 32 MiB L2, LPDDR5X.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/opening-infographic.png">
  <img src="assets/opening-infographic.png" alt="Mjolnir image vs the official vLLM image — end-to-end decode and prefill" width="100%">
</picture>
</p>

> <sub>End to end on a Jetson AGX Thor (MTP, 3 speculative tokens), one request at 8K context: decode <b>+50.8%</b>, prefill <b>+67.1%</b>, time to first token <b>4.13 s → 3.02 s</b>, inter-token latency <b>44.4 ms → 29.4 ms</b> vs the official vLLM image — <a href="#official-vllm-image-vs-mjolnir">all cells</a>.</sub>

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmarks/kernel-speedup-dark.png">
  <img src="assets/benchmarks/kernel-speedup.png" alt="Mjolnir decode speedup over FlashInfer at every context length" width="100%">
</picture>
</p>

| decode kernel, one step (ncu, cold L2) | result |
|---|---|
| **vs FlashInfer**, geomean over L = 128 4K 8K 16K 32K 64K 128K 256K | **1.28× faster** (from 1.09× at 16K to 1.71× at 128) |
| **vs FlashAttention 2** (stock vLLM's sm_110 default) | **2.01× faster** |
| faster than FlashInfer beyond the ±3% noise band | **8 of 8** context lengths |
| at the longest context (256K tokens) | **1.37×** vs FlashInfer |
| one decode step at 64K tokens | **602.3 µs** (FlashInfer 814.9 µs, FlashAttention 2 1.28 ms) |
| fp8 KV (what `--attention-backend MJOLNIR` serves today) | **1.17×** vs FlashInfer, geomean |
| correctness vs a bit-faithful emulator | **90 / 90** checks pass |


> [!NOTE]
> <sub>Every number above is computed from the raw results of `mjolnir verify sweep` (run `20261011-113515`, [`benchmarks/raw/verify/`](benchmarks/raw/verify/)) when this README is rendered — no hand-typed numbers. Hover the [interactive version](https://thor-arena.github.io/mjolnir/) for any point.</sub>

## Quickstart

> [!IMPORTANT]
> Requirements: Jetson AGX Thor (JetPack 7.x, `sm_110a`), Docker + NVIDIA Container Toolkit, [`uv`](https://docs.astral.sh/uv/).

```bash
# the CLI (the package is mjolnir-cli; the command is `mjolnir`)
uv tool install mjolnir-cli

# one-time: Thor host setup for stable numbers (clocks, power mode, fans, swap)
mjolnir hw setup

# serve: Qwen3.8-27B NVFP4 on the Mjolnir image, --attention-backend MJOLNIR, port 6001
mjolnir serve up
```

```bash
curl -s http://localhost:6001/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "Qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}'
```

**Use the image directly** (any model, any vLLM flags):

```bash
docker run --rm -it --gpus all --ipc host -p 8000:8000 -e VLLM_KV_CACHE_LAYOUT=HND \
  narandill/mjolnir-vllm:v0.31.0 <model> --kv-cache-dtype fp8 --attention-backend MJOLNIR
```

**Or bring your own vLLM 0.31.0**: `pip install mjolnir-kernel` registers the backend as `CUSTOM` through vLLM's plugin system (`--attention-backend CUSTOM`); [`thor-mjolnir-attention-backend.patch`](docker/vllm-thor/patches/thor-mjolnir-attention-backend.patch) adds the `MJOLNIR` name.

## Benchmarks

All kernel numbers are **ncu kernel time** (decode + split-KV merge), cold L2, base clocks, one sequence with 24 query / 4 KV heads, head dim 256, 128-token pages — the attention shape of Qwen3.8-27B — at **every context length from 128 to 256K tokens**. Legs:

| leg | what runs | KV cache |
|---|---|---|
| **Mjolnir · e2m3 KV** | this kernel, its fastest config (6-bit KV read directly by the tensor cores) | e2m3, head-major pages |
| **Mjolnir · fp8 KV** | this kernel, what `--attention-backend MJOLNIR` serves today | fp8 e4m3, head-major pages |
| **FlashInfer** | tensor-core decode as vLLM's FlashInfer backend runs it | fp8 e4m3, NHD pages |
| **FlashAttention 2** | `vllm_flash_attn` FA2 — stock vLLM 0.31.0's `sm_110` default | bf16 (FA2 has no fp8 KV) |

Measured on a dedicated Thor: a same-run ratio repeats within ±3% between runs, so ratios inside that band are drawn gray (L=16K is the exception — it is bimodal across runs; see [`docs/methodology/benchmarking.md`](docs/methodology/benchmarking.md)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmarks/kernel-latency-dark.png">
  <img src="assets/benchmarks/kernel-latency.png" alt="Decode latency per step vs context length" width="100%">
</picture>

| L | FlashInfer | FA2 | Mjolnir fp8 | Mjolnir e2m3 | fp8 vs FI | e2m3 vs FI | fp8 vs FA2 | e2m3 vs FA2 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **128** | 22.1 µs | 20.3 µs | 13.8 µs | **12.9 µs** | $\color{#16a34a}{\mathbf{1.60\times}}$ | $\color{#16a34a}{\mathbf{1.71\times}}$ | $\color{#16a34a}{\mathbf{1.47\times}}$ | $\color{#16a34a}{\mathbf{1.57\times}}$ |
| **4K** | 60.9 µs | 95.2 µs | 56.4 µs | **51.9 µs** | $\color{#16a34a}{\mathbf{1.08\times}}$ | $\color{#16a34a}{\mathbf{1.17\times}}$ | $\color{#16a34a}{\mathbf{1.69\times}}$ | $\color{#16a34a}{\mathbf{1.83\times}}$ |
| **8K** | 96.8 µs | 180.7 µs | 87.2 µs | **84.6 µs** | $\color{#16a34a}{\mathbf{1.11\times}}$ | $\color{#16a34a}{\mathbf{1.14\times}}$ | $\color{#16a34a}{\mathbf{2.07\times}}$ | $\color{#16a34a}{\mathbf{2.13\times}}$ |
| **16K** | 189.3 µs | 353.3 µs | **169.3 µs** | 173.2 µs | $\color{#16a34a}{\mathbf{1.12\times}}$ | $\color{#16a34a}{\mathbf{1.09\times}}$ | $\color{#16a34a}{\mathbf{2.09\times}}$ | $\color{#16a34a}{\mathbf{2.04\times}}$ |
| **32K** | 384.0 µs | 668.9 µs | 362.6 µs | **334.6 µs** | $\color{#16a34a}{\mathbf{1.06\times}}$ | $\color{#16a34a}{\mathbf{1.15\times}}$ | $\color{#16a34a}{\mathbf{1.84\times}}$ | $\color{#16a34a}{\mathbf{2.00\times}}$ |
| **64K** | 814.9 µs | 1.28 ms | 693.2 µs | **602.3 µs** | $\color{#16a34a}{\mathbf{1.18\times}}$ | $\color{#16a34a}{\mathbf{1.35\times}}$ | $\color{#16a34a}{\mathbf{1.85\times}}$ | $\color{#16a34a}{\mathbf{2.13\times}}$ |
| **128K** | 1.48 ms | 2.42 ms | 1.30 ms | **1.10 ms** | $\color{#16a34a}{\mathbf{1.14\times}}$ | $\color{#16a34a}{\mathbf{1.35\times}}$ | $\color{#16a34a}{\mathbf{1.87\times}}$ | $\color{#16a34a}{\mathbf{2.21\times}}$ |
| **256K** | 2.83 ms | 4.63 ms | 2.45 ms | **2.07 ms** | $\color{#16a34a}{\mathbf{1.16\times}}$ | $\color{#16a34a}{\mathbf{1.37\times}}$ | $\color{#16a34a}{\mathbf{1.89\times}}$ | $\color{#16a34a}{\mathbf{2.24\times}}$ |
| geomean |  |  |  |  | $\color{#16a34a}{\mathbf{1.17\times}}$ | $\color{#16a34a}{\mathbf{1.28\times}}$ | $\color{#16a34a}{\mathbf{1.84\times}}$ | $\color{#16a34a}{\mathbf{2.01\times}}$ |

> <sub>Bold = fastest at that `L`. Ratio columns: baseline time ÷ this kernel's time (green = faster beyond noise).</sub>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmarks/kernel-bandwidth-dark.png">
  <img src="assets/benchmarks/kernel-bandwidth.png" alt="Effective KV bandwidth vs the LPDDR5X peak" width="100%">
</picture>

### Official vLLM image vs Mjolnir

Native vLLM 0.31.0 — FlashInfer, on the **official** `vllm/vllm-openai` image, untouched — against the Mjolnir backend on the Mjolnir image: same model, same serving config — **speculative decoding on both sides: MTP, 3 speculative tokens** — measured back to back by `mjolnir bench e2e` (llama-benchy, gated clean windows). Native backends are never measured on the Mjolnir image: it patches FlashInfer paths too.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmarks/e2e-decode-dark.png">
  <img src="assets/benchmarks/e2e-decode.png" alt="End-to-end decode throughput: official vLLM vs Mjolnir" width="100%">
</picture>

Each cell: official → **Mjolnir** (gain; positive = Mjolnir better: green = better beyond the ±3% noise band, grey = within it, red = worse beyond it). With speculative decoding the decode rate also depends on how many draft tokens are accepted, which varies by prompt, so single decode cells move by more than the band from run to run.

| context | requests | decode (t/s, ↑) | prefill (t/s, ↑) | TTFT (ms, ↓) | ITL (ms, ↓) |
|--:|--:|--:|--:|--:|--:|
| no ctx | 1 | 22.5 → **35.2** $\color{#16a34a}{\text{+56.7\\%}}$ | 2,929 → **3,983** $\color{#16a34a}{\text{+36.0\\%}}$ | 833 → **606** $\color{#16a34a}{\text{+37.6\\%}}$ | 45.0 → **28.6** $\color{#16a34a}{\text{+57.3\\%}}$ |
| 4K | 1 | 23.1 → **33.6** $\color{#16a34a}{\text{+45.7\\%}}$ | 2,820 → **6,570** $\color{#16a34a}{\text{+133.0\\%}}$ | 2,314 → **1,666** $\color{#16a34a}{\text{+38.9\\%}}$ | 43.5 → **30.1** $\color{#16a34a}{\text{+44.6\\%}}$ |
| 8K | 1 | 22.8 → **34.3** $\color{#16a34a}{\text{+50.8\\%}}$ | 2,562 → **4,281** $\color{#16a34a}{\text{+67.1\\%}}$ | 4,132 → **3,023** $\color{#16a34a}{\text{+36.7\\%}}$ | 44.4 → **29.4** $\color{#16a34a}{\text{+51.1\\%}}$ |
| 16K | 1 | 23.2 → *not measured* | 2,205 → *not measured* | 8,493 → *not measured* | 43.3 → *not measured* |
| 32K | 1 | 23.2 → *not measured* | 1,762 → *not measured* | 19,894 → *not measured* | 43.2 → *not measured* |
| 64K | 1 | 22.6 → *not measured* | 1,255 → *not measured* | 54,008 → *not measured* | 44.3 → *not measured* |
| 128K | 1 | 20.5 → *not measured* | 791 → *not measured* | 168,458 → *not measured* | 49.2 → *not measured* |
| no ctx | 2 | 46.5 → **63.3** $\color{#16a34a}{\text{+36.2\\%}}$ | 2,744 → **3,588** $\color{#16a34a}{\text{+30.8\\%}}$ | 1,433 → **1,000** $\color{#16a34a}{\text{+43.2\\%}}$ | 40.5 → **29.6** $\color{#16a34a}{\text{+36.7\\%}}$ |
| 4K | 2 | 43.6 → **63.5** $\color{#16a34a}{\text{+45.7\\%}}$ | 2,728 → **3,590** $\color{#16a34a}{\text{+31.6\\%}}$ | 4,444 → **3,334** $\color{#16a34a}{\text{+33.3\\%}}$ | 43.1 → **29.7** $\color{#16a34a}{\text{+44.9\\%}}$ |
| 8K | 2 | 45.6 → **57.6** $\color{#16a34a}{\text{+26.4\\%}}$ | 2,515 → **3,461** $\color{#16a34a}{\text{+37.6\\%}}$ | 8,081 → **5,758** $\color{#16a34a}{\text{+40.3\\%}}$ | 42.1 → **32.1** $\color{#16a34a}{\text{+31.2\\%}}$ |
| 16K | 2 | 45.6 → *not measured* | 2,195 → *not measured* | 16,734 → *not measured* | 42.3 → *not measured* |
| 32K | 2 | 36.7 → *not measured* | 1,752 → *not measured* | 39,043 → *not measured* | 47.3 → *not measured* |
| no ctx | 4 | 85.0 → **104.5** $\color{#16a34a}{\text{+22.9\\%}}$ | 2,912 → **3,423** $\color{#16a34a}{\text{+17.6\\%}}$ | 2,781 → **2,225** $\color{#16a34a}{\text{+25.0\\%}}$ | 44.1 → **31.6** $\color{#16a34a}{\text{+39.7\\%}}$ |
| 4K | 4 | 78.5 → **104.2** $\color{#16a34a}{\text{+32.8\\%}}$ | 2,722 → **3,458** $\color{#16a34a}{\text{+27.0\\%}}$ | 8,993 → **6,937** $\color{#16a34a}{\text{+29.6\\%}}$ | 46.8 → **32.9** $\color{#16a34a}{\text{+42.1\\%}}$ |
| 8K | 4 | 68.1 → **68.0** $\color{grey}{\text{−0.1\\%}}$ | 2,491 → **3,375** $\color{#16a34a}{\text{+35.5\\%}}$ | 16,121 → **11,218** $\color{#16a34a}{\text{+43.7\\%}}$ | 49.1 → **39.2** $\color{#16a34a}{\text{+25.2\\%}}$ |
| 16K | 4 | 42.0 → *not measured* | 2,172 → *not measured* | 32,309 → *not measured* | 60.0 → *not measured* |
| 32K | 4 | 11.6 → *not measured* | 1,738 → *not measured* | 65,787 → *not measured* | 161.8 → *not measured* |

> [!NOTE]
> Incomplete: the Mjolnir side has no run at 16K, 32K, 64K, 128K context yet (`mjolnir bench e2e --only mjolnir --depths 16384,32768,65536,131072`).

> <sub>official: `vllm/vllm-openai:v0.31.0-ubuntu2404 · NVFP4 (vLLM 0.31.0 (official image))` · Mjolnir: `narandill/mjolnir-vllm:v0.31.0 · NVFP4_MJOLNIR_E2M3_LMH8_VT64K (Mjolnir fp6 KV + fp8 lm_head + 64K draft vocab)` · TTFT = time to first token; ITL = inter-token latency (1000 / per-request decode rate) · from [`benchmarks/history.jsonl`](benchmarks/history.jsonl).</sub>

## How the kernel works

A decode step is one query token per sequence against the whole KV cache: at 64K tokens that is 128 MiB of K and V per layer, read once. The design goal is to keep the LPDDR5X busy and everything else out of the way.

| | design choice | why on Thor |
|---|---|---|
| ⚙️ | **Warp-specialized `tcgen05` pipeline** — a TMA warp streams K/V pages through an SMEM ring, an MMA warp runs `S = QKᵀ` and `O += PV` on the tensor cores into TMEM, softmax warpgroups read S from TMEM and write P | the tensor cores and TMEM keep scores off the register file; the TMA engine walks the page table so the SMs never issue loads |
| 📐 | **Occupancy-pinned split-KV** — every step fills exactly 2 CTAs × 20 SMs; long sequences split into chunks, a small reduction kernel merges them | one full wave at every L; a fixed padded grid replays under full CUDA graphs |
| 🧊 | **fp8 everything** — Q quantized per (sequence, KV head) on the fly, fp8 KV, P stored as e4m3 | one MMA kind, no conversion warps, half the KV bytes of bf16 |
| 🗂️ | **Head-major (HND) pages** — each KV head's page is one contiguous block | longer DRAM bursts per TMA box: [hndsweep](docs/kernel/hndsweep.md) |
| 🪶 | **6-bit KV (e2m3) read directly by the MMA** (`kind::f8f6f4`, TMA unpacks to byte containers) | 25% fewer bytes at near-fp8 accuracy: [kvfmt](docs/kernel/kvfmt.md), [quant](docs/kernel/quant.md), [capacity](docs/kernel/kv-cache-size.md) |
| 🚿 | **L2 evict-first on K/V loads**, a 2-stage ring for fp6 | KV is streamed once; keep L2 for the rest of the model: [ring](docs/kernel/ring.md), [ab](docs/kernel/ab.md) |

The full walkthrough — one chapter per experiment, each with its chart, data table and the command that reproduces it — is the **[kernel case study](docs/kernel/README.md)**:

| chapter | question it answers |
|---|---|
| [Correctness matrix](docs/kernel/correctness.md)  | kernel vs the bit-faithful emulator: every layout, format, page size, q_len, Q dtype |
| [Headline L-sweep](docs/kernel/sweep.md)  | Mjolnir vs FlashInfer vs FlashAttention 2, ncu kernel time at every L |
| [Head-major pages](docs/kernel/hndsweep.md)  | FlashInfer on NHD and HND vs Mjolnir on HND — is HND a kernel win or a layout win? |
| [KV layouts and formats](docs/kernel/kvfmt.md)  | fp8 / fp6 / fp4 / nvfp4 KV on NHD / interleaved / HND pages (ncu + wall clock) |
| [KV format accuracy](docs/kernel/quant.md)  | relative L2 error of each KV format vs unquantized KV, at every L |
| [KV cache capacity](docs/kernel/kv-cache-size.md)  | tokens of KV each config fits (vLLM's startup sizing), and the concurrency that buys per context |
| [Speculative decoding](docs/kernel/spec.md)  | MTP acceptance per draft position and decode t/s for 1-4 draft tokens, per config — the best draft count |
| [Full profiles per KV format](docs/kernel/fp6prof.md)  | full ncu profiles: L2 sectors vs requests vs time, warp stalls, per KV format |
| [KV ring depth](docs/kernel/ring.md)  | TMA→MMA ring depth (--kv_stages) sweep at every L |
| [L2 prefetch](docs/kernel/prefetch.md)  | TMA L2 prefetch distance (--l2_prefetch) sweep |
| [Programmatic dependent launch](docs/kernel/pdl.md)  | PDL on/off wall clock (decode overlaps the previous kernel; merge launches early) |
| [Flag A/B](docs/kernel/ab.md)  | any harness flag sets, ncu + wall clock (default: evict-first x PDL) |
| [Accuracy](docs/kernel/accuracy.md) *(pending)* | Does a serving config keep the model's answers? Tool calling (standard + Hard Mode), GSM8K / MMLU / IFEval on demand |


## What's in the image

[`narandill/mjolnir-vllm:v0.31.0`](https://hub.docker.com/r/narandill/mjolnir-vllm) = the official `vllm/vllm-openai:v0.31.0` image + the `mjolnir-kernel` package + a hand-adapted patch stack for Thor. Tags follow the upstream vLLM version.

| patch | what it does |
|---|---|
| **`thor-mjolnir-attention-backend`** | adds `MJOLNIR` to vLLM's attention-backend enum → `--attention-backend MJOLNIR` |
| [#50885](https://github.com/vllm-project/vllm/pull/50885) | FULL cudagraph mode for speculative decode on the FlashInfer-native backend |
| [#49652](https://github.com/vllm-project/vllm/pull/49652) | draft-decode cudagraphs under dynamic speculative decoding |
| [#54165](https://github.com/vllm-project/vllm/pull/54165) | Mamba/GDN `copy_kv_cache` restore under speculative decode |
| [#55519](https://github.com/vllm-project/vllm/pull/55519) | no false-positive KV-reuse warning on the GDN hybrid |
| Thor `sm_110` patches | GDN prefill enablement, FA4 hd256 fp8-KV + 1-CTA decode, fused draft-decode plan advance, DSpark non-causal draft cudagraph, draft-cudagraph shape gate, the `MJOLNIR` backend name + FA4 fallback default, the `fp6_e2m3` KV cache dtype |

> [!TIP]
> Details: [`docker/vllm-thor/PATCHES.md`](docker/vllm-thor/PATCHES.md).

## The CLI

```bash
mjolnir                                 # help + next steps
mjolnir serve [up|down|status|logs]     # the vLLM container (Mjolnir image)
mjolnir model | image                   # pick the model config / image
mjolnir verify <stage>                  # correctness + ncu benchmarks -> benchmarks/raw/verify
mjolnir verify bench | all              # every benchmark stage (all = + correctness)
mjolnir verify show | list              # re-print saved results
mjolnir bench e2e | compare             # official vLLM image vs Mjolnir: decode, prefill, TTFT, ITL
mjolnir bench perf | ab                 # end-to-end serving benchmarks -> history.jsonl
mjolnir plot --readme [--site DIR]      # charts + README + docs (+ interactive site) from raw data
mjolnir autoresearch init|run|ledger    # the autonomous kernel-research loop
mjolnir gate | hw | litellm | perses    # clean-window gate, host setup, side stacks
```

> [!TIP]
> Every `verify` stage defaults to `L = 128 4K 8K 16K 32K 64K 128K 256K`; `--ls`, `--iters`, `--image`, `--dry-run` work everywhere.

## Contributing

Mjolnir is meant to be a **community case study**: every claim is reproducible with one command, and every experiment — kept or discarded — leaves raw data behind.

- **Run the benchmarks on your Thor** — `mjolnir verify bench` writes JSON you can open a PR with; the README, docs and charts re-render from it.
- **Try a kernel idea** — change `src/mjolnir/mjolnir/t5_decode.py`, prove it with `mjolnir verify correctness` and an `mjolnir verify ab --variants ...` run, and send the raw result with the PR.
- **Let the agents try** — `mjolnir autoresearch run` drives an OpenCode research loop (web + arXiv research, implementation, debugging) where every experiment is logged on [the ledger](research/README.md).
- Guidelines: [CONTRIBUTING.md](.github/CONTRIBUTING.md) · [docs index](docs/README.md).

## Repository layout

```
├── src/
│   ├── mjolnir/                the kernel package (PyPI: mjolnir-kernel, import `mjolnir`)
│   │   └── mjolnir/            t5_decode.py (kernel + harness) · runner.py · backend.py
│   └── mjolnir-cli/            the CLI package (PyPI: mjolnir-cli, command `mjolnir`)
│       └── mjolnir_cli/        verify/ · charts.py · readme.py · autoresearch.py · serve/bench/...
├── docker/vllm-thor/           the image: Dockerfile, patches/, apply/verify gates, PATCHES.md
├── configs/                    model configs (NVFP4_MJOLNIR default, NVFP4 = FlashInfer, NVFP4_FA2)
├── benchmarks/raw/verify/      raw verify results (one JSON per stage run) — the charts' only input
├── assets/benchmarks/          generated charts (light + dark)
├── docs/kernel/                the kernel case study (rendered from *.md.j2 + raw data)
├── research/                   the autoresearch program + ledger
└── .opencode/                  OpenCode agents (autoresearch, researchers, engineer, debugger)
```

## Security

The served API is unauthenticated; expose it deliberately. See [SECURITY.md](.github/SECURITY.md).

## License

Mjolnir is open source under the [Apache License 2.0](LICENSE) — the license of vLLM and FlashInfer, which it builds on — with one exception: the kernel, [`src/mjolnir/mjolnir/t5_decode.py`](src/mjolnir/mjolnir/t5_decode.py), is derived from an NVIDIA CUTLASS CuTe DSL example and stays under that example's **BSD-3-Clause** license, with NVIDIA's notice kept at the top of the file. Third-party code, the vLLM / FlashInfer / FlashAttention files the image patches, and what the [`narandill/mjolnir-vllm`](https://hub.docker.com/r/narandill/mjolnir-vllm) image redistributes are listed in [NOTICE](NOTICE).

<p align="center"><sub>⚡ Mjölnir — built on and for the edge.</sub></p>
<p align="center"><sub>Generated from <code>README.md.j2</code> by <code>mjolnir plot --readme</code>.</sub></p>
