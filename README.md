<p align="center">
  <img src="assets/hero.png" alt="Mjolnir — Jetson AGX Thor vLLM serving CLI with dedicated paged fp8 decode-attention kernel" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/mjolnir-kernel/"><img src="https://img.shields.io/pypi/v/mjolnir-kernel?label=mjolnir-kernel&color=eb6834" alt="mjolnir-kernel on PyPI"></a>
  <a href="https://pypi.org/project/mjolnir-cli/"><img src="https://img.shields.io/pypi/v/mjolnir-cli?label=mjolnir-cli&color=2a78d6" alt="mjolnir-cli on PyPI"></a>
  <a href="https://hub.docker.com/r/narandill/mjolnir-vllm"><img src="https://img.shields.io/docker/v/narandill/mjolnir-vllm?label=docker&sort=semver&color=1baf7a" alt="narandill/mjolnir-vllm on Docker Hub"></a>
  <img src="https://img.shields.io/badge/vLLM-0.31.0-black" alt="vLLM 0.31.0">
  <img src="https://img.shields.io/badge/Jetson-AGX%20Thor%20sm__110-76b900" alt="NVIDIA Jetson AGX Thor">
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

> <sub>End to end on a Jetson AGX Thor (MTP, 3 speculative tokens), one request at 32K context: decode <b>+0.4%</b>, prefill <b>+92.5%</b>, time to first token <b>19.96 s → 11.33 s</b>, inter-token latency <b>42.0 ms → 42.1 ms</b> vs the official vLLM image — <a href="#official-vllm-image-vs-mjolnir">all cells</a>.</sub>

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
| correctness vs a bit-faithful emulator | **89 / 89** checks pass |


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
| no ctx | 1 | 24.6 → **25.0** ($\color{grey}{\text{+1.7\%}}$) | — | 819 → **676** ($\color{#16a34a}{\text{+21.1\%}}$) | 40.9 → **40.4** ($\color{grey}{\text{+1.3\%}}$) |
| 4K | 1 | 25.6 → **26.1** ($\color{grey}{\text{+2.1\%}}$) | 2,852 → **7,825** ($\color{#16a34a}{\text{+174.3\%}}$) | 2,289 → **1,822** ($\color{#16a34a}{\text{+25.6\%}}$) | 39.4 → **38.8** ($\color{grey}{\text{+1.6\%}}$) |
| 8K | 1 | 22.9 → **24.0** ($\color{#16a34a}{\text{+4.6\%}}$) | 2,559 → **4,988** ($\color{#16a34a}{\text{+94.9\%}}$) | 4,136 → **3,087** ($\color{#16a34a}{\text{+34.0\%}}$) | 44.1 → **41.9** ($\color{#16a34a}{\text{+5.4\%}}$) |
| 16K | 1 | 23.5 → **24.6** ($\color{#16a34a}{\text{+4.7\%}}$) | 2,203 → **4,004** ($\color{#16a34a}{\text{+81.7\%}}$) | 8,501 → **5,637** ($\color{#16a34a}{\text{+50.8\%}}$) | 43.3 → **40.8** ($\color{#16a34a}{\text{+6.3\%}}$) |
| 32K | 1 | 23.9 → **23.9** ($\color{grey}{\text{+0.4\%}}$) | 1,756 → **3,382** ($\color{#16a34a}{\text{+92.5\%}}$) | 19,960 → **11,330** ($\color{#16a34a}{\text{+76.2\%}}$) | 42.0 → **42.1** ($\color{grey}{\text{−0.2\%}}$) |
| no ctx | 2 | 44.3 → **47.2** ($\color{#16a34a}{\text{+6.7\%}}$) | 2,724 → **3,386** ($\color{#16a34a}{\text{+24.3\%}}$) | 1,443 → **1,066** ($\color{#16a34a}{\text{+35.4\%}}$) | 42.7 → **40.0** ($\color{#16a34a}{\text{+6.7\%}}$) |
| 4K | 2 | 45.1 → **45.3** ($\color{grey}{\text{+0.6\%}}$) | 2,746 → **3,480** ($\color{#16a34a}{\text{+26.7\%}}$) | 4,415 → **3,438** ($\color{#16a34a}{\text{+28.4\%}}$) | 41.5 → **41.4** ($\color{grey}{\text{+0.3\%}}$) |
| 8K | 2 | 43.9 → **46.6** ($\color{#16a34a}{\text{+6.1\%}}$) | 2,504 → **3,390** ($\color{#16a34a}{\text{+35.4\%}}$) | 8,118 → **5,945** ($\color{#16a34a}{\text{+36.6\%}}$) | 43.1 → **40.4** ($\color{#16a34a}{\text{+6.7\%}}$) |
| 16K | 2 | 42.4 → **42.8** ($\color{grey}{\text{+0.9\%}}$) | 2,189 → **3,300** ($\color{#16a34a}{\text{+50.8\%}}$) | 16,781 → **10,919** ($\color{#16a34a}{\text{+53.7\%}}$) | 44.8 → **43.1** ($\color{#16a34a}{\text{+4.0\%}}$) |
| 32K | 2 | 33.7 → **36.9** ($\color{#16a34a}{\text{+9.5\%}}$) | 1,748 → **3,083** ($\color{#16a34a}{\text{+76.4\%}}$) | 39,137 → **21,899** ($\color{#16a34a}{\text{+78.7\%}}$) | 50.5 → **46.1** ($\color{#16a34a}{\text{+9.6\%}}$) |
| no ctx | 4 | 85.3 → **75.5** ($\color{#dc2626}{\text{−11.5\%}}$) | 2,927 → **3,333** ($\color{#16a34a}{\text{+13.9\%}}$) | 2,767 → **2,283** ($\color{#16a34a}{\text{+21.2\%}}$) | 42.6 → **44.7** ($\color{#dc2626}{\text{−4.6\%}}$) |
| 4K | 4 | 82.9 → **76.8** ($\color{#dc2626}{\text{−7.5\%}}$) | 2,717 → **3,397** ($\color{#16a34a}{\text{+25.0\%}}$) | 9,012 → **7,033** ($\color{#16a34a}{\text{+28.1\%}}$) | 44.7 → **46.2** ($\color{#dc2626}{\text{−3.3\%}}$) |
| 8K | 4 | 70.2 → **56.1** ($\color{#dc2626}{\text{−20.1\%}}$) | 2,478 → **3,349** ($\color{#16a34a}{\text{+35.2\%}}$) | 16,208 → **11,286** ($\color{#16a34a}{\text{+43.6\%}}$) | 47.3 → **53.4** ($\color{#dc2626}{\text{−11.4\%}}$) |
| 16K | 4 | 41.3 → **43.9** ($\color{#16a34a}{\text{+6.2\%}}$) | 2,163 → **3,215** ($\color{#16a34a}{\text{+48.7\%}}$) | 32,462 → **21,083** ($\color{#16a34a}{\text{+54.0\%}}$) | 60.2 → **59.8** ($\color{grey}{\text{+0.7\%}}$) |
| 32K | 4 | 11.5 → **17.4** ($\color{#16a34a}{\text{+51.4\%}}$) | 1,733 → **3,024** ($\color{#16a34a}{\text{+74.5\%}}$) | 65,974 → **36,736** ($\color{#16a34a}{\text{+79.6\%}}$) | 161.8 → **120.2** ($\color{#16a34a}{\text{+34.6\%}}$) |

> <sub>official: `vllm/vllm-openai:v0.31.0-ubuntu2404 · NVFP4 (vLLM 0.31.0 (official image))` · Mjolnir: `narandill/mjolnir-vllm:v0.31.0 · NVFP4_MJOLNIR_E2M3 (Mjolnir fp6 KV)` · TTFT = time to first token; ITL = inter-token latency (1000 / per-request decode rate) · from [`benchmarks/history.jsonl`](benchmarks/history.jsonl).</sub>

## How the kernel works

A decode step is one query token per sequence against the whole KV cache: at 64K tokens that is 128 MiB of K and V per layer, read once. The design goal is to keep the LPDDR5X busy and everything else out of the way.

| | design choice | why on Thor |
|---|---|---|
| ⚙️ | **Warp-specialized `tcgen05` pipeline** — a TMA warp streams K/V pages through an SMEM ring, an MMA warp runs `S = QKᵀ` and `O += PV` on the tensor cores into TMEM, softmax warpgroups read S from TMEM and write P | the tensor cores and TMEM keep scores off the register file; the TMA engine walks the page table so the SMs never issue loads |
| 📐 | **Occupancy-pinned split-KV** — every step fills exactly 2 CTAs × 20 SMs; long sequences split into chunks, a small reduction kernel merges them | one full wave at every L; a fixed padded grid replays under full CUDA graphs |
| 🧊 | **fp8 everything** — Q quantized per (sequence, KV head) on the fly, fp8 KV, P stored as e4m3 | one MMA kind, no conversion warps, half the KV bytes of bf16 |
| 🗂️ | **Head-major (HND) pages** — each KV head's page is one contiguous block | longer DRAM bursts per TMA box: [hndsweep](docs/kernel/hndsweep.md) |
| 🪶 | **6-bit KV (e2m3) read directly by the MMA** (`kind::f8f6f4`, TMA unpacks to byte containers) | 25% fewer bytes at near-fp8 accuracy: [kvfmt](docs/kernel/kvfmt.md), [quant](docs/kernel/quant.md) |
| 🚿 | **L2 evict-first on K/V loads**, a 2-stage ring for fp6 | KV is streamed once; keep L2 for the rest of the model: [ring](docs/kernel/ring.md), [ab](docs/kernel/ab.md) |

The full walkthrough — one chapter per experiment, each with its chart, data table and the command that reproduces it — is the **[kernel case study](docs/kernel/README.md)**:

| chapter | question it answers |
|---|---|
| [Correctness matrix](docs/kernel/correctness.md)  | kernel vs the bit-faithful emulator: every layout, format, page size, q_len, Q dtype |
| [Headline L-sweep](docs/kernel/sweep.md)  | Mjolnir vs FlashInfer vs FlashAttention 2, ncu kernel time at every L |
| [Head-major pages](docs/kernel/hndsweep.md)  | FlashInfer on NHD and HND vs Mjolnir on HND — is HND a kernel win or a layout win? |
| [KV layouts and formats](docs/kernel/kvfmt.md)  | fp8 / fp6 / fp4 / nvfp4 KV on NHD / interleaved / HND pages (ncu + wall clock) |
| [KV format accuracy](docs/kernel/quant.md)  | relative L2 error of each KV format vs unquantized KV, at every L |
| [Full profiles per KV format](docs/kernel/fp6prof.md)  | full ncu profiles: L2 sectors vs requests vs time, warp stalls, per KV format |
| [KV ring depth](docs/kernel/ring.md)  | TMA→MMA ring depth (--kv_stages) sweep at every L |
| [L2 prefetch](docs/kernel/prefetch.md)  | TMA L2 prefetch distance (--l2_prefetch) sweep |
| [Programmatic dependent launch](docs/kernel/pdl.md)  | PDL on/off wall clock (decode overlaps the previous kernel; merge launches early) |
| [Flag A/B](docs/kernel/ab.md)  | any harness flag sets, ncu + wall clock (default: evict-first x PDL) |


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
