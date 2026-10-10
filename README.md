<p align="center">
  <img src="assets/header.png" alt="Mjolnir" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/mjolnir-kernel/"><img src="https://img.shields.io/pypi/v/mjolnir-kernel?label=mjolnir-kernel&color=eb6834" alt="mjolnir-kernel on PyPI"></a>
  <a href="https://pypi.org/project/mjolnir-cli/"><img src="https://img.shields.io/pypi/v/mjolnir-cli?label=mjolnir-cli&color=2a78d6" alt="mjolnir-cli on PyPI"></a>
  <a href="https://hub.docker.com/r/narandill/mjolnir-vllm"><img src="https://img.shields.io/docker/v/narandill/mjolnir-vllm?label=docker&sort=semver&color=1baf7a" alt="narandill/mjolnir-vllm on Docker Hub"></a>
  <img src="https://img.shields.io/badge/vLLM-0.30.0-black" alt="vLLM 0.30.0">
  <img src="https://img.shields.io/badge/Jetson-AGX%20Thor%20sm__110-76b900" alt="NVIDIA Jetson AGX Thor">
  <a href="https://github.com/thor-arena/mjolnir/actions/workflows/ci.yml"><img src="https://github.com/thor-arena/mjolnir/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
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

**Mjolnir is a paged fp8 decode-attention kernel for the NVIDIA Jetson AGX Thor (`sm_110`)**, written in CuTe DSL on Blackwell's `tcgen05` tensor cores, TMEM and TMA, and shipped as a vLLM attention backend:

```bash
vllm serve <model> --attention-backend MJOLNIR
```

Every generated token re-reads the whole KV cache, so on a 273 GB/s edge GPU **decode speed is a memory problem**. Stock vLLM 0.30.0 runs FlashAttention 2 on Thor, which has no fp8 KV path and reads twice the bytes; FlashInfer is the fastest stock option. Mjolnir is built for this exact chip: 20 SMs, 2 CTAs per SM, a 32 MiB L2, LPDDR5X.

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/opening-infographic.png">
  <img src="assets/opening-infographic.png" alt="Mjolnir image vs the official vLLM image — end-to-end decode and prefill" width="100%">
</picture>
</p>

<p align="center"><sub>End to end on a Jetson AGX Thor, one request at 8K context: decode <b>+4.0%</b>, prefill <b>+19.0%</b>, time to first token <b>4.30 s → 3.63 s</b>, inter-token latency <b>40.9 ms → 39.3 ms</b> vs the official vLLM image — <a href="#official-vllm-image-vs-mjolnir">all cells</a>.</sub></p>


> [!NOTE]
> The headline benchmark has not been recorded in this checkout yet. Run `mjolnir verify bench` on a Thor and `mjolnir plot --readme` to render the charts and numbers from the raw results.

## Quickstart

> [!NOTE]
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
  narandill/mjolnir-vllm:v0.30.0 <model> --kv-cache-dtype fp8 --attention-backend MJOLNIR
```

**Or bring your own vLLM 0.30.0**: `pip install mjolnir-kernel` registers the backend as `CUSTOM` through vLLM's plugin system (`--attention-backend CUSTOM`); [`thor-mjolnir-attention-backend.patch`](docker/vllm-thor/patches/thor-mjolnir-attention-backend.patch) adds the `MJOLNIR` name.

## Benchmarks

All kernel numbers are **ncu kernel time** (decode + split-KV merge), cold L2, base clocks, one sequence with 24 query / 4 KV heads, head dim 256, 128-token pages — the attention shape of Qwen3.8-27B — at **every context length from 128 to 256K tokens**. Legs:

| leg | what runs | KV cache |
|---|---|---|
| **Mjolnir · e2m3 KV** | this kernel, its fastest config (6-bit KV read directly by the tensor cores) | e2m3, head-major pages |
| **Mjolnir · fp8 KV** | this kernel, what `--attention-backend MJOLNIR` serves today | fp8 e4m3, head-major pages |
| **FlashInfer** | tensor-core decode as vLLM's FlashInfer backend runs it | fp8 e4m3, NHD pages |
| **FlashAttention 2** | `vllm_flash_attn` FA2 — stock vLLM 0.30.0's `sm_110` default | bf16 (FA2 has no fp8 KV) |

The box is shared with a live server, so ±7% between runs is noise; ratios inside that band are drawn gray. Methodology: [`docs/methodology/benchmarking.md`](docs/methodology/benchmarking.md).

> Pending: `mjolnir verify sweep`.

### Official vLLM image vs Mjolnir

The same model and serving config on the official `vllm/vllm-openai` image and on the Mjolnir image, measured back to back by `mjolnir bench e2e` (llama-benchy, gated clean windows). Each cell: official → **Mjolnir** (gain; positive = Mjolnir better, ✅ beyond ±7%).

| context | requests | decode (t/s, ↑) | prefill (t/s, ↑) | TTFT (ms, ↓) | ITL (ms, ↓) |
|--:|--:|--:|--:|--:|--:|
| no ctx | 1 | 22.5 → **25.7** (+14.3% ✅) | 2,839 → **3,522** (+24.0% ✅) | 874 → **729** (+19.9% ✅) | 44.4 → **38.9** (+14.3% ✅) |
| 4K | 1 | 24.3 → **24.5** (+0.9% 🟰) | 2,894 → **3,290** (+13.7% ✅) | 2,271 → **2,013** (+12.8% ✅) | 41.1 → **40.8** (+0.9% 🟰) |
| 8K | 1 | 24.4 → **25.4** (+4.0% 🟰) | 2,468 → **2,938** (+19.0% ✅) | 4,296 → **3,631** (+18.3% ✅) | 40.9 → **39.3** (+4.0% 🟰) |
| no ctx | 2 | 44.5 → **48.5** (+9.0% ✅) | 2,671 → **3,118** (+16.7% ✅) | 1,475 → **1,255** (+17.5% ✅) | — |
| 4K | 2 | 45.7 → **47.6** (+4.3% 🟰) | 2,779 → **3,124** (+12.4% ✅) | 4,360 → **3,875** (+12.5% ✅) | — |
| 8K | 2 | 44.5 → **46.1** (+3.8% 🟰) | 2,468 → **2,862** (+16.0% ✅) | 8,237 → **7,096** (+16.1% ✅) | — |
| no ctx | 4 | 88.5 → **84.5** (−4.6% 🟰) | 2,892 → **3,325** (+15.0% ✅) | 2,800 → **2,433** (+15.1% ✅) | — |
| 4K | 4 | 81.0 → **83.8** (+3.5% 🟰) | 2,736 → **3,104** (+13.4% ✅) | 8,946 → **7,882** (+13.5% ✅) | — |
| 8K | 4 | 68.2 → **70.7** (+3.6% 🟰) | 2,457 → **2,800** (+14.0% ✅) | 16,344 → **14,324** (+14.1% ✅) | — |

<sub>official: `vllm/vllm-openai:v0.30.0-ubuntu2404 · NVFP4 (Native FlashInfer)` · Mjolnir: `narandill/mjolnir-vllm:v14 · NVFP4 (Mjolnir FlashInfer)` · TTFT = time to first token; ITL = inter-token latency (1000 / per-request decode rate) · from [`benchmarks/history.jsonl`](benchmarks/history.jsonl).</sub>

### Attention backends on the Mjolnir image

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/benchmarks/e2e-decode-dark.png">
  <img src="assets/benchmarks/e2e-decode.png" alt="End-to-end decode throughput per attention backend" width="100%">
</picture>

**Decode tokens/s** (llama-benchy, gated clean windows; Δ vs FlashInfer):

| backend (image) | c1 · no ctx | c1 · 4K ctx | c1 · 8K ctx | c4 · no ctx | c4 · 4K ctx | c4 · 8K ctx |
|---|--:|--:|--:|--:|--:|--:|
| **FlashInfer** | 25.7 | 24.5 | 25.4 | 84.5 | 83.8 | 70.7 |
| **FlashInfer (stock vLLM image)** | 22.5 $\color{#dc2626}{\text{−12.5\%}}$ | 24.3 $\color{grey}{\text{−0.9\%}}$ | 24.4 $\color{grey}{\text{−3.9\%}}$ | 88.5 $\color{grey}{\text{+4.8\%}}$ | 81.0 $\color{grey}{\text{−3.4\%}}$ | 68.2 $\color{grey}{\text{−3.5\%}}$ |

**Prompt tokens/s:**

| backend (image) | c1 · no ctx | c1 · 4K ctx | c1 · 8K ctx | c4 · no ctx | c4 · 4K ctx | c4 · 8K ctx |
|---|--:|--:|--:|--:|--:|--:|
| **FlashInfer** | 3,522 | 3,290 | 2,938 | 3,325 | 3,104 | 2,800 |
| **FlashInfer (stock vLLM image)** | 2,839 $\color{#dc2626}{\text{−19.4\%}}$ | 2,894 $\color{#dc2626}{\text{−12.0\%}}$ | 2,468 $\color{#dc2626}{\text{−16.0\%}}$ | 2,892 $\color{#dc2626}{\text{−13.0\%}}$ | 2,736 $\color{#dc2626}{\text{−11.9\%}}$ | 2,457 $\color{#dc2626}{\text{−12.3\%}}$ |

<sub>From [`benchmarks/history.jsonl`](benchmarks/history.jsonl) (`mjolnir bench ab`). Prefill runs on FlashInfer under the Mjolnir backend too, so prompt processing should match FlashInfer.</sub>

## How the kernel works

A decode step is one query token per sequence against the whole KV cache: at 64K tokens that is 128 MiB of K and V per layer, read once. The design goal is to keep the LPDDR5X busy and everything else out of the way.

| | design choice | why on Thor |
|---|---|---|
| ⚙️ | **Warp-specialized tcgen05 pipeline** — a TMA warp streams K/V pages through an SMEM ring, an MMA warp runs `S = QKᵀ` and `O += PV` on the tensor cores into TMEM, softmax warpgroups read S from TMEM and write P | the tensor cores and TMEM keep scores off the register file; the TMA engine walks the page table so the SMs never issue loads |
| 📐 | **Occupancy-pinned split-KV** — every step fills exactly 2 CTAs × 20 SMs; long sequences split into chunks, a small reduction kernel merges them | one full wave at every L; a fixed padded grid replays under full CUDA graphs |
| 🧊 | **fp8 everything** — Q quantized per (sequence, KV head) on the fly, fp8 KV, P stored as e4m3 | one MMA kind, no conversion warps, half the KV bytes of bf16 |
| 🗂️ | **Head-major (HND) pages** — each KV head's page is one contiguous block | longer DRAM bursts per TMA box: [hndsweep](docs/kernel/hndsweep.md) |
| 🪶 | **6-bit KV (e2m3) read directly by the MMA** (`kind::f8f6f4`, TMA unpacks to byte containers) | 25% fewer bytes at near-fp8 accuracy: [kvfmt](docs/kernel/kvfmt.md), [quant](docs/kernel/quant.md) |
| 🚿 | **L2 evict-first on K/V loads**, a 3-stage ring for fp6 | KV is streamed once; keep L2 for the rest of the model: [ring](docs/kernel/ring.md), [ab](docs/kernel/ab.md) |

The full walkthrough — one chapter per experiment, each with its chart, data table and the command that reproduces it — is the **[kernel case study](docs/kernel/README.md)**:

| chapter | question it answers |
|---|---|
| [Correctness matrix](docs/kernel/correctness.md) *(pending)* | kernel vs the bit-faithful emulator: every layout, format, page size, q_len, Q dtype |
| [Headline L-sweep](docs/kernel/sweep.md) *(pending)* | Mjolnir vs FlashInfer vs FlashAttention 2, ncu kernel time at every L |
| [Head-major pages](docs/kernel/hndsweep.md) *(pending)* | FlashInfer on NHD and HND vs Mjolnir on HND — is HND a kernel win or a layout win? |
| [KV layouts and formats](docs/kernel/kvfmt.md) *(pending)* | fp8 / fp6 / fp4 / nvfp4 KV on NHD / interleaved / HND pages (ncu + wall clock) |
| [KV format accuracy](docs/kernel/quant.md) *(pending)* | relative L2 error of each KV format vs unquantized KV, at every L |
| [Full profiles per KV format](docs/kernel/fp6prof.md) *(pending)* | full ncu profiles: L2 sectors vs requests vs time, warp stalls, per KV format |
| [KV ring depth](docs/kernel/ring.md) *(pending)* | TMA→MMA ring depth (--kv_stages) sweep at every L |
| [L2 prefetch](docs/kernel/prefetch.md) *(pending)* | TMA L2 prefetch distance (--l2_prefetch) sweep |
| [Programmatic dependent launch](docs/kernel/pdl.md) *(pending)* | PDL on/off wall clock (decode overlaps the previous kernel; merge launches early) |
| [Flag A/B](docs/kernel/ab.md) *(pending)* | any harness flag sets, ncu + wall clock (default: evict-first x PDL) |


## What's in the image

[`narandill/mjolnir-vllm:v0.30.0`](https://hub.docker.com/r/narandill/mjolnir-vllm) = the official `vllm/vllm-openai:v0.30.0` image + the `mjolnir-kernel` package + a hand-adapted patch stack for Thor. Tags follow the upstream vLLM version.

| patch | what it does |
|---|---|
| **`thor-mjolnir-attention-backend`** | adds `MJOLNIR` to vLLM's attention-backend enum → `--attention-backend MJOLNIR` |
| [#50885](https://github.com/vllm-project/vllm/pull/50885) | FULL cudagraph mode for speculative decode on the FlashInfer-native backend |
| [#49652](https://github.com/vllm-project/vllm/pull/49652) | draft-decode cudagraphs under dynamic speculative decoding |
| [#54165](https://github.com/vllm-project/vllm/pull/54165) | Mamba/GDN `copy_kv_cache` restore under speculative decode |
| [#55390](https://github.com/vllm-project/vllm/pull/55390) | MTP `draft_spec_decode` annotation for the draft group |
| [#55519](https://github.com/vllm-project/vllm/pull/55519) | no false-positive KV-reuse warning on the GDN hybrid |
| Thor `sm_110` patches | GDN prefill enablement, FA4 hd256 fp8-KV + 1-CTA decode, fused draft-decode plan advance, DSpark non-causal draft cudagraph, draft-cudagraph shape gate |

Details: [`docker/vllm-thor/PATCHES.md`](docker/vllm-thor/PATCHES.md).

## The CLI

```bash
mjolnir                                 # help + next steps
mjolnir serve [up|down|status|logs]     # the vLLM container (Mjolnir image)
mjolnir model | image                   # pick the model config / image
mjolnir verify <stage>                  # correctness + ncu benchmarks -> benchmarks/raw/verify
mjolnir verify bench | all              # every benchmark stage (all = + correctness)
mjolnir verify show | list              # re-print saved results
mjolnir bench e2e | compare            # official vLLM image vs Mjolnir: decode, prefill, TTFT, ITL
mjolnir bench perf | ab                 # end-to-end serving benchmarks -> history.jsonl
mjolnir plot --readme [--site DIR]      # charts + README + docs (+ interactive site) from raw data
mjolnir autoresearch init|run|ledger    # the autonomous kernel-research loop
mjolnir gate | hw | litellm | perses    # clean-window gate, host setup, side stacks
```

Every `verify` stage defaults to L = 128 4K 8K 16K 32K 64K 128K 256K; `--ls`, `--iters`, `--image`, `--dry-run` work everywhere.

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

[MIT](LICENSE)

<p align="center"><sub>⚡ Mjölnir — built on and for the edge. Generated from <code>README.md.j2</code> by <code>mjolnir plot --readme</code>; edit the template, not this file.</sub></p>
