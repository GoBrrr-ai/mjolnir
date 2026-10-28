# Jetson Orin family board matrix — decision-grade hardware research

**Date:** 2026-10-22 · **Requester:** kernel/benchmarks repo (`mjolnir`, Orin sm_87 family restoration under `src/mjolnir/mjolnir/arch/orin/`)
**Scope:** official NVIDIA sources only (jetson developer pages, module data sheets, JetPack docs/blogs) + deviceQuery cross-checks (marked unofficial). Two GitHub-thread claims verified/refuted.

---

## Key Findings

### 0. The die: one GA10B (Tegra234) for the whole family

- The Orin family is **seven modules with identical architecture**; the AGX Orin Developer Kit "can also emulate any of the Jetson Orin modules" — i.e. one die (GA10B / Tegra234), cut down by fusing TPCs/SMs, CPU clusters, DLA and memory channels per module ([family page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)).
- Official die facts ([AGX Orin Technical Brief, NVIDIA PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)): "up to 8 Texture Processing Clusters (TPCs), **up to 16 Streaming Multiprocessors (SMs)**, 192 KB of L1-cache per SM, and **4 MB of L2 Cache**. There are **128 CUDA cores per SM** … and **four 3rd Generation Tensor cores per SM**." Also: "**Jetson AGX Orin 32GB will have 2x 4 Core Clusters, and 7 TPCs with 14 SMs**" (⇒ AGX 64GB = 8 TPCs / 16 SMs).
- Per-SM tensor throughput (derived, cross-checked against NVIDIA's own tables): fp16 dense = 2048 FLOP/clk/SM (16 SM × 2048 × 1.3 GHz = 42.6 ≈ official 43 dense FP16 TFLOPS ✓), INT8 dense = 2× fp16 = 4096 FLOP/clk/SM (16 × 4096 × 1.3 = 85.2 ≈ official 85 dense INT8 TOPS ✓), sparse = 2× dense.

### 1. The board matrix (module-level; dev kits use the same modules)

| Board | CUDA cores | **SMs** | Tensor Cores | fp16 dense/sparse TFLOPS (GPU) | INT8 dense/sparse TOPS (GPU) | Memory | Mem BW | GPU max clock | Power (module) | L2 (observed) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Jetson AGX Orin 64GB** (= Developer Kit) | 2048 | **16** | 64 | 43 / 85 | 85 / 170 | 64 GB 256-bit LPDDR5 | **204.8 GB/s** | **1.3 GHz** | 15–60 W (MAXN=60) | 4 MB (4,194,304 B) |
| **Jetson AGX Orin Industrial** | 2048 | **16** | 64 | 39 / 78 | 78 / 156 | 64 GB 256-bit LPDDR5 **+ ECC** | 204.8 GB/s | 1.2 GHz | 15–75 W | (same die; not directly observed) |
| **Jetson AGX Orin 32GB** | 1792 | **14** | 56 | 27 / 54 std; ≈37/75 Super (derived) | 54 / 108 std | 32 GB 256-bit LPDDR5 | 204.8 GB/s | **930 MHz std → 1.3 GHz Super** (JP 7.2) | 15–40 W std; up to 60 W Super | 4 MB (4,194,304 B) |
| **Jetson Orin NX 16GB** | 1024 | **8** | 32 | 15 / 30 std; 19 / 38 Super | 30 / 60 std; 38 / 77 Super | 16 GB 128-bit LPDDR5 | **102.4 GB/s** | 918 MHz → 1173 MHz (MAXN SUPER) | 10/15/25 W + 40 W/MAXN SUPER | 2 MB (2,097,152 B) |
| **Jetson Orin NX 8GB** | 1024 | **8** | 32 | 13 / 25 std; 19 / 38 Super | 25 / 50 std; 38 / 77 Super | 8 GB 128-bit LPDDR5 | 102.4 GB/s | 765 MHz → 1173 MHz (MAXN SUPER) | 10/15/20 W + 40 W/MAXN SUPER | not directly observed (same 8-SM GPU as NX16 ⇒ 2 MB expected) |
| **Jetson Orin Nano 8GB** (= Nano Super Dev Kit) | 1024 | **8** | 32 | 10 / 20 std; 17 / 33 Super | 20 / 40 std; 33 / 67 Super | 8 GB **128-bit** LPDDR5 | 68 GB/s std → **102 GB/s** Super | 625 MHz → 1020 MHz (MAXN SUPER) | 7/15 W → 15/25 W/MAXN SUPER | 2 MB (JP 5/6 deviceQuery); 4 MB reported once on JP 7.2.1 |
| **Jetson Orin Nano 4GB** | 512 | **4** | 16 | 5 / 10 std; 8.5 / 17 Super | 10 / 20 std; 17 / 34 Super | 4 GB 64-bit LPDDR5 | 34 → 51 GB/s | 625 → 1020 MHz | 7/10 W → 10/25 W/MAXN SUPER | not observed (4 SMs ⇒ ~1 MB expected) |
| **DGX Orin / DGX Orin Station** | 2048 | **16** | 64 | 43 / 85 | 85 / 170 | 64 GB LPDDR5 (module) | 204.8 GB/s | 1.3 GHz | (server/workstation) | 4 MB |

Confidence per column family:
- **Cores/TCs/GPU/clocks/memory/BW/power for AGX + NX**: HIGH — official family page table ([colspan-parsed](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)), official [Orin NX data sheet DS-10712-001 v1.7 (Feb 2026)](https://developer.nvidia.com/downloads/jetson-orin-nx-module-series-data-sheet), official [JetPack 6.2 Super-Mode blog Table 2](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/), [JetPack 7.2 blog](https://developer.nvidia.com/blog/deploy-agentic-ready-ai-at-the-edge-with-memory-efficiency-in-nvidia-jetpack-7-2/), cross-checked with deviceQuery outputs (below).
- **SM counts**: HIGH for AGX 64GB (16), AGX 32GB (14) — official tech brief statement + deviceQuery; HIGH for Nano 8GB (8) and NX 16GB (8) — deviceQuery + cores/128; MEDIUM for Nano 4GB (4) and NX 8GB (8) — cores/128 + official blog table (no deviceQuery seen for those two).
- **L2 per board**: MEDIUM — deviceQuery observations, no official per-board statement except the 4 MB die figure.
- **AGX 32GB standard power 15–40 W**: MEDIUM — distributor docs ([TwoWin](https://twowintech.com/product/nvidia-jetson-agx-orin-32gb-64gb-200-275-tops-developer-kit/), Advantech) + JP 7.2 blog "up to 60W" for Super; the family page lists 15–60 W (Super-era).
- **DGX Orin row**: LOW — official pages retired (see Gaps); GPU identity follows from it being an AGX Orin 64GB-module system.

Notes on the sparse/dense trap: NVIDIA's headline "AI Performance" (275/248/241/157/117/67/34 TOPS) is **sparse INT8, GPU+DLA combined** (e.g. AGX 64GB: GPU 170 sparse + 2×DLA 105 = 275; NX 16GB: 77+80 = 157; AGX 32GB std: 108+92 = 200; AGX 32GB Super: ≈149+92 = 241). GPU-only dense/sparse are the compute-comparison table rows ([family page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)). Kernel-relevant number is **dense**.

### 2. Shared memory on GA10B — identical across boards, and NOT 100 KB

- **164 KB per SM, 163 KB max per thread block (opt-in), 48 KB default per kernel** — identical on every Orin board (same die).
- Official CUDA docs (compute-capability table): CC 8.7 = "164 KB" per SM and "163 KB" per thread block — the **8.0/A100-class values, not the 8.6 100/99 KB values** ([CUDA 11.8 Programming Guide, archived static table](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html)).
- deviceQuery on AGX Orin 64GB, AGX Orin 32GB and Orin Nano 8GB all report `Total shared memory per multiprocessor: 167936` (= 164 KB) and default per-block 49152; NVIDIA moderator (AastaLLL) explains the opt-in path via `cudaFuncSetAttribute` / `cudaDevAttrMaxSharedMemoryPerBlockOptin` ([forum thread](https://forums.developer.nvidia.com/t/orin-shared-memory-size-documentation/217043)).
- ⚠️ **Correction to the repo's internal brief** (`task/research/arxiv-orin-decode-attention-kernel.md` line 3 says "100KB smem/block"): Orin is 164 KB/SM / 163 KB max block. 100/99 KB is desktop GA10x (sm_86/sm_89).

### 3. L2 cache and persistence

- Die-level: **4 MB L2** on GA10B ([tech brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)).
- Observed `l2CacheSize` (deviceQuery, unofficial): AGX 64GB = 4 MB ([forum](https://forums.developer.nvidia.com/t/orin-shared-memory-size-documentation/217043)); AGX 32GB = 4 MB ([forum](https://forums.developer.nvidia.com/t/jetson-agx-orin-32gb-measured-memory-bandwidth-much-lower-than-theoretical-spec/331766)) — the 7-TPC/14-SM 32GB keeps the full 4 MB; **Orin NX 16GB = 2 MB** ([GitHub deviceQuery, JetPack-era](https://github.com/atinfinity/l4t-base-docker)); **Orin Nano 8GB = 2 MB** on JetPack 5/6 ([forum](https://forums.developer.nvidia.com/t/jetson-orin-nano-8gb-cache-line-size-and-page-size/306256)) but **4 MB** in one JetPack 7.2.1 report ([forum](https://forums.developer.nvidia.com/t/cuda-device-cant-be-detected-on-jetson-orin-nano-running-jp7/383847/33)) — unresolved; verify on target BSP.
- **L2 persistence (`cudaAccessPolicyWindow`) is supported**: official CUDA guide — "Devices of compute capability 8.x allow a portion of L2 cache to be set-aside for persisting data accesses" ([CUDA 11.8 guide](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html)). All Orin = CC 8.7 ⇒ supported family-wide; size the window to the observed L2 above.
- deviceQuery's "Memory Bus Width" is **unreliable on Tegra** (AGX Orin prints "128-bit" though it is 256-bit; an NX 16GB prints "64-bit" though it is 128-bit) — use data-sheet widths.

### 4. JetPack / L4T / CUDA compatibility (official archive)

From the [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive) and [downloads page](https://developer.nvidia.com/embedded/jetpack/downloads):

| JetPack | L4T | CUDA | Boards |
|---|---|---|---|
| 5.0–5.0.2 | 34.1–35.1 | 11.4 | AGX Orin DevKit only (of Orin) |
| 5.1–5.1.7 (current 5.1.7) | 35.2.1–35.6.5 | 11.4 | AGX Orin series + Orin NX; **Orin Nano series from 5.1.1**; 5.1.5 adds Super Mode; Xavier too |
| 6.0–6.2.3 (current 6.2.3) | 36.3–36.5.2 | 12.x | **All Orin series**; 6.2 introduced Nano/NX Super Modes (new flashing configs); AGX Orin 32GB NOT included yet |
| 7.0 / 7.1 | 38.2 / 38.4 | 13.0/13.x | **Thor only — no Orin** |
| **7.2 / 7.2.1 (latest)** | 39.2 / 39.2.1 | **13.0 / 13.2.2** | Thor + **"Jetson Orin Family"**; adds **Super Mode for AGX Orin 32GB** (930 MHz→1.3 GHz, 200→241 TOPS, up to 60 W); unified Ubuntu 24.04 / kernel 6.8 / CUDA 13 stack across Orin+Thor |

- So **AGX Orin on JetPack 7 starts at 7.2** (7.0/7.1 are Thor-only) — matches the repo's "Orin: JP 7.2.1+" note ([JP 7.2 blog](https://developer.nvidia.com/blog/deploy-agentic-ready-ai-at-the-edge-with-memory-efficiency-in-nvidia-jetpack-7-2/)).
- Orin product lifecycle extended **through 2032** ([JP 6.2 blog](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/)).
- JetPack 7.2.1 ISO flashes the Orin Nano Dev Kit in **Super Mode configuration by default** ([downloads page](https://developer.nvidia.com/embedded/jetpack/downloads)); Orin Nano on JP 7 runs CUDA 13.2 (driver 595.78) per forum deviceQuery.

### 5. FlashInfer multi-CTA sampler claim — verified/refuted

The claim: *"FlashInfer's multi-CTA sampler needs >16 SMs; Orin AGX (96 SMs) fine, Orin NX (16 SMs) borderline, Nano (6) deadlocks" — flashinfer issue #4544.*

Actual issue and resolution:
- [#4544](https://github.com/flashinfer-ai/flashinfer/issues/4544) is about **NVIDIA A16 (sm_86 / GA107, ~10 SMs)** hanging in the sampler under TP=4 + MTP — it does not mention Jetson at all. The board SM counts in the paraphrase are not from the issue.
- Maintainer **bkryu** (COLLABORATOR): *"the multi-CTA sampler kernel is optimized and functional for GPUs with **over 16 SMs**; we did not put into account A16 with 10 SMs … Because the kernel requires **more than 16 concurrently running CTAs** to avoid deadlock, supporting GPUs with fewer than 16 SMs would require a new kernel variant"* ([comment](https://github.com/flashinfer-ai/flashinfer/issues/4544#issuecomment-5334641245)).
- Fix: [PR #4595](https://github.com/flashinfer-ai/flashinfer/pull/4595) (merged 2026-08-19): *"Prevent `top_k_mask_logits` from launching its multi-CTA radix kernel on **GPUs with 16 or fewer SMs**. The multi-CTA implementation uses a software barrier that requires all participating CTAs to remain resident concurrently."* Returns `cudaErrorNotSupported` and keeps the single-CTA path.

Verdict on each number:
- **"Orin AGX (96 SMs) fine" — WRONG twice.** AGX Orin 64GB has **16 SMs** (2048 cores; official tech brief + deviceQuery), not 96. And 16 SMs is **exactly the rejected boundary** of PR #4595 ("16 or fewer SMs") — so the multi-CTA sampler is **not** "fine" on AGX Orin; post-#4595 it should fail fast with `cudaErrorNotSupported`, and pre-#4595 it was at deadlock risk. **The multi-CTA sampler is unsupported on the entire Orin family** (16 SMs max). Use `VLLM_USE_FLASHINFER_SAMPLER=0` or the single-CTA/`filter_apply_order="joint"` path.
- **"Orin NX (16 SMs) borderline" — WRONG.** Orin NX 16GB and 8GB both have **8 SMs** (1024 cores / 128; DS v1.7 + JP 6.2 blog).
- **"Nano (6 SMs) deadlocks" — WRONG count, right failure mode.** Orin Nano 8GB = **8 SMs** (three independent deviceQuery outputs, incl. one on JetPack 7.2.1/CUDA 13.2); Nano 4GB = 4 SMs. **No Orin board has 6 SMs.** The deadlock conclusion for ≤16-SM boards is correct per the maintainer, but so is AGX Orin (16) — the claim's "AGX fine" exception is false.
- **"AGX Orin 64GB = 16 SMs (2048 cores)" — CORRECT** (2048/128 = 16; deviceQuery "(016) Multiprocessors, (128) CUDA Cores/MP: 2048"; tech brief "up to 16 SMs").

---

## Source Breakdown (claim → source → confidence)

| Claim | Source | Confidence | Contradictions |
|---|---|---|---|
| AGX Orin 64GB: 2048 cores, 64 TC, 1.3 GHz, 204.8 GB/s, 15–60 W | [Family page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) + [tech brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf) | High | None |
| AGX Orin = 16 SMs, GA10B 128 cores/SM, 4 TC/SM, 192 KB L1/SM, 4 MB L2 | [Tech brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf) + [deviceQuery](https://forums.developer.nvidia.com/t/orin-shared-memory-size-documentation/217043) | High | None |
| AGX Orin 32GB: 1792 cores, 56 TC, 14 SMs (7 TPCs), 930 MHz std, 200 TOPS std | [Tech brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf) + [deviceQuery (14 MP, 930 MHz)](https://forums.developer.nvidia.com/t/jetson-agx-orin-32gb-measured-memory-bandwidth-much-lower-than-theoretical-spec/331766) | High | Family-page spec table lists 241 TOPS/1.3 GHz — resolved as the JP 7.2 Super Mode (next row) |
| AGX Orin 32GB Super Mode: 930→1300 MHz, 200→241 TOPS, up to 60 W (JetPack 7.2) | [JP 7.2 blog](https://developer.nvidia.com/blog/deploy-agentic-ready-ai-at-the-edge-with-memory-efficiency-in-nvidia-jetpack-7-2/) | High | The same family page's compute-comparison table still shows the 200-TOPS std values (stale) |
| AGX Orin Industrial: 2048 cores/64 TC @ 1.2 GHz, 248 TOPS, 15–75 W, ECC LPDDR5, −40–85 °C, 10-yr life | [Family page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) (GPU-row colspan groups Industrial under "2048-core/64 TC"; 248 = 16 SM × 2048 FLOP/clk × 1.2 GHz ✓) | High | None |
| Orin NX 16GB/8GB: 1024 cores, 32 TC (8 SMs), 918/765 MHz std → 1173 MHz MAXN SUPER, 128-bit LPDDR5 102.4 GB/s | [DS-10712-001 v1.7](https://developer.nvidia.com/downloads/jetson-orin-nx-module-series-data-sheet) + [JP 6.2 blog Table 2](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/) + [family page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) | High | Some partner pages (Connect Tech, Assured Systems) still list NX8 = 512 cores/16 TCs — contradicts NVIDIA's own current DS/blog; treat as stale |
| Orin Nano 8GB: 1024 cores, 32 TC, 8 SMs, 625→1020 MHz, 68→102 GB/s | [JP 6.2 blog Table 2](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/) + [deviceQuery ×3](https://forums.developer.nvidia.com/t/jetson-orin-nano-8gb-cache-line-size-and-page-size/306256) | High | None |
| Orin Nano 4GB: 512 cores, 16 TC, 4 SMs, 625→1020 MHz, 34→51 GB/s | [JP 6.2 blog Table 2](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/) + family page | High | None |
| Nano 8GB bus = 128-bit | Family page row ("8GB 128-bit LPDDR5 102 GB/s") + BW arithmetic (16 B × 6.4 GT/s = 102.4) | High | (my initial 64-bit guess was wrong — corrected via bandwidth arithmetic) |
| smem on CC 8.7 = 164 KB/SM, 163 KB max block, 48 KB default | [CUDA 11.8 guide table](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html) + [deviceQuery 167936 on 3 boards](https://forums.developer.nvidia.com/t/orin-shared-memory-size-documentation/217043) | High | Repo's earlier note "100KB smem/block" is wrong for Orin |
| L2 persistence supported on 8.x | [CUDA 11.8 guide](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html) | High | None |
| L2 = 4 MB (AGX), 2 MB (NX16, Nano8 JP5/6), Nano-JP7.2.1 anomaly 4 MB | deviceQuery links above | Medium | Nano 8GB: 2 MB (JP5/6) vs 4 MB (one JP 7.2.1 output) — unresolved |
| FlashInfer multi-CTA sampler requires >16 SMs; guarded at ≤16 SMs | [issue #4544 comment](https://github.com/flashinfer-ai/flashinfer/issues/4544#issuecomment-5334641245) + [PR #4595](https://github.com/flashinfer-ai/flashinfer/pull/4595) | High | The GitHub-thread paraphrase's board SM counts (96/16/6) are all wrong |
| JetPack version→board matrix | [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive) + [downloads](https://developer.nvidia.com/embedded/jetpack/downloads) | High | None |
| DGX Orin/Station use the AGX Orin 64GB module (GPU identical) | Could not reach a live official page (retired) | **Low** | — |

---

## Methodology

- **Searches (sequential, one at a time):** Orin NX data sheet PDF → AGX Orin data sheet PDF → "Orin NX 8GB 512 cores" history → deviceQuery AGX (16 MP) → deviceQuery L2 → deviceQuery 8.7 → Orin Nano deviceQuery → Orin NX L2 → Jetson L2 docs → DGX Orin specs (×4, all dead ends) → "AGX Orin 32GB 241 TOPS" → Orin smem optin (×2, empty).
- **Pages fully read (web_md / raw HTML / PDF→text):** family page (raw HTML table with colspans parsed), AGX Orin technical brief PDF, Orin NX DS v1.7 PDF, JetPack 6.2 blog, JetPack 7.2 blog, JetPack downloads, JetPack archive, CUDA 11.8 guide (CC table + persistence + accessPolicyWindow), forum threads #217043, #331766, #306256, #383847 (both pages), #232540, GitHub atinfinity/l4t-base-docker README, Wikipedia DGX, flashinfer #4544 + #4595 (github tools).
- **GitHub tools used:** `github_issue_read` ×3 (flashinfer #4544, comments, #4595).
- **Raw artifacts:** `task/research/orin-board-matrix/` (saved page dumps + pdftotext of DS-10712-001 v1.7 and the tech brief; forum JSON extractions).
- **Skipped and why:** Orin Nano data-sheet PDF (login-walled; content covered by JP 6.2 blog Table 2); original 2022 NX DS v1.0 (mirror served v1.7); DGX Orin archived pages (Internet Archive offline; live pages 404; 4 searches empty).

## Gaps and Caveats

1. **DGX Orin / DGX Orin Station**: NVIDIA's product pages, docs and datasheet are retired (404s); Internet Archive was offline; search indexes return nothing. The row assumes the known module identity (AGX Orin 64GB module ⇒ same GPU) — treat as **unverified against a live official page**. If needed, verify from a physical unit's `deviceQuery`.
2. **AGX Orin 32GB standard power envelope** (15–40 W vs 60 W): official pages list only the Super-era 15–60 W; the 15–40 W std figure comes from distributor docs + the JP 7.2 blog's "higher power envelopes up to 60 W" phrasing. Confirm with `nvpmodel -q` on hardware.
3. **Orin Nano 8GB L2** on JetPack 7 (2 MB vs 4 MB reports) — unresolved; matters for access-policy-window sizing. Verify with `deviceQuery` on the target BSP.
4. **NX 8GB / Nano 4GB L2** — no deviceQuery found; inferred from SM count (2 MB / ~1 MB expected).
5. **fp16 with fp32-accumulate rate on sm_87**: NVIDIA's 43-dense-TFLOPS figure doesn't state accumulator precision. Desktop GA10x (sm_86) halves fp16 rate with fp32 acc; A100 doesn't; Orin's class (164 KB smem, A100-like) suggests full rate but this was **not verified from fetched sources** — microbenchmark `mma.sync` fp16/fp32-acc on the Thor/Orin before trusting 43 TFLOPS for an fp32-acc kernel.
6. **Original-launch NX8 silicon story**: three official sources agree NX8 = 1024 cores/32 TCs today (and the JP 6.2 blog's "original" column lists the same), but some partner pages still show 512/16; the 2022 v1.0 data sheet could not be retrieved to fully close the history.
