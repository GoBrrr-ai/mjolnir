# mjolnir-kernel

**Mjolnir** — a paged decode-attention kernel for NVIDIA Jetson AGX Thor
(`sm_110`, 20 SMs), written in CuTe DSL on Blackwell `tcgen05` tensor cores, and
packaged as a vLLM attention backend.

```bash
vllm serve <model> --attention-backend MJOLNIR      # the mjolnir-vllm image
vllm serve <model> --attention-backend CUSTOM       # stock vLLM 0.32.0 + this package
```

This package is the kernel only. Serving, calibration and benchmarking are the
[`mjolnir-cli`](https://pypi.org/project/mjolnir-cli/) package; benchmarks
(Mjolnir vs FlashInfer vs FlashAttention 2 across context lengths), the design,
and one chapter per kernel feature live in the project repository:
**[GoBrrr-ai/mjolnir](https://github.com/GoBrrr-ai/mjolnir)** — numbers there
are rendered from the raw results, so they are never stale.

## What it does

The backend subclasses a stock vLLM backend — the *fallback* — and inherits
its KV-cache allocation, cache write, prefill and validation:

- **on the image**, FlashAttention 4 (`FLASH_ATTN`, resolved to FA4 on sm_110
  by the image's patch — its prefill is ~40% faster than FlashInfer's on
  Thor). If FA4's one-time fp8-KV probe fails on the device, the backend logs
  it and uses FlashInfer instead;
- **on a stock vLLM**, FlashInfer (stock FA on sm_110 is FA2, which cannot
  read an fp8 KV cache);
- `MJOLNIR_ATTN_FALLBACK` picks another either way.

Steps that are **decode only** run on Mjolnir:

- every request has the same `q_len ≤ 1 + num_speculative_tokens` (so MTP /
  n-gram speculative verification runs on Mjolnir too);
- CUDA-graph padding requests (zero tokens) may trail the batch;
- the step is causal — or the non-causal block of a parallel drafter (DSpark,
  DFlash: every query token sees every key) — and has no cascade prefix.

Every other step (prefill, mixed, cascade) is built and run by the base backend,
unchanged.

A decode step:

1. **Plan** — an occupancy-pinned split-KV plan (`plan_split_kv`): exactly one
   wave of 2 CTAs × 20 SMs. The plan lives in persistent device buffers padded to
   a fixed grid, so full CUDA graphs replay new plans without re-capture.
2. **Quantize Q** — dynamic e4m3 per `(sequence, KV head)` by default: fp8 Q,
   fp8 KV and fp8 P go straight into the tensor cores, no conversion warps.
3. **Stream KV** — TMA loads pages into a shared-memory ring
   (L2 evict-first: KV is read once), `tcgen05` MMAs accumulate S and O in TMEM,
   softmax warpgroups run the online softmax.
4. **Merge** — a small second kernel combines the split-KV partials by LSE.

The KV cache is read through runtime strides: vLLM's `(B,H,N,2D)`, FlashInfer
NHD/HND, or FlashAttention `(2,B,…)`, uint8 or fp8, with per-tensor or per-head
scales. Q and O are zero-copy for `q_len == 1`, even as strided views of the
fused QKV buffer.

## Install

**On the image** (`narandill/mjolnir-vllm:v0.32.0`) it is pre-installed and the
`MJOLNIR` backend name is patched into vLLM — nothing to do.

**On a stock vLLM 0.32.0** (inside a container that has CUDA 13 and `sm_110`):

```bash
pip install "mjolnir-kernel[dsl]"          # [dsl] pulls nvidia-cutlass-dsl
VLLM_KV_CACHE_LAYOUT=HND vllm serve Qwen/Qwen3-8B \
    --kv-cache-dtype fp8 --attention-backend CUSTOM
```

The package registers a `vllm.general_plugins` entry point (`mjolnir:register`)
that maps `AttentionBackendEnum.CUSTOM` — and `MJOLNIR`, when the vLLM build has
that member — to `mjolnir.backend.MjolnirAttentionBackend` in every vLLM process.
The two-line enum patch for vLLM is
[`thor-mjolnir-attention-backend.patch`](https://github.com/GoBrrr-ai/mjolnir/blob/main/docker/vllm-thor/patches/thor-mjolnir-attention-backend.patch).

Look for this line in the server log:

```
Mjolnir decode enabled: h_q=… h_k=… d=… page=… q_len<=… Q=fp8 ctas/SM=2 …
```

## Requirements

- An fp8 (`fp8` / `fp8_e4m3`) KV cache — or, on the image, a 6-bit
  `fp6_e2m3` one with calibrated scales (see below); head-major pages
  (`VLLM_KV_CACHE_LAYOUT=HND`) are fastest.
- `head_size % 128 == 0`; block size 16, 32, 64 or 128.
- A bf16 or fp16 model.
- No sliding window, soft-cap, sinks or ALiBi; no DCP or PCP.

If a layer does not qualify, the builder logs the reason and that KV-cache group
runs on the fallback backend.

## Configuration

| variable | default | meaning |
|---|---|---|
| `MJOLNIR_ATTN_DISABLE` | – | `1`: the plugin does not register; everything runs on the fallback |
| `MJOLNIR_ATTN_FALLBACK` | `FLASH_ATTN` (FA4) on the image, `FLASHINFER` on stock vLLM | base backend (`FLASHINFER`, `TRITON_ATTN`, …) for prefill and mixed steps |
| `MJOLNIR_ATTN_Q_DTYPE` | `fp8` | `fp8` (dynamic per-(seq, KV head) quantization) or `bf16` |
| `MJOLNIR_ATTN_MAX_QLEN` | `1 + num_speculative_tokens` | the largest q_len Mjolnir takes |
| `MJOLNIR_ATTN_CTAS` | `2` | CTAs per SM (2 is fastest on Thor) |
| `MJOLNIR_ATTN_MAX_SPLITS` | `16` | max KV chunks per sequence (1 = never split, no merge kernel). On Orin it caps the split capacity the kernel's occupancy picks (8 for the mma kernel on AGX Orin) |

| `MJOLNIR_ATTN_KV_STAGES` | `0` | KV ring depth override (0 = the tuned default) |
| `MJOLNIR_ATTN_PDL` | `0` | `1`: programmatic dependent launch |
| `MJOLNIR_ATTN_L2_PREFETCH` | `0` | TMA-prefetch K/V this many tiles ahead into L2 |
| `MJOLNIR_ATTN_KV_EVICT_FIRST` | `1` for fp8 Q, `0` for bf16 Q | L2 evict-first policy on K/V loads |
| `MJOLNIR_ATTN_PRECOMPILE` | `all` | compile every variant at start-up; `none` compiles on first use |
| `MJOLNIR_ATTN_KERNEL_CACHE` | on | compiled variants are exported to `$VLLM_CACHE_ROOT/mjolnir/kernels` (else `~/.cache/mjolnir/kernels`) and loaded on the next start instead of recompiling; `0` = off, a path = that directory. Keyed by variant, kernel source, CuTe DSL version and GPU |
| `MJOLNIR_ATTN_DRAFT_T5` | `1` | `0`: the drafter's layers (MTP / EAGLE) run on the fallback — isolates the draft-step cost (`mjolnir verify profile`). fp8 KV only: refused on fp6 KV, whose fallback cannot be CUDA-graphed |
| `MJOLNIR_ATTN_NONCAUSAL` | `1` | `0`: a parallel drafter's non-causal block steps (DSpark / DFlash) go to the fallback instead of the decode kernel — fp8 KV only for the same reason as above |
| `MJOLNIR_ATTN_FUSED_DRAFT` | `1` | `0`: tell vLLM the backend cannot advance draft metadata in place → one graph replay per draft step instead of the fused draft loop |
| `MJOLNIR_LM_HEAD` | – | `fp8`: quantize the lm_head to e4m3 per row at load, Triton fp8-weight GEMM for the verify and every draft (`mjolnir.lmhead`; config `NVFP4_MJOLNIR_E2M3_LMH8_VT64K`); the bf16 copy is freed automatically (untied heads only, +2.5 GB KV) |
| `MJOLNIR_DRAFT_VOCAB` | – | path to a `draft_vocab.json` (`mjolnir calibrate --draft-vocab`): the drafter's logits only over those tokens (VocabTrim); `MJOLNIR_DRAFT_VOCAB_SIZE` keeps the first N (a shorter file is padded with the lowest unused ids, with a warning). The kept head stays in the drafter's native quant (NVFP4 4-bit in-kernel, FP8 row-slice) — never up-scaled to bf16; a bf16 head is the one exception (fp8 only when `MJOLNIR_LM_HEAD=fp8` is set). `MJOLNIR_DRAFT_VOCAB_NATIVE=0` forces the dequant→fp8 path |
 | `MJOLNIR_ATTN_VERBOSE` | – | `1`: print the kernel's compile-time notes (ring stages, TMEM columns) — the standalone harness always does |
 | `MJOLNIR_ATTN_BANNER` | `1` | `0`: skip the Mjolnir hammer banner in the vLLM boot log (`mjolnir.banner.emit`, once per start, coloured with the server's formatter) |
| `MJOLNIR_ATTN_KV_LAYOUT` | auto | force the KV view: `BHN2D BNH2D B2NHD B2HND 2BNHD 2BHND` |
| `MJOLNIR_ATTN_FUSED_QPACK` | `1` | `0`: quantize fp8 Q with PyTorch ops (~10 launches) instead of one Triton kernel |
| `MJOLNIR_ATTN_KV_SCALES` | – | `fp6_e2m3` KV: the per-(layer, KV head, channel) scales file from `python -m mjolnir.calibrate` |
| `MJOLNIR_ATTN_KV_SCALE_HEADROOM` | `1.0` | `fp6_e2m3` KV: multiply every calibrated scale |
| `MJOLNIR_ATTN_KV_CALIBRATE` | – | `1`: record the per-channel amax of every K/V written (what `mjolnir.calibrate` reads) |

The pre-rename `THOR_T5_*` spellings are still honored. Each default is backed
by a measurement — see the
[kernel chapters](https://github.com/GoBrrr-ai/mjolnir/tree/main/docs/kernel).

## fp6 (e2m3) KV in vLLM

On the image, `--kv-cache-dtype fp6_e2m3` stores K/V as packed 6-bit e2m3
codes (25% fewer bytes than fp8) with static per-(layer, KV head, channel)
scales — per channel because K has outlier channels a head-wide scale would
crush the rest under. Mjolnir writes that cache (`mjolnir/kvfp6.py`), decodes
from it directly (K's channel scales folded into Q, V's into the output), and
expands the pages a prefill / mixed step reads to e4m3 for the FA4 fallback
(channel scales applied, one descale per head). The scales come from a calibration run on fp8 KV:

```bash
python -m mjolnir.calibrate --config <serve.yaml> --out kv_scales/e2m3.json   # in the image
MJOLNIR_ATTN_KV_SCALES=kv_scales/e2m3.json vllm serve ... --kv-cache-dtype fp6_e2m3 --attention-backend MJOLNIR
```

(`mjolnir calibrate` / `mjolnir serve` do both for you —
[docs/faq/fp6-kv.md](https://github.com/GoBrrr-ai/mjolnir/blob/main/docs/faq/fp6-kv.md).)

## Beyond fp8: the harness

`mjolnir/arch/thor/decode.py` is the kernel; `mjolnir/arch/thor/harness.py`
(`python -m mjolnir.arch.thor.harness --help`) is its standalone harness, with a bit-faithful emulator as the
correctness reference. The harness can also read KV formats vLLM cannot write
yet — fp6 (`e2m3`/`e3m2`) and fp4 (`e2m1`) read directly by `tcgen05
kind::f8f6f4`, and `nvfp4` — and defaults to e2m3 on head-major pages for fp8 Q.
How they compare in speed and accuracy:
[`kvfmt`](https://github.com/GoBrrr-ai/mjolnir/blob/main/docs/kernel/kvfmt.md) and
[`quant`](https://github.com/GoBrrr-ai/mjolnir/blob/main/docs/kernel/quant.md).
Wiring a vLLM cache-write path for fp6 is an open contribution.

## Platforms and architectures

The kernel is written for one architecture today — Thor (`sm_110`) — but the
package is organized by platform: `mjolnir/arch/` is the implemented-kernel
registry (`mjolnir.arch.thor` re-exports the exact runner surface the backend
consumes), and `mjolnir/platforms.py` is the torch-free leaf that maps a
compute capability to it. Thor and Jetson Orin (`sm_87`) are implemented;
DGX Spark (`sm_121`) is declared research — named in the matrix, no kernel
yet. Orin ships one kernel: the `mma.m16n8k16` decode kernel (fp8 and
fp6-e2m3 KV, hd 256, M ≤ 8, bf16 Q) — see `mjolnir/arch/orin/README.md`.
The retired FMA GEMV lives in the `gemv-archive` git tag.

`kernel_for((11, 0))` returns `mjolnir.arch.thor`; any other compute
capability raises `PlatformNotSupported` naming the requested `sm_XX` and the
shipped platforms. The vLLM backend never sees that exception: its metadata
builder consults the pure `unsupported_reason()` and takes the normal
ineligible-config path — that layer's decode steps run on the fallback
backend, the serve continues. On a host without a usable CUDA device,
`MJOLNIR_ARCH` (`sm_110` or `11.0`) pins the architecture instead of the
torch read (the standalone harness).

There is no remembered platform state anywhere in the stack: the CLI side
resolves its platform per invocation (flag > `$MJOLNIR_PLATFORM` > host
detection — four source classes with provenance, the `/tmp/nv_jetson_model`
model copy corroboration-only), so the kernel is always resolved from the
device the run actually happens on. See
[docs/faq/platforms.md](../../docs/faq/platforms.md).

## Tests

From a checkout, on a Thor:

```bash
mjolnir verify correctness      # kernel vs the emulator, the full case matrix
mjolnir verify vllm             # runner tests + greedy MJOLNIR-vs-FLASHINFER smoke runs
```

or directly in the container: `python tests/kernel/test_runner.py`,
`python tests/kernel/smoke_vllm.py [--eager] [--spec 2]` (they find the package
installed, via `$MJOLNIR_SRC`, or by walking up to `src/mjolnir`).

Mjolnir rounds P (and, by default, Q) to e4m3, so greedy outputs can flip on
near-tie tokens and then drift apart — FlashInfer's own greedy output differs
between CUDA-graph and eager mode. The smoke test therefore compares
per-step logprobs, not text: the chosen tokens must get the same logprob from
both backends on the shared prefix, and every divergence must be a near-tie
(each token in the other's top-5, at most 1 nat apart). The matching prefix
is printed as information.

## Files

| file | what |
|---|---|
| `mjolnir/arch/thor/` | the Thor kernel family: `decode.py` (the CuTe-DSL kernel + merge), `call.py`, `const.py`, `plan.py`, `runner.py`, `harness.py` (driver + emulator) |
| `mjolnir/runner.py` | plan buffers, compile cache, launch, KV views (no vLLM import) |
| `mjolnir/backend.py` | the vLLM backend, metadata builder, impl |
| `mjolnir/platforms.py` | arch → kernel resolution (`kernel_for`, `PlatformNotSupported`), `MJOLNIR_ARCH` detection — torch-free |
| `mjolnir/arch/thor.py` | the Thor (`sm_110`) kernel surface: re-exports of the runner |
| `mjolnir/__init__.py` | `register()` — the vLLM plugin entry point |

Apache-2.0, except `mjolnir/arch/thor/decode.py`, which is derived from an NVIDIA
CUTLASS CuTe DSL example and stays under BSD-3-Clause — see `LICENSE` and
`NOTICE` in this package.
