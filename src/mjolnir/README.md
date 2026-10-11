# mjolnir-kernel

**Mjolnir** — a paged decode-attention kernel for NVIDIA Jetson AGX Thor
(`sm_110`, 20 SMs), written in CuTe DSL on Blackwell `tcgen05` tensor cores, and
packaged as a vLLM attention backend.

```bash
vllm serve <model> --attention-backend MJOLNIR      # the mjolnir-vllm image
vllm serve <model> --attention-backend CUSTOM       # stock vLLM 0.31.0 + this package
```

Benchmarks (Mjolnir vs FlashInfer vs FlashAttention 2 at 128 to 256K tokens of
context), the design, and one chapter per kernel feature live in the project
repository: **[thor-arena/mjolnir](https://github.com/thor-arena/mjolnir)** —
numbers there are rendered from the raw results, so they are never stale.

## What it does

The backend subclasses a stock vLLM backend (FlashInfer by default,
`MJOLNIR_ATTN_FALLBACK` picks another) and inherits its KV-cache allocation,
cache write, prefill and validation. Steps that are **decode only** run on
Mjolnir:

- every request has the same `q_len ≤ 1 + num_speculative_tokens` (so MTP /
  n-gram speculative verification runs on Mjolnir too);
- CUDA-graph padding requests (zero tokens) may trail the batch;
- the step is causal and has no cascade prefix.

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

**On the image** (`narandill/mjolnir-vllm:v0.31.0`) it is pre-installed and the
`MJOLNIR` backend name is patched into vLLM — nothing to do.

**On a stock vLLM 0.31.0** (inside a container that has CUDA 13 and `sm_110`):

```bash
pip install "mjolnir-kernel[dsl]"          # [dsl] pulls nvidia-cutlass-dsl
VLLM_KV_CACHE_LAYOUT=HND vllm serve Qwen/Qwen3-8B \
    --kv-cache-dtype fp8 --attention-backend CUSTOM
```

The package registers a `vllm.general_plugins` entry point (`mjolnir:register`)
that maps `AttentionBackendEnum.CUSTOM` — and `MJOLNIR`, when the vLLM build has
that member — to `mjolnir.backend.MjolnirAttentionBackend` in every vLLM process.
The two-line enum patch for vLLM is
[`thor-mjolnir-attention-backend.patch`](https://github.com/thor-arena/mjolnir/blob/main/docker/vllm-thor/patches/thor-mjolnir-attention-backend.patch).

Look for this line in the server log:

```
Mjolnir decode enabled: h_q=… h_k=… d=… page=… q_len<=… Q=fp8 ctas/SM=2 …
```

## Requirements

- An fp8 (`fp8` / `fp8_e4m3`) KV cache; head-major pages
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
| `MJOLNIR_ATTN_FALLBACK` | `FLASHINFER` | base backend (`FLASH_ATTN`, `TRITON_ATTN`, …) for prefill and mixed steps |
| `MJOLNIR_ATTN_Q_DTYPE` | `fp8` | `fp8` (dynamic per-(seq, KV head) quantization) or `bf16` |
| `MJOLNIR_ATTN_MAX_QLEN` | `1 + num_speculative_tokens` | the largest q_len Mjolnir takes |
| `MJOLNIR_ATTN_CTAS` | `2` | CTAs per SM (2 is fastest on Thor) |
| `MJOLNIR_ATTN_MAX_SPLITS` | `16` | max KV chunks per sequence (1 = never split, no merge kernel) |
| `MJOLNIR_ATTN_KV_STAGES` | `0` | KV ring depth override (0 = the tuned default) |
| `MJOLNIR_ATTN_PDL` | `0` | `1`: programmatic dependent launch |
| `MJOLNIR_ATTN_L2_PREFETCH` | `0` | TMA-prefetch K/V this many tiles ahead into L2 |
| `MJOLNIR_ATTN_KV_EVICT_FIRST` | `1` for fp8 Q, `0` for bf16 Q | L2 evict-first policy on K/V loads |
| `MJOLNIR_ATTN_PRECOMPILE` | `all` | compile every variant at start-up; `none` compiles on first use |
| `MJOLNIR_ATTN_KV_LAYOUT` | auto | force the KV view: `BHN2D BNH2D B2NHD B2HND 2BNHD 2BHND` |

The pre-rename `THOR_T5_*` spellings are still honored. Each default is backed
by a measurement — see the
[kernel chapters](https://github.com/thor-arena/mjolnir/tree/main/docs/kernel).

## Beyond fp8: the harness

`mjolnir/t5_decode.py` is the kernel **and** its standalone harness
(`python -m mjolnir.t5_decode --help`) with a bit-faithful emulator as the
correctness reference. The harness can also read KV formats vLLM cannot write
yet — fp6 (`e2m3`/`e3m2`) and fp4 (`e2m1`) read directly by `tcgen05
kind::f8f6f4`, and `nvfp4` — and defaults to e2m3 on head-major pages for fp8 Q.
How they compare in speed and accuracy:
[`kvfmt`](https://github.com/thor-arena/mjolnir/blob/main/docs/kernel/kvfmt.md) and
[`quant`](https://github.com/thor-arena/mjolnir/blob/main/docs/kernel/quant.md).
Wiring a vLLM cache-write path for fp6 is an open contribution.

## Tests

From a checkout, on a Thor:

```bash
mjolnir verify correctness      # kernel vs the emulator, the full case matrix
mjolnir verify vllm             # runner tests + greedy MJOLNIR-vs-FLASHINFER smoke runs
```

or directly in the container: `python tests/kernel/test_runner.py`,
`python tests/kernel/smoke_vllm.py [--eager] [--spec 2]` (they find the package
installed, via `$MJOLNIR_SRC`, or by walking up to `src/mjolnir`).

Mjolnir rounds P (and, by default, Q) to e4m3, so greedy outputs match
FlashInfer for the leading tokens and may drift later; the smoke test checks a
minimum matching prefix and a mean matching fraction.

## Files

| file | what |
|---|---|
| `mjolnir/t5_decode.py` | the CuTe-DSL kernel, the split-KV merge, the harness and the emulator |
| `mjolnir/runner.py` | plan buffers, compile cache, launch, KV views (no vLLM import) |
| `mjolnir/backend.py` | the vLLM backend, metadata builder, impl |
| `mjolnir/__init__.py` | `register()` — the vLLM plugin entry point |

Apache-2.0, except `mjolnir/t5_decode.py`, which is derived from an NVIDIA
CUTLASS CuTe DSL example and stays under BSD-3-Clause — see `LICENSE` and
`NOTICE` in this package.
