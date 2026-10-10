# vllm-thor-t5

T5 fp8 paged decode attention (CuTe DSL, tcgen05) as a vLLM attention backend
for Jetson AGX Thor (sm_110, 20 SMs).

The backend is a subclass of a stock vLLM backend (FlashInfer by default).
It inherits the KV-cache layout, the KV-cache write, prefill and validation
from that backend. Steps that are **decode only** run on T5:

- every request has the same `q_len ≤ 1 + num_speculative_tokens`;
- CUDA-graph padding requests (zero tokens) may trail the batch;
- the step is causal and has no cascade prefix.

Every other step (prefill, mixed, cascade) is built and run by the base
backend, unchanged.

## How a decode step runs

- **Plan.** The builder makes an occupancy-pinned split-KV plan
  (`plan_split_kv`). The plan is copied into persistent device buffers and
  padded to a fixed grid. Full CUDA graphs therefore replay new plans
  without being re-captured.
- **Compile.** Compiled kernels are cached per `(q_len, reduction)` for the
  whole process. They are compiled at start-up, or on first use from
  `build()`, which runs outside graph capture.
- **Q quantization.** The default is dynamic e4m3 per `(seq, kv head)`.
  This gives the fully fp8 path: fp8 Q, fp8 KV and fp8 P, with no
  conversion warps. Use `THOR_T5_Q_DTYPE=bf16` to keep Q in bf16.
- **Memory layout.**
  - Q and O are zero-copy for `q_len == 1`, even when the query is a
    strided view of the fused QKV buffer.
  - For `q_len > 1`, Q and O are packed and unpacked.
  - The KV cache is read through runtime strides, so any layout works:
    vLLM-main `(B,H,N,2D)`, FlashInfer NHD/HND, or FlashAttention
    `(2,B,…)`, with uint8 or fp8 element type.
  - KV scales can be per-tensor or per-head.

## Performance on Thor (sm_110, 20 SMs)

Measured with ncu: cold L2, base clocks, one sequence, 24 query / 4 KV heads, `d = 256`.
The table is from one 4-way sweep; T5 legs ran with evict-first on.

| L | T5-fp8 vs FlashInfer | T5-bf16 vs FlashInfer |
|---|---|---|
| 128 | 1.95× | 2.15× |
| 4K–16K | 1.00–1.13× | 0.96–1.10× |
| 32K–64K | 0.93–1.01× | 0.82–0.96× |
| 128K–256K | 0.91–0.94× | 0.86–0.89× |
| geomean | **1.08×** | 1.04× |

The pure-FMA GEMV kernel measured 0.57× of FlashInfer, so it's retired from the default sweeps.

**Defaults and the measurements behind them:**

- **2 CTAs per SM** (`THOR_T5_CTAS=2`). At 1 CTA/SM, T5-fp8 drops to 0.80× of FlashInfer at 128K–256K.
- **Evict-first on for fp8 Q, off for bf16 Q** (`THOR_T5_KV_EVICT_FIRST`).
  - fp8: it won every A/B, by up to 29% wall clock at 4K–16K, and was neutral at long L.
  - bf16: it cost 4–18% at 32K–256K.
  - The kernel harness uses the same rule by default (`--kv_evict_first` auto).
- **PDL off.** Its A/B results changed sign between runs.

**Measurement noise.** On this box a single result moves about ±7% between runs. FlashInfer's best and average times also differ by 10–15% at long L. The sweep report therefore marks ±7% as "within noise" (🟰, `NOISE_PCT=7`) and profiles 50 iterations per point (`ITERS=50`). Compare A/B variants only within the same run.

**Known gap.** T5-fp8 is consistently 6–9% behind FlashInfer at 128K–256K, beyond the noise: 179–187 GB/s against 191–205 GB/s. Several knobs have been tried without closing it:

- deeper KV rings
- L2 prefetch
- PDL
- an in-kernel split merge

The next step is an ncu memory-workload profile at 256K.

## Experimental: KV layouts and formats (kernel harness only)

The kernel can also read three other KV-cache layouts and formats. They are A/B-testable in the harness and in `verify.sh`, but not wired into vLLM yet. Each format would need its own vLLM cache-write path.

| flag | what it does |
|---|---|
| `--kv_layout hnd` | head-major pages, `(pages, 2, h_k, PS, d)`: each KV head's page is one contiguous block (vLLM can allocate this with `VLLM_KV_CACHE_LAYOUT=HND`) |
| `--kv_format e3m2` / `e2m3` | fp6 KV, read directly by `tcgen05 kind::f8f6f4` (TMA unpacks to 1-byte SMEM containers); 25% fewer bytes, per-head descale |
| `--kv_format e2m1` | fp4 KV, read directly the same way; 50% fewer bytes, per-head descale only |
| `--kv_format nvfp4` | e2m1 + one e4m3 scale per 16 elements (+ a global scale); a convert warpgroup produces e4m3 in SMEM for the normal fp8 MMA; 44% fewer bytes |

All four need fp8 Q.

**Harness default: e2m3 on HND, 3-stage KV ring.** With fp8 Q, `t5_decode.py`
and every `verify.sh`/sweep stage now run **e2m3 KV on head-major pages** unless
flags say otherwise (`--kv_format auto` → e2m3 for fp8 Q, fp8
for bf16 Q; `--kv_layout hnd`). It measured 1.10–1.15× FlashInfer-HND at
64K–256K and −11% wall clock vs fp8-HND at 64K, at rel L2 error 0.051 (fp8:
0.045). The ring depth for fp6 at 2 CTAs/SM (d=256) is 3 stages instead of the
whole-tile 4: 3 beat 4 by 4–5% at 64K/256K in two separate A/Bs (−13% wall vs
fp8-HND at 64K); 6 stages was the slowest. fp8 keeps 4 (4 beat 3 there).
`--kv_stages N` overrides. The old leg is `--kv_format fp8 --kv_layout nhd`. The correctness
matrix still pins fp8/NHD per case (the vLLM path) and adds `default:` cases.
**vLLM is unchanged:** the backend reads vLLM's fp8 cache, so it runs fp8 KV
until a vLLM e2m3 cache-write path exists.

```bash
./verify.sh correctness   # includes HND / fp6 / fp4 / nvfp4 cases
./verify.sh quant         # accuracy: rel L2 error vs unquantized KV, per format
./verify.sh kvfmt         # speed: fp8, interleaved, hnd, fp6/fp4 on hnd (ncu + wall),
                          #   vs FlashInfer on NHD *and* HND (cells: "vs FI /vs FI-HND")
./verify.sh hndsweep      # full L sweep: FI-tc, FI-NHD, FI-HND vs T5-fp8 on HND
./verify.sh fp6prof       # full ncu profiles hnd / e2m3-hnd / e2m1-hnd at 64K + side-by-side
```

First kvfmt results (ncu, vs FlashInfer on NHD, at 8K / 64K / 256K):

| variant | vs FI-NHD | wall @64K vs fp8 | rel L2 error |
|---|---|---|---|
| fp8 (NHD) | 1.05 / 0.98 / 0.94 | — | 0.045 |
| interleaved | 0.93 / 0.99 / 0.92 | +3% | 0.045 |
| hnd | 1.13 / 1.13 / 1.08 | −9% | 0.045 |
| e2m3 + hnd | 1.21 / 1.19 / 1.16 | −18% | 0.051 |
| e2m1 + hnd | 1.79 / 2.14 / 2.13 | −50% | 0.20 |

These were measured against FlashInfer on NHD only. The FI-HND legs (`finhd`/`fihnd`,
driver `tests/thor_t5/fi_layout_driver.py`, same data in both layouts) show
whether the HND gain is a T5 advantage or a layout advantage both kernels share.

## Requirements

- An fp8 (`fp8` / `fp8_e4m3`) KV cache.
- `head_size % 128 == 0`.
- Block size 16, 32, 64 or 128.
- A bf16 or fp16 model.
- No sliding window, soft-cap, sinks or ALiBi.
- No DCP or PCP.

If any of these is not met, the builder logs the reason and the whole
KV-cache group runs on the fallback backend.

## Install

Install inside the vLLM container:

```bash
pip install -e ~/mjolnir --no-deps     # the project that holds src/vllm_thor_t5
```

The project's `pyproject.toml` needs the plugin entry point (and
`vllm_thor_t5` among its packages, e.g. `[tool.setuptools.packages.find] where = ["src"]`):

```toml
[project.entry-points."vllm.general_plugins"]
thor_t5 = "vllm_thor_t5:register"
```

The entry point maps
`AttentionBackendEnum.CUSTOM` to `vllm_thor_t5.backend.ThorT5AttentionBackend`
in every vLLM process.

## Use

```bash
vllm serve Qwen/Qwen3-8B --kv-cache-dtype fp8 --attention-backend CUSTOM
# older releases: VLLM_ATTENTION_BACKEND=CUSTOM vllm serve ...
```

Look for this line in the log:

`Thor T5 decode enabled: h_q=… h_k=… d=… page=… q_len<=… Q=fp8 ctas/SM=2 …`

## Environment

| variable | default | meaning |
|---|---|---|
| `THOR_T5_DISABLE` | – | `1`: plugin does not register; the builder routes everything to the fallback |
| `THOR_T5_FALLBACK` | `FLASHINFER` | base backend (`FLASH_ATTN`, `TRITON_ATTN`, …); prefill and mixed steps run on it |
| `THOR_T5_Q_DTYPE` | `fp8` | `fp8` (dynamic per-(seq, kv head) Q quantization) or `bf16` |
| `THOR_T5_MAX_QLEN` | `1 + num_speculative_tokens` | cap on the q_len T5 takes (longer goes to the fallback) |
| `THOR_T5_CTAS` | `2` | CTAs per SM (2 is fastest on Thor; 1 uses an 8-stage ring) |
| `THOR_T5_MAX_SPLITS` | `16` | max KV chunks per sequence (1 = never split, no reduction) |
| `THOR_T5_KV_STAGES` | `0` | KV ring depth override (0 = tuned default: one tile at 2/SM) |
| `THOR_T5_PDL` | `0` | `1`: programmatic dependent launch — decode overlaps the previous kernel's tail, the split-KV merge launches early |
| `THOR_T5_L2_PREFETCH` | `0` | TMA-prefetch K/V this many 128-row tiles ahead into L2 (try 1–4 at long context) |
| `THOR_T5_KV_EVICT_FIRST` | `1` for fp8 Q, `0` for bf16 Q | `1`: K/V TMA loads carry an L2 EVICT_FIRST policy (KV is streamed once) |
| `THOR_T5_PRECOMPILE` | `all` | compile every variant at start-up; `none` compiles on first use |
| `THOR_T5_KV_LAYOUT` | auto | force the KV view: `BHN2D BNH2D B2NHD B2HND 2BNHD 2BHND` |

## Tests

```bash
# kernel/runner vs the bit-faithful emulator (no vLLM needed)
python tests/thor_t5/test_runner.py
# end to end: greedy CUSTOM vs FLASHINFER (fp8 KV both sides)
python tests/thor_t5/smoke_vllm.py               # CUDA graphs
python tests/thor_t5/smoke_vllm.py --eager
python tests/thor_t5/smoke_vllm.py --spec 2      # ngram spec decode -> q_len 3
# (the tests find src/vllm_thor_t5 by walking up; or set THOR_T5_SRC)
# or everything in the container, from ~/mjolnir:
./verify.sh vllm
```

T5 rounds P to e4m3 and, by default, Q as well. Greedy outputs therefore
match FlashInfer for the leading tokens and may drift later. The smoke test
checks two thresholds:

- every prompt matches for at least `--min_prefix` tokens;
- the mean matching fraction is at least `--min_match`.

## Files

- `src/vllm_thor_t5/t5_decode.py` is the kernel and its standalone harness
  (`python t5_decode.py --help`). It is the single source: `verify.sh` and
  `ncu-t5-fp8-sweep.sh` profile and test this file (override with
  `KERNEL=` / `T5_KERNEL=`).
- `src/vllm_thor_t5/runner.py` holds the plan buffers, compile cache, launch and
  KV views. It does not use vLLM.
- `src/vllm_thor_t5/backend.py` holds the vLLM backend, builder, impl and
  metadata.
