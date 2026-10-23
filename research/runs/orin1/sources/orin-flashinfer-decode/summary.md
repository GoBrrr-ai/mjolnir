# Orin decode-attention: beating FlashInfer 0.7.0.post1 under vLLM 0.32.0 (sm_87)

Research date: 2026-10-22. Target: Jetson AGX Orin 64GB (sm_87, GA10B), FlashInfer 0.7.0.post1
decode path in vLLM 0.32.0, fp8 e4m3 KV, hd 256, GQA 24:4 (g=6), MTP q_len 1..8, pages 16/128.

## Headline correction (decision-grade)

**vLLM 0.32.0 never uses FlashInfer's GEMV decode kernel on Orin.** `_get_decode_wrapper`
constructs `BatchDecodeWithPagedKVCacheWrapper(..., use_tensor_cores=True)` **unconditionally**
([flashinfer.py v0.32.0, L1332-1340](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/flashinfer.py))
with the comment "Tensor cores are enabled by default because the perf would be at least as good
as cuda cores for all attention ops in latest gpus." With `use_tensor_cores=True`,
[decode.py (v0.7.0.post1)](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/flashinfer/decode.py)
routes plan/run to `get_batch_prefill_module("fa2", ...)` — i.e. the **FA2 tensor-core kernel**
(`BatchPrefillWithPagedKVCacheKernel`). fp8 KV on non-SM90 → `determine_attention_backend` returns
"fa2" ([utils.py](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/flashinfer/utils.py)).
XQA is SM90/SM100/SM120-only in FlashInfer ([issue #2522](https://github.com/flashinfer-ai/flashinfer/issues/2522)),
so on sm_87 the baseline to beat is **FA2-TC decode (mma.m16n8k16.f32.f16.f16.f32) with fp8 KV
converted in registers by software `fast_dequant_f8f16x4`**.

## Q1 — AGX Orin 64GB hardware facts

Primary source: [NVIDIA Jetson AGX Orin Technical Brief (PDF)](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf) (fetched, extracted with pdftotext):

- "up to 16 Streaming Multiprocessors (SM's), 192 KB of L1-cache per SM, and 4 MB of L2 Cache. There are 128 CUDA cores per SM ... and four 3rd Generation Tensor cores per SM. Jetson AGX Orin 64GB has 2048 CUDA cores and 64 Tensor cores with up to 170 Sparse TOPs of INT8 Tensor compute, and up to 5.3 FP32 TFLOPs of CUDA compute."
- 275 TOPS total INT8 (GPU 170 sparse + DLA 105); 32GB module = 14 SMs / 200 TOPS.
- Memory: LPDDR5 "max clock speed of 3200 MHz, with 6400 Gbps per pin, enabling 204.8 GB/s"; 256-bit bus; 64GB.
- GPU max freq 1.3 GHz (AGX Orin 64GB row); 32GB = 930MHz-class die.
- Also [developer blog](https://developer.nvidia.com/blog/delivering-server-class-performance-at-the-edge-with-nvidia-jetson-orin/): 2048 cores + 64 TC, 204.8 GB/s, 275 TOPS.

Derived (flagged): fp16 dense ≈ 42.5 TFLOPS (16 SM × 2048 FLOP/clk/SM × 1.3 GHz; consistent with 170 sparse INT8 TOPS → 85 dense INT8 → 42.5 dense FP16). **No fp8 Tensor Cores on sm_87** — CUDA guide §20.7: 4th-gen TCs with fp8 are CC 8.9/9.0 only; CC 8.0/8.6/8.7 have 3rd-gen TCs (fp16/bf16/tf32/int8/fp64).

CC 8.7 specifics ([CUDA C++ Programming Guide §20.7 + Table 26/27](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)):
- 128 FP32 cores/SM, 2 FP64/SM, 4 3rd-gen TC/SM.
- Unified L1/smem = **192 KB/SM**; smem carveout up to **164 KB/SM**; **max 163 KB per block** (the 128 KB/100 KB / 99 KB numbers are CC 8.6/8.9, NOT Orin).
- Table 27 (columns 7.5|8.0|8.6|8.7|8.9|9.0|10+): resident blocks/SM 16 (8.7), resident warps 48, resident threads **1536**, 64 K 32-bit registers/SM, 255 regs/thread, 1024 threads/block.
- Feature table: "L2 Cache Residency Management" = 7.x No / 8.x Yes → access-policy-window/evict hints supported on sm_87.
- GA10B supports: `cp.async` (PTX cp.async sm_80+; cp-size ∈ {4,8,16}; `cp.async.cg` with 16 B — FlashInfer emits exactly this, [cp_async.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/include/flashinfer/cp_async.cuh)); `createpolicy` cache-eviction policies (PTX ISA: "creates a cache eviction policy ... `.level::cache_hint`"); `discard` (128-byte granularity, "destructively discarded without writing back" — PTX ISA); `__ldg`/`ld.global.nc` (CC 5.0+; on Ampere read-only data cached in unified L1).
- **fp8: no HW cvt, no fp8 mma on sm_87.** FlashInfer gates `FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED` to `__CUDA_ARCH__ >= 900` ([vec_dtypes.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/include/flashinfer/vec_dtypes.cuh)); `mma...e4m3` requires ≥890 ([mma.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/include/flashinfer/mma.cuh)).

## Q2 — FlashInfer decode kernels (v0.7.0.post1)

GEMV path ([decode.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/include/flashinfer/attention/decode.cuh)):
- `vec_size = max(16/sizeof(DTypeKV), HEAD_DIM/32)`: fp8 hd256 → **16**; bf16 → 8. `bdx = HEAD_DIM/vec_size`: fp8 → **16**, bf16 → 32. `bdy = GROUP_SIZE` (g=6). `num_threads = max(128, bdx*bdy)`; `bdz = num_threads/(bdx*bdy)` (integer div → 1). So fp8 g=6 block = **(16,6,1) = 96 threads**; bf16 = (32,6,1) = 192.
- `tile_size_per_bdx = 1` for GROUP_SIZE>1; for MHA g=1: fp8 → 2 (batch) / 2 (single, hd<512), bf16 → 4. fp8+GQA note in code confirms "When FP8+GQA shrinks the K+V buffer (tile_size_per_bdx=1)".
- `DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM`: **compute_capability ≥ 8 → 2 stages; else 1** ([utils.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/include/flashinfer/utils.cuh)). SM80/86/87/89 all get 2.
- smem for hd256 fp8 g6 (derived from the launch formula): 2 stages × (K+V) = 2×2×6×1×256×1 B = 6144 B + 1024 B (offsets/md) ≈ **7.2 KB total**; per stage K+V = 3072 B. Each thread cp.async's 16 B (vec_bits=128) — `.cg` ([cp_async.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/include/flashinfer/cp_async.cuh)).
- Compute: per thread 16 fp8 → `cast_load` to float vec (software cvt on sm87) → fp32 FMA + `shfl_xor` reduce over bdx=16. Grid (padded_batch, num_kv_heads). Split-KV: if `batch×num_kv_heads < blocks_per_sm×num_SM` binary-search chunks to fill the grid; merge = separate `VariableLengthMergeStates` kernel (LSE). `get_heuristic_num_threads`: g==8 fp8 → 256 ("not enough registers for 512 threads") — single-decode only.

TC path (FA2, [prefill.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/include/flashinfer/attention/prefill.cuh)):
- Decode (packed_qo_len = q_len×g = 6 ≤ 16) → `FA2DetermineCtaTileQ` → **CTA_TILE_Q=16** ("1x4 warp layout": NUM_WARPS_Q=1, NUM_WARPS_KV=4, 128 threads). CTA_TILE_Q=32 only for hd_vo≥512; 64 for packed>16; 128 for >64 & hd<256.
- NUM_MMA_KV from launcher budget: kFixedSmem=16×256×2=8192 (Q bf16); kKVSmemPerMmaKV=(256+256)×16×4×1=32768 (fp8, no repack at CTA_TILE_Q=16); Orin 164KB/SM → **2 CTAs/SM → cap 83968 B → NUM_MMA_KV=2 → CTA_TILE_KV=128 KV rows/iter** (bf16: 65536 B/step → NUM_MMA_KV=1 → 64 rows). Derived from launcher formula + CC 8.7 limits.
- MMA: `mma.sync.aligned.m16n8k16.row.col.f32.f16/bf16.f16.f32` ([mma.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.0.post1/include/flashinfer/mma.cuh), enabled `__CUDA_ARCH__ >= 800`).
- fp8 KV staging: smem holds fp8 (swizzle 128B); in the QK loop: `ldmatrix` → `frag_layout_swizzle_16b_to_8b` → `vec_cast<half/bf16, fp8>::cast<8>` (= `fast_dequant_f8f16x4` software path on sm87: prmt + mask/shift + 2×HMUL2 bias per 4 values) → HMMA. No smem repack at CTA_TILE_Q=16 (`use_kv_repack` requires CTA_TILE_Q>16); the repack staging buffer exists only for prefill shapes.
- Chosen when: FlashInfer API default `use_tensor_cores=False` ("Will be faster for large group size in grouped query attention"); **vLLM 0.32.0 always passes True**. MTP (q_len_per_req>1) → `is_causal=True`, uniform_q_len plan.

## Q3 — vLLM 0.32.0 decode path ([flashinfer.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/flashinfer.py), [utils/flashinfer.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/utils/flashinfer.py))

- CC gate: SM80 ≤ cc ≤ SM121 (SM75 excluded, bug #3620). Supported head sizes [64,128,256,512]; kernel block sizes [16,32,64]; kv dtypes incl. fp8/fp8_e4m3/fp8_e5m2/nvfp4.
- On SM87: `_get_flashinfer_trtllm_api_decode_kernel()` → XQA only on SM90/SM12x, TRTLLM_GEN only SM100 → `use_trtllm_decode_attention=False` → native wrapper. `use_trtllm_attention` decode auto: `num_tokens <= 256 and kv_cache_dtype == "auto"` → False with fp8.
- Decode wrapper: `BatchDecodeWithPagedKVCacheWrapper(workspace, layout, use_cuda_graph=…, paged_kv_*_buffers, use_tensor_cores=True, backend="auto")` — unconditional.
- Q dtype on SM87 with fp8 KV = **model dtype (bf16)**: "Architectures with only fa2 (e.g. SM89, SM120) cannot consume FP8 queries, so keep the model dtype for Q there."
- run(): `decode_wrapper.run(q, kv_cache, q_scale, k_scale, v_scale, out=...)` — per-tensor scales folded into bmm1/bmm2 scale host-side.
- **CUDAGraph**: `get_cudagraph_support` → `UNIFORM_SINGLE_TOKEN_DECODE` on SM87 (uniform-batch requires `can_use_trtllm_attention`, i.e. XQA/trtllm-gen archs). Under spec decode (q_len>1) that means PIECEWISE downgrade (cf. vllm #49547). `reorder_batch_threshold = 1`.

## Q4 — TRT-LLM XQA

- XQA = TRT-LLM's tensor-core MQA/GQA decode kernel for the generation phase ("XQA kernel provides optimization for MQA and GQA during generation phase... Using tensor cores for acceleration, reducing data loading and conversion" — [XQA-kernel.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/XQA-kernel.md)).
- **Source is public**: [cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention) — `decoderXQARunner.cpp`, `xqaParams.h`, `decoderXQAImplJIT/` (NVRTC JIT compile engine, cubin cache).
- Published perf (blog): Llama-70B FP8 on H200, ISL/OSL 128/2048: 1,227 → 2,941 tok/s/GPU (**2.4×**) 1-GPU; 13,232 → 25,300 (1.9×) 8-GPU; TPOT flattens with XQA enabled. Shipped in v0.8. **No Orin/Ampere XQA numbers published.**
- In FlashInfer, XQA is exposed for SM90/SM100/SM120 only ([issue #2522](https://github.com/flashinfer-ai/flashinfer/issues/2522) — open, unassigned). vLLM 0.32.0 does not route SM87 to XQA.
- NVIDIA Orin decode-attention content: none found beyond the [Jetson deployment blog](https://developer.nvidia.com/blog/frontier-reasoning-reaches-the-edge-how-to-deploy-and-optimize-models-on-nvidia-jetson/) (Qwen3.8-27B dense on AGX Orin: NVFP4 + DFlash2 spec decode, "up to 6.28× decode throughput over BF16 on Jetson", no per-kernel decode attention data) and the Jetson AI Lab benchmarking tutorial.

## Q5 — Ampere bandwidth-bound decode levers

- `cp.async.cg` 16 B bypasses L1 (PTX: cp-size 4/8/16; `.cg` shown with 16) — FlashInfer's standard KV load; also `.L2::128B` prefetch variant available and used (`PrefetchMode::kPrefetch`).
- L2 evict hints: supported on all CC 8.x (Table 26); `createpolicy` + `ld.global.L2::evict_first/last` usable on sm_87; FlashInfer decode does **not** apply any evict policy to the KV stream (opportunity).
- `discard.na`: 128-byte granularity, discard-without-writeback semantics (PTX ISA) — usable for scrubbing consumed smem→L2 lines on Ampere (verify arch gate; semantics captured, exact arch sentence not extracted).
- fp8→fp16 cvt on GA10B: **software only** (`FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED` ≥900). e4m3 path ≈ 7-8 ALU ops per 4 values (byte_perm + shift/mask + 2×HMUL2 bias); e5m2 ≈ 2 byte_perms per 4 values. These issue on the same pipes as the FMAs/HMMA — a real per-value tax on sm_87.
- `ld.global.nc`/`__ldg`: alive on Ampere; read-only data cached in unified L1 (guide §10.10/§20.7). For single-use KV streams it buys little; L2 evict-first policy on the KV stream is the higher-value hint.
- smem: Orin is CC 8.7 = **163 KB/block, 164 KB/SM carveout** — NOT the 99-100 KB of SM86/89. FlashInfer's Ampere FA2 decode is budgeted to 2 CTAs/SM ≈ 84 KB cap → NUM_MMA_KV=2 (128 KV rows/iter). A kernel willing to run 1 CTA/SM can stage ~4× the KV per iteration on Orin.

## Q6 — Public Orin decode benchmarks

- [Jetson AI Lab GenAI benchmarking](https://www.jetson-ai-lab.com/tutorials/genai-benchmarking/): vLLM container `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`; example result table: Llama-3.1-8B W4A16 — output throughput 44.19 tok/s @ concurrency 1, TPOT 22.61 ms, TTFT 32.02 ms (labeled "example output"; treat as indicative, not a certified Orin run).
- [NVIDIA dev blog (Sep 2026)](https://developer.nvidia.com/blog/frontier-reasoning-reaches-the-edge-how-to-deploy-and-optimize-models-on-nvidia-jetson/): Qwen3.8-27B (dense 27B) on AGX Orin/Thor with NVFP4 + DFlash2; "up to 6.28× decode throughput speedup over BF16 on Jetson" (chart-based; no absolute Orin tok/s in text).
- [FlashInfer issue #2579](https://github.com/flashinfer-ai/flashinfer/issues/2579) "Architecture sm87 not supported" (Feb 2026, closed not_planned): MLC-LLM on Orin AGX hit the sm87 arch gate; workaround `FLASHINFER_CUDA_ARCH_LIST="8.7"` → "**I notice an increase of 25% in tokens/second with my LLMs**"; follow-up comment: tested on Orin Nano and AGX, "inference increased by around 25% ... kernels for the 8x seem to work pretty well"; maintainer yzh119: "not too much architecture difference between sm87 and other ampere architecture."
- No published FlashInfer-vs-TRT-LLM decode comparison on Orin found.

## Synthesis — what it takes to beat FlashInfer decode on Orin (ranked levers)

Baseline to beat: FA2-TC decode — m16n8k16 HMMA, CTA_TILE_Q=16 (1 Q-warp + 4 KV-warps, 128 thr), NUM_MMA_KV=2 → 128 KV rows/iter, fp8 KV in smem with **per-value in-register software cvt before every HMMA** (sm_87 has no fp8 cvt/mma), ~84 KB smem cap (2 CTA/SM budget), split-KV fills ≤ 2 CTAs/SM × 16 SM = 32 CTAs, separate LSE-merge launch (no PDL on Orin — `device_support_pdl` requires major ≥ 9).

1. **Use the full 163 KB/block smem of sm_87.** FlashInfer's Ampere tuning assumes the 100 KB-class parts; on Orin you can double+ the KV tile per iteration (deeper pipeline / fewer syncs) or run 1 CTA/SM with a much larger KV step.
2. **Kill the fp8 cvt tax.** Convert fp8→bf16 once into smem (repack) with the extra smem, or do a pure-FMA fp32 GEMV where each 16 B fp8 load is converted once and FMAs run at 128/clk/SM with no HMMA dependency chain. Every avoided `fast_dequant` op is issue slots back.
3. **Fill all 16 SMs at small batch with fused split-KV+merge.** FlashInfer's merge is a second kernel (no PDL on Orin → hard serialization); a single-kernel split+combine or last-CTA merge removes a launch-latency floor per decode step.
4. **L2 policy on the KV stream**: `createpolicy`/evict_first on single-use KV lines (supported on 8.x, unused by FlashInfer decode) plus `.L2::128B` prefetch; keep q/scales resident.
5. **Sub-byte KV (fp6/fp4)** — the only lever that scales with L on a 204.8 GB/s part; sm_87 pays software cvt (prmt-based, as in FlashInfer's fp4 path), which the extra smem and FMA budget can hide.
6. **MTP shape (q_len 1..8)**: FA2-TC handles uniform q_len>1 with is_causal; a GEMV-style kernel that verifies 6 q-heads × M tokens per KV row reads each KV byte once per KV row instead of per (tile,row) mma pass — byte-efficiency at hd256 fp8 is decided by how few times each KV byte is streamed.

## Gaps
- TRT-LLM XQA Ampere tile shapes / sm_87 support not read in source (only path + blog).
- Exact PTX arch-gate sentence for `discard` (sm_80+) not captured; semantics captured.
- Per-instruction throughput (PRMT/HMUL2 per SM/clk) on GA10B not sourced from a primary doc.
- No official absolute tok/s for 27B on AGX Orin 64GB found in text form (jetson-ai-lab models page is JS-rendered).
- jetson-ai-lab example table is illustrative ("example output"), not a labeled Orin measurement.
