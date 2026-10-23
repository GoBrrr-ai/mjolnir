# Why the GEMV FMA cannot win on Orin — and what can

Date: 2026-10-23. Evidence: ncu full-section profiles of the Mjolnir GEMV
(`t5_fp8-hnd-L4096.raw.csv`, run 20261023-135335) and FlashInfer's decode
kernel (same GPU, same replay conditions), plus the committed events-timed
sweep (20261023-095858). Board: AGX Orin 64GB, sm_87 (16 SMs × 128 cores,
~1.3 GHz, 204.8 GB/s LPDDR5, 4 MB L2). Shape: b=1, M=1, GQA 24:4, hd256,
fp8 KV, L=4096.

## The head-to-head (ncu replay, base clocks, cold L2)

| metric | FlashInfer (NHD) | Mjolnir GEMV (HND) |
|---|---|---|
| duration | 152–210 µs | ~1.0–1.75 ms |
| Memory throughput | **52%** | **7.9%** |
| Compute (SM) throughput | 34% | **59.2%** |
| Achieved occupancy | 16.4% | 14.9% |
| grid × block | 32 × 128 | 64 × 96 |
| tensor-op instructions | HMMA | **zero** |

The occupancy is nearly IDENTICAL (~16%). The difference is the instruction
mix: FlashInfer keeps the memory pipe at 52% while the tensor cores absorb
the math; the GEMV pins the SM at 59% (issue-bound) while the memory starves
at 8%.

## The GEMV's pipe breakdown (ncu, per-SMSP)

- ALU pipe: **59.6% of peak** — the top pipe is INTEGER/address work
- FMA pipe: 20.6% — the actual math
- LSU: 4.4%; tensor (HMMA/IMMA): 0
- `sm__inst_executed`: 51.5% of peak sustained
- The stalls: math_pipe_throttle 0.35/issue, barrier 0.60, wait 0.38 —
  the scalar chain is issue-throttled and the cp.async barriers bite

The address/loop arithmetic alone eats ~3× the issue budget of the math.
Each KV row costs 16 FFMA + the address increments per thread; the Ampere's
scalar fp32 path retires 2 FLOP per issued instruction, so the decode's
FLOP:byte ratio at hd256 (≈24 FLOP/byte) lands the kernel on the
scalar-issue roofline: ~8.4 MB at 4K ⇒ the measured ~19–22 GB/s, flat.

## The flat-bandwidth proof (events-timed, boost clocks)

Effective KV bandwidth of the GEMV vs L (committed sweep): 19, 19, 20, 21,
22, 22, 22 GB/s for 4K→256K. FlashInfer: 30 → 156 GB/s (76% of peak at
128K). The GEMV's ceiling is structural — it does not rise with more work
or more parallelism.

## The split-scaling negative result (measured 2026-10-23)

`--max_splits` at L=4096 (kernel-only medians): ns=16 → 453 µs (19 GB/s),
ns=32 → 466, ns=128 → 620 (13.5), ns=512 → 1424 (5.9). More CTAs of the
same scalar kernel make it WORSE — parallelism is not the lever; the
instruction mix is. The Alg.1 default (ns=16) is already optimal for this
kernel.

## Conclusion

The GEMV FMA is structurally wrong for Ampere at hd256/GQA decode: the
scalar fp32 issue roofline caps it at ~20 GB/s regardless of scheduling,
while the tensor cores sit idle and the DRAM starves. FlashInfer wins NOT
because of occupancy (theirs matches ours) but because MMA instructions
move 4096 FLOP per issue slot and its loads are wide, smem-staged, and
cp.async-bulk — the memory pipe stays fed.

## The fix (ranked)

1. **MMA-based decode core** (m16n8k16 bf16): QK^T with the 6 GQA query
   heads padded into the M dimension (the padding waste is irrelevant —
   the MMA math is ~0.6 µs of the budget), PV likewise; KV staged through
   smem (cp.async, 128B-aligned), converted e4m3/e2m3→bf16 in smem or
   registers before the MMA. The scalar-issue roofline disappears; the
   kernel enters the bandwidth-bound regime where Mjolnir's fp6 KV byte
   win finally converts (192 MB at 128K fp6 vs 256 MB fp8).
2. **Dequant-in-smem for e2m3**: the LUT/bit-math unpack to bf16 tiles
   (the conversion cost ~5 µs at 4K — negligible), then the standard MMA.
   The format win survives the rewrite; the kernel-side unpack is already
   written (the GEMV's e2m3 path) and re-targets to tile granularity.
3. **Keep the plan/merge**: the Alg.1 split planner and the torch-op LSE
   combine are family-independent; the combine's ~0.2% cost is unchanged.
4. The expected floor: fp8 KV at 4K ≈ 41 µs (DRAM) + latency ⇒ ~100–150 µs
   (parity with FI); fp6 at 128K ⇒ ~1.2 ms vs FI's 1.64 ms — the 1.3× byte
   win converts only after the issue roofline is gone.
5. The int8-IMMA alternative (the e4m3→int8 rescale, mma.s8) exists but
   costs accuracy and adds a rescale pass; the bf16-MMA path is simpler
   and the math is free. Only revisit if the cvt traffic shows up in ncu.

## The tensor-core roofline check (why the MMA suffices at every L)

sm_87 bf16 tensor cores: ~512 FLOP/cycle/SM ⇒ 16 SMs × 512 × 1.3 GHz ≈
10.6 TFLOP/s. Decode FLOPs per step (GQA 24:4, hd256): 2 × 2 × L × 24 ×
256 — at 128K: 3.2 GFLOP ⇒ ~0.3 ms on TC, while the KV load (192 MB fp6 /
256 MB fp8) at the achievable ~150 GB/s is 1.3–1.7 ms. The math hides
under the loads at every L — the kernel stays memory-bound, which is the
regime where Mjolnir's fp6 byte win converts. The scalar path's roofline
for the same FLOPs: 3.2 GFLOP / 5.3 TFLOP/s = 0.6 ms — our measured 12.9
ms at 128K is 21× that: the per-element addressing (the measured ALU 3×
the math), the barriers, and the issue throttling, all of which the MMA
removes.

## The NVIDIA-examples findings (web survey, 2026-10-23)

- **FlashInfer ships BOTH paths, and they are different kernels.** Its GEMV
  (`decode.cuh`) is structurally OURS (the per-token 512B staging, bdx=16
  threads/row, bdy=6, 128 threads launched / 96 active) — and it is NOT
  what vLLM 0.32.0 runs on the Orin. The serving path is the TC kernel:
  `FA2DetermineCtaTileQ → CTA_TILE_Q=16` (the 6 q-heads occupy 6 of 16
  M-rows), a 1×4 warp layout (128 threads), the fp8 KV staged in smem
  with the 128B swizzle and converted per-fragment by
  `fast_dequant_f8f16x4` before every `mma.m16n8k16` (`prefill.cuh`,
  `vec_dtypes.cuh`). Our measured FI numbers (52% memory, 34% SM) are
  THIS kernel.
- **The geometry the fast kernels share**: ≥64 KV rows per smem stage
  (FA2 kBlockN=64 for hd256; FlashInfer TC 128 rows/iter; llama.cpp
  32–64 per MMA step with 128 double-buffered; XQA ≤128-token blocks) —
  our GEMV stages ONE token (512B) per stage: the latency-coverage
  failure. The 16-byte `cp.async.cg` everywhere. 128–256 threads,
  ≥2 CTAs/SM, the split-KV tuned to 2–8 waves (the XQA:
  `__launch_bounds__(256, 4)`; the FA2 heuristic: no split if the grid
  fills 0.8×SMs, else the wave-efficiency ≥0.85, max 128; the XQA
  planner: the historyLength/128 start, capped by sqrt(8×blocks), tuned
  toward 8 waves).
- **GQA packing**: the 6 q-heads → the M tile (the FlashInfer 16, the
  llama.cpp 8) — the padding rows are zero-work, no extra KV traffic.
  The anti-pattern is FA2-Ampere's per-q-head CTAs (the 6× logical KV
  re-streaming — survivable on the A100's 40MB L2, fatal on the Orin's
  4MB).
- **The merge**: the FlashInfer's is a second launch, and PDL does not
  exist below sm_90 — on the Orin it is hard serialization per step.
  Keep the separate merge (ours, ~0.2%) but count the launch gap; the
  fused-merge experiment on the THOR failed (0.56× at 4K) — retest only
  with the MMA kernel's cheaper epilogue.
- **The memory system**: no ECC on the AGX Orin 64GB devkit (the
  Industrial-only feature); the sustained STREAM-class = 60–80% of
  204.8 = **120–165 GB/s**; the decode writes only O (the asymmetry is
  second-order). The L2 hints (evict_first on the KV stream,
  evict_last on Q/scales, the 128B prefetch) are available CC 8.0+ and
  FlashInfer's decode sets NO policy — a free gap.
- **The methodology correction**: ncu's `--clock-control` is
  unsupported on GA10b (Orin) — it silently no-ops; our profiles ran
  under the box's jetson_clocks lock (active since the setup) — the
  ratios are valid, and the correct recipe is jetson_clocks +
  `--clock-control none`. The XQA Ampere inner loop is closed (cubins);
  the llama.cpp/FA2/FlashInfer sources above are the open references.
- **The TC roofline correction**: the sm_87 tensor figure is 1024–2048
  dense bf16 FLOP/clk/SM (the derived 21–43 TFLOPS at 1.3 GHz) — either
  way the 3.2 GFLOP of 128K decode hides under the 1.3–1.7 ms of KV
  loads; the conclusion is unchanged, the margin larger.

## The consolidated plan

**Phase 1 — the MMA decode core** (the rewrite; `mjolnir/arch/orin/`
gains an mma kernel beside the GEMV, the harness/verify machinery is
already arch-agnostic):
- CTA = one (kv-head, chunk); 128 threads, 1 Q-warp + 4 KV-warps (the
  FlashInfer TC layout) or 2 CTAs/SM (the llama.cpp geometry)
- Q: the 6 heads padded into the M=16 tile; the smem-staged once per CTA
- KV: the 64–128 rows/stage smem ring (3–4 stages, ~100 KB), the 16B
  cp.async.cg, the 128B swizzle
- The math: QK^T and PV as `mma.m16n8k16` bf16; the fp8→bf16 per
  fragment in the consumer stage (`fast_dequant_f8f16x4`-style); the
  e2m3 via the LUT unpack to bf16 tiles (the fragment layout per
  BitDecoding — Phase 2)
- The scales: the K per-channel folded into Q (S = (Q∘s)·K̂ᵀ), the
  per-token V on the PV epilogue — the SageAttention trick (the scales
  outside the mma) keeps the dequant off the critical path
- The split planner: the byte-equalized stream-K-ish chunking with the
  partials ≈ 2× the SMs, the existing torch-op LSE combine
- The expected: the fp8 4K from 453 µs → the ~100–150 µs class (the
  parity to 1.5× vs the FI), and the bandwidth-bound regime at the long
  L where the fp6 byte win converts (the 1.3× structural)

**Phase 2 — e2m3 in the fragment layout** (the BitDecoding mechanism):
the codes packed so the LUT/dequant lands in the MMA registers, no
reshuffle. The 10/10 correctness + the accuracy data (0.0196 rel L2)
carry over.

**Phase 3 — the polish**: the L2 evict_first/prefetch hints on the KV
stream, the merge-launch placement, the per-L split tuning (the FA2
heuristic as the starting point), the M-policy (the GEMV returns only
if the issue-efficient M=1 case wins the profiled A/B).

## What would NOT help

- More splits / more CTAs (measured: worse — for THIS kernel; the stream-K
  recipe applies to the MMA kernel, where the byte-equalized wave is the goal)
- Deeper cp.async rings (the memory pipe is at 8% — nothing to deepen)
- Wider per-thread vectors (already 16B; the bottleneck is issue, not load width)
- Occupancy tuning (the FI wins at the SAME 16%)

## The literature (arxiv survey, 2026-10-23)

The decisive claims, with IDs:

- **BitDecoding (2503.18773)** — the paper that claims TC decode beats
  CUDA-core GEMV decode AT BATCH 1 on Ampere: "existing systems decode
  inefficiently by relying solely on CUDA cores, underutilizing Tensor
  Cores"; GQA's key-sharing enables the TC dot products. 4.8× over fp16
  FlashDecoding-v2 on Ampere; 3× single-batch 128K decode. Mechanism:
  pack the low-bit codes in the ldmatrix fragment layout so the dequant
  lands directly in the TC registers (no reshuffle), the warp-level
  dequant + the software-pipelined dequant→mma, XOR-swizzled smem.
- **FlashDecoding++ (2311.01282)** — the crossover: at M=1 the CUDA-core
  GEMV wins by ~22% (the TC = 82% of the GEMV) — but that is the
  ISSUE-EFFICIENT GEMV, not ours; at M=4 the TC is ~2× the GEMV. Our
  MTP verify shape (M=4..8) is squarely on the TC side. Also: pad Q to
  the mma M-tile (8/16), never to 64 (>87% wasted work). And the
  asynchronized softmax (the declared max) removes the cross-CTA sync.
- **QServe (2405.04532)** — the cost model: 1 CUDA-core op ≈ 50 INT4-TC
  ops on the A100-class tensor core; a dequant-on-CUDA-cores decode goes
  1.2× SLOWER unless the kernel stays memory-bound with the in-register
  dequant. Our measured failure mode exactly.
- **LeanAttention (2405.10480)** — the split recipe for the MMA kernel:
  stream-K over the KV tokens (equalize the bytes/SM, the partials ≈ the
  SM count, the catch-up tree merge), the occupancy ≈ 100% at every L;
  the byte-equivalent tile for d=256 fp8: the 64–128 rows/stage — a
  3-stage ring of the 32-row stages fits the Orin's 100 KB smem.
- **FlashInfer (2501.01005) / DeFT (2404.00242) / Tree Attention
  (2408.04093)** — the merge is launch-bound: fewer partials, tree
  merge, persistent CTAs.
- **The quantized-decode convergence** (KIVI 2402.02750, KVQuant
  2401.18079 ~1.7× vs the fp16 GEMV decode, SKVQ 2405.06219, ZipCache
  2405.14256, SageAttention2 2411.10958): dequant fused into the matmul,
  the K per-channel scales folded into Q (S = (Q∘s)·K̂ᵀ), the per-token
  V scales on the PV epilogue, the SageAttention trick: the scales
  OUTSIDE the mma at the per-thread granularity = the zero-dequant
  overhead. On sm_87 (no fp8/fp6 TC): the e4m3/e2m3 → bf16 cvt in the
  consumer stage of the ring, then the standard m16n8k16 — exactly our
  plan; BitDecoding's fragment-layout packing is directly implementable
  on our HND pages.
- **The arithmetic-intensity check (2507.15465)**: the GQA decode
  attention ≈ 1–2 Op/B ≪ the ridge point → the kernel's only job is to
  be a byte pump; the FMA-vs-MMA matters only for keeping the issue rate
  OFF the load path — our exact failure mode, named.
- The negatives (verified): no KIVAR, no FP8-LC; no arXiv paper reports
  the kernel-level achieved GB/s for the decode attention on the Orin —
  our measurements are novel.

## The revised plan (the literature folded in)

1. **Rewrite the decode core on mma.m16n8k16 bf16** (BitDecoding +
   SageAttention2): the 6 GQA q-heads padded to the M=8 tile (never 64),
   the KV staged through smem, the codes unpacked/dequantized in the
   consumer stage (the fragment layout for the e2m3), the K-scales
   folded into Q, the V-scales on the PV epilogue. Expected 2–4× on top
   of the scheduling fix — and it is the PRECONDITION: our measured
   issue roofline caps the GEMV at ~20 GB/s at any grid.
2. **Stream-K token-parallel split-KV for the new kernel** (LeanAttention):
   equalize the bytes/SM, ~2 CTAs/SM × 16 SMs, the tree LSE merge. For
   the GEMV this was measured-negative; for the MMA kernel it is the
   occupancy recipe. 3–6×.
3. **A 3-stage cp.async ring of the 32-row stages** (~100 KB smem) — the
   byte budget from the LeanTile geometry; the HND fp8 rows are 256B
   contiguous (no Thor-style 64B-request wall on the Orin). 1.5–2×.
4. **Async/declared-max softmax** (FlashDecoding++): the split partials
   independent, no cross-CTA sync; the tree merge (our torch-op combine
   stays, the partial count stays ≈ the SM count).
5. Keep the M-policy honest: the M=1 GEMV could return as a special case
   ONLY if the profiled high-occupancy version wins — the FlashDecoding++
   crossover says it needs to be issue-efficient first, which ours is not.

