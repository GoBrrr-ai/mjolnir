# Orin kernel family — the mma.sync decode kernel (sm_87)

This subpackage is the Mjolnir kernel for **Jetson AGX Orin 64GB (`sm_87`,
GA10B Ampere)** — targeted at that one architecture (Thor keeps the
`tcgen05` `t5_*` family; this family never runs on sm_100/sm_110). The
family's only kernel is the tensor-core `mma.m16n8k16` decode kernel.

## What the kernel is

One CTA per `(kv-head, split, batch)`, 128 threads = 4 warps; the M tile
= 64 rows split across the warps (one 16-row mma atom each); the KV rows
stage through `cp.async` (32 rows per stage), dequantize fp8→bf16 in
smem, and ride the reference's `ldmatrix → mma.m16n8k16` path. `M ≤ 8`
(the MTP=7 verify shape), split-KV with chunk-normalized fp32 partials +
the fused LSE combine, causal and non-causal (the MTP drafts see one
fewer future key each). **KV formats: fp8 e4m3 and fp6 e2m3** — the fp6
rows read packed (d·3/4 = 192 bytes for d=256, 4 codes per 3 bytes LE)
through the 64-entry `kvfp6.e2m3_values` LUT; the per-channel scales
fold into Q (`q_mul`) and post-multiply the output (`o_mul`) in the
runner, so the kernel's math is untouched (M1+M2).

## The target to beat

The native sm_87 path in vLLM 0.32.0 is **FlashInfer's tensor-core FA2
decode** — confirmed at both tags: `vllm/v1/attention/backends/flashinfer.py`
constructs `BatchDecodeWithPagedKVCacheWrapper(..., use_tensor_cores=True)`
unconditionally, which routes to the FA2 prefill kernel
(`get_batch_prefill_module("fa2", ...)`); XQA/trtllm-gen are SM90/100/120-only.
Its Ampere shape for our model (packed qo = q_len·g = 6..24 ≤ 16 →
`CTA_TILE_Q = 16`, 1 Q-warp + 4 KV-warps = 128 threads):

* **NUM_MMA_KV = 2 → 128 KV rows per iteration** (fp8: 32 KB K+V smem per
  step; the launcher's 2-CTA/SM budget against the *100 KB-class* Ampere
  parts — Orin's 163 KB is headroom it never uses);
* fp8 KV is converted to fp16 **in registers before every HMMA**
  (`fast_dequant_f8f16x4`, ≈2–4 ALU ops per 4 values — no fp8 cvt or mma
  exists below sm_89) — a per-value tax the mma pays once per loaded
  vector in smem instead;
* Q stays **bf16** on SM87 + fp8 KV ("architectures with only fa2 cannot
  consume FP8 queries");
* split-KV is capped at 2 CTAs/SM (32 CTAs on Orin's 16 SMs), the LSE
  merge is a **second kernel launch**, and PDL does not exist on
  Ampere (`device_support_pdl` requires major ≥ 9) — a hard serial
  launch floor per decode step that Thor does not pay.

Hardware reference (AGX Orin 64GB): 2048 cores / **16 SMs** / 64 3rd-gen
Tensor Cores / 5.3 TFLOPS fp32 / ~42.5 TFLOPS fp16 (derived) /
**204.8 GB/s** LPDDR5 / 4 MB L2 / 163 KB smem per block / 1.3 GHz.

## Measured status

* **The mma kernel**: `mjolnir verify sweep` same-run vs FlashInfer
  FA2-TC across the M/L band (research/runs/orin1/notes/
  mma-kernel-design-2.md).
* **The mma kernel (2026-10-26)**: serving-validated end-to-end — the
  A/B/C matrix byte-identical correct, the spec sweep 14/14 (Mjolnir
  19.9 t/s/req vs FlashInfer 18.9 at c=1, +5.0% aggregate at c=4,
  identical per-position acceptance), fp6 (M1+M2) probe-validated on
  both pool layouts and the image canary (`kernel_test_orin_gates.py`).

## The fp6 e2m3 KV path (M1+M2, landed)

* **M1 (in-kernel)**: the mma reads the packed rows directly — the
  192-byte row moves as 12 × 16-byte cp.async chunks; the codes
  unpack at bit 6i and dequantize through the LUT into the bf16 mma
  operands. The per-head descales ride the accumulator
  (`scale_eff` / the epilogue) exactly like fp8.
* **M2 (per-channel)**: the per-(layer, KV head, channel) scales from
  `mjolnir calibrate` fold into Q pre-launch (`q_mul`) and
  post-multiply the output (`o_mul`) — the kernel untouched
  (`mjolnir/kvfp6.py`, the calibration `kv_scales/e2m3-awq.json`).

## hd128 (shipped, 2026-10-31)

The mma serves **head_size 128 and 256 for fp8** from one D-parameterized
body: the staging, the fragment counts and the PV ranges all derive from
`self.HEAD_DIM`, the constructor takes the head size, and the runner
gates `{128, 256}`, keys the binary on it (the TVM-FFI dims bake it) and
passes it to the split-KV combine — whose elements-per-thread is
`d / NUM_THREADS`. That combine was the whole hd128 bug: it baked
D=256, so at hd128 its second element per thread read and wrote past
the row — masked whenever `max_splits=1` (the write-through path), a
9.2 rel-err otherwise. fp6 stays **hd256-only**: the e2m3 D2 recipe's
byte-lane phase is static per thread and assumes 64 groups per row
(a per-iteration phase would serve hd128's 32).

Verified on the harness (rel err 0.0016–0.0019 vs the fp64 reference):
L 1K–262K, q_len 3/4/5, HND + NHD, split caps 2/4/8; the hd256 fp8 and
fp6 regressions clean. At M=5 / L=64K the hd128 shape graphs 600 µs vs
hd256's 967 µs (1.61× — an hd128 model streams half the KV bytes), and
the kernel drops to 128 regs / 27 KB smem (4 CTAs/SM by registers) from
hd256's 236 regs / 83 KB (2 CTAs/SM).

## Not portable from Thor (deliberate)

* `tcgen05 kind::f8f6f4` (the Thor fp6 read) — Ampere has no such MMA;
  fp6 read here is unpack + `mma.sync`.
* PDL / programmatic dependent launch — sm_90+.
* fp8 lm_head (`MJOLNIR_LM_HEAD=fp8`) — no fp8 tensor cores on Ampere;
  the Triton fp8 GEMM would be compute-bound. Orin keeps a bf16 lm_head.
* The Thor-tuned ring depths / split plans — retuned for 16 SMs,
  100 KiB smem and LPDDR5x latency; the sm_110 numbers do not transfer.
