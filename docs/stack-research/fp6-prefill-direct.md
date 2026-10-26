# fp6 KV direct-read prefill on sm_87 — research and decision (2026-10-26)

The question: can the Orin (sm_87) prefill fallback read packed fp6 (e2m3) KV
**directly — no scratchpad, no per-step tax** — and if not, is there a better
all-platform KV format to replace it with? Researched via two parallel studies
(FlashInfer internals at our pinned v0.7.0.post1; the low-bit-KV literature).
Full reports: `task/research/fp6-prefill-direct-serve-sm87/summary.md`,
`task/research/arxiv-fp6-kv-prefill-dequant.md`.

## Verdict

**Keep fp6 everywhere. Kill the scratchpad by teaching FlashInfer's Ampere
prefill to unpack fp6 in-kernel.** No format replacement; NVFP4-KV parks as a
future additional format.

## Findings

1. **The scratchpad is a system hack, not a hardware necessity.** sm_87 has no
   fp8 tensor cores (they start at sm_89), so *every* quantized-KV path on
   Ampere dequantizes to fp16/bf16 in-kernel. FlashInfer already does this for
   fp8 KV on sm_80: register dequant per fragment at `CTA_TILE_Q=16`
   (`fast_dequant_f8f16x4`), an smem staging repack at larger tiles
   (`repack_fp8_tile_to_bf16`, PR #3485). Their own words (PR #4746): *"FP8 KV
   — which also dequantizes — is essentially free on the same hardware."*
2. **Sub-byte KV on Ampere is already shipped in our exact stack.** NVFP4 KV
   (software unpack + smem repack) runs on pre-SM100 in the pinned tag: PR
   #4746 + #4769 took sm_80 NVFP4 prefill from 6.8–8.2× fp8 down to
   **1.900 vs 1.858 ms** (≈ parity; prefill b1 4096², hd128). vLLM 0.32.0
   wires `_nvfp4_kv_on_fa2()` for sm_80/90/120. The machinery fp6 needs is
   byte-verified in our image.
3. **Nobody ships fp6 KV prefill** — `abs:"FP6" AND abs:"KV cache"` → 0 arXiv
   hits; every KV-quant paper (KIVI, KVQuant, ZipCache, QServe, Atom) fuses
   dequant into *decode* and prefills on fp16. The transferable recipes exist
   though: FP6-LLM (2401.14112, bit-plane register stitch, dequant fully
   hidden under mma on A100), SageAttention (native int8-K Ampere prefill,
   1.37× on A100), Marlin (≈free register dequant). The failure mode is
   measured: naive unpacks cost 1.1–1.2× (QServe); a function-local LUT
   nvcc places in local memory cost FlashInfer 3.6× until moved to
   prmt-indexed registers (PR #4746).
4. **Per-channel scales need zero kernel changes.** The produce stage bakes
   `code·s_c` into e4m3 bytes (fits e4m3's ±448 up to the calibrated amax —
   the exact numerics our validated fp8-scratch path uses today). No Q
   folding, no epilogue patch, scales at 1.
5. **No better replacement**: NVFP4-KV is lossier, needs per-16 block-scale
   calibration, and the Thor kernel doesn't read fp4 yet (the existing "fp4
   with finer scales" open item). It would ride the same FlashInfer seam
   later. fp2/int2: nothing. **And the fp6 configs are the best servers on
   this board** — the 20261026-175121 spec sweep: `AWQ_MJOLNIR_E2M3_VT64K`
   25.8 t/s vs official 18.4 at k=5 (+40%), acceptance preserved.

## The recipe (path A — patch the pinned FlashInfer)

Structure facts (byte-verified at v0.7.0.post1): the Ampere prefill is
FA2-derived CUDA, 2-stage cp.async K/V pipeline, `mma.sync.m16n8k16`,
CTA_TILE_Q ∈ {16, 64, 128}. The fp4 path already proves the pattern
"load packed → repack in smem → feed the 16-bit path". The refined design —
**in-place expansion, so the entire downstream is the untouched fp8 path**:

1. **`flashinfer/jit/utils.py`** — a KV dtype entry for packed fp6. `uint8` is
   taken (`__nv_fp4x2_e2m1`); route our dtype through an explicit string
   (`__nv_fp6_e2m3_sv`) on the customize seam rather than the torch-dtype map.
2. **`include/flashinfer/attention/prefill.cuh`** — two pieces only:
   - `produce_kv` fp6 branch: per 16-code smem slot, cp.async **16 B with
     src-size 12** (PTX zero-fills the remainder — the fp4 `load_64b_async`
     precedent) — row width is hd·0.75 = 192 B for hd256, so 16-B chunks never
     cross rows; GMEM pointer advance per column 12 B instead of 16 (0.75×
     the bytes). smem cost identical to fp8.
   - an **in-place expand pass** at exactly the hook where the fp4
     `repack_kv_tile_to_16b` sits (after `wait_group` + `block.sync`, before
     the fragment loads — prefill.cuh:2589/2650): each thread expands its own
     slots — read the slot's 12 packed bytes to registers, write 16 e4m3
     bytes (64-entry e2m3→e4m3 LUT, prmt-indexed in registers — never a
     function-local array, the measured 3.6× trap) times `s_c` into the same
     16 B slot. Same thread↔slot mapping as the produce loop; no cross-thread
     hazard; the repack smem buffer is not even needed.
   - **Why this is enough**: for 1-byte KV the kernel dequantizes per
     *fragment* (`ldmatrix → frag_layout_swizzle_16b_to_8b →
     vec_cast<DTypeQ, DTypeKV>::cast<8>`, compute_qk) — from the moment the
     smem tile holds e4m3 bytes, both tile paths (`CTA_TILE_Q=16` register
     dequant; large-tile `repack_fp8_tile_to_bf16`) run unchanged. **No
     compute_qk changes at all.**
3. **Scales tensor**: `const half* kv_scale` ([num_kv_heads, head_dim], the
   existing per-(layer, kv-head, channel) calibration) passed as an additional
   kernel param via the `gen_customize_batch_prefill_module` seam; the expand
   pass reads `s_c` for its slot's channel column.
4. **`flashinfer/prefill.py`** — allow the dtype for batch prefill on sm_80+
   (sentinel-gated).
5. **vLLM side (`src/mjolnir`, probe-gated `MJOLNIR_ATTN_FP6_DIRECT`)**: when
   fp6 + fallback + sm_80-family, skip the scratch build and hand the packed
   pool + scale tensor to the patched prefill. Thor (FA4 fallback) keeps the
   scratch until an FA4-side equivalent exists — Thor's prefill tax is a
   separate follow-up.

Expected: fp6 prefill ≈ fp8 ±5% (per the fp4-on-sm80 data), **growing win with
context length** (0.75× the KV bytes in a bandwidth-bound regime), the
per-step sync and the no-graphs restriction gone, and the
`MJOLNIR_ATTN_DRAFT_T5=0`-on-fp6 refusal revisitable.

## Derived facts (read from the extracted image sources, /tmp/opencode/fi-src/)

- `upcast_size<T>() = sizeof(b128_t)/sizeof(T)` (permuted_smem.cuh:43) — 16
  elements per 16-B smem column for any 1-byte KV type. The produce loop's
  per-j gmem advance is `(pack_ratio) * upcast_size`: fp8 `8*16=128` codes,
  fp4 `4*16=64` bytes, **fp6 `6*16=96` bytes** — each j-chunk covers 128
  codes; the j-loop bound `NUM_MMA_D/(8/sizeof(DTypeKV))` is unchanged for a
  1-byte fp6 struct (`sizeof == 1`), so the smem side needs no edits.
- The cp.async src-size is a **runtime operand** in the existing
  `pred_load_128b` (`"r"(src_in_bytes)` with zero-fill) — a 12-byte partial
  slot load is a copy of that asm with a constant 12.
- **The scales plumbing already exists**: `batch_prefill` ships
  `kv_cache_sf` (packed-uint8 NVFP4 KV + per-block UE4M3 scale tensors,
  fa2-only) — prefill.py:1385. The fp6 scale tensor rides the same
  extra-tensor-param path with a per-channel layout instead of per-16 blocks.
- fp4 keeps its packed codes in smem and dequantizes per-fragment
  (`compute_qk`: `ldmatrix → frag_layout_swizzle_16b_to_8b/4b →
  vec_cast<DTypeQ, DTypeKV>::cast<8>`) — so a 1-byte fp6 type whose
  `vec_cast` delegates to the e4m3 conversion makes the smem tile, post
  in-place expansion, byte-identical to fp8 with **zero compute_qk changes**.

## Status

- [x] Research (this doc)
- [x] Sources extracted from the image (`/tmp/opencode/fi-src/`); all hunks
      authored → `docker/vllm-thor/patches/fi-fp6-kv-ampere-prefill.patch`
      (6 files: vec_dtypes, cp_async, permuted_smem, prefill, modules.py,
      jit_utils.py), registered in `FLASHINFER_PATCH_ORDER` +
      `FLASHINFER_FIXES` sentinels + `PATCHES.md`
- [x] Compile-validated CPU-side: the patch applies to a pristine tree, and
      the fp6 expand instantiates clean for sm_87 (`nvcc -c`, exit 0) —
      `/tmp/opencode/fi-test/`
- [ ] Image build of the side tag `v0.32.0-fp6direct` (in flight) + the
      unpack canary in `image gates`
- [ ] vLLM-side wiring (`MJOLNIR_ATTN_FP6_DIRECT`), probe-gated
- [ ] Runtime: greedy parity vs fp8, then the same-run A/B
      (fp6-direct vs fp8-scratch vs fp8) across L
- [ ] Default flip decision per the bench discipline (a default flips only
      with a same-run A/B win across L)
