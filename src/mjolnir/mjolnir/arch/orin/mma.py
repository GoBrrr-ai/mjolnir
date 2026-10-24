"""The sm_87 MMA decode kernel — Phase 1 (M1): fp8 KV, bf16 Q, q_len <= 8
(the MTP=7 verify shape: 7 drafts + the real token = 8 query tokens).

The design lives in research/runs/orin1/notes/mma-kernel-design.md; the
evidence for why the GEMV's scalar-issue roofline caps it at ~20 GB/s
lives in gemv-vs-mma-bottleneck.md. The structure follows NVIDIA's
Ampere FlashAttention-2 DSL example (research/runs/orin1/sources/
ampere-fa2-dsl/fa2_ampere.py):

- CTA = one (kv-head, split, batch); 128 threads = 4 warps
- The M tile = 64 rows: the 4 warps SPLIT THE M (the warp arrangement
  (4,1,1), one 16-row mma atom per warp) — the reference's structure.
  The q_len*6 rows (<= 48) pad into one tile; the per-warp row stats
  are quad-local (no cross-warp reduction, no stats smem).
- The N of QK^T = the 32 stage rows, the FULL tile per warp — so the
  PV's A operand sees the K=16 per mma atom (an N-split warp
  arrangement would halve it and break the PV).
- The KV: 32 rows per stage — the fp8 rows cp.async'd raw (the paged
  gather, the GEMV's make_ptr idiom, the tail clamp), dequantized
  fp8->bf16 IN SMEM, then the reference's ldmatrix -> mma.m16n8k16 path
- The bf16 smem tiles are UNSWIZZLED in M1 (the plain row-major — the
  swizzle is the M2 bank-conflict pass). The V dequantizes IN REGISTERS
  (the even/odd two-pass over the transposed u16 view of the raw fp8,
  the same x4 recipe as the smem path): acc_O splits into the
  even/odd-d halves, re-interleaved through smem at the epilogue. No
  sV16 tile — the freed 16 KB funds a 4th ring pair (3 groups in
  flight).
- The P is written back through smem (reusing the Q tile's space) and
  re-read as the PV's A operand through the standard copy atoms — the
  S-fragment-to-A-operand retile the reference does with
  logical_divide, expressed with atoms instead (the per-thread S
  fragment shape differs between warp arrangements; the atoms are
  geometry-agnostic).
- The causal mask is PER-TOKEN (the MTP drafts see one fewer future
  key each): the row (m, r) attends to the keys gk <= kv_len - q_len +
  m — the GEMV ref_decode contract; the M=1 degenerates to the no-op.
- The descales follow the GEMV's contract: the K scale folds into
  scale_eff (the S side), the V scale divides at the epilogue, the
  LSE = (m*scale_eff + log2 l) * ln2 with the GEMV's exact coords.
- The split-KV partial O/LSE per (split, row) — merged by the existing
  torch combine.

Status: M1 in progress. NOT wired into the default paths; `--impl mma`
exercises it.
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp import LdMatrix8x8x16bOp, MmaF16BF16Op

import cuda.bindings.driver as cuda

from mjolnir.arch.orin.decode import DescaleTensors, GEMVDecodeKernel

# The e4m3 -> bf16 x4 dequant (the FlashInfer/Marlin fast_dequant_f8f16x4,
# bf16 branch — verified bit-exact on sm_87 for normals and zero):
# _DEQ_MASK_LSB keeps the 7 value bits of each packed fp8 half (the C++
# MASK1>>7 is an ARITHMETIC shift on a signed int — the sign extension is
# what makes the mask 0x7F007F00), _DEQ_MASK_SIGN the sign bits, _DEQ_BIAS
# the packed bf16x2 2^120 (= the e4m3->bf16 exponent bias applied through
# one fma; also maps the zero code to 0.0).
_DEQ_MASK_LSB = cutlass.Int32(0x7F007F00)
_DEQ_MASK_SIGN = cutlass.Int32(0x80008000)
_DEQ_BIAS = cutlass.Int32(0x7B807B80)


class MMADecodeKernel:
    """The traced CuTe kernel. The arg surface mirrors the GEMV's
    GEMVDecodeKernel.__call__ so the runner's packing is shared."""

    HEAD_DIM = GEMVDecodeKernel.HEAD_DIM  # 256
    M_TILE = 64  # the 4 warps x 16-row mma atoms; q_len*6 <= 48 pads in
    STAGE_ROWS = 32  # the KV rows per smem stage
    NUM_THREADS = 128  # 4 warps
    MAX_Q_LEN = 8  # the MTP=7 verify shape (7 drafts + the real token)

    def __init__(
        self,
        qhead_per_kvhead: int,
        tile_size_per_bdx: int = 4,  # unused (the GEMV API compat)
        page_size: int = 0,
        causal: bool = True,
        kv_format: str = "fp8",
    ):
        assert kv_format == "fp8", f"the M1 MMA kernel takes fp8 KV only, got {kv_format!r}"
        assert qhead_per_kvhead >= 1
        self.g = qhead_per_kvhead
        self.page_size = page_size
        self.causal = bool(causal)

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None = None,
        mCuSeqlensQ: cute.Tensor | None = None,
        mCuSeqlensK: cute.Tensor | None = None,
        mDescale: DescaleTensors | None = None,
        mCodeLUT: cute.Tensor | None = None,
        mO_partial: cute.Tensor | None = None,
        mLSE_partial: cute.Tensor | None = None,
        mPageTable: cute.Tensor | None = None,
        seqlen_q: cutlass.Constexpr[int] = 1,
        seqlen_k: Int32 = 0,
        softmax_scale_log2: Float32 = 0.0,
        num_splits: Int32 = 1,
        stream: cuda.CUstream = None,
    ):
        assert const_expr(mCodeLUT is None), "e2m3 lands in M5 (fp8 first)"
        assert const_expr(1 <= seqlen_q <= self.MAX_Q_LEN), (
            f"seqlen_q {seqlen_q} outside 1..{self.MAX_Q_LEN} (the MTP=7 verify shape)"
        )
        assert const_expr(mK.element_type.width == 8 and mV.element_type.width == 8), (
            "the M1 MMA kernel takes fp8 KV only"
        )
        assert const_expr(mQ.element_type == cutlass.BFloat16), (
            "the fp8 Q lands with M3 (the pack is bf16-Q only for now)"
        )
        batch_size = mQ.shape[0]
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mDescale,
            mO_partial,
            mLSE_partial,
            mPageTable,
            seqlen_q,
            seqlen_k,
            softmax_scale_log2,
            num_splits,
        ).launch(
            grid=(cute.size(mV.shape[-2]), num_splits, batch_size),
            block=[self.NUM_THREADS, 1, 1],
            stream=stream,
        )

    # ------------------------------------------------------------------
    @cute.jit
    def g2s_stage(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPageTable,
        sK8a: cute.Tensor,
        sK8b: cute.Tensor,
        sK8c: cute.Tensor,
        sK8d: cute.Tensor,
        sK8e: cute.Tensor,
        sV8a: cute.Tensor,
        sV8b: cute.Tensor,
        sV8c: cute.Tensor,
        sV8d: cute.Tensor,
        sV8e: cute.Tensor,
        sK8v: cute.Tensor,
        sV8v: cute.Tensor,
        tidx,
        kv_start,
        row0,
        kv_len_c,
        num_head_kv,
        kv_head,
        batch_idx,
        kv_row0,
        buf,
    ) -> None:
        """One KV stage: gmem -> smem fp8 (32 rows x 256 B = 4 x 16B chunks
        per thread) into buffer `buf` of the double-buffered staging tiles.
        The tail rows CLAMP to the last valid row (the GEMV's trick — no
        branch); the softmax masks the overhang by kv_len_c. The prefetch
        of a not-yet-valid stage clamps row0 the same way: the duplicated
        rows are never read."""
        D = self.HEAD_DIM
        NT = self.NUM_THREADS
        SR = self.STAGE_ROWS
        g2s_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            mK.element_type,
            num_bits_per_copy=128,
        )
        for i in cutlass.range_constexpr(SR * D // NT // 16):
            e16 = tidx + i * NT  # the 16-byte chunk index within the stage
            r = e16 // (D // 16)
            ck = (e16 % (D // 16)) * 16
            gk_row = cutlass.min(kv_start + row0 + r, kv_start + kv_len_c - 1)
            if const_expr(mPageTable is not None):
                page = mPageTable[batch_idx, gk_row // self.page_size]
                kv_base = (page * self.page_size + gk_row % self.page_size) * num_head_kv + kv_head
            else:
                kv_base = (kv_row0 + gk_row) * num_head_kv + kv_head
            k_src = cute.make_tensor(
                cute.make_ptr(
                    mK.element_type,
                    (mK.iterator + kv_base * D + ck).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(16),
            )
            kdst = sK8a.iterator
            vdst = sV8a.iterator
            if buf == 1:
                kdst = sK8b.iterator
                vdst = sV8b.iterator
            if buf == 2:
                kdst = sK8c.iterator
                vdst = sV8c.iterator
            if buf == 3:
                kdst = sK8d.iterator
                vdst = sV8d.iterator
            if buf == 4:
                kdst = sK8e.iterator
                vdst = sV8e.iterator
            k_dst = cute.make_tensor(
                cute.make_ptr(
                    mK.element_type,
                    # The u16-view swizzled offset (the register-path K
                    # dequant ldmatrix reads the view) — the byte offset
                    # is 2x the view element index.
                    (kdst + Int32(2) * cute.crd2idx((r, ck // 2), sK8v.layout)).toint(),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                ),
                cute.make_layout(16),
            )
            v_src = cute.make_tensor(
                cute.make_ptr(
                    mV.element_type,
                    (mV.iterator + kv_base * D + ck).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(16),
            )
            v_dst = cute.make_tensor(
                cute.make_ptr(
                    mV.element_type,
                    # The u16-view swizzled offset (the register-path V
                    # dequant ldmatrix reads the view): view col j holds
                    # the fp8 d {2j, 2j+1} — the byte offset is 2x.
                    (vdst + Int32(2) * cute.crd2idx((r, ck // 2), sV8v.layout)).toint(),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                ),
                cute.make_layout(16),
            )
            cute.copy(g2s_atom, k_src, k_dst)
            cute.copy(g2s_atom, v_src, v_dst)

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mDescale: DescaleTensors | None,
        mO_partial: cute.Tensor | None,
        mLSE_partial: cute.Tensor | None,
        mPageTable: cute.Tensor | None,
        seqlen_q: cutlass.Constexpr[int],
        seqlen_k: Int32,
        softmax_scale_log2: Float32,
        num_splits: Int32,
    ):
        D = self.HEAD_DIM
        MT = self.M_TILE
        SR = self.STAGE_ROWS
        NT = self.NUM_THREADS
        G = self.g

        kv_head = cute.arch.block_idx()[0]
        split_idx = cute.arch.block_idx()[1]
        batch_idx = cute.arch.block_idx()[2]
        tidx = cute.arch.thread_idx()[0]

        is_varlen_q = const_expr(mCuSeqlensQ is not None)
        is_varlen_k = const_expr(mCuSeqlensK is not None)

        if const_expr(is_varlen_q):
            num_head = mQ.shape[1]
            q_row_base = Int32(mCuSeqlensQ[batch_idx])
        else:
            num_head = mQ.shape[2]
            q_row_base = batch_idx * seqlen_q
        if const_expr(is_varlen_k):
            kv_row0 = Int32(mCuSeqlensK[batch_idx])
            kv_len = Int32(mCuSeqlensK[batch_idx + 1]) - kv_row0
        else:
            kv_row0 = batch_idx * seqlen_k
            kv_len = seqlen_k
        num_head_kv = mV.shape[-2]

        kv_chunk = (kv_len + num_splits - 1) // num_splits
        kv_start = split_idx * kv_chunk
        kv_len_c = cutlass.min(kv_chunk, cutlass.max(Int32(0), kv_len - kv_start))

        # The per-(batch, kv_head) descales (the GEMV's exact semantics).
        qk_descale = Float32(1.0)
        v_descale = Float32(1.0)
        if const_expr(mDescale is not None):
            if const_expr(mDescale.q_descale is not None):
                qk_descale = qk_descale * Float32(mDescale.q_descale[batch_idx, kv_head])
            if const_expr(mDescale.k_descale is not None):
                qk_descale = qk_descale * Float32(mDescale.k_descale[batch_idx, kv_head])
            if const_expr(mDescale.v_descale is not None):
                v_descale = Float32(mDescale.v_descale[batch_idx, kv_head])
        scale_eff = softmax_scale_log2 * qk_descale

        # The O/LSE flat row offset (the GEMV's exact math). The head*D term
        # is added per-row in the epilogue with the global head index — no
        # kv_head term here, it would double-count.
        o_off = q_row_base * num_head * D
        if const_expr(mO_partial is not None):
            if const_expr(is_varlen_q):
                split_stride = mO_partial.shape[1] * num_head * D
            else:
                split_stride = mO_partial.shape[1] * mO_partial.shape[2] * num_head * D
            o_off = o_off + split_idx * split_stride

        # ---- smem ---------------------------------------------------------
        # The bf16 operand tiles carry the FA2-style XOR swizzle (the
        # (8, 64)-element atom, Swizzle<3,3,3>): plain 512 B row-major rows
        # put every ldmatrix 8-row group in the same banks — an 8-way
        # conflict per ldmatrix (the M1 profile's dominant stall:
        # short_scoreboard 8.87 of 13.09 cycles/instr). The fp8 staging
        # tiles stay flat (the dequant reads them by the 16 B vector).
        bf16_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, 64), stride=(64, 1)),
        )
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(cutlass.BFloat16, cute.tile_to_shape(bf16_atom, (MT, D), (0, 1)), byte_alignment=16)
        # The fp8 staging tiles, DOUBLE-BUFFERED with the second buffers
        # ALIASED into sQ: the Q tile is dead once its fragments are
        # hoisted (they live in registers from the fill+ldsm before the
        # stage loop), so its 32 KB host the prefetch buffers. Total stays
        # 80 KB -> 2 CTAs/SM: the ring hides the g2s DRAM latency (the #1
        # stall at locked clocks) without paying the occupancy.
        sK8 = smem.allocate_tensor(mK.element_type, cute.make_layout(SR * D), byte_alignment=16)
        sV8 = smem.allocate_tensor(mV.element_type, cute.make_layout(SR * D), byte_alignment=16)
        # The 4th pair: the V register path freed sV16 — 4 buffers with
        # wait_group(3) keeps 3 groups in flight (48 KB/CTA = 96 KB/SM).
        sK8d = smem.allocate_tensor(mK.element_type, cute.make_layout(SR * D), byte_alignment=16)
        sV8d = smem.allocate_tensor(mV.element_type, cute.make_layout(SR * D), byte_alignment=16)
        # The 5th pair: the K register path freed sK16 — 5 buffers with
        # wait_group(4) keeps 4 groups in flight (64 KB/CTA = 128 KB/SM,
        # FlashInfer's exact in-flight).
        sK8e = smem.allocate_tensor(mK.element_type, cute.make_layout(SR * D), byte_alignment=16)
        sV8e = smem.allocate_tensor(mV.element_type, cute.make_layout(SR * D), byte_alignment=16)
        # the offsets are in BF16 ELEMENTS (SR*D bytes = SR*D//2 elements):
        # the four 8 KB buffers tile sQ's 32 KB exactly.
        sK8b = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=cutlass.Float8E4M3FN), cute.make_layout(SR * D))
        sV8b = cute.make_tensor(
            cute.recast_ptr(sQ.iterator + SR * D // 2, dtype=cutlass.Float8E4M3FN), cute.make_layout(SR * D)
        )
        sK8c = cute.make_tensor(
            cute.recast_ptr(sQ.iterator + SR * D, dtype=cutlass.Float8E4M3FN), cute.make_layout(SR * D)
        )
        sV8c = cute.make_tensor(
            cute.recast_ptr(sQ.iterator + 3 * SR * D // 2, dtype=cutlass.Float8E4M3FN), cute.make_layout(SR * D)
        )
        # The K register path: the (SR, D/2) bf16 u16 VIEWS of the raw fp8
        # buffers (each u16 = 2 consecutive fp8 of one key) — the ldmatrix
        # source for the in-register even/odd K dequant. No sK16 tile.
        sK8v = cute.make_tensor(
            cute.recast_ptr(sK8.iterator, dtype=cutlass.BFloat16),
            cute.tile_to_shape(bf16_atom, (SR, D // 2), (0, 1)),
        )
        sK8bv = cute.make_tensor(
            cute.recast_ptr(sK8b.iterator, dtype=cutlass.BFloat16),
            cute.tile_to_shape(bf16_atom, (SR, D // 2), (0, 1)),
        )
        sK8cv = cute.make_tensor(
            cute.recast_ptr(sK8c.iterator, dtype=cutlass.BFloat16),
            cute.tile_to_shape(bf16_atom, (SR, D // 2), (0, 1)),
        )
        sK8dv = cute.make_tensor(
            cute.recast_ptr(sK8d.iterator, dtype=cutlass.BFloat16),
            cute.tile_to_shape(bf16_atom, (SR, D // 2), (0, 1)),
        )
        sK8ev = cute.make_tensor(
            cute.recast_ptr(sK8e.iterator, dtype=cutlass.BFloat16),
            cute.tile_to_shape(bf16_atom, (SR, D // 2), (0, 1)),
        )
        # The V register path: the (SR, D/2) bf16 u16 VIEWS of the raw fp8
        # buffers — each u16 = the fp8 d {2j, 2j+1} of one key. The
        # transposed compositions (the reference's sVt pattern) feed the
        # transpose=True B-atom; a directly-allocated transposed layout
        # loses the 16B alignment the ldsm needs.
        sV8v = cute.make_tensor(
            cute.recast_ptr(sV8.iterator, dtype=cutlass.BFloat16),
            cute.tile_to_shape(bf16_atom, (SR, D // 2), (0, 1)),
        )
        sV8bv = cute.make_tensor(
            cute.recast_ptr(sV8b.iterator, dtype=cutlass.BFloat16),
            cute.tile_to_shape(bf16_atom, (SR, D // 2), (0, 1)),
        )
        sV8cv = cute.make_tensor(
            cute.recast_ptr(sV8c.iterator, dtype=cutlass.BFloat16),
            cute.tile_to_shape(bf16_atom, (SR, D // 2), (0, 1)),
        )
        sV8dv = cute.make_tensor(
            cute.recast_ptr(sV8d.iterator, dtype=cutlass.BFloat16),
            cute.tile_to_shape(bf16_atom, (SR, D // 2), (0, 1)),
        )
        sV8ev = cute.make_tensor(
            cute.recast_ptr(sV8e.iterator, dtype=cutlass.BFloat16),
            cute.tile_to_shape(bf16_atom, (SR, D // 2), (0, 1)),
        )
        sV8vt = cute.composition(sV8v, cute.make_layout((D // 2, SR), stride=(SR, 1)))
        sV8bvt = cute.composition(sV8bv, cute.make_layout((D // 2, SR), stride=(SR, 1)))
        sV8cvt = cute.composition(sV8cv, cute.make_layout((D // 2, SR), stride=(SR, 1)))
        sV8dvt = cute.composition(sV8dv, cute.make_layout((D // 2, SR), stride=(SR, 1)))
        sV8evt = cute.composition(sV8ev, cute.make_layout((D // 2, SR), stride=(SR, 1)))

        # ---- stage the Q once: the q_rows real rows, the zero padding ---
        # mQ: (b, M, h_q, d) bf16; the M row (token m, head-in-group r) at
        # the (batch*seqlen_q + m, kv_head*g + r) — the token-major rows.
        for i in cutlass.range_constexpr(MT * D // NT):
            e = tidx + i * NT
            R = e // D
            c = e % D
            if seqlen_q * G > R:
                m = R // G
                r = R % G
                sQ[R, c] = cutlass.BFloat16(mQ[batch_idx, Int32(m), kv_head * G + r, c])
            else:
                sQ[R, c] = cutlass.BFloat16(0.0)

        # ---- the mma machinery ------------------------------------------
        tiled_mma = cute.make_tiled_mma(
            MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
            (4, 1, 1),
            permutation_mnk=(MT, SR, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)
        # The RAW K fragment: the (SR, D/2) u16 view (2 fp8 per element) —
        # dequanted in registers per pass below. The A side matches in
        # view units (the even/odd-d Q fragments filled directly from sQ).
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK8v))
        # The RAW V fragment: built from the (n, k)-shaped composition (the
        # sVt idiom) — the u16 elements are 2 fp8 each, dequanted in
        # registers per pass below.
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sV8vt))
        acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((MT, SR)), cutlass.Float32)
        # The acc_O split by d parity (the two dequant passes each cover
        # half the columns; the same total registers).
        acc_O_even = cute.make_rmem_tensor(thr_mma.partition_shape_C((MT, D // 2)), cutlass.Float32)
        acc_O_odd = cute.make_rmem_tensor(thr_mma.partition_shape_C((MT, D // 2)), cutlass.Float32)
        acc_O_even.fill(0.0)
        acc_O_odd.fill(0.0)
        n_row_slots = cutlass.const_expr(
            cute.size(thr_mma.partition_shape_C((MT, SR))[0][0]) * cute.size(thr_mma.partition_shape_C((MT, SR))[1])
        )
        row_max = cute.make_rmem_tensor((n_row_slots,), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((n_row_slots,), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        smem_copy_K = cute.make_tiled_copy_B(
            cute.make_copy_atom(LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16),
            tiled_mma,
        )
        smem_copy_V = cute.make_tiled_copy_B(
            cute.make_copy_atom(LdMatrix8x8x16bOp(transpose=True, num_matrices=4), cutlass.BFloat16),
            tiled_mma,
        )
        smem_thr_K = smem_copy_K.get_slice(tidx)
        smem_thr_V = smem_copy_V.get_slice(tidx)
        tSsK_a = smem_thr_K.partition_S(sK8v)
        tSsK_b = smem_thr_K.partition_S(sK8bv)
        tSsK_c = smem_thr_K.partition_S(sK8cv)
        tSsK_d = smem_thr_K.partition_S(sK8dv)
        tSsK_e = smem_thr_K.partition_S(sK8ev)
        tSsV_a = smem_thr_V.partition_S(sV8vt)
        tSsV_b = smem_thr_V.partition_S(sV8bvt)
        tSsV_c = smem_thr_V.partition_S(sV8cvt)
        tSsV_d = smem_thr_V.partition_S(sV8dvt)
        tSsV_e = smem_thr_V.partition_S(sV8evt)
        # The copy-side fragment views: the mma fragments' layouts differ
        # from the smem-copy atoms' — the retile bridges them (the
        # reference's *_copy_view idiom).
        tSrK_copy = smem_thr_K.retile(tSrK)
        tOrVt_copy = smem_thr_V.retile(tOrVt)
        # The u32 view of the raw V fragment: the even/odd register dequant
        # operates on the register pairs (each = 2 fragment elements = 2
        # u16 view columns = 4 fp8).
        tOrVtVu = cute.make_tensor(
            cute.recast_ptr(tOrVt.iterator, dtype=cutlass.Uint32),
            cute.make_layout(cute.size(tOrVt) // 2),
        )
        # The u32 view of the raw K fragment (the same recipe as tOrVtVu).
        tSrKu = cute.make_tensor(
            cute.recast_ptr(tSrK.iterator, dtype=cutlass.Uint32),
            cute.make_layout(cute.size(tSrK) // 2),
        )

        # The even/odd-d Q fragments: the register-path K dequant runs two
        # passes over the (SR, D/2) fp8 view (the mma's k-element = one u16
        # = 2 fp8 of one key), so the A side must match in view units —
        # pass even consumes Q[2j], pass odd Q[2j+1]. The fragments are
        # filled directly from sQ (a generic S2R copy atom's per-lane
        # layout does not match the mma A-fragment — dbg35; a composed
        # strided view mis-decodes — dbg37). The decoded m16k16 A-fragment
        # map (dbg36, 8 elements per k-step): row m0 + 8*((e//2)%2), view
        # column k0 + (e%2) + 8*(e//4); row base = 16*warp + lane/4 (the
        # (4,1,1) arrangement gives each warp a disjoint 16-row band),
        # k0 = 2*(lane%4).
        tSrQ_even = cute.make_rmem_tensor(thr_mma.partition_shape_A((MT, D // 2)), cutlass.BFloat16)
        tSrQ_odd = cute.make_rmem_tensor(thr_mma.partition_shape_A((MT, D // 2)), cutlass.BFloat16)
        # The barrier is required: the flat fill scatters each sQ row across
        # all 4 warps, so no warp can read before every warp has stored.
        cute.arch.sync_threads()
        for ks in cutlass.range_constexpr(D // 32):
            for e in cutlass.range_constexpr(8):
                m = 16 * (tidx // 32) + tidx % 32 // 4 + 8 * ((e // 2) % 2)
                j = 16 * ks + 2 * (tidx % 4) + (e % 2) + 8 * (e // 4)
                tSrQ_even[None, None, ks][e] = sQ[m, Int32(2) * j]
                tSrQ_odd[None, None, ks][e] = sQ[m, Int32(2) * j + 1]
        # The barrier: the ring's prefetch buffers alias sQ, which the
        # fills just finished reading — no warp may write them (the stage
        # loop's first g2s into buf 1) until every warp's fill is done.
        cute.arch.sync_threads()

        # The stage loop over this CTA's KV chunk, 2-stage pipelined:
        # the prologue fills buf 0, then each iteration issues the NEXT
        # stage's g2s into the other buffer before consuming the current
        # one — the cp.async DRAM latency overlaps the dequant+mma.
        # The prefetch's rows clamp to the chunk end (branch-free; the
        # duplicated rows are never read).
        n_stages = cute.ceil_div(kv_len_c, SR)
        if kv_len_c > 0:  # a degenerate empty chunk must not read row -1
            self.g2s_stage(
                mK,
                mV,
                mPageTable,
                sK8,
                sK8b,
                sK8c,
                sK8d,
                sK8e,
                sV8,
                sV8b,
                sV8c,
                sV8d,
                sV8e,
                sK8v,
                sV8v,
                tidx,
                kv_start,
                0,
                kv_len_c,
                num_head_kv,
                kv_head,
                batch_idx,
                kv_row0,
                0,
            )
            cute.arch.cp_async_commit_group()
            self.g2s_stage(
                mK,
                mV,
                mPageTable,
                sK8,
                sK8b,
                sK8c,
                sK8d,
                sK8e,
                sV8,
                sV8b,
                sV8c,
                sV8d,
                sV8e,
                sK8v,
                sV8v,
                tidx,
                kv_start,
                cutlass.min(SR, kv_len_c - 1),
                kv_len_c,
                num_head_kv,
                kv_head,
                batch_idx,
                kv_row0,
                1,
            )
            cute.arch.cp_async_commit_group()
            self.g2s_stage(
                mK,
                mV,
                mPageTable,
                sK8,
                sK8b,
                sK8c,
                sK8d,
                sK8e,
                sV8,
                sV8b,
                sV8c,
                sV8d,
                sV8e,
                sK8v,
                sV8v,
                tidx,
                kv_start,
                cutlass.min(2 * SR, kv_len_c - 1),
                kv_len_c,
                num_head_kv,
                kv_head,
                batch_idx,
                kv_row0,
                2,
            )
            cute.arch.cp_async_commit_group()
            self.g2s_stage(
                mK,
                mV,
                mPageTable,
                sK8,
                sK8b,
                sK8c,
                sK8d,
                sK8e,
                sV8,
                sV8b,
                sV8c,
                sV8d,
                sV8e,
                sK8v,
                sV8v,
                tidx,
                kv_start,
                cutlass.min(3 * SR, kv_len_c - 1),
                kv_len_c,
                num_head_kv,
                kv_head,
                batch_idx,
                kv_row0,
                3,
            )
            cute.arch.cp_async_commit_group()
        for stage in cutlass.range(n_stages):
            par = stage % 5
            row0 = stage * SR
            row0_next = cutlass.min((stage + 4) * SR, kv_len_c - 1)
            self.g2s_stage(
                mK,
                mV,
                mPageTable,
                sK8,
                sK8b,
                sK8c,
                sK8d,
                sK8e,
                sV8,
                sV8b,
                sV8c,
                sV8d,
                sV8e,
                sK8v,
                sV8v,
                tidx,
                kv_start,
                row0_next,
                kv_len_c,
                num_head_kv,
                kv_head,
                batch_idx,
                kv_row0,
                (stage + 4) % 5,
            )
            cute.arch.cp_async_commit_group()
            # <= 4 groups outstanding: everything up to `stage` has landed.
            cute.arch.cp_async_wait_group(4)
            cute.arch.sync_threads()

            # 2. the QK^T: the acc_S = Q * K^T for this stage's rows, in
            #    TWO register-dequant passes over the raw u16 K fragment
            #    (the FlashInfer/Marlin fast_dequant_f8f16x4 recipe — the
            #    same x4 path the smem dequant used, verified bit-exact on
            #    sm_87). Pass even: the u16 pairs' LOW bytes (the d = 2j
            #    columns); pass odd: the high bytes (d = 2j+1), no shift.
            #    One accumulator: the dequant + ldmatrix between the passes
            #    breaks the HMMA dependency chain (the old 2-acc reason),
            #    and one acc saves 16 registers at the ptxas ceiling.
            acc_S.fill(0.0)
            tSsK_sel = tSsK_a
            if par == 1:
                tSsK_sel = tSsK_b
            if par == 2:
                tSsK_sel = tSsK_c
            if par == 3:
                tSsK_sel = tSsK_d
            if par == 4:
                tSsK_sel = tSsK_e
            for ks in cutlass.range_constexpr(D // 32):
                cute.copy(smem_copy_K, tSsK_sel[None, None, ks], tSrK_copy[None, None, ks])
            for i in cutlass.range_constexpr(cute.size(tSrK) // 2):
                qp = (tSrKu[i] & 0x00FF00FF) << 8
                out = (qp & _DEQ_MASK_SIGN) | ((qp & _DEQ_MASK_LSB) >> 4)
                tSrKu[i] = cute.arch.fma_packed_bf16x2(out, _DEQ_BIAS, cutlass.Int32(0))
            for ks in cutlass.range_constexpr(D // 32):
                cute.gemm(tiled_mma, acc_S, tSrQ_even[None, None, ks], tSrK[None, None, ks], acc_S)
            for ks in cutlass.range_constexpr(D // 32):
                cute.copy(smem_copy_K, tSsK_sel[None, None, ks], tSrK_copy[None, None, ks])
            for i in cutlass.range_constexpr(cute.size(tSrK) // 2):
                out = (tSrKu[i] & _DEQ_MASK_SIGN) | ((tSrKu[i] & _DEQ_MASK_LSB) >> 4)
                tSrKu[i] = cute.arch.fma_packed_bf16x2(out, _DEQ_BIAS, cutlass.Int32(0))
            for ks in cutlass.range_constexpr(D // 32):
                cute.gemm(tiled_mma, acc_S, tSrQ_odd[None, None, ks], tSrK[None, None, ks], acc_S)

            # 4. the online softmax (the quad-local row stats), the P to
            #    smem, and the O rescale. The warps own disjoint 16-row
            #    bands (the (4,1,1) arrangement), so the row stats reduce
            #    across the quad only. The mask: the stage-local row bound
            #    plus the per-token causal tail (the row's token m attends
            #    to the keys gk <= kv_len - q_len + m — the GEMV
            #    ref_decode contract).
            self.softmax_p(
                thr_mma,
                acc_S,
                acc_O_even,
                acc_O_odd,
                row_max,
                row_sum,
                scale_eff,
                row0,
                kv_len_c,
                kv_len,
                kv_start,
                seqlen_q,
                n_row_slots,
                tidx,
            )

            # 5. PV: the acc_O += P * V — the S fragment reinterpreted in
            #    place as the A operand (the reference's logical_divide).
            #    The V is the RAW u16 view: two register-dequant passes
            #    (even d, odd d) into the acc_O halves.
            tSsV_sel = tSsV_a
            if par == 1:
                tSsV_sel = tSsV_b
            if par == 2:
                tSsV_sel = tSsV_c
            if par == 3:
                tSsV_sel = tSsV_d
            if par == 4:
                tSsV_sel = tSsV_e
            self.pv(
                tiled_mma,
                acc_S,
                smem_copy_V,
                tSsV_sel,
                tOrVt,
                tOrVt_copy,
                tOrVtVu,
                acc_O_even,
                acc_O_odd,
            )
            # The barrier BEFORE the next stage's ring overwrite: the PV's
            # ldmatrix reads the raw V buffers at the very END of the body
            # (the old smem-dequant path read them inside the barrier'd
            # dequant block) — without this, a fast warp's next-issue
            # cp.async can land on a buffer a lagging warp is still
            # reading (the M=8 short-chunk race, run-to-run varying).
            cute.arch.sync_threads()

        # 6. the epilogue.
        self.epilogue(
            acc_O_even,
            acc_O_odd,
            row_max,
            row_sum,
            n_row_slots,
            thr_mma,
            tidx,
            mO,
            mLSE,
            mO_partial,
            mLSE_partial,
            o_off,
            num_head,
            v_descale,
            scale_eff,
            seqlen_q,
            is_varlen_q,
            q_row_base,
            batch_idx,
            kv_head,
            split_idx,
        )

    # -- the helpers ------------------------------------------------------
    @staticmethod
    @cute.jit
    def _acc_mn_view(acc: cute.Tensor) -> cute.Tensor:
        """The (M, N)-indexed view of an mma accumulator fragment (the
        reference's _make_acc_tensor_mn_view — the layout must be
        COMPOSED with the fragment's actual layout, not a fresh compact
        one)."""
        col_major = cute.make_layout(acc.layout.shape)
        mn = cute.make_layout(
            (
                (col_major.shape[0][1], col_major.shape[1]),
                (col_major.shape[0][0], col_major.shape[2]),
            ),
            stride=(
                (col_major.stride[0][1], col_major.stride[1]),
                (col_major.stride[0][0], col_major.stride[2]),
            ),
        )
        mn = cute.composition(acc.layout, mn)
        return cute.make_tensor(acc.iterator, mn)

    @cute.jit
    def softmax_p(
        self,
        thr_mma,
        acc_S: cute.Tensor,
        acc_O_even: cute.Tensor,
        acc_O_odd: cute.Tensor,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        scale_eff: Float32,
        row0: Int32,
        kv_len_c: Int32,
        kv_len: Int32,
        kv_start: Int32,
        seqlen_q: cutlass.Constexpr[int],
        n_row_slots: cutlass.Constexpr[int],
        tidx: Int32,
    ):
        """The per-stage online softmax + the P to smem. Quad-local: the
        warps own disjoint 16-row bands (the (4,1,1) arrangement), so the
        row stats reduce across the 4-lane butterfly group only. The
        row/col coordinates come from a partitioned identity tensor (the
        reference's tScS pattern — the coords are 4-deep; [2] is the row,
        [3] the column)."""
        MT = self.M_TILE
        G = self.g
        acc_S_mn = self._acc_mn_view(acc_S)
        acc_O_even_mn = self._acc_mn_view(acc_O_even)
        acc_O_odd_mn = self._acc_mn_view(acc_O_odd)

        mcS = cute.make_identity_tensor((1, 1, MT, self.STAGE_ROWS))
        cS = mcS[0, 0, None, None]
        tScS = thr_mma.partition_C(cS)
        tScS_mn = self._acc_mn_view(tScS)

        # 1. the per-row max (the quad-reduced) with the mask, then the
        #    running O/sum rescale. The M tile is 64 rows; q_len*6 <= 48
        #    are real — the warps whose ENTIRE 16-row band is padding
        #    (band start >= seqlen_q*G, compile-time-foldable per kernel
        #    instance) compute nothing: their Q is zero, their fake causal
        #    limit passes the mask, and they would produce exp2=1 garbage
        #    the epilogue never stores. The skip is WARP-UNIFORM — the
        #    butterfly shuffles below are warp-collective and deadlock on
        #    a divergent branch. A partially-padded band (one warp at
        #    most) keeps the full path; its padded rows are masked.
        if Int32(16) * (tidx // 32) < seqlen_q * G:
            for r in cutlass.range_constexpr(n_row_slots):
                row = tScS_mn[r, 0][2]
                m_local = -cutlass.Float32.inf
                tok = row // G
                for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                    stage_row = row0 + tScS_mn[0, c][3]
                    gk = kv_start + stage_row
                    lim = kv_len - seqlen_q + tok
                    if stage_row < kv_len_c and gk <= lim:
                        m_local = cute.arch.fmax(m_local, acc_S_mn[r, c])
                m_local = cute.arch.fmax(
                    m_local,
                    cute.arch.shuffle_sync_bfly(m_local, offset=2, mask=-1, mask_and_clamp=31),
                )
                m_local = cute.arch.fmax(
                    m_local,
                    cute.arch.shuffle_sync_bfly(m_local, offset=1, mask=-1, mask_and_clamp=31),
                )
                # the empty row (all masked) keeps the O at 0 via m_new = 0.
                m_new = m_local
                if m_new == -cutlass.Float32.inf:
                    m_new = Float32(0.0)
                rescale = cute.math.exp2((row_max[r] - m_new) * scale_eff, fastmath=True)
                for c in cutlass.range_constexpr(cute.size(acc_O_even_mn.shape[1])):
                    acc_O_even_mn[r, c] = acc_O_even_mn[r, c] * rescale
                    acc_O_odd_mn[r, c] = acc_O_odd_mn[r, c] * rescale
                row_sum[r] = row_sum[r] * rescale
                row_max[r] = m_new

        # 2. the P = exp2(S*scale - m_new) in place; the per-thread partial
        #    sums; the group reduction; the bf16 P to the smem round-trip.
        if Int32(16) * (tidx // 32) < seqlen_q * G:
            for r in cutlass.range_constexpr(n_row_slots):
                row = tScS_mn[r, 0][2]
                s_partial = Float32(0.0)
                tok = row // G
                for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                    stage_row = row0 + tScS_mn[0, c][3]
                    p = Float32(0.0)
                    if stage_row < kv_len_c and kv_start + stage_row <= kv_len - seqlen_q + tok:
                        p = cute.math.exp2(acc_S_mn[r, c] * scale_eff - row_max[r] * scale_eff, fastmath=True)
                    acc_S_mn[r, c] = p
                    s_partial += p
                s_partial += cute.arch.shuffle_sync_bfly(s_partial, offset=2, mask=-1, mask_and_clamp=31)
                s_partial += cute.arch.shuffle_sync_bfly(s_partial, offset=1, mask=-1, mask_and_clamp=31)
                row_sum[r] += s_partial
        else:
            # the fully-padded warps: P = 0 so the PV's HMMA on their rows
            # accumulates nothing (the acc_O rows are never stored).
            for r in cutlass.range_constexpr(n_row_slots):
                for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                    acc_S_mn[r, c] = Float32(0.0)

    @cute.jit
    def pv(
        self,
        tiled_mma,
        acc_S: cute.Tensor,
        smem_copy_V,
        tSsV,
        tOrVt,
        tOrVt_copy,
        tOrVtVu,
        acc_O_even: cute.Tensor,
        acc_O_odd: cute.Tensor,
    ):
        """The PV: the P (the S fragment reinterpreted as the A operand —
        the reference's logical_divide on the fragment layout) x the V,
        in TWO passes over the raw u16 fragment: pass even dequants the
        low bytes of each u16 (the d=2j column) into acc_O_even, pass
        odd the high bytes (d=2j+1) into acc_O_odd. The raw fp8 smem is
        untouched — the dequant lives in the fragment registers."""
        rP = cute.make_fragment_like(acc_S, cutlass.BFloat16)
        rP.store(acc_S.load().to(cutlass.BFloat16))
        div = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (div.shape[0], div.shape[2][0]),
                div.shape[1],
                div.shape[2][1],
            ),
            stride=(
                (div.stride[0], div.stride[2][0]),
                div.stride[1],
                div.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
        for ks in cutlass.range_constexpr(cute.size(tOrVt.layout.shape[2])):
            cute.copy(smem_copy_V, tSsV[None, None, ks], tOrVt_copy[None, None, ks])
        for i in cutlass.range_constexpr(cute.size(tOrVt) // 2):
            qp = (tOrVtVu[i] & 0x00FF00FF) << 8
            out = (qp & _DEQ_MASK_SIGN) | ((qp & _DEQ_MASK_LSB) >> 4)
            tOrVtVu[i] = cute.arch.fma_packed_bf16x2(out, _DEQ_BIAS, cutlass.Int32(0))
        for ks in cutlass.range_constexpr(cute.size(tOrVt.layout.shape[2])):
            cute.gemm(tiled_mma, acc_O_even, tOrS[None, None, ks], tOrVt[None, None, ks], acc_O_even)
        for ks in cutlass.range_constexpr(cute.size(tOrVt.layout.shape[2])):
            cute.copy(smem_copy_V, tSsV[None, None, ks], tOrVt_copy[None, None, ks])
        for i in cutlass.range_constexpr(cute.size(tOrVt) // 2):
            out = (tOrVtVu[i] & _DEQ_MASK_SIGN) | ((tOrVtVu[i] & _DEQ_MASK_LSB) >> 4)
            tOrVtVu[i] = cute.arch.fma_packed_bf16x2(out, _DEQ_BIAS, cutlass.Int32(0))
        for ks in cutlass.range_constexpr(cute.size(tOrVt.layout.shape[2])):
            cute.gemm(tiled_mma, acc_O_odd, tOrS[None, None, ks], tOrVt[None, None, ks], acc_O_odd)

    @cute.jit
    def epilogue(
        self,
        acc_O_even: cute.Tensor,
        acc_O_odd: cute.Tensor,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        n_row_slots: cutlass.Constexpr[int],
        thr_mma,
        tidx: Int32,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        mO_partial: cute.Tensor | None,
        mLSE_partial: cute.Tensor | None,
        o_off: Int32,
        num_head: Int32,
        v_descale: Float32,
        scale_eff: Float32,
        seqlen_q: cutlass.Constexpr[int],
        is_varlen_q: cutlass.Constexpr[bool],
        q_row_base: Int32,
        batch_idx: Int32,
        kv_head: Int32,
        split_idx: Int32,
    ):
        """The normalize + the store. The final mode: the O + the LSE at
        o_off; the split mode: the fp32 partials (the GEMV's offset
        math). The M row R -> (m, r) = divmod(R, 6) (the token-major);
        the O at o_off + m*row_stride + (kv_head*6 + r)*D. Each thread
        stores the fragment slots it owns — the slot's global (row, col)
        comes from a partitioned identity tensor, so the writes never
        collide (exactly one thread owns each (row, col))."""
        MT = self.M_TILE
        D = self.HEAD_DIM
        G = self.g
        o_elem = mO_partial.element_type if const_expr(mO_partial is not None) else mO.element_type
        o_base = mO_partial.iterator if const_expr(mO_partial is not None) else mO.iterator
        row_stride = num_head * D  # the dense stride of one token row in O
        q_rows = seqlen_q * G

        acc_O_even_mn = self._acc_mn_view(acc_O_even)
        acc_O_odd_mn = self._acc_mn_view(acc_O_odd)
        mcO = cute.make_identity_tensor((1, 1, MT, D // 2))
        cO = mcO[0, 0, None, None]
        tScO_mn = self._acc_mn_view(thr_mma.partition_C(cO))

        # 1. the per-slot normalize (the row_sum is group-uniform, so the
        #    scale is identical across the group — no cross-thread issue).
        for r in cutlass.range_constexpr(n_row_slots):
            s = row_sum[r]
            scale = 1.0 if (s == 0.0 or s != s) else cute.arch.rcp_approx(s)
            for c in cutlass.range_constexpr(cute.size(acc_O_even_mn.shape[1])):
                acc_O_even_mn[r, c] = acc_O_even_mn[r, c] * (scale * v_descale)
                acc_O_odd_mn[r, c] = acc_O_odd_mn[r, c] * (scale * v_descale)

        # 2. the O store: the slot's (row, col) from the identity coords —
        #    the acc's n indexes the D/2 view columns; the even acc holds
        #    the d = 2j outputs, the odd acc the d = 2j+1.
        for acc_i in cutlass.range_constexpr(2):
            acc_mn = acc_O_even_mn if acc_i == 0 else acc_O_odd_mn
            for r in cutlass.range_constexpr(n_row_slots):
                R = tScO_mn[r, 0][2]
                if q_rows > R:
                    m = R // G
                    head = kv_head * G + (R % G)
                    for c in cutlass.range_constexpr(cute.size(acc_mn.shape[1])):
                        i = Int32(2) * tScO_mn[0, c][3] + acc_i
                        v = acc_mn[r, c].to(o_elem)
                        o_dst = cute.make_tensor(
                            cute.make_ptr(
                                o_elem,
                                (o_base + o_off + m * row_stride + head * D + i).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout(1),
                        )
                        o_dst[0] = v

        # 3. the LSE: the group's first lane writes its rows' slots (the
        #    value is group-uniform after the reduction — the duplicate
        #    writes are benign).
        if const_expr(mLSE is not None or mLSE_partial is not None):
            lane = tidx % 32
            lse_tensor = mLSE_partial if const_expr(mLSE_partial is not None) else mLSE
            for r in cutlass.range_constexpr(n_row_slots):
                R = tScO_mn[r, 0][2]
                if q_rows > R and lane == (R % 8) * 4:
                    m = R // G
                    qo_head = kv_head * G + (R % G)
                    l_r = row_sum[r]
                    lse = (
                        row_max[r] * scale_eff + cute.math.log2(cutlass.max(l_r, Float32(1e-30)), fastmath=True)
                    ) * Float32(math.log(2.0))
                    if const_expr(mLSE_partial is not None):
                        if const_expr(is_varlen_q):
                            lse_tensor[split_idx, qo_head, Int32(q_row_base + m)] = lse
                        else:
                            lse_tensor[split_idx, batch_idx, qo_head, Int32(m)] = lse
                    else:
                        if const_expr(is_varlen_q):
                            lse_tensor[qo_head, Int32(q_row_base + m)] = lse
                        else:
                            lse_tensor[Int32(batch_idx), qo_head, Int32(m)] = lse
