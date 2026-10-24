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
  swizzle is the M2 bank-conflict pass). The V is staged plain and read
  through a composition to (d, row) — what the transpose=True B-atom
  wants; a directly-allocated transposed layout loses the 16B alignment
  the ldsm requires.
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
        # The MT=16 experimental path (the (1,4,1) N-split arrangement,
        # q_len <= 2): opt-in via the env so the default stays the proven
        # MT=64 kernel until it gates and benches.
        import os

        self.small_tile = os.environ.get("MJOLNIR_MMA16", "") == "1"
        # The 3-CTA occupancy probe: nvvm.minctasm (= __launch_bounds__
        # min-blocks) makes ptxas cap the registers to <=170/thread.
        # Probe only — the smem still binds at 2 CTAs until the register
        # paths land, so this measures the cap's spill/ILP cost alone.
        self.regcap = os.environ.get("MJOLNIR_REGCAP", "") == "1"
        # The full 3-CTA package: the K register path (no sK16), the sV16
        # aliased over the dead-after-hoist sQ + a pad, the 2-pair ring,
        # and nvvm.minctasm=3. MT=16 only.
        self.regpath = os.environ.get("MJOLNIR_REGPATH", "") == "1"

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
        if const_expr(self.small_tile and seqlen_q * self.g <= 32):
            self.kernel_small(
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
                16 if seqlen_q * self.g <= 16 else 32,
            ).launch(
                grid=(cute.size(mV.shape[-2]), num_splits, batch_size),
                block=[self.NUM_THREADS, 1, 1],
                min_blocks_per_mp=3 if (self.regcap or self.regpath) else 0,
                stream=stream,
            )
            return
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
        sV8a: cute.Tensor,
        sV8b: cute.Tensor,
        sV8c: cute.Tensor,
        tidx,
        kv_start,
        row0,
        kv_len_c,
        num_head_kv,
        kv_head,
        batch_idx,
        kv_row0,
        buf,
        k_dst_view: cute.Tensor | None = None,
    ) -> None:
        """One KV stage: gmem -> smem fp8 (32 rows x 256 B = 4 x 16B chunks
        per thread) into buffer `buf` of the double-buffered staging tiles.
        The tail rows CLAMP to the last valid row (the GEMV's trick — no
        branch); the softmax masks the overhang by kv_len_c. The prefetch
        of a not-yet-valid stage clamps row0 the same way: the duplicated
        rows are never read. With k_dst_view set (the register path), the
        K bytes land at the view's swizzled offsets so the ldmatrix read
        of the (SR, D/2) u16 view sees logical (row, view-col) data."""
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
            if const_expr(k_dst_view is None):
                koff = (kdst + e16 * 16).toint()
            else:
                koff = (kdst + Int32(2) * cute.crd2idx((r, ck // 2), k_dst_view.layout)).toint()
            k_dst = cute.make_tensor(
                cute.make_ptr(
                    mK.element_type,
                    koff,
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
                    (vdst + e16 * 16).toint(),
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
        # The u32 views of the fp8 staging tiles: the 4-wide dequant reads.
        sK8u = cute.make_tensor(cute.recast_ptr(sK8.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4))
        sV8u = cute.make_tensor(cute.recast_ptr(sV8.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4))
        sK8bu = cute.make_tensor(cute.recast_ptr(sK8b.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4))
        sV8bu = cute.make_tensor(cute.recast_ptr(sV8b.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4))
        sK8cu = cute.make_tensor(cute.recast_ptr(sK8c.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4))
        sV8cu = cute.make_tensor(cute.recast_ptr(sV8c.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4))
        sK16 = smem.allocate_tensor(cutlass.BFloat16, cute.tile_to_shape(bf16_atom, (SR, D), (0, 1)), byte_alignment=16)
        # The V staged in the plain (row, d) orientation; the transposed
        # view for the transpose=True B-atom is a composition (the
        # reference's sVt pattern — a directly-allocated transposed
        # layout loses the 16B alignment the ldsm needs).
        sV16 = smem.allocate_tensor(cutlass.BFloat16, cute.tile_to_shape(bf16_atom, (SR, D), (0, 1)), byte_alignment=16)
        sVt = cute.composition(
            sV16,
            cute.make_layout((D, SR), stride=(SR, 1)),
        )

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
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK16))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((MT, SR)), cutlass.Float32)
        acc_S2 = cute.make_rmem_tensor(thr_mma.partition_shape_C((MT, SR)), cutlass.Float32)
        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((MT, D)), cutlass.Float32)
        acc_O.fill(0.0)
        n_row_slots = cutlass.const_expr(
            cute.size(thr_mma.partition_shape_C((MT, SR))[0][0]) * cute.size(thr_mma.partition_shape_C((MT, SR))[1])
        )
        row_max = cute.make_rmem_tensor((n_row_slots,), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((n_row_slots,), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        smem_copy_Q = cute.make_tiled_copy_A(
            cute.make_copy_atom(LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16),
            tiled_mma,
        )
        smem_copy_K = cute.make_tiled_copy_B(
            cute.make_copy_atom(LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16),
            tiled_mma,
        )
        smem_copy_V = cute.make_tiled_copy_B(
            cute.make_copy_atom(LdMatrix8x8x16bOp(transpose=True, num_matrices=4), cutlass.BFloat16),
            tiled_mma,
        )
        smem_thr_Q = smem_copy_Q.get_slice(tidx)
        smem_thr_K = smem_copy_K.get_slice(tidx)
        smem_thr_V = smem_copy_V.get_slice(tidx)
        tSsQ = smem_thr_Q.partition_S(sQ)
        tSsK = smem_thr_K.partition_S(sK16)
        tOsVt = smem_thr_V.partition_S(sVt)
        # The copy-side fragment views: the mma fragments' layouts differ
        # from the smem-copy atoms' — the retile bridges them (the
        # reference's *_copy_view idiom).
        tSrQ_copy = smem_thr_Q.retile(tSrQ)
        tSrK_copy = smem_thr_K.retile(tSrK)
        tOrVt_copy = smem_thr_V.retile(tOrVt)

        # The Q fragments load once: the Q tile is stage-invariant (the
        # per-stage reload cost 32 KB of ldmatrix traffic per stage). The
        # barrier is required: the flat fill scatters each sQ row across
        # all 4 warps, so no warp can read before every warp has stored.
        cute.arch.sync_threads()
        for k in cutlass.range_constexpr(D // 16):
            cute.copy(smem_copy_Q, tSsQ[None, None, k], tSrQ_copy[None, None, k])
        # The barrier: the ring's prefetch buffers alias sQ, which the
        # hoist just finished reading — no warp may write them (the stage
        # loop's first g2s into buf 1) until every warp's ldsm is done.
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
                sV8,
                sV8b,
                sV8c,
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
                sV8,
                sV8b,
                sV8c,
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
        for stage in cutlass.range(n_stages):
            par = stage % 3
            row0 = stage * SR
            row0_next = cutlass.min((stage + 2) * SR, kv_len_c - 1)
            self.g2s_stage(
                mK,
                mV,
                mPageTable,
                sK8,
                sK8b,
                sK8c,
                sV8,
                sV8b,
                sV8c,
                tidx,
                kv_start,
                row0_next,
                kv_len_c,
                num_head_kv,
                kv_head,
                batch_idx,
                kv_row0,
                (stage + 2) % 3,
            )
            cute.arch.cp_async_commit_group()
            # <= 2 groups outstanding: everything up to `stage` has landed.
            cute.arch.cp_async_wait_group(2)
            cute.arch.sync_threads()

            # 2. the dequant fp8 -> bf16 in smem, 4 values per iteration —
            #    the FlashInfer/Marlin fast_dequant_f8f16x4 recipe (prmt +
            #    shift/mask + one packed-bf16 fma per 4 values), verified
            #    bit-exact on sm_87 for normals and zero; subnormal codes
            #    map to their normal-formula neighbors (the same documented
            #    caveat as _e4m3_to_f32, absorbed by the tolerance). The
            #    old element-wise path cost ~8 ALU ops/value with
            #    byte-wide LDS and made the kernel issue-bound (the ALU
            #    pipe was ~50% of all executed instructions).
            srcu_k = sK8u
            srcu_v = sV8u
            if par == 1:
                srcu_k = sK8bu
                srcu_v = sV8bu
            if par == 2:
                srcu_k = sK8cu
                srcu_v = sV8cu
            for i in cutlass.range_constexpr(SR * D // NT // 4):
                e4 = (tidx + i * NT) * 4
                r = e4 // D
                c = e4 % D
                qk = srcu_k[e4 // 4]
                qp = cute.arch.prmt(qk, qk, 0x1302)
                out1 = (qp & _DEQ_MASK_SIGN) | ((qp & _DEQ_MASK_LSB) >> 4)
                out2 = ((qp << 8) & _DEQ_MASK_SIGN) | (((qp << 8) & _DEQ_MASK_LSB) >> 4)
                out1 = cute.arch.fma_packed_bf16x2(out1, _DEQ_BIAS, cutlass.Int32(0))
                out2 = cute.arch.fma_packed_bf16x2(out2, _DEQ_BIAS, cutlass.Int32(0))
                sK16[r, c] = (out1 & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                sK16[r, c + 1] = ((out1 >> 16) & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                sK16[r, c + 2] = (out2 & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                sK16[r, c + 3] = ((out2 >> 16) & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                qv = srcu_v[e4 // 4]
                qp = cute.arch.prmt(qv, qv, 0x1302)
                out1 = (qp & _DEQ_MASK_SIGN) | ((qp & _DEQ_MASK_LSB) >> 4)
                out2 = ((qp << 8) & _DEQ_MASK_SIGN) | (((qp << 8) & _DEQ_MASK_LSB) >> 4)
                out1 = cute.arch.fma_packed_bf16x2(out1, _DEQ_BIAS, cutlass.Int32(0))
                out2 = cute.arch.fma_packed_bf16x2(out2, _DEQ_BIAS, cutlass.Int32(0))
                sV16[r, c] = (out1 & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                sV16[r, c + 1] = ((out1 >> 16) & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                sV16[r, c + 2] = (out2 & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                sV16[r, c + 3] = ((out2 >> 16) & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
            cute.arch.sync_threads()

            # 3. the QK^T: the acc_S = Q * K^T for this stage's rows (the
            #    Q fragments loaded once before the loop). Two independent
            #    accumulator chains over the k halves: a single accumulation
            #    is a 16-deep serial HMMA chain (the wait stalls were the #2
            #    cost at locked clocks); the merge is 16 adds.
            acc_S.fill(0.0)
            acc_S2.fill(0.0)
            for k in cutlass.range_constexpr(D // 16):
                cute.copy(smem_copy_K, tSsK[None, None, k], tSrK_copy[None, None, k])
            for k in cutlass.range_constexpr(D // 32):
                cute.gemm(tiled_mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)
                cute.gemm(tiled_mma, acc_S2, tSrQ[None, None, k + D // 32], tSrK[None, None, k + D // 32], acc_S2)
            for i in cutlass.range_constexpr(cute.size(acc_S)):
                acc_S[i] = acc_S[i] + acc_S2[i]

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
                acc_O,
                row_max,
                row_sum,
                scale_eff,
                row0,
                kv_len_c,
                kv_len,
                kv_start,
                seqlen_q,
                n_row_slots,
            )

            # 5. PV: the acc_O += P * V — the S fragment reinterpreted in
            #    place as the A operand (the reference's logical_divide).
            self.pv(tiled_mma, acc_S, smem_copy_V, tOsVt, tOrVt, tOrVt_copy, acc_O)

        # 6. the epilogue.
        self.epilogue(
            acc_O,
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

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel_small(
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
        mt: cutlass.Constexpr[int],
    ):
        """The small-M variant (mt = 16 or 32): the warps N-split BOTH
        gemms — the arrangement (mt/16, 4/(mt/16), 1) gives the warp w
        the S columns of its n-atom slice and the O columns of its d
        slice, 16 HMMAs each. The row stats merge across the warps
        through a smem slab initialized neutral (the max -inf, the sum
        0) so every warp may merge all 4 entries without decoding the
        arrangement; the P round-trips through a (mt, 32) bf16 tile
        aliasing sK16's head. The deferred-frame softmax (no per-stage
        O rescale)."""
        D = self.HEAD_DIM
        MT = mt
        SR = self.STAGE_ROWS
        NT = self.NUM_THREADS
        G = self.g

        kv_head = cute.arch.block_idx()[0]
        split_idx = cute.arch.block_idx()[1]
        batch_idx = cute.arch.block_idx()[2]
        tidx = cute.arch.thread_idx()[0]
        warp = tidx // 32

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

        o_off = q_row_base * num_head * D
        if const_expr(mO_partial is not None):
            if const_expr(is_varlen_q):
                split_stride = mO_partial.shape[1] * num_head * D
            else:
                split_stride = mO_partial.shape[1] * mO_partial.shape[2] * num_head * D
            o_off = o_off + split_idx * split_stride

        bf16_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, 64), stride=(64, 1)),
        )
        smem = cutlass.utils.SmemAllocator()
        # the Q tile (16, 256) bf16 = 8 KB — the 4x-smaller M tile.
        sQ = smem.allocate_tensor(cutlass.BFloat16, cute.tile_to_shape(bf16_atom, (MT, D), (0, 1)), byte_alignment=16)
        if const_expr(mt == 16 and self.regpath):
            # THE 3-CTA PACKAGE (MJOLNIR_REGPATH=1, MT=16): the K dequant
            # lives in the fragment registers (no sK16); the sV16 overlays
            # the sQ + this pad — both dead after the A-fragment fill —
            # and the ring is 2-pair (a, c; the refill issues after the
            # post-PV barrier since it retargets the consumed buffer).
            # 48.5 KB dynamic -> 3 CTAs/SM at the <=170-register cap.
            sPad = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(SR * D // 2), byte_alignment=16)
            sK8 = smem.allocate_tensor(mK.element_type, cute.make_layout(SR * D), byte_alignment=16)
            sV8 = smem.allocate_tensor(mV.element_type, cute.make_layout(SR * D), byte_alignment=16)
            sK8c = smem.allocate_tensor(mK.element_type, cute.make_layout(SR * D), byte_alignment=16)
            sV8c = smem.allocate_tensor(mV.element_type, cute.make_layout(SR * D), byte_alignment=16)
            sK8b = sK8c  # the buf-1 slot never issued in the 2-pair ring
            sV8b = sV8c
            sK8u = cute.make_tensor(cute.recast_ptr(sK8.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4))
            sV8u = cute.make_tensor(cute.recast_ptr(sV8.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4))
            sK8bu = sK8cu = cute.make_tensor(
                cute.recast_ptr(sK8c.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4)
            )
            sV8bu = sV8cu = cute.make_tensor(
                cute.recast_ptr(sV8c.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4)
            )
            # the (SR, D/2) bf16 VIEW of the fp8 K bytes: each u16 = 2
            # consecutive fp8 of ONE key, swizzled for the ldmatrix.
            sK8v = cute.make_tensor(
                cute.recast_ptr(sK8.iterator, dtype=cutlass.BFloat16),
                cute.tile_to_shape(bf16_atom, (SR, D // 2), (0, 1)),
            )
            sV16 = cute.make_tensor(sQ.iterator, cute.tile_to_shape(bf16_atom, (SR, D), (0, 1)))
            sStat = smem.allocate_tensor(cutlass.Float32, cute.make_layout(4 * MT * 2), byte_alignment=16)
            sP = smem.allocate_tensor(
                cutlass.BFloat16, cute.tile_to_shape(bf16_atom, (MT, SR), (0, 1)), byte_alignment=16
            )
        else:
            sK8 = smem.allocate_tensor(mK.element_type, cute.make_layout(SR * D), byte_alignment=16)
            sV8 = smem.allocate_tensor(mV.element_type, cute.make_layout(SR * D), byte_alignment=16)
            # the pair b: the K aliases the sQ head; for MT=32 the 16 KB sQ
            # also hosts the V (the 2-CTA budget — a real third buffer would
            # push the dynamic smem to 89 KB and collapse the occupancy to
            # 1 CTA/SM).
            sK8b = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=cutlass.Float8E4M3FN), cute.make_layout(SR * D))
            if const_expr(MT == 32):
                sV8b = cute.make_tensor(
                    cute.recast_ptr(sQ.iterator + SR * D // 2, dtype=cutlass.Float8E4M3FN), cute.make_layout(SR * D)
                )
            else:
                sV8b = smem.allocate_tensor(mV.element_type, cute.make_layout(SR * D), byte_alignment=16)
            sK8c = smem.allocate_tensor(mK.element_type, cute.make_layout(SR * D), byte_alignment=16)
            sV8c = smem.allocate_tensor(mV.element_type, cute.make_layout(SR * D), byte_alignment=16)
            sK8u = cute.make_tensor(cute.recast_ptr(sK8.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4))
            sV8u = cute.make_tensor(cute.recast_ptr(sV8.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4))
            sK8bu = cute.make_tensor(
                cute.recast_ptr(sK8b.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4)
            )
            sV8bu = cute.make_tensor(
                cute.recast_ptr(sV8b.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4)
            )
            sK8cu = cute.make_tensor(
                cute.recast_ptr(sK8c.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4)
            )
            sV8cu = cute.make_tensor(
                cute.recast_ptr(sV8c.iterator, dtype=cutlass.Uint32), cute.make_layout(SR * D // 4)
            )
            sK16 = smem.allocate_tensor(
                cutlass.BFloat16, cute.tile_to_shape(bf16_atom, (SR, D), (0, 1)), byte_alignment=16
            )
            sV16 = smem.allocate_tensor(
                cutlass.BFloat16, cute.tile_to_shape(bf16_atom, (SR, D), (0, 1)), byte_alignment=16
            )
            sStat = smem.allocate_tensor(cutlass.Float32, cute.make_layout(4 * MT * 2), byte_alignment=16)
            # the P tile (16, 32) bf16 aliasing sK16's head: the QK's ldmatrix
            # reads complete before the stats-merge barrier, the P writes
            # follow it, and the next dequant waits behind the post-PV barrier.
            sP = cute.make_tensor(sK16.iterator, cute.tile_to_shape(bf16_atom, (MT, SR), (0, 1)))
        sVt = cute.composition(sV16, cute.make_layout((D, SR), stride=(SR, 1)))
        # the neutral slab: the warps of the other m-bands leave -inf/0 —
        # the merge may read all 4 entries without decoding the arrangement.
        for i in cutlass.range_constexpr(4 * MT):
            sStat[i * 2] = -cutlass.Float32.inf
            sStat[i * 2 + 1] = Float32(0.0)

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

        m_atoms = MT // 16
        n_arr = 4 // m_atoms
        tiled_mma = cute.make_tiled_mma(
            MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
            (m_atoms, n_arr, 1),
            permutation_mnk=(MT, SR, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)
        if const_expr(mt == 16 and self.regpath):
            # the register path: the B fragment loads the RAW u16 pairs
            # from the swizzled view; the even/odd dequant happens on the
            # fragment registers (the 2-pass gemm below). The A side is
            # the even/odd-d Q fragments — the (1,4,1) arrangement puts
            # every warp on m-atom 0, so the row base has no warp term.
            tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK8v))
        else:
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK16))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((MT, SR)), cutlass.Float32)
        if const_expr(not (mt == 16 and self.regpath)):
            acc_S2 = cute.make_rmem_tensor(thr_mma.partition_shape_C((MT, SR)), cutlass.Float32)
        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((MT, D)), cutlass.Float32)
        acc_O.fill(0.0)
        n_row_slots = cutlass.const_expr(
            cute.size(thr_mma.partition_shape_C((MT, SR))[0][0]) * cute.size(thr_mma.partition_shape_C((MT, SR))[1])
        )
        row_max = cute.make_rmem_tensor((n_row_slots,), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((n_row_slots,), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        smem_copy_Q = cute.make_tiled_copy_A(
            cute.make_copy_atom(LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16),
            tiled_mma,
        )
        smem_copy_K = cute.make_tiled_copy_B(
            cute.make_copy_atom(LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16),
            tiled_mma,
        )
        smem_copy_V = cute.make_tiled_copy_B(
            cute.make_copy_atom(LdMatrix8x8x16bOp(transpose=True, num_matrices=4), cutlass.BFloat16),
            tiled_mma,
        )
        smem_thr_Q = smem_copy_Q.get_slice(tidx)
        smem_thr_K = smem_copy_K.get_slice(tidx)
        smem_thr_V = smem_copy_V.get_slice(tidx)
        tSsQ = smem_thr_Q.partition_S(sQ)
        if const_expr(mt == 16 and self.regpath):
            tSsK = smem_thr_K.partition_S(sK8v)
        else:
            tSsK = smem_thr_K.partition_S(sK16)
        tOsVt = smem_thr_V.partition_S(sVt)
        tSsP = smem_thr_Q.partition_S(sP)
        if const_expr(mt == 16 and self.regpath):
            # the u32 view of the K fragment: the even/odd dequant operates
            # on the register pairs (each = 2 fragment elements = 4 fp8 =
            # 4 consecutive d of the pair's 2 view columns).
            tSrKu = cute.make_tensor(
                cute.recast_ptr(tSrK.iterator, dtype=cutlass.Uint32),
                cute.make_layout(cute.size(tSrK) // 2),
            )
        else:
            tSrQ_copy = smem_thr_Q.retile(tSrQ)
        tSrK_copy = smem_thr_K.retile(tSrK)
        tOrVt_copy = smem_thr_V.retile(tOrVt)
        tSrP = thr_mma.make_fragment_A(thr_mma.partition_A(sP))
        tSrP_copy = smem_thr_Q.retile(tSrP)

        cute.arch.sync_threads()
        if const_expr(mt == 16 and self.regpath):
            # the even/odd-d Q fragments, filled directly from the sQ with
            # the decoded m16k16 A map (the dbg36 recipe minus the warp
            # term: the (1,4,1) keeps every warp on m-atom 0). A generic
            # S2R copy atom's per-lane layout does not match the mma
            # A-fragment, so the fill is manual.
            tSrQ_even = cute.make_rmem_tensor(thr_mma.partition_shape_A((MT, D // 2)), cutlass.BFloat16)
            tSrQ_odd = cute.make_rmem_tensor(thr_mma.partition_shape_A((MT, D // 2)), cutlass.BFloat16)
            for ks in cutlass.range_constexpr(D // 32):
                for e in cutlass.range_constexpr(8):
                    m = tidx % 32 // 4 + 8 * ((e // 2) % 2)
                    j = 16 * ks + 2 * (tidx % 4) + (e % 2) + 8 * (e // 4)
                    tSrQ_even[None, None, ks][e] = sQ[m, Int32(2) * j]
                    tSrQ_odd[None, None, ks][e] = sQ[m, Int32(2) * j + 1]
        else:
            for k in cutlass.range_constexpr(D // 16):
                cute.copy(smem_copy_Q, tSsQ[None, None, k], tSrQ_copy[None, None, k])
        cute.arch.sync_threads()

        n_stages = cute.ceil_div(kv_len_c, SR)
        if kv_len_c > 0:
            self.g2s_stage(
                mK,
                mV,
                mPageTable,
                sK8,
                sK8b,
                sK8c,
                sV8,
                sV8b,
                sV8c,
                tidx,
                kv_start,
                0,
                kv_len_c,
                num_head_kv,
                kv_head,
                batch_idx,
                kv_row0,
                0,
                sK8v if const_expr(mt == 16 and self.regpath) else None,
            )
            cute.arch.cp_async_commit_group()
            self.g2s_stage(
                mK,
                mV,
                mPageTable,
                sK8,
                sK8b,
                sK8c,
                sV8,
                sV8b,
                sV8c,
                tidx,
                kv_start,
                cutlass.min(SR, kv_len_c - 1),
                kv_len_c,
                num_head_kv,
                kv_head,
                batch_idx,
                kv_row0,
                2 if const_expr(mt == 16 and self.regpath) else 1,
                sK8v if const_expr(mt == 16 and self.regpath) else None,
            )
            cute.arch.cp_async_commit_group()
        for stage in cutlass.range(n_stages):
            row0 = stage * SR
            if const_expr(mt == 16 and self.regpath):
                # the 2-pair ring: the wait consumes the oldest of the two
                # outstanding groups (everything up to `stage` has landed);
                # the refill for stage+2 re-targets the buffer this stage
                # just consumed, so it issues after the post-PV barrier at
                # the loop bottom.
                par = 2 * (stage % 2)
                cute.arch.cp_async_wait_group(1)
            else:
                par = stage % 3
                row0_next = cutlass.min((stage + 2) * SR, kv_len_c - 1)
                self.g2s_stage(
                    mK,
                    mV,
                    mPageTable,
                    sK8,
                    sK8b,
                    sK8c,
                    sV8,
                    sV8b,
                    sV8c,
                    tidx,
                    kv_start,
                    row0_next,
                    kv_len_c,
                    num_head_kv,
                    kv_head,
                    batch_idx,
                    kv_row0,
                    (stage + 2) % 3,
                    None,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(2)
            cute.arch.sync_threads()

            srcu_k = sK8u
            srcu_v = sV8u
            if par == 1:
                srcu_k = sK8bu
                srcu_v = sV8bu
            if par == 2:
                srcu_k = sK8cu
                srcu_v = sV8cu
            for i in cutlass.range_constexpr(SR * D // NT // 4):
                e4 = (tidx + i * NT) * 4
                r = e4 // D
                c = e4 % D
                if const_expr(not (mt == 16 and self.regpath)):
                    qk = srcu_k[e4 // 4]
                    qp = cute.arch.prmt(qk, qk, 0x1302)
                    out1 = (qp & _DEQ_MASK_SIGN) | ((qp & _DEQ_MASK_LSB) >> 4)
                    out2 = ((qp << 8) & _DEQ_MASK_SIGN) | (((qp << 8) & _DEQ_MASK_LSB) >> 4)
                    out1 = cute.arch.fma_packed_bf16x2(out1, _DEQ_BIAS, cutlass.Int32(0))
                    out2 = cute.arch.fma_packed_bf16x2(out2, _DEQ_BIAS, cutlass.Int32(0))
                    sK16[r, c] = (out1 & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                    sK16[r, c + 1] = ((out1 >> 16) & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                    sK16[r, c + 2] = (out2 & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                    sK16[r, c + 3] = ((out2 >> 16) & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                qv = srcu_v[e4 // 4]
                qp = cute.arch.prmt(qv, qv, 0x1302)
                out1 = (qp & _DEQ_MASK_SIGN) | ((qp & _DEQ_MASK_LSB) >> 4)
                out2 = ((qp << 8) & _DEQ_MASK_SIGN) | (((qp << 8) & _DEQ_MASK_LSB) >> 4)
                out1 = cute.arch.fma_packed_bf16x2(out1, _DEQ_BIAS, cutlass.Int32(0))
                out2 = cute.arch.fma_packed_bf16x2(out2, _DEQ_BIAS, cutlass.Int32(0))
                sV16[r, c] = (out1 & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                sV16[r, c + 1] = ((out1 >> 16) & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                sV16[r, c + 2] = (out2 & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
                sV16[r, c + 3] = ((out2 >> 16) & 0xFFFF).to(cutlass.Uint16).bitcast(cutlass.BFloat16)
            cute.arch.sync_threads()

            acc_S.fill(0.0)
            if const_expr(mt == 16 and self.regpath):
                # the register-path QK: pass even masks the LOW byte of each
                # fragment u16 into the recipe position (the even d); pass
                # odd needs no shift. The dequant + ldmatrix between the
                # passes breaks the HMMA dependency chain, and the single
                # acc_S (no acc_S2) saves 16 registers under the cap.
                for k in cutlass.range_constexpr(D // 32):
                    cute.copy(smem_copy_K, tSsK[None, None, k], tSrK_copy[None, None, k])
                for i in cutlass.range_constexpr(cute.size(tSrK) // 2):
                    qp = (tSrKu[i] & 0x00FF00FF) << 8
                    out = (qp & _DEQ_MASK_SIGN) | ((qp & _DEQ_MASK_LSB) >> 4)
                    tSrKu[i] = cute.arch.fma_packed_bf16x2(out, _DEQ_BIAS, cutlass.Int32(0))
                for k in cutlass.range_constexpr(D // 32):
                    cute.gemm(tiled_mma, acc_S, tSrQ_even[None, None, k], tSrK[None, None, k], acc_S)
                for k in cutlass.range_constexpr(D // 32):
                    cute.copy(smem_copy_K, tSsK[None, None, k], tSrK_copy[None, None, k])
                for i in cutlass.range_constexpr(cute.size(tSrK) // 2):
                    out = (tSrKu[i] & _DEQ_MASK_SIGN) | ((tSrKu[i] & _DEQ_MASK_LSB) >> 4)
                    tSrKu[i] = cute.arch.fma_packed_bf16x2(out, _DEQ_BIAS, cutlass.Int32(0))
                for k in cutlass.range_constexpr(D // 32):
                    cute.gemm(tiled_mma, acc_S, tSrQ_odd[None, None, k], tSrK[None, None, k], acc_S)
            else:
                acc_S2.fill(0.0)
                for k in cutlass.range_constexpr(D // 16):
                    cute.copy(smem_copy_K, tSsK[None, None, k], tSrK_copy[None, None, k])
                for k in cutlass.range_constexpr(D // 32):
                    cute.gemm(tiled_mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)
                    cute.gemm(tiled_mma, acc_S2, tSrQ[None, None, k + D // 32], tSrK[None, None, k + D // 32], acc_S2)
                for i in cutlass.range_constexpr(cute.size(acc_S)):
                    acc_S[i] = acc_S[i] + acc_S2[i]

            # the row stats: the per-warp partial max over THIS warp's 8
            # S columns (the quad butterfly), stored to the slab; the
            # merge after the barrier gives every warp the global row max.
            acc_S_mn = self._acc_mn_view(acc_S)
            acc_O_mn = self._acc_mn_view(acc_O)
            mcS = cute.make_identity_tensor((1, 1, MT, SR))
            cS = mcS[0, 0, None, None]
            tScS_mn = self._acc_mn_view(thr_mma.partition_C(cS))
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
                if m_local == -cutlass.Float32.inf:
                    m_local = Float32(0.0)
                if tidx % 4 == 0:
                    sStat[(warp * MT + row) * 2] = m_local
            cute.arch.sync_threads()
            for r in cutlass.range_constexpr(n_row_slots):
                row = tScS_mn[r, 0][2]
                m_new = -cutlass.Float32.inf
                for w in cutlass.range_constexpr(4):
                    m_new = cute.arch.fmax(m_new, sStat[(w * MT + row) * 2])
                if m_new == -cutlass.Float32.inf:
                    m_new = Float32(0.0)
                # the deferred-frame anchor: the first stage sets it; the
                # re-anchor only on the >60-log2 drift.
                if row_max[r] == -cutlass.Float32.inf:
                    row_max[r] = m_new
                elif (m_new - row_max[r]) * scale_eff > 60.0:
                    factor = cute.math.exp2((row_max[r] - m_new) * scale_eff, fastmath=True)
                    for c in cutlass.range_constexpr(cute.size(acc_O_mn.shape[1])):
                        acc_O_mn[r, c] = acc_O_mn[r, c] * factor
                    row_sum[r] = row_sum[r] * factor
                    row_max[r] = m_new

            # the P into acc_S + the (16, 32) bf16 tile; the per-warp
            # partial sums to the slab for the second merge.
            for r in cutlass.range_constexpr(n_row_slots):
                row = tScS_mn[r, 0][2]
                tok = row // G
                for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                    stage_row = row0 + tScS_mn[0, c][3]
                    p = Float32(0.0)
                    if stage_row < kv_len_c and kv_start + stage_row <= kv_len - seqlen_q + tok:
                        p = cute.math.exp2(acc_S_mn[r, c] * scale_eff - row_max[r] * scale_eff, fastmath=True)
                    acc_S_mn[r, c] = p
                    sP[row, Int32(tScS_mn[0, c][3])] = p.to(cutlass.BFloat16)
            for r in cutlass.range_constexpr(n_row_slots):
                s_partial = Float32(0.0)
                for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                    s_partial += acc_S_mn[r, c]
                # the quad reduction: the lane holds only its 2 columns'
                # partial — the butterfly merges the quad's 4 before the
                # store (the same reduction the max above does).
                s_partial += cute.arch.shuffle_sync_bfly(s_partial, offset=2, mask=-1, mask_and_clamp=31)
                s_partial += cute.arch.shuffle_sync_bfly(s_partial, offset=1, mask=-1, mask_and_clamp=31)
                if tidx % 4 == 0:
                    row = tScS_mn[r, 0][2]
                    sStat[(warp * MT + row) * 2 + 1] = s_partial
            cute.arch.sync_threads()
            for r in cutlass.range_constexpr(n_row_slots):
                row = tScS_mn[r, 0][2]
                s_total = Float32(0.0)
                for w in cutlass.range_constexpr(4):
                    s_total += sStat[(w * MT + row) * 2 + 1]
                row_sum[r] += s_total

            # the PV: the A = the ldmatrix from the COMPLETED smem P tile
            # (each warp wrote its own columns; the barrier above makes
            # the tile whole) — the per-warp B covers its d-quarter.
            for k in cutlass.range_constexpr(SR // 16):
                cute.copy(smem_copy_Q, tSsP[None, None, k], tSrP_copy[None, None, k])
            for k in cutlass.range_constexpr(cute.size(tOrVt.layout.shape[2])):
                cute.copy(smem_copy_V, tOsVt[None, None, k], tOrVt_copy[None, None, k])
                cute.gemm(tiled_mma, acc_O, tSrP[None, None, k], tOrVt[None, None, k], acc_O)
            cute.arch.sync_threads()
            if const_expr(mt == 16 and self.regpath):
                # the 2-pair refill: the buffer this stage consumed is now
                # free (the barrier above); stage+2 reuses it.
                row0_next = cutlass.min((stage + 2) * SR, kv_len_c - 1)
                self.g2s_stage(
                    mK,
                    mV,
                    mPageTable,
                    sK8,
                    sK8b,
                    sK8c,
                    sV8,
                    sV8b,
                    sV8c,
                    tidx,
                    kv_start,
                    row0_next,
                    kv_len_c,
                    num_head_kv,
                    kv_head,
                    batch_idx,
                    kv_row0,
                    2 * ((stage + 2) % 2),
                    sK8v,
                )
                cute.arch.cp_async_commit_group()

        self.epilogue(
            acc_O,
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
        acc_O: cute.Tensor,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        scale_eff: Float32,
        row0: Int32,
        kv_len_c: Int32,
        kv_len: Int32,
        kv_start: Int32,
        seqlen_q: cutlass.Constexpr[int],
        n_row_slots: cutlass.Constexpr[int],
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
        acc_O_mn = self._acc_mn_view(acc_O)

        mcS = cute.make_identity_tensor((1, 1, MT, self.STAGE_ROWS))
        cS = mcS[0, 0, None, None]
        tScS = thr_mma.partition_C(cS)
        tScS_mn = self._acc_mn_view(tScS)

        # 1. the per-row max (the quad-reduced) with the mask, then the
        #    running O/sum rescale.
        for r in cutlass.range_constexpr(n_row_slots):
            m_local = -cutlass.Float32.inf
            row = tScS_mn[r, 0][2]  # the tile row (0..MT-1)
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
            # The DEFERRED-FRAME softmax: the O accumulates in a FIXED
            # max-frame (row_max = the anchor) — no per-stage acc_O
            # rescale (128 fp32 fmul/stage/lane saved; the frame
            # cancels in the normalize and the LSE is frame-invariant).
            # The P = exp2(S - anchor) may exceed 1 by the logit drift;
            # re-anchor only if the logits climb >60 log2 units above
            # the frame (the exp2(60) P ~1e18, far inside the
            # bf16/fp32 range with the V descales).
            if row_max[r] == -cutlass.Float32.inf:
                # the first stage: set the frame; the acc_O is still 0.
                row_max[r] = m_new
            elif (m_new - row_max[r]) * scale_eff > 60.0:
                factor = cute.math.exp2((row_max[r] - m_new) * scale_eff, fastmath=True)
                for c in cutlass.range_constexpr(cute.size(acc_O_mn.shape[1])):
                    acc_O_mn[r, c] = acc_O_mn[r, c] * factor
                row_sum[r] = row_sum[r] * factor
                row_max[r] = m_new

        # 2. the P = exp2(S*scale - m_new) in place; the per-thread partial
        #    sums; the group reduction; the bf16 P to the smem round-trip.
        for r in cutlass.range_constexpr(n_row_slots):
            s_partial = Float32(0.0)
            row = tScS_mn[r, 0][2]
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

    @cute.jit
    def pv(
        self,
        tiled_mma,
        acc_S: cute.Tensor,
        smem_copy_V,
        tOsVt,
        tOrVt,
        tOrVt_copy,
        acc_O: cute.Tensor,
    ):
        """The PV: the P (the S fragment reinterpreted as the A operand —
        the reference's logical_divide on the fragment layout) x the V."""
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
        for k in cutlass.range_constexpr(cute.size(tOrVt.layout.shape[2])):
            cute.copy(smem_copy_V, tOsVt[None, None, k], tOrVt_copy[None, None, k])
            cute.gemm(tiled_mma, acc_O, tOrS[None, None, k], tOrVt[None, None, k], acc_O)

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
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

        acc_O_mn = self._acc_mn_view(acc_O)
        mcO = cute.make_identity_tensor((1, 1, MT, D))
        cO = mcO[0, 0, None, None]
        tScO_mn = self._acc_mn_view(thr_mma.partition_C(cO))

        # 1. the per-slot normalize (the row_sum is group-uniform, so the
        #    scale is identical across the group — no cross-thread issue).
        for r in cutlass.range_constexpr(n_row_slots):
            s = row_sum[r]
            scale = 1.0 if (s == 0.0 or s != s) else cute.arch.rcp_approx(s)
            for c in cutlass.range_constexpr(cute.size(acc_O_mn.shape[1])):
                acc_O_mn[r, c] = acc_O_mn[r, c] * (scale * v_descale)

        # 2. the O store: the slot's (row, col) from the identity coords.
        for r in cutlass.range_constexpr(n_row_slots):
            R = tScO_mn[r, 0][2]
            if q_rows > R:
                m = R // G
                head = kv_head * G + (R % G)
                for c in cutlass.range_constexpr(cute.size(acc_O_mn.shape[1])):
                    i = tScO_mn[0, c][3]
                    v = acc_O_mn[r, c].to(o_elem)
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
