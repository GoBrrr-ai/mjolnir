"""The sm_87 MMA decode kernel — Phase 1 (M1): fp8 KV, bf16 Q, seqlen_q=1.

The design lives in research/runs/orin1/notes/mma-kernel-design.md; the
evidence for why the GEMV's scalar-issue roofline caps it at ~20 GB/s
lives in gemv-vs-mma-bottleneck.md. The structure adapts NVIDIA's
Ampere FlashAttention-2 DSL example (research/runs/orin1/sources/
ampere-fa2-dsl/fa2_ampere.py) to the decode shape:

- CTA = one (kv-head, split, batch); 128 threads = 4 warps
- The M tile: the 6 GQA q-heads padded into the mma M=16 rows (one q
  token; the MTP shapes land in M3)
- The KV: 32 rows per stage — the fp8 rows cp.async'd raw (the paged
  gather, the GEMV's make_ptr idiom, the tail clamp instead of the
  branch), dequantized fp8->bf16 IN SMEM, then the reference's
  ldmatrix -> mma.m16n8k16 path
- The warps split the stage's rows (the N of QK^T) => the row softmax
  stats reduce across the warps through smem (the reference's 4 warps
  split M, so its stats are warp-local). The PV splits the head_dim
  across the warps (each warp accumulates its 8-wide d slices — 32
  fp32 registers/thread), so O never reduces across warps.
- The V smem is staged TRANSPOSED ([d][row]) like the reference's sVt,
  which is what its transpose=True B-atom reads.
- The descales follow the GEMV's contract: the K scale folds into
  scale_eff (the S side), the V scale divides at the epilogue, the
  LSE = (m + log2 l) * ln2.
- The split-KV partial O/LSE per (split, row) — the GEMV's exact flat
  offset math; merged by the existing torch combine.

Status: M1 in progress. NOT wired into the default paths; `--impl mma`
exercises it.
"""

from __future__ import annotations


import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp import LdMatrix8x8x16bOp, MmaF16BF16Op
from cutlass.pipeline import NamedBarrier

import cuda.bindings.driver as cuda

from mjolnir.arch.orin.decode import DescaleTensors, GEMVDecodeKernel, _e4m3_to_f32


class MMADecodeKernel:
    """The traced CuTe kernel. The arg surface mirrors the GEMV's
    GEMVDecodeKernel.__call__ so the runner's packing is shared."""

    HEAD_DIM = GEMVDecodeKernel.HEAD_DIM  # 256
    M_TILE = 16  # the mma M: the 6 GQA heads + the padding rows
    STAGE_ROWS = 32  # the KV rows per smem stage
    NUM_THREADS = 128  # 4 warps
    WARPS = 4

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
        self.cta_sync_barrier = NamedBarrier(barrier_id=1, num_threads=self.NUM_THREADS)
        self.stats_barrier = NamedBarrier(barrier_id=2, num_threads=self.NUM_THREADS)

    # -- the smem layout helper (the reference's 8x64 swizzle atom) -------
    @staticmethod
    @cute.jit
    def _bf16_tile_layout(rows: cutlass.Constexpr[int], cols: cutlass.Constexpr[int]):
        atom = cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1)))
        return cute.tile_to_shape(atom, (rows, cols), order=(0, 1))

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
        assert const_expr(seqlen_q == 1), "the MTP shapes (seqlen_q>1) land in M3"
        assert const_expr(mK.element_type.width == 8 and mV.element_type.width == 8), (
            "the M1 MMA kernel takes fp8 KV only"
        )
        assert const_expr(mQ.element_type == cutlass.BFloat16), "the fp8 Q lands with M3 (the pack is bf16-Q only for now)"
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

        # The O/LSE flat row offset (the GEMV's exact math; the q rows of
        # this kv_head start at qo_head = kv_head * g).
        o_off = q_row_base * num_head * D + kv_head * self.g * D
        if const_expr(mO_partial is not None):
            if const_expr(is_varlen_q):
                split_stride = mO_partial.shape[1] * num_head * D
            else:
                split_stride = mO_partial.shape[1] * mO_partial.shape[2] * num_head * D
            o_off = o_off + split_idx * split_stride

        # ---- smem -------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(cutlass.BFloat16, self._bf16_tile_layout(MT, D), byte_alignment=16)
        sK8 = smem.allocate_tensor(mK.element_type, cute.make_layout(SR * D), byte_alignment=16)
        sV8 = smem.allocate_tensor(mV.element_type, cute.make_layout(SR * D), byte_alignment=16)
        sK16 = smem.allocate_tensor(cutlass.BFloat16, self._bf16_tile_layout(SR, D), byte_alignment=16)
        # The V staged TRANSPOSED ([d][row]) — the reference's sVt, which
        # its transpose=True B-atom reads.
        sV16 = smem.allocate_tensor(cutlass.BFloat16, self._bf16_tile_layout(D, SR), byte_alignment=16)
        # The cross-warp row-stat scratch: [warp][MT] for the max, then
        # [WARPS*MT .. ] for the partial sums.
        sStats = smem.allocate_tensor(cutlass.Float32, cute.make_layout(self.WARPS * MT * 2), byte_alignment=16)

        g2s_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            mK.element_type,
            num_bits_per_copy=128,
        )

        # ---- stage the Q once: the 6 real rows, the zero padding -------
        # mQ: (b, M, h_q, d) bf16; row (token, head) at
        # (batch*seqlen_q + m)*num_head + head — the M=1: m=0.
        for i in cutlass.range_constexpr(MT * D // NT):
            e = tidx + i * NT
            r = e // D
            c = e % D
            if r < self.g:
                sQ[r, c] = cutlass.BFloat16(mQ[batch_idx, Int32(0), kv_head * self.g + r, c])
            else:
                sQ[r, c] = cutlass.BFloat16(0.0)

        # ---- the mma machinery ------------------------------------------
        tiled_mma = cute.make_tiled_mma(
            MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
            (1, self.WARPS, 1),
            permutation_mnk=(MT, self.WARPS * 8, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)
        sQ_t = cute.make_tensor(sQ.iterator, self._bf16_tile_layout(MT, D))
        sK16_t = cute.make_tensor(sK16.iterator, self._bf16_tile_layout(SR, D))
        sV16_t = cute.make_tensor(sV16.iterator, self._bf16_tile_layout(D, SR))

        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ_t))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK16_t))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sV16_t))
        acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((MT, SR)), cutlass.Float32)
        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((MT, D)), cutlass.Float32)
        acc_O.fill(0.0)
        row_max = cute.make_rmem_tensor((MT,), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((MT,), cutlass.Float32)
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
        tSsQ = smem_thr_Q.partition_S(sQ_t)
        tSsK = smem_thr_K.partition_S(sK16_t)
        tOsVt = smem_thr_V.partition_S(sV16_t)

        # The stage loop over this CTA's KV chunk (single-buffered in M1).
        n_stages = cute.ceil_div(kv_len_c, SR)
        for stage in cutlass.range(n_stages):
            row0 = stage * SR

            # 1. g2s: 32 rows x 256 B = 4 x 16B chunks per thread. The tail
            #    rows CLAMP to the last valid row (the GEMV's trick — no
            #    branch); the softmax masks the overhang by kv_len_c.
            for i in cutlass.range_constexpr(SR * D // NT // 16):
                e16 = tidx + i * NT  # the 16-byte chunk index within the stage
                r = e16 // (D // 16)
                ck = (e16 % (D // 16)) * 16
                gk_row_raw = kv_start + row0 + r
                gk_row = cutlass.min(gk_row_raw, kv_start + kv_len_c - 1)
                if const_expr(mPageTable is not None):
                    page = mPageTable[batch_idx, gk_row // self.page_size]
                    kv_base = (page * self.page_size + (gk_row - page * self.page_size)) * num_head_kv + kv_head
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
                k_dst = cute.make_tensor(
                    cute.make_ptr(
                        mK.element_type,
                        (sK8.iterator + e16 * 16).toint(),
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
                        (sV8.iterator + e16 * 16).toint(),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout(16),
                )
                cute.copy(g2s_atom, k_src, k_dst)
                cute.copy(g2s_atom, v_src, v_dst)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # 2. the dequant fp8 -> bf16 in smem. The K: the same [row][d]
            #    shape; the V: transposed to [d][row] for the B-atom.
            for i in cutlass.range_constexpr(SR * D // NT):
                e = tidx + i * NT
                r = e // D
                c = e % D
                sK16[r, c] = cutlass.BFloat16(_e4m3_to_f32(sK8[e]))
                sV16[c, r] = cutlass.BFloat16(_e4m3_to_f32(sV8[e]))
            cute.arch.sync_threads()

            # 3. QK^T: the acc_S = Q * K^T for this stage's rows.
            acc_S.fill(0.0)
            n_kslices = D // 16
            for k in cutlass.range_constexpr(n_kslices):
                cute.copy(smem_copy_Q, tSsQ[None, None, k], tSrQ[None, None, k])
                cute.copy(smem_copy_K, tSsK[None, None, k], tSrK[None, None, k])
            for k in cutlass.range_constexpr(n_kslices):
                cute.gemm(tiled_mma, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)

            # 4. the online softmax (the cross-warp row stats).
            self.online_softmax(
                acc_S,
                acc_O,
                row_max,
                row_sum,
                sStats,
                scale_eff,
                row0,
                kv_len_c,
            )

            # 5. PV: the acc_O += P * V (the warp's d-slices).
            self.pv(tiled_mma, smem_copy_V, tOsVt, tOrVt, acc_S, acc_O, MT, SR)

        # 6. the epilogue.
        self.epilogue(
            acc_O,
            row_sum,
            row_max,
            scale_eff,
            mO,
            mLSE,
            mO_partial,
            mLSE_partial,
            o_off,
            num_head,
            D,
            v_descale,
            seqlen_q,
            is_varlen_q,
            batch_idx,
            kv_head,
            split_idx,
        )

    # -- the helpers ------------------------------------------------------
    @staticmethod
    @cute.jit
    def _acc_mn_view(acc: cute.Tensor) -> cute.Tensor:
        """The (M, N)-indexed view of an mma accumulator fragment (the
        reference's _make_acc_tensor_mn_view)."""
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
        return cute.make_tensor(acc.iterator, mn)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        acc_O: cute.Tensor,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        sStats: cute.Tensor,
        scale_eff: Float32,
        row0: Int32,
        kv_len_c: Int32,
    ):
        """The per-stage online softmax. The warps split the stage's rows
        (the N of the S tile), so the row stats reduce across the warps
        through sStats: each warp's quad-reduced per-row partial lands in
        its smem slot, then every warp computes the identical global m/l.
        The kv_len_c bound masks the chunk's overhang rows (the clamp
        duplicated the last valid row into them)."""
        MT = self.M_TILE
        acc_S_mn = self._acc_mn_view(acc_S)
        acc_O_mn = self._acc_mn_view(acc_O)
        warp_idx = cute.arch.thread_idx()[0] // 32

        # 1. the per-warp partial row max (the quad-reduced).
        for r in cutlass.range_constexpr(MT):
            m_local = -cutlass.Float32.inf
            for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                if c < kv_len_c - row0:  # the stage-local row bound
                    m_local = cute.arch.fmax(m_local, acc_S_mn[r, c])
            m_local = cute.arch.shuffle_sync_bfly(m_local, offset=1)
            m_local = cute.arch.shuffle_sync_bfly(m_local, offset=2)
            sStats[warp_idx * MT + r] = m_local
        self.stats_barrier.arrive_and_wait()

        # 2. the global row max + the O/sum rescale.
        for r in cutlass.range_constexpr(MT):
            m_new = -cutlass.Float32.inf
            for w in cutlass.range_constexpr(self.WARPS):
                m_new = cute.arch.fmax(m_new, sStats[w * MT + r])
            if m_new == -cutlass.Float32.inf:
                m_new = Float32(0.0)  # the all-masked row: keep the O at 0
            m_prev = row_max[r]
            row_max[r] = m_new
            rescale = cute.math.exp2(m_prev * scale_eff - m_new * scale_eff, fastmath=True)
            for c in cutlass.range_constexpr(cute.size(acc_O_mn.shape[1])):
                acc_O_mn[r, c] = acc_O_mn[r, c] * rescale
            row_sum[r] = row_sum[r] * rescale
        self.stats_barrier.arrive_and_wait()

        # 3. the P = exp2(S*scale - m_new) in place; the partial sums.
        for r in cutlass.range_constexpr(MT):
            s_partial = Float32(0.0)
            for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                v = acc_S_mn[r, c]
                if c < kv_len_c - row0:
                    p = cute.math.exp2(v * scale_eff - row_max[r] * scale_eff, fastmath=True)
                else:
                    p = Float32(0.0)
                acc_S_mn[r, c] = p
                s_partial += p
            sStats[self.WARPS * MT + warp_idx * MT + r] = s_partial
        self.stats_barrier.arrive_and_wait()

        # 4. the global row sum.
        for r in cutlass.range_constexpr(MT):
            s_total = Float32(0.0)
            for w in cutlass.range_constexpr(self.WARPS):
                s_total += sStats[self.WARPS * MT + w * MT + r]
            row_sum[r] += s_total

    @cute.jit
    def pv(
        self,
        tiled_mma,
        smem_copy_V,
        tOsVt,
        tOrVt,
        acc_S: cute.Tensor,
        acc_O: cute.Tensor,
        m_block: cutlass.Constexpr[int],
        n_block: cutlass.Constexpr[int],
    ):
        """The PV: the P (the S fragment reinterpreted as the A operand —
        the 16x8 D-layout of the QK mma matches the 16x8 K-layout of the
        PV mma, so the conversion is in place) x the V fragments."""
        rP = cute.make_fragment_like(acc_S, cutlass.BFloat16)
        for i in cutlass.range_constexpr(cute.size(rP)):
            rP[i] = cutlass.BFloat16(acc_S[i])
        # ((the atom-M), (the atom-N)) -> ((the atom-M, 2), the rest) — the
        # reference's logical_divide trick: the PV's A operand expects the
        # K dim split in 2s.
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            cute.copy(smem_copy_V, tOsVt[None, None, k], tOrVt[None, None, k])
            cute.gemm(tiled_mma, acc_O, tOrS[None, None, k], tOrVt[None, None, k], acc_O)

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        row_sum: cute.Tensor,
        row_max: cute.Tensor,
        scale_eff: Float32,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        mO_partial: cute.Tensor | None,
        mLSE_partial: cute.Tensor | None,
        o_off: Int32,
        num_head: Int32,
        D: cutlass.Constexpr[int],
        v_descale: Float32,
        seqlen_q: cutlass.Constexpr[int],
        is_varlen_q: cutlass.Constexpr[bool],
        batch_idx: Int32,
        kv_head: Int32,
        split_idx: Int32,
    ):
        """The normalize + the store. The final mode: the O + the LSE at
        o_off; the split mode: the fp32 partials (the GEMV's offset
        math). Only the 6 real head rows are written; the row r's writer
        is the thread with lane%4==0 and lane//4==r (it holds the
        quad-reduced row sum)."""
        MT = self.M_TILE
        acc_O_mn = self._acc_mn_view(acc_O)
        tidx = cute.arch.thread_idx()[0]
        lane = tidx % 32
        o_elem = mO_partial.element_type if const_expr(mO_partial is not None) else mO.element_type
        o_base = mO_partial.iterator if const_expr(mO_partial is not None) else mO.iterator
        row_stride = num_head * D  # the dense stride of one token row in O

        # the O: the row r's fragment cols spread over the 4 lanes of the
        # row's quad; each lane writes its own 2-column pairs through the
        # whole d via the acc_O fragment's natural order.
        for i in cutlass.range_constexpr(cute.size(acc_O_mn.shape[1])):
            for r in cutlass.range_constexpr(MT):
                if r < self.g:
                    s = row_sum[r]
                    scale = 1.0 if (s == 0.0 or s != s) else cute.arch.rcp_approx(s)
                    v = (acc_O_mn[r, i] * scale * v_descale).to(o_elem)
                    o_dst = cute.make_tensor(
                        cute.make_ptr(
                            o_elem,
                            (o_base + o_off + r * row_stride + i).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        ),
                        cute.make_layout(1),
                    )
                    # the rmem->gmem: the direct element write via the ptr
                    o_dst[0] = v

        # the LSE: one writer per row (the quad's first lane).
        if const_expr(mLSE is not None or mLSE_partial is not None):
            lse_tensor = mLSE_partial if const_expr(mLSE_partial is not None) else mLSE
            for r in cutlass.range_constexpr(MT):
                if r < self.g and lane == r * 4:
                    # the row r's quad leader holds the (quad-reduced) max
                    # and sum in its row_max/row_sum registers.
                    m_r = row_max[r]
                    l_r = row_sum[r]
                    lse = (m_r * scale_eff + cute.math.log2(cutlass.max(l_r, Float32(1e-30)))) * 0.6931471805599453
                    qo_head = kv_head * self.g + r
                    if const_expr(mLSE_partial is not None):
                        if const_expr(is_varlen_q):
                            lse_tensor[split_idx, qo_head, batch_idx * seqlen_q + 0] = lse
                        else:
                            lse_tensor[split_idx, qo_head, batch_idx] = lse
                    else:
                        if const_expr(is_varlen_q):
                            lse_tensor[qo_head, batch_idx * seqlen_q + 0] = lse
                        else:
                            lse_tensor[qo_head, batch_idx] = lse
