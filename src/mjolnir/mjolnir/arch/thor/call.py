"""The kernel's host-side builder: tensor/layout construction, TMA atoms, launch.

`_CallMixin.__call__` runs at trace time (plain Python over cute tensors); the
traced `decode`/`reduction` kernels live in t5_kernel.py. Split out of
t5_decode.py for the file-length ratchet.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
from cutlass.cute.typing import *
from typing import Tuple

from mjolnir.quant import NVFP4_GROUP
from mjolnir.arch.thor.const import (
    _note,
    log2_e,
    mma_modes,
    swizzle_align,
    warpgroup_warps,
)
from mjolnir.arch.thor.plan import tile_s_rows


class _CallMixin:
    """Host-side builder mixin for MixedInputFusedMultiHeadAttentionDecode."""

    def _kv_types(self, kv_dtype, mma_dtype):
        """KV-format type plumbing, kept OUT of the traced __call__ (the DSL
        rewrites if/else and ternaries there). Returns (NVFP4, NARROW,
        MMA A-operand dtype, SMEM ring dtype, packed stages)."""
        nvfp4 = self.kv_format == "nvfp4"
        narrow = self.kv_format in ("e3m2", "e2m3", "e2m1")
        if nvfp4:
            assert kv_dtype is cutlass.Float4E2M1FN, kv_dtype
            assert mma_dtype.width == 8, "nvfp4 KV needs fp8 Q"
            return True, False, mma_dtype, mma_dtype, 2
        if narrow:
            assert kv_dtype.width < 8 and mma_dtype.width == 8, (kv_dtype, mma_dtype)
            # A = K/V in fp6/fp4 (MMA reads them directly), SMEM 1-B containers
            return False, True, kv_dtype, cutlass.Int8, 0
        assert kv_dtype is cutlass.Float8E4M3FN, kv_dtype
        return False, False, mma_dtype, kv_dtype, 0

    def _default_kv_stages(self, stages_per_tile):
        """(default ring depth, round smem-limited depths to whole tiles).
        Plain Python (not traced). fp6 KV at 2 CTAs/SM, d=256 (4 stages per
        tile): 2 stages. Depth is monotonic for fp6 at long L — 2 < 3 < 4 <
        5 < 6 in ncu time; 2 beat 3 by 11-14% at 64K/128K/256K (verify ring
        20261011-091005), after 3 had beaten 4 by 4-5% in two earlier A/Bs.
        fp8 keeps the whole tile (4) until its own ring sweep says otherwise."""
        if self.max_kv_stages > 0:
            return self.max_kv_stages, False
        if self.kv_format in ("e3m2", "e2m3") and self.ctas_per_sm == 2 and stages_per_tile == 4:
            return 2, False
        return stages_per_tile * (1 if self.ctas_per_sm == 2 else 2), True

    def _kv_tx_dtype(self, kv_smem_dtype):
        """Element type whose size TMA credits to the KV stage barrier: the
        packed gmem type for the narrow formats, else the SMEM type."""
        return {
            "e3m2": cutlass.Float6E3M2FN,
            "e2m3": cutlass.Float6E2M3FN,
            "e2m1": cutlass.Float4E2M1FN,
        }.get(self.kv_format, kv_smem_dtype)

    def can_implement(self, num_seqs, heads_q, heads_k, headdim, q_dtype, kv_dtype):
        want = {
            "fp8": cutlass.Float8E4M3FN,
            "e3m2": cutlass.Float6E3M2FN,
            "e2m3": cutlass.Float6E2M3FN,
            "e2m1": cutlass.Float4E2M1FN,
            "nvfp4": cutlass.Float4E2M1FN,
        }[self.kv_format]
        if kv_dtype is not want:
            raise testing.CantImplementError(f"kv_format {self.kv_format} needs KV dtype {want}, got {kv_dtype}")
        if self.kv_format != "fp8" and q_dtype is not cutlass.Float8E4M3FN:
            raise testing.CantImplementError(f"kv_format {self.kv_format} runs on the fp8-Q path only")
        if q_dtype not in (cutlass.BFloat16, cutlass.Float8E4M3FN):
            raise testing.CantImplementError("Q must be BFloat16 or Float8E4M3FN")
        if headdim != self.headdim or headdim % 128 != 0:
            raise testing.CantImplementError(f"headdim({headdim}) must be a multiple of 128")
        if heads_q % heads_k != 0:
            raise testing.CantImplementError(f"heads_q({heads_q}) must be a multiple of heads_k({heads_k})")
        fp8 = q_dtype.width == 8
        if fp8:
            need_cvt = 1 if self.kv_format == "nvfp4" else 0
            if self.convert_warpgroups != need_cvt:
                raise testing.CantImplementError(
                    f"fp8-Q path with kv_format {self.kv_format} needs convert_warpgroups={need_cvt}"
                )
            if self.grouped_head_tile % 16 != 0:
                raise testing.CantImplementError("fully-fp8 mma needs grouped_head_tile % 16 == 0")
        elif self.convert_warpgroups < 1:
            raise testing.CantImplementError("bf16 Q path needs cvt warpgroups")

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[
            cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32
        ],  # num_seqs, h_k, n_q (= q_len * h_r packed rows), d
        max_blocks: cutlass.Int32,  # block_table row length
        num_pages: cutlass.Int32,  # pages in the KV pool
        num_work: cutlass.Int32,  # work items in the plan (grid.x)
        strides: Tuple[
            cutlass.Int32,
            cutlass.Int32,  # q_seq, o_seq
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,  # k_page, k_row, k_head
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,  # v_page, v_row, v_head
            cutlass.Int32,
            cutlass.Int32,  # q_descale seq, head
            cutlass.Int32,
            cutlass.Int32,  # k_descale seq, head
            cutlass.Int32,
            cutlass.Int32,  # v_descale seq, head
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,  # ks_page, ks_row, ks_head
            cutlass.Int32,
            cutlass.Int32,
            cutlass.Int32,  # vs_page, vs_row, vs_head
        ],  # element strides (of the KV element type); see STRIDE_FIELDS
        q_iter: cute.Pointer,  # (b, h_k, n_q, d): rows d-contiguous, seq stride free
        k_iter: cute.Pointer,  # pages x rows x heads x d, d contiguous, any other strides
        v_iter: cute.Pointer,  # same for V (may alias K's buffer, e.g. interleaved)
        ks_iter: cute.Pointer,  # nvfp4 only: K scales (e4m3, one per 16 d), else unused
        vs_iter: cute.Pointer,  # nvfp4 only: V scales
        block_table_iter: cute.Pointer,  # (b, max_blocks) int32
        seq_lens_iter: cute.Pointer,  # (b,) int32
        work_iter: cute.Pointer,  # (num_work, 4) int32: seq, tile_begin, tile_end, split
        n_splits_iter: cute.Pointer,  # (b,) int32: chunks per sequence
        q_descale_iter: cute.Pointer,  # fp32, indexed (seq, kv head) via strides
        k_descale_iter: cute.Pointer,  # (stride 0, 0 = one per-tensor scale)
        v_descale_iter: cute.Pointer,
        o_iter: cute.Pointer,  # (b, h_k, n_q, d): rows d-contiguous, seq stride free
        lse_iter: cute.Pointer,  # (b, h_k, n_q) fp32
        o_partial_iter: cute.Pointer,  # (max_splits, b, h_k, n_q, d) fp32
        m_partial_iter: cute.Pointer,  # (max_splits, b, h_k, n_q) fp32
        l_partial_iter: cute.Pointer,  # (max_splits, b, h_k, n_q) fp32
        scale_qs: cutlass.Float32,  # softmax scale (1/sqrt(d) by default)
        scale_o: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        ##############################
        # TiledMma creation
        ##############################
        mma_dtype = q_iter.dtype
        kv_dtype = k_iter.dtype
        acc_dtype = cutlass.Float32
        NVFP4, NARROW, kv_mma_dtype, kv_smem_dtype, self.packed_stages = self._kv_types(kv_dtype, mma_dtype)
        assert v_iter.dtype is kv_dtype
        assert o_partial_iter.dtype is acc_dtype
        assert m_partial_iter.dtype is acc_dtype
        assert l_partial_iter.dtype is acc_dtype
        assert lse_iter.dtype is acc_dtype

        FP8 = mma_dtype.width == 8
        if cutlass.const_expr(FP8):
            assert self.convert_warpgroups == int(NVFP4)
        else:
            assert self.convert_warpgroups >= 1

        PS = self.page_size
        max_splits = self.max_splits

        # Block tile sets the granularity at which threadblocks consume work
        blk_tile_s = tile_s_rows
        blk_tile_h = self.grouped_head_tile
        blk_tile_d = self.headdim
        blk_tile_shd = (blk_tile_s, blk_tile_h, blk_tile_d)

        # MMA tile sets the granularity at which TMAs + MMAs are issued
        mma_tile_m = 128
        mma_tile_n = self.grouped_head_tile
        mma_tile_k = 128
        mma_tile_mnk = (mma_tile_m, mma_tile_n, mma_tile_k)

        # A-operand source: fp8 reads K/V straight from the TMA'd SMEM ring;
        # bf16 reads the cvt output from TMEM.
        a_operand_source = tcgen05.OperandSource.SMEM if cutlass.const_expr(FP8) else tcgen05.OperandSource.TMEM

        # GEMM1: S(s, n) = K(s, d) . Q(n, d) — A-operand K.
        # Narrow formats: A = K/V in fp6/fp4, B = Q/P in e4m3 (kind::f8f6f4
        # takes mixed A/B types). nvfp4 is converted to e4m3 before the MMA.
        # (kv_mma_dtype / kv_smem_dtype come from self._kv_types)
        tiled_mma_kq = sm100_utils.make_trivial_tiled_mma(
            kv_mma_dtype,
            mma_dtype,
            OperandMajorMode.K,  # K
            OperandMajorMode.K,  # Q
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
            a_operand_source,
        )

        # GEMM2: O(d, n) = V(d, s) . P(s, n). V is headdim-major (MN-major)
        # in SMEM on the fp8 path; on the bf16 path V comes from TMEM, which
        # must be K-major. The a-major must match smem_layout_v's orientation.
        v_major_mode = OperandMajorMode.MN if cutlass.const_expr(FP8) else OperandMajorMode.K
        tiled_mma_vp = sm100_utils.make_trivial_tiled_mma(
            kv_mma_dtype,
            mma_dtype,
            v_major_mode,
            OperandMajorMode.MN,  # P
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
            a_operand_source,
        )

        # Q stages
        self.q_stages = blk_tile_d // mma_tile_k

        # Perf heuristics
        max_cvt_stages = 4 if self.grouped_head_tile == 32 else 8
        # KV ring depth: a stage is freed only when the GEMM that read it
        # completes, so loads for tile s+2 wait on GEMM2(s). Deeper rings let
        # the loader run further ahead of the softmax/GEMM chain at long L.
        # Default depth = a WHOLE number of KV tiles (one tile = tiles_dk K
        # stages + tiles_dm V stages; 4 at d=256): 1 tile at 2 CTAs/SM, 2 at
        # 1 CTA/SM. Measured on Thor (ncu, cold L2, d=256): 2/SM depth 4 beat
        # 2, 3, 5, 6 at 8K/64K/256K, and 1/SM depth 8 beat 13. Ring depths
        # that are not tile multiples rotate a tile's K/V across slots and
        # stream worse on LPDDR5X; deeper is not better.
        # Exception, fp6 KV (e3m2/e2m3) at 2/SM, d=256: shallower is faster —
        # depth 2 beat 3 by 11-14% at 64K-256K, 3 beat 4 by 4-5%, 6 was worst
        # — see _default_kv_stages.
        stages_per_tile = (blk_tile_d // mma_tile_k) + (blk_tile_d // mma_tile_m)
        # Plain ternaries, not if/else: the DSL stages if/else into regions
        # whose bindings don't escape (NameError after the block).
        max_kv_stages, round_to_tile = self._default_kv_stages(stages_per_tile)

        # TMEM: S stages + O + (bf16 only) the cvt ring
        tmem_alloc_cols = mma_tile_n * self.sp_stages
        tmem_alloc_cols += mma_tile_n * self.o_stages * (blk_tile_d // mma_tile_m)
        tmem_capacity = 512 // self.ctas_per_sm  # 512 cols per SM, split by CTAs
        cvt_stage_cols = mma_tile_k * mma_dtype.width // 32
        if cutlass.const_expr(FP8):
            self.cvt_stages = 0
        else:
            self.cvt_stages = min((tmem_capacity - tmem_alloc_cols) // cvt_stage_cols, max_cvt_stages)
        tmem_alloc_cols += self.cvt_stages * cvt_stage_cols
        self.tmem_alloc_cols = 2 ** math.ceil(math.log2(tmem_alloc_cols))  # PO2
        # A second co-resident CTA blocks in tcgen05.alloc if the first holds
        # more than its share.
        assert self.tmem_alloc_cols <= tmem_capacity

        _note(f"\tcvt stages: {self.cvt_stages}")

        # SMEM budget -> KV stages
        mbarrier_reserved_bytes = 768
        smem_alloc_bits = mbarrier_reserved_bytes * 8
        smem_alloc_bits += mma_tile_n * acc_dtype.width  # colmax
        smem_alloc_bits += mma_tile_n * warpgroup_warps * acc_dtype.width  # colsum
        smem_alloc_bits += mma_tile_n * mma_tile_k * self.q_stages * mma_dtype.width  # Q
        smem_alloc_bits += mma_tile_m * mma_tile_n * self.sp_stages * mma_dtype.width  # P
        smem_alloc_bits += 3 * swizzle_align * 8  # worst-case padding: Q, KV, P
        # nvfp4: packed e2m1 tiles (TMA -> convert warpgroup), one K + one V
        # tile in flight; the e4m3 ring below is what the MMA reads
        smem_alloc_bits += self.packed_stages * (blk_tile_s * blk_tile_d * 4 + 128 * 8)

        # Thor (sm_110a): 228 KB smem per SM, 1 KB reserved per resident CTA,
        # at most 227 KB (232448 B) per CTA. At 2 CTAs/SM each gets ~113 KB
        # (6 KV stages instead of 8 — still 96 KB of KV in flight per CTA).
        smem_capacity = min(232448, 233472 // self.ctas_per_sm - 1024)
        # SMEM ring dtype (from self._kv_types): 1-byte containers for the
        # narrow formats (TMA unpacks), e4m3 for fp8 and converted nvfp4
        self.kv_stages = (smem_capacity * 8 - smem_alloc_bits) // (mma_tile_m * mma_tile_k * kv_smem_dtype.width)
        self.kv_stages = min(self.kv_stages, max_kv_stages)
        # smem-limited below the target: keep a whole number of tiles
        self.kv_stages = (
            self.kv_stages - self.kv_stages % stages_per_tile
            if round_to_tile and self.kv_stages >= stages_per_tile
            else self.kv_stages
        )

        _note(f"\tkv stages: {self.kv_stages}, kv format: {self.kv_format}, packed stages: {self.packed_stages}")
        _note(f"\tpdl: {self.pdl}, l2 prefetch: {self.l2_prefetch} tiles ahead, kv evict_first: {self.kv_evict_first}")

        ##############################
        # Global tensors
        ##############################
        b, h_k, n_q, d = problem_shape
        (
            q_seq,
            o_seq,
            k_page,
            k_row,
            k_head,
            v_page,
            v_row,
            v_head,
            qd_seq,
            qd_head,
            kd_seq,
            kd_head,
            vd_seq,
            vd_head,
            ks_page,
            ks_row,
            ks_head,
            vs_page,
            vs_row,
            vs_head,
        ) = strides

        # Q: (n_q, d, (h_k, b)). Rows of one sequence are packed (n_q, h_k)
        # with d contiguous; the per-sequence stride is free so vLLM's
        # fused-QKV `query` view is read in place (TMA strides only need
        # 16-B multiples).
        q = cute.make_tensor(
            q_iter,
            cute.make_layout((n_q, d, (h_k, b)), stride=(d, 1, (n_q * d, q_seq))),
        )
        # KV cache, any layout with d contiguous (vLLM NHD, the interleaved
        # (blocks, heads, tokens, 2*d) layout, FlashInfer (blocks, 2, ...)):
        #   K view: (PS, d, (h_k, num_pages))   — GEMM1 A-operand (M=s, K=d)
        #   V view: (d, PS, (h_k, num_pages))   — GEMM2 A-operand (M=d, K=s)
        # A page of one KV head is a (PS, d) block: page index and head
        # together are the L-mode, so a page-table lookup is just a TMA
        # coordinate.
        k = cute.make_tensor(
            k_iter,
            cute.make_layout((PS, d, (h_k, num_pages)), stride=(k_row, 1, (k_head, k_page))),
        )
        v = cute.make_tensor(
            v_iter,
            cute.make_layout((d, PS, (h_k, num_pages)), stride=(1, v_row, (v_head, v_page))),
        )
        # nvfp4: packed K and V are both token-major (PS, d) blocks; scales are
        # (PS, d/16) e4m3 per (head, page). Unused (dummy) on other formats.
        v_tok = cute.make_tensor(
            v_iter,
            cute.make_layout((PS, d, (h_k, num_pages)), stride=(v_row, 1, (v_head, v_page))),
        )
        n_groups = d // NVFP4_GROUP
        k_scales = cute.make_tensor(
            ks_iter,
            cute.make_layout((PS, n_groups, (h_k, num_pages)), stride=(ks_row, 1, (ks_head, ks_page))),
        )
        v_scales = cute.make_tensor(
            vs_iter,
            cute.make_layout((PS, n_groups, (h_k, num_pages)), stride=(vs_row, 1, (vs_head, vs_page))),
        )
        block_table = cute.make_tensor(
            block_table_iter,
            cute.make_layout((b, max_blocks), stride=(max_blocks, 1)),  # row-major
        )
        seq_lens = cute.make_tensor(seq_lens_iter, cute.make_layout((b,)))
        work = cute.make_tensor(work_iter, cute.make_layout((num_work, 4), stride=(4, 1)))
        if cutlass.const_expr(self.fused_merge):
            # + the appended counter region (see SplitPlan.n_splits_tensor),
            # laid out (q tiles, kv heads, seq) after the per-seq n_splits
            n_tiles_c = cute.ceil_div(n_q, self.grouped_head_tile)
            n_splits = cute.make_tensor(n_splits_iter, cute.make_layout((b + n_tiles_c * h_k * b,)))
        else:
            n_splits = cute.make_tensor(n_splits_iter, cute.make_layout((b,)))
        # Descales indexed by coord_hb = (kv head, seq); stride 0 broadcasts
        # a per-tensor scale (vLLM layer._k_scale) with no copy.
        q_descale = cute.make_tensor(q_descale_iter, cute.make_layout(((h_k, b),), stride=((qd_head, qd_seq),)))
        k_descale = cute.make_tensor(k_descale_iter, cute.make_layout(((h_k, b),), stride=((kd_head, kd_seq),)))
        v_descale = cute.make_tensor(v_descale_iter, cute.make_layout(((h_k, b),), stride=((vd_head, vd_seq),)))

        o = cute.make_tensor(
            o_iter,
            cute.make_layout((d, n_q, (h_k, b)), stride=(1, d, (n_q * d, o_seq))),
        )
        lse = cute.make_tensor(
            lse_iter,
            cute.make_ordered_layout(shape=(n_q, (h_k, b)), order=(0, (1, 2))),
        )
        o_partial = cute.make_tensor(
            o_partial_iter,
            cute.make_ordered_layout(shape=(d, n_q, (h_k, b), max_splits), order=(0, 1, (2, 3), 4)),
        )
        m_partial = cute.make_tensor(
            m_partial_iter,
            cute.make_ordered_layout(shape=(n_q, (h_k, b), max_splits), order=(0, (1, 2), 3)),
        )
        l_partial = cute.make_tensor(
            l_partial_iter,
            cute.make_ordered_layout(shape=(n_q, (h_k, b), max_splits), order=(0, (1, 2), 3)),
        )

        ##############################
        # SMEM layouts
        ##############################
        # (MMA, MMA_M/N, MMA_K, Stages)
        smem_layout_q = sm100_utils.make_smem_layout_b(tiled_mma_kq, mma_tile_mnk, mma_dtype, self.q_stages)
        smem_layout_k = sm100_utils.make_smem_layout_a(tiled_mma_kq, mma_tile_mnk, kv_smem_dtype, self.kv_stages)
        smem_layout_v = sm100_utils.make_smem_layout_a(
            tiled_mma_vp, mma_tile_mnk, kv_smem_dtype, self.kv_stages, is_k_major=False
        )  # V is headdim-major (GEMM2 M-major) in gmem + smem

        # O tile shape (used only to derive the TMEM O fragment / softmax
        # partitions; the epilogue stores O from registers).
        smem_layout_atom_o = tcgen05.make_smem_layout_atom(tcgen05.mma.SmemLayoutAtomKind.MN_SW128, acc_dtype)
        smem_layout_o = cute.tile_to_shape(smem_layout_atom_o, (blk_tile_d, blk_tile_h), order=(1, 0))
        smem_layout_o = cute.flat_divide(smem_layout_o, (mma_tile_m, mma_tile_n))

        ##############################
        # TMA atoms
        ##############################
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()

        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            q,
            cute.select(smem_layout_q, mma_modes),
            mma_tile_mnk,
            tiled_mma_kq,
        )

        if cutlass.const_expr(NVFP4):
            # Packed e2m1 boxes of one page x the whole head dim, into a plain
            # row-major (token, d) packed tile; the convert warpgroup reads it.
            packed_box = cute.make_layout((PS, blk_tile_d), stride=(blk_tile_d, 1))
            tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op, k, packed_box, (PS, blk_tile_d)
            )
            tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op, v_tok, packed_box, (PS, blk_tile_d)
            )
        elif cutlass.const_expr(PS == blk_tile_s):
            # One page == one KV tile: the same full-stage TMA box as a dense
            # KV tensor, indexed by (head, page) in the L-mode.
            tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                k,
                cute.select(smem_layout_k, mma_modes),
                mma_tile_mnk,
                tiled_mma_kq,
                internal_type=kv_smem_dtype,
            )
            tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                v,
                cute.select(smem_layout_v, mma_modes),
                mma_tile_mnk,
                tiled_mma_vp,
                internal_type=kv_smem_dtype,
            )
        else:
            # PS < 128: a KV tile spans 128 / PS pages, one TMA box per page.
            # The MMA-partitioned stage (MMA, #MMA_M, #MMA_K) flattens colex
            # to m + 128 * k, so (rows, cols) sub-boxes are relabeled with a
            # (1, 128)-strided layout; the swizzle stays in the ComposedLayout.
            k_box = cute.composition(
                cute.select(smem_layout_k, mma_modes),
                cute.make_layout((PS, mma_tile_k), stride=(1, mma_tile_m)),
            )  # (s rows of one page, d)
            v_box = cute.composition(
                cute.select(smem_layout_v, mma_modes),
                cute.make_layout((mma_tile_m, PS), stride=(1, mma_tile_m)),
            )  # (d, s rows of one page)
            tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op, k, k_box, (PS, mma_tile_k), internal_type=kv_smem_dtype
            )
            tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op, v, v_box, (mma_tile_m, PS), internal_type=kv_smem_dtype
            )

        ##############################
        # Decode kernel launch
        ##############################
        scale_qs_log2_e = scale_qs * log2_e

        n_tiles = cute.ceil_div(n_q, blk_tile_h)
        grid = (num_work, n_tiles, h_k)  # one CTA per (work item, q tile, kv head)

        self.decode(
            blk_tile_shd,
            mma_tile_mnk,
            tiled_mma_kq,
            tiled_mma_vp,
            mma_dtype,
            kv_smem_dtype,
            smem_layout_q,
            tma_atom_q,
            tma_tensor_q,
            smem_layout_k,
            tma_atom_k,
            tma_tensor_k,
            smem_layout_v,
            tma_atom_v,
            tma_tensor_v,
            smem_layout_o,
            block_table,
            seq_lens,
            work,
            n_splits,
            q_descale,
            k_descale,
            v_descale,
            k_scales,
            v_scales,
            o,
            lse,
            o_partial,
            m_partial,
            l_partial,
            h_k,
            n_q,
            b,  # num_seqs: the counter region base offset
            num_pages,
            scale_qs,
            scale_qs_log2_e,
            scale_o,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=self.ctas_per_sm,  # caps regs/thread for 2 CTAs/SM
            **self.launch_extra,
        )

        ##############################
        # Split-KV reduction (merges only sequences the plan split)
        ##############################
        # Compiled out when the plan splits no sequence (every sequence wrote
        # its final output in decode): saves a ~7 us no-op launch.
        if cutlass.const_expr(max_splits > 1 and self.with_reduction):
            h_q = h_k * n_q
            o_r = cute.make_tensor(o_iter, cute.make_layout((d, h_q, b), stride=(1, d, o_seq)))
            lse_r = cute.make_tensor(lse_iter, cute.make_layout((h_q, b)))
            o_partial_r = cute.make_tensor(o_partial_iter, cute.make_layout((d, h_q, b, max_splits)))
            m_partial_r = cute.make_tensor(m_partial_iter, cute.make_layout((h_q, b, max_splits)))
            l_partial_r = cute.make_tensor(l_partial_iter, cute.make_layout((h_q, b, max_splits)))
            d_per_blk = 128
            d_blks = cute.ceil_div(d, d_per_blk)
            self.reduction(
                o_r,
                lse_r,
                o_partial_r,
                m_partial_r,
                l_partial_r,
                n_splits,
                scale_o,
                self.pdl,
            ).launch(
                grid=[d_blks, h_q, b],
                block=[d_per_blk, 1, 1],
                cluster=[1, 1, 1],
                stream=stream,
                **self.launch_extra,
            )
