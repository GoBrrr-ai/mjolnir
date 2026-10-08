# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import math
import os
from typing import Type, Tuple, Optional
from functools import partial

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import *

# Kernel invariants
mma_modes = (0, 1, 2)
mma_dice = (None, None, None)  # (MMA, #MMA_M, #MMA_K)
cpy_dice = (None,) + mma_dice  # (CPY, #CPY_MMA, #CPY_M, #CPY_K)
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128

# Math helpers
log2_e = math.log2(math.e)  # change exponential base
use_tensor_ssa_math = False  # experimental
fadd2 = cute.arch.add_packed_f32x2
fmul2 = cute.arch.mul_packed_f32x2
ffma2 = cute.arch.fma_packed_f32x2
exp2 = partial(cute.math.exp2, fastmath=True)
def _warp_fmax_shfl(value, mask=0xFFFFFFFF):
    """Warp-wide fmax via a butterfly shfl. ``warp_redux_sync(kind="fmax")``
    emits ``redux.f32``, which ptxas rejects on sm_110a (redux is int-only)."""
    for offset in (16, 8, 4, 2, 1):
        value = cute.arch.fmax(
            value, cute.arch.shuffle_sync_bfly(value, offset=offset, mask=mask), nan=True
        )
    return value


warp_fmax = _warp_fmax_shfl
smem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")
gmem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="gpu")


class MixedInputFusedMultiHeadAttentionDecode:
    def __init__(
        self,
        headdim,
        block_scaledim,  # headdim per scale factor; scale factor shape is (batches, heads_k, seqlen, headdim / block_scaledim)
        grouped_head_tile,  # GQA packing tile size, can be less than group size
        convert_warpgroups=1,  # Multiple warpgroups striding on convert stages
        page_size=0,  # 0 = dense KV; 128 = paged pool (num_pages, page_size, H_kv, D)
    ):
        self.headdim = headdim
        self.grouped_head_tile = grouped_head_tile
        self.block_scaledim = block_scaledim
        self.scaledim = headdim // block_scaledim
        self.convert_warpgroups = convert_warpgroups
        self.page_size = page_size
        assert page_size in (0, 128), "page_size must be 0 (dense) or 128 (paged)"

        assert headdim % block_scaledim == 0
        assert grouped_head_tile % 8 == 0 and 0 < grouped_head_tile <= 32

        warpgroup_id = 0

        self.softmax_warpgroup_id = warpgroup_id
        warpgroup_id += 1

        self.cvt_warpgroup_ids = tuple(
            range(warpgroup_id, warpgroup_id + convert_warpgroups)
        )
        warpgroup_id += convert_warpgroups

        # Why 2 MMA+TMA warps when not MMA bound?
        # Less register pressure per warp promotes concise SASS
        # hides MMA 'switching' latency that gets exposed with less concise SASS
        # We would have 2 leftover warps if we do warpgroup reg realloc
        # and less register pressure gives more realloc flexibility
        self.mma_kq_warp_id = warpgroup_id * warpgroup_warps + 0
        self.mma_vp_warp_id = warpgroup_id * warpgroup_warps + 1
        self.tma_kv_warp_id = warpgroup_id * warpgroup_warps + 2
        self.tma_qo_warp_id = warpgroup_id * warpgroup_warps + 3
        self.mma_tma_warpgroup_id = warpgroup_id
        warpgroup_id += 1

        self.threads_per_cta = warpgroup_id * warpgroup_threads

        self.use_reg_reconfig = grouped_head_tile > 16
        max_regs_per_wg_thread = 64 * 1024 // warpgroup_threads  # 64K regs per SM
        self.mma_tma_regs = 72
        self.cvt_regs = 112
        self.softmax_regs = (
            max_regs_per_wg_thread
            - self.mma_tma_regs
            - self.cvt_regs * convert_warpgroups
        )
        self.softmax_regs = max(128, min(256, self.softmax_regs))
        assert (
            self.mma_tma_regs + self.softmax_regs + self.cvt_regs * convert_warpgroups
        ) <= max_regs_per_wg_thread or not self.use_reg_reconfig

        self.bs_stages = 2
        self.sp_stages = 2
        self.o_stages = 1

    def can_implement(
        self,
        problem_shape,
        kv_splits,
        q_dtype,
        kv_dtype,
        o_dtype,
        acc_dtype,
    ):
        b, h_q, h_k, s_k, d = problem_shape

        if kv_dtype is cutlass.Float8E4M3:
            raise ValueError("use Float8E4M3FN instead of Float8E4M3")

        if d % 64 != 0:
            raise testing.CantImplementError(f"headdim({d}) must be multiple of 64")

        if h_q % h_k != 0:
            raise testing.CantImplementError(
                f"heads_q({h_q}) must be a multiple of heads_k({h_k})"
            )

        align_scale_bits = 128  # TMA requirement
        if self.scaledim * q_dtype.width < align_scale_bits:
            align_seq = align_scale_bits // (self.scaledim * q_dtype.width)
            if s_k % align_seq != 0:
                raise testing.CantImplementError(
                    f"seqlen({s_k}) must be a multiple of {align_seq}"
                )

        if kv_dtype.width < 8 and d % 128 != 0:  # TMA requirement
            raise testing.CantImplementError(
                f"headdim({d}) must be multiple of 128 for {kv_dtype} KV"
            )

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[
            cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32
        ],  # b, h_q, h_k, s_k, d
        kv_splits: cutlass.Int32,  # threadblocks per sequence
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        k_scale_iter: cute.Pointer,
        v_scale_iter: cute.Pointer,
        o_iter: cute.Pointer,
        m_iter: cute.Pointer,  # colmax_s, must be -inf initialized
        l_iter: cute.Pointer,  # logsumexp
        o_partial_iter: cute.Pointer,  # partial O per kv split
        m_partial_iter: cute.Pointer,  # partial colmax_s per kv split
        l_partial_iter: cute.Pointer,  # partial colsum_p per kv split
        scale_qs: cutlass.Float32,
        scale_o: cutlass.Float32,
        stream: cuda.CUstream,
        mPageTable_iter: Optional[cute.Pointer] = None,  # paged page table (batch, max_pages) int32
        seqlen_k: Optional[cutlass.Int32] = None,  # per-batch seqlen (paged)
        num_pages: Optional[cutlass.Int32] = None,  # total physical pages in the paged pool
    ):
        ##############################
        # TiledMma creation
        ##############################
        mma_dtype = q_iter.dtype
        acc_dtype = o_partial_iter.dtype
        assert acc_dtype is cutlass.Float32  # don't support other acc types for now

        # Block tile sets the granularity at which threadblocks consume work
        blk_tile_s = 128
        blk_tile_h = self.grouped_head_tile
        blk_tile_d = self.headdim
        blk_tile_shd = (blk_tile_s, blk_tile_h, blk_tile_d)

        # MMA tile sets the granularity at which TMAs + MMAs are issued
        mma_tile_m = 128
        mma_tile_n = self.grouped_head_tile
        mma_tile_k = 128 if self.headdim % 128 == 0 else 64
        mma_tile_mnk = (mma_tile_m, mma_tile_n, mma_tile_k)
        assert self.headdim % mma_tile_k == 0

        # GEMM1: (S_K, H_R, D, (H_K, B))
        tiled_mma_kq = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            OperandMajorMode.K,  # K
            OperandMajorMode.K,  # Q
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
            tcgen05.OperandSource.TMEM,  # converted K in tmem
        )

        # GEMM2: (D, H_R, S_K, (H_K, B))
        tiled_mma_vp = sm100_utils.make_trivial_tiled_mma(  #
            mma_dtype,
            mma_dtype,
            OperandMajorMode.K,  # V
            OperandMajorMode.MN,  # P
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
            tcgen05.OperandSource.TMEM,  # converted V in tmem
        )

        # Calculate Q stages
        self.q_stages = blk_tile_d // mma_tile_k

        # Perf heuristics
        cap_kv_stages = k_iter.dtype.width >= 8
        max_cvt_stages = 4 if self.grouped_head_tile == 32 and mma_tile_k == 128 else 8
        max_kv_stages = 8 if mma_tile_k == 128 else 14

        # Calculate KV tmem stages
        tmem_alloc_cols = mma_tile_n * self.sp_stages
        tmem_alloc_cols += mma_tile_n * self.o_stages * (blk_tile_d // mma_tile_m)
        tmem_capacity = 512
        cvt_stage_cols = mma_tile_k * mma_dtype.width // 32
        self.cvt_stages = (tmem_capacity - tmem_alloc_cols) // cvt_stage_cols
        self.cvt_stages = (
            min(self.cvt_stages, max_cvt_stages) if cap_kv_stages else self.cvt_stages
        )

        tmem_alloc_cols += self.cvt_stages * cvt_stage_cols
        self.tmem_alloc_cols = 2 ** math.ceil(
            math.log2(tmem_alloc_cols)
        )  # Tmem alloc must be PO2

        print(f"\tcvt stages: {self.cvt_stages}")

        # Calculate KV smem stages
        self.mbarrier_reserved_bytes = 768
        smem_alloc_bits = self.mbarrier_reserved_bytes * 8
        smem_alloc_bits += mma_tile_n * acc_dtype.width  # colmax
        smem_alloc_bits += (
            self.scaledim * blk_tile_s * self.bs_stages * mma_dtype.width
        )  # block scale
        smem_alloc_bits += mma_tile_n * warpgroup_warps * acc_dtype.width  # colsum
        smem_alloc_bits += (
            mma_tile_n * mma_tile_k * self.q_stages * mma_dtype.width
        )  # Q
        smem_alloc_bits += (
            mma_tile_m * mma_tile_n * self.sp_stages * mma_dtype.width
        )  # P

        # Thor (sm_110a) smem/SM is 232448 B, not the sm_100 (B200) value —
        # the reference hardcodes "sm_100" which over-allocates on Thor.
        smem_capacity = 232448
        kv_smem_dtype = cutlass.Int8 if k_iter.dtype.width < 8 else k_iter.dtype
        self.kv_stages = (smem_capacity * 8 - smem_alloc_bits) // (
            mma_tile_m * mma_tile_k * kv_smem_dtype.width
        )
        self.kv_stages = (
            min(self.kv_stages, max_kv_stages) if cap_kv_stages else self.kv_stages
        )

        print(f"\tkv stages: {self.kv_stages}")

        ##############################
        # TMA creation
        ##############################
        b, h_q, h_k, s_k, d = problem_shape
        h_r = h_q // h_k

        q = cute.make_tensor(
            q_iter,
            cute.make_ordered_layout(shape=(h_r, d, (h_k, b)), order=(1, 0, (2, 3))),
        )

        PS = self.page_size
        PAGED = mPageTable_iter is not None
        # Pool-form tensors are built with PYTHON TERNARIES, not staged
        # if/else: the host jit traces BOTH blocks of a staged if, and the
        # join back requires one type on every path (None vs Tensor fails).
        # A ternary is plain lazy Python — each leg builds exactly its own
        # tensors and nothing crosses a control-flow region. pool_pages is a
        # dummy (1) on a dense compile so the paged operand still traces.
        pool_pages = num_pages if num_pages is not None else 1
        # Paged pool: (num_pages, PS, H_kv, D), D contiguous (D-last).
        k = (
            cute.make_tensor(
                k_iter,
                cute.make_ordered_layout(
                    shape=(pool_pages, PS, h_k, d), order=(3, 2, 1, 0)
                ),
            )
            if PAGED
            else cute.make_tensor(
                k_iter,
                cute.make_ordered_layout(
                    shape=(s_k, d, (h_k, b)), order=(1, 0, (2, 3))
                ),
            )
        )
        v = (
            # D-MAJOR pool: (num_pages, PS, h_k, d) — d contiguous, exactly
            # like the K pool. The V smem stage is headdim-major (GEMM2
            # M-major: d contiguous, is_k_major=False), so the 16-B cp.async
            # run is along d on BOTH the gmem and smem sides (see the V
            # fill — structurally identical to the K leg).
            cute.make_tensor(
                v_iter,
                cute.make_ordered_layout(
                    shape=(pool_pages, PS, h_k, d), order=(3, 2, 1, 0)
                ),
            )
            if PAGED
            else cute.make_tensor(
                v_iter,
                cute.make_ordered_layout(
                    shape=(d, s_k, (h_k, b)), order=(0, 1, (2, 3))
                ),
            )
        )
        mPageTable = (
            cute.make_tensor(mPageTable_iter, cute.make_layout((b, pool_pages)))
            if mPageTable_iter is not None
            else None
        )
        assert k_iter.dtype is not q_iter.dtype
        assert v_iter.dtype is k_iter.dtype

        o_partial = cute.make_tensor(
            o_partial_iter,
            cute.make_ordered_layout(
                shape=(d, h_r, (h_k, b), kv_splits), order=(0, 1, (2, 3), 4)
            ),
        )

        m = cute.make_tensor(
            m_iter,
            cute.make_ordered_layout(
                shape=(h_r, (h_k, b)),
                order=(0, (1, 2)),
            ),
        )
        assert m_iter.dtype is acc_dtype

        m_partial = cute.make_tensor(
            m_partial_iter,
            cute.make_ordered_layout(
                shape=(h_r, (h_k, b), kv_splits),
                order=(0, (1, 2), 3),
            ),
        )
        assert m_partial_iter.dtype is acc_dtype

        l_partial = cute.make_tensor(
            l_partial_iter,
            cute.make_ordered_layout(
                shape=(h_r, (h_k, b), kv_splits),
                order=(0, (1, 2), 3),
            ),
        )
        assert l_partial_iter.dtype is acc_dtype

        align_scale_bits = 128  # TMA requirement
        if cutlass.const_expr(self.scaledim * mma_dtype.width >= align_scale_bits):
            scale_layout = cute.make_ordered_layout(
                shape=(self.scaledim, s_k, (h_k, b)), order=(0, 1, (2, 3))
            )
        else:
            align_seq = align_scale_bits // (self.scaledim * mma_dtype.width)
            s_ks = (align_seq, s_k // align_seq)
            scale_layout = cute.make_ordered_layout(
                shape=(self.scaledim, s_ks, (h_k, b)), order=(0, (1, 2), (3, 4))
            )

        # Paged scale pool: (num_pages, h_k, PS, scaledim), scaledim-last —
        # h_k-MAJOR (the KV pool's h_k and PS modes swapped): with
        # scaledim == 1 a PS x scaledim tile is only 256 B, and the KV-pool
        # order (PS before h_k) would leave the h_k >= 1 tiles 2 B-aligned
        # (cp.async needs >= 4 B). h_k-major makes every (page, h_k) tile
        # base a multiple of PS*scaledim*2 B (256 B here), so the paged
        # tma_qo cp.async scale-gather (TMA can't express the page-table
        # gather) is legal; the tile lands in the same sBS stage as the
        # dense TMA path. (Ternaries, not staged if/else — see KV pools.)
        # PS is 0 on a dense compile — a dummy tile size there (the pool
        # tensors are unused on the dense leg; the ternaries below select
        # the dense layouts).
        PS_pool = PS if PS > 0 else 128
        scale_pool_layout = cute.make_ordered_layout(
            shape=(pool_pages, h_k, PS_pool, self.scaledim), order=(3, 2, 1, 0)
        )
        k_scale = (
            cute.make_tensor(k_scale_iter, scale_pool_layout)
            if PAGED
            else cute.make_tensor(k_scale_iter, scale_layout)
        )
        v_scale = (
            cute.make_tensor(v_scale_iter, scale_pool_layout)
            if PAGED
            else cute.make_tensor(v_scale_iter, scale_layout)
        )
        assert k_scale_iter.dtype is mma_dtype
        assert v_scale_iter.dtype is mma_dtype

        # (MMA, MMA_M/N, MMA_K, Stages)
        smem_layout_q = sm100_utils.make_smem_layout_b(
            tiled_mma_kq, mma_tile_mnk, q_iter.dtype, self.q_stages
        )
        smem_layout_k = sm100_utils.make_smem_layout_a(
            tiled_mma_kq, mma_tile_mnk, kv_smem_dtype, self.kv_stages
        )
        smem_layout_v = sm100_utils.make_smem_layout_a(
            tiled_mma_vp, mma_tile_mnk, kv_smem_dtype, self.kv_stages, is_k_major=False
        )  # V is always headdim-major (GEMM2 M-major) in gmem+smem

        smem_layout_bs = cute.make_layout((self.scaledim, blk_tile_s, self.bs_stages))

        smem_layout_atom_o = tcgen05.make_smem_layout_atom(
            tcgen05.mma.SmemLayoutAtomKind.MN_SW128, o_partial_iter.dtype
        )
        smem_layout_o = cute.tile_to_shape(
            smem_layout_atom_o, (blk_tile_d, blk_tile_h), order=(1, 0)
        )
        smem_layout_o = cute.flat_divide(smem_layout_o, (mma_tile_m, mma_tile_n))

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            q,
            cute.select(smem_layout_q, mma_modes),
            mma_tile_mnk,
            tiled_mma_kq,
        )
        # TMA atoms for KV + scale are created for BOTH legs. The paged leg
        # does not *use* the KV/scale TMA atoms (it loads KV + scale via a
        # cp.async page-table gather in the tma_kv / tma_qo PAGED branches),
        # but the decode kernel takes them unconditionally, so they must exist.
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
        tma_atom_ks, tma_tensor_ks = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            k_scale,
            cute.select(smem_layout_bs, mode=[0, 1]),
            smem_layout_bs.shape[:2],
        )
        tma_atom_vs, tma_tensor_vs = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            v_scale,
            cute.select(smem_layout_bs, mode=[0, 1]),
            smem_layout_bs.shape[:2],
        )
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            o_partial,
            cute.select(smem_layout_o, mode=[0, 1]),
            mma_tile_mnk[:2],
        )

        # K scale and V scale will have the same TMA tensor (coord tensor)
        # only difference is base ptr which is stored in copy atom
        tma_tensor_bs = tma_tensor_ks

        ##############################
        # Decode Kernel launch
        ##############################
        scale_qs_log2_e = scale_qs * log2_e

        n_tiles = cute.ceil_div(h_r, blk_tile_h)
        l_tiles = b * h_k
        grid = (kv_splits, n_tiles, l_tiles)

        self.decode(
            blk_tile_shd,
            mma_tile_mnk,
            tiled_mma_kq,
            tiled_mma_vp,
            q_iter.dtype,
            smem_layout_q,
            tma_atom_q,
            tma_tensor_q,
            k_iter.dtype,
            smem_layout_k,
            tma_atom_k,
            tma_tensor_k,
            v_iter.dtype,
            smem_layout_v,
            tma_atom_v,
            tma_tensor_v,
            smem_layout_bs,
            tma_atom_ks,
            tma_atom_vs,
            tma_tensor_bs,
            tma_tensor_vs,
            o_partial_iter.dtype,
            smem_layout_o,
            tma_atom_o,
            tma_tensor_o,
            m,
            m_partial,
            l_partial,
            scale_qs,
            scale_qs_log2_e,
            mPageTable,
            seqlen_k,
            num_pages,
            k,
            v,
            k_scale,
            v_scale,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

        ##############################
        # Reduction Kernel launch
        ##############################
        o = cute.make_tensor(o_iter, cute.make_layout((d, h_q, b)))
        m = cute.make_tensor(m_iter, cute.make_layout((h_q, b)))
        l = cute.make_tensor(l_iter, cute.make_layout((h_q, b)))

        o_partial = cute.make_tensor(
            o_partial_iter, cute.make_layout((d, h_q, b, kv_splits))
        )
        m_partial = cute.make_tensor(
            m_partial_iter, cute.make_layout((h_q, b, kv_splits))
        )
        l_partial = cute.make_tensor(
            l_partial_iter, cute.make_layout((h_q, b, kv_splits))
        )

        d_per_blk = 128
        d_blks = cute.ceil_div(d, d_per_blk)

        self.reduction(o, m, l, o_partial, m_partial, l_partial, scale_o).launch(
            grid=[d_blks, h_q, b],
            block=[d_per_blk, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def decode(
        self,
        # MMA
        blk_tile_shd: cute.Tile,
        mma_tile_mnk: cute.Tile,
        tiled_mma_kq: cute.TiledMma,
        tiled_mma_vp: cute.TiledMma,
        # Q
        q_dtype: Type[cutlass.Numeric],
        smem_layout_q: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        # K
        k_dtype: Type[cutlass.Numeric],
        smem_layout_k: cute.ComposedLayout,
        tma_atom_k: cute.CopyAtom,
        mK: cute.Tensor,
        # V
        v_dtype: Type[cutlass.Numeric],
        smem_layout_v: cute.ComposedLayout,
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        # K/V block scale
        smem_layout_bs: cute.Layout,
        tma_atom_ks: cute.CopyAtom,
        tma_atom_vs: cute.CopyAtom,
        mBS: cute.Tensor,
        mVBS: cute.Tensor,  # V-scale coord tensor (the PAGED gather needs its own pool)
        # O
        o_dtype: Type[cutlass.Numeric],
        smem_layout_o: cute.ComposedLayout,
        tma_atom_o: cute.CopyAtom,
        mO: cute.Tensor,
        # Rest
        mM: cute.Tensor,
        mM_partial: cute.Tensor,
        mL_partial: cute.Tensor,
        scale_qs: cutlass.Float32,
        scale_qs_log2_e: cutlass.Float32,
        # Paged KV (None for dense)
        mPageTable: Optional[cute.Tensor] = None,
        seqlen_k: Optional[cutlass.Int32] = None,
        num_pages: Optional[cutlass.Int32] = None,
        # Raw (non-TMA) pool tensors — the paged cp.async gathers index
        # .iterator + flat offset. The TMA coord tensors' iterators are
        # descriptor objects, not raw pointers. (Dense compiles pass the
        # dense raw tensors; the paged branches are folded away.)
        mK_raw: Optional[cute.Tensor] = None,
        mV_raw: Optional[cute.Tensor] = None,
        mBS_raw: Optional[cute.Tensor] = None,
        mVBS_raw: Optional[cute.Tensor] = None,
    ):
        # Read special registers
        kv_splits, tiles_hr, tiles_hb = cute.arch.grid_dim()
        kv_split_idx, coord_hr, coord_hb = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
        warpgroup_tidx = tidx % warpgroup_threads
        warpgroup_widx = warp_idx % warpgroup_warps
        init_warp = 0

        # No multicast
        mcast_coord = 0
        mcast_layout = cute.make_layout((1, 1, 1, 1))  # vmnk

        # Alias types
        mma_dtype = q_dtype
        acc_dtype = o_dtype
        kv_smem_dtype = cutlass.Int8 if k_dtype.width < 8 else k_dtype

        # Paged KV: mPageTable is None for dense.
        PAGED = cutlass.const_expr(mPageTable is not None)

        # Shapes for MMA tile indexing (Read TMA partition for example)
        blk_tile_s, blk_tile_h, blk_tile_d = blk_tile_shd
        mma_tile_m, mma_tile_n, mma_tile_k = mma_tile_mnk
        tiles_dm, tiles_sk = cute.ceil_div(
            (blk_tile_d, blk_tile_s), (mma_tile_m, mma_tile_k)
        )
        tiles_dk, tiles_sm = cute.ceil_div(
            (blk_tile_d, blk_tile_s), (mma_tile_k, mma_tile_m)
        )
        # Paged: one KV tile == one page (PS==blk_tile_s); the batch's tile
        # count is its own seqlen / page size, not the pool's page count.
        # Python ternary, not staged if/else: bindings made inside a staged
        # region are region-local (the DSL scope rule forbids reading them
        # after the region), and a ternary is plain lazy Python — only the
        # selected leg traces.
        tiles_s = (
            cute.ceil_div(seqlen_k, blk_tile_s)
            if PAGED
            else cute.ceil_div(mK.shape[0], blk_tile_s)
        )
        iters_s = cute.ceil_div(tiles_s - kv_split_idx, kv_splits)
        prefetch_iters = self.sp_stages - 1
        if iters_s < prefetch_iters:
            prefetch_iters = iters_s
        assert tiles_sm == 1

        # Runtime checks
        exit_early = kv_split_idx >= tiles_s
        lane_store_max = mma_tile_n == warp_threads or lane_idx < mma_tile_n

        # Smem alloc helper
        svector_align = 16
        stensor_align = 128
        smem = utils.SmemAllocator()

        ##############################
        # Prefetch TMA descriptor
        ##############################
        if warp_idx == init_warp and not exit_early:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_ks)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_vs)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)
        init_warp += 1

        ##############################
        # Tmem Allocation
        ##############################
        tmem_ptr_smem_ptr = smem.allocate_array(cutlass.Int32)
        if warp_idx == init_warp and not exit_early:
            cute.arch.alloc_tmem(self.tmem_alloc_cols, tmem_ptr_smem_ptr)
        init_warp += 1

        ##############################
        # Pipeline Allocation + Init
        ##############################
        # Allocate Mbarriers
        q_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.q_stages * 2)
        kv_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.kv_stages * 2)
        bs_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.bs_stages * 2)
        cvt_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.cvt_stages * 2)
        s_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.sp_stages * 2)
        p_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.sp_stages * 2)
        o_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.o_stages * 2)

        # Declare named barriers
        softmax_nbar = pipeline.NamedBarrier(
            barrier_id=1, num_threads=warpgroup_threads
        )
        mma_kq_nbar = pipeline.NamedBarrier(barrier_id=2, num_threads=64)
        mma_vp_nbar = pipeline.NamedBarrier(barrier_id=3, num_threads=64)

        # Alias thread cooperatives
        elect_one_cooperative = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        warpgroup_cooperative = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, warpgroup_threads
        )
        mma_group = elect_one_cooperative
        tma_group = elect_one_cooperative
        # Paged KV: the cp.async page-table gather is issued by every thread of
        # the tma_kv warp, and the cp.async.mbarrier.arrive.noinc is per-thread,
        # so the producer group is the full warp (not the elect_one TMA issuer).
        cpasync_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, warp_threads)
        cvt_group = warpgroup_cooperative
        cvt_groups = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, warpgroup_threads * self.convert_warpgroups
        )
        softmax_group = warpgroup_cooperative

        # Initialize pipelines
        q_producer, q_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.q_stages,
            producer_group=tma_group,
            consumer_group=softmax_group,  # Reuse Q consumer mbarriers to sync O store
            tx_count=cute.size_in_bytes(q_dtype, cute.select(smem_layout_q, mma_modes)),
            barrier_storage=q_pipeline_ptr,
            tidx=mcast_coord,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()
        # Paged KV: the producer is a cp.async page-table gather (no TMA
        # descriptor / tx_count); the full barrier is signaled via
        # cp.async.mbarrier.arrive.noinc (PipelineCpAsync producer op).
        # Ternary (not staged if/else) — see tiles_s above.
        kv_producer, kv_consumer = (
            pipeline.PipelineCpAsync.create(
                num_stages=self.kv_stages,
                producer_group=cpasync_group,
                consumer_group=cvt_group,
                barrier_storage=kv_pipeline_ptr,
                defer_sync=True,
            ).make_participants()
            if PAGED
            else pipeline.PipelineTmaAsync.create(
                num_stages=self.kv_stages,
                producer_group=tma_group,
                consumer_group=cvt_group,
                tx_count=cute.size_in_bytes(
                    k_dtype, cute.select(smem_layout_k, mma_modes)
                ),
                barrier_storage=kv_pipeline_ptr,
                tidx=mcast_coord,
                cta_layout_vmnk=mcast_layout,
                defer_sync=True,
            ).make_participants()
        )
        # Paged scale: the producer is a cp.async page-table gather (no TMA
        # descriptor / tx_count); the full barrier is signaled via
        # cp.async.mbarrier.arrive.noinc (PipelineCpAsync producer op).
        # Ternary (not staged if/else) — see tiles_s above.
        bs_producer, bs_consumer = (
            pipeline.PipelineCpAsync.create(
                num_stages=self.bs_stages,
                producer_group=cpasync_group,
                consumer_group=cvt_groups,
                barrier_storage=bs_pipeline_ptr,
                defer_sync=True,
            ).make_participants()
            if PAGED
            else pipeline.PipelineTmaAsync.create(
                num_stages=self.bs_stages,
                producer_group=tma_group,
                consumer_group=cvt_groups,
                tx_count=cute.size_in_bytes(
                    mma_dtype, cute.select(smem_layout_bs, mode=[0, 1])
                ),
                barrier_storage=bs_pipeline_ptr,
                tidx=mcast_coord,
                cta_layout_vmnk=mcast_layout,
                defer_sync=True,
            ).make_participants()
        )
        cvt_producer, cvt_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.cvt_stages,
            producer_group=cvt_group,
            consumer_group=mma_group,
            barrier_storage=cvt_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        s_producer, s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.sp_stages,
            producer_group=mma_group,
            consumer_group=softmax_group,
            barrier_storage=s_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.sp_stages,
            producer_group=softmax_group,
            consumer_group=mma_group,
            barrier_storage=p_pipeline_ptr,
            defer_sync=True,
        ).make_participants()
        o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.o_stages,
            producer_group=mma_group,
            consumer_group=softmax_group,
            barrier_storage=o_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        # Ensure visibility of local mbarrier inits and tmem alloc
        cute.arch.sync_threads()

        ##############################
        # MMA Partition + Allocate
        ##############################
        # Threadblock slice
        thrblk_mma_kq = tiled_mma_kq.get_slice(0)
        thrblk_mma_vp = tiled_mma_vp.get_slice(0)

        # M - colmax
        sM_layout = cute.make_layout(shape=(mma_tile_m, mma_tile_n), stride=(0, 1))
        sM = smem.allocate_tensor(acc_dtype, sM_layout, svector_align)
        tCsM = thrblk_mma_kq.partition_C(sM)

        # L - colsum
        sL_layout = cute.make_layout(
            shape=(mma_tile_m, mma_tile_n, warpgroup_warps), stride=(0, 1, mma_tile_n)
        )
        sL = smem.allocate_tensor(acc_dtype, sL_layout, svector_align)
        tCsL = thrblk_mma_kq.partition_C(sL)

        # BS - block scale
        sBS = smem.allocate_tensor(
            mma_dtype, smem_layout_bs, stensor_align
        )  # (SCALE, TILE_S, bs_stages)

        # Q
        tBsQ = smem.allocate_tensor(
            q_dtype, smem_layout_q.outer, stensor_align, smem_layout_q.inner
        )  # (MMA, #MMA_N, #MMA_K, q_stages)

        # K
        tAsK = smem.allocate_tensor(
            kv_smem_dtype, smem_layout_k.outer, stensor_align, smem_layout_k.inner
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        tAtK_cvt_shape = tiled_mma_kq.partition_shape_A(
            (mma_tile_m, mma_tile_k, self.cvt_stages)
        )  # (MMA, #MMA_M, #MMA_K, cvt_stages)
        tAtK_cvt = thrblk_mma_kq.make_fragment_A(tAtK_cvt_shape)

        # V
        tAsV_iterator = cute.recast_ptr(
            tAsK.iterator, smem_layout_v.inner, dtype=kv_smem_dtype
        )  # KV share input buffers
        tAsV = cute.make_tensor(
            tAsV_iterator, smem_layout_v.outer
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        tAtV_cvt_shape = tiled_mma_vp.partition_shape_A(
            (mma_tile_m, mma_tile_k, self.cvt_stages)
        )  # (MMA, #MMA_M, #MMA_K, cvt_stages)
        tAtV_cvt = thrblk_mma_vp.make_fragment_A(tAtV_cvt_shape)

        # S
        tCtS_shape = tiled_mma_kq.partition_shape_C(
            (mma_tile_m, mma_tile_n, self.sp_stages)
        )
        tCtS = thrblk_mma_kq.make_fragment_C(
            tCtS_shape
        )  # (MMA_MN, #MMA_M=1, #MMA_N=1, sp_stages)

        # P - Treat MN C tile of BMM0 as NM B tile of BMM1
        # (MMA_NK, #MMA_N, #MMA_K=MMA_TILE_M/MMA_K, sp_stages)
        mma_tile_nm = (None, mma_tile_n, mma_tile_m)
        tBsP_nm_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_vp, mma_tile_nm, mma_dtype, self.sp_stages
        )
        tBsP_nm = smem.allocate_tensor(
            mma_dtype, tBsP_nm_layout.outer, stensor_align, tBsP_nm_layout.inner
        )

        # Tile for NK B tile iteration
        # (MMA_NK, #MMA_N, #MMA_K=MMA_TILE_K/MMA_K, #TILES_SK=MMA_TILE_M/MMA_TILE_K, sp_stages)
        tBsP_nk_tile = thrblk_mma_vp.partition_shape_B((mma_tile_n, mma_tile_k))
        tBsP_nk = cute.local_tile(tBsP_nm, tBsP_nk_tile, (0, 0, None, None))

        # Reshape NM B tile of BMM1 to become MN C tile of BMM0
        # (MMA_NK, #MMA_N, #MMA_K=MMA_TILE_M/MMA_K, sp_stages) ->
        # (MMA_MN, #MMA_M, #MMA_N, sp_stages)
        tCsP_tile = cute.make_ordered_layout(tCtS_shape, order=((2, 0), 3, 1, 4))
        tCsP = cute.composition(tBsP_nm, tCsP_tile)

        # O
        sO_iterator = cute.recast_ptr(
            tBsQ.iterator, smem_layout_o.inner, dtype=o_dtype
        )  # Reuse QKV smem for O TMA store
        sO_mma = cute.make_tensor(
            sO_iterator, smem_layout_o.outer
        )  # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN)
        tCsO = thrblk_mma_vp.partition_C(
            sO_mma
        )  # (MMA, #MMA_M, #MMA_N, #TILE_DM, #TILE_HN)
        tCtO = thrblk_mma_vp.make_fragment_C(tCsO.shape)

        # Tmem tensor allocation
        tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Int32, 16, tmem_ptr_smem_ptr)
        tmem_offset = 0

        tAtK_cvt = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=mma_dtype), tAtK_cvt.layout
        )
        tAtV_cvt = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=mma_dtype), tAtV_cvt.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tAtK_cvt)

        tCtS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtS.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS)

        tCtO = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtO.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtO)

        print(
            f"\t{tmem_offset} tmem cols used, {self.tmem_alloc_cols} tmem cols allocated"
        )
        assert tmem_offset <= self.tmem_alloc_cols

        ##############################
        # Exit early
        ##############################
        if exit_early:
            noop = None  # early return not supported # noqa: F841

        ##############################
        # TMA KV Dispatch
        ##############################
        elif warp_idx == self.tma_kv_warp_id:
            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            # const_expr CALL form (not a bare `if PAGED:`): a Name condition
            # is a staged runtime if — the DSL traces BOTH arms (scf.if
            # regions), so the dense arm's local_tile/TMA-partition would
            # run on the pool tensors during a paged compile. The const_expr
            # call form is a true compile-time branch: only the selected
            # arm traces on each leg.
            if cutlass.const_expr(PAGED):
                #
                # Paged KV: per-stage cp.async page-table gather into the
                # swizzled KV smem stage tiles. PS == blk_tile_s, so one KV
                # tile == one physical pool page; mPageTable[b, s] maps tile s
                # to the physical page. The acquisition pattern mirrors the
                # dense TMA leg exactly: tiles_dk K stages + tiles_dm V stages
                # per KV tile (each stage is a (blk_tile_s, d-half) fragment —
                # NOT the whole tile), same handle order, same commit points.
                #
                # Stage 2D: both K and V stage fragments are (s, d) row-major
                # in smem (d contiguous: K A-linear = s*TILE_S + d, V the
                # same), so one ordered 2D relabels the mma-mode stage view
                # back to (s, d); the swizzle rides the pointer (the
                # ComposedLayout inner is untouched by the composition).
                # partition_D drops the pointer alignment, so the final
                # per-(row-group, d-chunk) fragment is rewrapped with the
                # 16-B alignment the 128-bit cp.async atom requires (the
                # swizzle permutes 16-B chunks within 128-B rows, so 16 B is
                # guaranteed).
                #
                kv_async_copy_elems = 128 // kv_smem_dtype.width  # fp8 -> 16
                kv_bytes = kv_smem_dtype.width // 8
                kv_gmem_k_block = math.gcd(blk_tile_d, blk_tile_d, 128 // kv_bytes)
                kv_threads_per_row = kv_gmem_k_block // kv_async_copy_elems
                kv_rows_per_pass = warp_threads // kv_threads_per_row
                kv_atom = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(
                        cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
                    ),
                    kv_smem_dtype,
                    num_bits_per_copy=128,
                )
                kv_thr_layout = cute.make_ordered_layout(
                    (kv_rows_per_pass, kv_threads_per_row), order=(1, 0)
                )
                kv_val_layout = cute.make_layout((1, kv_async_copy_elems))
                kv_gmem_tiled_copy = cute.make_tiled_copy_tv(
                    kv_atom, kv_thr_layout, kv_val_layout
                )
                # lane_idx (the warp-local index), not tidx (the CTA-wide
                # thread ID): the TiledCopy's 32-thread geometry is the
                # warp's, so the slice coordinate is the lane.
                kv_gmem_thr_copy = kv_gmem_tiled_copy.get_slice(lane_idx)
                # The V leg is a TRANSPOSE of the K leg: the K smem stage is
                # (M=s, K=d) but the V smem stage is (M=d, K=s) (the GEMM2
                # A-operand, M-major, is_k_major=False). Both are headdim-
                # major (d contiguous), so the 16-B cp.async run is along d on
                # both the gmem and smem sides for BOTH legs — but the V tile
                # is oriented (d, s) (d first) vs the K tile's (s, d). That
                # flips which value-mode carries the 16-elem run (mode 0 for V,
                # mode 1 for K) and the tiler / stage-2D ordering, so the V
                # leg gets its own TiledCopy + stage_2d + (d, s) tiler + a
                # (d, s) page view (see the V fill).
                kv_val_layout_v = cute.make_layout((kv_async_copy_elems, 1))
                kv_gmem_tiled_copy_v = cute.make_tiled_copy_tv(
                    kv_atom, kv_thr_layout, kv_val_layout_v
                )
                kv_gmem_thr_copy_v = kv_gmem_tiled_copy_v.get_slice(lane_idx)
                k_stage_2d = cute.make_ordered_layout(
                    (blk_tile_s, mma_tile_k), order=(0, 1)
                )
                v_stage_2d = cute.make_ordered_layout(
                    (mma_tile_m, mma_tile_k), order=(0, 1)
                )
                # Pool strides (num_pages, PS, h_k, d), D-last: page stride
                # PS*h_k*d, row stride h_k*d, h_k stride d.
                h_k = mK_raw.shape[2]
                page_stride = blk_tile_s * h_k * blk_tile_d
                row_stride = h_k * blk_tile_d
                b_index = coord_hb // h_k
                h_k_index = coord_hb % h_k

                if os.environ.get("SPARKCVT_PROBE") == "1":
                    import sys as _sys

                    print(
                        "PROBE tAsK[stage0]:",
                        tAsK[None, None, None, 0].layout,
                        file=_sys.stderr,
                    )
                    print(
                        "PROBE tAsV[stage0]:",
                        tAsV[None, None, None, 0].layout,
                        file=_sys.stderr
                    )
                    print("PROBE k_stage_2d:", k_stage_2d, file=_sys.stderr)
                    print("PROBE v_stage_2d:", v_stage_2d, file=_sys.stderr)
                    print(
                        "PROBE compK:",
                        cute.composition(
                            tAsK[None, None, None, 0], k_stage_2d
                        ).layout,
                        file=_sys.stderr,
                    )
                    print(
                        "PROBE compV:",
                        cute.composition(
                            tAsV[None, None, None, 0], v_stage_2d
                        ).layout,
                        file=_sys.stderr,
                    )

                #
                # Sequence loop
                #
                prefetch_tiles = prefetch_iters * kv_splits
                for s in cutlass.range(
                    kv_split_idx, prefetch_tiles + tiles_s, kv_splits
                ):
                    # Load K (tile s): tiles_dk d-half stages. The gmem source
                    # is the page's (PS, d) view tiled to the (s, d-half)
                    # stage shape, partitioned by the SAME TiledCopy as the
                    # dst (tc_spark idiom — the per-THREAD slice goes to
                    # cute.copy, not the tiled copy; the dynamic phys_page
                    # rides the page view's iterator and lowers through the
                    # partitioned cp.async view).
                    if s < tiles_s:
                        phys_page = mPageTable[b_index, s]
                        mK_page = mK_raw[phys_page, None, h_k_index, None]
                        for dk in cutlass.range_constexpr(tiles_dk):
                            k_handle = kv_producer.acquire_and_advance()
                            stage = k_handle.index
                            tKsK = kv_gmem_thr_copy.partition_D(
                                cute.composition(
                                    tAsK[None, None, None, stage], k_stage_2d
                                )
                            )
                            gK_h = cute.local_tile(
                                mK_page, (blk_tile_s, mma_tile_k), (0, dk)
                            )
                            tKgK = kv_gmem_thr_copy.partition_S(gK_h)
                            for m in cutlass.range_constexpr(
                                cute.size(tKsK, mode=[1])
                            ):
                                # No row predicate: the driver asserts
                                # seqlen_k % PS == 0 and the page table is
                                # identity, so every page of every real
                                # sequence is fully in-bounds (each page is
                                # exactly PS == blk_tile_s rows). The dense
                                # TMA leg is unpredicated for the same reason.
                                # (Partial-page support needs the tc_spark
                                # predicate_k idiom when it lands here.)
                                for k in cutlass.range_constexpr(
                                    cute.size(tKsK, mode=[2])
                                ):
                                    tKsK_k = tKsK[None, m, k]
                                    tKgK_k = tKgK[None, m, k]
                                    # partition_S drops the pointer alignment
                                    # (1-elem default) — re-stamp the 16-B
                                    # alignment the 128-bit cp.async atom
                                    # requires. Sound: every offset in the gmem
                                    # base is a multiple of 16 B (131 072-B
                                    # page, 1 024-B row, 128-B d-half, 16-B
                                    # thread band) on top of the 16-B-aligned
                                    # pool base.
                                    tKgK_k = cute.make_tensor(
                                        cute.make_ptr(
                                            kv_smem_dtype,
                                            tKgK_k.iterator.toint(),
                                            cute.AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        tKgK_k.layout,
                                    )
                                    cute.copy(
                                        kv_gmem_thr_copy,
                                        tKgK_k,
                                        tKsK_k,
                                    )
                            k_handle.commit()

                    # Load V (tile s - prefetch_tiles): tiles_dm d-half
                    # stages. The V leg is a transpose of the K leg: the V
                    # smem stage is (M=d, K=s) (GEMM2 A-operand, M-major,
                    # is_k_major=False) vs the K stage's (M=s, K=d). Both are
                    # headdim-major (d contiguous), and the d-major V pool has
                    # d contiguous too, so the 16-B cp.async run is along d on
                    # BOTH sides. The V tile is oriented (d, s) (d first): the
                    # (d, s) page view + (d, s) tiler + V TiledCopy (16 along
                    # the d mode) keep the run along d contiguously.
                    if s >= prefetch_tiles:
                        phys_page = mPageTable[b_index, s - prefetch_tiles]
                        mV_page = mV_raw[phys_page, None, h_k_index, None]
                        # (d, s) view of the (s, d) page: d (the V smem's
                        # M-axis) is contiguous in the d-major pool.
                        mV_page_ds = cute.make_tensor(
                            mV_page.iterator,
                            cute.select(mV_page.layout, mode=(1, 0)),
                        )
                        for dm in cutlass.range_constexpr(tiles_dm):
                            v_handle = kv_producer.acquire_and_advance()
                            stage = v_handle.index
                            tVsV = kv_gmem_thr_copy_v.partition_D(
                                cute.composition(
                                    tAsV[None, None, None, stage], v_stage_2d
                                )
                            )
                            gV_h = cute.local_tile(
                                mV_page_ds, (mma_tile_m, mma_tile_k), (dm, 0)
                            )
                            tVgV = kv_gmem_thr_copy_v.partition_S(gV_h)
                            for m in cutlass.range_constexpr(
                                cute.size(tVsV, mode=[1])
                            ):
                                # No row predicate (see the K leg above).
                                for k in cutlass.range_constexpr(
                                    cute.size(tVsV, mode=[2])
                                ):
                                    tVsV_k = tVsV[None, m, k]
                                    tVgV_k = tVgV[None, m, k]
                                    # partition_S alignment re-stamp (see the
                                    # K leg above).
                                    tVgV_k = cute.make_tensor(
                                        cute.make_ptr(
                                            kv_smem_dtype,
                                            tVgV_k.iterator.toint(),
                                            cute.AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        tVgV_k.layout,
                                    )
                                    cute.copy(
                                        kv_gmem_thr_copy_v,
                                        tVgV_k,
                                        tVsV_k,
                                    )
                            v_handle.commit()
            else:
                # Apply block tiler and slice
                gK = cute.local_tile(
                    mK, tiler=(blk_tile_s, blk_tile_d), coord=(None, 0, coord_hb)
                )  # (TILE_S, TILE_D, #TILE_S)
                gV = cute.local_tile(
                    mV, tiler=(blk_tile_d, blk_tile_s), coord=(0, None, coord_hb)
                )  # (TILE_D, TILE_S, #TILE_S)

                # Apply MMA tiler and MMA partition
                gK_mma = cute.flat_divide(
                    gK, (mma_tile_m, mma_tile_k)
                )  # (MMA_TILE_M, MMA_TILE_K, #TILE_SM, #TILE_DK, #TILE_S)
                gV_mma = cute.flat_divide(
                    gV, (mma_tile_m, mma_tile_k)
                )  # (MMA_TILE_M, MMA_TILE_K, #TILE_DM, #TILE_SK, #TILE_S)
                tAgK = thrblk_mma_kq.partition_A(
                    gK_mma
                )  # (MMA, #MMA_M, #MMA_K, #TILE_SM, #TILE_DK, #TILE_S)
                tAgV = thrblk_mma_vp.partition_A(
                    gV_mma
                )  # (MMA, #MMA_M, #MMA_K, #TILE_DM, #TILE_SK, #TILE_S)

                # TMA partition
                # (MMA, #MMA_M, #MMA_K, Rest...) -> (TMA, Rest...)
                tGSsK, tGSgK = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_k,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(tAsK, 0, 3),
                    gmem_tensor=cute.group_modes(tAgK, 0, 3),
                )

                tGSsV, tGSgV = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_v,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(tAsV, 0, 3),
                    gmem_tensor=cute.group_modes(tAgV, 0, 3),
                )

                #
                # Sequence loop
                #
                prefetch_tiles = prefetch_iters * kv_splits
                for s in cutlass.range(
                    kv_split_idx, prefetch_tiles + tiles_s, kv_splits
                ):
                    # Load K
                    if s < tiles_s:
                        tGSgK_s = tGSgK[None, None, None, s]
                        for dk in cutlass.range_constexpr(tiles_dk):
                            k_handle = kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_k,
                                tGSgK_s[None, 0, dk],
                                tGSsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )

                    # Load V
                    if s >= prefetch_tiles:
                        tGSgV_s = tGSgV[None, None, None, s - prefetch_tiles]
                        for sk in cutlass.range_constexpr(tiles_sk):
                            for dm in cutlass.range_constexpr(tiles_dm):
                                v_handle = kv_producer.acquire_and_advance()
                                cute.copy(
                                    tma_atom_v,
                                    tGSgV_s[None, dm, sk],
                                    tGSsV[None, v_handle.index],
                                    tma_bar_ptr=v_handle.barrier,
                                )

        ##############################
        # TMA QO Dispatch
        ##############################
        elif warp_idx == self.tma_qo_warp_id:
            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            # Apply block tiler and slice
            gQ = cute.local_tile(
                mQ, tiler=(blk_tile_h, blk_tile_d), coord=(coord_hr, 0, coord_hb)
            )  # (TILE_H, TILE_D)
            gO = cute.local_tile(
                mO,
                tiler=(blk_tile_d, blk_tile_h),
                coord=(0, coord_hr, coord_hb, kv_split_idx),
            )  # (TILE_D, TILE_H)
            # Apply MMA tiler and MMA partition
            gQ_mma = cute.flat_divide(
                gQ, (mma_tile_n, mma_tile_k)
            )  # (MMA_TILE_N, MMA_TILE_K, #TILE_HN, #TILE_DK)
            gO_mma = cute.flat_divide(
                gO, (mma_tile_m, mma_tile_n)
            )  # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN)
            tBgQ = thrblk_mma_kq.partition_B(
                gQ_mma
            )  # (MMA, #MMA_N, #MMA_K, #TILE_HN, #TILE_DK)

            # TMA partition
            tGSsQ, tGSgQ = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tBsQ, 0, 3),
                gmem_tensor=cute.group_modes(tBgQ, 0, 3),
            )

            tSGsO, tSGgO = cute.nvgpu.cpasync.tma_partition(
                tma_atom_o,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(sO_mma, 0, 2),
                gmem_tensor=cute.group_modes(gO_mma, 0, 2),
            )

            # Load Q
            for dk in cutlass.range_constexpr(tiles_dk):
                q_handle = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tGSgQ[None, 0, dk],  # stages_q == tiles_dk by construction
                    tGSsQ[None, dk],
                    tma_bar_ptr=q_handle.barrier,
                )

            # Sequence Loop
            prefetch_tiles = prefetch_iters * kv_splits
            # const_expr CALL form — see the tma_kv warp above (a bare Name
            # condition traces both arms on every compile).
            if cutlass.const_expr(PAGED):
                #
                # Paged scale: per-tile cp.async page-table gather. PS ==
                # blk_tile_s, so one scale tile == one pool page; the pool is
                # (num_pages, h_k, PS, scaledim) h_k-major (scaledim-last),
                # so within a (page, h_k) the PS x scaledim tile is a single
                # contiguous gmem block AND its base is
                # PS*scaledim*2-B-aligned (cp.async legal — see
                # scale_pool_layout). The per-row gather collapses to a 1D
                # copy.
                # Mirrors the KV gather in tma_kv: every thread of the warp
                # issues its own cp.async, then handle.commit()
                # (cp.async.mbarrier.arrive.noinc, the PipelineCpAsync
                # producer op). The smem stage tile is a contiguous block in
                # the affine sBS layout (col-major), so a 1D sBS view +
                # partition_D is exact (no swizzle to ride along). K and V
                # scale tiles alternate through the same sBS buffer (the cvt
                # warp's sVS reinterprets sBS) — same as the dense leg.
                #
                bs_tile_elems = blk_tile_s * self.scaledim
                # 128-bit cp.async only (this cutlass's simt_async_copy
                # cp_size supports 128 b only — a 64-b atom fails
                # verification): 8 bf16 elems per thread. The TiledCopy is
                # bs_active_threads wide (exactly one stage tile: 16 x 8 =
                # 128 elems); the warp's lanes >= bs_active_threads alias
                # onto it (lane % 16) — a benign duplicate write of the same
                # 16-B chunk — so the row pred stays a single condition.
                bs_async_copy_elems = 128 // mma_dtype.width  # 8 (bf16)
                assert bs_tile_elems % bs_async_copy_elems == 0
                bs_copy_bytes = bs_async_copy_elems * mma_dtype.width // 8  # 16
                bs_active_threads = bs_tile_elems // bs_async_copy_elems  # 16
                bs_g2s_atom = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(
                        cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
                    ),
                    mma_dtype,
                    num_bits_per_copy=128,
                )
                bs_thr_layout = cute.make_layout((bs_active_threads,))
                bs_val_layout = cute.make_layout((bs_async_copy_elems,))
                bs_gmem_tiled_copy = cute.make_tiled_copy_tv(
                    bs_g2s_atom, bs_thr_layout, bs_val_layout
                )
                # Lanes 16..31 alias onto threads 0..15 (duplicate, identical
                # data — no pred needed to fence them).
                bs_gmem_thr_copy = bs_gmem_tiled_copy.get_slice(
                    lane_idx % bs_active_threads
                )
                sBS_1d = cute.make_tensor(
                    sBS.iterator,
                    cute.make_layout((bs_tile_elems, self.bs_stages)),
                )  # (TILE_S*SCALE, bs_stages) — stage tile contiguous
                # h_k from the RAW scale pool (num_pages, h_k, PS, scaledim).
                h_k = mBS_raw.shape[1]
                b_index = coord_hb // h_k
                h_k_index = coord_hb % h_k

                for s in cutlass.range(
                    kv_split_idx, prefetch_tiles + tiles_s, kv_splits
                ):
                    # Load K scale (tile s)
                    if s < tiles_s:
                        bs_handle = bs_producer.acquire_and_advance()
                        stage = bs_handle.index
                        phys_page = mPageTable[b_index, s]
                        tBSsBS = bs_gmem_thr_copy.partition_D(sBS_1d[None, stage])
                        # Base of the (page, h_k) scale tile in the (page,
                        # h_k, PS, scaledim) h_k-major pool, from the base
                        # iterator (the GEMV idiom — this cutlass has no
                        # cute.elem_pointer): tile stride = PS*scaledim, so
                        # base = (page*h_k + h_k) * PS*scaledim. The 1D
                        # TiledCopy maps thread t -> chunk [8t, 8t+8) of the
                        # 128-element tile (lanes alias mod bs_active_threads).
                        bs_gmem_ptr = cute.make_ptr(
                            mma_dtype,
                            (
                                mBS_raw.iterator
                                + (phys_page * h_k + h_k_index)
                                * (blk_tile_s * self.scaledim)
                                + (lane_idx % bs_active_threads)
                                * bs_async_copy_elems
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=bs_copy_bytes,
                        )
                        src = cute.make_tensor(
                            bs_gmem_ptr, cute.make_layout((bs_async_copy_elems,))
                        )
                        should_load = cute.make_fragment_like(
                            tBSsBS[None, 0], cute.Boolean
                        )
                        # OOB tail rows (seqlen % PS != 0): skip the chunk like
                        # the KV gather's per-row predicate (aliased lane).
                        should_load.fill(
                            s * blk_tile_s
                            + ((lane_idx % bs_active_threads)
                               * bs_async_copy_elems
                               + (bs_async_copy_elems - 1)) // self.scaledim
                            < seqlen_k
                        )
                        cute.copy(
                            bs_gmem_tiled_copy,
                            src,
                            tBSsBS[None, 0],
                            pred=should_load,
                        )
                        bs_handle.commit()

                    # Load V scale (tile s - prefetch_tiles)
                    if s >= prefetch_tiles:
                        bs_handle = bs_producer.acquire_and_advance()
                        stage = bs_handle.index
                        phys_page = mPageTable[b_index, s - prefetch_tiles]
                        tBSsBS = bs_gmem_thr_copy.partition_D(sBS_1d[None, stage])
                        bs_gmem_ptr = cute.make_ptr(
                            mma_dtype,
                            (
                                mVBS_raw.iterator
                                + (phys_page * h_k + h_k_index)
                                * (blk_tile_s * self.scaledim)
                                + (lane_idx % bs_active_threads)
                                * bs_async_copy_elems
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=bs_copy_bytes,
                        )
                        src = cute.make_tensor(
                            bs_gmem_ptr, cute.make_layout((bs_async_copy_elems,))
                        )
                        should_load = cute.make_fragment_like(
                            tBSsBS[None, 0], cute.Boolean
                        )
                        should_load.fill(
                            (s - prefetch_tiles) * blk_tile_s
                            + ((lane_idx % bs_active_threads)
                               * bs_async_copy_elems
                               + (bs_async_copy_elems - 1)) // self.scaledim
                            < seqlen_k
                        )
                        cute.copy(
                            bs_gmem_tiled_copy,
                            src,
                            tBSsBS[None, 0],
                            pred=should_load,
                        )
                        bs_handle.commit()
            else:
                # Dense scale: TMA. gBS / the tma_partition views are bound
                # here (not in an earlier region) — region-local bindings
                # can't be read from a later sibling region (DSL scope rule).
                gBS = cute.local_tile(
                    mBS, tiler=(self.scaledim, blk_tile_s), coord=(0, None, coord_hb)
                )  # (SCALE, TILE_S, #TILE_S)
                tGSsBS, tGSgBS = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_ks,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(sBS, 0, 2),
                    gmem_tensor=cute.group_modes(gBS, 0, 2),
                )
                for s in cutlass.range(
                    kv_split_idx, prefetch_tiles + tiles_s, kv_splits
                ):
                    if s < tiles_s:
                        bs_handle = bs_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_ks,
                            tGSgBS[None, s],
                            tGSsBS[None, bs_handle.index],
                            tma_bar_ptr=bs_handle.barrier,
                        )

                    if s >= prefetch_tiles:
                        bs_handle = bs_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_vs,
                            tGSgBS[None, s - prefetch_tiles],
                            tGSsBS[None, bs_handle.index],
                            tma_bar_ptr=bs_handle.barrier,
                        )

            # Store O
            for dm in cutlass.range_constexpr(tiles_dm):
                q_producer.acquire_and_advance()  # Reuse Q load barriers to sync O store
                cute.copy(tma_atom_o, tSGsO[None, dm, 0], tSGgO[None, dm, 0])

        ##############################
        # Convert Dispatch
        ##############################
        elif warpgroup_idx in self.cvt_warpgroup_ids:
            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.cvt_regs)

            # Intermediate convert type
            cvt_type = cutlass.Float32
            if cutlass.const_expr(
                mma_dtype is cutlass.BFloat16
                and k_dtype in (cutlass.Int4, cutlass.Int8)
            ):
                cvt_type = mma_dtype

            # Initialize for multiple warpgroups if necessary
            convert_phase = 0
            if cutlass.const_expr(self.convert_warpgroups > 1):
                assert tiles_dk % self.convert_warpgroups == 0
                assert (tiles_dm * tiles_sk) % self.convert_warpgroups == 0
                convert_phase = warpgroup_idx % self.convert_warpgroups
                for _ in cutlass.range(convert_phase):
                    kv_consumer.advance()
                    cvt_producer.advance()

            # Construct tiled copy and partition K
            mma_k_bits = mma_tile_k * mma_dtype.width
            tmem_store_atom_k = cute.make_copy_atom(
                tcgen05.St16x256bOp(tcgen05.Repetition(mma_k_bits // 256)),
                mma_dtype,
            )
            smem_load_atom_k = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x16x8bOp(
                    num_matrices=4,
                    unpack_bits=(k_dtype.width if k_dtype.width < 8 else None),
                ),
                kv_smem_dtype,
            )

            tmem_store_k = tcgen05.make_tmem_copy(
                tmem_store_atom_k, tAtK_cvt[mma_dice + (0,)]
            )
            thr_store_k = tmem_store_k.get_slice(warpgroup_tidx)
            tKrK_cvt_shape = thr_store_k.partition_S(tAtK_cvt).shape[:-1]
            tKtK_cvt = thr_store_k.partition_D(tAtK_cvt)

            smem_load_k = cute.make_tiled_copy_S(smem_load_atom_k, tmem_store_k)
            thr_load_k = smem_load_k.get_slice(warpgroup_tidx)
            tKsK = thr_load_k.partition_S(tAsK)
            tKrK_shape = thr_load_k.partition_D(tAsK).shape[:-1]

            # Construct tiled copy and partition V
            tmem_store_atom_v = cute.make_copy_atom(
                tcgen05.St16x256bOp(tcgen05.Repetition(mma_k_bits // 256)), mma_dtype
            )
            smem_load_op_v = cute.nvgpu.warp.LdMatrix16x16x8bOp(
                transpose=True,
                num_matrices=2,
                unpack_bits=(v_dtype.width if v_dtype.width < 8 else None),
            )
            smem_load_atom_v = cute.make_copy_atom(smem_load_op_v, kv_smem_dtype)

            tmem_store_v = tcgen05.make_tmem_copy(
                tmem_store_atom_v, tAtV_cvt[mma_dice + (0,)]
            )
            thr_store_v = tmem_store_v.get_slice(warpgroup_tidx)
            tVrV_cvt_shape = thr_store_v.partition_S(tAtV_cvt).shape[:-1]
            tVtV_cvt = thr_store_v.partition_D(tAtV_cvt)

            smem_load_v = cute.make_tiled_copy_S(smem_load_atom_v, tmem_store_v)
            thr_load_v = smem_load_v.get_slice(warpgroup_tidx)
            tVsV = thr_load_v.partition_S(tAsV)
            tVrV_shape = thr_load_v.partition_D(tAsV).shape[:-1]

            # Partition KS - K block scale
            sKS_layout = cute.make_layout(
                shape=(
                    blk_tile_s,
                    (self.block_scaledim, self.scaledim),
                    self.bs_stages,
                ),
                stride=(self.scaledim, (0, 1), blk_tile_s * self.scaledim),
            )
            sKS = cute.make_tensor(
                sBS.iterator, sKS_layout
            )  # (TILE_S, TILE_D, bs_stages)
            sKS_mma = cute.group_modes(
                cute.flat_divide(sKS, (mma_tile_m, mma_tile_k)), 2, 4
            )  # (MMA_TILE_M, MMA_TILE_K, (#TILE_SM, #TILE_DK), bs_stages)
            tAsKS = thrblk_mma_kq.partition_A(
                sKS_mma
            )  # (MMA, #MMA_M, #MMA_K, (#TILE_SM, #TILE_DK), bs_stages)
            tKsKS = thr_load_k.partition_D(
                tAsKS
            )  # (CPY, CPY_MMA, CPY_M, CPY_K, #TILE, bs_stages)

            # Partition VS - V block scale
            sVS_layout = cute.make_layout(
                shape=(
                    (self.block_scaledim, self.scaledim),
                    blk_tile_s,
                    self.bs_stages,
                ),
                stride=((0, 1), self.scaledim, blk_tile_s * self.scaledim),
            )
            sVS = cute.make_tensor(
                sBS.iterator, sVS_layout
            )  # (TILE_D, TILE_S, bs_stages)
            sVS_mma = cute.group_modes(
                cute.flat_divide(sVS, (mma_tile_m, mma_tile_k)), 2, 4
            )  # (MMA_TILE_M, MMA_TILE_K, (#TILE_DM, #TILE_SK), bs_stages)
            tAsVS = thrblk_mma_vp.partition_A(
                sVS_mma
            )  # (MMA, #MMA_M, #MMA_K, (#TILE_DM, #TILE_SK), bs_stages)
            tVsVS = thr_load_v.partition_D(
                tAsVS
            )  # (CPY, CPY_MMA, CPY_M, CPY_K, #TILE, bs_stages)

            #
            # Sequence loop
            #
            for s in cutlass.range(prefetch_iters + iters_s):
                if s < iters_s:
                    # Load K scale
                    bs_handle = bs_consumer.wait_and_advance()
                    tKrKS = cute.make_rmem_tensor_like(
                        tKsKS[cpy_dice + (None, 0)]
                    )  # 'like' preserves 0 strides
                    cute.autovec_copy(tKsKS[cpy_dice + (None, bs_handle.index)], tKrKS)
                    cute.arch.fence_view_async_shared()
                    bs_handle.release()

                    # Convert and scale K
                    for dk in cutlass.range(
                        tiles_dk // self.convert_warpgroups, unroll=2
                    ):
                        tKrK = cute.make_rmem_tensor(tKrK_shape, kv_smem_dtype)
                        tKrK_cvt = cute.make_rmem_tensor(tKrK_cvt_shape, mma_dtype)

                        kv_handle = kv_consumer.wait_and_advance()
                        cute.copy(thr_load_k, tKsK[cpy_dice + (kv_handle.index,)], tKrK)
                        cute.arch.fence_view_async_shared()
                        kv_handle.release()

                        # Sign extend unpacked int4 to int8
                        if cutlass.const_expr(k_dtype is cutlass.Int4):
                            tKrK_unpacked_i4_vec = tKrK.load().maybe_downcast()
                            tKrK_i8_vec = cute.arch.sext_unpacked_i4_i8_intrinsic(
                                tKrK_unpacked_i4_vec, cute.size(tKrK_shape)
                            )
                            tKrK.store(
                                cute.TensorSSA(tKrK_i8_vec, tKrK_shape, cutlass.Int8)
                            )

                        coord_dk = dk * self.convert_warpgroups + convert_phase
                        scale_k = tKrKS[cpy_dice + (coord_dk,)].load()
                        tKrK_ssa = tKrK.load().to(cvt_type).to(mma_dtype) * scale_k
                        tKrK_cvt.store(tKrK_ssa.reshape(tKrK_cvt_shape))

                        cvt_handle = cvt_producer.acquire_and_advance()
                        cute.copy(
                            thr_store_k,
                            tKrK_cvt,
                            tKtK_cvt[cpy_dice + (cvt_handle.index,)],
                        )
                        cute.arch.fence_view_async_tmem_store()
                        cvt_handle.commit()

                        # Advance again for multiple warpgroups
                        for _ in cutlass.range_constexpr(self.convert_warpgroups - 1):
                            kv_consumer.advance()
                            cvt_producer.advance()

                if s >= prefetch_iters:
                    # Load V scale
                    bs_handle = bs_consumer.wait_and_advance()
                    tVrVS = cute.make_rmem_tensor_like(
                        tVsVS[cpy_dice + (None, 0)]
                    )  # 'like' preserves 0 strides
                    cute.autovec_copy(tVsVS[cpy_dice + (None, bs_handle.index)], tVrVS)
                    cute.arch.fence_view_async_shared()
                    bs_handle.release()

                    # Convert and scale V
                    for dmsk in cutlass.range(
                        tiles_dm * tiles_sk // self.convert_warpgroups, unroll=2
                    ):
                        tVrV = cute.make_rmem_tensor(tVrV_shape, kv_smem_dtype)
                        tVrV_cvt = cute.make_rmem_tensor(tVrV_cvt_shape, mma_dtype)

                        kv_handle = kv_consumer.wait_and_advance()
                        cute.copy(thr_load_v, tVsV[cpy_dice + (kv_handle.index,)], tVrV)
                        cute.arch.fence_view_async_shared()
                        kv_handle.release()

                        # Sign extend unpacked int4 to int8
                        if cutlass.const_expr(v_dtype is cutlass.Int4):
                            tVrV_unpacked_i4_vec = tVrV.load().maybe_downcast()
                            tVrV_i8_vec = cute.arch.sext_unpacked_i4_i8_intrinsic(
                                tVrV_unpacked_i4_vec, cute.size(tVrV_shape)
                            )
                            tVrV.store(
                                cute.TensorSSA(tVrV_i8_vec, tVrV_shape, cutlass.Int8)
                            )

                        coord_dmsk = dmsk * self.convert_warpgroups + convert_phase
                        scale_v = tVrVS[cpy_dice + (coord_dmsk,)].load()
                        tVrV_ssa = tVrV.load().to(cvt_type).to(mma_dtype) * scale_v
                        tVrV_cvt.store(tVrV_ssa.reshape(tVrV_cvt_shape))

                        cvt_handle = cvt_producer.acquire_and_advance()
                        cute.copy(
                            thr_store_v,
                            tVrV_cvt,
                            tVtV_cvt[cpy_dice + (cvt_handle.index,)],
                        )
                        cute.arch.fence_view_async_tmem_store()
                        cvt_handle.commit()

                        # Advance again for multiple warpgroups
                        for _ in cutlass.range_constexpr(self.convert_warpgroups - 1):
                            kv_consumer.advance()
                            cvt_producer.advance()

        ##############################
        # MMA KQ Dispatch
        ##############################
        elif warp_idx == self.mma_kq_warp_id:
            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            # Setup mma descriptors
            tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)

            # Wait for Q
            for dk in cutlass.range_constexpr(tiles_dk):
                q_consumer.wait_and_advance()

            # Sequence loop
            s_token = True  # Producer always acquires first
            for s in cutlass.range(iters_s):
                # BMM1
                k_token = cvt_consumer.try_wait()
                tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                s_handle = s_producer.acquire_and_advance(s_token)
                for dk in cutlass.range_constexpr(tiles_dk):
                    is_last_iter = dk == tiles_dk - 1
                    k_handle = cvt_consumer.wait_and_advance(k_token)
                    # Signal BMM2 to start
                    if is_last_iter:
                        mma_kq_nbar.arrive()
                    for mma_k in cutlass.range_constexpr(tAtK_cvt.shape[2]):
                        cute.gemm(
                            tiled_mma_kq,
                            tCtS[mma_dice + (s_handle.index,)],
                            tAtK_cvt[None, None, mma_k, k_handle.index],
                            tBsQ_desc[None, None, mma_k, dk],
                            tCtS[mma_dice + (s_handle.index,)],
                        )
                        if dk == 0 and mma_k == 0:
                            tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, True)
                    k_handle.release()
                    if not is_last_iter:
                        k_token = cvt_consumer.try_wait()
                s_handle.commit()

                # Advance and wait for BMM 2
                if s > 0:
                    for _ in cutlass.range_constexpr(tiles_dm * tiles_sk):
                        cvt_consumer.advance()
                    mma_vp_nbar.arrive_and_wait()
                    s_token = s_producer.try_acquire()

        ##############################
        # MMA VP Dispatch
        ##############################
        elif warp_idx == self.mma_vp_warp_id:
            # Free registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            # Setup mma descriptors
            tiled_mma_vp.set(tcgen05.Field.ACCUMULATE, True)
            tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)

            # Advance and wait for BMM1
            for _ in cutlass.range_constexpr(tiles_dk):
                cvt_consumer.advance()
            mma_kq_nbar.arrive_and_wait()

            # Sequence loop
            p_token = False
            o_token = True  # Producer always acquires first
            for s in cutlass.range(iters_s):
                # Advance and wait for BMM1
                if s < iters_s - 1:
                    for _ in cutlass.range_constexpr(tiles_dk):
                        cvt_consumer.advance()
                    mma_kq_nbar.arrive_and_wait()
                    p_token = p_consumer.try_wait()

                # BMM2
                v_token = cvt_consumer.try_wait()
                p_handle = p_consumer.wait_and_advance(p_token)
                o_handle = o_producer.acquire_and_advance(o_token)
                for sk in cutlass.range_constexpr(tiles_sk):
                    for dm in cutlass.range_constexpr(tiles_dm):
                        is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                        v_handle = cvt_consumer.wait_and_advance(v_token)
                        # Signal BMM1 to start
                        if is_last_iter:
                            mma_vp_nbar.arrive()
                        for mma_k in cutlass.range_constexpr(tAtV_cvt.shape[2]):
                            cute.gemm(
                                tiled_mma_vp,
                                tCtO[mma_dice + (dm, 0)],
                                tAtV_cvt[None, None, mma_k, v_handle.index],
                                tBsP_desc[None, None, mma_k, sk, p_handle.index],
                                tCtO[mma_dice + (dm, 0)],
                            )
                        v_handle.release()
                        if not is_last_iter:
                            v_token = cvt_consumer.try_wait()
                p_handle.release()
                o_handle.commit()
                o_token = o_producer.try_acquire()

            # Wait for signal to dealloc tmem, then dealloc
            o_producer.tail()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)

        ##############################
        # Softmax + Correction Dispatch
        ##############################
        elif warpgroup_idx == self.softmax_warpgroup_id:
            # Alloc registers
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_increase(self.softmax_regs)

            # Construct tiled copies
            tmem_op_width = 32
            tmem_op_repeat = tcgen05.Repetition(
                mma_tile_n * acc_dtype.width // tmem_op_width
            )
            tmem_load_atom_s = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tmem_op_repeat), acc_dtype
            )
            tmem_load_s = tcgen05.make_tmem_copy(
                tmem_load_atom_s, tCtS[mma_dice + (0,)]
            )
            thr_load_s = tmem_load_s.get_slice(warpgroup_tidx)

            tmem_store_atom_o = cute.make_copy_atom(
                tcgen05.St32x32bOp(tmem_op_repeat), o_dtype
            )
            tmem_store_o = tcgen05.make_tmem_copy(
                tmem_store_atom_o, tCtO[mma_dice + (0, 0)]
            )
            thr_store_o = tmem_store_o.get_slice(warpgroup_tidx)

            # Partition S and P
            tStS = thr_load_s.partition_S(
                tCtS
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, stages_sp)
            tSsP = thr_load_s.partition_D(
                tCsP
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, stages_sp)

            # Partition O
            tStO = thr_load_s.partition_S(
                tCtO
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, #TILE_HN)
            tSsO = thr_load_s.partition_D(
                tCsO
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, #TILE_HN)
            tSrO = cute.make_rmem_tensor_like(tSsO)

            # Partition colmax and initialize in RF
            tSsM = thr_load_s.partition_D(tCsM)  # (CPY, #CPY_MMA, #CPY_M, #CPY_N)
            tSrM_prev = cute.make_rmem_tensor_like(tSsM)
            tSrM_prev.fill(-cutlass.Float32.inf)

            # Partition colsum and initialize in RF
            # Each thread maintains a local colsum in RF, smem reduction happens after loop
            tSsL = thr_load_s.partition_D(
                tCsL
            )  # (CPY, #CPY_MMA, #CPY_M, #CPY_N, WARPS)
            tSrL = cute.make_rmem_tensor_like(tSsL[cpy_dice + (0,)])
            tSrL.fill(cutlass.Float32(0))

            assert warp_threads >= cute.size(tSsM)

            # get gmem colmax + colsum to store to
            inbound_hr = coord_hr * blk_tile_h + lane_idx < mM.shape[0]
            gM = cute.local_tile(
                mM, tiler=(mma_tile_n, 1), coord=(coord_hr, coord_hb)
            )  # (TILE_H) = (MMA_TILE_N)
            gM_partial = cute.local_tile(
                mM_partial,
                tiler=(mma_tile_n, 1),
                coord=(coord_hr, coord_hb, kv_split_idx),
            )
            gL_partial = cute.local_tile(
                mL_partial,
                tiler=(mma_tile_n, 1),
                coord=(coord_hr, coord_hb, kv_split_idx),
            )

            # Initialize O
            tSrO.fill(cutlass.Float32(0))
            cute.copy(thr_store_o, tSrO, tStO)

            # Initialize colsum and colmax in smem and wait
            if warpgroup_widx == 0 and lane_store_max:
                tSsM[lane_idx] = -cutlass.Float32.inf
            if warpgroup_widx == 1 and lane_store_max:
                tSsL[lane_idx] = cutlass.Float32(0)
            softmax_nbar.arrive_and_wait()

            #
            # Sequence loop
            #
            for s in cutlass.range(iters_s):
                # Load S from tmem
                s_handle = s_consumer.wait_and_advance()
                tSrS = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
                cute.copy(tmem_load_s, tStS[cpy_dice + (s_handle.index,)], tSrS)
                cute.arch.fence_view_async_tmem_load()
                s_handle.release()

                # Reduce colmax in warp RF
                tSrM = cute.make_rmem_tensor_like(tSsM)
                tSrM_lane = cutlass.Float32(0)  # Avoid dynamic register indexing
                for i in cutlass.range_constexpr(cute.size(tSrS)):
                    tSrM[i] = warp_fmax(tSrS[i])
                    if i == lane_idx:
                        tSrM_lane = tSrM[i]

                # Reduce colmax in smem
                if lane_store_max:
                    smem_fmax(tSsM.iterator + tSsM.layout(lane_idx), tSrM_lane)

                # Wait for colmax then load
                softmax_nbar.arrive_and_wait()
                cute.autovec_copy(tSsM, tSrM)

                # Compute online softmax
                tSrP = cute.make_rmem_tensor(tSsP.shape[:-1], mma_dtype)
                if cutlass.const_expr(use_tensor_ssa_math):
                    tSrP_f32 = exp2(scale_qs_log2_e * (tSrS.load() - tSrM.load()))
                    tSrP.store(tSrP_f32.to(mma_dtype))  # convert
                else:
                    tSrP_f32 = cute.make_rmem_tensor(tSrS.shape, acc_dtype)
                    for i in cutlass.range_constexpr(0, cute.size(tSrS), 2):
                        p_f32x2 = fadd2(
                            (tSrS[i], tSrS[i + 1]), (-tSrM[i], -tSrM[i + 1])
                        )
                        p_f32x2 = fmul2(p_f32x2, (scale_qs_log2_e, scale_qs_log2_e))
                        tSrP_f32[i] = exp2(p_f32x2[0])
                        tSrP_f32[i + 1] = exp2(p_f32x2[1])
                    tSrP.store(tSrP_f32.load().to(mma_dtype))

                # Store P to smem
                p_handle = p_producer.acquire_and_advance()
                cute.autovec_copy(tSrP, tSsP[cpy_dice + (p_handle.index,)])
                cute.arch.fence_view_async_shared()
                p_handle.commit()

                # Compute correction and correct colsum
                if cutlass.const_expr(use_tensor_ssa_math):
                    correction = exp2(
                        scale_qs_log2_e * (tSrM_prev.load() - tSrM.load())
                    )
                    tSrL.store(tSrL.load() * correction + tSrP_f32)
                else:
                    correction = cute.make_rmem_tensor_like(tSrM)
                    for i in cutlass.range_constexpr(0, cute.size(tSrM), 2):
                        c_f32x2 = fadd2(
                            (tSrM_prev[i], tSrM_prev[i + 1]), (-tSrM[i], -tSrM[i + 1])
                        )
                        c_f32x2 = fmul2(c_f32x2, (scale_qs_log2_e, scale_qs_log2_e))
                        c_f32x2 = (exp2(c_f32x2[0]), exp2(c_f32x2[1]))
                        correction[i] = c_f32x2[0]
                        correction[i + 1] = c_f32x2[1]
                        l_f32x2 = ffma2(
                            c_f32x2,
                            (tSrL[i], tSrL[i + 1]),
                            (tSrP_f32[i], tSrP_f32[i + 1]),
                        )
                        tSrL[i] = l_f32x2[0]
                        tSrL[i + 1] = l_f32x2[1]

                # Correct O
                if s > 0:
                    # Wait for O
                    o_handle = o_consumer.wait_and_advance()

                    # Apply correction
                    for dm in cutlass.range_constexpr(tiles_dm):
                        tSrO_dm = cute.make_rmem_tensor_like(tSsO[cpy_dice + (0, 0)])
                        cute.copy(thr_load_s, tStO[cpy_dice + (dm, 0)], tSrO_dm)

                        for i in cutlass.range_constexpr(0, cute.size(tSrO_dm), 2):
                            o_f32x2 = fmul2(
                                (tSrO_dm[i], tSrO_dm[i + 1]),
                                (correction[i], correction[i + 1]),
                            )
                            tSrO_dm[i] = o_f32x2[0]
                            tSrO_dm[i + 1] = o_f32x2[1]

                        cute.copy(thr_store_o, tSrO_dm, tStO[cpy_dice + (dm, 0)])

                    # Notify MMA
                    cute.arch.fence_view_async_tmem_store()
                    o_handle.release()

                # Update colmax
                tSrM_prev.store(tSrM.load())

            #
            # Softmax Epilogue
            #

            # Reduce colsum in warp RF
            tSrL_lane = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(cute.size(tSrL)):
                tSrL[i] = cute.arch.warp_reduction_sum(tSrL[i])
                if i == lane_idx:
                    tSrL_lane = tSrL[i]

            # Store partial colsum in smem
            if lane_store_max:
                tSsL[cpy_dice + (warpgroup_widx,)][lane_idx] = tSrL_lane

            # Wait for colsum
            softmax_nbar.arrive_and_wait()

            if warpgroup_widx == 0 and lane_store_max and inbound_hr:
                # Load colsum and colmax
                sL_lane_wg = sL[0, lane_idx, None]
                sL_lane = sL_lane_wg[0] + sL_lane_wg[1] + sL_lane_wg[2] + sL_lane_wg[3]
                sM_lane = sM[0, lane_idx]

                # Scale colmax
                sM_lane = sM_lane * scale_qs

                # Store colsum and colmax
                gL_partial[lane_idx] = sL_lane
                gM_partial[lane_idx] = sM_lane
                gmem_fmax(gM.iterator + gM.layout(lane_idx), sM_lane)

            o_handle = o_consumer.wait_and_advance()
            cute.copy(thr_load_s, tStO, tSrO)
            cute.arch.fence_view_async_tmem_load()
            o_handle.release()  # Final release signals tmem dealloc

            # Store O to smem
            for dm in cutlass.range_constexpr(tiles_dm):
                tOrO_dm = tSrO[cpy_dice + (dm, 0)]
                tOsO_dm = tSsO[cpy_dice + (dm, 0)]

                cute.autovec_copy(tOrO_dm, tOsO_dm)
                cute.arch.fence_view_async_shared()

                # Reuse Q consumer barriers to notify O TMA store
                q_consumer.release()
                q_consumer.advance()

        return

    @staticmethod
    @cute.kernel
    def reduction(
        o: cute.Tensor,
        m: cute.Tensor,
        l: cute.Tensor,
        o_partial: cute.Tensor,
        m_partial: cute.Tensor,
        l_partial: cute.Tensor,
        scale_o: cutlass.Float32,
    ):
        d_blk_idx, coord_h, coord_b = cute.arch.block_idx()
        d_per_blk, _, _ = cute.arch.block_dim()
        d_idx, _, _ = cute.arch.thread_idx()
        coord_d = d_blk_idx * d_per_blk + d_idx

        o_dhb = cutlass.Float32(0)
        m_hb = m[coord_h, coord_b]
        l_hb = cutlass.Float32(0)

        o_partial_dhb = o_partial[coord_d, coord_h, coord_b, None]
        m_partial_hb = m_partial[coord_h, coord_b, None]
        l_partial_hb = l_partial[coord_h, coord_b, None]

        for partial_idx in cutlass.range(o_partial.shape[-1]):
            correction = exp2(log2_e * (m_partial_hb[partial_idx] - m_hb))
            o_dhb += correction * o_partial_dhb[partial_idx]
            l_hb += correction * l_partial_hb[partial_idx]

        if coord_d < o.shape[0]:
            o[coord_d, coord_h, coord_b] = o.element_type(scale_o * (o_dhb / l_hb))

        if d_blk_idx == 0 and d_idx == 0:
            l[coord_h, coord_b] = m_hb + cute.math.log(l_hb, fastmath=False)

        return


def run(
    batches: int = 1,
    seqlen: int = 1024,
    heads_q: int = 24,
    heads_k: int = 4,
    headdim: int = 256,
    block_scaledim: int = 256,
    kv_splits: int = 0,
    q_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
    kv_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
    o_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    tolerance: float = 0.1,
    scale_q: float = 1.0,
    scale_o: float = 1.0,
    scale_s: float = 0.0,
    warmup_iterations: int = 0,
    iterations: int = 0,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    paged: bool = False,
    page_size: int = 128,
    **kwargs,
):
    print("Running Blackwell SM100 Mixed Input FMHA Decode test with:")
    print(f"\tbatches: {batches}, seqlen: {seqlen}")
    print(f"\theads_q: {heads_q}, heads_k: {heads_k}")
    print(f"\theaddim: {headdim}, block_scaledim: {block_scaledim}")
    print(f"\tkv_splits: {kv_splits}")
    print(f"\tq_dtype: {q_dtype}")
    print(f"\tkv_dtype: {kv_dtype}")
    print(f"\to_dtype: {o_dtype}")
    print(f"\tacc_dtype: {acc_dtype}")
    print(f"\ttolerance: {tolerance}")
    print(f"\tscale_q: {scale_q}")
    print(f"\tscale_o: {scale_o}")
    print(f"\tscale_s: {scale_s}")
    print(f"\twarmup_iterations: {warmup_iterations}")
    print(f"\titerations: {iterations}")
    print(f"\tskip_ref_check: {skip_ref_check}")
    print(f"\tuse_cold_l2: {use_cold_l2}")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    #
    # Config Kernel
    #
    grouped_heads = heads_q // heads_k
    convert_warpgroups = 1
    if headdim == 512 and grouped_heads == 8 and kv_dtype.width == 4:
        convert_warpgroups = 4
    elif headdim > 128:
        convert_warpgroups = 2
    fmha = MixedInputFusedMultiHeadAttentionDecode(
        headdim=headdim,
        block_scaledim=block_scaledim,
        grouped_head_tile=min(cute.round_up(grouped_heads, 8), 32),
        convert_warpgroups=convert_warpgroups,
        page_size=page_size if paged else 0,
    )

    if scale_s == 0.0:  # default to 1/sqrt(d)
        scale_s = 1.0 / math.sqrt(headdim)
    scale_qs = scale_q * scale_s

    if kv_splits == 0:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        sm_count = 148 if sm_count <= 0 else sm_count
        grid_yz = batches * heads_k * math.ceil(grouped_heads / fmha.grouped_head_tile)
        kv_splits = sm_count // grid_yz  # 1 wave
        kv_splits = max(1, kv_splits)
        if sm_count == 148 and grid_yz == 32:
            kv_splits = 9  # 2 waves
        print(f"\tauto kv_splits: {kv_splits}")

    seqlen_q = 1
    seqlen_k = seqlen

    problem_shape = (batches, heads_q, heads_k, seqlen_k, headdim)

    fmha.can_implement(problem_shape, kv_splits, q_dtype, kv_dtype, o_dtype, acc_dtype)

    #
    # Allocate Tensors
    #
    torch.manual_seed(1111)

    def create_tensor(shape, dtype, init=True):
        init_type = cutlass_torch.TensorInitType.RANDOM
        init_config = cutlass_torch.RandomInitConfig(min_val=-2, max_val=2)

        if init is False or init is None:
            init_type = cutlass_torch.TensorInitType.SKIP
            init_config = None
        elif isinstance(init, int) or isinstance(init, float):
            init_type = cutlass_torch.TensorInitType.SCALAR
            init_config = cutlass_torch.ScalarInitConfig(value=init)
        elif isinstance(init, tuple) or isinstance(init, list):
            if len(init) == 2:
                init_type = cutlass_torch.TensorInitType.RANDOM
                init_config = cutlass_torch.RandomInitConfig(
                    min_val=init[0], max_val=init[1]
                )
            if len(init) == 3:
                init_type = cutlass_torch.TensorInitType.GAUSSIAN
                init_config = cutlass_torch.RandomInitConfig(
                    mean=init[0], std=init[1], scale=init[2]
                )

        f32_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
            shape,
            torch.float32,
            permute_order=None,
            init_type=init_type,
            init_config=init_config,
        )

        _, torch_tensor = cutlass_torch.cute_tensor_like(
            f32_torch_tensor,
            dtype,
            is_dynamic_layout=True,
            assumed_align=16,
        )

        # Create dtype cute tensor with offset (gpu)
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
        cute_tensor.element_type = dtype

        return (
            f32_torch_tensor,
            cute_tensor,
            torch_tensor,
        )

    qo_shape = (kv_splits, batches, heads_q, seqlen_q, headdim)
    kv_shape = (batches, heads_k, seqlen_k, headdim)
    scale_shape = (batches, heads_k, seqlen_k, fmha.scaledim)

    _init_mag = int(os.environ.get("SPARKCVT_INIT", "8"))
    _qkv_init = [-_init_mag, _init_mag - 1]
    q_ref, q_cute, q_torch = create_tensor(qo_shape[1:], q_dtype, init=_qkv_init)
    k_ref, k_cute, k_torch = create_tensor(kv_shape, kv_dtype, init=_qkv_init)
    v_ref, v_cute, v_torch = create_tensor(kv_shape, kv_dtype, init=_qkv_init)
    k_scale_ref, k_scale_cute, k_scale_torch = create_tensor(
        scale_shape, q_dtype, init=[-2, 2]
    )
    v_scale_ref, v_scale_cute, v_scale_torch = create_tensor(
        scale_shape, q_dtype, init=[-2, 2]
    )
    if os.environ.get("SPARKCVT_PAT") == "1":
        # Readable K-fill pattern: K[s, 0] = s % 7 (fp8-exact), K[:, d>0] = 0,
        # Q = e_0, V constant, unit scales. A correct K fill gives scores
        # S[s] = s % 7 -> m = 6 (times scale_qs); a d-shift or s/d transpose
        # gives m = 0.
        # Row- AND column-dependent (any s/d remap or shift changes the
        # scores): K[s, d] = (s + d) % 7, Q[d] = d % 3, V[s, d] = s % 7 —
        # all fp8/bf16-exact.
        dev = k_torch.device
        s_idx = torch.arange(seqlen_k, device=dev, dtype=torch.float32)
        d_idx = torch.arange(headdim, device=dev, dtype=torch.float32)
        k_pat = ((s_idx[:, None] + d_idx[None, :]) % 7)  # (s, d)
        v_pat = (s_idx[:, None] % 7)  # (s, d) broadcast
        q_pat = (d_idx % 3)  # (d,)
        for _t in (q_torch, q_ref):
            _t.zero_()
            for _h in range(heads_q):
                _t[0, _h, 0, :].copy_(q_pat.to(_t.dtype))
        for _t in (k_torch, k_ref):
            _t.zero_()
            for _h in range(heads_k):
                _t[0, _h, :, :].copy_(k_pat.to(_t.dtype))
        for _t in (v_torch, v_ref):
            _t.zero_()
            for _h in range(heads_k):
                _t[0, _h, :, :].copy_(v_pat.to(_t.dtype))
        for _t in (
            k_scale_torch,
            k_scale_ref,
            v_scale_torch,
            v_scale_ref,
        ):
            _t.fill_(1.0)
    if os.environ.get("SPARKCVT_PAT2") == "1":
        # K = 0 (scores all 0 -> uniform softmax), Q = 1, V random:
        # O = mean_s(V_s) — isolates the V gather (and V scale).
        for _t in (q_torch, q_ref):
            _t.fill_(1.0)
        for _t in (k_torch, k_ref):
            _t.zero_()
    if os.environ.get("SPARKCVT_PAT3") == "1":
        # V row-dependent: V[s, d] = s % 7 (fp8-exact), K = 0, Q = 1 ->
        # O[d] = mean_s(s % 7); any s-remap of the V gather breaks it.
        dev = k_torch.device
        _vpat = (
            torch.arange(seqlen_k, device=dev, dtype=torch.float32) % 7
        )
        for _t in (q_torch, q_ref):
            _t.fill_(1.0)
        for _t in (k_torch, k_ref):
            _t.zero_()
        if os.environ.get("SPARKCVT_PAT4") == "1":
            # V[s, h, d] = h: O[g] = the h_k-mixture mean of group g.
            for _t in (v_torch, v_ref):
                for _h in range(heads_k):
                    _t[0, _h, :, :].fill_(_h)
        else:
            for _t in (v_torch, v_ref):
                _t.zero_()
                for _h in range(heads_k):
                    _vpat_h = (
                        (torch.arange(seqlen_k, device=dev) + 37 * _h) % 7
                    ).to(_t.dtype)
                    _t[0, _h, :, :].copy_(_vpat_h[:, None])
    if os.environ.get("SPARKCVT_PAT6") == "1":
        # V[s, d] = ((s + d) % 7) - 3 (fp8-exact, SIGNED -3..3, varies along
        # BOTH s and d), K = 0, Q = 1. Identical structure to PAT5 but signed:
        # if PAT5 passes and PAT6 fails, the bug is sign handling (not mapping).
        dev = k_torch.device
        s_idx = torch.arange(seqlen_k, device=dev, dtype=torch.float32)
        d_idx = torch.arange(headdim, device=dev, dtype=torch.float32)
        _vpat6 = (((s_idx[:, None] + d_idx[None, :]) % 7) - 3)  # (s, d)
        for _t in (q_torch, q_ref):
            _t.fill_(1.0)
        for _t in (k_torch, k_ref):
            _t.zero_()
        for _t in (v_torch, v_ref):
            _t.zero_()
            for _h in range(heads_k):
                _t[0, _h, :, :].copy_(_vpat6.to(_t.dtype))
    if os.environ.get("SPARKCVT_PATSCALE") == "1":
        # K=V=Q=1 (trivially-correct data path) with NON-UNIFORM per-s
        # scales -> isolates the K/V SCALE gather. Every other pattern test
        # uses scale=1, which masks any scale-gather misread (wrong s/h_k/
        # page all still read 1.0). Here a wrong per-s scale changes the
        # softmax (k_scale) and/or the output (v_scale).
        s_idx = torch.arange(seqlen_k, device=k_torch.device, dtype=torch.float32)
        for _t in (q_torch, q_ref):
            _t.fill_(1.0)
        for _t in (k_torch, k_ref):
            _t.fill_(1.0)
        for _t in (v_torch, v_ref):
            _t.fill_(1.0)
        for _t in (k_scale_torch, k_scale_ref):
            _t.zero_()
            for _h in range(heads_k):
                _t[0, _h, :, 0].copy_(((s_idx % 4) + 1).to(_t.dtype))
        for _t in (v_scale_torch, v_scale_ref):
            _t.zero_()
            for _h in range(heads_k):
                _t[0, _h, :, 0].copy_(((s_idx % 3) + 1).to(_t.dtype))
    if os.environ.get("SPARKCVT_PATS") == "1":
        # K[s, d] = s % 5, V[s, d] = s % 5 (both s-dependent, fp8-exact),
        # Q = 1, scale = 1. softmax(256 * (s % 5)) peaks at s % 5 == 4, so
        # O ~= the V value at the peak. A CONSISTENT s-permutation of K and
        # V is masked (softmax/mean invariance), but an INCONSISTENT one
        # (K's s-order != V's s-order, from the different K/V fill phases)
        # pairs the wrong K-s with the wrong V-s and changes O. Isolates
        # K/V s-alignment.
        s_idx = torch.arange(seqlen_k, device=k_torch.device, dtype=torch.float32)
        _kpat = (s_idx % 5)[:, None]  # (s, 1) -> broadcast over d
        for _t in (q_torch, q_ref):
            _t.fill_(1.0)
        for _t in (k_torch, k_ref):
            _t.zero_()
            for _h in range(heads_k):
                _t[0, _h, :, :].copy_(_kpat.to(_t.dtype))
        for _t in (v_torch, v_ref):
            _t.zero_()
            for _h in range(heads_k):
                _t[0, _h, :, :].copy_(_kpat.to(_t.dtype))
    if os.environ.get("SPARKCVT_PAT5") == "1":
        # V[s, d] = (s + d) % 7 (fp8-exact, varies along BOTH s and d),
        # K = 0, Q = 1 -> O[d] = mean_s((s + d) % 7). Unlike PAT3/PAT4
        # (constant along d), this isolates the V d-mapping (any V s/d
        # remap or transpose breaks it). Non-negative, so sign is not a
        # factor (rules in/out a conversion sign bug vs a mapping bug).
        dev = k_torch.device
        s_idx = torch.arange(seqlen_k, device=dev, dtype=torch.float32)
        d_idx = torch.arange(headdim, device=dev, dtype=torch.float32)
        _vpat5 = ((s_idx[:, None] + d_idx[None, :]) % 7)  # (s, d)
        for _t in (q_torch, q_ref):
            _t.fill_(1.0)
        for _t in (k_torch, k_ref):
            _t.zero_()
        for _t in (v_torch, v_ref):
            _t.zero_()
            for _h in range(heads_k):
                _t[0, _h, :, :].copy_(_vpat5.to(_t.dtype))
    if os.environ.get("SPARKCVT_SCALE1") == "1":
        # Unit scales: isolates the K/V gather from the scale gather.
        for _t in (
            k_scale_ref,
            k_scale_torch,
            v_scale_ref,
            v_scale_torch,
        ):
            _t.fill_(1.0)
    _, o_cute, o_torch = create_tensor(qo_shape[1:], o_dtype, init=False)
    _, m_cute, m_torch = create_tensor(qo_shape[1:-1], acc_dtype, init=-math.inf)
    _, l_cute, l_torch = create_tensor(qo_shape[1:-1], acc_dtype, init=False)
    _, o_partial_cute, o_partial_torch = create_tensor(qo_shape, acc_dtype, init=0)
    _, m_partial_cute, m_partial_torch = create_tensor(
        qo_shape[:-1], acc_dtype, init=-math.inf
    )
    _, l_partial_cute, l_partial_torch = create_tensor(qo_shape[:-1], acc_dtype, init=0)

    #
    # Paged KV: reshape the dense KV into the paged pool + the identity page table.
    # The kernel's PAGED leg (mPageTable_iter is not None) reads the pool via a
    # cp.async page-table gather; the identity table makes the pool bit-exact
    # with the dense KV, so the torch refcheck (dense) still applies.
    #
    paged_args = (None, None, None)
    if paged:
        PS = page_size
        pages_per_bh = seqlen_k // PS
        assert seqlen_k % PS == 0, (
            f"seqlen ({seqlen_k}) must be a multiple of page_size ({PS})"
        )
        num_pages = batches * pages_per_bh

        def to_pool(t, d_last):
            # t: (b, h_k, s_k, D) dense -> (num_pages, PS, h_k, d_last) pool.
            # The permute must put PS (dim 3) BEFORE h_k (dim 1) so the
            # .reshape(num_pages, PS, h_k, d_last) is order-consistent; the old
            # .permute(0,2,1,3,4) (h before PS) made the reshape copy scramble
            # the h/PS axes (the kernel KV pool is PS-major, (num_pages, PS,
            # h_k, d) D-last — see the PAGED k/v layout).
            return (
                t.reshape(batches, heads_k, pages_per_bh, PS, d_last)
                .permute(0, 2, 3, 1, 4)
                .reshape(num_pages, PS, heads_k, d_last)
                .contiguous()
            )

        def to_scale_pool(t, d_last):
            # (b, h_k, s_k, d_last) -> (num_pages, h_k, PS, d_last): h_k-MAJOR
            # (see scale_pool_layout — cp.async needs 4-B-aligned tile bases,
            # which only h_k-major gives when scaledim == 1).
            return (
                t.reshape(batches, heads_k, pages_per_bh, PS, d_last)
                .permute(0, 2, 1, 3, 4)
                .reshape(num_pages, heads_k, PS, d_last)
                .contiguous()
            )

        k_pool_torch = to_pool(k_torch, headdim)
        v_pool_torch = to_pool(v_torch, headdim)
        k_scale_pool_torch = to_scale_pool(k_scale_torch, fmha.scaledim)
        v_scale_pool_torch = to_scale_pool(v_scale_torch, fmha.scaledim)

        k_cute = from_dlpack(k_pool_torch, assumed_align=16)
        k_cute.element_type = kv_dtype
        v_cute = from_dlpack(v_pool_torch, assumed_align=16)
        v_cute.element_type = kv_dtype
        k_scale_cute = from_dlpack(k_scale_pool_torch, assumed_align=16)
        k_scale_cute.element_type = q_dtype
        v_scale_cute = from_dlpack(v_scale_pool_torch, assumed_align=16)
        v_scale_cute.element_type = q_dtype

        # Identity page table: (b, num_pages) int32. Batch b's tile s maps to the
        # physical page b*pages_per_bh + s; only the first pages_per_bh columns
        # are read (the kernel iterates tile_s < ceil_div(seqlen_k, PS)).
        mPageTable_torch = torch.zeros(
            batches, num_pages, dtype=torch.int32, device=k_torch.device
        )
        for bb in range(batches):
            for s in range(pages_per_bh):
                mPageTable_torch[bb, s] = bb * pages_per_bh + s
        mPageTable_cute = from_dlpack(mPageTable_torch, assumed_align=16)
        mPageTable_cute.element_type = cutlass.Int32

        paged_args = (
            mPageTable_cute.iterator,
            cutlass.Int32(seqlen_k),
            cutlass.Int32(num_pages),
        )

    #
    # Compile
    #
    current_stream = cutlass_torch.default_stream()
    compiled_fmha = cute.compile(
        fmha,
        problem_shape,
        kv_splits,
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        k_scale_cute.iterator,
        v_scale_cute.iterator,
        o_cute.iterator,
        m_cute.iterator,
        l_cute.iterator,
        o_partial_cute.iterator,
        m_partial_cute.iterator,
        l_partial_cute.iterator,
        scale_qs,
        scale_o,
        current_stream,
        paged_args[0],
        paged_args[1],
        paged_args[2],
        options="--opt-level 2",
    )
    print("Finished Compiling")

    #
    # Refcheck
    #
    def run_torch_fmha(
        q_ref, k_ref, v_ref, k_scale_ref, v_scale_ref, scale_qs=1.0, scale_o=1.0
    ):
        for i in range(0, headdim // block_scaledim):
            j = i * block_scaledim
            k_ref[..., j : j + block_scaledim] *= k_scale_ref[..., i : i + 1]
            v_ref[..., j : j + block_scaledim] *= v_scale_ref[..., i : i + 1]

        with sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH], set_priority=True
        ):
            o_ref = scaled_dot_product_attention(
                q_ref,
                k_ref,
                v_ref,
                attn_mask=None,
                dropout_p=0.0,
                scale=scale_qs,
                is_causal=False,
                enable_gqa=(heads_q != heads_k),
            )

        return o_ref * scale_o

    if not skip_ref_check:
        # Execute kernel once for reference checking
        print("Running...")
        compiled_fmha(
            problem_shape,
            kv_splits,
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            k_scale_cute.iterator,
            v_scale_cute.iterator,
            o_cute.iterator,
            m_cute.iterator,
            l_cute.iterator,
            o_partial_cute.iterator,
            m_partial_cute.iterator,
            l_partial_cute.iterator,
            scale_qs,
            scale_o,
            current_stream,
            paged_args[0],
            paged_args[1],
            paged_args[2],
        )
        print("Verifying results...")
        # SPARKCVT_REF8=1: compare against the EXACT fp8/bf16 values the
        # kernel reads (the *_torch pool data) rather than the float32
        # pre-rounding refs — isolates a real data-path bug from the
        # harness's float-vs-fp8 ref mismatch.
        if os.environ.get("SPARKCVT_REF8") == "1":
            o_ref = run_torch_fmha(
                q_torch.float().cpu(),
                k_torch.float().cpu(),
                v_torch.float().cpu(),
                k_scale_torch.float().cpu(),
                v_scale_torch.float().cpu(),
                scale_qs,
                scale_o,
            )
        else:
            o_ref = run_torch_fmha(
                q_ref, k_ref, v_ref, k_scale_ref, v_scale_ref, scale_qs, scale_o
            )
        if os.environ.get("SPARKCVT_DIAG") == "1":
            od = (o_torch.float().cpu() - o_ref).abs()
            print(
                f"DIAG max={od.max().item():.4g} "
                f"mean={od.mean().item():.4g} "
                f"frac>0.1={(od > 0.1).float().mean().item():.3f} "
                f"shape={tuple(od.shape)}"
            )
            ph = od.amax(dim=(0, 2, 3))
            print("DIAG per-hq max:", torch.round(ph, decimals=3).tolist())
            _gqa = heads_q // heads_k
            _cands = []
            for _h in range(heads_k):
                _cands.append(
                    float(
                        ((torch.arange(seqlen_k) + 37 * _h) % 7).float()
                        .mean()
                    )
                )
            _kern_m = o_torch.float().cpu().mean(dim=(0, 2, 3))
            _ref_m = o_ref.float().mean(dim=(0, 2, 3))
            print(
                "DIAG per-hq kern-mean:",
                [round(x, 4) for x in _kern_m.tolist()],
            )
            print(
                "DIAG per-hq ref-mean:",
                [round(x, 4) for x in _ref_m.tolist()],
            )
            print(
                "DIAG V-mean-cands h_k0..3:",
                [round(x, 4) for x in _cands],
            )
            _diffh = _kern_m - _ref_m
            print(
                "DIAG per-hq signed diff:",
                [round(x, 4) for x in _diffh.tolist()],
            )
            _i, _j, _k, _l = (0, 12, 0, 82)
            _hk = _j // (heads_q // heads_k)
            print(
                "DIAG O[0,12,0,82] kern=",
                o_torch.float().cpu()[_i, _j, _k, _l].item(),
                "ref=",
                o_ref[_i, _j, _k, _l].item(),
            )
            print(
                "DIAG Vcol(h_k",
                _hk,
                ",d82) mean=",
                v_ref[_i, _hk, :, _l].float().mean().item(),
                "Vcol vals[0:4]=",
                v_ref[_i, _hk, :4, _l].float().tolist(),
            )
            _d0 = o_torch.float().cpu()[0, 0, 0, :] - o_ref[0, 0, 0, :].float()
            _bad = [
                _d for _d in range(256) if abs(_d0[_d].item()) > 1e-6
            ]
            print("DIAG d-diff[0:16]=", [round(x, 4) for x in _d0[:16].tolist()])
            print("DIAG d-diff[128:144]=", [round(x, 4) for x in _d0[128:144].tolist()])
            print(
                "DIAG #bad d=",
                len(_bad),
                "bad range=[",
                min(_bad) if _bad else None,
                ",",
                max(_bad) if _bad else None,
                "]",
                "first20=",
                _bad[:20],
            )
            rph = o_ref.abs().amax(dim=(0, 2, 3))
            print("DIAG per-hq REF max:", torch.round(rph, decimals=3).tolist())
            kph = o_torch.float().cpu().abs().amax(dim=(0, 2, 3))
            print("DIAG per-hq KERN max:", torch.round(kph, decimals=3).tolist())
            if not m_torch.is_nested:
                gqa = heads_q // heads_k
                k_exp = k_ref.repeat_interleave(gqa, dim=1)
                s_ref = q_ref @ k_exp.transpose(-1, -2)
                m_ref = s_ref.max(dim=-1).values * scale_qs
                md = (m_torch.float().cpu() - m_ref).abs()
                print(
                    "DIAG m: max_diff=",
                    md.max().item(),
                    "ref[0,:6,0]=",
                    torch.round(m_ref[0, :6, 0], decimals=3).tolist(),
                    "kern[0,:6,0]=",
                    torch.round(m_torch.float().cpu()[0, :6, 0], decimals=3).tolist(),
                )
                # Transposed-storage hypothesis: if the cp.async fill put
                # K[d, s] at storage(s, d), the scores would be (K @ q)[s].
                Kx = k_ref[0][torch.arange(heads_q, device=k_ref.device) // gqa]
                qm = q_ref[0, :, 0, :]
                s_trans = torch.einsum("hsd,hd->hs", Kx, qm)  # (h_q, s)
                print(
                    "DIAG m_trans(0,:6)=",
                    torch.round(
                        s_trans[:6].max(dim=1).values * scale_qs, decimals=3
                    ).tolist(),
                )
        torch.testing.assert_close(
            o_ref, o_torch.float().cpu(), atol=tolerance, rtol=1e-05
        )

    def generate_tensors():
        _, q_cute, _ = create_tensor(qo_shape[1:], q_dtype, init=[-8, 7])
        _, k_cute, k_torch_g = create_tensor(kv_shape, kv_dtype, init=[-8, 7])
        _, v_cute, v_torch_g = create_tensor(kv_shape, kv_dtype, init=[-8, 7])
        _, k_scale_cute, k_scale_torch_g = create_tensor(
            scale_shape, q_dtype, init=[-8, 7]
        )
        _, v_scale_cute, v_scale_torch_g = create_tensor(
            scale_shape, q_dtype, init=[-8, 7]
        )
        _, o_cute, _ = create_tensor(qo_shape[1:], o_dtype, init=False)
        _, m_cute, _ = create_tensor(qo_shape[1:-1], acc_dtype, init=-math.inf)
        _, l_cute, _ = create_tensor(qo_shape[1:-1], acc_dtype, init=False)
        _, o_partial_cute, _ = create_tensor(qo_shape, acc_dtype, init=0)
        _, m_partial_cute, _ = create_tensor(qo_shape[:-1], acc_dtype, init=-math.inf)
        _, l_partial_cute, _ = create_tensor(qo_shape[:-1], acc_dtype, init=0)

        if paged:
            PS = page_size
            pages_per_bh = seqlen_k // PS
            num_pages = batches * pages_per_bh

            def to_pool_g(t, d_last):
                # KV pool: (num_pages, PS, h_k, d_last) PS-major (see to_pool).
                return (
                    t.reshape(batches, heads_k, pages_per_bh, PS, d_last)
                    .permute(0, 2, 3, 1, 4)
                    .reshape(num_pages, PS, heads_k, d_last)
                    .contiguous()
                )

            def to_scale_pool_g(t, d_last):
                # scale pool: (num_pages, h_k, PS, d_last) h_k-major (see
                # to_scale_pool — cp.async needs 4-B-aligned tile bases, which
                # only h_k-major gives when scaledim == 1).
                return (
                    t.reshape(batches, heads_k, pages_per_bh, PS, d_last)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(num_pages, heads_k, PS, d_last)
                    .contiguous()
                )

            k_pool_torch = to_pool_g(k_torch_g, headdim)
            v_pool_torch = to_pool_g(v_torch_g, headdim)
            k_scale_pool_torch = to_scale_pool_g(k_scale_torch_g, fmha.scaledim)
            v_scale_pool_torch = to_scale_pool_g(v_scale_torch_g, fmha.scaledim)
            k_cute = from_dlpack(k_pool_torch, assumed_align=16)
            k_cute.element_type = kv_dtype
            v_cute = from_dlpack(v_pool_torch, assumed_align=16)
            v_cute.element_type = kv_dtype
            k_scale_cute = from_dlpack(k_scale_pool_torch, assumed_align=16)
            k_scale_cute.element_type = q_dtype
            v_scale_cute = from_dlpack(v_scale_pool_torch, assumed_align=16)
            v_scale_cute.element_type = q_dtype

        args = testing.JitArguments(
            problem_shape,
            kv_splits,
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            k_scale_cute.iterator,
            v_scale_cute.iterator,
            o_cute.iterator,
            m_cute.iterator,
            l_cute.iterator,
            o_partial_cute.iterator,
            m_partial_cute.iterator,
            l_partial_cute.iterator,
            scale_qs,
            scale_o,
            current_stream,
            paged_args[0],
            paged_args[1],
            paged_args[2],
        )
        args.add_to_scope(
            [
                q_cute,
                k_cute,
                v_cute,
                k_scale_cute,
                v_scale_cute,
                o_cute,
                m_cute,
                l_cute,
                o_partial_cute,
                m_partial_cute,
                l_partial_cute,
            ]
        )
        return args

    #
    # Profile
    #
    workspace_count = 1
    if use_cold_l2:
        q_torch_effective = q_torch.values() if q_torch.is_nested else q_torch
        k_torch_effective = k_torch.values() if k_torch.is_nested else k_torch
        v_torch_effective = v_torch.values() if v_torch.is_nested else v_torch
        k_scale_torch_effective = (
            k_scale_torch.values() if k_scale_torch.is_nested else k_scale_torch
        )
        v_scale_torch_effective = (
            v_scale_torch.values() if v_scale_torch.is_nested else v_scale_torch
        )
        o_torch_effective = o_torch.values() if o_torch.is_nested else o_torch
        m_torch_effective = m_torch.values() if m_torch.is_nested else m_torch
        l_torch_effective = l_torch.values() if l_torch.is_nested else l_torch
        o_partial_torch_effective = (
            o_partial_torch.values() if o_partial_torch.is_nested else o_partial_torch
        )
        m_partial_torch_effective = (
            m_partial_torch.values() if m_partial_torch.is_nested else m_partial_torch
        )
        l_partial_torch_effective = (
            l_partial_torch.values() if l_partial_torch.is_nested else l_partial_torch
        )
        one_workspace_bytes = (
            q_torch_effective.numel() * q_torch_effective.element_size()
            + k_torch_effective.numel() * k_torch_effective.element_size()
            + v_torch_effective.numel() * v_torch_effective.element_size()
            + k_scale_torch_effective.numel() * k_scale_torch_effective.element_size()
            + v_scale_torch_effective.numel() * v_scale_torch_effective.element_size()
            + o_torch_effective.numel() * o_torch_effective.element_size()
            + m_torch_effective.numel() * m_torch_effective.element_size()
            + l_torch_effective.numel() * l_torch_effective.element_size()
            + o_partial_torch_effective.numel()
            * o_partial_torch_effective.element_size()
            + m_partial_torch_effective.numel()
            * m_partial_torch_effective.element_size()
            + l_partial_torch_effective.numel()
            * l_partial_torch_effective.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = 0.0
    if iterations > 0:
        exec_time = testing.benchmark(
            compiled_fmha,
            workspace_generator=generate_tensors,
            workspace_count=workspace_count,
            stream=current_stream,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        )
        print(exec_time)

    return exec_time  # Return execution time in microseconds


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str):
        try:
            return tuple(cutlass.int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(description="Example of FMHA on Blackwell.")

    parser.add_argument(
        "--batches", "--batch", "--b", type=int, default=1, help="batch size"
    )

    parser.add_argument(
        "--seqlen",
        "--seqlen_k",
        "--seq",
        "--s",
        type=int,
        default=1024,
        help="key/value sequence length",
    )

    parser.add_argument(        "--heads_q", "--h_q", type=int, default=24, help="query heads")

    parser.add_argument(
        "--heads_k", "--h_k", type=int, default=4, help="key/value heads"
    )

    parser.add_argument(
        "--headdim", "--d", type=int, default=256, help="head dimmension"
    )

    parser.add_argument(
        "--block_scaledim",
        "--bs",
        type=int,
        default=256,
        help="headdim per scale factor",
    )

    parser.add_argument(
        "--kv_splits",
        "--splits",
        type=int,
        default=0,
        help="threadblocks per sequence",
    )

    parser.add_argument(
        "--q_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
        help="query data type",
    )

    parser.add_argument(
        "--kv_dtype",
        type=cutlass.dtype,
        default=cutlass.Float8E4M3FN,
        help="key/value data type",
    )

    parser.add_argument(
        "--o_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
        help="output data type",
    )

    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
        help="accumulator/reduction data type",
    )

    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )

    parser.add_argument(
        "--scale_q",
        type=float,
        default=1.0,
        help="query scale factor",
    )

    parser.add_argument(
        "--scale_o",
        type=float,
        default=1.0,
        help="output scale factor",
    )

    parser.add_argument(
        "--scale_s",
        "--scale",
        type=float,
        default=0.0,
        help="score (Q*K) scale factor; if zero, defaults to 1/sqrt(D)",
    )

    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=0,
        help="Number of iterations for warmup",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of iterations after warmup",
    )

    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip reference check",
    )

    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    parser.add_argument(
        "--paged",
        action="store_true",
        default=False,
        help="Use paged KV (cp.async page-table gather) instead of dense TMA",
    )

    parser.add_argument(
        "--page_size",
        type=int,
        default=128,
        help="Paged KV page size in rows (must equal the kernel tile_s)",
    )

    kwargs = vars(parser.parse_args())
    run(**kwargs)

    print("PASS")
