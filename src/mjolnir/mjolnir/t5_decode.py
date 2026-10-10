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

"""Paged-KV fp8 decode attention for Jetson Thor (sm_110a), vLLM-shaped.

Inputs follow vLLM's paged KV cache:
  q          (num_seqs * q_len, h_q, d)          bf16, or e4m3 + q_descale
  k/v cache  (num_pages, page_size, h_k, d)      e4m3
  block_table(num_seqs, max_blocks)               int32
  seq_lens   (num_seqs,)                          int32 (includes the new tokens)
  q/k/v_descale (num_seqs, h_k)                   fp32 (per-tensor scales broadcast)
Outputs:
  o          (num_seqs * q_len, h_q, d)          bf16
  lse        (num_seqs, h_k, q_len * h_r)        fp32, natural-log LSE

Two compute paths, selected by the Q dtype:
  bf16 Q : fp8 KV -> bf16 via the cvt warpgroups -> TMEM A-operand
  e4m3 Q : fully-fp8, the mma reads K/V straight from the TMA'd SMEM ring

Query packing. The kernel works on N = q_len * h_r rows per KV head, packed
qi-major (n = qi * h_r + hr). For q_len == 1 that is exactly vLLM's layout
(no copy). For q_len > 1 (speculative decoding) the caller permutes q
(b, q_len, h_k, h_r, d) -> (b, h_k, q_len, h_r, d) and o back; see
pack_q / unpack_o below. Query token qi of a sequence of length L may
attend to keys 0 .. L - q_len + qi (causal tail).

Split-KV scheduling is occupancy-pinned (FlashInfer/FlashDecoding style):
a host-side plan (plan_split_kv) picks the smallest contiguous KV chunk such
that all chunks of all sequences fit in one wave of CTA slots, and emits one
work item (seq, first tile, end tile, split slot) per chunk. Each CTA streams
its chunk front to back. Long sequences get more chunks, short ones fewer,
so variable-length batches balance. A sequence with a single chunk writes its
final output directly; the reduction only merges sequences that were split.

Workspace (o/m/l partials) needs NO initialization: the reduction only reads
the splits a sequence actually used, and padding work items (begin == end)
exit at once, so a fixed-size padded work list is CUDA-graph friendly.
max_splits == 1 compiles the reduction out entirely.
"""

import argparse
import math
import os
from typing import Type, Tuple
from functools import partial

import torch

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
tile_s_rows = 128  # KV rows per tile (= GEMM1 M)

# SW128 smem atoms repeat every 1024 B and both TMA and the UMMA descriptor
# (base_offset = 0) assume the tile base is 1024-B aligned.
swizzle_align = 1024

# fp8 P is stored as e4m3(p * FP8_P_SCALE): p in [0, 1] then spans e4m3's
# normal range instead of flushing small probabilities to subnormals / 0.
# 1 / FP8_P_SCALE is folded into the output scale.
FP8_P_SCALE = 448.0

# KV cache formats (fp8-Q path). "fp8": e4m3 (default). Narrow formats read
# DIRECTLY by tcgen05 kind::f8f6f4 (TMA unpacks the packed gmem bytes into
# 1-byte SMEM containers): "e3m2"/"e2m3" (fp6), "e2m1" (fp4); scales are the
# usual per-tensor / per-head / per-(seq, head) descales. "nvfp4": packed
# e2m1 + one e4m3 scale per 16 head-dim elements, converted to e4m3 in SMEM
# by a convert warpgroup, then the normal fp8 MMA.
KV_FORMATS = ("fp8", "e3m2", "e2m3", "e2m1", "nvfp4")
KV_FORMAT_BITS = {"fp8": 8, "e3m2": 6, "e2m3": 6, "e2m1": 4, "nvfp4": 4}
# Harness (benchmark CLI) defaults: e2m3 KV on head-major pages, the fastest
# measured config at near-fp8 accuracy (rel L2 0.051 vs fp8 0.045). The kernel
# class and the vLLM runner keep fp8 (vLLM writes an fp8 cache).
HARNESS_KV_FORMAT_FP8Q = "e2m3"
HARNESS_KV_LAYOUT = "hnd"
# NVFP4 conversion computes e4m3(code * scale * 2^-3): with codes <= 6 and
# e4m3 scales <= 448 the product stays <= 336 (< 448, no saturation). The
# host folds the 2^3 back into the K/V descale.
NVFP4_SHIFT = 0.125
NVFP4_GROUP = 16

# Math helpers
log2_e = math.log2(math.e)  # change exponential base
fadd2 = cute.arch.add_packed_f32x2
fmul2 = cute.arch.mul_packed_f32x2
ffma2 = cute.arch.fma_packed_f32x2
exp2 = partial(cute.math.exp2, fastmath=True)


def _warp_fmax_shfl(value, mask=0xFFFFFFFF):
    """Warp-wide fmax via a butterfly shfl. ``warp_redux_sync(kind="fmax")``
    emits ``redux.f32``, which ptxas rejects on sm_110a (redux is int-only)."""
    for offset in (16, 8, 4, 2, 1):
        value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=offset, mask=mask), nan=True)
    return value


warp_fmax = _warp_fmax_shfl
smem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")


class MixedInputFusedMultiHeadAttentionDecode:
    def __init__(
        self,
        headdim,
        grouped_head_tile,  # packed query rows (q_len * h_r) per CTA tile
        convert_warpgroups=1,  # bf16 path: warpgroups striding over cvt stages; fp8: 0
        page_size=128,  # vLLM block size: 16, 32, 64 or 128
        q_len=1,  # query tokens per sequence (> 1 for speculative decoding)
        max_splits=16,  # partial-slot capacity per sequence (compile-time)
        ctas_per_sm=2,  # co-resident CTAs per SM; 2 is the Thor default (1/SM: 0.80x FI
        #                 at 128K-256K), each CTA gets half the smem/regs/TMEM
        with_reduction=True,  # False: no split in this plan -> no reduction launch
        max_kv_stages=0,  # KV ring depth; 0 = whole KV tiles (1 tile at 2/SM, 2 at 1/SM; fp6 at 2/SM d=256: 3)
        pdl=False,  # programmatic dependent launch for decode + reduction
        l2_prefetch=0,  # TMA-prefetch K/V tiles this many tiles ahead into L2 (0 = off)
        kv_evict_first=False,  # TMA K/V loads with an L2 EVICT_FIRST cache policy
        kv_format="fp8",  # KV cache format, see KV_FORMATS
    ):
        self.max_kv_stages = max_kv_stages
        self.pdl = bool(pdl)
        # Extra .launch() kwargs, built here in plain Python: the DSL rewrites
        # conditional expressions inside __call__ into regions that can't see
        # `self`. Empty when PDL is off (works with DSLs without use_pdl).
        self.launch_extra = {"use_pdl": True} if self.pdl else {}
        self.l2_prefetch = int(l2_prefetch)
        self.kv_evict_first = bool(kv_evict_first)
        assert kv_format in KV_FORMATS, f"kv_format must be one of {KV_FORMATS}"
        self.kv_format = kv_format
        if kv_format == "nvfp4":
            assert convert_warpgroups == 1, "nvfp4 KV needs exactly one convert warpgroup"
            assert self.l2_prefetch == 0, "l2_prefetch is not implemented for nvfp4 KV"
        assert 0 <= self.l2_prefetch <= 8, "l2_prefetch: 0..8 tiles"
        if self.l2_prefetch > 0:
            assert hasattr(cute, "prefetch"), (
                "l2_prefetch needs cute.prefetch (TMA prefetch) — newer nvidia-cutlass-dsl"
            )
        if self.pdl:
            assert hasattr(cute.arch, "griddepcontrol_wait"), (
                "pdl needs cute.arch.griddepcontrol_* — newer nvidia-cutlass-dsl"
            )
        self.with_reduction = with_reduction
        self.headdim = headdim
        self.grouped_head_tile = grouped_head_tile
        self.convert_warpgroups = convert_warpgroups
        self.page_size = page_size
        self.q_len = q_len
        self.max_splits = max_splits
        self.ctas_per_sm = ctas_per_sm

        assert headdim % 128 == 0, "headdim must be a multiple of 128"
        assert page_size in (16, 32, 64, 128), "page_size must be 16, 32, 64 or 128"
        # Ld32x32b / St32x32b repetition (= tile width) must be a power of two.
        assert grouped_head_tile in (8, 16, 32), "grouped_head_tile must be 8, 16 or 32"
        assert convert_warpgroups >= 0
        assert q_len >= 1 and max_splits >= 1
        assert ctas_per_sm in (1, 2)

        warpgroup_id = 0

        self.softmax_warpgroup_id = warpgroup_id
        warpgroup_id += 1

        # Fully-fp8 launches with convert_warpgroups == 0: no cvt warpgroup
        # exists at all (begin == end, so the cvt dispatch never matches).
        self.cvt_wg_begin = warpgroup_id
        self.cvt_wg_end = warpgroup_id + convert_warpgroups
        warpgroup_id += convert_warpgroups

        # 2 MMA warps (less register pressure, hides mma switching latency)
        # + 2 TMA warps (KV ring / Q).
        self.mma_kq_warp_id = warpgroup_id * warpgroup_warps + 0
        self.mma_vp_warp_id = warpgroup_id * warpgroup_warps + 1
        self.tma_kv_warp_id = warpgroup_id * warpgroup_warps + 2
        self.tma_q_warp_id = warpgroup_id * warpgroup_warps + 3
        warpgroup_id += 1

        self.threads_per_cta = warpgroup_id * warpgroup_threads

        # Register budget. The launch uses min_blocks_per_mp = ctas_per_sm, so
        # ptxas caps every thread at R = 64K / (threads * ctas) rounded DOWN to
        # a multiple of 8 (bf16 at 2 CTAs/SM: 384 threads -> R = 80, not 85).
        # The CTA's register pool is R * threads; setmaxnreg must stay inside
        # it, a .dec may not exceed R and an .inc may not go below R (either
        # is an illegal instruction). Budget in per-warpgroup-thread units.
        self.use_reg_reconfig = grouped_head_tile > 16
        num_warpgroups = self.threads_per_cta // warpgroup_threads
        launch_regs = min(255, 64 * 1024 // (self.threads_per_cta * ctas_per_sm)) // 8 * 8
        pool_regs = launch_regs * num_warpgroups
        # 2 CTAs/SM with a cvt warpgroup is the tight case (pool 240): give
        # the MMA/TMA warps less so softmax keeps >= 96.
        self.mma_tma_regs = min(launch_regs, 56 if (ctas_per_sm == 2 and convert_warpgroups > 0) else 72)
        self.cvt_regs = min(launch_regs, 112 if ctas_per_sm == 1 else 88)
        self.softmax_regs = pool_regs - self.mma_tma_regs - self.cvt_regs * convert_warpgroups
        self.softmax_regs = min(256, self.softmax_regs) // 8 * 8  # setmaxnreg: 24..256, x8
        if self.use_reg_reconfig:
            assert self.softmax_regs >= max(96, launch_regs), (
                f"softmax warpgroup gets {self.softmax_regs} regs (launch {launch_regs}, "
                f"pool {pool_regs}); use ctas_per_sm=1 or fewer cvt warpgroups"
            )
            assert (self.mma_tma_regs + self.cvt_regs * convert_warpgroups + self.softmax_regs) <= pool_regs

        self.sp_stages = 2
        self.o_stages = 1

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
        tile): 3 stages — measured 4-5% faster than the whole-tile 4 at
        64K/256K in two Thor A/Bs (fp8 is the opposite: 4 beat 3)."""
        if self.max_kv_stages > 0:
            return self.max_kv_stages, False
        if self.kv_format in ("e3m2", "e2m3") and self.ctas_per_sm == 2 and stages_per_tile == 4:
            return 3, False
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
        # Exception, fp6 KV (e3m2/e2m3) at 2/SM, d=256: depth 3 beat 4 by 4-5%
        # at 64K/256K in two A/Bs (6 was worst) — see _default_kv_stages.
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

        print(f"\tcvt stages: {self.cvt_stages}")

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

        print(f"\tkv stages: {self.kv_stages}, kv format: {self.kv_format}, packed stages: {self.packed_stages}")
        print(f"\tpdl: {self.pdl}, l2 prefetch: {self.l2_prefetch} tiles ahead, kv evict_first: {self.kv_evict_first}")

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

    @cute.kernel
    def decode(
        self,
        # MMA
        blk_tile_shd: cute.Tile,
        mma_tile_mnk: cute.Tile,
        tiled_mma_kq: cute.TiledMma,
        tiled_mma_vp: cute.TiledMma,
        q_dtype: Type[cutlass.Numeric],
        kv_smem_dtype: Type[cutlass.Numeric],
        # Q
        smem_layout_q: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        # K
        smem_layout_k: cute.ComposedLayout,
        tma_atom_k: cute.CopyAtom,
        mK: cute.Tensor,
        # V
        smem_layout_v: cute.ComposedLayout,
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        # O tile shape
        smem_layout_o: cute.ComposedLayout,
        # Paging / varlen / scales
        mBlockTable: cute.Tensor,
        mSeqLens: cute.Tensor,
        mWork: cute.Tensor,
        mNSplits: cute.Tensor,
        mQD: cute.Tensor,
        mKD: cute.Tensor,
        mVD: cute.Tensor,
        mKS: cute.Tensor,  # nvfp4 K scales (PS, d/16, (h_k, pages)), else unused
        mVS: cute.Tensor,
        # Outputs
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mO_partial: cute.Tensor,
        mM_partial: cute.Tensor,
        mL_partial: cute.Tensor,
        # Scalars
        h_k: cutlass.Int32,
        n_q: cutlass.Int32,
        num_pages: cutlass.Int32,
        scale_qs: cutlass.Float32,
        scale_qs_log2_e: cutlass.Float32,
        scale_o: cutlass.Float32,
    ):
        # Read special registers
        work_idx, coord_hr, h_k_idx = cute.arch.block_idx()
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
        acc_dtype = cutlass.Float32

        # Compile-time selectors (plain Python values)
        FP8 = cutlass.const_expr(mma_dtype.width == 8)
        NVFP4 = cutlass.const_expr(self.kv_format == "nvfp4")
        SINGLE_SPLIT = cutlass.const_expr(self.max_splits == 1)
        PS = self.page_size
        p_scale = FP8_P_SCALE if FP8 else 1.0

        # Shapes for MMA tile indexing
        blk_tile_s, blk_tile_h, blk_tile_d = blk_tile_shd
        mma_tile_m, mma_tile_n, mma_tile_k = mma_tile_mnk
        tiles_dm, tiles_sk = cute.ceil_div((blk_tile_d, blk_tile_s), (mma_tile_m, mma_tile_k))
        tiles_dk, tiles_sm = cute.ceil_div((blk_tile_d, blk_tile_s), (mma_tile_k, mma_tile_m))
        assert tiles_sm == 1
        assert tiles_sk == 1
        pages_per_tile = blk_tile_s // PS
        L2P = self.l2_prefetch  # compile-time

        # PDL: this grid may start while the previous kernel in the stream
        # (QKV/rope, KV-cache write, Q quantization, an earlier layer's
        # reduction reading the partial buffers) is still finishing. Wait for
        # it before the first global read/write; then let our dependent (the
        # reduction) launch early — it waits for this grid in turn, so its
        # launch latency and prologue hide behind our tail.
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        # Work item from the plan: this CTA streams the CONTIGUOUS tile range
        # [tile_begin, tile_end) of one sequence and writes partial slot
        # split_slot (or the final output if the sequence has one chunk).
        b_idx = mWork[work_idx, 0]
        tile_begin = mWork[work_idx, 1]
        tile_end = mWork[work_idx, 2]
        split_slot = mWork[work_idx, 3]
        coord_hb = h_k_idx + h_k * b_idx  # (h_k, b) L-mode index
        seq_len = mSeqLens[b_idx]
        n_splits_seq = mNSplits[b_idx]
        iters_s = tile_end - tile_begin
        exit_early = iters_s <= 0  # padding work item
        prefetch_iters = self.sp_stages - 1
        if iters_s < prefetch_iters:
            prefetch_iters = iters_s

        n_base = coord_hr * blk_tile_h
        lane_store_max = mma_tile_n == warp_threads or lane_idx < mma_tile_n

        # Smem alloc helper
        svector_align = 16
        smem = utils.SmemAllocator()

        ##############################
        # Prefetch TMA descriptor
        ##############################
        if warp_idx == init_warp and not exit_early:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
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
        q_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.q_stages * 2)
        pk_pipeline_ptr = smem.allocate_array(cutlass.Int64, max(1, self.packed_stages) * 2)
        kv_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.kv_stages * 2)
        cvt_pipeline_ptr = smem.allocate_array(cutlass.Int64, max(1, self.cvt_stages * 2))
        s_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.sp_stages * 2)
        p_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.sp_stages * 2)
        o_pipeline_ptr = smem.allocate_array(cutlass.Int64, self.o_stages * 2)

        # Named barriers
        softmax_nbar = pipeline.NamedBarrier(barrier_id=1, num_threads=warpgroup_threads)
        mma_kq_nbar = pipeline.NamedBarrier(barrier_id=2, num_threads=64)
        mma_vp_nbar = pipeline.NamedBarrier(barrier_id=3, num_threads=64)

        # Thread cooperatives
        elect_one_cooperative = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        warpgroup_cooperative = pipeline.CooperativeGroup(pipeline.Agent.Thread, warpgroup_threads)
        mma_group = elect_one_cooperative
        tma_group = elect_one_cooperative
        cvt_group = warpgroup_cooperative
        softmax_group = warpgroup_cooperative

        q_producer, q_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.q_stages,
            producer_group=tma_group,
            consumer_group=softmax_group,
            tx_count=cute.size_in_bytes(q_dtype, cute.select(smem_layout_q, mma_modes)),
            barrier_storage=q_pipeline_ptr,
            tidx=mcast_coord,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()

        # KV ring (always TMA; a stage is filled by 1 box or 128/PS page boxes,
        # tx_count is the full stage either way — OOB boxes are zero-filled
        # and still credit their bytes).
        #  fp8 : UMMA consumer (PipelineTmaUmma): released by tcgen05.commit
        #        once the mma that read the stage completes; each stage is read
        #        by exactly one mma warp -> consumer count 1.
        #  bf16: cvt consumer (PipelineTmaAsync). The cvt warpgroups STRIDE
        #        over stages, so each stage is released by ONE warpgroup.
        # TMA credits the barrier with the bytes it READS from gmem: for the
        # narrow formats that is the packed size (fp6: 12 KB, fp4: 8 KB per
        # 128x128 stage), not the unpacked 1-B/elem SMEM size — counting
        # the SMEM bytes would leave the stage barrier waiting forever.
        kv_tx_dtype = self._kv_tx_dtype(kv_smem_dtype)
        kv_tx_count = cute.size_in_bytes(kv_tx_dtype, cute.select(smem_layout_k, mma_modes))
        pk_producer, pk_consumer = None, None
        if cutlass.const_expr(NVFP4):
            # TMA -> packed e2m1 ring -> convert warpgroup (one K or V tile
            # per stage) ...
            pk_producer, pk_consumer = pipeline.PipelineTmaAsync.create(
                num_stages=self.packed_stages,
                producer_group=tma_group,
                consumer_group=cvt_group,
                tx_count=blk_tile_s * blk_tile_d // 2,  # packed bytes of a tile
                barrier_storage=pk_pipeline_ptr,
                tidx=mcast_coord,
                cta_layout_vmnk=mcast_layout,
                defer_sync=True,
            ).make_participants()
            # ... -> e4m3 ring (the normal fp8 A-operand ring) -> MMA
            kv_producer, kv_consumer = pipeline.PipelineAsyncUmma.create(
                num_stages=self.kv_stages,
                producer_group=cvt_group,
                consumer_group=mma_group,
                barrier_storage=kv_pipeline_ptr,
                defer_sync=True,
            ).make_participants()
            cvt_producer, cvt_consumer = None, None
            a_consumer = kv_consumer
        elif cutlass.const_expr(FP8):
            kv_producer, kv_consumer = pipeline.PipelineTmaUmma.create(
                num_stages=self.kv_stages,
                producer_group=tma_group,
                consumer_group=mma_group,
                tx_count=kv_tx_count,
                barrier_storage=kv_pipeline_ptr,
                cta_layout_vmnk=mcast_layout,
                defer_sync=True,
            ).make_participants()
            cvt_producer, cvt_consumer = None, None
            a_consumer = kv_consumer
        else:
            kv_producer, kv_consumer = pipeline.PipelineTmaAsync.create(
                num_stages=self.kv_stages,
                producer_group=tma_group,
                consumer_group=cvt_group,
                tx_count=kv_tx_count,
                barrier_storage=kv_pipeline_ptr,
                tidx=mcast_coord,
                cta_layout_vmnk=mcast_layout,
                defer_sync=True,
            ).make_participants()
            cvt_producer, cvt_consumer = pipeline.PipelineAsyncUmma.create(
                num_stages=max(1, self.cvt_stages),
                producer_group=cvt_group,
                consumer_group=mma_group,
                barrier_storage=cvt_pipeline_ptr,
                defer_sync=True,
            ).make_participants()
            a_consumer = cvt_consumer

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
        thrblk_mma_kq = tiled_mma_kq.get_slice(0)
        thrblk_mma_vp = tiled_mma_vp.get_slice(0)

        # M - colmax
        sM_layout = cute.make_layout(shape=(mma_tile_m, mma_tile_n), stride=(0, 1))
        sM = smem.allocate_tensor(acc_dtype, sM_layout, svector_align)
        tCsM = thrblk_mma_kq.partition_C(sM)

        # L - colsum
        sL_layout = cute.make_layout(shape=(mma_tile_m, mma_tile_n, warpgroup_warps), stride=(0, 1, mma_tile_n))
        sL = smem.allocate_tensor(acc_dtype, sL_layout, svector_align)
        tCsL = thrblk_mma_kq.partition_C(sL)

        # Q (swizzled: 1024-B aligned)
        tBsQ = smem.allocate_tensor(
            q_dtype, smem_layout_q.outer, swizzle_align, smem_layout_q.inner
        )  # (MMA, #MMA_N, #MMA_K, q_stages)

        # K (swizzled KV ring: 1024-B aligned)
        tAsK = smem.allocate_tensor(
            kv_smem_dtype, smem_layout_k.outer, swizzle_align, smem_layout_k.inner
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        if cutlass.const_expr(FP8):
            tAtK_cvt = thrblk_mma_kq.make_fragment_A(tAsK)
        else:
            tAtK_cvt_shape = tiled_mma_kq.partition_shape_A((mma_tile_m, mma_tile_k, self.cvt_stages))
            tAtK_cvt = thrblk_mma_kq.make_fragment_A(tAtK_cvt_shape)

        # V (shares the KV ring)
        tAsV_iterator = cute.recast_ptr(tAsK.iterator, smem_layout_v.inner, dtype=kv_smem_dtype)
        tAsV = cute.make_tensor(tAsV_iterator, smem_layout_v.outer)  # (MMA, #MMA_M, #MMA_K, kv_stages)
        if cutlass.const_expr(NVFP4):
            # packed e2m1 tiles, row-major (token, d) per stage
            sKVp = smem.allocate_tensor(
                cutlass.Float4E2M1FN,
                cute.make_layout(
                    (blk_tile_s, blk_tile_d, self.packed_stages),
                    stride=(blk_tile_d, 1, blk_tile_s * blk_tile_d),
                ),
                128,
            )
        if cutlass.const_expr(FP8):
            tAtV_cvt = thrblk_mma_vp.make_fragment_A(tAsV)
        else:
            tAtV_cvt_shape = tiled_mma_vp.partition_shape_A((mma_tile_m, mma_tile_k, self.cvt_stages))
            tAtV_cvt = thrblk_mma_vp.make_fragment_A(tAtV_cvt_shape)

        # S
        tCtS_shape = tiled_mma_kq.partition_shape_C((mma_tile_m, mma_tile_n, self.sp_stages))
        tCtS = thrblk_mma_kq.make_fragment_C(tCtS_shape)

        # P - Treat MN C tile of BMM0 as NM B tile of BMM1
        mma_tile_nm = (None, mma_tile_n, mma_tile_m)
        tBsP_nm_layout = sm100_utils.make_smem_layout_b(tiled_mma_vp, mma_tile_nm, mma_dtype, self.sp_stages)
        tBsP_nm = smem.allocate_tensor(mma_dtype, tBsP_nm_layout.outer, swizzle_align, tBsP_nm_layout.inner)
        tBsP_nk_tile = thrblk_mma_vp.partition_shape_B((mma_tile_n, mma_tile_k))
        tBsP_nk = cute.local_tile(tBsP_nm, tBsP_nk_tile, (0, 0, None, None))
        tCsP_tile = cute.make_ordered_layout(tCtS_shape, order=((2, 0), 3, 1, 4))
        tCsP = cute.composition(tBsP_nm, tCsP_tile)

        # O tile view (shape source only — aliases Q smem, never written)
        sO_iterator = cute.recast_ptr(tBsQ.iterator, smem_layout_o.inner, dtype=acc_dtype)
        sO_mma = cute.make_tensor(sO_iterator, smem_layout_o.outer)
        tCsO = thrblk_mma_vp.partition_C(sO_mma)
        tCtO = thrblk_mma_vp.make_fragment_C(tCsO.shape)

        # TMEM tensors
        tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Int32, 16, tmem_ptr_smem_ptr)
        tmem_offset = 0

        if cutlass.const_expr(not FP8):
            tAtK_cvt = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + tmem_offset, dtype=mma_dtype),
                tAtK_cvt.layout,
            )
            tAtV_cvt = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + tmem_offset, dtype=mma_dtype),
                tAtV_cvt.layout,
            )
            tmem_offset += tcgen05.find_tmem_tensor_col_offset(tAtK_cvt)

        tCtS = cute.make_tensor(cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtS.layout)
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS)

        tCtO = cute.make_tensor(cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtO.layout)
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtO)

        print(f"\t{tmem_offset} tmem cols used, {self.tmem_alloc_cols} tmem cols allocated")
        assert tmem_offset <= self.tmem_alloc_cols

        ##############################
        # Exit early (this split has no tiles for this sequence)
        ##############################
        if exit_early:
            noop = None  # early return not supported # noqa: F841

        ##############################
        # TMA KV Dispatch
        ##############################
        elif warp_idx == self.tma_kv_warp_id:
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            # KV is streamed exactly once: EVICT_FIRST keeps it from pushing
            # Q / block table / partials out of L2 (CUTLASS CacheHintSm90
            # EVICT_FIRST policy word).
            kv_hint = {}
            if cutlass.const_expr(self.kv_evict_first):
                kv_hint = {"cache_policy": cutlass.Int64(0x12F0000000000000)}

            # Walk this CTA's contiguous chunk: step i loads K of tile
            # tile_begin + i and V of tile tile_begin + i - prefetch_iters
            # (K runs prefetch_iters tiles ahead of V, same order as the
            # consumers drain the ring).
            if cutlass.const_expr(NVFP4):
                # Packed e2m1 K / V tiles (token-major, whole head dim) into
                # the packed ring, one box per page; the convert warpgroup
                # turns them into e4m3 stages. Same K-ahead-of-V order.
                sKVp_box = cute.flat_divide(sKVp, (PS, blk_tile_d))  # (PS, D, pages_per_tile, 1, packed_stages)
                gKp_box = cute.flat_divide(mK, (PS, blk_tile_d))  # (PS, D, 1, 1, L)
                gVp_box = cute.flat_divide(mV, (PS, blk_tile_d))
                tGSsKp, tGSgKp = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_k,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(sKVp_box, 0, 2),
                    gmem_tensor=cute.group_modes(gKp_box, 0, 2),
                )  # (TMA, pages_per_tile, 1, stages), (TMA, 1, 1, L)
                tGSsVp, tGSgVp = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_v,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(sKVp_box, 0, 2),
                    gmem_tensor=cute.group_modes(gVp_box, 0, 2),
                )
                for s in cutlass.range(iters_s + prefetch_iters):
                    if s < iters_s:
                        t = tile_begin + s
                        pk_handle = pk_producer.acquire_and_advance()
                        for j in cutlass.range_constexpr(pages_per_tile):
                            blk = t * pages_per_tile + j
                            page = num_pages  # past the end: OOB, zero-filled
                            if blk * PS < seq_len:
                                page = mBlockTable[b_idx, blk]
                            l_idx = h_k_idx + h_k * page
                            cute.copy(
                                tma_atom_k,
                                tGSgKp[None, 0, 0, l_idx],
                                tGSsKp[None, j, 0, pk_handle.index],
                                tma_bar_ptr=pk_handle.barrier,
                                **kv_hint,
                            )
                    if s >= prefetch_iters:
                        t = tile_begin + s - prefetch_iters
                        pk_handle = pk_producer.acquire_and_advance()
                        for j in cutlass.range_constexpr(pages_per_tile):
                            blk = t * pages_per_tile + j
                            page = num_pages
                            if blk * PS < seq_len:
                                page = mBlockTable[b_idx, blk]
                            l_idx = h_k_idx + h_k * page
                            cute.copy(
                                tma_atom_v,
                                tGSgVp[None, 0, 0, l_idx],
                                tGSsVp[None, j, 0, pk_handle.index],
                                tma_bar_ptr=pk_handle.barrier,
                                **kv_hint,
                            )
            elif cutlass.const_expr(PS == blk_tile_s):
                # One page per tile. L-mode = (h_k, num_pages).
                gK = cute.local_tile(mK, tiler=(blk_tile_s, blk_tile_d), coord=(0, 0, None))  # (TILE_S, TILE_D, L)
                gV = cute.local_tile(mV, tiler=(blk_tile_d, blk_tile_s), coord=(0, 0, None))  # (TILE_D, TILE_S, L)
                gK_mma = cute.flat_divide(
                    gK, (mma_tile_m, mma_tile_k)
                )  # (MMA_TILE_M, MMA_TILE_K, #TILE_SM, #TILE_DK, L)
                gV_mma = cute.flat_divide(
                    gV, (mma_tile_m, mma_tile_k)
                )  # (MMA_TILE_M, MMA_TILE_K, #TILE_DM, #TILE_SK, L)
                tAgK = thrblk_mma_kq.partition_A(gK_mma)
                tAgV = thrblk_mma_vp.partition_A(gV_mma)
                tGSsK, tGSgK = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_k,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(tAsK, 0, 3),
                    gmem_tensor=cute.group_modes(tAgK, 0, 3),
                )  # (TMA, stages), (TMA, #TILE_SM, #TILE_DK, L)
                tGSsV, tGSgV = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_v,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(tAsV, 0, 3),
                    gmem_tensor=cute.group_modes(tAgV, 0, 3),
                )  # (TMA, stages), (TMA, #TILE_DM, #TILE_SK, L)

                for s in cutlass.range(iters_s + prefetch_iters):
                    # L2 prefetch: K+V of tile s + L2P (and, on the first step,
                    # s + 1 .. s + L2P - 1) so DRAM streams ahead of the ring.
                    if cutlass.const_expr(L2P > 0):
                        for pp in cutlass.range_constexpr(1, L2P + 1):
                            if s + pp < iters_s:
                                if cutlass.const_expr(pp == L2P):
                                    page_p = mBlockTable[b_idx, tile_begin + s + pp]
                                    l_p = h_k_idx + h_k * page_p
                                    for dk in cutlass.range_constexpr(tiles_dk):
                                        cute.prefetch(tma_atom_k, tGSgK[None, 0, dk, l_p])
                                    for dm in cutlass.range_constexpr(tiles_dm):
                                        cute.prefetch(tma_atom_v, tGSgV[None, dm, 0, l_p])
                                else:
                                    if s == 0:
                                        page_p = mBlockTable[b_idx, tile_begin + pp]
                                        l_p = h_k_idx + h_k * page_p
                                        for dk in cutlass.range_constexpr(tiles_dk):
                                            cute.prefetch(tma_atom_k, tGSgK[None, 0, dk, l_p])
                                        for dm in cutlass.range_constexpr(tiles_dm):
                                            cute.prefetch(tma_atom_v, tGSgV[None, dm, 0, l_p])
                    # Load K (tile tile_begin + s)
                    if s < iters_s:
                        page = mBlockTable[b_idx, tile_begin + s]
                        l_idx = h_k_idx + h_k * page
                        for dk in cutlass.range_constexpr(tiles_dk):
                            k_handle = kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_k,
                                tGSgK[None, 0, dk, l_idx],
                                tGSsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                                **kv_hint,
                            )
                    # Load V (tile tile_begin + s - prefetch_iters)
                    if s >= prefetch_iters:
                        page = mBlockTable[b_idx, tile_begin + s - prefetch_iters]
                        l_idx = h_k_idx + h_k * page
                        for dm in cutlass.range_constexpr(tiles_dm):
                            v_handle = kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v,
                                tGSgV[None, dm, 0, l_idx],
                                tGSsV[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                                **kv_hint,
                            )
            else:
                # PS < 128: pages_per_tile boxes per stage. Logical views of
                # the ring: K (s, d, stage), V (d, s, stage).
                ring_2d = cute.make_layout(
                    (mma_tile_m, mma_tile_k, self.kv_stages),
                    stride=(1, mma_tile_m, mma_tile_m * mma_tile_k),
                )
                sK_sd = cute.composition(tAsK, ring_2d)
                sV_ds = cute.composition(tAsV, ring_2d)
                sK_box = cute.flat_divide(sK_sd, (PS, mma_tile_k))  # (PS, K, #J, 1, stages)
                sV_box = cute.flat_divide(sV_ds, (mma_tile_m, PS))  # (M, PS, 1, #J, stages)
                gK_box = cute.flat_divide(mK, (PS, mma_tile_k))  # (PS, K, 1, #DK, L)
                gV_box = cute.flat_divide(mV, (mma_tile_m, PS))  # (M, PS, #DM, 1, L)
                tGSsK, tGSgK = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_k,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(sK_box, 0, 2),
                    gmem_tensor=cute.group_modes(gK_box, 0, 2),
                )  # (TMA, #J, 1, stages), (TMA, 1, #DK, L)
                tGSsV, tGSgV = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_v,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(sV_box, 0, 2),
                    gmem_tensor=cute.group_modes(gV_box, 0, 2),
                )  # (TMA, 1, #J, stages), (TMA, #DM, 1, L)

                for s in cutlass.range(iters_s + prefetch_iters):
                    # L2 prefetch (see the PS == 128 path); pages past the
                    # sequence end are skipped, not zero-filled.
                    if cutlass.const_expr(L2P > 0):
                        for pp in cutlass.range_constexpr(1, L2P + 1):
                            if s + pp < iters_s:
                                do_p = s == 0
                                if cutlass.const_expr(pp == L2P):
                                    do_p = s >= 0
                                if do_p:
                                    t_p = tile_begin + s + pp
                                    for j in cutlass.range_constexpr(pages_per_tile):
                                        blk_p = t_p * pages_per_tile + j
                                        if blk_p * PS < seq_len:
                                            page_p = mBlockTable[b_idx, blk_p]
                                            l_p = h_k_idx + h_k * page_p
                                            for dk in cutlass.range_constexpr(tiles_dk):
                                                cute.prefetch(tma_atom_k, tGSgK[None, 0, dk, l_p])
                                            for dm in cutlass.range_constexpr(tiles_dm):
                                                cute.prefetch(tma_atom_v, tGSgV[None, dm, 0, l_p])
                    # Load K (tile tile_begin + s)
                    if s < iters_s:
                        t = tile_begin + s
                        for dk in cutlass.range_constexpr(tiles_dk):
                            k_handle = kv_producer.acquire_and_advance()
                            for j in cutlass.range_constexpr(pages_per_tile):
                                # Pages past the sequence end get an
                                # out-of-bounds page index: TMA zero-fills
                                # them and never touches the block table
                                # padding (which may hold garbage).
                                blk = t * pages_per_tile + j
                                page = num_pages
                                if blk * PS < seq_len:
                                    page = mBlockTable[b_idx, blk]
                                l_idx = h_k_idx + h_k * page
                                cute.copy(
                                    tma_atom_k,
                                    tGSgK[None, 0, dk, l_idx],
                                    tGSsK[None, j, 0, k_handle.index],
                                    tma_bar_ptr=k_handle.barrier,
                                    **kv_hint,
                                )
                    # Load V (tile tile_begin + s - prefetch_iters)
                    if s >= prefetch_iters:
                        t = tile_begin + s - prefetch_iters
                        for dm in cutlass.range_constexpr(tiles_dm):
                            v_handle = kv_producer.acquire_and_advance()
                            for j in cutlass.range_constexpr(pages_per_tile):
                                blk = t * pages_per_tile + j
                                page = num_pages
                                if blk * PS < seq_len:
                                    page = mBlockTable[b_idx, blk]
                                l_idx = h_k_idx + h_k * page
                                cute.copy(
                                    tma_atom_v,
                                    tGSgV[None, dm, 0, l_idx],
                                    tGSsV[None, 0, j, v_handle.index],
                                    tma_bar_ptr=v_handle.barrier,
                                    **kv_hint,
                                )

        ##############################
        # TMA Q Dispatch
        ##############################
        elif warp_idx == self.tma_q_warp_id:
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            gQ = cute.local_tile(mQ, tiler=(blk_tile_h, blk_tile_d), coord=(coord_hr, 0, coord_hb))  # (TILE_H, TILE_D)
            gQ_mma = cute.flat_divide(gQ, (mma_tile_n, mma_tile_k))  # (MMA_TILE_N, MMA_TILE_K, #TILE_HN, #TILE_DK)
            tBgQ = thrblk_mma_kq.partition_B(gQ_mma)
            tGSsQ, tGSgQ = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tBsQ, 0, 3),
                gmem_tensor=cute.group_modes(tBgQ, 0, 3),
            )
            for dk in cutlass.range_constexpr(tiles_dk):
                q_handle = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tGSgQ[None, 0, dk],  # q_stages == tiles_dk
                    tGSsQ[None, dk],
                    tma_bar_ptr=q_handle.barrier,
                )

        ##############################
        # Convert Dispatch (bf16 only; empty range on the fp8 launch)
        ##############################
        elif warpgroup_idx >= self.cvt_wg_begin and warpgroup_idx < self.cvt_wg_end:
            if cutlass.const_expr(not FP8):
                if cutlass.const_expr(self.use_reg_reconfig):
                    cute.arch.setmaxregister_decrease(self.cvt_regs)

                cvt_type = cutlass.Float32  # e4m3 -> f32 -> bf16

                # Multiple warpgroups stride over the KV / cvt stages
                convert_phase = 0
                if cutlass.const_expr(self.convert_warpgroups > 1):
                    assert tiles_dk % self.convert_warpgroups == 0
                    assert (tiles_dm * tiles_sk) % self.convert_warpgroups == 0
                    convert_phase = (warpgroup_idx - self.cvt_wg_begin) % self.convert_warpgroups
                    for _ in cutlass.range(convert_phase):
                        kv_consumer.advance()
                        cvt_producer.advance()

                # K: smem (ldmatrix) -> rf -> cvt -> TMEM
                mma_k_bits = mma_tile_k * mma_dtype.width
                tmem_store_atom_k = cute.make_copy_atom(
                    tcgen05.St16x256bOp(tcgen05.Repetition(mma_k_bits // 256)),
                    mma_dtype,
                )
                smem_load_atom_k = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x16x8bOp(num_matrices=4),
                    kv_smem_dtype,
                )
                tmem_store_k = tcgen05.make_tmem_copy(tmem_store_atom_k, tAtK_cvt[mma_dice + (0,)])
                thr_store_k = tmem_store_k.get_slice(warpgroup_tidx)
                tKrK_cvt_shape = thr_store_k.partition_S(tAtK_cvt).shape[:-1]
                tKtK_cvt = thr_store_k.partition_D(tAtK_cvt)
                smem_load_k = cute.make_tiled_copy_S(smem_load_atom_k, tmem_store_k)
                thr_load_k = smem_load_k.get_slice(warpgroup_tidx)
                tKsK = thr_load_k.partition_S(tAsK)
                tKrK_shape = thr_load_k.partition_D(tAsK).shape[:-1]

                # V
                tmem_store_atom_v = cute.make_copy_atom(
                    tcgen05.St16x256bOp(tcgen05.Repetition(mma_k_bits // 256)),
                    mma_dtype,
                )
                smem_load_atom_v = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix16x16x8bOp(transpose=True, num_matrices=2),
                    kv_smem_dtype,
                )
                tmem_store_v = tcgen05.make_tmem_copy(tmem_store_atom_v, tAtV_cvt[mma_dice + (0,)])
                thr_store_v = tmem_store_v.get_slice(warpgroup_tidx)
                tVrV_cvt_shape = thr_store_v.partition_S(tAtV_cvt).shape[:-1]
                tVtV_cvt = thr_store_v.partition_D(tAtV_cvt)
                smem_load_v = cute.make_tiled_copy_S(smem_load_atom_v, tmem_store_v)
                thr_load_v = smem_load_v.get_slice(warpgroup_tidx)
                tVsV = thr_load_v.partition_S(tAsV)
                tVrV_shape = thr_load_v.partition_D(tAsV).shape[:-1]

                for s in cutlass.range(prefetch_iters + iters_s):
                    if s < iters_s:
                        for dk in cutlass.range(tiles_dk // self.convert_warpgroups, unroll=2):
                            tKrK = cute.make_rmem_tensor(tKrK_shape, kv_smem_dtype)
                            tKrK_cvt = cute.make_rmem_tensor(tKrK_cvt_shape, mma_dtype)

                            kv_handle = kv_consumer.wait_and_advance()
                            cute.copy(thr_load_k, tKsK[cpy_dice + (kv_handle.index,)], tKrK)
                            cute.arch.fence_view_async_shared()
                            kv_handle.release()

                            tKrK_ssa = tKrK.load().to(cvt_type).to(mma_dtype)
                            tKrK_cvt.store(tKrK_ssa.reshape(tKrK_cvt_shape))

                            cvt_handle = cvt_producer.acquire_and_advance()
                            cute.copy(
                                thr_store_k,
                                tKrK_cvt,
                                tKtK_cvt[cpy_dice + (cvt_handle.index,)],
                            )
                            cute.arch.fence_view_async_tmem_store()
                            cvt_handle.commit()

                            for _ in cutlass.range_constexpr(self.convert_warpgroups - 1):
                                kv_consumer.advance()
                                cvt_producer.advance()

                    if s >= prefetch_iters:
                        for dmsk in cutlass.range(tiles_dm * tiles_sk // self.convert_warpgroups, unroll=2):
                            tVrV = cute.make_rmem_tensor(tVrV_shape, kv_smem_dtype)
                            tVrV_cvt = cute.make_rmem_tensor(tVrV_cvt_shape, mma_dtype)

                            kv_handle = kv_consumer.wait_and_advance()
                            cute.copy(thr_load_v, tVsV[cpy_dice + (kv_handle.index,)], tVrV)
                            cute.arch.fence_view_async_shared()
                            kv_handle.release()

                            tVrV_ssa = tVrV.load().to(cvt_type).to(mma_dtype)
                            tVrV_cvt.store(tVrV_ssa.reshape(tVrV_cvt_shape))

                            cvt_handle = cvt_producer.acquire_and_advance()
                            cute.copy(
                                thr_store_v,
                                tVrV_cvt,
                                tVtV_cvt[cpy_dice + (cvt_handle.index,)],
                            )
                            cute.arch.fence_view_async_tmem_store()
                            cvt_handle.commit()

                            for _ in cutlass.range_constexpr(self.convert_warpgroups - 1):
                                kv_consumer.advance()
                                cvt_producer.advance()

            if cutlass.const_expr(NVFP4):
                # nvfp4 -> e4m3: packed (token, d) tile in SMEM, e4m3 scale
                # per 16 d elements from gmem; writes the e4m3 A-operand
                # stages the fp8 MMA reads (K: (s, d) K-major, V: (d, s)
                # MN-major — both addressed through their logical views).
                # Thread t owns d-chunk c = t % 8 (32 elements = 2 scale
                # groups) of rows t // 8 + 16 * i: a warp reads 8 consecutive
                # 16-B chunks of 4 rows (conflict-free) and the chunk lands
                # in one half-head-dim e4m3 stage (c // 4).
                if cutlass.const_expr(self.use_reg_reconfig):
                    cute.arch.setmaxregister_decrease(self.cvt_regs)
                CH = 32  # elements per chunk
                n_chunks = blk_tile_d // CH  # 8 at d = 256
                rows_per_pass = warpgroup_threads // n_chunks  # 16
                cvt_c = warpgroup_tidx % n_chunks
                cvt_r0 = warpgroup_tidx // n_chunks
                half = cvt_c // (mma_tile_k // CH)  # which 128-d e4m3 stage
                cvt_cc = cvt_c % (mma_tile_k // CH)  # chunk inside that stage
                ring_2d = cute.make_layout(
                    (mma_tile_m, mma_tile_k, self.kv_stages),
                    stride=(1, mma_tile_m, mma_tile_m * mma_tile_k),
                )
                sK_sd = cute.composition(tAsK, ring_2d)  # (s, d, stage)
                sV_ds = cute.composition(tAsV, ring_2d)  # (d, s, stage)
                shift = cutlass.Float32(NVFP4_SHIFT)

                for s in cutlass.range(prefetch_iters + iters_s):
                    for kv in cutlass.range_constexpr(2):  # 0: K tile, 1: V tile
                        do_tile = s < iters_s
                        t = tile_begin + s
                        if cutlass.const_expr(kv == 1):
                            do_tile = s >= prefetch_iters
                            t = tile_begin + s - prefetch_iters
                        if do_tile:
                            pk_handle = pk_consumer.wait_and_advance()
                            h0 = kv_producer.acquire_and_advance()
                            h1 = kv_producer.acquire_and_advance()
                            dst_stage = h0.index
                            if half == 1:
                                dst_stage = h1.index
                            for i in cutlass.range_constexpr(blk_tile_s // rows_per_pass):
                                r = cvt_r0 + rows_per_pass * i
                                # scale page: rows past the sequence end were
                                # zero-filled by TMA (code 0 -> 0); read page 0
                                blk = t * pages_per_tile + r // PS
                                page_s = 0
                                if blk * PS < seq_len:
                                    page_s = mBlockTable[b_idx, blk]
                                l_s = h_k_idx + h_k * page_s
                                rsc = cute.make_rmem_tensor((2,), cutlass.Float8E4M3FN)
                                if cutlass.const_expr(kv == 0):
                                    gsc = cute.local_tile(mKS[r % PS, None, l_s], (2,), (cvt_c,))
                                else:
                                    gsc = cute.local_tile(mVS[r % PS, None, l_s], (2,), (cvt_c,))
                                cute.autovec_copy(gsc, rsc)
                                rscf = cute.make_rmem_tensor((2,), cutlass.Float32)
                                rscf.store(rsc.load().to(cutlass.Float32))
                                sc0 = rscf[0] * shift
                                sc1 = rscf[1] * shift
                                # 32 packed codes -> f32 -> * scale -> e4m3
                                src = cute.local_tile(sKVp[r, None, pk_handle.index], (CH,), (cvt_c,))
                                rc = cute.make_rmem_tensor((CH,), cutlass.Float4E2M1FN)
                                cute.autovec_copy(src, rc)
                                rf = cute.make_rmem_tensor((CH,), cutlass.Float32)
                                rf.store(rc.load().to(cutlass.Float32))
                                for j in cutlass.range_constexpr(CH):
                                    if cutlass.const_expr(j < NVFP4_GROUP):
                                        rf[j] = rf[j] * sc0
                                    else:
                                        rf[j] = rf[j] * sc1
                                r8 = cute.make_rmem_tensor((CH,), mma_dtype)
                                r8.store(rf.load().to(mma_dtype))
                                if cutlass.const_expr(kv == 0):
                                    dst = cute.local_tile(sK_sd[r, None, dst_stage], (CH,), (cvt_cc,))
                                else:
                                    dst = cute.local_tile(sV_ds[None, r, dst_stage], (CH,), (cvt_cc,))
                                cute.autovec_copy(r8, dst)
                            # generic-proxy SMEM writes -> visible to tcgen05
                            cute.arch.fence_view_async_shared()
                            pk_handle.release()
                            h0.commit()
                            h1.commit()

        ##############################
        # MMA KQ Dispatch
        ##############################
        elif warp_idx == self.mma_kq_warp_id:
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)

            # Wait for Q
            for dk in cutlass.range_constexpr(tiles_dk):
                q_consumer.wait_and_advance()

            s_token = True  # Producer always acquires first
            for s in cutlass.range(iters_s):
                # BMM1: A-operand K (SMEM ring stage for fp8, TMEM for bf16);
                # a_consumer.release() is a tcgen05.commit in both cases.
                k_token = a_consumer.try_wait()
                tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                s_handle = s_producer.acquire_and_advance(s_token)
                for dk in cutlass.range_constexpr(tiles_dk):
                    is_last_iter = dk == tiles_dk - 1
                    k_handle = a_consumer.wait_and_advance(k_token)
                    if is_last_iter:
                        mma_kq_nbar.arrive()  # signal BMM2 to start
                    # shape[2] = UMMA instructions per stage along K
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
                        k_token = a_consumer.try_wait()
                s_handle.commit()

                # Skip the V stages (mma_vp's) and wait for BMM2
                if s > 0:
                    for _ in cutlass.range_constexpr(tiles_dm * tiles_sk):
                        a_consumer.advance()
                    mma_vp_nbar.arrive_and_wait()
                    s_token = s_producer.try_acquire()

        ##############################
        # MMA VP Dispatch
        ##############################
        elif warp_idx == self.mma_vp_warp_id:
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_decrease(self.mma_tma_regs)

            tiled_mma_vp.set(tcgen05.Field.ACCUMULATE, True)
            tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)

            # Skip the first K stages (mma_kq's) and wait for BMM1
            for _ in cutlass.range_constexpr(tiles_dk):
                a_consumer.advance()
            mma_kq_nbar.arrive_and_wait()

            p_token = False
            o_token = True  # Producer always acquires first
            for s in cutlass.range(iters_s):
                if s < iters_s - 1:
                    for _ in cutlass.range_constexpr(tiles_dk):
                        a_consumer.advance()
                    mma_kq_nbar.arrive_and_wait()
                    p_token = p_consumer.try_wait()

                # BMM2: A-operand V
                v_token = a_consumer.try_wait()
                p_handle = p_consumer.wait_and_advance(p_token)
                o_handle = o_producer.acquire_and_advance(o_token)
                for sk in cutlass.range_constexpr(tiles_sk):
                    for dm in cutlass.range_constexpr(tiles_dm):
                        is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                        v_handle = a_consumer.wait_and_advance(v_token)
                        if is_last_iter:
                            mma_vp_nbar.arrive()  # signal BMM1 to start
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
                            v_token = a_consumer.try_wait()
                p_handle.release()
                o_handle.commit()
                o_token = o_producer.try_acquire()

            # Wait for the softmax's final O read, then dealloc TMEM
            o_producer.tail()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)

        ##############################
        # Softmax + Correction + Epilogue Dispatch
        ##############################
        elif warpgroup_idx == self.softmax_warpgroup_id:
            if cutlass.const_expr(self.use_reg_reconfig):
                cute.arch.setmaxregister_increase(self.softmax_regs)

            # Per-(seq, kv head) descales. q/k descale fold into the softmax
            # exponent (S is only ever max'ed and exponentiated), v descale
            # into the output scale.
            qk_descale = mQD[coord_hb] * mKD[coord_hb]
            cta_c = scale_qs_log2_e * qk_descale  # exponent base-2 scale
            cta_scale = scale_qs * qk_descale  # natural-log LSE scale
            v_descale = mVD[coord_hb]

            # Tiled copies
            tmem_op_width = 32
            tmem_op_repeat = tcgen05.Repetition(mma_tile_n * acc_dtype.width // tmem_op_width)
            tmem_load_atom_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tmem_op_repeat), acc_dtype)
            tmem_load_s = tcgen05.make_tmem_copy(tmem_load_atom_s, tCtS[mma_dice + (0,)])
            thr_load_s = tmem_load_s.get_slice(warpgroup_tidx)

            tmem_store_atom_o = cute.make_copy_atom(tcgen05.St32x32bOp(tmem_op_repeat), acc_dtype)
            tmem_store_o = tcgen05.make_tmem_copy(tmem_store_atom_o, tCtO[mma_dice + (0, 0)])
            thr_store_o = tmem_store_o.get_slice(warpgroup_tidx)

            # Partition S and P. Thread t holds TMEM lane t == S row t of the
            # tile; element i of its fragment is column (query row) i.
            tStS = thr_load_s.partition_S(tCtS)
            tSsP = thr_load_s.partition_D(tCsP)

            # Partition O
            tStO = thr_load_s.partition_S(tCtO)
            tSsO = thr_load_s.partition_D(tCsO)
            tSrO = cute.make_rmem_tensor_like(tSsO)

            # colmax (running, raw S units)
            tSsM = thr_load_s.partition_D(tCsM)
            tSrM_prev = cute.make_rmem_tensor_like(tSsM)
            tSrM_prev.fill(-cutlass.Float32.inf)

            # colsum (per thread, reduced after the loop)
            tSsL = thr_load_s.partition_D(tCsL)
            tSrL = cute.make_rmem_tensor_like(tSsL[cpy_dice + (0,)])
            tSrL.fill(cutlass.Float32(0))

            assert warp_threads >= cute.size(tSsM)

            # Causal tail for q_len > 1: column n (= qi * h_r + hr) may see
            # keys 0 .. seq_len - q_len + qi.
            if cutlass.const_expr(self.q_len > 1):
                h_r_true = n_q // self.q_len
                col_lim = cute.make_fragment_like(tSrM_prev, cutlass.Int32)
                for i in cutlass.range_constexpr(cute.size(col_lim)):
                    col_lim[i] = seq_len - self.q_len + (n_base + i) // h_r_true

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
                tile_idx = tile_begin + s
                row = tile_idx * blk_tile_s + warpgroup_tidx

                # Load S from tmem
                s_handle = s_consumer.wait_and_advance()
                tSrS = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
                cute.copy(tmem_load_s, tStS[cpy_dice + (s_handle.index,)], tSrS)
                cute.arch.fence_view_async_tmem_load()
                s_handle.release()

                # Mask keys past the sequence end (select, so garbage/zero
                # K rows from padding or OOB pages can't leak in)
                if cutlass.const_expr(self.q_len == 1):
                    if row >= seq_len:
                        tSrS.fill(-cutlass.Float32.inf)
                else:
                    for i in cutlass.range_constexpr(cute.size(tSrS)):
                        if row > col_lim[i]:
                            tSrS[i] = -cutlass.Float32.inf

                # colmax in warp RF
                tSrM = cute.make_rmem_tensor_like(tSsM)
                tSrM_lane = cutlass.Float32(0)  # Avoid dynamic register indexing
                for i in cutlass.range_constexpr(cute.size(tSrS)):
                    tSrM[i] = warp_fmax(tSrS[i])
                    if i == lane_idx:
                        tSrM_lane = tSrM[i]

                # colmax in smem (running across this CTA's tiles)
                if lane_store_max:
                    smem_fmax(tSsM.iterator + tSsM.layout(lane_idx), tSrM_lane)

                softmax_nbar.arrive_and_wait()
                cute.autovec_copy(tSsM, tSrM)

                # A column with no visible key so far has max -inf: use 0 as
                # the exponent reference so p = exp2(-inf) = 0 and the
                # correction exp2(-inf) = 0 instead of NaN from (-inf) - (-inf).
                m_use = cute.make_rmem_tensor_like(tSrM)
                for i in cutlass.range_constexpr(cute.size(tSrM)):
                    m_use[i] = tSrM[i]
                    if tSrM[i] == -cutlass.Float32.inf:
                        m_use[i] = cutlass.Float32(0.0)

                # Online softmax
                tSrP = cute.make_rmem_tensor(tSsP.shape[:-1], mma_dtype)
                tSrP_f32 = cute.make_rmem_tensor(tSrS.shape, acc_dtype)
                for i in cutlass.range_constexpr(0, cute.size(tSrS), 2):
                    p_f32x2 = fadd2((tSrS[i], tSrS[i + 1]), (-m_use[i], -m_use[i + 1]))
                    p_f32x2 = fmul2(p_f32x2, (cta_c, cta_c))
                    tSrP_f32[i] = exp2(p_f32x2[0])
                    tSrP_f32[i + 1] = exp2(p_f32x2[1])
                if cutlass.const_expr(FP8):
                    p_scale_f32 = cutlass.Float32(p_scale)
                    tSrPq_f32 = cute.make_rmem_tensor(tSrS.shape, acc_dtype)
                    for i in cutlass.range_constexpr(0, cute.size(tSrS), 2):
                        pq_f32x2 = fmul2((tSrP_f32[i], tSrP_f32[i + 1]), (p_scale_f32, p_scale_f32))
                        tSrPq_f32[i] = pq_f32x2[0]
                        tSrPq_f32[i + 1] = pq_f32x2[1]
                    tSrP.store(tSrPq_f32.load().to(mma_dtype))
                else:
                    tSrP.store(tSrP_f32.load().to(mma_dtype))

                # Store P to smem (generic writes -> async-proxy UMMA read)
                p_handle = p_producer.acquire_and_advance()
                cute.autovec_copy(tSrP, tSsP[cpy_dice + (p_handle.index,)])
                cute.arch.fence_view_async_shared()
                p_handle.commit()

                # Correction factor and colsum (unscaled p)
                correction = cute.make_rmem_tensor_like(tSrM)
                for i in cutlass.range_constexpr(0, cute.size(tSrM), 2):
                    c_f32x2 = fadd2((tSrM_prev[i], tSrM_prev[i + 1]), (-m_use[i], -m_use[i + 1]))
                    c_f32x2 = fmul2(c_f32x2, (cta_c, cta_c))
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

                # Correct O — only when some column max actually moved. In
                # decode the running max settles after a few tiles, so most
                # iterations skip the full O TMEM load/scale/store round trip
                # that otherwise sits on the critical path before GEMM2(s).
                # tSrM / tSrM_prev come from the shared colmax, so the test is
                # warpgroup-uniform; when nothing moved every correction is
                # exactly exp2(0) = 1, so skipping is bit-identical.
                if s > 0:
                    rescale = cutlass.Boolean(False)
                    for i in cutlass.range_constexpr(cute.size(tSrM)):
                        if tSrM[i] != tSrM_prev[i]:
                            rescale = cutlass.Boolean(True)
                    o_handle = o_consumer.wait_and_advance()
                    if rescale:
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
                        cute.arch.fence_view_async_tmem_store()
                    o_handle.release()

                tSrM_prev.store(tSrM.load())

            #
            # Epilogue
            #
            # colsum: warp reduce, then per-warp partials in smem
            tSrL_lane = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(cute.size(tSrL)):
                tSrL[i] = cute.arch.warp_reduction_sum(tSrL[i])
                if i == lane_idx:
                    tSrL_lane = tSrL[i]
            if lane_store_max:
                tSsL[cpy_dice + (warpgroup_widx,)][lane_idx] = tSrL_lane
            softmax_nbar.arrive_and_wait()

            # Row statistics (one lane per query column)
            if warpgroup_widx == 0 and lane_store_max and n_base + lane_idx < n_q:
                sL_lane_wg = sL[0, lane_idx, None]
                sL_lane = sL_lane_wg[0] + sL_lane_wg[1] + sL_lane_wg[2] + sL_lane_wg[3]
                sM_lane = sM[0, lane_idx] * cta_scale  # -inf stays -inf
                n_lane = n_base + lane_idx
                if cutlass.const_expr(SINGLE_SPLIT):
                    mLSE[n_lane, coord_hb] = sM_lane + cute.math.log(sL_lane, fastmath=False)
                else:
                    if n_splits_seq == 1:  # unsplit sequence: final LSE now
                        mLSE[n_lane, coord_hb] = sM_lane + cute.math.log(sL_lane, fastmath=False)
                    else:
                        mM_partial[n_lane, coord_hb, split_slot] = sM_lane
                        mL_partial[n_lane, coord_hb, split_slot] = sL_lane

            # Final O from TMEM
            o_handle = o_consumer.wait_and_advance()
            cute.copy(thr_load_s, tStO, tSrO)
            cute.arch.fence_view_async_tmem_load()
            o_handle.release()  # Final release signals tmem dealloc

            # Store O straight from registers: thread t owns headdim row
            # dm * 128 + t, element i is query column n_base + i, so each
            # store instruction writes 128 consecutive d values (coalesced).
            # Unsplit sequences (one chunk) are normalized and written as the
            # final bf16 output; split ones write fp32 partials.
            out_scale = scale_o * v_descale * (1.0 / p_scale)
            part_scale = v_descale * (1.0 / p_scale)
            inv_l = cute.make_rmem_tensor_like(tSrM_prev)
            for i in cutlass.range_constexpr(cute.size(inv_l)):
                # + tiny: an empty row (seq_len 0, e.g. CUDA-graph padding)
                # has l = 0 and O = 0 -> writes 0 instead of NaN. l >= 1
                # otherwise, so the epsilon is exact in fp32.
                l_i = sL[0, i, 0] + sL[0, i, 1] + sL[0, i, 2] + sL[0, i, 3] + cutlass.Float32(1e-30)
                inv_l[i] = out_scale / l_i
            write_final = cutlass.Boolean(True)
            if cutlass.const_expr(not SINGLE_SPLIT):
                write_final = n_splits_seq == 1
            if write_final:
                for dm in cutlass.range_constexpr(tiles_dm):
                    tSrO_dm = tSrO[cpy_dice + (dm, 0)]
                    d_idx = dm * mma_tile_m + warpgroup_tidx
                    for i in cutlass.range_constexpr(cute.size(tSrO_dm)):
                        if n_base + i < n_q:
                            mO[d_idx, n_base + i, coord_hb] = mO.element_type(tSrO_dm[i] * inv_l[i])
            else:
                for dm in cutlass.range_constexpr(tiles_dm):
                    tSrO_dm = tSrO[cpy_dice + (dm, 0)]
                    d_idx = dm * mma_tile_m + warpgroup_tidx
                    for i in cutlass.range_constexpr(cute.size(tSrO_dm)):
                        if n_base + i < n_q:
                            mO_partial[d_idx, n_base + i, coord_hb, split_slot] = tSrO_dm[i] * part_scale

        return

    @staticmethod
    @cute.kernel
    def reduction(
        o: cute.Tensor,  # (d, h_q, b)
        lse: cute.Tensor,  # (h_q, b)
        o_partial: cute.Tensor,  # (d, h_q, b, max_splits)
        m_partial: cute.Tensor,  # (h_q, b, max_splits)
        l_partial: cute.Tensor,  # (h_q, b, max_splits)
        n_splits: cute.Tensor,  # (b,) chunks per sequence from the plan
        scale_o: cutlass.Float32,
        pdl: cutlass.Constexpr = False,  # static kernel: no self
    ):
        if cutlass.const_expr(pdl):
            # launched early (PDL): the partials are only complete once the
            # decode grid has fully finished
            cute.arch.griddepcontrol_wait()
        d_blk_idx, coord_h, coord_b = cute.arch.block_idx()
        d_per_blk, _, _ = cute.arch.block_dim()
        d_idx, _, _ = cute.arch.thread_idx()
        coord_d = d_blk_idx * d_per_blk + d_idx

        # Only the splits this sequence actually ran: the partial buffers of
        # the others were never written (no workspace init needed). A
        # sequence with one chunk already wrote its final output in decode.
        kv_splits = o_partial.shape[3]  # max_splits (compile-time)
        n_valid = n_splits[coord_b]
        if n_valid > kv_splits:
            n_valid = kv_splits

        # Issue every split's loads up front (kv_splits is compile-time, so
        # this unrolls into independent predicated loads): one DRAM latency
        # instead of a chain of ~3 * kv_splits dependent ones.
        m_vals = cute.make_rmem_tensor((kv_splits,), cutlass.Float32)
        l_vals = cute.make_rmem_tensor((kv_splits,), cutlass.Float32)
        o_vals = cute.make_rmem_tensor((kv_splits,), cutlass.Float32)
        for j in cutlass.range_constexpr(kv_splits):
            m_vals[j] = -cutlass.Float32.inf
            l_vals[j] = cutlass.Float32(0)
            o_vals[j] = cutlass.Float32(0)
            if j < n_valid:
                m_vals[j] = m_partial[coord_h, coord_b, j]
                l_vals[j] = l_partial[coord_h, coord_b, j]
                o_vals[j] = o_partial[coord_d, coord_h, coord_b, j]

        m_max = -cutlass.Float32.inf
        for j in cutlass.range_constexpr(kv_splits):
            m_max = cute.arch.fmax(m_max, m_vals[j])

        o_acc = cutlass.Float32(0)
        l_acc = cutlass.Float32(0)
        for j in cutlass.range_constexpr(kv_splits):
            w = cutlass.Float32(0)
            if m_vals[j] != -cutlass.Float32.inf:
                w = exp2(log2_e * (m_vals[j] - m_max))
            l_acc += w * l_vals[j]
            o_acc += w * o_vals[j]

        inv_l = cutlass.Float32(0)
        if l_acc > cutlass.Float32(0):
            inv_l = 1.0 / l_acc

        # n_valid <= 1: decode already wrote the final O / LSE — leave it.
        if n_valid > 1:
            if coord_d < o.shape[0]:
                o[coord_d, coord_h, coord_b] = o.element_type(scale_o * o_acc * inv_l)
            if d_blk_idx == 0 and d_idx == 0:
                lse[coord_h, coord_b] = m_max + cute.math.log(l_acc, fastmath=False)

        return


##############################################################################
# Host helpers (also what a vLLM backend would call)
##############################################################################


class SplitPlan:
    """Host-side occupancy-pinned split-KV plan (see plan_split_kv)."""

    def __init__(self, work, n_splits, chunk_tiles, slots):
        self.work = work  # list of (seq, tile_begin, tile_end, split_slot)
        self.n_splits = n_splits  # chunks per sequence
        self.chunk_tiles = chunk_tiles  # tiles per chunk (last chunk may be short)
        self.slots = slots  # work items that fit one wave

    @property
    def num_work(self):
        return len(self.work)

    def work_tensor(self, device="cpu"):
        t = torch.tensor(self.work if self.work else [(0, 0, 0, 0)], dtype=torch.int32)
        return t.reshape(-1, 4).contiguous().to(device)

    def n_splits_tensor(self, device="cpu"):
        return torch.tensor(self.n_splits, dtype=torch.int32).to(device)

    def __repr__(self):
        real = sum(1 for w in self.work if w[2] > w[1])
        return (
            f"SplitPlan(chunk={self.chunk_tiles} tiles, work={real}"
            f"{f'+{self.num_work - real} pad' if self.num_work > real else ''}, "
            f"slots/wave={self.slots}, max chunks/seq={max(self.n_splits)})"
        )


def plan_split_kv(
    seq_lens,
    heads_k,
    n_q_tiles,
    sm_count,
    ctas_per_sm=1,
    max_splits=16,
    force_splits=0,
    min_chunk_tiles=1,
    pad_to=0,
):
    """Occupancy-pinned split-KV plan (the FlashInfer PrefillPlan / FlashDecoding
    scheme), in units of 128-row KV tiles.

    The grid is (work items, q tiles, kv heads), so one wave holds
        slots = sm_count * ctas_per_sm // (heads_k * n_q_tiles)
    work items. Pick the SMALLEST contiguous chunk size such that the total
    number of chunks over all sequences fits in one wave; every sequence is
    then cut into ceil(tiles / chunk) contiguous chunks. Long sequences get
    more chunks, short ones fewer, so varlen batches balance, and each CTA
    streams one contiguous KV range front to back.

    If the batch alone already fills the wave, nothing is split (each
    sequence is one work item, ordered longest first for the waves).
    force_splits > 0 cuts every sequence into that many chunks instead.
    pad_to > 0 pads the work list to a fixed length with empty items
    (begin == end, they exit at once) — keeps the grid fixed for CUDA graphs.
    """
    tiles = [max(1, math.ceil(int(L) / tile_s_rows)) for L in seq_lens]
    T = max(tiles)
    slots = max(1, (sm_count * ctas_per_sm) // (heads_k * n_q_tiles))
    lo = max(min_chunk_tiles, math.ceil(T / max_splits))  # cap chunks/seq
    if force_splits > 0:
        chunk = max(lo, math.ceil(T / force_splits))
    elif len(tiles) >= slots or max_splits == 1:
        chunk = T
    else:
        hi = T
        while lo < hi:  # smallest chunk whose total work fits one wave
            mid = (lo + hi) // 2
            if sum(math.ceil(t / mid) for t in tiles) <= slots:
                hi = mid
            else:
                lo = mid + 1
        chunk = lo
    work, n_splits = [], []
    for b, t in enumerate(tiles):
        k = math.ceil(t / chunk)
        n_splits.append(k)
        for j in range(k):
            work.append((b, j * chunk, min((j + 1) * chunk, t), j))
    work.sort(key=lambda w: -(w[2] - w[1]))  # longest first (stable)
    if pad_to:
        assert pad_to >= len(work), f"pad_to {pad_to} < {len(work)} work items"
        work += [(0, 0, 0, 0)] * (pad_to - len(work))
    assert max(n_splits) <= max_splits
    return SplitPlan(work, n_splits, chunk, slots)


STRIDE_FIELDS = (
    "q_seq",
    "o_seq",
    "k_page",
    "k_row",
    "k_head",
    "v_page",
    "v_row",
    "v_head",
    "qd_seq",
    "qd_head",
    "kd_seq",
    "kd_head",
    "vd_seq",
    "vd_head",
    "ks_page",
    "ks_row",
    "ks_head",
    "vs_page",
    "vs_row",
    "vs_head",
)


def _descale_strides(t, num_seqs, heads_k):
    """(seq, head) element strides of a descale tensor: a 1-element tensor
    is a per-tensor scale (0, 0); (h_k,) per head; (b, h_k) per seq+head."""
    if t.numel() == 1:
        return 0, 0
    if t.dim() == 1 and t.shape[0] == heads_k:
        return 0, t.stride(0)
    assert t.dim() == 2 and t.shape[1] == heads_k, f"descale shape {tuple(t.shape)}"
    return t.stride(0), t.stride(1)


def _kv_elem_strides(t, kv_bits, name):
    """(page, row, head) strides of a KV view in units of the KV ELEMENT.
    8-bit formats: the view's own strides. Sub-byte formats (fp6 / fp4 /
    nvfp4) are passed as uint8 BYTE views (pages, PS, h_k, d * bits / 8);
    their byte strides are converted to element strides."""
    assert t.stride(3) == 1, f"{name}: head dim must be contiguous"
    byte_strides = [x * t.element_size() for x in t.stride()[:3]]
    assert all(x % 16 == 0 for x in byte_strides), f"{name}: TMA needs 16-byte-multiple strides, got {byte_strides} B"
    if kv_bits == 8:
        return tuple(t.stride()[:3])
    assert t.element_size() == 1, f"{name}: sub-byte KV must be a uint8 byte view"
    assert all((x * 8) % kv_bits == 0 for x in byte_strides), (
        f"{name}: byte strides {byte_strides} are not whole {kv_bits}-bit elements"
    )
    return tuple(x * 8 // kv_bits for x in byte_strides)


def kernel_strides(q, o, k_cache, v_cache, q_descale, k_descale, v_descale, kv_bits=8, k_scales=None, v_scales=None):
    """Element strides for the kernel's `strides` argument.

    q, o      : (b, h_k, n_q, d) views (rows d-contiguous; seq stride free)
    k/v_cache : (num_pages, page_size, h_k, d) LOGICAL views of any physical
                layout with d contiguous (vLLM NHD, interleaved K|V, FlashInfer,
                head-major HND); for sub-byte formats uint8 byte views with
                last dim d * kv_bits / 8
    *_descale : 1-element, (h_k,) or (b, h_k) fp32 tensors
    k/v_scales: nvfp4 only, (num_pages, page_size, h_k, d / 16) e4m3 (uint8)
    """
    b, h_k, n_q, d = q.shape
    for name, t in (("q", q), ("o", o)):
        assert t.stride(3) == 1 and t.stride(2) == d and t.stride(1) == n_q * d, (
            f"{name}: rows must be packed (n_q, h_k) with d contiguous, got {t.stride()}"
        )
    ks = (0, 0, 0)
    vs = (0, 0, 0)
    if k_scales is not None:
        assert k_scales.stride(3) == 1 and v_scales.stride(3) == 1
        ks = tuple(k_scales.stride()[:3])
        vs = tuple(v_scales.stride()[:3])
    vals = (
        q.stride(0),
        o.stride(0),
        *_kv_elem_strides(k_cache, kv_bits, "k_cache"),
        *_kv_elem_strides(v_cache, kv_bits, "v_cache"),
        *_descale_strides(q_descale, b, h_k),
        *_descale_strides(k_descale, b, h_k),
        *_descale_strides(v_descale, b, h_k),
        *ks,
        *vs,
    )
    assert all(0 <= x < 2**31 for x in vals), f"stride out of int32 range: {vals}"
    return tuple(int(x) for x in vals)


# ----------------------------------------------------------------------------
# Narrow KV formats: value tables, quantization and bit packing
# ----------------------------------------------------------------------------
_MINIFLOAT = {  # format: (exponent bits, mantissa bits, bias); no inf / nan
    "e3m2": (3, 2, 3),
    "e2m3": (2, 3, 1),
    "e2m1": (2, 1, 1),
}


def minifloat_table(fmt):
    """Values of every code of an fp6 / fp4 format (code = sign|exp|mant)."""
    eb, mb, bias = _MINIFLOAT[fmt]
    vals = []
    for code in range(1 << (1 + eb + mb)):
        sign = -1.0 if code >> (eb + mb) else 1.0
        e = (code >> mb) & ((1 << eb) - 1)
        m = code & ((1 << mb) - 1)
        if e == 0:
            v = (m / (1 << mb)) * 2.0 ** (1 - bias)
        else:
            v = (1 + m / (1 << mb)) * 2.0 ** (e - bias)
        vals.append(sign * v)
    return torch.tensor(vals, dtype=torch.float32)


def minifloat_quantize(x, fmt):
    """Round-to-nearest (saturating) codes of x in an fp6 / fp4 format.
    Returns (codes uint8, decoded float values)."""
    table = minifloat_table(fmt)
    pos = table[: len(table) // 2]  # non-negative half, ascending
    ax = x.abs().clamp(max=float(pos[-1]))
    idx = torch.bucketize(ax, pos)  # first pos >= ax
    lo = (idx - 1).clamp(min=0)
    hi = idx.clamp(max=len(pos) - 1)
    pick_lo = (ax - pos[lo]) <= (pos[hi] - ax)
    mag = torch.where(pick_lo, lo, hi)
    neg = (x < 0) & (pos[mag] > 0)
    codes = (mag + neg.to(mag.dtype) * (len(table) // 2)).to(torch.uint8)
    return codes, table[codes.long()]


def pack_subbyte(codes, bits):
    """Pack codes (..., d) into bytes (..., d * bits / 8): little-endian
    bitstream, element i at bits [bits*i, bits*i + bits) (fp4: element 2i in
    the low nibble)."""
    c = codes.to(torch.int64)
    d = c.shape[-1]
    if bits == 8:
        return codes.to(torch.uint8)
    if bits == 4:
        c = c.reshape(*c.shape[:-1], d // 2, 2)
        return (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)
    assert bits == 6 and d % 4 == 0
    c = c.reshape(*c.shape[:-1], d // 4, 4)
    w = c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)
    out = torch.stack([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF], dim=-1)
    return out.reshape(*c.shape[:-2], d // 4 * 3).to(torch.uint8)


def nvfp4_quantize(x, group=NVFP4_GROUP):
    """NVFP4: e2m1 codes, one e4m3 scale per `group` elements of the last dim
    and one fp32 global scale. Returns (codes uint8, scales e4m3, global
    scale, values the T5 kernel feeds its fp8 MMA, descale to pass).
    Kernel value = e4m3(e2m1 * scale * 2^-3); descale = global * 2^3."""
    *lead, d = x.shape
    xg = x.reshape(*lead, d // group, group)
    amax_g = xg.abs().amax(dim=-1)
    gs = float(x.abs().amax().clamp(min=1e-12)) / (6.0 * 448.0)
    sc = (amax_g / 6.0 / gs).clamp(max=448.0).to(torch.float8_e4m3fn)
    scf = sc.float()
    denom = (scf * gs).unsqueeze(-1)
    q = torch.where(denom > 0, xg / denom.clamp(min=1e-30), torch.zeros_like(xg))
    codes, _ = minifloat_quantize(q, "e2m1")
    lut = minifloat_table("e2m1")
    kern = (lut[codes.long()] * scf.unsqueeze(-1) * NVFP4_SHIFT).to(torch.float8_e4m3fn)
    return (codes.reshape(*lead, d), sc, gs, kern.float().reshape(*lead, d), gs / NVFP4_SHIFT)


def pack_q(q, num_seqs, q_len, heads_k):
    """vLLM q (num_seqs * q_len, h_q, d) -> kernel q (b, h_k, q_len * h_r, d).
    A free view for q_len == 1."""
    _, h_q, d = q.shape
    h_r = h_q // heads_k
    if q_len == 1:
        return q.view(num_seqs, heads_k, h_r, d)
    return (
        q.view(num_seqs, q_len, heads_k, h_r, d)
        .permute(0, 2, 1, 3, 4)
        .reshape(num_seqs, heads_k, q_len * h_r, d)
        .contiguous()
    )


def unpack_o(o, num_seqs, q_len, heads_k):
    """kernel o (b, h_k, q_len * h_r, d) -> vLLM (num_seqs * q_len, h_q, d)."""
    b, h_k, n_q, d = o.shape
    h_r = n_q // q_len
    if q_len == 1:
        return o.view(num_seqs, h_k * h_r, d)
    return o.view(b, h_k, q_len, h_r, d).permute(0, 2, 1, 3, 4).reshape(b * q_len, h_k * h_r, d)


def quantize_q_fp8(q, num_seqs, q_len, heads_k):
    """Per-(seq, kv head) e4m3 quantization of q (num_seqs * q_len, h_q, d).
    Returns (q_fp8 same shape, q_descale (num_seqs, h_k) fp32)."""
    _, h_q, d = q.shape
    h_r = h_q // heads_k
    qg = q.float().view(num_seqs, q_len, heads_k, h_r, d)
    amax = qg.abs().amax(dim=(1, 3, 4)).clamp(min=1e-6)  # (b, h_k)
    q_descale = (amax / 448.0).contiguous()
    q8 = (qg / q_descale[:, None, :, None, None]).to(torch.float8_e4m3fn)
    return q8.view(num_seqs * q_len, h_q, d), q_descale


def _to_cute(t, dtype=None):
    """torch -> cute tensor; fp8 goes through a uint8 view (dlpack-safe)."""
    if t.dtype == torch.float8_e4m3fn:
        ct = from_dlpack(t.view(torch.uint8), assumed_align=16)
        ct.element_type = cutlass.Float8E4M3FN
        return ct
    ct = from_dlpack(t, assumed_align=16)
    if dtype is not None:
        ct.element_type = dtype
    return ct


def emulate_decode(
    q_packed,  # (b, h_k, n_q, d) float — the exact values the kernel reads
    k_pool,  # (num_pages, PS, h_k, d) float
    v_pool,
    block_table,  # (b, max_blocks) int
    seq_lens,  # (b,) int
    q_descale,
    k_descale,
    v_descale,  # (b, h_k) float
    scale_qs,
    scale_o,
    q_len,
    chunk_tiles,
    fp8,
):
    """Bit-faithful model of the kernel's arithmetic (contiguous chunk split, running
    max, -inf guard, P rounding to e4m3(p * 448) or bf16, descale folding).
    Also returns a plain fp32 attention for the fp8-vs-exact gap."""
    b, h_k, n_q, d = q_packed.shape
    PS = k_pool.shape[1]
    h_r = n_q // q_len
    p_scale = FP8_P_SCALE if fp8 else 1.0
    o_emu = torch.zeros(b, h_k, n_q, d)
    lse_emu = torch.zeros(b, h_k, n_q)
    o_exact = torch.zeros(b, h_k, n_q, d)
    for bi in range(b):
        L = int(seq_lens[bi])
        n_blk = math.ceil(L / PS)
        pages = block_table[bi, :n_blk].long()
        Kb = k_pool[pages].reshape(-1, h_k, d)[:L].permute(1, 0, 2)  # (h_k, L, d)
        Vb = v_pool[pages].reshape(-1, h_k, d)[:L].permute(1, 0, 2)
        S = torch.einsum("gnd,gld->gnl", q_packed[bi], Kb)  # raw scores
        qi = torch.arange(n_q) // h_r
        lim = L - q_len + qi  # (n_q,)
        masked = torch.arange(L)[None, :] > lim[:, None]  # (n_q, L)
        S = S.masked_fill(masked[None], -math.inf)
        qk = q_descale[bi] * k_descale[bi]  # (h_k,)
        c = (scale_qs * log2_e * qk)[:, None]  # (h_k, 1)
        cta_scale = (scale_qs * qk)[:, None]
        vd = v_descale[bi][:, None, None]

        tiles = math.ceil(L / tile_s_rows)
        o_parts, m_parts, l_parts = [], [], []
        n_chunks = math.ceil(tiles / chunk_tiles)
        for j in range(n_chunks):  # contiguous chunks, as in the plan
            m = torch.full((h_k, n_q), -math.inf)
            l = torch.zeros(h_k, n_q)
            o = torch.zeros(h_k, n_q, d)
            for t in range(j * chunk_tiles, min((j + 1) * chunk_tiles, tiles)):
                sl = slice(t * tile_s_rows, min((t + 1) * tile_s_rows, L))
                St = S[:, :, sl]
                m_new = torch.maximum(m, St.amax(dim=-1))
                m_use = torch.where(m_new == -math.inf, torch.zeros_like(m_new), m_new)
                p = torch.exp2((St - m_use[..., None]) * c[..., None])
                corr = torch.exp2((m - m_use) * c)
                l = l * corr + p.sum(dim=-1)
                if fp8:
                    pq = (p * p_scale).to(torch.float8_e4m3fn).float()
                else:
                    pq = p.to(torch.bfloat16).float()
                o = o * corr[..., None] + torch.einsum("gnl,gld->gnd", pq, Vb[:, sl])
                m = m_new
            o_parts.append(o * vd / p_scale)
            m_parts.append(m * cta_scale)
            l_parts.append(l)
        m_all = torch.stack(m_parts).amax(dim=0)
        w = [torch.exp(mp - m_all) for mp in m_parts]
        o_sum = sum(wi[..., None] * op for wi, op in zip(w, o_parts))
        l_sum = sum(wi * lp for wi, lp in zip(w, l_parts))
        o_emu[bi] = scale_o * o_sum / l_sum[..., None]
        lse_emu[bi] = m_all + torch.log(l_sum)

        # exact fp32 attention (no P rounding) for reference
        P = torch.softmax(S * cta_scale[..., None], dim=-1)
        o_exact[bi] = scale_o * torch.einsum("gnl,gld->gnd", P, Vb) * vd
    return o_emu, lse_emu, o_exact


def run(
    batches: int = 1,
    seqlen: int = 1024,
    varlen: bool = False,
    q_len: int = 1,
    heads_q: int = 24,
    heads_k: int = 4,
    headdim: int = 256,
    page_size: int = 128,
    kv_splits: int = 0,
    q_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
    kv_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
    tolerance: float = 0.05,
    scale_s: float = 0.0,
    scale_o: float = 1.0,
    warmup_iterations: int = 0,
    iterations: int = 0,
    skip_ref_check: bool = False,
    seed: int = 1111,
    ctas_per_sm: int = 2,
    no_l2_flush: bool = False,
    kv_stages: int = 0,
    contiguous_pages: bool = False,
    max_splits: int = 16,
    pad_work_to: int = 0,
    kv_layout: str = "nhd",
    strided_qo: bool = False,
    descale: str = "per_seq_head",
    pdl: bool = False,
    l2_prefetch: int = 0,
    kv_evict_first: int = -1,  # -1 auto: on for fp8 Q (vLLM default), off for bf16 Q
    kv_format: str = "fp8",
    kv_from_float: bool = False,
    **kwargs,
):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")
    torch.manual_seed(seed)
    dev = "cuda"

    fp8 = q_dtype.width == 8
    num_seqs = batches
    if kv_evict_first < 0:
        # matches the vLLM backend default: EVICT_FIRST wins on the fp8 path,
        # but costs the bf16 path 4-18% at 32K-256K
        kv_evict_first = 1 if fp8 else 0
    h_r = heads_q // heads_k
    n_q = q_len * h_r
    PS = page_size

    # Kernel config. Default 2 CTAs/SM (40-CTA one-wave plan for one
    # sequence) with a one-tile KV ring: measured best on Thor. At 2 CTAs/SM
    # the bf16 path uses a single cvt warpgroup so two CTAs fit the
    # 64K-register file.
    assert kv_format in KV_FORMATS
    if kv_format != "fp8":
        assert fp8, f"kv_format {kv_format} runs on the fp8-Q path (--q_dtype Float8E4M3FN)"
        kv_dtype = {
            "e3m2": cutlass.Float6E3M2FN,
            "e2m3": cutlass.Float6E2M3FN,
            "e2m1": cutlass.Float4E2M1FN,
            "nvfp4": cutlass.Float4E2M1FN,
        }[kv_format]
    kv_bits = KV_FORMAT_BITS[kv_format]
    from_float = kv_format != "fp8" or kv_from_float
    if fp8:
        convert_warpgroups = 1 if kv_format == "nvfp4" else 0
    elif headdim > 128 and ctas_per_sm == 1:
        convert_warpgroups = 2
    else:
        convert_warpgroups = 1
    ght_base = 16 if fp8 else 8
    # Rows per CTA tile: next power of two >= n_q (min 8 bf16 / 16 fp8, max 32;
    # wider n_q is tiled across CTAs). TMEM ld/st widths must be powers of two.
    grouped_head_tile = max(ght_base, min(32, 1 << max(0, (n_q - 1).bit_length())))
    n_tiles = math.ceil(n_q / grouped_head_tile)

    # Sequence lengths (include the q_len new tokens). A decode step always
    # has at least the new token(s) in the cache, so L < q_len is clamped.
    if seqlen < q_len:
        print(f"\tseqlen {seqlen} < q_len {q_len}: using seqlen = {q_len}")
        seqlen = q_len
    if varlen:
        seq_lens = torch.randint(q_len, seqlen + 1, (num_seqs,), dtype=torch.int32)
        seq_lens[0] = seqlen  # keep the max for the split heuristic
    else:
        seq_lens = torch.full((num_seqs,), seqlen, dtype=torch.int32)
    max_seqlen = int(seq_lens.max())

    hw = cutlass.utils.HardwareInfo()
    sm_count = hw.get_device_multiprocessor_count()
    sm_count = 20 if sm_count <= 0 else sm_count  # Jetson AGX Thor: 20 SMs
    # Occupancy-pinned split-KV plan (contiguous chunks, one wave).
    plan = plan_split_kv(
        seq_lens.tolist(),
        heads_k,
        n_tiles,
        sm_count,
        ctas_per_sm=ctas_per_sm,
        max_splits=max_splits,
        force_splits=kv_splits,
        pad_to=pad_work_to,
    )

    print("Paged FMHA decode (sm_110) with:")
    print(f"\tnum_seqs: {num_seqs}, max seqlen: {max_seqlen}, varlen: {varlen}")
    print(f"\tseq_lens: {seq_lens.tolist()[:8]}{' ...' if num_seqs > 8 else ''}")
    print(f"\tq_len: {q_len}, heads_q: {heads_q}, heads_k: {heads_k}, headdim: {headdim}")
    grid_ctas = plan.num_work * heads_k * n_tiles
    print(
        f"\tpage_size: {PS}, sm_count: {sm_count}, ctas_per_sm: {ctas_per_sm}, "
        f"grid: {grid_ctas} CTAs, pages: {'contiguous' if contiguous_pages else 'shuffled'}"
    )
    print(f"\tplan: {plan}")
    print(f"\tq_dtype: {q_dtype}, kv_dtype: {kv_dtype}")

    fmha = MixedInputFusedMultiHeadAttentionDecode(
        headdim=headdim,
        grouped_head_tile=grouped_head_tile,
        convert_warpgroups=convert_warpgroups,
        page_size=PS,
        q_len=q_len,
        max_splits=max_splits,
        ctas_per_sm=ctas_per_sm,
        max_kv_stages=kv_stages,
        with_reduction=max(plan.n_splits) > 1,
        pdl=pdl,
        l2_prefetch=l2_prefetch,
        kv_evict_first=bool(kv_evict_first),
        kv_format=kv_format,
    )
    fmha.can_implement(num_seqs, heads_q, heads_k, headdim, q_dtype, kv_dtype)

    if scale_s == 0.0:
        scale_s = 1.0 / math.sqrt(headdim)

    #
    # Paged KV pool with a SHUFFLED page assignment; unreferenced pages are
    # poisoned with e4m3 NaN so any stray read shows up in the output.
    #
    pages_per_seq = [math.ceil(int(L) / PS) for L in seq_lens]
    max_blocks = max(pages_per_seq)
    used_pages = sum(pages_per_seq)
    num_pages = used_pages + max(4, used_pages // 4)
    # Shuffled pages exercise the paging logic; contiguous pages (identity-
    # ordered block table) match a freshly allocated cache and the FI driver.
    perm = torch.arange(num_pages) if contiguous_pages else torch.randperm(num_pages)
    block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32)
    cursor = 0
    for bi, n_blk in enumerate(pages_per_seq):
        block_table[bi, :n_blk] = perm[cursor : cursor + n_blk].to(torch.int32)
        cursor += n_blk
    unused = perm[cursor:]

    # Descales: per (seq, kv head), per kv head, or one per-tensor scale
    # (vLLM's layer._k_scale) — the kernel takes them via strides.
    def make_descale():
        if descale == "per_tensor":
            return torch.rand(1) * 1.5 + 0.5
        if descale == "per_head":
            return torch.rand(heads_k) * 1.5 + 0.5
        return torch.rand(num_seqs, heads_k) * 1.5 + 0.5

    def expand_descale(t):  # logical (b, h_k) for the reference
        t = t.float()
        if t.numel() == 1:
            return t.reshape(1, 1).expand(num_seqs, heads_k)
        if t.dim() == 1:
            return t.reshape(1, heads_k).expand(num_seqs, heads_k)
        return t

    row_bytes = headdim * kv_bits // 8

    def kv_pool():
        """-> (bytes (pages, PS, h_k, row_bytes) uint8, kernel values (float,
        what the MMA multiplies), descale, nvfp4 scale bytes or None,
        unquantized float pool or None). Unreferenced pages are poisoned:
        e4m3 NaN for fp8, the largest code (and scale) otherwise."""
        shape = (num_pages, PS, heads_k, headdim)
        if not from_float:  # legacy fp8 test data: raw e4m3 values, random descales
            x8 = (torch.randn(shape) * 2.0).to(torch.float8_e4m3fn)
            xb = x8.view(torch.uint8).clone()
            xb[unused] = 0x7F  # e4m3fn NaN
            return xb, xb.view(torch.float8_e4m3fn).float(), make_descale(), None, None
        x = torch.randn(shape)  # the "real" KV the cache quantizes
        if kv_format == "nvfp4":
            codes, sc, gs, vals, desc = nvfp4_quantize(x)
            codes[unused] = 7  # +6.0
            sc = sc.view(torch.uint8).clone()
            sc[unused] = 0x7E  # e4m3 448
            vals = vals.clone()
            vals[unused] = float("nan")  # never read by the reference
            return pack_subbyte(codes, 4), vals, torch.tensor([desc]), sc, x
        amax = x.abs().amax(dim=(0, 1, 3)).clamp(min=1e-6)  # per kv head
        if kv_format == "fp8":
            kd = amax / 448.0
            x8 = (x / kd.view(1, 1, -1, 1)).to(torch.float8_e4m3fn)
            xb = x8.view(torch.uint8).clone()
            xb[unused] = 0x7F
            return xb, xb.view(torch.float8_e4m3fn).float(), kd, None, x
        fmax = float(minifloat_table(kv_format).max())
        kd = amax / fmax
        codes, vals = minifloat_quantize(x / kd.view(1, 1, -1, 1), kv_format)
        codes[unused] = (1 << (kv_bits - 1)) - 1  # largest positive code
        return pack_subbyte(codes, kv_bits), vals, kd, None, x

    k_bytes, k_pool, k_descale, k_sc, k_true = kv_pool()
    v_bytes, v_pool, v_descale, v_sc, v_true = kv_pool()
    if from_float:
        print(
            f"\tkv: format {kv_format} ({kv_bits} bit) quantized from float data; "
            f"descale {'per-tensor (nvfp4 global)' if kv_format == 'nvfp4' else 'per kv head'}"
            f" (--descale ignored)"
        )

    q_vllm = torch.randn(num_seqs * q_len, heads_q, headdim).to(torch.bfloat16)
    if fp8:
        q_in, q_descale = quantize_q_fp8(q_vllm, num_seqs, q_len, heads_k)
    else:
        q_in, q_descale = q_vllm, torch.ones(1)  # per-tensor 1.0
    q_packed = pack_q(q_in, num_seqs, q_len, heads_k).contiguous()

    # Device tensors. Q/O optionally live inside wider rows (like vLLM's
    # fused-QKV `query` view); the KV cache optionally uses another vLLM
    # physical layout — the kernel sees only strided logical views.
    row = heads_k * n_q * headdim
    pad = (2 * heads_k * headdim) if strided_qo else 0

    def strided_rows(src_packed, dtype):
        buf = torch.zeros(num_seqs, row + pad, dtype=dtype, device=dev)
        view = buf[:, :row].view(num_seqs, heads_k, n_q, headdim)
        if src_packed is not None:
            if src_packed.dtype == torch.float8_e4m3fn:
                view.view(torch.uint8).copy_(src_packed.view(torch.uint8).to(dev))
            else:
                view.copy_(src_packed.to(dev))
        return view

    q_d = strided_rows(q_packed, q_packed.dtype)
    # KV bytes in the chosen physical layout; the kernel sees logical
    # (pages, PS, h_k, row) views. fp8 views are retyped e4m3, sub-byte
    # formats stay uint8 byte views (strides converted to elements).
    rb = row_bytes
    if kv_layout == "nhd":  # separate (pages, PS, h_k, row) K and V tensors
        k_d, v_d = k_bytes.to(dev), v_bytes.to(dev)
    elif kv_layout == "hnd":  # head-major: (pages, 2, h_k, PS, row)
        buf = torch.empty(num_pages, 2, heads_k, PS, rb, dtype=torch.uint8, device=dev)
        buf[:, 0] = k_bytes.permute(0, 2, 1, 3).to(dev)
        buf[:, 1] = v_bytes.permute(0, 2, 1, 3).to(dev)
        k_d, v_d = buf[:, 0].transpose(1, 2), buf[:, 1].transpose(1, 2)
    elif kv_layout == "interleaved":  # vLLM main FA: (pages, h_k, PS, 2*row)
        buf = torch.empty(num_pages, heads_k, PS, 2 * rb, dtype=torch.uint8, device=dev)
        buf[..., :rb] = k_bytes.permute(0, 2, 1, 3).to(dev)
        buf[..., rb:] = v_bytes.permute(0, 2, 1, 3).to(dev)
        kv = buf.transpose(1, 2)  # (pages, PS, h_k, 2*row)
        k_d, v_d = kv[..., :rb], kv[..., rb:]
    elif kv_layout == "flashinfer":  # FlashInfer NHD: (pages, 2, PS, h_k, row)
        buf = torch.empty(num_pages, 2, PS, heads_k, rb, dtype=torch.uint8, device=dev)
        buf[:, 0] = k_bytes.to(dev)
        buf[:, 1] = v_bytes.to(dev)
        k_d, v_d = buf[:, 0], buf[:, 1]
    else:
        raise ValueError(f"unknown kv_layout {kv_layout}")
    if kv_bits == 8:
        k_d, v_d = k_d.view(torch.float8_e4m3fn), v_d.view(torch.float8_e4m3fn)
    if kv_format == "nvfp4":
        ks_d = k_sc.to(dev).view(torch.float8_e4m3fn)
        vs_d = v_sc.to(dev).view(torch.float8_e4m3fn)
    else:  # unused by the kernel; any valid e4m3 pointer
        ks_d = vs_d = torch.zeros(1, 1, 1, NVFP4_GROUP, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    bt_d = block_table.to(dev)
    sl_d = seq_lens.to(dev)
    work_d = plan.work_tensor(dev)
    ns_d = plan.n_splits_tensor(dev)
    qd_d = q_descale.float().contiguous().to(dev)
    kd_d = k_descale.float().contiguous().to(dev)
    vd_d = v_descale.float().contiguous().to(dev)
    o_d = strided_rows(None, torch.bfloat16)
    strides = kernel_strides(
        q_d,
        o_d,
        k_d,
        v_d,
        qd_d,
        kd_d,
        vd_d,
        kv_bits=kv_bits,
        k_scales=ks_d if kv_format == "nvfp4" else None,
        v_scales=vs_d if kv_format == "nvfp4" else None,
    )
    print(
        f"\tlayout: kv={kv_layout} strided_qo={strided_qo} descale={descale}; "
        f"strides={dict(zip(STRIDE_FIELDS, strides))}"
    )
    lse_d = torch.empty(num_seqs, heads_k, n_q, dtype=torch.float32, device=dev)
    # Workspace: deliberately NOT initialized (the kernel must not need it)
    op_d = torch.full((max_splits, num_seqs, heads_k, n_q, headdim), float("nan"), device=dev)
    mp_d = torch.full((max_splits, num_seqs, heads_k, n_q), float("nan"), device=dev)
    lp_d = torch.full((max_splits, num_seqs, heads_k, n_q), float("nan"), device=dev)

    def kv_ptr(t):
        if kv_bits == 8:
            return _to_cute(t).iterator
        from cutlass.cute.runtime import make_ptr  # sub-byte element pointer

        return make_ptr(kv_dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    kv_ptrs = [kv_ptr(k_d), kv_ptr(v_d), _to_cute(ks_d).iterator, _to_cute(vs_d).iterator]
    cts = [
        _to_cute(q_d),
        _to_cute(bt_d),
        _to_cute(sl_d),
        _to_cute(work_d),
        _to_cute(ns_d),
        _to_cute(qd_d),
        _to_cute(kd_d),
        _to_cute(vd_d),
        _to_cute(o_d),
        _to_cute(lse_d),
        _to_cute(op_d),
        _to_cute(mp_d),
        _to_cute(lp_d),
    ]
    stream = cutlass_torch.default_stream()
    args = (
        (num_seqs, heads_k, n_q, headdim),
        max_blocks,
        num_pages,
        max(1, plan.num_work),
        strides,
        cts[0].iterator,
        *kv_ptrs,
        *[t.iterator for t in cts[1:]],
        scale_s,
        scale_o,
        stream,
    )
    compiled = cute.compile(fmha, *args, options="--opt-level 2")
    print("Finished Compiling")

    if not skip_ref_check:
        compiled(*args)
        torch.cuda.synchronize()
        print("Verifying results...")
        o_emu, lse_emu, o_exact = emulate_decode(
            q_packed.float(),
            k_pool.float(),
            v_pool.float(),
            block_table,
            seq_lens,
            expand_descale(q_descale.float()),
            expand_descale(k_descale.float()),
            expand_descale(v_descale.float()),
            scale_s,
            scale_o,
            q_len,
            plan.chunk_tiles,
            fp8,
        )
        o_kern = o_d.float().cpu()
        lse_kern = lse_d.cpu()
        if from_float:
            # what the cache format costs: kernel vs attention over the
            # UNQUANTIZED float KV (same q, P rounding aside)
            ones = torch.ones(num_seqs, heads_k)
            _, _, o_true = emulate_decode(
                q_packed.float(),
                k_true,
                v_true,
                block_table,
                seq_lens,
                expand_descale(q_descale.float()),
                ones,
                ones,
                scale_s,
                scale_o,
                q_len,
                plan.chunk_tiles,
                fp8,
            )
            d_true = (o_kern - o_true).abs()
            rel = (d_true.norm() / o_true.norm().clamp(min=1e-12)).item()
            print(
                f"kv quantization error ({kv_format}): |kern-fp32 unquantized| "
                f"max={d_true.max().item():.4g} mean={d_true.mean().item():.4g} "
                f"rel_l2={rel:.4g}"
            )
        d_emu = (o_kern - o_emu).abs()
        d_exact = (o_kern - o_exact).abs()
        d_lse = (lse_kern - lse_emu).abs()
        print(
            f"check: |kern-emu| max={d_emu.max().item():.4g} mean={d_emu.mean().item():.4g}; "
            f"|lse-emu| max={d_lse.max().item():.4g}; "
            f"|kern-exact fp32| max={d_exact.max().item():.4g} "
            f"(P rounding to {'e4m3' if fp8 else 'bf16'})"
        )
        if torch.isnan(o_kern).any():
            raise AssertionError("NaN in output (stray read of a poisoned page?)")
        torch.testing.assert_close(o_kern, o_emu, atol=tolerance, rtol=2e-2)
        torch.testing.assert_close(lse_kern, lse_emu, atol=1e-2, rtol=1e-3)
        # vLLM-layout round trip sanity
        _ = unpack_o(o_d, num_seqs, q_len, heads_k)

    exec_us = 0.0
    if iterations > 0:
        # Cold L2 (32 MiB on Thor): flush between timed launches, unless an
        # external profiler controls the cache (ncu --cache-control all) —
        # then the flush kernel is just noise in the profile.
        flush = None
        if not no_l2_flush:
            flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
        for _ in range(warmup_iterations):
            compiled(*args)
        times = []
        for _ in range(iterations):
            if flush is not None:
                flush.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            compiled(*args)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        times.sort()
        exec_us = times[len(times) // 2]
        kv_bytes = int(seq_lens.sum()) * heads_k * headdim * 2  # K + V, 1 B each
        io_bytes = q_d.numel() * q_d.element_size() + o_d.numel() * 2
        gbps = (kv_bytes + io_bytes) / (exec_us * 1e-6) / 1e9
        cache = "hot/unflushed L2" if flush is None else "cold L2"
        if os.environ.get("CUDA_INJECTION64_PATH"):
            # ncu / nsys inject a library and serialize + replay every launch:
            # host-side event timing then measures the profiler, not the kernel.
            print(
                f"median {exec_us:.1f} us host-timed UNDER A PROFILER (includes "
                f"replay/flush overhead) — use the profiler's gpu__time_duration"
            )
        else:
            print(f"median {exec_us:.2f} us, {gbps:.3g} GB/s ({cache}, {kv_bytes + io_bytes} B of KV+Q+O)")

    return exec_us


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paged fp8 decode attention (sm_110).")
    parser.add_argument("--batches", "--batch", "--b", "--num_seqs", type=int, default=1)
    parser.add_argument(
        "--seqlen", "--seqlen_k", "--s", type=int, default=1024, help="(max) kv sequence length incl. new tokens"
    )
    parser.add_argument("--varlen", action="store_true", help="random per-sequence lengths in [q_len, seqlen]")
    parser.add_argument("--q_len", type=int, default=1, help="query tokens per sequence (spec decode)")
    parser.add_argument("--heads_q", "--h_q", type=int, default=24)
    parser.add_argument("--heads_k", "--h_k", type=int, default=4)
    parser.add_argument("--headdim", "--d", type=int, default=256)
    parser.add_argument("--page_size", type=int, default=128)
    parser.add_argument(
        "--ctas_per_sm",
        type=int,
        default=2,
        choices=(1, 2),
        help="co-resident CTAs per SM (2 = 40 CTAs/wave, measured best on Thor with the default ring depth)",
    )
    parser.add_argument(
        "--kv_stages",
        type=int,
        default=0,
        help="KV ring depth (0 = default: 1 KV tile at 2 CTAs/SM, 2 tiles at 1 CTA/SM)",
    )
    parser.add_argument(
        "--contiguous_pages", action="store_true", help="identity-ordered block table (default: shuffled pages)"
    )
    parser.add_argument(
        "--kv_splits",
        "--splits",
        type=int,
        default=0,
        help="force N contiguous chunks per sequence (0 = occupancy-pinned plan)",
    )
    parser.add_argument("--max_splits", type=int, default=16, help="partial-slot capacity per sequence (compile-time)")
    parser.add_argument(
        "--pad_work_to", type=int, default=0, help="pad the work list to N items (fixed grid, CUDA graphs)"
    )
    parser.add_argument(
        "--kv_layout",
        default=HARNESS_KV_LAYOUT,
        choices=("nhd", "hnd", "interleaved", "flashinfer"),
        help="physical KV cache layout (vLLM NHD, head-major HND "
        "(pages, 2, h_k, PS, d), vLLM-main interleaved K|V, "
        f"FlashInfer NHD); default {HARNESS_KV_LAYOUT}",
    )
    parser.add_argument(
        "--kv_format",
        default="auto",
        choices=("auto",) + KV_FORMATS,
        help="KV cache format (fp8-Q path for non-fp8): e4m3; fp6 "
        "e3m2/e2m3 or fp4 e2m1 read directly by the MMA (per "
        "kv-head descale); nvfp4 (e2m1 + e4m3 per-16 scales, "
        "converted to e4m3 in-kernel); default auto = "
        f"{HARNESS_KV_FORMAT_FP8Q} for fp8 Q, fp8 for bf16 Q",
    )
    parser.add_argument(
        "--kv_from_float",
        action="store_true",
        help="fp8: quantize KV from float data with per-head "
        "descales and report the quantization error (always "
        "on for the other formats) — the accuracy A/B baseline",
    )
    parser.add_argument(
        "--strided_qo", action="store_true", help="Q/O rows inside wider rows (like vLLM's fused-QKV view)"
    )
    parser.add_argument(
        "--pdl",
        action="store_true",
        help="programmatic dependent launch (decode waits on the previous "
        "kernel, reduction launches early); time with --iterations/nsys, "
        "ncu serializes kernels and hides the effect",
    )
    parser.add_argument(
        "--l2_prefetch", type=int, default=0, help="TMA-prefetch K/V this many 128-row tiles ahead into L2 (0 = off)"
    )
    parser.add_argument(
        "--kv_evict_first",
        type=int,
        nargs="?",
        const=1,
        default=-1,
        choices=(-1, 0, 1),
        help="TMA K/V loads with an L2 EVICT_FIRST cache policy; "
        "default auto = on for fp8 Q, off for bf16 Q (the vLLM "
        "defaults); 0/1 force it",
    )
    parser.add_argument(
        "--descale",
        default="per_seq_head",
        choices=("per_seq_head", "per_head", "per_tensor"),
        help="K/V descale granularity (vLLM default: per_tensor)",
    )
    parser.add_argument("--q_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--kv_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--tolerance", type=float, default=0.05, help="atol vs the bit-faithful emulation")
    parser.add_argument("--scale_s", "--scale", type=float, default=0.0)
    parser.add_argument("--scale_o", type=float, default=1.0)
    parser.add_argument("--warmup_iterations", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument(
        "--no_l2_flush",
        action="store_true",
        help="don't flush L2 between timed launches (use under ncu --cache-control all)",
    )
    # Accepted for old command lines; paging is always on now.
    parser.add_argument("--paged", action="store_true", help=argparse.SUPPRESS)

    kwargs = vars(parser.parse_args())
    if kwargs["kv_format"] == "auto":  # sub-8-bit formats need fp8 Q
        kwargs["kv_format"] = HARNESS_KV_FORMAT_FP8Q if kwargs["q_dtype"] is cutlass.Float8E4M3FN else "fp8"
    run(**kwargs)

    print("PASS")
