# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Mieszko Syty (Mjolnir modifications)
# SPDX-License-Identifier: BSD-3-Clause
#
# Derived from NVIDIA CUTLASS, examples/python/CuTeDSL/cute/blackwell/kernel/
# attention/mixed_input_fmha/mixed_input_fmha_decode.py; this file as a whole
# stays under the BSD-3-Clause license below (see NOTICE).

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
attend to keys 0 .. L - q_len + qi (causal tail). With ``causal=False`` every
query token sees all L keys — the block attention of parallel drafters
(DSpark / DFlash: one bonus query and K mask queries attend to the context
and to each other).

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

from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05

from mjolnir.quant import NVFP4_GROUP, NVFP4_SHIFT
from mjolnir.t5_call import _CallMixin
from mjolnir.t5_const import (
    FP8_P_SCALE,
    _note,
    cpy_dice,
    exp2,
    fadd2,
    ffma2,
    fmul2,
    log2_e,
    mma_dice,
    mma_modes,
    smem_fmax,
    swizzle_align,
    warp_fmax,
    warp_threads,
    warpgroup_threads,
    warpgroup_warps,
)
from cutlass.cute.typing import *

# Compile-time notes (stages, tmem use): printed by the standalone harness
# (__main__), quiet inside vLLM unless MJOLNIR_ATTN_VERBOSE=1.

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


class MixedInputFusedMultiHeadAttentionDecode(_CallMixin):
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
        max_kv_stages=0,  # KV ring depth; 0 = whole KV tiles (1 tile at 2/SM, 2 at 1/SM; fp6 at 2/SM d=256: 2)
        pdl=False,  # programmatic dependent launch for decode + reduction
        l2_prefetch=0,  # TMA-prefetch K/V tiles this many tiles ahead into L2 (0 = off)
        kv_evict_first=False,  # TMA K/V loads with an L2 EVICT_FIRST cache policy
        kv_format="fp8",  # KV cache format, see KV_FORMATS
        causal=True,  # False: every query token sees all seq_len keys (parallel-drafter blocks)
        fused_merge=False,  # E1: last CTA per (seq, q tile, kv head) merges the splits in-kernel
    ):
        self.causal = bool(causal)
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
        self.fused_merge = bool(fused_merge)
        if self.fused_merge:
            assert max_splits > 1, "fused_merge needs split capacity (max_splits > 1)"
            assert hasattr(cute.arch, "atomic_add"), "fused_merge needs cute.arch.atomic_add"
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

    ##############################################################################
    # Host helpers (also what a vLLM backend would call)
    ##############################################################################

    # ----------------------------------------------------------------------------

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
        num_seqs: cutlass.Int32,
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
        MAX_SPLITS = self.max_splits  # plain int: loop bounds in the fused merge
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
        merge_flag_ptr = smem.allocate_array(cutlass.Int32, 1)  # E1 last-CTA election flag

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

        _note(f"\t{tmem_offset} tmem cols used, {self.tmem_alloc_cols} tmem cols allocated")
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
            # keys 0 .. seq_len - q_len + qi. Non-causal: every column sees
            # keys 0 .. seq_len - 1 (the q_len == 1 mask below).
            if cutlass.const_expr(self.q_len > 1 and self.causal):
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
                if cutlass.const_expr(self.q_len == 1 or not self.causal):
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

                # E1 fused last-CTA merge: after every softmax thread issued its
                # partial stores, one elected thread increments this group's
                # counter (acq_rel, gpu scope: publishes the stores, acquires
                # the other CTAs'); the last CTA combines all splits' partials
                # for its rows and writes the final O + LSE itself — no
                # reduction launch, one DRAM round trip (FlashInfer D128
                # producer structure; the counter self-reset keeps it
                # graph-safe, and the next grid's griddepcontrol_wait orders
                # after this grid completes).
                if cutlass.const_expr(self.fused_merge):
                    softmax_nbar.arrive_and_wait()
                    cnt_idx = num_seqs + (coord_hr * h_k + h_k_idx) * num_seqs + b_idx
                    if warpgroup_tidx == 0:
                        merge_flag_ptr[0] = 0
                        old = cute.arch.atomic_add(
                            mNSplits.iterator + mNSplits.layout(cnt_idx),
                            cutlass.Int32(1),
                            sem="acq_rel",
                            scope="gpu",
                        )
                        if old == n_splits_seq - 1:
                            merge_flag_ptr[0] = 1
                            mNSplits[cnt_idx] = 0  # self-reset for the next launch
                    softmax_nbar.arrive_and_wait()
                    if merge_flag_ptr[0]:
                        for i in cutlass.range_constexpr(cute.size(tSrO[cpy_dice + (0, 0)])):
                            if n_base + i < n_q:
                                m_max = -cutlass.Float32.inf
                                for j in cutlass.range_constexpr(MAX_SPLITS):
                                    if j < n_splits_seq:
                                        m_max = cute.arch.fmax(m_max, mM_partial[n_base + i, coord_hb, j])
                                l_acc = cutlass.Float32(0)
                                o_acc = cute.make_rmem_tensor((tiles_dm,), cutlass.Float32)
                                for dm in cutlass.range_constexpr(tiles_dm):
                                    o_acc[dm] = cutlass.Float32(0)
                                for j in cutlass.range_constexpr(MAX_SPLITS):
                                    if j < n_splits_seq:
                                        m_j = mM_partial[n_base + i, coord_hb, j]
                                        w = cutlass.Float32(0)
                                        if m_j != -cutlass.Float32.inf:
                                            w = exp2(log2_e * (m_j - m_max))
                                        l_acc += w * mL_partial[n_base + i, coord_hb, j]
                                        for dm in cutlass.range_constexpr(tiles_dm):
                                            d_idx = dm * mma_tile_m + warpgroup_tidx
                                            o_acc[dm] += w * mO_partial[d_idx, n_base + i, coord_hb, j]
                                inv_l_m = cutlass.Float32(0)
                                if l_acc > cutlass.Float32(0):
                                    inv_l_m = 1.0 / l_acc
                                for dm in cutlass.range_constexpr(tiles_dm):
                                    d_idx = dm * mma_tile_m + warpgroup_tidx
                                    mO[d_idx, n_base + i, coord_hb] = mO.element_type(scale_o * o_acc[dm] * inv_l_m)
                                if warpgroup_tidx == 0:
                                    mLSE[n_base + i, coord_hb] = m_max + cute.math.log(l_acc, fastmath=False)

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


if __name__ == "__main__":  # the standalone harness lives in mjolnir.t5_harness
    from mjolnir.t5_harness import main

    main()
