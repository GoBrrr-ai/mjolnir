"""Pure-FMA GEMV decode kernel for head_dim=256, M<=8 — the Orin (``sm_87``)
kernel family. TARGETS SM_87 ONLY: the runner and the vLLM backend's platform
guard never instantiate it on any other compute capability.

Restored from the ``gemv-archive`` git tag (2026-10-22,
``deprecated/gemv/kernel/sm100_hd256_decode_gemv.py``, tag commit ``6989e30``)
— a CuTe-DSL port of the FlashInfer fast-decode GEMV kernel
(``flashinfer/attention/decode.cuh``, ``use_tensor_cores=False`` path) — and
re-targeted at Jetson AGX Orin (GA10B): Mjolnir naming, ``MJOLNIR_ATTN_*``
knobs, and the sm_87 shared-memory budget replace the FA4/Blackwell ones
(the budget assert is in ``__init__``). The FMA compute path below is the
sm_87 *baseline variant*: the roofline for GQA g=6 at head_dim 256 asks
~93% of Orin's fp32 FMA peak just to saturate its ~205 GB/s LPDDR5x, so an
mma.sync tensor-core variant is the planned default — the loader, ring,
split-KV plan and combine below survive that refactor unchanged.

Architecture (M=1, dense/varlen/paged KV, SplitKV-capable since Phase B1):

- grid  : ``(num_kv_heads, num_splits, batch_size)`` — one CTA per
  (kv_head, kv-chunk, batch). ``num_splits = 1`` is the Phase-A serial
  form (write-through, no partials).
- block : ``16 * g`` threads (``g = qhead_per_kvhead``); ``tx = tidx % 16``
  is the 16-wide head-dim slice, ``ty = tidx // 16`` the q-head in the GQA
  group. Each 16-thread group is exactly one half-warp, so a 4-step
  butterfly shfl (offsets 1,2,4,8) reduces a 256-dot score within the group.
- state : per thread ``(m, l, O[16] fp32)`` in registers; scores are made
  group-uniform by the shfl reduce, so the epilogue needs no cross-thread
  merge — each thread writes its own 16-wide d-slice of the output row.
- QK    : 16-element FMA dot per thread + butterfly reduce, log2-domain
  online softmax (``exp2``), ``qk_descale`` folded into the score scale.
- PV    : FMA ``O += p * V``; ``v_descale`` applied once in the epilogue.
- KV    : cp.async 128-bit into an S-stage smem ring (S=16; S=2 is the
  original double buffer — see ``docs/gemv-ring-fix-bench.md``), K and V
  waited separately per iteration (FlashInfer consumer-loop order).

SplitKV (Phase B1, FlashInfer Alg.1 — scheduler.cuh): with
``num_splits > 1`` each CTA scans its chunk
``[s·ceil(kv_len/num_splits), min((s+1)·ceil(...), kv_len))`` (empty chunks
past the end take the ``kv_len == 0`` path) and emits a **chunk-normalized**
partial: ``O_partial[split] = o·v_descale / l`` (fp32) and
``LSE_partial[split] = (m + log2 l)·ln2`` (natural units). The host then
merges the partials with the in-tree ``_flash_attn_fwd_combine`` (the A1b
verified ⊕-merge): ``O = Σ_s exp(lse_s − L)·O_s``, ``L = max_s lse_s +
ln Σ_s exp(lse_s − max)``. With ``num_splits = 1`` the Phase-A epilogue
writes the final O directly (write-through, no partials/combine —
FlashInfer App. D.2).

Paged KV (Phase B1): ``page_size`` (constructor constant) is 0 for dense or
16/128 for the paged pool layout ``(num_pages, page_size, H_kv, D)``. Each
row is gathered via the int32 page table:
``page = row // page_size``, ``off = row − page·page_size``,
``phys = mPageTable[batch, page]·page_size + off`` (decode.cuh's page-table
walk; K and V of a row share the page). Tail rows are clamped to the last
valid row BEFORE the page lookup, so every cp.async stays in-bounds (the
Phase-A clamp semantics) and clamped rows are masked in compute.

Tail handling: gmem row indices are *clamped* to ``kv_len - 1`` so every
cp.async stays in-bounds and commit-group counts stay uniform (no
predicated loads); clamped rows are masked to ``-inf`` in compute (K data
unused; V data finite, weighted by ``p = 0``). A batch (chunk) with
``kv_len == 0`` writes ``O = 0`` / ``LSE = -inf``.

DSL notes: no nested functions and no early exits — the base-DSL
preprocessor rejects closure captures inside staged if/for bodies, so all
addressing is inlined. ``num_splits`` is a runtime value (the grid's y
dimension), so one split binary serves every ns; the partial-tensor
presence (``mO_partial is not None``) is the compile-time split flag.
"""

from typing import NamedTuple, Optional

import math
import os

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync

# The kernel targets sm_87 only. The vLLM backend resolves the family through
# mjolnir.platforms (which routes (8, 7) here once the runner lands) and the
# runner asserts the device before instantiating — this constant is the
# machine-readable statement of that.
TARGET_CC = (8, 7)

# GA10B (CC 8.7) max shared memory per block: 163 KiB (the CUDA guide's CC
# 8.7 column — A100-class, NOT the 100 KiB of the CC 8.6/8.9 GA10x parts),
# minus CUDA's 1 KiB reservation. FlashInfer's Ampere tuning assumes the
# 100 KB-class parts; Orin can stage ~2x the KV per iteration.
MAX_BLOCK_SMEM_SM87 = 162 * 1024


# The descales, formerly FA4's ``DescaleTensors`` (a NamedTuple of three
# optional cute tensors with a ``__new_from_mlir_values__`` tracing hook).
# Defined here because the kernel package must not import vLLM — and the FA4
# interface does not exist on Ampere anyway.
class DescaleTensors(NamedTuple):
    q_descale: Optional[cute.Tensor] = None
    k_descale: Optional[cute.Tensor] = None
    v_descale: Optional[cute.Tensor] = None

    def __new_from_mlir_values__(self, values):
        # None fields carry no MLIR value, so `values` only covers the
        # non-None fields IN ORDER — rehydrate positionally (padding at the
        # end would shift k<-v<-None and silently swap the descales).
        it = iter(values)
        return DescaleTensors(
            *(None if f is None else next(it) for f in (self.q_descale, self.k_descale, self.v_descale))
        )


def _e4m3_to_f32(x):
    """Software e4m3fn -> fp32 for sm_87.

    No fp8 cvt exists before sm_89 (and the DSL's ``.to(Float32)`` lowers to
    the SM90+ ``cvt.packfloat``), so the code byte is bit-assembled: normals
    are exact — ``(em + 120*8) << 20`` places the 3-bit exponent (fp32 bias
    127 - e4m3 bias 7 = +120, carried by the integer add) and the 3-bit
    mantissa exactly in the fp32 fields, and the sign bit reaches bit 31 via
    ``<< 24`` from bit 7. The zero code selects to 0.0; subnormal codes
    (em < 8) come out as their normal-formula neighbors — an absolute error
    < 0.008 on a |v| < 0.875 value, far inside the correctness tolerance
    (the prmt + magic-constant fp16 path is the bit-exact follow-up).
    """
    b = x.bitcast(cutlass.Uint8).to(cutlass.Uint32)
    em = b & cutlass.Uint32(0x7F)
    bits = ((b & cutlass.Uint32(0x80)) << cutlass.Uint32(24)) | ((em + cutlass.Uint32(0x3C0)) << cutlass.Uint32(20))
    v = bits.bitcast(cutlass.Float32)
    nz = (b != cutlass.Uint32(0)).to(cutlass.Uint32).to(cutlass.Float32)
    return v * nz


def _elem_to_f32(x, elem):
    """Element -> fp32: fp8 through the sm_87 software path, bf16/fp16 via HW."""
    if const_expr(elem is cutlass.Float8E4M3FN):
        return _e4m3_to_f32(x)
    return x.to(Float32)


class GEMVDecodeKernel:
    """GEMV decode kernel for head_dim=256 on Jetson Orin (``sm_87``).

    Q dtype in {bf16, fp16, e4m3}; KV dtype in {bf16, fp16, e4m3}.
    Output dtype is taken from the caller-provided ``mO`` tensor (final,
    non-split) or ``mO_partial`` (fp32, split mode).

    ``page_size`` (constructor constant): 0 = dense KV; 16 or 128 = paged
    pool ``K/V: (num_pages, page_size, H_kv, D)`` gathered through
    ``mPageTable``: ``(batch, max_pages)`` int32.
    """

    HEAD_DIM = 256
    VEC = 16  # elements per thread (128-bit for fp8 KV; 2x128-bit for 16-bit KV)

    def __init__(
        self,
        qhead_per_kvhead: int,
        tile_size_per_bdx: int = 4,
        page_size: int = 0,
        causal: bool = True,
    ):
        assert qhead_per_kvhead >= 1, "qhead_per_kvhead must be >= 1"
        assert self.HEAD_DIM % (16 * self.VEC) == 0
        assert page_size in (0, 16, 32, 64, 128), "page_size must be 0 (dense) or 16/32/64/128"
        self.g = qhead_per_kvhead
        self.page_size = page_size
        self.causal = bool(causal)
        self.bdx = self.HEAD_DIM // self.VEC  # 16 threads per KV row
        self.bdy = qhead_per_kvhead  # one q-head per (ty) thread
        # rows per (ty) thread. The tile-size knob (default: the constructor
        # value, 4 — the s4t4 commit-coarsening; in-flight = 4 stages * 6 *
        # 4 * 256 * 2 = 48 KiB per K/V buffer, same as the shipped t1ns20
        # but with 4x coarser commits).
        _tile_env = int(os.environ.get("MJOLNIR_ATTN_TILE", "0") or 0)
        self.tile = _tile_env if _tile_env > 0 else tile_size_per_bdx
        # KV ring depth (in-flight stages). Same semantics as the Thor
        # ring-depth knob (MJOLNIR_ATTN_KV_STAGES; 0 = the tuned default,
        # 4 here — the s4t4 commit-coarsening: 4.4% faster at 32K/64K, 0.5%
        # faster at 8K; the L2 sectors are identical, the gain is purely
        # from fewer commit points: 8 vs 32). The rolling
        # cp_async_wait_group(2*S-1) loop and the S-stage prologue preload
        # are S-agnostic (verified: only this value + smem shapes change);
        # S*bdy*tile = resident rows, 2*S*R*D*elemsize = the ring bytes.
        _stages_env = int(os.environ.get("MJOLNIR_ATTN_KV_STAGES", "0") or 0)
        self.stages = _stages_env if _stages_env > 0 else 4
        self.rows_per_stage = self.bdy * self.tile  # R
        self.num_threads = self.bdx * self.bdy
        # sm_87 budget: the ring is TWO smem buffers of S*R*D elements (K
        # and V each — the allocator in ``kernel``). This checks the worst
        # case (2-byte elements); fp8/fp6 KV (1 byte) allows twice the
        # stages at the same tile. MAX_BLOCK_SMEM_SM87 is the GA10B 100 KiB
        # opt-in budget minus CUDA's 1 KiB reservation.
        ring_bytes = 2 * self.stages * self.rows_per_stage * self.HEAD_DIM * 2
        assert ring_bytes <= MAX_BLOCK_SMEM_SM87, (
            f"ring {ring_bytes} B (stages={self.stages}, rows_per_stage="
            f"{self.rows_per_stage}, 2B worst case) exceeds the sm_87 block "
            f"budget ({MAX_BLOCK_SMEM_SM87} B) — reduce MJOLNIR_ATTN_KV_STAGES "
            "or MJOLNIR_ATTN_TILE (fp8/fp6 KV may double the stages)"
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor] = None,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mDescale: Optional[DescaleTensors] = None,
        mO_partial: Optional[cute.Tensor] = None,
        mLSE_partial: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        seqlen_q: cutlass.Constexpr[int] = 1,
        seqlen_k: Int32 = 0,
        softmax_scale_log2: Float32 = 0.0,
        num_splits: Int32 = 1,
        # Always keep stream as the last parameter (EnvStream: obtained
        # implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        is_varlen_q = const_expr(mCuSeqlensQ is not None)
        if const_expr(is_varlen_q):
            batch_size = mCuSeqlensQ.shape[0] - 1
        else:
            batch_size = mQ.shape[0]
        num_kv_heads = mV.shape[-2]
        assert const_expr((mO_partial is None) == (mLSE_partial is None))
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
            grid=[num_kv_heads, num_splits, batch_size],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mDescale: Optional[DescaleTensors],
        mO_partial: Optional[cute.Tensor],
        mLSE_partial: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        seqlen_q: cutlass.Constexpr[int],
        seqlen_k: Int32,
        softmax_scale_log2: Float32,
        num_splits: Int32,
    ):
        # M (seqlen_q) is a compile-time value: it sizes the per-thread register
        # state (Q/O/M*VEC) and bounds the unrolled M-loops in the compute. One
        # binary per M (the dispatch bakes M into the compile-cache key). M=1
        # sizes the state exactly as before (M*VEC == VEC) — bit-identical to the
        # Phase A/B1 M=1 decode; M>1 is the MTP verify extension (causal tail
        # per the t5 contract when self.causal).
        M = seqlen_q
        D = self.HEAD_DIM
        VEC = self.VEC
        BDX = self.bdx
        TILE = self.tile
        S = self.stages
        R = self.rows_per_stage
        PS = self.page_size

        IS_SPLIT = const_expr(mO_partial is not None)
        PAGED = const_expr(mPageTable is not None)

        kv_head = cute.arch.block_idx()[0]
        split_idx = cute.arch.block_idx()[1]
        batch_idx = cute.arch.block_idx()[2]
        tidx = cute.arch.thread_idx()[0]
        tx = tidx % BDX
        ty = tidx // BDX
        qo_head = kv_head * self.g + ty

        is_varlen_q = const_expr(mCuSeqlensQ is not None)
        is_varlen_k = const_expr(mCuSeqlensK is not None)

        # Per-batch bases / lengths.
        if const_expr(is_varlen_q):
            num_head = mQ.shape[1]
            q_row_base = Int32(mCuSeqlensQ[batch_idx])  # global row of the query
        else:
            num_head = mQ.shape[2]
            q_row_base = batch_idx * seqlen_q
        if const_expr(is_varlen_k):
            kv_row0 = Int32(mCuSeqlensK[batch_idx])  # global KV row of the batch
            kv_len = Int32(mCuSeqlensK[batch_idx + 1]) - kv_row0
        else:
            kv_row0 = batch_idx * seqlen_k
            kv_len = seqlen_k
        kv_row_stride = mV.shape[-2] * D  # elements per KV row (all heads)
        num_head_kv = mV.shape[-2]

        # ---- SplitKV: this CTA's KV chunk (identity at num_splits=1) --------
        kv_chunk = (kv_len + num_splits - 1) // num_splits
        kv_start = split_idx * kv_chunk
        # Cap by the chunk length: without the min(), every non-last split
        # over-scans to the end of the sequence (kv_len - kv_start > kv_chunk),
        # double-counting rows. ns=1 is unaffected (kv_chunk == kv_len).
        kv_len_c = cutlass.min(kv_chunk, cutlass.max(Int32(0), kv_len - kv_start))

        # Per-(batch, kv_head) descales (identity when absent). Index with a
        # 2D tuple coord, NOT a flat scalar: the descale memref is
        # (batch, H_kv) and a scalar coord into a multi-rank tensor does not
        # land at the flat offset (same class of bug as the LSE coord).
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

        # O row offset (the Q load computes its per-token offset inline). In
        # split mode the O/LSE epilogue targets mO_partial/mLSE_partial at the
        # same flat row offset plus the split block (num_splits * rows * H * D
        # elems).
        o_off = q_row_base * num_head * D + qo_head * D + tx * VEC
        if const_expr(IS_SPLIT):
            # Stride of one split block = number of elements in mO_partial with
            # its leading (split) mode removed. NOT shape[0]*shape[1]*... (that
            # folds the split count into the stride and over-strides by
            # num_splits). Dense (ns,B,Sq,H,D) -> B*Sq*H*D;
            # varlen (ns,total_q,H,D) -> total_q*H*D.
            if const_expr(is_varlen_q):
                split_stride = mO_partial.shape[1] * num_head * D
            else:
                split_stride = mO_partial.shape[1] * mO_partial.shape[2] * num_head * D
            o_off = o_off + split_idx * split_stride
        # LSE coord (NOT a flat scalar — the DSL's scalar coord into a
        # multi-rank memref does not land where the flat offset says): it is
        # computed per token (lse_c_m) in the empty-chunk and epilogue branches
        # below, with the seqlen (token) sub-mode advancing by m.

        # ---- smem: 2-stage double buffer, K and V (flat, row-major) -------
        smem = cutlass.utils.SmemAllocator()
        kv_elems = S * R * D
        k_smem = smem.allocate_tensor(mK.element_type, cute.make_layout(kv_elems), byte_alignment=16)
        v_smem = smem.allocate_tensor(mV.element_type, cute.make_layout(kv_elems), byte_alignment=16)

        k_g2s_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            mK.element_type,
            num_bits_per_copy=128,
        )
        v_g2s_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            mV.element_type,
            num_bits_per_copy=128,
        )
        s2r_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mK.element_type, num_bits_per_copy=128)
        v_s2r_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mV.element_type, num_bits_per_copy=128)
        q_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mQ.element_type, num_bits_per_copy=128)
        # O target dtype: the final O (mO) or the fp32 partial (mO_partial).
        # A 128-bit copy moves 128//width elements (4 for fp32 partials).
        o_elem = mO_partial.element_type if const_expr(IS_SPLIT) else mO.element_type
        o_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), o_elem, num_bits_per_copy=128)

        # Each copy atom moves 128 bits per instruction. For 16-bit dtypes
        # that is 8 elements, so a 16-element VEC tile needs N_SUB sub-copies
        # (8-bit: EPC=16 -> N_SUB=1, a single copy as before). The DSL does
        # NOT auto-tile an atom across a larger fragment (verified: a 128-bit
        # atom over a 16-elem bf16 tile copies only the first 8 elements).
        EPC_K = 16 if mK.element_type.width == 8 else 8
        N_SUB_K = VEC // EPC_K
        EPC_V = 16 if mV.element_type.width == 8 else 8
        N_SUB_V = VEC // EPC_V
        EPC_Q = 16 if mQ.element_type.width == 8 else 8
        N_SUB_Q = VEC // EPC_Q
        EPC_O = 128 // o_elem.width
        N_SUB_O = VEC // EPC_O

        # ---- gmem row offset for a clamped local row -----------------------
        # Dense: batch row (kv_row0 + kv_start + crow) in the (B, L) row space.
        # Paged: sequence row (kv_start + crow) -> page table -> pool row.
        # Both must include the split's chunk start (kv_start); ns=1 has
        # kv_start=0 so the omission was invisible in Phase A.
        # (Inlined at each of the 4 cp.async sites — no nested helpers.)
        # ---- empty sequence / empty chunk: O = 0, LSE = -inf ----------------
        # No early exit (the DSL forbids it in kernels) — branch instead.
        if kv_len_c == 0:
            o_frag = cute.make_rmem_tensor(VEC, o_elem)
            o_frag.fill(0.0)
            row_stride = num_head * D
            for m in cutlass.range_constexpr(M):
                o_off_m = o_off + m * row_stride
                for sub in cutlass.range_constexpr(N_SUB_O):
                    o_sub = cute.make_rmem_tensor(EPC_O, o_elem)
                    for i in cutlass.range_constexpr(EPC_O):
                        o_sub[i] = o_frag[sub * EPC_O + i]
                    o_dst = cute.make_tensor(
                        cute.make_ptr(
                            o_elem,
                            (
                                (mO_partial.iterator if const_expr(IS_SPLIT) else mO.iterator) + o_off_m + sub * EPC_O
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(EPC_O),
                    )
                    cute.copy(o_atom, o_sub, o_dst)
                if tx == 0:
                    if const_expr(mLSE is not None or mLSE_partial is not None):
                        lse_tensor = mLSE_partial if const_expr(IS_SPLIT) else mLSE
                        if const_expr(IS_SPLIT and is_varlen_q):
                            lse_c_m = (split_idx, qo_head, q_row_base + m)
                        elif const_expr(IS_SPLIT):
                            lse_c_m = (split_idx, batch_idx, qo_head, m)
                        elif const_expr(is_varlen_q):
                            lse_c_m = (qo_head, q_row_base + m)
                        else:
                            lse_c_m = (batch_idx, qo_head, m)
                        lse_tensor[lse_c_m] = -Float32.inf
        else:
            # ---- Q: load this thread's d-slice of its q-head, for all M -----
            # tokens (M=1 -> the single decode row; M>1 -> the MTP verify,
            # causal tail per the t5 contract when self.causal). token
            # m sits at global row (q_row_base + m): dense q_row_base =
            # batch_idx*M, varlen = cu_seqlens_q[batch_idx] (M=1 there).
            q_f32 = cute.make_rmem_tensor(M * VEC, Float32)
            for m in cutlass.range_constexpr(M):
                q_off_m = (q_row_base + m) * num_head * D + qo_head * D + tx * VEC
                for sub in cutlass.range_constexpr(N_SUB_Q):
                    q_src = cute.make_tensor(
                        cute.make_ptr(
                            mQ.element_type,
                            (mQ.iterator + q_off_m + sub * EPC_Q).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(EPC_Q),
                    )
                    q_sub = cute.make_rmem_tensor(EPC_Q, mQ.element_type)
                    cute.copy(q_atom, q_src, q_sub)
                    for i in cutlass.range_constexpr(EPC_Q):
                        q_f32[m * VEC + sub * EPC_Q + i] = _elem_to_f32(q_sub[i], mQ.element_type)

            # ---- register state (per token; M=1 == the Phase A/B1 M=1) -------
            # o_f32/q_f32 are M*VEC (M=1 -> VEC, unchanged); m_run/l_run/s_vec
            # carry the per-token online-softmax state (M=1 -> one element each).
            o_f32 = cute.make_rmem_tensor(M * VEC, Float32)
            o_f32.fill(0.0)
            s_vec = cute.make_rmem_tensor(M * R, Float32)
            m_run = cute.make_rmem_tensor(M, Float32)
            # Finite sentinel, not -inf: a token whose rows are ALL causally
            # masked in this chunk (short sequence, deep split) keeps m_run
            # here, and exp2(m_prev - m_run) must not hit -inf - -inf = NaN.
            # exp2(-1e30 - finite) is still exactly 0, so visible rows behave
            # bit-identically to the -inf init.
            m_run.fill(Float32(-1e30))
            l_run = cute.make_rmem_tensor(M, Float32)
            l_run.fill(0.0)

            # ---- prologue: preload the two stages ----------------------------
            # Row indices are clamped to kv_len_c-1 (in-bounds tail); the
            # commits below keep the per-thread group count uniform.
            for st in cutlass.range_constexpr(S):
                # Each ty loads its own TILE row(s) into slots
                # (st*R + ty*TILE .. st*R + ty*TILE + TILE-1) — distinct slots,
                # no overwrites. (Reading later uses slot st*R + j, j in [0, R).)
                for j in cutlass.range_constexpr(TILE):
                    row = st * R + ty * TILE + j
                    crow = row if row < kv_len_c else kv_len_c - 1
                    if const_expr(PAGED):
                        # Paged gather: page = row//PS, off = row - page*PS,
                        # phys pool row = page_table[batch, page]*PS + off.
                        grow = kv_start + crow
                        page = grow // PS
                        pidx = mPageTable[batch_idx, page]
                        kv_base = (pidx * PS + (grow - page * PS)) * num_head_kv + kv_head
                        k_off = kv_base * D + tx * VEC
                        v_off = kv_base * D + tx * VEC
                    else:
                        k_off = (kv_row0 + kv_start + crow) * kv_row_stride + kv_head * D + tx * VEC
                        v_off = (kv_row0 + kv_start + crow) * kv_row_stride + kv_head * D + tx * VEC
                    ks_off = (st * R + ty * TILE + j) * D + tx * VEC
                    for sub in cutlass.range_constexpr(N_SUB_K):
                        k_src = cute.make_tensor(
                            cute.make_ptr(
                                mK.element_type,
                                (mK.iterator + k_off + sub * EPC_K).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_K),
                        )
                        ks = cute.make_tensor(
                            cute.make_ptr(
                                mK.element_type,
                                (k_smem.iterator + ks_off + sub * EPC_K).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_K),
                        )
                        cute.copy(k_g2s_atom, k_src, ks)
                cute.arch.cp_async_commit_group()
                for j in cutlass.range_constexpr(TILE):
                    row = st * R + ty * TILE + j
                    crow = row if row < kv_len_c else kv_len_c - 1
                    if const_expr(PAGED):
                        grow = kv_start + crow
                        page = grow // PS
                        pidx = mPageTable[batch_idx, page]
                        kv_base = (pidx * PS + (grow - page * PS)) * num_head_kv + kv_head
                        v_off = kv_base * D + tx * VEC
                    else:
                        v_off = (kv_row0 + kv_start + crow) * kv_row_stride + kv_head * D + tx * VEC
                    vs_off = (st * R + ty * TILE + j) * D + tx * VEC
                    for sub in cutlass.range_constexpr(N_SUB_V):
                        v_src = cute.make_tensor(
                            cute.make_ptr(
                                mV.element_type,
                                (mV.iterator + v_off + sub * EPC_V).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_V),
                        )
                        vs = cute.make_tensor(
                            cute.make_ptr(
                                mV.element_type,
                                (v_smem.iterator + vs_off + sub * EPC_V).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_V),
                        )
                        cute.copy(v_g2s_atom, v_src, vs)
                cute.arch.cp_async_commit_group()

            # ---- main pipeline -------------------------------------------------
            num_chunks = (kv_len_c + R - 1) // R
            for it in cutlass.range(num_chunks):
                st = it % S

                # K[st] ready -> QK for all R rows of the stage, all M tokens.
                # Causal per the t5 contract (the M-loop mask below); the K
                # fragment is read from smem ONCE per row (shared by the M
                # tokens) to avoid an Mx smem re-read; the M dots reuse it.
                cute.arch.cp_async_wait_group(2 * S - 1)
                cute.arch.sync_threads()
                m_prev = cute.make_rmem_tensor(M, Float32)
                for m in cutlass.range_constexpr(M):
                    m_prev[m] = m_run[m]
                for j in cutlass.range_constexpr(R):
                    ks_off = (st * R + j) * D + tx * VEC
                    k_frag = cute.make_rmem_tensor(VEC, mK.element_type)
                    for sub in cutlass.range_constexpr(N_SUB_K):
                        k_slice = cute.make_tensor(
                            cute.make_ptr(
                                mK.element_type,
                                (k_smem.iterator + ks_off + sub * EPC_K).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_K),
                        )
                        k_sub = cute.make_rmem_tensor(EPC_K, mK.element_type)
                        cute.copy(s2r_atom, k_slice, k_sub)
                        for i in cutlass.range_constexpr(EPC_K):
                            k_frag[sub * EPC_K + i] = k_sub[i]
                    for m in cutlass.range_constexpr(M):
                        s = Float32(0.0)
                        for sub in cutlass.range_constexpr(N_SUB_K):
                            for i in cutlass.range_constexpr(EPC_K):
                                s = s + q_f32[m * VEC + sub * EPC_K + i] * _elem_to_f32(
                                    k_frag[sub * EPC_K + i], mK.element_type
                                )
                        # 16-lane butterfly reduce (offsets < 16 stay in ty group).
                        s = s + cute.arch.shuffle_sync_bfly(s, offset=1)
                        s = s + cute.arch.shuffle_sync_bfly(s, offset=2)
                        s = s + cute.arch.shuffle_sync_bfly(s, offset=4)
                        s = s + cute.arch.shuffle_sync_bfly(s, offset=8)
                        s = s * scale_eff
                        row = it * R + j
                        if const_expr(M > 1 and self.causal):
                            # The t5 contract (vLLM verify): seq_lens INCLUDE
                            # the M query tokens and row m attends to
                            # KV[0 .. kv_len - M + m] — the earlier query
                            # tokens are already in the cache, the later ones
                            # are not. The bound is the min of the loaded
                            # chunk length and the causal tail measured from
                            # the GLOBAL length (kv_len - kv_start): a
                            # non-final split's chunk is shorter than the
                            # remaining keys, and masking against kv_len_c
                            # alone would hide the chunk's last M-1 rows,
                            # while masking against the tail alone would
                            # count the clamped duplicate rows past kv_len_c.
                            s = s if row < cutlass.min(kv_len_c, kv_len - kv_start - (M - 1 - m)) else -Float32.inf
                        else:
                            s = s if row < kv_len_c else -Float32.inf
                        s_vec[m * R + j] = s
                        m_run[m] = s if s > m_run[m] else m_run[m]

                # Online rescale (log2 domain), per token. m_prev[m] - m_run[m]
                # is -inf only when m_prev[m] == -inf (first chunk) -> exp2(-inf)=0.
                for m in cutlass.range_constexpr(M):
                    o_scale = cute.math.exp2(m_prev[m] - m_run[m], fastmath=True)
                    l_run[m] = l_run[m] * o_scale
                    for i in cutlass.range_constexpr(VEC):
                        o_f32[m * VEC + i] = o_f32[m * VEC + i] * o_scale
                    for j in cutlass.range_constexpr(R):
                        p = cute.math.exp2(s_vec[m * R + j] - m_run[m], fastmath=True)
                        s_vec[m * R + j] = p
                        l_run[m] = l_run[m] + p

                # Refill K for chunk it+S into the just-consumed stage.
                cute.arch.sync_threads()
                nxt = it + S
                # The unconditional commit keeps the in-flight group count
                # uniform, so wait_group(2*S-1) stays valid for every
                # iteration including the tail (clamped rows are in-bounds).
                # Same slot mapping as the prologue: row nxt*R + ty*TILE + j
                # goes into slot st*R + ty*TILE + j of the freed stage.
                for j in cutlass.range_constexpr(TILE):
                    row = nxt * R + ty * TILE + j
                    crow = row if row < kv_len_c else kv_len_c - 1
                    if const_expr(PAGED):
                        grow = kv_start + crow
                        page = grow // PS
                        pidx = mPageTable[batch_idx, page]
                        kv_base = (pidx * PS + (grow - page * PS)) * num_head_kv + kv_head
                        k_off = kv_base * D + tx * VEC
                    else:
                        k_off = (kv_row0 + kv_start + crow) * kv_row_stride + kv_head * D + tx * VEC
                    ks_off = (st * R + ty * TILE + j) * D + tx * VEC
                    for sub in cutlass.range_constexpr(N_SUB_K):
                        k_src = cute.make_tensor(
                            cute.make_ptr(
                                mK.element_type,
                                (mK.iterator + k_off + sub * EPC_K).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_K),
                        )
                        ks = cute.make_tensor(
                            cute.make_ptr(
                                mK.element_type,
                                (k_smem.iterator + ks_off + sub * EPC_K).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_K),
                        )
                        cute.copy(k_g2s_atom, k_src, ks)
                cute.arch.cp_async_commit_group()

                # V[st] ready -> PV, all M tokens. The V fragment is read from
                # smem ONCE per row (shared by the M tokens); each token
                # accumulates p_m * V into its own o_f32[m] row.
                cute.arch.cp_async_wait_group(2 * S - 1)
                cute.arch.sync_threads()
                for j in cutlass.range_constexpr(R):
                    vs_off = (st * R + j) * D + tx * VEC
                    v_frag = cute.make_rmem_tensor(VEC, mV.element_type)
                    for sub in cutlass.range_constexpr(N_SUB_V):
                        v_slice = cute.make_tensor(
                            cute.make_ptr(
                                mV.element_type,
                                (v_smem.iterator + vs_off + sub * EPC_V).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_V),
                        )
                        v_sub = cute.make_rmem_tensor(EPC_V, mV.element_type)
                        cute.copy(v_s2r_atom, v_slice, v_sub)
                        for i in cutlass.range_constexpr(EPC_V):
                            v_frag[sub * EPC_V + i] = v_sub[i]
                    for m in cutlass.range_constexpr(M):
                        p = s_vec[m * R + j]
                        for sub in cutlass.range_constexpr(N_SUB_V):
                            for i in cutlass.range_constexpr(EPC_V):
                                d = sub * EPC_V + i
                                o_f32[m * VEC + d] = o_f32[m * VEC + d] + p * _elem_to_f32(
                                    v_frag[sub * EPC_V + i], mV.element_type
                                )

                # Refill V for chunk it+S into the just-consumed stage.
                cute.arch.sync_threads()
                for j in cutlass.range_constexpr(TILE):
                    row = nxt * R + ty * TILE + j
                    crow = row if row < kv_len_c else kv_len_c - 1
                    if const_expr(PAGED):
                        grow = kv_start + crow
                        page = grow // PS
                        pidx = mPageTable[batch_idx, page]
                        kv_base = (pidx * PS + (grow - page * PS)) * num_head_kv + kv_head
                        v_off = kv_base * D + tx * VEC
                    else:
                        v_off = (kv_row0 + kv_start + crow) * kv_row_stride + kv_head * D + tx * VEC
                    vs_off = (st * R + ty * TILE + j) * D + tx * VEC
                    for sub in cutlass.range_constexpr(N_SUB_V):
                        v_src = cute.make_tensor(
                            cute.make_ptr(
                                mV.element_type,
                                (mV.iterator + v_off + sub * EPC_V).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_V),
                        )
                        vs = cute.make_tensor(
                            cute.make_ptr(
                                mV.element_type,
                                (v_smem.iterator + vs_off + sub * EPC_V).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            cute.make_layout(EPC_V),
                        )
                        cute.copy(v_g2s_atom, v_src, vs)
                cute.arch.cp_async_commit_group()

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ---- epilogue ------------------------------------------------------
            # Non-split (write-through): O = (o/l)·v_descale in the final
            # dtype, LSE = (m+log2 l)·ln2 into mLSE.
            # Split: the SAME chunk-normalized values go to the fp32 partial
            # O_partial[split] / LSE_partial[split]; the host combine
            # (flash_fwd_combine) applies the global rescale
            # O = Σ_s exp(lse_s − L)·O_s, L = max + ln Σ exp(lse_s − max).
            o_frag = cute.make_rmem_tensor(VEC, o_elem)
            o_base = mO_partial.iterator if const_expr(IS_SPLIT) else mO.iterator
            row_stride = num_head * D  # dense stride of one token row in O/LSE
            for m in cutlass.range_constexpr(M):
                # l_run == 0 (every row masked for this token) -> O = 0,
                # not 0 * inf; the max is a no-op whenever anything was seen.
                scale_out = v_descale / cutlass.max(l_run[m], Float32(1e-30))
                for i in cutlass.range_constexpr(VEC):
                    o_frag[i] = (o_f32[m * VEC + i] * scale_out).to(o_elem)
                o_off_m = o_off + m * row_stride
                for sub in cutlass.range_constexpr(N_SUB_O):
                    o_sub = cute.make_rmem_tensor(EPC_O, o_elem)
                    for i in cutlass.range_constexpr(EPC_O):
                        o_sub[i] = o_frag[sub * EPC_O + i]
                    o_dst = cute.make_tensor(
                        cute.make_ptr(
                            o_elem,
                            (o_base + o_off_m + sub * EPC_O).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(EPC_O),
                    )
                    cute.copy(o_atom, o_sub, o_dst)
                if tx == 0:
                    if const_expr(mLSE is not None or mLSE_partial is not None):
                        lse_tensor = mLSE_partial if const_expr(IS_SPLIT) else mLSE
                        # token m's LSE coordinate: the seqlen (token) sub-mode
                        # advances by m (M=1 -> the existing 0 / q_row_base base).
                        if const_expr(IS_SPLIT and is_varlen_q):
                            lse_c_m = (split_idx, qo_head, q_row_base + m)
                        elif const_expr(IS_SPLIT):
                            lse_c_m = (split_idx, batch_idx, qo_head, m)
                        elif const_expr(is_varlen_q):
                            lse_c_m = (qo_head, q_row_base + m)
                        else:
                            lse_c_m = (batch_idx, qo_head, m)
                        lse_tensor[lse_c_m] = (m_run[m] + cute.math.log2(l_run[m], fastmath=True)) * Float32(
                            math.log(2.0)
                        )
