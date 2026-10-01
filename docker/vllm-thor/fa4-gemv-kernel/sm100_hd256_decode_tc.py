"""Tensor-core (mma.sync) M=1 hd256 decode kernel on SM100/SM110 — first correct pass.

The FMA GEMV decode kernel (``sm100_hd256_decode_gemv.py``) is compute/issue-bound
(ncu: Compute SOL 68% vs Memory SOL 26% at L=32768): its per-KV-row FMA dot + butterfly
shuffle + FMA PV caps achieved-BW ~100 GB/s regardless of L. This kernel does QK and PV
with ``mma.sync m16n8k16`` (the ``flash_fwd.py`` Sm80 mma structure) so the compute is
"free" relative to the bytes streamed — the kernel becomes a memory-bound streamer like
FlashInfer's FA2-tc decode. See ``docs/fa4-hd256-fp8/gemv-long-context-diagnosis.md``.

Shape: M=1, GQA 24:4 (group 6 q-heads per kv-head), head_dim=256. The GQA group is the
"m" (query) dimension of the mma (``PackGQA``), padded to ``TILE_M`` (64); the KV rows
are the "n" dimension (tiled by ``TILE_N``). Each CTA owns one kv_head and streams its
KV through the mma.

Scope (correctness-first): dense + varlen (cu_seqlens_k), bf16 Q + bf16 or fp8-e4m3
KV (fp8 with the per-(batch, kv_head) descales: qk_descale folds into the softmax
scale, v_descale into the epilogue — the GEMV/FI descale scheme), num_splits=1,
single-stage cp.async KV. SplitKV (long-L perf), paged KV, and the deep KV ring are
the follow-up passes.

Correctness gate: ``verify-tc-decode.py`` vs the fp32 reference (``mjolnir verify``).
"""

from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr, Float8E4M3FN
from cutlass.cute.nvgpu import cpasync, warp
from quack import layout_utils

from vllm.vllm_flash_attn.cute import ampere_helpers as sm80_utils
from vllm.vllm_flash_attn.cute import utils
from vllm.vllm_flash_attn.cute.pack_gqa import pack_gqa_layout, PackGQA
from vllm.vllm_flash_attn.cute.flash_fwd_sm100 import DescaleTensors
from vllm.vllm_flash_attn.cute.softmax import Softmax
from vllm.vllm_flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
DBG_STAGE = 0  # 0=full 1=after-Qload 2=after-KVload 3=after-QK 4=after-PV


@cute.jit
def _mask_partial_kv(
    acc_S: cute.Tensor,
    tScS: cute.Tensor,
    nb: Int32,
    kv_len: Int32,
    tile_n,
):
    """Write -inf into the OOB kv-columns of the acc_S rows, through mn-view
    row round-trips (the exact access path online_softmax consumes — a linear
    `acc_S[i]` store alone is dropped before the row-vector loads see it).
    Per-element stores are UNCONDITIONAL (a runtime-`if` around an rmem
    element store is dropped at trace); the OOB select is a runtime ternary
    (the GEMV `s = s if ... else -inf` idiom). Per-thread the mn view's
    (r, c) slot maps to tScS own-modes (r%2, r//2, c) (probe-verified; a
    coord read through the mn reshape is garbage)."""
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
    _nr = cute.size(acc_S_mn.shape[0])
    _nc = cute.size(acc_S_mn.shape[1])
    for _r in cutlass.range_constexpr(_nr):
        _row_t = cute.make_rmem_tensor(_nc, Float32)
        _row_t.store(acc_S_mn[_r, None].load())
        for _c in cutlass.range_constexpr(_nc):
            _v = _row_t[_c]
            _row_t[_c] = (
                -Float32.inf
                if nb * tile_n + tScS[(_r % 2, _r // 2), 0, _c][1] >= kv_len
                else _v
            )
        acc_S_mn[_r, None].store(_row_t.load())


@cute.jit
def _qk_mma_fp8(
    tiled_mma_qk: cute.TiledMma,
    acc_S: cute.Tensor,
    tSrQ: cute.Tensor,
    tSrK: cute.Tensor,
    tSsK: cute.Tensor,
    tSrK_cp: cute.Tensor,
    smem_thr_copy_K: cute.TiledCopy,
):
    """fp8 QK mma (Q already in regs; K read element-wise from smem).
    Mirrors sm80_utils.gemm(A_in_regs=True) but k-consistent with the fp8 mma
    (k-tile 32, D/32 blocks). The K smem->reg load is element-wise (the b16
    ldmatrix B-path faults on the fp8 B-operand on sm_110a). tSrK_cp =
    retile(tSrK) is the mma B-fragment in the COPY's value order, and tSsK is
    the smem in the same copy order, so a straight per-element copy aligns
    (a manual _e//per_k, _e%per_k decomposition does NOT — tSrK's layout
    interleaves the k-blocks, k-mode stride 32 < the d-mode stride 256)."""
    num_k = cute.size(tSsK.shape[2])
    for _e in cutlass.range_constexpr(cute.size(tSrK_cp)):
        tSrK_cp[_e] = tSsK[_e]
    for k in cutlass.range_constexpr(num_k):
        cute.gemm(tiled_mma_qk, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)


@cute.jit
def _pv_mma_fp8(
    tiled_mma_pv: cute.TiledMma,
    acc_O: cute.Tensor,
    tOrP: cute.Tensor,
    tOrVt: cute.Tensor,
    tOsVt: cute.Tensor,
    smem_thr_copy_V: cute.TiledCopy,
):
    # fp8 leg PV: load V into the bf16 PV B-fragment ELEMENT-WISE (no ldmatrix,
    # which faults on the fp8 B-operand on sm_110a) + the bf16 PV mma. The read
    # goes through the SAME swizzle-aware smem view the proven bf16 ldmatrix
    # path uses (tOsVt = smem_thr_copy_V.partition_S(sVt), retiled into the
    # fragment's copy-value order) — NOT a raw sV[n_kv,d] index (which does not
    # survive the smem swizzle: the V=n_kv fragment dump showed raw sV reads
    # land +16 n_kv off on the c>=1 columns). retile puts tOrVt in the copy's
    # value order; autovec_copy moves the swizzled smem elements into it (the
    # O-store path proves this smem->reg form on sm_110a) with the implicit
    # fp8->bf16 cast.
    tOrVt_cv = smem_thr_copy_V.retile(tOrVt)
    for _e in cutlass.range_constexpr(cute.size(tOrVt_cv)):
        tOrVt_cv[_e] = tOsVt[_e]
    for k in cutlass.range_constexpr(cute.size(tOrP.shape[2])):
        cute.gemm(tiled_mma_pv, acc_O, tOrP[None, None, k], tOrVt[None, None, k], acc_O)


@cute.jit
def _convert_v_to_bf16(sV: cute.Tensor, sVb: cute.Tensor, tidx,
                       TILE_N: cutlass.Constexpr, D: cutlass.Constexpr,
                       num_threads: cutlass.Constexpr):
    # fp8 leg only: stage the fp8 V (just cp.async'd into sV) into the bf16 V
    # buffer sVb so the PV mma can ldmatrix it like the bf16 leg (an fp8
    # B-operand faults on sm_110a). Element-wise over the shared (n_kv, d)
    # logical grid: sV and sVb have different swizzles but the same (n, d, 1)
    # logical shape, so an identical (n, d) index reads/writes the same logical
    # element in each (CuTe applies each tensor's own swizzle). Thread-strided;
    # per is a compile-time Python int (TILE_N*D//num_threads), not cute.size
    # (which is symbolic and defeats range_constexpr).
    per = int(TILE_N) * int(D) // int(num_threads)
    for _i in cutlass.range_constexpr(per):
        _f = tidx * per + _i
        _n = _f // D
        _d = _f - _n * D
        sVb[_n, _d, 0] = sV[_n, _d, 0].to(sVb.element_type)


class BlackwellHd256DecodeTC:
    """Tensor-core (mma.sync) M=1 hd256 decode kernel on SM100/SM110."""

    HEAD_DIM = 256
    TILE_M = 64    # mma M-tile: the GQA group (6) padded (Sm80 4-warp config)
    TILE_N = 64    # mma N-tile: kv-rows per n_block
    NUM_THREADS = 128  # 4 warps (matches the FI FA2-tc decode geometry)

    def __init__(self, qhead_per_kvhead: int, page_size: int = 0):
        assert qhead_per_kvhead >= 1, "qhead_per_kvhead must be >= 1"
        assert self.HEAD_DIM % 16 == 0
        self.g = qhead_per_kvhead
        self.page_size = page_size
        self.num_threads = self.NUM_THREADS

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
        mPageTable: Optional[cute.Tensor] = None,
        seqlen_q: Int32 = 1,
        seqlen_k: Int32 = 0,
        softmax_scale_log2: Float32 = 0.0,
        num_splits: Int32 = 1,
        stream: cuda.CUstream = None,
    ):
        is_varlen_q = const_expr(mCuSeqlensQ is not None)
        if const_expr(is_varlen_q):
            batch_size = mCuSeqlensQ.shape[0] - 1
        else:
            batch_size = mQ.shape[0]
        num_kv_heads = mV.shape[-2]
        # Stamp 16-byte divisibility on the dynamic strides so every derived copy
        # pointer (base + dynamic row/head/batch offset) provably stays 128-bit
        # aligned for the cp.async gmem->smem loads (the Sm80 pattern, flash_fwd.py:716).
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        # Host-side layout prep (the Sm80 pattern): permute to (seqlen, d, heads,
        # batch) and pack the GQA group into the M-dimension.
        if const_expr(mCuSeqlensQ is None):
            QO_trans = [1, 3, 2, 0]
        else:
            QO_trans = [0, 2, 1]
        if const_expr(mCuSeqlensK is None):
            KV_trans = [1, 3, 2, 0]
        else:
            KV_trans = [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=QO_trans))
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=QO_trans))
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=KV_trans))
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=KV_trans))
        num_head_kv = mV.shape[2]
        mQ = pack_gqa_layout(mQ, self.g, num_head_kv, head_idx=2)
        mO = pack_gqa_layout(mO, self.g, num_head_kv, head_idx=2)
        self.kernel(
            mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK, mDescale,
            seqlen_q, seqlen_k, softmax_scale_log2, num_splits, mPageTable,
        ).launch(
            grid=[num_kv_heads, 1, batch_size],
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
        seqlen_q: Int32,
        seqlen_k: Int32,
        softmax_scale_log2: Float32,
        num_splits: Int32,
        mDbgV: Optional[cute.Tensor] = None,
    ):
        D = self.HEAD_DIM
        G = self.g
        TILE_M = self.TILE_M
        TILE_N = self.TILE_N
        # Q and O are bf16 (the I/O dtype); K/V stream as fp8 e4m3 (halving the
        # DRAM bytes) or stay bf16 (regression path). The fp8 QK/PV mmas are the
        # native mma.sync (MmaFP8Op), so Q and P are cast to fp8 in-regs and the
        # descales fold into the score scale (qk) and the output (v) — exactly
        # the GEMV/FI descale scheme.
        DTYPE = mO.element_type         # bf16 (O)
        KVTYPE = mK.element_type         # fp8 e4m3 or bf16 (K, V streamed)
        FP8T = Float8E4M3FN
        FP8_KV = const_expr(KVTYPE.width == 8)
        import os as _os_ew
        _FORCE_EW = _os_ew.environ.get("TC_FORCE_EW") == "1"
        # Q smem dtype: fp8 e4m3 on the fp8 leg (the interface pre-casts Q —
        # the native fp8 mma.sync needs an fp8 A-operand and its A fragment
        # (m16n8k32: 4-b32 regs of 4 consecutive k) has no ldmatrix 8-bit A
        # load; Q is read once from smem as a plain (autovec) copy). bf16
        # elsewhere (the ldmatrix regression path).
        QDTYPE = FP8T if const_expr(FP8_KV) else DTYPE
        # P-operand dtype: always bf16 (the PV mma is always bf16, see above).
        PT = DTYPE

        kv_head = cute.arch.block_idx()[0]
        batch_idx = cute.arch.block_idx()[2]
        tidx = cute.arch.thread_idx()[0]

        is_varlen_q = const_expr(mCuSeqlensQ is not None)
        is_varlen_k = const_expr(mCuSeqlensK is not None)
        # mQ/mO/mK/mV arrive already permuted + GQA-packed (host-side in __call__).
        # mLSE is packed here (element-wise store; no 128-bit gmem alignment needed).
        if const_expr(mLSE is not None):
            num_head_kv = mO.shape[2]
            if const_expr(not is_varlen_q):
                # dense mLSE is (B, L_q, H); pack_gqa_layout(head_idx=1) wants the
                # head dim at mode 1, so permute to (L_q, H, B) first (the varlen
                # mLSE is already (total_q, H) with the head dim at mode 1).
                mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[1, 2, 0]))
            mLSE = pack_gqa_layout(mLSE, G, num_head_kv, head_idx=1)

        # Per-batch bases / lengths.
        if const_expr(is_varlen_q):
            q_row_base = Int32(mCuSeqlensQ[batch_idx])
        else:
            q_row_base = batch_idx * seqlen_q
        if const_expr(is_varlen_k):
            kv_row0 = Int32(mCuSeqlensK[batch_idx])
            kv_len = Int32(mCuSeqlensK[batch_idx + 1]) - kv_row0
        else:
            kv_row0 = batch_idx * seqlen_k
            kv_len = seqlen_k
        num_nblocks = (kv_len + TILE_N - 1) // TILE_N

        # Per-(batch, kv_head) descales (identity when absent). The fp8 mma.sync
        # computes S_raw = Q_fp8 . K_fp8 and O_raw = P_fp8 . V_fp8, so the true
        # scores are scaled by q_descale*k_descale (folded into the softmax scale)
        # and the output by v_descale (applied once in the epilogue).
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

        # ---- mma infrastructure (mma.sync, Sm80 pattern) ---------------------
        # QK mma: fp8 KV -> the native fp8 mma.sync m16n8k32 (MmaFP8Op, k-tile
        # 32); bf16 KV -> m16n8k16 (the regression path).
        if const_expr(FP8_KV):
            mma_op = warp.MmaFP8Op(FP8T, Float32, (16, 8, 32))
            mma_k = 32
        else:
            mma_op = warp.MmaF16BF16Op(DTYPE, Float32, (16, 8, 16))
            mma_k = 16
        tiled_mma_qk = cute.make_tiled_mma(
            mma_op,
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, mma_k),
        )
        # PV mma: ALWAYS bf16 m16n8k16, even on the fp8 leg. The back-to-back
        # C->A reshape (quack convert_layout_acc_frgA) is k=16-geometry only
        # ("TODO: Sm90 FP8" upstream): it yields 8-element A-blocks per
        # 16-col k-window, but a fp8 k=32 A-fragment needs 16 elements per
        # k-block -> the k-loop over-ranges and reads undefined e4m3 register
        # bits (NaN). A fp8 PV mma would need a P smem round-trip (FA2
        # pattern); instead P is cast to bf16 in-regs (8-bit mantissa >=
        # e4m3's 3) and V is cast per-element from fp8 smem.
        if const_expr(FP8_KV):
            mma_pv_op = warp.MmaF16BF16Op(DTYPE, Float32, (16, 8, 16))
        else:
            mma_pv_op = mma_op
        tiled_mma_pv = cute.make_tiled_mma(
            mma_pv_op,
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)

        # Q smem: fp8 on the fp8 leg (pre-cast Q), bf16 otherwise; O smem stays
        # bf16 (O is written once); K/V stream fp8.
        sQ_atom = sm80_utils.get_smem_layout_atom(QDTYPE, D)
        sK_atom = sm80_utils.get_smem_layout_atom(KVTYPE, D)
        sV_atom = sm80_utils.get_smem_layout_atom(KVTYPE, D)
        sO_atom = sm80_utils.get_smem_layout_atom(DTYPE, D)
        sQ_layout = cute.tile_to_shape(sQ_atom, (TILE_M, D), (0, 1))
        sK_layout = cute.tile_to_shape(sK_atom, (TILE_N, D, 1), (0, 1, 2))
        sV_layout = cute.tile_to_shape(sV_atom, (TILE_N, D, 1), (0, 1, 2))
        sO_layout = cute.tile_to_shape(sO_atom, (TILE_M, D), (0, 1))
        # fp8 leg: the PV mma is bf16 (an fp8 B-operand faults on sm_110a), so V
        # is staged in a bf16 smem buffer (sVb) and ldmatrix'd like the bf16 leg.
        # V is loaded fp8 (the DRAM win) into sV, then converted sV->sVb per block
        # so the proven bf16 ldmatrix reads it. sVb is allocated only here (the
        # bf16 leg stays at its 128 KiB footprint, so the struct is conditional).
        sVb_atom = sm80_utils.get_smem_layout_atom(DTYPE, D)
        sVb_layout = cute.tile_to_shape(sVb_atom, (TILE_N, D, 1), (0, 1, 2))

        smem = cutlass.utils.SmemAllocator()
        if const_expr(FP8_KV):
            @cute.struct
            class _SharedStorage:
                sV: cute.struct.Align[cute.struct.MemRange[KVTYPE, cute.cosize(sV_layout)], 1024]
                sVb: cute.struct.Align[cute.struct.MemRange[DTYPE, cute.cosize(sVb_layout)], 1024]
                sO: cute.struct.Align[cute.struct.MemRange[DTYPE, cute.cosize(sO_layout)], 1024]
                sQ: cute.struct.Align[cute.struct.MemRange[QDTYPE, cute.cosize(sQ_layout)], 1024]
                sK: cute.struct.Align[cute.struct.MemRange[KVTYPE, cute.cosize(sK_layout)], 1024]
            storage = smem.allocate(_SharedStorage)
            sVb = storage.sVb.get_tensor(sVb_layout)
        else:
            @cute.struct
            class _SharedStorage:
                sV: cute.struct.Align[cute.struct.MemRange[KVTYPE, cute.cosize(sV_layout)], 1024]
                sO: cute.struct.Align[cute.struct.MemRange[DTYPE, cute.cosize(sO_layout)], 1024]
                sQ: cute.struct.Align[cute.struct.MemRange[QDTYPE, cute.cosize(sQ_layout)], 1024]
                sK: cute.struct.Align[cute.struct.MemRange[KVTYPE, cute.cosize(sK_layout)], 1024]
            storage = smem.allocate(_SharedStorage)
            sVb = None
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sV_layout)
        sO = storage.sO.get_tensor(sO_layout)
        # The PV mma reads V in bf16: sV directly (bf16 leg) or the fp8->bf16
        # staged copy sVb (fp8 leg).
        if const_expr(FP8_KV):
            sVt = layout_utils.transpose_view(sVb)
        else:
            sVt = layout_utils.transpose_view(sV)
        # NOTE: the KV smem OOB rows (beyond the per-batch seqlen) are only read
        # by the mma when a block is partial. In production the KV is a pool
        # (always larger than any single sequence), so those rows hold finite
        # data and the -inf score mask makes their P@V contribution 0. An exactly
        # sized KV (a short unit test) would read true OOB -> NaN; the verify
        # pads the KV to a whole tile for that reason.

        # gmem -> smem tiled copies (Q/K/V/O). Q cp.async as QDTYPE (fp8 on the
        # fp8 leg); O as bf16; K/V as KVTYPE.
        q_async_copy_elems = 128 // QDTYPE.width         # Q: 16 (fp8) / 8 (bf16)
        o_async_copy_elems = 128 // DTYPE.width          # O: bf16 -> 8 elems
        kv_async_copy_elems = 128 // KVTYPE.width        # K/V: 16 (fp8) / 8 (bf16)
        atom_async_copy_q = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), QDTYPE,
            num_bits_per_copy=128)
        atom_async_copy_kv = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), KVTYPE,
            num_bits_per_copy=128)
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), DTYPE, num_bits_per_copy=128)
        tQK_shape_dim_1 = sQ_atom.outer.shape[1] // q_async_copy_elems
        tK_shape_dim_1 = sK_atom.outer.shape[1] // kv_async_copy_elems
        tV_shape_dim_1 = sV_atom.outer.shape[1] // kv_async_copy_elems
        tQ_layout = cute.make_ordered_layout(
            (self.num_threads // tQK_shape_dim_1, tQK_shape_dim_1), order=(1, 0))
        tK_layout = cute.make_ordered_layout(
            (self.num_threads // tK_shape_dim_1, tK_shape_dim_1), order=(1, 0))
        tV_layout = cute.make_ordered_layout(
            (self.num_threads // tV_shape_dim_1, tV_shape_dim_1), order=(1, 0))
        tO_shape_dim_1 = sO_atom.outer.shape[1] // o_async_copy_elems
        tO_layout = cute.make_ordered_layout(
            (self.num_threads // tO_shape_dim_1, tO_shape_dim_1), order=(1, 0))
        vQ_layout = cute.make_layout((1, q_async_copy_elems))
        vO_layout = cute.make_layout((1, o_async_copy_elems))
        vKV_layout = cute.make_layout((1, kv_async_copy_elems))
        gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async_copy_q, tQ_layout, vQ_layout)
        gmem_tiled_copy_K = cute.make_tiled_copy_tv(atom_async_copy_kv, tK_layout, vKV_layout)
        gmem_tiled_copy_V = cute.make_tiled_copy_tv(atom_async_copy_kv, tV_layout, vKV_layout)
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)
        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)

        # mma fragments + smem ldmatrix copy atoms.
        # Q: the fp8 leg has NO ldmatrix for Q — the fp8 A fragment (m16n8k32:
        # 4 b32 regs of 4 consecutive k) has no 8-bit ldmatrix A load, and Q is
        # read from smem once per CTA as a plain autovec copy. The bf16 leg
        # keeps the ldmatrix (regression path).
        # K/V: the 16-bit ldmatrix (LdMatrix8x8x16bOp) — typed at fp8 on the
        # fp8 leg (the fp8 mma B fragment is the b16 fragment at 2x k-density;
        # see the atom section), bf16 on the regression leg.
        if const_expr(FP8_KV):
            tCsQ = thr_mma_qk.partition_A(sQ)
            tSrQ = thr_mma_qk.make_fragment_A(tCsQ)
        else:
            tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((TILE_M, D))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)
        if const_expr(FP8_KV):
            # The fp8 mma B fragment (m16n8k32) is the m16n8k16 16-bit fragment
            # at 2x k-density, so the 16-bit ldmatrix typed at fp8 composes
            # exactly (the quack production-fp8-GEMM pattern —
            # task/research/cute-dsl-ldmatrix-fp8-b-operand/summary.md). The
            # 8-bit LdMatrix16x8x8bOp is NOT usable for k-major fp8 B here:
            # it is not PTX-faithful (lowers to m16n16.b8 + permutations), its
            # tiled per-thread bases carry an 8-byte phase in this k-major
            # smem (a genuine cudaErrorMisalignedAddress, not just the static
            # check), and make_tiled_copy_B with it fetches the wrong k.
            # K smem is k-major (transpose=False, as the bf16 K leg); sVt is
            # n-major (transpose=True, as the bf16 V leg).
            smem_copy_atom_K = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), FP8T)
            smem_copy_atom_V = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), FP8T)
        else:
            smem_copy_atom_Q = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), DTYPE)
            smem_copy_atom_K = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), DTYPE)
            smem_copy_atom_V = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), DTYPE)
        if const_expr(not FP8_KV):
            smem_thr_copy_Q = utils.make_tiled_copy_A(
                smem_copy_atom_Q, tiled_mma_qk).get_slice(tidx)
            tSsQ = smem_thr_copy_Q.partition_S(sQ)
        smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_K, tiled_mma_qk).get_slice(tidx)
        if const_expr(FP8_KV):
            # The PV mma is bf16 (see above) but the V smem is fp8. tOsVt is
            # only a coordinate source for the per-element V reads in
            # _pv_mma_fp8, so it must carry the PV mma's B-fragment per-thread
            # geometry (the make_tiled_copy_B invariant: retiled smem view
            # element order == mma fragment order) — pair a bf16-typed atom
            # with tiled_mma_pv, NOT the fp8 QK mma (whose k=32 geometry has
            # half the k-blocks of the bf16 PV mma's k=16 -> wrong V mapping).
            # partition_S keeps the fp8 sVt addressing; the atom only
            # contributes the ldmatrix TV structure.
            smem_copy_atom_V_pv = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), DTYPE)
            smem_thr_copy_V = utils.make_tiled_copy_B(
                smem_copy_atom_V_pv, tiled_mma_pv).get_slice(tidx)
        else:
            smem_thr_copy_V = utils.make_tiled_copy_B(
                smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        if const_expr(FP8_KV):
            # The ldmatrix copy view for K (the fp8 leg prefetches K in-regs
            # per k-block; the bf16 leg's sm80_utils.gemm does its own retile).
            tSrK_cp = smem_thr_copy_K.retile(tSrK)
            # partition_S drops the pointer-alignment assumption, deriving the
            # 8-bit element default, while the b16 ldmatrix atom requires its
            # 16B row starts. Here the stamp is SOUND (unlike the old 8-bit
            # atom, whose per-thread bases carried a genuine 8-byte phase):
            # the b16 per-thread 16B row starts are 128n + 16*k16 (K leg) /
            # 16*d16 + 128n (V leg) — 16-byte-aligned pre-swizzle, and this
            # swizzle only XORs bits >= 4 (task/research/cute-dsl-ldmatrix-
            # fp8-b-operand/summary.md).
            tSsK = cute.make_tensor(
                cute.make_ptr(KVTYPE, tSsK.iterator.toint(), cute.AddressSpace.smem,
                              assumed_align=16),
                tSsK.layout)
            # tOsVt needs no re-stamp here: the fp8 leg's V is the bf16 staged
            # copy sVb (transposed), so partition_S keeps the bf16 16-bit
            # default the b16 ldmatrix wants — identical to the bf16 leg.

        import os as _dbg_os
        if _dbg_os.environ.get("TC_DBG_LAYOUT") == "1":
            print("DBG tOrVt.shape =", tOrVt.shape)
            print("DBG tOrVt.layout =", tOrVt.layout)
            print("DBG tOsVt.shape =", tOsVt.shape)
            print("DBG tOsVt.layout =", tOsVt.layout)
            print("DBG tOsVt_cv.shape =", smem_thr_copy_V.retile(tOrVt).shape)
            print("DBG tSrK.shape =", tSrK.shape)
            print("DBG tSrK.layout =", tSrK.layout)
            print("DBG tSsK.shape =", tSsK.shape)
            print("DBG tSsK.layout =", tSsK.layout)
            print("DBG per_k =", cute.size(tSsK.shape[0]) * cute.size(tSsK.shape[1]))

        acc_S_shape = thr_mma_qk.partition_shape_C((TILE_M, TILE_N))
        acc_S = cute.make_rmem_tensor(acc_S_shape, Float32)

        # online softmax (num_rows = the per-thread M-rows of the O mma-C fragment,
        # so the row_scale size matches acc_O_mn mode[0] in rescale_O).
        num_rows = cute.size(layout_utils.reshape_acc_to_mn(acc_O), mode=[0])
        # scale_eff carries the qk descale (the fp8 QK mma scores are raw; the
        # true scores are raw * q_descale * k_descale * softmax_scale). Identity
        # in the bf16 / no-descale path (scale_eff == softmax_scale_log2).
        softmax = Softmax.create(
            scale_eff, num_rows, arch=110, softmax_scale=scale_eff)
        softmax.reset()

        # ---- KV gmem source views (dense + varlen) -------------------------
        if const_expr(is_varlen_k):
            mK_cur = cute.domain_offset((kv_row0, 0), mK[None, None, kv_head])
            mV_cur = cute.domain_offset((kv_row0, 0), mV[None, None, kv_head])
        else:
            mK_cur = mK[None, None, kv_head, batch_idx]
            mV_cur = mV[None, None, kv_head, batch_idx]
        gK = cute.local_tile(mK_cur, (TILE_N, D), (None, 0))
        gV = cute.local_tile(mV_cur, (TILE_N, D), (None, 0))

        # ---- Q load (PackGQA: the GQA group is the M-dimension) -------------
        if const_expr(is_varlen_q):
            mQ_cur = cute.domain_offset((q_row_base, 0), mQ[None, None, kv_head])
        else:
            mQ_cur = mQ[None, None, kv_head, batch_idx]
        pack_gqa = PackGQA(TILE_M, D, False, G)
        pack_gqa.load_Q(mQ_cur, sQ, gmem_tiled_copy_Q, tidx, 0, seqlen=seqlen_q)
        cute.arch.cp_async_commit_group()
        # Q -> regs (A_in_regs): rotate smem_q into the mma A fragment.
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if const_expr(FP8_KV):
            # Plain smem->reg copy: no ldmatrix for the fp8 A-operand (see the
            # fragment section). Element-wise (no vectorized load/store API
            # through the composed/swizzled smem layout). One-time cost — Q
            # is loaded once per CTA and reused across all n_blocks.
            for _q in cutlass.range_constexpr(cute.size(tSrQ)):
                tSrQ[_q] = tCsQ[_q]
            tSrQ_mma = tSrQ
        else:
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)
            tSrQ_mma = tSrQ

        # S-coord fragment for the n-OOB mask (identity tensor over the S tile).
        # Indexed by its OWN modes (see the mask loops below): reshape_acc_to_mn
        # on a coordinate tensor does not yield the true (m, n) coords, so the
        # mask must read tScS[(a, b), 0, c] directly (verified probe: raw-fragment
        # element access + per-element store + SSA ternary all land reliably,
        # while the mn-view coord reshape is garbage).
        cS = cute.make_identity_tensor((TILE_M, TILE_N))
        tScS = thr_mma_qk.partition_C(cS)

        # KV gmem partition + row predicates (the tail n_block is predicated to
        # avoid OOB gmem reads; the mma n-OOB mask handles the -inf scores).
        cK = cute.make_identity_tensor((TILE_N, D))
        tKcK = gmem_thr_copy_K.partition_S(cK)
        t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
        tKpK = utils.predicate_k(tKcK, limit=D)
        tVcV = gmem_thr_copy_V.partition_S(cK)
        t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cK)
        tVpV = utils.predicate_k(tVcV, limit=D)
        tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
        tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)

        # NOTE on the V load: it is NOT predicated. The mma does P@V over the full
        # (TILE_N, D) V tile; a partial block leaves rows >= kv_len read from gmem.
        # P[OOB] == 0 (score -inf), so those rows contribute 0 * V[OOB]; V[OOB] must
        # be FINITE. Dense/varlen decode always serves a sequence from a KV pool that
        # is >= any single sequence, so rows beyond kv_len are in-bounds pool data
        # (finite); an exactly sized unit-test KV is padded to a whole tile for the
        # same reason. (The K load stays predicated: its OOB rows only feed the S
        # mma, which the -inf score mask discards, so they need not be finite.)

        # ---- mainloop over n_blocks (single-stage KV) -----------------------
        # is_first must be a compile-time constant (Softmax.online_softmax takes a
        # Constexpr), so the first block (is_first=True) and the steady-state loop
        # (is_first=False) are each emitted inline (the Sm80 pattern; a nested fn
        # is not allowed here because staged if/for reject non-tensor closures).
        #
        # First n_block (is_first=True).
        if num_nblocks >= 1:
            nb = Int32(0)
            kv_limit = kv_len - nb * TILE_N
            if cutlass.const_expr(DBG_STAGE < 2):
                for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                    if t0KcK[0, n, 0][0] < kv_limit:
                        cute.copy(
                            gmem_thr_copy_K, tKgK[None, n, None, nb],
                            tKsK[None, n, None, 0], pred=tKpK[None, n, None])
                cute.arch.cp_async_commit_group()
                for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                    cute.copy(
                        gmem_thr_copy_V, tVgV[None, n, None, nb],
                        tVsV[None, n, None, 0], pred=tVpV[None, n, None])
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
            if cutlass.const_expr(DBG_STAGE < 3):
                if const_expr(FP8_KV):
                    _qk_mma_fp8(tiled_mma_qk, acc_S, tSrQ_mma, tSrK, tSsK, tSrK_cp,
                                smem_thr_copy_K)
                else:
                    sm80_utils.gemm(
                        thr_mma_qk, acc_S, tSrQ_mma, tSrK, tSsQ, tSsK[None, None, None, 0],
                        smem_thr_copy_Q, smem_thr_copy_K, A_in_regs=True)
                # OOB mask (first block, see steady-state note).
                if kv_len < TILE_N:
                    _mask_partial_kv(acc_S, tScS, nb, kv_len, TILE_N)
                row_scale = softmax.online_softmax(acc_S, is_first=cutlass.const_expr(True))
                softmax.rescale_O(acc_O, row_scale)
            if cutlass.const_expr(DBG_STAGE < 4):
                rP = cute.make_fragment_like(acc_S, PT)
                rP.store(acc_S.load().to(PT))
                tOrP = layout_utils.reshape_acc_to_frgA(rP)
                if const_expr(FP8_KV):
                    # Stage V fp8 -> bf16, then the PV ldmatrix reads sVb exactly
                    # like the bf16 leg (proven; an fp8 B-operand faults on sm_110a).
                    _convert_v_to_bf16(sV, sVb, tidx, TILE_N, D, self.num_threads)
                if _FORCE_EW:
                    _pv_mma_fp8(tiled_mma_pv, acc_O, tOrP, tOrVt, tOsVt, smem_thr_copy_V)
                else:
                    sm80_utils.gemm_rs(
                        thr_mma_pv, acc_O, tOrP, tOrVt, tOsVt[None, None, None, 0],
                        smem_thr_copy_V)

        # Steady-state n_blocks (is_first=False).
        if cutlass.const_expr(DBG_STAGE < 5):
            for nb in cutlass.range(1, num_nblocks):
                kv_limit = kv_len - nb * TILE_N
                for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                    if t0KcK[0, n, 0][0] < kv_limit:
                        cute.copy(
                            gmem_thr_copy_K, tKgK[None, n, None, nb],
                            tKsK[None, n, None, 0], pred=tKpK[None, n, None])
                cute.arch.cp_async_commit_group()
                for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                    cute.copy(
                        gmem_thr_copy_V, tVgV[None, n, None, nb],
                        tVsV[None, n, None, 0], pred=tVpV[None, n, None])
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                if const_expr(FP8_KV):
                    _qk_mma_fp8(tiled_mma_qk, acc_S, tSrQ_mma, tSrK, tSsK, tSrK_cp,
                                smem_thr_copy_K)
                else:
                    sm80_utils.gemm(
                        thr_mma_qk, acc_S, tSrQ_mma, tSrK, tSsQ, tSsK[None, None, None, 0],
                        smem_thr_copy_Q, smem_thr_copy_K, A_in_regs=True)
                # OOB mask (see the first-block mask). The guard is the verify3
                # shape (a per-block single runtime-if): only the last block can
                # contain OOB columns, and full kv_len keeps the zero-overhead
                # path. (A `nb == num_nblocks - 1` guard here measurably
                # re-scheduled the hot loop and corrupted large-L single-batch
                # runs — keep the loop structure stable.)
                if kv_len % TILE_N != 0:
                    _mask_partial_kv(acc_S, tScS, nb, kv_len, TILE_N)
                row_scale = softmax.online_softmax(acc_S, is_first=cutlass.const_expr(False))
                softmax.rescale_O(acc_O, row_scale)
                rP = cute.make_fragment_like(acc_S, PT)
                rP.store(acc_S.load().to(PT))
                tOrP = layout_utils.reshape_acc_to_frgA(rP)
                if const_expr(FP8_KV):
                    _convert_v_to_bf16(sV, sVb, tidx, TILE_N, D, self.num_threads)
                if _FORCE_EW:
                    _pv_mma_fp8(tiled_mma_pv, acc_O, tOrP, tOrVt, tOsVt, smem_thr_copy_V)
                else:
                    sm80_utils.gemm_rs(
                        thr_mma_pv, acc_O, tOrP, tOrVt, tOsVt[None, None, None, 0],
                        smem_thr_copy_V)

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # ---- epilogue -------------------------------------------------------
        # finalize softmax -> row_scale = 1/l (row_sum holds the LSE, natural units).
        row_scale = softmax.finalize()
        softmax.rescale_O(acc_O, row_scale)
        # v_descale: the PV mma accumulated P @ V_raw, so O = (V/l) * v_descale.
        # Exact no-op in the bf16 / no-descale path (v_descale == 1.0). The LSE
        # (row_sum) is score-domain only and is not affected.
        acc_O.store(acc_O.load() * v_descale)

        if const_expr(mLSE is not None):
            if const_expr(is_varlen_q):
                mLSE_cur = mLSE[None, kv_head]
            else:
                mLSE_cur = mLSE[None, kv_head, batch_idx]
            pack_gqa.store_LSE(mLSE_cur, softmax.row_sum, tiled_mma_pv, tidx, 0, seqlen_q)

        # O: acc_O (rmem) -> sO (smem) -> rmem fragment -> gmem (the Sm80 store path).
        rO = cute.make_fragment_like(acc_O, DTYPE)
        rO.store(acc_O.load().to(DTYPE))
        smem_copy_atom_O = utils.get_smem_store_atom(110, DTYPE)
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma_pv).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)
        cute.arch.sync_threads()
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOrO = cute.make_fragment_like(tOsO, DTYPE)
        cute.autovec_copy(tOsO, tOrO)
        if const_expr(is_varlen_q):
            mO_cur = mO[None, None, kv_head]
        else:
            mO_cur = mO[None, None, kv_head, batch_idx]
        pack_gqa.store_O(mO_cur, tOrO, gmem_tiled_copy_O, tidx, 0, seqlen_q)
