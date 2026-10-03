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

Scope (correctness-first): dense + varlen (cu_seqlens_k) + paged (page table +
seqused_k; the per-n_block physical page is gathered in-kernel — see the
``is_paged`` branches below), bf16 Q + bf16 or fp8-e4m3 KV (fp8 with the
per-(batch, kv_head) descales: qk_descale folds into the softmax scale, v_descale
into the epilogue — the GEMV/FI descale scheme), single-stage cp.async KV.

SplitKV (FlashInfer Alg.1, the GEMV Phase-B1 reference): ``num_splits`` is a
runtime grid y-dimension (one binary serves every ns), so with ``num_splits > 1``
each CTA scans its **tile-aligned** KV chunk (chunk boundaries are TILE_N
multiples — the TC's cp.async reads whole 64-row tiles, unlike the GEMV's
row-aligned chunks) and emits the chunk-normalized fp32 partials
``mO_partial[split]`` / ``mLSE_partial[split]`` (the host packs them like the
final O/LSE with a leading split plane; the host combine re-merges). An empty
chunk (ns over-split past the end) writes O=0 / LSE=-inf (``Softmax.finalize``'s
zero row_sum path) — the combine discards it. ``num_splits = 1`` keeps the
serial scan (``kv_start_tile = 0`` → ``gtile == nb``) and the bf16 final O store.
The deep KV ring is the follow-up pass.

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


@cute.jit
def _mask_partial_kv(
    acc_S: cute.Tensor,
    thr_mma: cute.TiledMma,
    nb: Int32,
    kv_len: Int32,
    tile_m,
    tile_n,
):
    """Write -inf into the OOB kv-columns of the acc_S rows. The mma C-fragment
    is per-thread (each thread owns a 16-column block, not columns 0..15), so
    the mn-view column index is not the absolute kv column. The identity tensor
    ``cS`` partitioned by the same mma holds, per mn-slot, the coordinate each
    slot represents — ``tScS_mn[r, c][1]`` is the absolute kv column. That
    coordinate is abstract and constant-folds to a fixed value when used inline,
    so it is first materialized into rmem (the true per-thread column). The
    -inf select then goes through an mn-view row round-trip (the access path
    ``online_softmax`` consumes); a runtime-``if`` around an rmem element store
    is dropped at trace, so the store is unconditional and the OOB pick is a
    runtime ternary (the GEMV ``s = s if ... else -inf`` idiom)."""
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
    cS = cute.make_identity_tensor((tile_m, tile_n))
    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS))
    _nr = cute.size(acc_S_mn.shape[0])
    _nc = cute.size(acc_S_mn.shape[1])
    # Materialize the per-thread absolute kv column for every mn-slot to rmem.
    _col = cute.make_rmem_tensor((_nr, _nc), Int32)
    for _r in cutlass.range_constexpr(_nr):
        for _c in cutlass.range_constexpr(_nc):
            _col[_r, _c] = tScS_mn[_r, _c][1]
    for _r in cutlass.range_constexpr(_nr):
        _row_t = cute.make_rmem_tensor(_nc, Float32)
        _row_t.store(acc_S_mn[_r, None].load())
        for _c in cutlass.range_constexpr(_nc):
            _v = _row_t[_c]
            _row_t[_c] = (
                -Float32.inf
                if nb * tile_n + _col[_r, _c] >= kv_len
                else _v
            )
        acc_S_mn[_r, None].store(_row_t.load())


@cute.jit
def _qk_mma_fp8(
    tiled_mma_qk: cute.TiledMma,
    acc_S: cute.Tensor,
    tSrQ: cute.Tensor,
    tSrK: cute.Tensor,
    sK_b: cute.Tensor,
):
    """fp8 QK mma (Q already in regs; K loaded ELEMENT-WISE from the mma's own
    B-partition sK_b). sK_b = thr_mma_qk.partition_B(sK) maps each fragment
    slot _e to the correct (n_kv, d), so a straight per-_e copy (tSrK[_e] =
    sK_b[_e]) is correct. This BYPASSES the b16-ldmatrix-typed-fp8 K-major
    tiled-copy retile (tSrK_cp = retile(tSrK) / tSsK = partition_S), which is
    broken on sm_110a: its partition delivers only ~33 of the 256 d-values
    (probe kdump). The mma is the native fp8 m16n8k32 (k-tile 32, D/32 blocks)."""
    num_k = cute.size(sK_b.shape[2])
    for _e in cutlass.range_constexpr(cute.size(tSrK)):
        tSrK[_e] = sK_b[_e]
    for k in cutlass.range_constexpr(num_k):
        cute.gemm(tiled_mma_qk, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)


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
        # Paged pool page size (0 = dense; 64/128 = the flat-pool paged mode).
        # The paged mode requires TILE_N (64) <= page_size so a whole n_block
        # (TILE_N kv rows) fits inside ONE physical page — the per-tile page
        # gather then needs a single page-table read per n_block.
        assert page_size in (0, 64, 128), "page_size must be 0 (dense), 64, or 128"
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
        mSequsedK: Optional[cute.Tensor] = None,
        seqlen_q: Int32 = 1,
        seqlen_k: Int32 = 0,
        softmax_scale_log2: Float32 = 0.0,
        num_splits: Int32 = 1,
        stream: cuda.CUstream = None,
        mPin: Optional[cute.Tensor] = None,
        mO_partial: Optional[cute.Tensor] = None,
        mLSE_partial: Optional[cute.Tensor] = None,
        is_m1: cutlass.Constexpr[bool] = True,
    ):
        # SplitKV: the partial pair is all-or-nothing (the presence of mO_partial
        # is the compile-time split flag, the GEMV idiom).
        assert const_expr((mO_partial is None) == (mLSE_partial is None))
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
        # The split partials get the same alignment stamps (assume_tensor_aligned
        # passes None through).
        mO_partial = assume_tensor_aligned(mO_partial)
        mLSE_partial = assume_tensor_aligned(mLSE_partial)
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
        if const_expr(mPageTable is None):
            # dense/varlen: permute the KV to (seqlen, d, heads, batch) /
            # (total, d, heads) (the Sm80 pattern).
            mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=KV_trans))
            mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=KV_trans))
        # paged: mK/mV stay the flat-pool (NP*PS, H_kv, D) views (A5) — the
        # kernel gathers the per-n_block physical page (A8-A10), so the KV_trans
        # select is skipped (it would be out-of-rank on the 3D pool view).
        # The head axis sits at mode 1 (mode 2 is D) in the paged pool view.
        num_head_kv = mV.shape[1] if const_expr(mPageTable is not None) else mV.shape[2]
        mQ = pack_gqa_layout(mQ, self.g, num_head_kv, head_idx=2)
        mO = pack_gqa_layout(mO, self.g, num_head_kv, head_idx=2)
        self.kernel(
            mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK, mDescale,
            seqlen_q, seqlen_k, softmax_scale_log2, num_splits,
            mPageTable, mSequsedK, mPin, mO_partial, mLSE_partial,
            is_m1,
        ).launch(
            # SplitKV: one CTA per (kv_head, kv-chunk, batch) — num_splits is a
            # RUNTIME grid dim (the GEMV idiom: one split binary serves every ns;
            # ns=1 is the serial grid [H_kv, 1, B]).
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
        seqlen_q: Int32,
        seqlen_k: Int32,
        softmax_scale_log2: Float32,
        num_splits: Int32,
        mPageTable: Optional[cute.Tensor] = None,
        mSequsedK: Optional[cute.Tensor] = None,
        mPin: Optional[cute.Tensor] = None,
        mO_partial: Optional[cute.Tensor] = None,
        mLSE_partial: Optional[cute.Tensor] = None,
        is_m1: cutlass.Constexpr[bool] = True,
    ):
        D = self.HEAD_DIM
        G = self.g
        TILE_M = self.TILE_M
        TILE_N = self.TILE_N
        # Paged pool mode (A6): K/V arrive as flat-pool (NP*PS, H_kv, D) views;
        # the per-n_block physical page is gathered from mPageTable in-kernel and
        # the flat-pool tile coordinate (ktile) replaces the logical n_block. The
        # dense/varlen path below is byte-identical to the pre-paged kernel
        # (is_paged is compile-time folded; PS is unused there).
        is_paged = const_expr(mPageTable is not None)
        PS = const_expr(self.page_size)
        # Q and O are bf16 (the I/O dtype); K/V stream as fp8 e4m3 (halving the
        # DRAM bytes) or stay bf16 (regression path). The fp8 QK/PV mmas are the
        # native mma.sync (MmaFP8Op), so Q and P are cast to fp8 in-regs and the
        # descales fold into the score scale (qk) and the output (v) — exactly
        # the GEMV/FI descale scheme.
        DTYPE = mO.element_type         # bf16 (O)
        KVTYPE = mK.element_type         # fp8 e4m3 or bf16 (K, V streamed)
        # SplitKV: the partials' presence is the compile-time split flag
        # (the GEMV idiom). In split mode the chunk-normalized O partial is fp32
        # (the combine re-merges the planes; the LSE partial is already fp32).
        # ODTYPE types the O-store path ONLY (sO smem, the O copy atoms, the
        # rmem fragments) — the mma/softmax stay fp32/bf16 as before. Non-split
        # keeps bf16 everywhere (byte-identical footprint).
        IS_SPLIT = const_expr(mO_partial is not None)
        ODTYPE = Float32 if IS_SPLIT else DTYPE
        FP8T = Float8E4M3FN
        FP8_KV = const_expr(KVTYPE.width == 8)
        # Q smem dtype: fp8 e4m3 on the fp8 leg (the interface pre-casts Q —
        # the native fp8 mma.sync needs an fp8 A-operand and its A fragment
        # (m16n8k32: 4-b32 regs of 4 consecutive k) has no ldmatrix 8-bit A
        # load; Q is read once from smem as a plain (autovec) copy). bf16
        # elsewhere (the ldmatrix regression path).
        QDTYPE = FP8T if const_expr(FP8_KV) else DTYPE
        # P-operand dtype: always bf16 (the PV mma is always bf16, see above).
        PT = DTYPE

        kv_head = cute.arch.block_idx()[0]
        split_idx = cute.arch.block_idx()[1]   # SplitKV chunk (0 at num_splits=1)
        batch_idx = cute.arch.block_idx()[2]
        tidx = cute.arch.thread_idx()[0]

        is_varlen_q = const_expr(mCuSeqlensQ is not None)
        is_varlen_k = const_expr(mCuSeqlensK is not None)
        # mQ/mO/mK/mV arrive already permuted + GQA-packed (host-side in __call__).
        # mLSE is packed here (element-wise store; no 128-bit gmem alignment needed).
        # In split mode the base is mLSE_partial[split_idx]: the host packs the
        # partial like the final mLSE plus a leading (num_splits) plane, so the
        # sliced plane has the IDENTICAL (B, L_q, H) / (H, total_q) shape and the
        # packing below is unchanged — only the destination differs.
        if const_expr(mLSE is not None or mLSE_partial is not None):
            if const_expr(IS_SPLIT):
                # A bare mLSE_partial[split_idx] is an ELEMENT access in this
                # DSL (it returns a scalar, not a plane), so build the CTA's
                # split plane by offsetting the base pointer by
                # split_idx * (elems per split block) and taking the sub-layout
                # with the leading (ns) mode dropped — the same trick the GEMV
                # uses (split_idx*split_stride into a flat offset, see
                # sm100_hd256_decode_gemv.py). The plane is IDENTICAL to the
                # final mLSE, so the select/pack below are unchanged.
                if const_expr(is_varlen_q):
                    lse_stride = mLSE_partial.shape[1] * mLSE_partial.shape[2]
                    lse_sub = cute.select(mLSE_partial.layout, mode=[1, 2])
                else:
                    lse_stride = (mLSE_partial.shape[1] * mLSE_partial.shape[2]
                                  * mLSE_partial.shape[3])
                    lse_sub = cute.select(mLSE_partial.layout, mode=[1, 2, 3])
                mLSE = cute.make_tensor(mLSE_partial.iterator + split_idx * lse_stride, lse_sub)
            num_head_kv = mO.shape[2]
            if const_expr(not is_varlen_q):
                # dense mLSE is (B, H, L_q) (the interface's lse_shape is
                # (batch_size, num_head, seqlen_q)); pack_gqa_layout(head_idx=1)
                # wants (seqlen_q, nheads, batch) — head dim at mode 1 — so
                # permute to (L_q, H, B) first.
                #
                # The select is a COMPILE-TIME branch on is_m1 (M is in the tc_key
                # -> a separate binary per M): M=1 keeps the historical
                # mode=[1,2,0] (seqlen sub-mode = H) — the proven-safe decode
                # codegen — while M>1 needs the mode=[2,1,0] (seqlen sub-mode =
                # L_q>1) that the (B,H,M) layout actually is. The two selects give
                # IDENTICAL M=1 store addresses (the (seqlen, head) modes alias
                # at L_q=1), so M=1 stays bit-exact; but the [2,1,0] codegen
                # (seqlen sub-mode 1) is UNSAFE for the M=1 split cells (it
                # mis-generates the LSE-store TMA and corrupts the O), so the M=1
                # path must never use it.
                if const_expr(is_m1):
                    mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[1, 2, 0]))
                else:
                    mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[2, 1, 0]))
            else:
                # varlen mLSE is (H, total_q) — the head dim is at mode 0.
                # pack_gqa_layout(head_idx=1) wants it at mode 1, so permute to
                # (total_q, H). (The old comment claimed it was already (total_q, H);
                # the interface allocates (num_head, total_q), so it is not.)
                mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=[1, 0]))
            mLSE = pack_gqa_layout(mLSE, G, num_head_kv, head_idx=1)

        # Per-batch bases / lengths.
        if const_expr(is_varlen_q):
            q_row_base = Int32(mCuSeqlensQ[batch_idx])
        else:
            q_row_base = batch_idx * seqlen_q
        if const_expr(is_varlen_k):
            kv_row0 = Int32(mCuSeqlensK[batch_idx])
            kv_len = Int32(mCuSeqlensK[batch_idx + 1]) - kv_row0
        elif const_expr(is_paged):
            # A7: the per-batch ACTUAL KV length comes from seqused_k — the flat
            # pool is (typically) padded to whole pages, so kv_len != pool rows.
            # The tail mask below uses this kv_len exactly as in the dense path.
            kv_len = Int32(mSequsedK[batch_idx])
        else:
            kv_row0 = batch_idx * seqlen_k
            kv_len = seqlen_k
        # ---- SplitKV: this CTA's TILE-ALIGNED KV chunk (split mode only) ----
        # The TC's tiled cp.async reads whole TILE_N-row tiles, so chunk
        # boundaries are tile multiples (the GEMV's row-aligned chunks do not
        # map onto the tiled loads). num_splits is a runtime value (the grid's
        # y-dim), so one split binary serves every ns; num_nblocks below is
        # this split's tile count (empty chunks past the end get 0 — the
        # epilogue then writes O=0 / LSE=-inf via Softmax.finalize's zero
        # row_sum, and the host combine discards the plane).
        #
        # CONST_EXPR-gated (the IS_SPLIT = mO_partial-is-not-None idiom): in
        # non-split mode NO chunk ops enter the leg at all — kv_start_tile
        # stays undefined and gtile below is a pure alias of nb. That matters:
        # any extra tile-index arithmetic (even a folded `Int32(0) + nb`)
        # shifts the paged leg's pipeline scheduling enough to break the
        # bit-exact dense-vs-paged cross-check, and the dense large-L codegen
        # moves off the canonical kernel's (a borderline rel cell tipped).
        if const_expr(IS_SPLIT):
            kv_total_tiles = (kv_len + TILE_N - 1) // TILE_N
            kv_chunk_tiles = (kv_total_tiles + num_splits - 1) // num_splits
            kv_start_tile = split_idx * kv_chunk_tiles
            num_nblocks = cutlass.min(
                kv_chunk_tiles,
                cutlass.max(Int32(0), kv_total_tiles - kv_start_tile),
            )
        else:
            # Non-split serial scan: no chunk ops in this leg (keeps the
            # paged dense-vs-paged cross-check bit-exact and the dense
            # large-L codegen identical to the canonical kernel).
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
        # at the I/O dtype (bf16 non-split / fp32 split partial — O is written
        # once); K/V stream fp8.
        sQ_atom = sm80_utils.get_smem_layout_atom(QDTYPE, D)
        sK_atom = sm80_utils.get_smem_layout_atom(KVTYPE, D)
        sV_atom = sm80_utils.get_smem_layout_atom(KVTYPE, D)
        # O smem at ODTYPE: the split fp32 partial doubles the O tile
        # (64x256: 32 KiB bf16 -> 64 KiB fp32). Budget check (the K/V tiles
        # dominate): bf16 leg 32+64+32+32 = 160 KiB, fp8 leg 16+32+64+16+16 =
        # 144 KiB — both under the ~227 KiB sm_110a per-CTA limit.
        sO_atom = sm80_utils.get_smem_layout_atom(ODTYPE, D)
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
                sO: cute.struct.Align[cute.struct.MemRange[ODTYPE, cute.cosize(sO_layout)], 1024]
                sQ: cute.struct.Align[cute.struct.MemRange[QDTYPE, cute.cosize(sQ_layout)], 1024]
                sK: cute.struct.Align[cute.struct.MemRange[KVTYPE, cute.cosize(sK_layout)], 1024]
            storage = smem.allocate(_SharedStorage)
            sVb = storage.sVb.get_tensor(sVb_layout)
        else:
            @cute.struct
            class _SharedStorage:
                sV: cute.struct.Align[cute.struct.MemRange[KVTYPE, cute.cosize(sV_layout)], 1024]
                sO: cute.struct.Align[cute.struct.MemRange[ODTYPE, cute.cosize(sO_layout)], 1024]
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
        # fp8 leg); O as ODTYPE (bf16 / fp32 split partial); K/V as KVTYPE.
        q_async_copy_elems = 128 // QDTYPE.width         # Q: 16 (fp8) / 8 (bf16)
        o_async_copy_elems = 128 // ODTYPE.width         # O: bf16 -> 8 / fp32 -> 4
        kv_async_copy_elems = 128 // KVTYPE.width        # K/V: 16 (fp8) / 8 (bf16)
        atom_async_copy_q = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), QDTYPE,
            num_bits_per_copy=128)
        atom_async_copy_kv = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), KVTYPE,
            num_bits_per_copy=128)
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), ODTYPE, num_bits_per_copy=128)
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
        # sK_b: the mma's OWN B-operand smem view (partition_B) — each fragment
        # slot _e maps to the correct (n_kv, d). The K fragment is loaded from
        # this ELEMENT-WISE (tSrK[_e] = sK_b[_e]); the b16-ldmatrix-typed-fp8
        # K-major tiled-copy retile (tSrK_cp = retile(tSrK) / tSsK =
        # partition_S) is BROKEN on sm_110a — it delivers only ~33 of the 256
        # d-values (probe: kdump) because the partition's k-tile does not map to
        # the smem d-stride. (The N-major/transpose=True V path is fine.)
        sK_b = thr_mma_qk.partition_B(sK[None, None, 0])
        tSrK = thr_mma_qk.make_fragment_B(sK_b)
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
            # The PV mma is bf16 (see above) but the V smem is fp8. tOsVt is the
            # V ldmatrix source for the bf16 PV mma (gemm_rs), so it must carry
            # the PV mma's B-fragment per-thread
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

        # ---- KV gmem source views (dense + varlen + paged) ------------------
        if const_expr(is_paged):
            # A5: flat pool (NP*PS, D) per kv_head — the head axis is mode 1 of
            # the pool view. The row coordinate is the flat-pool tile index
            # (ktile), computed per n_block from the page table (A8-A10);
            # local_tile tiles (TILE_N, D) over it with a dynamic row coordinate.
            mK_cur = mK[None, kv_head, None]
            mV_cur = mV[None, kv_head, None]
        elif const_expr(is_varlen_k):
            mK_cur = cute.domain_offset((kv_row0, 0), mK[None, None, kv_head])
            mV_cur = cute.domain_offset((kv_row0, 0), mV[None, None, kv_head])
        else:
            mK_cur = mK[None, None, kv_head, batch_idx]
            mV_cur = mV[None, None, kv_head, batch_idx]
        gK = cute.local_tile(mK_cur, (TILE_N, D), (None, 0))
        gV = cute.local_tile(mV_cur, (TILE_N, D), (None, 0))

        # ---- Q load (PackGQA: the GQA group is the M-dimension) -------------
        if const_expr(is_varlen_q):
            # The GQA-packed M-mode is (qhead, seqlen): offset the seqlen sub-mode
            # by q_row_base (the batch's global query row) via the NESTED coord
            # ((0, q_row_base), 0) — a flat (q_row_base, 0) applies the offset to
            # the qhead sub-mode (or drops it), so every batch read row 0's Q and
            # the varlen QK scores (hence LSE) were wrong for batch>0 (the O was
            # unaffected for L=1 because P=1 for a single kv row). Same idiom as
            # the varlen O store below.
            mQ_cur = cute.domain_offset(((0, q_row_base), 0), mQ[None, None, kv_head])
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
            # gtile: the GLOBAL tile. In non-split mode it is a pure alias of
            # nb (no arithmetic) so the codegen stays bit-identical to the
            # canonical kernel; in split mode it is the CTA's chunk offset + nb.
            if const_expr(IS_SPLIT):
                gtile = kv_start_tile + nb
            else:
                gtile = nb
            kv_limit = kv_len - gtile * TILE_N
            if const_expr(is_paged):
                # A8 (+SplitKV): gather this tile's physical page + the
                # flat-pool tile coordinate (ktile) that replaces the logical
                # tile. The table column is the page containing global row
                # gtile*TILE_N (== column 0 at ns=1, nb=0). (R1-PROVEN: a
                # runtime ktile lowers to a runtime gmem address through the
                # partitioned cp.async view — the same idiom as
                # probe_tc_paged_ktile.py T2.)
                pidx = mPageTable[batch_idx, (gtile * TILE_N) // PS]
                ktile = (pidx * PS + (gtile * TILE_N) % PS) // TILE_N
                tile_coord = ktile
            else:
                tile_coord = gtile
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if t0KcK[0, n, 0][0] < kv_limit:
                    cute.copy(
                        gmem_thr_copy_K, tKgK[None, n, None, tile_coord],
                        tKsK[None, n, None, 0], pred=tKpK[None, n, None])
            cute.arch.cp_async_commit_group()
            if const_expr(is_paged):
                # A10: V's OWN page-table read (its own pidx/ktile; identical
                # value while K and V share one pool layout, but V is gathered
                # independently so the pools can diverge later). Same gtile.
                pidx_v = mPageTable[batch_idx, (gtile * TILE_N) // PS]
                ktile_v = (pidx_v * PS + (gtile * TILE_N) % PS) // TILE_N
                tile_coord_v = ktile_v
            else:
                tile_coord_v = gtile
            for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                cute.copy(
                    gmem_thr_copy_V, tVgV[None, n, None, tile_coord_v],
                    tVsV[None, n, None, 0], pred=tVpV[None, n, None])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            if const_expr(is_paged):
                # ---- sm_110a paged codegen pin (compiled in the paged leg) --
                # The paged variant of this kernel is miscompiled by the
                # toolchain's register allocation: the un-pinned paged leg
                # emits NaN O / -inf LSE while the dense leg is bit-correct,
                # with correct kernel source and a correct MLIR trace (the
                # only PTX difference is register binding). Three smem reads +
                # three gmem stores in the first block deterministically avoid
                # the miscompile (validated across dtype x L x batch x page-
                # table shuffle); the smem->gmem store class is required —
                # smem stores or gmem reads alone do not fix it. The values go
                # into a per-device scratch buffer (mPin, interface-owned) that
                # is never read back.
                if const_expr(mPin is not None):
                    if tidx == 0 and kv_head == 0 and batch_idx == 0:
                        # fp8-safe: the cast goes through Float32 (an fp8/i32
                        # direct cast does not lower in NVVM on sm_110a).
                        mPin[0] = sK[0, 0, 0].to(Float32)
                        mPin[1] = sK[63, 0, 0].to(Float32)
                        mPin[2] = sK[1, 64, 0].to(Float32)
            if const_expr(FP8_KV):
                _qk_mma_fp8(tiled_mma_qk, acc_S, tSrQ_mma, tSrK, sK_b)
            else:
                sm80_utils.gemm(
                    thr_mma_qk, acc_S, tSrQ_mma, tSrK, tSsQ, tSsK[None, None, None, 0],
                    smem_thr_copy_Q, smem_thr_copy_K, A_in_regs=True)
            # OOB mask (first block, see steady-state note). The mask checks the
            # ABSOLUTE row (gtile*TILE_N + col >= kv_len) — the global tile. In
            # non-split mode gtile is a pure alias of nb==0 here, so the guard is
            # the canonical `kv_len < TILE_N` — keep it on its own branch so the
            # non-split codegen stays bit-identical to the reference kernel (a
            # folded gtile term here shifts the dense/paged codegen; see the
            # steady-state note). In split mode the first block is at
            # gtile==kv_start_tile, which IS the partial tail whenever this split
            # owns the last tile — e.g. L=97, ns=2: split 1 has a single block at
            # gtile=1 (rows 64..127, kv_len=97), `kv_len < TILE_N` is false, and
            # the OOB rows 97..127 were never masked -> wrong LSE (3.2e-1). Guard
            # the split path on the block's own remaining rows instead.
            if const_expr(IS_SPLIT):
                if kv_len - gtile * TILE_N < TILE_N:
                    _mask_partial_kv(acc_S, thr_mma_qk, gtile, kv_len, TILE_M, TILE_N)
            else:
                if kv_len < TILE_N:
                    _mask_partial_kv(acc_S, thr_mma_qk, gtile, kv_len, TILE_M, TILE_N)
            row_scale = softmax.online_softmax(acc_S, is_first=cutlass.const_expr(True))
            softmax.rescale_O(acc_O, row_scale)
            rP = cute.make_fragment_like(acc_S, PT)
            rP.store(acc_S.load().to(PT))
            tOrP = layout_utils.reshape_acc_to_frgA(rP)
            if const_expr(FP8_KV):
                # Stage V fp8 -> bf16, then the PV ldmatrix reads sVb exactly
                # like the bf16 leg (proven; an fp8 B-operand faults on sm_110a).
                _convert_v_to_bf16(sV, sVb, tidx, TILE_N, D, self.num_threads)
                # Barrier: sVb is written by ALL threads above and read by ALL
                # threads in the gemm_rs ldmatrix below. Without it a warp can
                # ldmatrix-read sVb before another warp's convert retires —
                # 1st block reads uninitialized smem (NaN), later blocks read
                # the previous block's stale V (the large finite O error).
                cute.arch.sync_threads()
            sm80_utils.gemm_rs(
                thr_mma_pv, acc_O, tOrP, tOrVt, tOsVt[None, None, None, 0],
                smem_thr_copy_V)

        # Steady-state n_blocks (is_first=False).
        for nb in cutlass.range(1, num_nblocks):
            if const_expr(IS_SPLIT):
                gtile = kv_start_tile + nb  # GLOBAL tile (chunk offset + nb)
            else:
                gtile = nb                 # pure alias: canonical codegen
            kv_limit = kv_len - gtile * TILE_N
            if const_expr(is_paged):
                # A9 (+SplitKV): per-tile physical page gather (a RUNTIME page
                # index — the R1-PROVEN idiom): which physical page holds this
                # tile's TILE_N kv rows, then the flat-pool tile coordinate
                # within it. The table column is the page containing GLOBAL row
                # gtile*TILE_N, i.e. (gtile*TILE_N)//PS — NOT gtile//PS (the
                # tile is a block index, PS a row count; gtile//PS stays 0
                # until tile PS and would re-read page 0).
                pidx = mPageTable[batch_idx, (gtile * TILE_N) // PS]
                ktile = (pidx * PS + (gtile * TILE_N) % PS) // TILE_N
                tile_coord = ktile
            else:
                tile_coord = gtile
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if t0KcK[0, n, 0][0] < kv_limit:
                    cute.copy(
                        gmem_thr_copy_K, tKgK[None, n, None, tile_coord],
                        tKsK[None, n, None, 0], pred=tKpK[None, n, None])
            cute.arch.cp_async_commit_group()
            if const_expr(is_paged):
                # A10: V's own pidx/ktile (independent gather, see first block).
                # Same gtile / column index as K's A9 read (comment there).
                pidx_v = mPageTable[batch_idx, (gtile * TILE_N) // PS]
                ktile_v = (pidx_v * PS + (gtile * TILE_N) % PS) // TILE_N
                tile_coord_v = ktile_v
            else:
                tile_coord_v = gtile
            for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                cute.copy(
                    gmem_thr_copy_V, tVgV[None, n, None, tile_coord_v],
                    tVsV[None, n, None, 0], pred=tVpV[None, n, None])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            if const_expr(FP8_KV):
                _qk_mma_fp8(tiled_mma_qk, acc_S, tSrQ_mma, tSrK, sK_b)
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
                _mask_partial_kv(acc_S, thr_mma_qk, gtile, kv_len, TILE_M, TILE_N)
            row_scale = softmax.online_softmax(acc_S, is_first=cutlass.const_expr(False))
            softmax.rescale_O(acc_O, row_scale)
            rP = cute.make_fragment_like(acc_S, PT)
            rP.store(acc_S.load().to(PT))
            tOrP = layout_utils.reshape_acc_to_frgA(rP)
            if const_expr(FP8_KV):
                _convert_v_to_bf16(sV, sVb, tidx, TILE_N, D, self.num_threads)
                # sVb write (all threads) must complete before the gemm_rs
                # ldmatrix read (all threads) — see the first-block barrier.
                cute.arch.sync_threads()
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

        # LSE: the VALUE is the same in split mode (row_sum = the chunk LSE,
        # natural units; an empty chunk already holds -inf from finalize) — only
        # the destination differs (mLSE was repointed to mLSE_partial[split_idx]
        # in the packing block above).
        if const_expr(mLSE is not None or mLSE_partial is not None):
            if const_expr(is_varlen_q):
                # Varlen LSE (H, total_q) is transposed to (total_q, H) above and
                # packed to ((qhead, seqlen), nkv) — the nkv mode is flat, so
                # mLSE[None, kv_head] selects the right kv_head. The batch's LSE
                # rows start at seqlen index q_row_base: carry that base in the
                # store coordinate (row_base) rather than a domain_offset — a
                # domain_offset adds the dynamic q_row_base on top of the (already
                # dynamic) slice base and the DSL 16-B-aligns it in transit,
                # dropping/corrupting the per-CTA base (every CTA wrote the same
                # low cells -> cross-batch clobber). The O store is unaffected (it
                # goes through TiledCopy, whose domain_offset path is correct).
                mLSE_cur = mLSE[None, kv_head]
                pack_gqa.store_LSE(
                    mLSE_cur, softmax.row_sum, tiled_mma_pv, tidx, 0, seqlen_q,
                    row_base=q_row_base,
                )
            else:
                mLSE_cur = mLSE[None, kv_head, batch_idx]
                pack_gqa.store_LSE(mLSE_cur, softmax.row_sum, tiled_mma_pv, tidx, 0, seqlen_q)

        # O: acc_O (rmem) -> sO (smem) -> rmem fragment -> gmem (the Sm80 store
        # path). ODTYPE-typed: bf16 non-split (byte-identical to before) /
        # fp32 split partial. The VALUE is the same in split mode (acc_O after
        # finalize() + rescale_O + v_descale = the chunk-normalized O) — only
        # the destination (mO_partial[split_idx]) and dtype differ.
        rO = cute.make_fragment_like(acc_O, ODTYPE)
        rO.store(acc_O.load().to(ODTYPE))
        smem_copy_atom_O = utils.get_smem_store_atom(110, ODTYPE)
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma_pv).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)
        cute.arch.sync_threads()
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOrO = cute.make_fragment_like(tOsO, ODTYPE)
        cute.autovec_copy(tOsO, tOrO)
        if const_expr(IS_SPLIT):
            # mO_partial is host-packed like mO plus a leading (num_splits)
            # plane. A bare mO_partial[split_idx] is an ELEMENT access in this
            # DSL, so build the CTA's plane by offsetting the base pointer by
            # split_idx * (elems per split block) and dropping the (ns) mode —
            # the GEMV idiom — then apply the SAME QO permute + GQA pack the
            # final mO got in __call__ (the select mirrors QO_trans; the pack
            # mirrors pack_gqa_layout(mO, ..., head_idx=2)), so the store path
            # is identical — only the base + dtype differ.
            if const_expr(is_varlen_q):
                o_stride = mO_partial.shape[1] * mO_partial.shape[2] * mO_partial.shape[3]
                o_sub = cute.select(mO_partial.layout, mode=[1, 2, 3])
            else:
                o_stride = (mO_partial.shape[1] * mO_partial.shape[2]
                            * mO_partial.shape[3] * mO_partial.shape[4])
                o_sub = cute.select(mO_partial.layout, mode=[1, 2, 3, 4])
            mO_sp = cute.make_tensor(mO_partial.iterator + split_idx * o_stride, o_sub)
            if const_expr(is_varlen_q):
                mO_sp = cute.make_tensor(mO_sp.iterator, cute.select(mO_sp.layout, mode=[0, 2, 1]))
            else:
                mO_sp = cute.make_tensor(mO_sp.iterator, cute.select(mO_sp.layout, mode=[1, 3, 2, 0]))
            # mO.shape[2] is the packed nheads_kv (the LSE block's num_head_kv
            # idiom) — pack_gqa_layout's nheads_kv arg, NOT the pre-pack 24.
            mO_sp = pack_gqa_layout(mO_sp, G, mO.shape[2], head_idx=2)
        else:
            mO_sp = mO
        if const_expr(is_varlen_q):
            # The GQA-packed M-mode is (qhead, total_q): offset only the
            # total_q sub-mode by q_row_base (the batch's global query row) —
            # a scalar would land on the fastest (total_q) sub-mode only when
            # the strides happen to agree, so be explicit.
            mO_cur = cute.domain_offset(((0, q_row_base), 0), mO_sp[None, None, kv_head])
        else:
            mO_cur = mO_sp[None, None, kv_head, batch_idx]
        pack_gqa.store_O(mO_cur, tOrO, gmem_tiled_copy_O, tidx, 0, seqlen_q)
