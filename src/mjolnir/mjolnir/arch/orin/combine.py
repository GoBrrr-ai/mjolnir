"""The fused split-KV combine for the Orin MMA decode kernel.

The runner's partials are (ns, rows, h_q, d) fp32 O and (ns, b, h_q, ql)
fp32 LSE; the merge is O = Σ_s exp(lse_s − L)·O_s / Σ_s exp(lse_s − L)
with L = max_s lse_s (the chunk-normalized contract the kernels write).
The eager torch form issued ~20 small kernels (~150 us of GPU per call
— more than the decode kernel itself); this is one kernel, one pass
over the partials, the FlashInfer MergeStates cost class.

One block per (row, q-head): the block loads its ns LSE values, then
each thread accumulates its two d elements across the splits. d is
baked at 256 (the MMA kernel's HEAD_DIM); ns/rows/h_q/b/ql are runtime
Int32s so one compile serves every capture shape. The tensor layouts
are the runner's capacity buffers (fixed at construction, so the baked
extents are stable). The empty-chunk splits write lse = -inf -> weight
0; an all-empty row normalizes to 0 (the torch combine's clamp_min) —
L is floored at -FLT_MAX so that row never computes -inf - -inf.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

D = 256  # the MMA kernel's HEAD_DIM
NUM_THREADS = 128
_TINY = 1.1754944e-35  # torch.finfo(torch.float32).tiny, rounded
_NEG_FLOOR = -3.4028234e38  # -torch.finfo(torch.float32).max: the finite L floor


@cute.kernel
def combine_kernel(
    mO_part: cute.Tensor,  # (ns_cap*rows_cap*h_q*D) fp32 flat
    mLse: cute.Tensor,  # (ns_cap*b_cap*h_q*ql_cap) fp32 flat
    mOut: cute.Tensor,  # (rows_cap*h_q*D) bf16 flat
    max_ns: cutlass.Constexpr[int],  # the unroll/predication bound (= ns_cap)
    ns: Int32,
    rows: Int32,
    h_q: Int32,
    b: Int32,
    ql: Int32,
):
    bid = cute.arch.block_idx()[0]
    tidx = cute.arch.thread_idx()[0]
    r = bid // h_q
    h = bid - r * h_q
    bb = r // ql
    qq = r - bb * ql
    ls_s = b * h_q * ql
    ls_b = h_q * ql

    # Both loops unroll over max_ns with the s < ns predication: the
    # dynamic-range form serialized every split's load behind the loop
    # branch (~34 us; the FI merge overlaps all partial rows in flight
    # and lands at 11). Unrolled, the loads issue independently and the
    # latency hides under the acc chain.
    # 1. L = max_s lse[s, h, r], floored at a FINITE -FLT_MAX: a row whose
    #    every split is empty (all lse = -inf — a CUDA-graph padding request
    #    with seq_len 0, or a dummy/profile step) would otherwise get
    #    L = -inf and w = exp2(-inf - -inf) = NaN, a NaN output row instead
    #    of the documented 0 (and a NaN row poisons any per-tensor reduction
    #    downstream). With the floor, every -inf lse weighs exactly 0 and a
    #    real lse (>= -FLT_MAX) is unaffected.
    L = Float32(_NEG_FLOOR)
    for s in cutlass.range_constexpr(max_ns):
        if s < ns:
            L = cutlass.max(L, mLse[s * ls_s + bb * ls_b + h * ql + qq])

    # 2. the weighted sum over the splits; the exp weights recomputed per
    #    element pass (ns <= 64 exp2s are nothing next to the loads).
    ln2e = Float32(1.4426950408889634)  # 1/ln 2: exp(x) = exp2(x*log2 e)
    sw = Float32(0.0)
    acc0 = Float32(0.0)
    acc1 = Float32(0.0)
    for s in cutlass.range_constexpr(max_ns):
        if s < ns:
            w = cute.math.exp2((mLse[s * ls_s + bb * ls_b + h * ql + qq] - L) * ln2e, fastmath=True)
            sw = sw + w
            obase = s * rows * h_q * D + r * h_q * D + h * D
            acc0 = acc0 + w * mO_part[obase + tidx]
            acc1 = acc1 + w * mO_part[obase + tidx + NUM_THREADS]
    scale = 1.0 / cutlass.max(sw, Float32(_TINY))
    obase = r * h_q * D + h * D
    mOut[obase + tidx] = (acc0 * scale).to(mOut.element_type)
    mOut[obase + tidx + NUM_THREADS] = (acc1 * scale).to(mOut.element_type)


@cute.jit
def combine_launch(
    mO_part: cute.Tensor,
    mLse: cute.Tensor,
    mOut: cute.Tensor,
    max_ns: cutlass.Constexpr[int],
    ns: Int32,
    rows: Int32,
    h_q: Int32,
    b: Int32,
    ql: Int32,
):
    combine_kernel(mO_part, mLse, mOut, max_ns, ns, rows, h_q, b, ql).launch(
        grid=(rows * h_q, 1, 1),
        block=(NUM_THREADS, 1, 1),
    )
