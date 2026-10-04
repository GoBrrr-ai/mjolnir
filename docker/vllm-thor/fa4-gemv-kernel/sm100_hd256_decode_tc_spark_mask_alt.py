"""The tScS direct-read OOB kv-mask — the committed (canonical) variant, preserved
as a reference alongside the live mn-reshape variant that is primary in the kernel.

Two masks exist for the TC hd256 decode kernel's n-OOB (partial-kv) -inf write:

1. **mn-reshape (primary, in `sm100_hd256_decode_tc_spark.py` `_mask_partial_kv`):**
   reads the absolute kv column through
   `layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS))` (the identity tensor
   `cS` partitioned by the mma), then materializes it to rmem before the -inf
   select. This is what the kernel ships with.

2. **tScS direct-read (this file):** reads the absolute kv column straight from the
   mma C-fragment coordinate tensor `tScS` at its own modes
   `tScS[(_r % 2, _r // 2), 0, _c][1]` — "probe-verified" in the committed tree
   (a coord read through the mn reshape was reported as garbage in that context,
   hence the direct own-modes indexing).

The live kernel uses variant 1. Variant 2 is preserved here so the committed
approach is not lost; if the mn-reshape path ever regresses on a new CUDA/CuTe-DSL
base, variant 2 is the known-good fallback. The two differ only in how the
per-thread absolute kv column is obtained; the -inf select and the mn-view row
round-trip (the access path `online_softmax` consumes) are identical.

Signature difference: variant 2 takes the `tScS` fragment tensor directly
(`_mask_partial_kv(acc_S, tScS, nb, kv_len, tile_n)`); variant 1 takes the
`thr_mma` + `tile_m`/`tile_n` and builds `tScS_mn` internally.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils  # noqa: F401
from cutlass import layout_utils

from cutlass import Float32, Int32
from cutlass.cute import const_expr


def mask_partial_kv_tscs_direct(
    acc_S: cute.Tensor,
    tScS: cute.Tensor,
    nb: Int32,
    kv_len: Int32,
    tile_n,
):
    """Variant 2 — tScS direct-read OOB kv-mask (the committed canonical form).

    Write -inf into the OOB kv-columns of the acc_S rows, through mn-view row
    round-trips (the exact access path online_softmax consumes — a linear
    `acc_S[i]` store alone is dropped before the row-vector loads see it).
    Per-element stores are UNCONDITIONAL (a runtime-`if` around an rmem element
    store is dropped at trace); the OOB select is a runtime ternary (the GEMV
    `s = s if ... else -inf` idiom). Per-thread the mn view's (r, c) slot maps to
    tScS own-modes (r%2, r//2, c) (probe-verified; a coord read through the mn
    reshape is garbage in this context).
    """
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


# Alias so a caller can drop this in with the same name as the kernel's variant.
_mask_partial_kv = mask_partial_kv_tscs_direct
