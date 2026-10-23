"""The sm_87 MMA decode kernel — the design lives in
research/runs/orin1/notes/mma-kernel-design.md; the GEMV's scalar-issue
roofline (the ALU pipe 59.6% of peak, the memory 8%) is what this
replaces. Phase 1: fp8 KV, 128 threads (the 6 GQA heads padded into the
M=16 tile), a 3-stage cp.async ring of 32-row stages, the dequant in
the consumer stage, mma.m16n8k16 bf16 for QK^T and PV.

Status: SCAFFOLD. The M1 milestone (the single-stage skeleton vs the
reference) lands next; the flag exists so the verify stages can A/B the
moment it computes. Until then the runner refuses the impl honestly.
"""

from __future__ import annotations

HEAD_DIM = 256  # the GEMV family constant: hd256 only


class MMADecodeKernel:
    """The traced CuTe kernel. The signature mirrors the GEMV's
    (GEMVDecodeKernel.__call__) so the runner's arg packing is shared —
    see the design doc §7 for what changes (the smem ring, the mma
    fragments, the cross-warp softmax)."""

    HEAD_DIM = HEAD_DIM

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "the Orin MMA decode kernel is not implemented yet — "
            "see research/runs/orin1/notes/mma-kernel-design.md (M1 next); "
            "use --impl gemv"
        )
