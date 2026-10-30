"""The Orin (``sm_87``) kernel family: the mma.sync tensor-core decode kernel.

One subpackage per implemented architecture, sibling to
:mod:`mjolnir.arch.thor` (see that module's docstring for the surface
contract). The family mirrors the Thor layout:

* :mod:`mjolnir.arch.orin.mma` — the tensor-core (mma.sync m16n8k16)
  decode kernel: fp8 and fp6-e2m3 KV, bf16 Q, head_dim 256, q_len <= 8
  (decode and the MTP verify). The family's only kernel;
* :mod:`mjolnir.arch.orin.plan` — the split-KV planner (FlashInfer
  Alg. 1, board-agnostic: the SM count is a device query, so AGX Orin
  64/32 GB, Orin NX and Orin Nano all plan correctly);
* :mod:`mjolnir.arch.orin.combine` — the fused split-KV LSE merge;
* :mod:`mjolnir.arch.orin.runner` — the plan buffers, compile cache and
  launch; :class:`Runner` below is the surface the vLLM backend consumes.

Do not import this package on a device whose compute capability is not
(8, 7): the DSL compile and the module's budget asserts assume GA10B.
"""

import logging

from mjolnir.runner import env_flag

from .runner import OrinDecodeRunner

__all__ = ["OrinDecodeRunner", "Runner"]

_log = logging.getLogger("vllm.mjolnir")

MMA_HEAD_DIM = 256  # mma.MMADecodeKernel.HEAD_DIM (not imported: that traces CUTLASS)
MMA_MAX_Q_LEN = 8  # mma.MMADecodeKernel.MAX_Q_LEN
MMA_KV_FORMATS = ("fp8", "e2m3")  # the mma kernel's KV formats


class Runner(OrinDecodeRunner):
    """The backend-facing runner surface (``mjolnir.platforms.kernel_for``
    hands this to the vLLM backend). Accepts the shared backend kwargs and
    absorbs the Thor-only levers as inert attributes: the Orin grid has no
    CTAs/SM knob, and PDL / L2-prefetch / evict-first hints do not exist
    below sm_90 — the attributes exist so the backend's builder logging and
    metadata paths read them unchanged.

    Two backend kwargs are translated to this family's meaning:

    * ``max_splits`` (``MJOLNIR_ATTN_MAX_SPLITS``, "max KV chunks per
      sequence") caps the auto split capacity — it used to pin it, so the
      backend's default 16 silently replaced the mma's 2-CTA-occupancy
      optimum of 8 (one wave on 16 SMs) with the over-split 16;
    * ``q_mode``: the mma kernel takes a bf16 query, so the runner always
      runs bf16 Q."""

    def __init__(
        self,
        *,
        ctas_per_sm: int = 2,
        pdl: bool = False,
        l2_prefetch: int = 0,
        kv_evict_first: bool = False,
        max_splits: int = 0,
        **kw,
    ):
        head_size = kw["head_size"]
        kv_format = kw.get("kv_format", "fp8")
        max_q_len = kw.get("max_q_len", 1)
        if head_size != MMA_HEAD_DIM:
            raise ValueError(f"the Orin mma kernel serves head_size {MMA_HEAD_DIM} only (got {head_size})")
        if kv_format not in MMA_KV_FORMATS:
            raise ValueError(f"the Orin mma kernel serves {'/'.join(MMA_KV_FORMATS)} KV only (got {kv_format!r})")
        if max_q_len > MMA_MAX_Q_LEN:
            raise ValueError(f"the Orin mma kernel serves q_len <= {MMA_MAX_Q_LEN} (got {max_q_len})")
        if env_flag("MJOLNIR_ATTN_ORIN_IMPL", "") not in ("", "mma"):
            raise ValueError("MJOLNIR_ATTN_ORIN_IMPL: only 'mma' is supported on Orin")
        kw["q_mode"] = "bf16"
        super().__init__(split_cap=max_splits, **kw)
        self.ctas_per_sm = ctas_per_sm
        self.pdl = pdl
        self.l2_prefetch = l2_prefetch
        self.kv_evict_first = kv_evict_first
