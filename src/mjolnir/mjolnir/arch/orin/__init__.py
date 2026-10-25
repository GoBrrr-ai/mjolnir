"""The Orin (``sm_87``) kernel family: the mma.sync and GEMV FMA decode kernels.

One subpackage per implemented architecture, sibling to
:mod:`mjolnir.arch.thor` (see that module's docstring for the surface
contract). The family mirrors the Thor layout:

* :mod:`mjolnir.arch.orin.mma` — the tensor-core (mma.sync m16n8k16)
  decode kernel: fp8 KV, bf16 Q, head_dim 256, q_len <= 8 (decode and
  the MTP verify). The serving default where it applies;
* :mod:`mjolnir.arch.orin.decode` — the GEMV FMA kernel (restored from
  the ``gemv-archive`` tag, re-targeted at ``sm_87`` only): every KV
  format (fp8 / fp6 e2m3 / bf16), the fallback kernel of the family;
* :mod:`mjolnir.arch.orin.plan` — the split-KV planner (FlashInfer
  Alg. 1, board-agnostic: the SM count is a device query, so AGX Orin
  64/32 GB, Orin NX and Orin Nano all plan correctly);
* :mod:`mjolnir.arch.orin.combine` — the fused split-KV LSE merge;
* :mod:`mjolnir.arch.orin.runner` — the plan buffers, compile cache and
  launch; :class:`Runner` below is the surface the vLLM backend consumes
  (``MJOLNIR_ATTN_ORIN_IMPL`` = ``mma`` | ``gemv`` picks the kernel).

Do not import this package on a device whose compute capability is not
(8, 7): the DSL compile and the module's budget asserts assume GA10B.
"""

import logging

from mjolnir.runner import env_flag

from .runner import GEMVDecodeRunner

__all__ = ["GEMVDecodeRunner", "Runner", "select_impl"]

_log = logging.getLogger("vllm.mjolnir")

MMA_HEAD_DIM = 256  # mma.MMADecodeKernel.HEAD_DIM (not imported: that traces CUTLASS)
MMA_MAX_Q_LEN = 8  # mma.MMADecodeKernel.MAX_Q_LEN


def select_impl(*, head_size: int, kv_format: str, max_q_len: int, requested: str | None = None) -> tuple[str, str]:
    """The Orin serving kernel for this layer group: ``(impl, reason)``.

    ``MJOLNIR_ATTN_ORIN_IMPL`` = ``mma`` (default) | ``gemv``. The mma kernel
    (tensor-core, the faster one across the measured band) serves fp8 KV at
    head_dim 256 and q_len <= 8 with a bf16 query; anything else (fp6 /
    bf16 KV) runs the GEMV, which covers every format."""
    want = (requested if requested is not None else env_flag("MJOLNIR_ATTN_ORIN_IMPL", "mma")).lower() or "mma"
    if want not in ("mma", "gemv"):
        raise ValueError(f"MJOLNIR_ATTN_ORIN_IMPL={want!r}: expected mma or gemv")
    if want == "gemv":
        return "gemv", "requested"
    if kv_format != "fp8":
        return "gemv", f"the mma kernel reads fp8 KV only (kv={kv_format})"
    if head_size != MMA_HEAD_DIM:
        return "gemv", f"the mma kernel serves head_dim {MMA_HEAD_DIM} only (got {head_size})"
    if max_q_len > MMA_MAX_Q_LEN:
        return "gemv", f"the mma kernel serves q_len <= {MMA_MAX_Q_LEN} (got {max_q_len})"
    return "mma", "default"


class Runner(GEMVDecodeRunner):
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
    * ``q_mode``: the mma kernel takes a bf16 query (the fp8-Q pack is the
      GEMV's), so an mma runner always runs bf16 Q."""

    def __init__(
        self,
        *,
        ctas_per_sm: int = 2,
        pdl: bool = False,
        l2_prefetch: int = 0,
        kv_evict_first: bool = False,
        max_splits: int = 0,
        impl: str | None = None,
        **kw,
    ):
        impl, why = select_impl(
            head_size=kw["head_size"],
            kv_format=kw.get("kv_format", "fp8"),
            max_q_len=kw.get("max_q_len", 1),
            requested=impl,
        )
        if impl == "mma":
            kw["q_mode"] = "bf16"
        elif why not in ("default", "requested"):
            _log.info("Mjolnir (orin): the GEMV kernel serves this group: %s", why)
        super().__init__(impl=impl, split_cap=max_splits, **kw)
        self.ctas_per_sm = ctas_per_sm
        self.pdl = pdl
        self.l2_prefetch = l2_prefetch
        self.kv_evict_first = kv_evict_first
