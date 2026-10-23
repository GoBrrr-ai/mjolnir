"""The Orin (``sm_87``) kernel family: the GEMV FMA decode kernel.

One subpackage per implemented architecture, sibling to
:mod:`mjolnir.arch.thor` (see that module's docstring for the surface
contract). The family mirrors the Thor layout:

* :mod:`mjolnir.arch.orin.decode` — the GEMV FMA kernel (restored from
  the ``gemv-archive`` tag, re-targeted at ``sm_87`` only); the FMA
  path is the M=1 leg (FlashDecoding++ measured GEMV +22% over tensor
  cores at M=1), with an mma.sync variant planned for the M>1
  speculative-verify shape;
* :mod:`mjolnir.arch.orin.plan` — the split-KV planner (FlashInfer
  Alg. 1, board-agnostic: the SM count is a device query, so AGX Orin
  64/32 GB, Orin NX and Orin Nano all plan correctly);
* the runner and harness land here next — the surface the vLLM backend
  consumes (``Runner`` + the shared :mod:`mjolnir.runner` helpers) is
  not exported yet, and ``mjolnir.platforms._SHIPPED`` still marks orin
  as unimplemented until it exists.

Do not import this package on a device whose compute capability is not
(8, 7): the DSL compile and the module's budget asserts assume GA10B.
"""

from .runner import GEMVDecodeRunner

__all__ = ["GEMVDecodeRunner", "Runner"]


class Runner(GEMVDecodeRunner):
    """The backend-facing runner surface (``mjolnir.platforms.kernel_for``
    hands this to the vLLM backend). Accepts the shared backend kwargs and
    absorbs the Thor-only levers as inert attributes: the GEMV grid has no
    CTAs/SM knob, and PDL / L2-prefetch / evict-first hints do not exist
    below sm_90 — the attributes exist so the backend's builder logging and
    metadata paths read them unchanged."""

    def __init__(
        self,
        *,
        ctas_per_sm: int = 2,
        pdl: bool = False,
        l2_prefetch: int = 0,
        kv_evict_first: bool = False,
        **kw,
    ):
        super().__init__(**kw)
        self.ctas_per_sm = ctas_per_sm
        self.pdl = pdl
        self.l2_prefetch = l2_prefetch
        self.kv_evict_first = kv_evict_first
