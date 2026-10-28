"""The Thor (sm_110) kernel: the traced T5 decode family.

Layout (one directory per implemented architecture; the surface contract
is in the package docstring of :mod:`mjolnir.arch`):

* :mod:`mjolnir.arch.thor.decode` — the CuTe-DSL tcgen05 kernel (the
  single source of truth for the kernel; BSD-3-Clause, NVIDIA-derived);
* :mod:`mjolnir.arch.thor.call` — the host-side builder traced at
  compile time;
* :mod:`mjolnir.arch.thor.const` — shared kernel constants + math
  helpers (the leaf module);
* :mod:`mjolnir.arch.thor.plan` — the occupancy-pinned split-KV planner
  and the stride vectors;
* :mod:`mjolnir.arch.thor.runner` — the runner (plan buffers, compile
  cache, launch, KV views): what the vLLM backend instantiates;
* :mod:`mjolnir.arch.thor.harness` — the standalone harness (the
  benchmark driver + the emulator): ``python -m mjolnir.arch.thor.harness``.

The backend resolves the class through :func:`mjolnir.platforms.load_kernel`
at build time — it never imports a family module directly, so landing a
family edits nothing but ``platforms._SHIPPED`` and its own subpackage.
Importing this package pulls the CUTLASS DSL (heavy): only do it when a
device is real.
"""

from mjolnir.runner import (
    SUPPORTED_PAGE_SIZES,
    StepPlan,
    detect_kv_layout,
    env_flag,
    split_kv_cache,
)
from .runner import T5DecodeRunner

Runner = T5DecodeRunner

__all__ = [
    "SUPPORTED_PAGE_SIZES",
    "Runner",
    "StepPlan",
    "T5DecodeRunner",
    "detect_kv_layout",
    "env_flag",
    "split_kv_cache",
]
