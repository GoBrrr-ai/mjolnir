"""The Thor (sm_110) kernel: re-exports of the exact runner surface the vLLM
backend consumes. Adding an architecture = adding a sibling subpackage with
the same surface and a ``_SHIPPED`` row in ``mjolnir.platforms``:

* ``Runner`` — the decode-runner class the backend instantiates; the one
  name a family subpackage must define itself (all others are re-exports of
  shared, family-independent machinery from :mod:`mjolnir.runner`);
* the shared helpers every family consumes unchanged: ``StepPlan``,
  ``SUPPORTED_PAGE_SIZES``, ``detect_kv_layout``, ``env_flag``,
  ``split_kv_cache``.

The backend resolves the class through :func:`mjolnir.platforms.load_kernel`
at build time — it never imports a family module directly, so landing a
family edits nothing but ``platforms._SHIPPED`` and its own subpackage.
Importing this module pulls the CUTLASS DSL (heavy): only do it when a
device is real. The compiled-kernel cache fingerprint
(``runner._cache_fingerprint``) hashes the kernel module the shared runner
imports — family-blind until the runner itself is split per family: a
second family must extend the fingerprint (or key its own cache) or it
would reuse this family's objects.
"""

from mjolnir.runner import (
    SUPPORTED_PAGE_SIZES,
    StepPlan,
    T5DecodeRunner,
    detect_kv_layout,
    env_flag,
    split_kv_cache,
)

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
