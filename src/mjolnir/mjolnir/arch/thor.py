"""The Thor (sm_110) kernel: re-exports of the exact runner surface the vLLM
backend consumes. Adding an architecture = adding a sibling subpackage with
the same surface and a ``_SHIPPED`` row in ``mjolnir.platforms``."""

from mjolnir.runner import (
    SUPPORTED_PAGE_SIZES,
    StepPlan,
    T5DecodeRunner,
    detect_kv_layout,
    env_flag,
    split_kv_cache,
)

__all__ = [
    "SUPPORTED_PAGE_SIZES",
    "StepPlan",
    "T5DecodeRunner",
    "detect_kv_layout",
    "env_flag",
    "split_kv_cache",
]
