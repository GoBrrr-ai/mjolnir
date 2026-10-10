"""Mjolnir — fp8 paged decode attention for vLLM on Jetson AGX Thor (sm_110).

The kernel (codename **T5**, ``mjolnir.t5_decode``) is a CuTe-DSL tcgen05 +
TMEM + TMA decode kernel; ``mjolnir.backend`` wraps it as a vLLM attention
backend and ``mjolnir.runner`` holds the plan buffers, compile cache and KV
views.

``register()`` is the vLLM general-plugin entry point
(``vllm.general_plugins`` → ``mjolnir = "mjolnir:register"``). It points the
backend enum at ``mjolnir.backend.MjolnirAttentionBackend``:

* ``AttentionBackendEnum.MJOLNIR`` — present in the Mjolnir image (the
  ``thor-mjolnir-attention-backend`` patch adds the member), so
  ``vllm serve <model> --attention-backend MJOLNIR`` works;
* ``AttentionBackendEnum.CUSTOM`` — always, so a stock vLLM can use
  ``--attention-backend CUSTOM``.

The backend module (and with it vLLM's attention stack and the CUTLASS DSL)
is imported lazily, only when vLLM resolves the backend.
"""

from __future__ import annotations

import os

__all__ = ["register", "BACKEND_PATH", "BACKEND_NAME", "__version__"]
__version__ = "0.1.0"

BACKEND_NAME = "MJOLNIR"
BACKEND_PATH = "mjolnir.backend.MjolnirAttentionBackend"


def _disabled() -> bool:
    for name in ("MJOLNIR_ATTN_DISABLE", "THOR_T5_DISABLE"):
        if os.environ.get(name, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _registry():
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    except ImportError:  # older vLLM layout
        from vllm.attention.backends.registry import (  # type: ignore
            AttentionBackendEnum,
            register_backend,
        )
    return AttentionBackendEnum, register_backend


def has_native_member() -> bool:
    """True when this vLLM has the ``MJOLNIR`` enum member (patched vLLM)."""
    try:
        enum, _ = _registry()
    except ImportError:
        return False
    return BACKEND_NAME in enum.__members__


def register() -> None:
    if _disabled():
        return
    enum, register_backend = _registry()
    if BACKEND_NAME in enum.__members__:
        register_backend(enum[BACKEND_NAME], BACKEND_PATH)
    register_backend(enum.CUSTOM, BACKEND_PATH)
