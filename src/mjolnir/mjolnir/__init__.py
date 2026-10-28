"""Mjolnir — fp8 paged decode attention for vLLM on Jetson AGX Thor (sm_110).

The kernel (codename **T5**, ``mjolnir.arch.thor.decode``) is a CuTe-DSL tcgen05 +
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

__all__ = ["BACKEND_NAME", "BACKEND_PATH", "__version__", "register"]
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
        from vllm.attention.backends.registry import (
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
    # vLLM > 0.32.0: the fp6_e2m3 KV dtype registers through the pluggable
    # dtype API (vLLM #52670) instead of the image patch — a no-op on
    # 0.32.0 and on a base where the patch already added it.
    try:
        from .kvdtype import register as _register_kv_dtype

        _register_kv_dtype()
    except Exception as e:  # noqa: BLE001 - never break vLLM startup
        import logging

        logging.getLogger("vllm.mjolnir").warning("Mjolnir fp6 KV dtype not registered: %r", e)
    if os.environ.get("MJOLNIR_ATTN_DEBUG", "").strip():
        # The IMA hunt: the engine core is a child process that inherits this
        # env before its CUDA runtime initializes, so faulting kernels are
        # named at their launch instead of surfacing on a later sync.
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    enum, register_backend = _registry()
    if BACKEND_NAME in enum.__members__:
        register_backend(enum[BACKEND_NAME], BACKEND_PATH)
    register_backend(enum.CUSTOM, BACKEND_PATH)
    # opt-in lm_head experiments (fp8 head, trimmed draft vocabulary)
    if os.environ.get("MJOLNIR_LM_HEAD", "").strip() or os.environ.get("MJOLNIR_DRAFT_VOCAB", "").strip():
        try:
            from .lmhead import install_hooks

            install_hooks()
        except Exception as e:  # noqa: BLE001 - never break vLLM startup
            import logging

            logging.getLogger("vllm.mjolnir").warning("Mjolnir lm_head hooks not installed: %r", e)
