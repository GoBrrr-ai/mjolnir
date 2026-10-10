"""Thor T5 fp8 paged decode attention for vLLM (Jetson AGX Thor, sm_110).

`register()` is the vLLM general-plugin entry point: it maps the CUSTOM
attention backend to ThorT5AttentionBackend. The backend module (and with
it vLLM's attention stack, CUTLASS DSL, ...) is imported lazily, only
when vLLM resolves the backend.
"""

__all__ = ["register", "BACKEND_PATH"]
__version__ = "0.1.0"

BACKEND_PATH = "vllm_thor_t5.backend.ThorT5AttentionBackend"


def register() -> None:
    import os

    if os.environ.get("THOR_T5_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        return
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    except ImportError:  # older vLLM layout
        from vllm.attention.backends.registry import (  # type: ignore
            AttentionBackendEnum,
            register_backend,
        )
    register_backend(AttentionBackendEnum.CUSTOM, BACKEND_PATH)
