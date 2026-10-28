"""The fp6 (e2m3) KV-cache dtype as a vLLM pluggable dtype (vLLM #52670).

vLLM's ``register_kv_cache_dtype`` (merged 2026-10-20, first release after
0.32.0) lets a backend ship its own ``--kv-cache-dtype`` value without
patching vLLM core — exactly the four edits the
``thor-fp6-e2m3-kv-cache`` image patch hand-applies to 0.32.0 (the
``CacheDType`` literal, ``STR_DTYPE_TO_TORCH_DTYPE``, ``get_kv_quant_mode``
and ``is_quantized_kv_cache``). Once the base ships the API, this module
replaces that patch: the dtype is registered from the plugin entry point,
the storage dtype is the packed-uint8 cell view and ``KVQuantMode.CUSTOM``
marks the format as quantized-but-self-managed (the Mjolnir kernel reads
the packed pool directly; the FA4 fallback expands a step's pages to e4m3).

Inert on vLLM without the API (0.32.0 and older — the image patch owns the
dtype there) and on a base that already knows the dtype (a patched image
during a transition): both are correct no-ops.
"""

from __future__ import annotations


def register() -> bool:
    """Register ``fp6_e2m3`` with vLLM's pluggable KV-dtype API.

    Returns True when registered. False is the correct no-op when the API
    does not exist (vLLM <= 0.32.0: the image patch owns the dtype) or the
    dtype is already known (a patched base carrying both).
    """
    try:
        from vllm.config.cache import KV_CACHE_DTYPES, register_kv_cache_dtype
        from vllm.v1.kv_cache_interface import KVQuantMode
    except ImportError:
        return False
    if "fp6_e2m3" in KV_CACHE_DTYPES:
        return False
    import torch

    @register_kv_cache_dtype("fp6_e2m3")
    class _Fp6E2M3:
        name = "fp6_e2m3"

        def torch_dtype(self) -> torch.dtype:
            # The packed cell: 4 e2m3 codes in 3 bytes (d*6/8 bytes per K/V
            # row); the backend publishes the cell size via customize_spec.
            return torch.uint8

        def quant_mode(self) -> KVQuantMode:
            # Quantized, but no generic kernel mode: the Mjolnir backend
            # owns every kernel that touches the pool (decode reads it
            # directly; the fallback expands to e4m3 through kvfp6).
            return KVQuantMode.CUSTOM

    return True
