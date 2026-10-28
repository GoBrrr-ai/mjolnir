"""The fp6 KV-dtype plugin (``mjolnir.kvdtype``) — the #52670 contract.

The handler registers ``fp6_e2m3`` through vLLM's pluggable KV-dtype API
(``register_kv_cache_dtype``, merged 2026-10-20 — first release after
0.32.0) so the ``thor-fp6-e2m3-kv-cache`` image patch can retire at the
next base bump. Pinned here:

* without the API (vLLM <= 0.32.0, i.e. every env without a post-#52670
  vllm) ``register()`` is a False-returning no-op — no import error leaks;
* the handler answers the exact contract the merged PR's own tests pin:
  storage dtype ``uint8`` (the packed 4-codes-in-3-bytes cell) and
  ``KVQuantMode.CUSTOM`` (quantized, self-managed kernels);
* re-registration / a patched base that already knows the dtype is a
  no-op, not an overwrite.
"""

from __future__ import annotations

import mjolnir.kvdtype as kvdtype


def test_register_noop_without_the_vllm_api(monkeypatch):
    """No pluggable-dtype API (0.32.0 and older): clean False, no raise."""
    import builtins

    real_import = builtins.__import__

    def no_vllm(name, *args, **kwargs):
        if name.startswith("vllm"):
            raise ImportError(f"no vllm in this env ({name})")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_vllm)
    assert kvdtype.register() is False


def test_handler_contract(monkeypatch):
    """With the API present, the handler pins the fp6 storage + quant mode
    and skips when the dtype is already known (patched base)."""
    import sys
    import types

    registered: dict[str, object] = {}
    dtypes: list[str] = []

    class KVQuantMode:
        CUSTOM = "CUSTOM"

    fake_cache = types.ModuleType("vllm.config.cache")
    fake_cache.KV_CACHE_DTYPES = dtypes  # ty: ignore[unresolved-attribute] — test stub

    def register_kv_cache_dtype(name):
        def _decorate(cls):
            dtypes.append(name)
            registered[name] = cls
            return cls

        return _decorate

    fake_cache.register_kv_cache_dtype = register_kv_cache_dtype  # ty: ignore[unresolved-attribute] — test stub
    fake_v1 = types.ModuleType("vllm.v1.kv_cache_interface")
    fake_v1.KVQuantMode = KVQuantMode  # ty: ignore[unresolved-attribute] — test stub
    fake_vllm = types.ModuleType("vllm")
    fake_config = types.ModuleType("vllm.config")
    fake_torch = types.ModuleType("torch")
    fake_torch.uint8 = "torch.uint8"  # ty: ignore[unresolved-attribute] — test stub
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", fake_config)
    monkeypatch.setitem(sys.modules, "vllm.config.cache", fake_cache)
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.kv_cache_interface", fake_v1)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert kvdtype.register() is True
    assert "fp6_e2m3" in dtypes
    handler = registered["fp6_e2m3"]()  # ty: ignore[call-non-callable] — test stub
    assert handler.torch_dtype() is fake_torch.uint8
    assert handler.quant_mode() is KVQuantMode.CUSTOM

    # A base that already knows the dtype (the image patch, or a second
    # register()): no-op, no overwrite.
    assert kvdtype.register() is False
