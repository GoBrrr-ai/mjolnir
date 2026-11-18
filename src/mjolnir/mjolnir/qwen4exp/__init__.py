"""Qwen3.8-Flash-Next (vLLM ``qwen4_exp``) serving pieces for Thor.

Opt-in modules that the ``thor-qwen4-*`` image patches import (see
``docker/vllm-thor/PATCHES.md``, "Qwen3.8-Flash-Next (qwen4_exp) serving
stack"). Each is inert unless its environment switch is set:

* ``ple_disk`` / ``ple_disk_table`` / ``ple_disk_cache`` — a GPU slot-pool
  cache and prefetch in front of vLLM's PLE disk table (``disk_offload`` +
  ``MJOLNIR_PLE_DISK_CACHE_GB``).
"""
