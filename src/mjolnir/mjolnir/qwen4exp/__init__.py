"""Qwen3.8-Flash-Next (vLLM ``qwen4_exp``) serving pieces for Thor.

Opt-in modules that the ``thor-qwen4-*`` image patches import (see
``docker/vllm-thor/PATCHES.md``, "Qwen3.8-Flash-Next (qwen4_exp) serving
stack"). Each is inert unless its environment switch is set:

* ``ple_disk`` / ``ple_disk_table`` / ``ple_disk_cache`` — a GPU slot-pool
  cache and prefetch in front of vLLM's PLE disk table (``disk_offload`` +
  ``MJOLNIR_PLE_DISK_CACHE_GB``).
* ``draft_head`` — the MTP drafter's reduced vocabulary head
  (``MJOLNIR_DRAFT_VOCAB``) and its 8-bit options.
* ``screened_head`` — the screened target lm_head
  (``QWEN4EXP_LM_HEAD_SCREENED``, a backport of vLLM #60566).
* ``side_fp8`` — online FP8 for the target's BF16 dense side layers
  (``QWEN4EXP_SIDE_FP8``).
"""
