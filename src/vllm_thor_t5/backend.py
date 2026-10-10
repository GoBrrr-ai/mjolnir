"""vLLM attention backend: T5 fp8 paged decode on Jetson Thor (sm_110).

Design
------
ThorT5AttentionBackend is a *subclass of a stock backend* (FlashInfer by
default, THOR_T5_FALLBACK picks another). It inherits the KV-cache layout,
the KV-cache write, prefill, validation and everything else. Only two
things change:

  * the metadata builder: a step whose batch is decode-only (uniform
    q_len <= 1 + num_speculative_tokens, causal, no cascade) gets a T5 plan
    (occupancy-pinned split-KV, written into persistent buffers, padded to a
    fixed grid -> FULL CUDA graphs replay it). Any other step is built by
    the base builder and run by the base impl, unchanged.
  * the impl: forward() runs the T5 kernel when it gets T5 metadata,
    otherwise it calls the base forward.

Register with `--attention-backend CUSTOM` (the plugin entry point in
pyproject.toml registers the class at vLLM start-up).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from .runner import (
    SUPPORTED_PAGE_SIZES,
    StepPlan,
    T5DecodeRunner,
    detect_kv_layout,
    env_flag,
    split_kv_cache,
)

logger = logging.getLogger("vllm.thor_t5")  # under vLLM's handler

# --------------------------------------------------------------------------
# vLLM imports (paths moved between releases; try both)
# --------------------------------------------------------------------------
try:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
except ImportError:  # older layout
    from vllm.attention.backends.registry import AttentionBackendEnum  # type: ignore

try:
    from vllm.v1.attention.backend import AttentionCGSupport
except ImportError:
    from vllm.v1.attention.backends.utils import AttentionCGSupport  # type: ignore

FP8_CACHE_DTYPES = ("fp8", "fp8_e4m3")

# Builder-level step counters (per process): smoke tests read these to make
# sure decode steps really ran on T5.
STATS = {"t5_steps": 0, "fallback_steps": 0}


def _fallback_name() -> str:
    return env_flag("THOR_T5_FALLBACK", "FLASHINFER").upper() or "FLASHINFER"


def _base_backend():
    name = _fallback_name()
    assert name != "CUSTOM", "THOR_T5_FALLBACK cannot be CUSTOM"
    return AttentionBackendEnum[name].get_class()


_Base = _base_backend()
_BaseBuilder = _Base.get_builder_cls()
_BaseImpl = _Base.get_impl_cls()
_BASE_WRITES_KV_IN_FORWARD = getattr(_Base, "forward_includes_kv_cache_update", True)


def _max_q_len(vllm_config) -> int:
    spec = getattr(vllm_config, "speculative_config", None)
    n = getattr(spec, "num_speculative_tokens", None) if spec is not None else None
    cap = int(env_flag("THOR_T5_MAX_QLEN", "0") or 0)
    q = 1 + int(n or 0)
    return min(q, cap) if cap > 0 else q


def _gpu_sync_allowed():
    try:
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed

        return gpu_sync_allowed()
    except ImportError:
        import contextlib

        return contextlib.nullcontext()


# --------------------------------------------------------------------------
# Static eligibility (config level). Returns None if OK, else the reason.
# --------------------------------------------------------------------------
def _layers(vllm_config, layer_names):
    try:
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

        return get_layers_from_vllm_config(vllm_config, AttentionLayerBase, layer_names)
    except Exception:  # noqa: BLE001 - layout differs across vLLM versions
        return {}


def _layer_reason(impl) -> str | None:
    sw = getattr(impl, "sliding_window", None)
    if sw not in (None, (-1, -1), -1):
        return "sliding window"
    if getattr(impl, "logits_soft_cap", None) not in (None, 0, 0.0):
        return "logits soft-cap"
    if getattr(impl, "sinks", None) is not None:
        return "attention sinks"
    if getattr(impl, "alibi_slopes", None) is not None:
        return "ALiBi"
    return None


def ineligible_reason(kv_cache_spec, layer_names, vllm_config) -> str | None:
    if env_flag("THOR_T5_DISABLE") in ("1", "true", "yes"):
        return "THOR_T5_DISABLE set"
    cache_dtype = str(getattr(vllm_config.cache_config, "cache_dtype", "auto"))
    if cache_dtype not in FP8_CACHE_DTYPES:
        return f"kv cache dtype {cache_dtype!r} (T5 needs fp8_e4m3)"
    head = getattr(kv_cache_spec, "head_size", None)
    if head is None or head % 128:
        return f"head_size {head} (needs a multiple of 128)"
    if getattr(kv_cache_spec, "head_size_v", head) != head:
        return "K/V head sizes differ"
    if getattr(kv_cache_spec, "sliding_window", None):
        return "sliding-window KV spec"
    bs = getattr(kv_cache_spec, "block_size", None)
    if bs not in SUPPORTED_PAGE_SIZES:
        return f"block size {bs} (supported: {SUPPORTED_PAGE_SIZES})"
    if vllm_config.model_config.dtype not in (torch.bfloat16, torch.float16):
        return f"model dtype {vllm_config.model_config.dtype}"
    pc = vllm_config.parallel_config
    if getattr(pc, "decode_context_parallel_size", 1) > 1:
        return "decode context parallel"
    if getattr(pc, "prefill_context_parallel_size", 1) > 1:
        return "prefill context parallel"
    for name, layer in _layers(vllm_config, layer_names).items():
        r = _layer_reason(getattr(layer, "impl", None))
        if r:
            return f"layer {name}: {r}"
    return None


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
class ThorT5Metadata:
    """Decode-only step metadata for T5 (one per KV-cache group).

    Carries the attributes generic vLLM code reads from attention metadata
    (num_actual_tokens, slot_mapping, ...); FlashInfer-specific fields are
    absent because the base impl never sees this object."""

    is_t5 = True

    def __init__(self, *, plan: StepPlan, runner: T5DecodeRunner, cam):
        self.plan = plan
        self.runner = runner
        self.num_actual_tokens = cam.num_actual_tokens
        self.num_reqs = cam.num_reqs
        self.max_query_len = cam.max_query_len
        self.max_seq_len = cam.max_seq_len
        self.query_start_loc = cam.query_start_loc
        self.seq_lens = cam.seq_lens
        self.block_table = cam.block_table_tensor
        self.slot_mapping = cam.slot_mapping
        self.num_decodes = cam.num_reqs
        self.num_decode_tokens = cam.num_actual_tokens
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.use_cascade = False
        self.causal = True


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------
class ThorT5MetadataBuilder(_BaseBuilder):  # type: ignore[misc, valid-type]
    # Shared-metadata shortcut across groups would bypass our plan.
    supports_update_block_table: bool = False
    supports_draft_decode_metadata_update: bool = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs)
        # Draft-graph metadata refresh is base-backend specific.
        self.supports_draft_decode_metadata_update = False
        self._t5_reason = ineligible_reason(kv_cache_spec, layer_names, vllm_config)
        self.t5: T5DecodeRunner | None = None
        if self._t5_reason is not None:
            logger.warning(
                "Thor T5 decode disabled for %s: %s -> %s for all steps",
                layer_names[:1],
                self._t5_reason,
                _fallback_name(),
            )
            return
        mc, pc = vllm_config.model_config, vllm_config.parallel_config
        num_heads = None
        try:
            from vllm.v1.attention.backends.utils import get_num_attention_heads_from_layers

            num_heads = get_num_attention_heads_from_layers(vllm_config, layer_names)
        except Exception:  # noqa: BLE001
            pass
        if num_heads is None:
            num_heads = mc.get_num_attention_heads(pc)
        q_mode = env_flag("THOR_T5_Q_DTYPE", "fp8").lower()
        q_mode = {"bf16": "bf16", "bfloat16": "bf16", "fp8": "fp8", "e4m3": "fp8"}[q_mode]
        self.t5 = T5DecodeRunner(
            num_heads=num_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            head_size=kv_cache_spec.head_size,
            page_size=kv_cache_spec.block_size,
            out_dtype=mc.dtype,
            max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
            max_q_len=_max_q_len(vllm_config),
            q_mode=q_mode,
            ctas_per_sm=int(env_flag("THOR_T5_CTAS", "2")),
            max_splits=int(env_flag("THOR_T5_MAX_SPLITS", "16")),
            kv_stages=int(env_flag("THOR_T5_KV_STAGES", "0")),
            # EVICT_FIRST: default ON for the fully-fp8 path — the one change
            # that won in every Thor A/B (b=1 wall clock -8..-29% at 4K-16K,
            # ncu neutral at 64K-256K); bf16-Q long-L was mixed -> opt-in.
            # PDL: opt-in — its A/Bs flipped sign between runs (-17% .. +5%)
            # on a busy box; re-measure on an idle one before defaulting it.
            pdl=env_flag("THOR_T5_PDL", "0").lower() in ("1", "true", "yes"),
            l2_prefetch=int(env_flag("THOR_T5_L2_PREFETCH", "0")),
            kv_evict_first=env_flag("THOR_T5_KV_EVICT_FIRST", "1" if q_mode == "fp8" else "0").lower()
            in ("1", "true", "yes"),
            device=device,
        )
        pre = env_flag("THOR_T5_PRECOMPILE", "all").lower()
        if pre in ("all", "1", "true", "yes"):
            logger.info("Thor T5: compiling decode kernels (q_len 1..%d, %s Q)", self.t5.max_q_len, q_mode)
            self.t5.precompile()
        logger.info(
            "Thor T5 decode enabled: h_q=%d h_k=%d d=%d page=%d q_len<=%d Q=%s "
            "ctas/SM=%d slots(q1)=%d pdl=%s l2_prefetch=%d "
            "kv_evict_first=%s fallback=%s",
            num_heads,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            kv_cache_spec.block_size,
            self.t5.max_q_len,
            q_mode,
            self.t5.ctas_per_sm,
            self.t5.slots(1),
            self.t5.pdl,
            self.t5.l2_prefetch,
            self.t5.kv_evict_first,
            _fallback_name(),
        )

    # -- CUDA graph support ------------------------------------------------
    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        base = super().get_cudagraph_support(vllm_config, kv_cache_spec)
        # Mixed batches always go to the base backend, so never promise more
        # than it does for those; uniform decode is T5's and graph-safe.
        if base == AttentionCGSupport.ALWAYS:
            return base
        if _max_q_len(vllm_config) > 1:
            return max(base, AttentionCGSupport.UNIFORM_BATCH, key=lambda e: e.value)
        return max(base, AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE, key=lambda e: e.value)

    @classmethod
    def get_varlen_cudagraph_max_query_len(cls, vllm_config, kv_cache_spec):
        return None  # T5 graphs replay uniform-q_len batches only

    # -- build ---------------------------------------------------------------
    def _t5_q_len(self, cam) -> int | None:
        """q_len if this step is a uniform decode batch T5 can take."""
        t5 = self.t5
        if t5 is None:
            return None
        q_len = int(cam.max_query_len)
        if not 1 <= q_len <= t5.max_q_len or cam.num_reqs > t5.max_num_seqs:
            return None
        causal = getattr(cam, "causal", True)
        if causal is not True:
            return None
        if getattr(cam, "dcp_local_seq_lens", None) is not None:
            return None
        if q_len == 1 and cam.num_actual_tokens <= cam.num_reqs:
            return 1
        # Uniform: every request has q_len tokens, except trailing CUDA-graph
        # padding requests with 0.
        qsl = cam.query_start_loc_cpu[: cam.num_reqs + 1]
        lens = (qsl[1:] - qsl[:-1]).tolist()
        n_real = 0
        while n_real < len(lens) and lens[n_real] == q_len:
            n_real += 1
        if any(x != 0 for x in lens[n_real:]):
            return None
        return q_len

    def _seq_lens_cpu(self, cam):
        ub = getattr(cam, "seq_lens_cpu_upper_bound", None)
        if ub is not None:
            # An upper bound is fine for planning: the kernel masks with the
            # exact device seq_lens, extra tiles are empty.
            return ub
        legacy = getattr(cam, "seq_lens_cpu", None)
        if legacy is not None:
            return legacy
        with _gpu_sync_allowed():
            return cam.seq_lens.cpu()

    def build(self, common_prefix_len: int, common_attn_metadata, fast_build: bool = False):
        cam = common_attn_metadata
        q_len = self._t5_q_len(cam) if common_prefix_len == 0 else None
        if q_len is None:
            STATS["fallback_steps"] += 1
            return super().build(common_prefix_len, cam, fast_build)
        STATS["t5_steps"] += 1
        t5 = self.t5
        num_seqs = cam.num_reqs
        plan = t5.plan(
            self._seq_lens_cpu(cam),
            num_seqs,
            q_len,
            cam.block_table_tensor,
            cam.seq_lens,
        )
        # Never compiled inside graph capture: build() runs outside it.
        t5.ensure_compiled(q_len, plan.with_reduction)
        return ThorT5Metadata(plan=plan, runner=t5, cam=cam)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build(0, common_attn_metadata)

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        # Cascade steps are built by the base builder (common_prefix_len > 0);
        # keep the base decision.
        return super().use_cascade_attention(*args, **kwargs)


# --------------------------------------------------------------------------
# Impl
# --------------------------------------------------------------------------
class ThorT5Impl(_BaseImpl):  # type: ignore[misc, valid-type]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # T5 quantizes Q itself; a pre-quantized query (layer-level fusion)
        # would change the input dtype of every step, so keep it bf16.
        self.supports_quant_query_input = False
        self._kv_layout: str | None = None
        self._t5_reason = _layer_reason(self)

    def fused_output_quant_supported(self, quant_key) -> bool:
        return False  # T5 writes bf16/fp16 output

    def _kv_views(self, kv_cache: torch.Tensor, runner: T5DecodeRunner):
        if self._kv_layout is None:
            self._kv_layout = detect_kv_layout(kv_cache.shape, runner.h_k, runner.page_size, runner.d)
            logger.info(
                "Thor T5: KV cache %s %s -> layout %s", tuple(kv_cache.shape), tuple(kv_cache.stride()), self._kv_layout
            )
        return split_kv_cache(kv_cache, self._kv_layout, runner.d)

    def _write_kv(self, layer, key, value, kv_cache, k_cache, v_cache, md):
        """Only for base backends whose forward() also writes the KV cache
        (forward_includes_kv_cache_update = True, e.g. older vLLM)."""
        if not _BASE_WRITES_KV_IN_FORWARD:
            return
        if key is None or value is None:
            return
        if getattr(self, "kv_sharing_target_layer_name", None) is not None:
            return
        n = md.slot_mapping.shape[0]
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key[:n],
            value[:n],
            k_cache,
            v_cache,
            md.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
        *args: Any,
        **kwargs: Any,
    ):
        if not getattr(attn_metadata, "is_t5", False):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
                *args,
                **kwargs,
            )
        if self._t5_reason is not None:
            raise RuntimeError(f"Thor T5 got a layer it cannot run ({self._t5_reason}); set THOR_T5_DISABLE=1")
        assert output is not None and output_scale is None
        md: ThorT5Metadata = attn_metadata
        runner = md.runner
        k_cache, v_cache = self._kv_views(kv_cache, runner)
        if k_cache.shape[1] != runner.page_size:
            raise RuntimeError(
                f"Thor T5: KV page size {k_cache.shape[1]} != planned {runner.page_size}; "
                "set THOR_T5_KV_LAYOUT or THOR_T5_DISABLE=1"
            )
        self._write_kv(layer, key, value, kv_cache, k_cache, v_cache, md)
        if query.dtype not in (torch.bfloat16, torch.float16):
            # pre-quantized fp8 query from a layer fusion: dequantize
            query = (query.float() * layer._q_scale).to(torch.bfloat16)
        runner.run(
            md.plan,
            query,
            output,
            k_cache,
            v_cache,
            layer._k_scale,
            layer._v_scale,
            self.scale,
        )
        return output


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------
class ThorT5AttentionBackend(_Base):  # type: ignore[misc, valid-type]
    accept_output_buffer: bool = True
    base_backend: ClassVar[type] = _Base

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return ThorT5Impl

    @staticmethod
    def get_builder_cls():
        return ThorT5MetadataBuilder

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # The uniform-decode check reads query_start_loc_cpu; it must be exact.
        return False
