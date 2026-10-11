"""vLLM attention backend: the Mjolnir (T5) fp8 paged decode on Jetson Thor (sm_110).

Design
------
MjolnirAttentionBackend is a *subclass of a stock backend*, the "fallback":
FlashAttention 4 on the Mjolnir image (its prefill is ~40% faster than
FlashInfer's on Thor), FlashInfer on a stock vLLM (whose FA on sm_110 cannot
read an fp8 KV cache); MJOLNIR_ATTN_FALLBACK picks another. It inherits the
KV-cache layout, the KV-cache write, prefill, validation and everything else.
Only two things change:

  * the metadata builder: a step whose batch is decode-only (uniform
    q_len <= 1 + num_speculative_tokens, causal, no cascade) gets a T5 plan
    (occupancy-pinned split-KV, written into persistent buffers, padded to a
    fixed grid -> FULL CUDA graphs replay it). Any other step is built by
    the base builder and run by the base impl, unchanged.
  * the impl: forward() runs the T5 kernel when it gets T5 metadata,
    otherwise it calls the base forward.

Select it with `--attention-backend MJOLNIR` (the Mjolnir image's vLLM has
the enum member) or `--attention-backend CUSTOM` (stock vLLM); the
`vllm.general_plugins` entry point registers the class at vLLM start-up.
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

logger = logging.getLogger("vllm.mjolnir")  # under vLLM's handler

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


def _default_fallback() -> tuple[str, bool]:
    """(backend name, explicitly chosen?). The Mjolnir image's vLLM carries
    the default (``fa_utils.mjolnir_fallback_name``: FLASH_ATTN, which the same
    patch resolves to FA4 on sm_110); a stock vLLM gets FLASHINFER."""
    explicit = (env_flag("MJOLNIR_ATTN_FALLBACK", "") or "").strip().upper()
    if explicit:
        return explicit, True
    try:
        from vllm.v1.attention.backends.fa_utils import mjolnir_fallback_name
    except ImportError:  # stock vLLM: no Thor FA4 FP8-KV support
        return "FLASHINFER", False
    return mjolnir_fallback_name(), False


def _fa4_serves_fp8_kv() -> bool | None:
    """Can FlashAttention serve this model's (fp8) KV cache here? None when
    there is no config to ask (import outside vLLM's model construction)."""
    try:
        from vllm.config import get_current_vllm_config_or_none
        from vllm.v1.attention.backends.fa_utils import flash_attn_supports_kv_cache_dtype
    except ImportError:
        return None
    cfg = get_current_vllm_config_or_none()
    if cfg is None or cfg.model_config is None:
        return None
    cache_dtype = str(getattr(cfg.cache_config, "cache_dtype", "auto"))
    if cache_dtype not in FP8_CACHE_DTYPES:
        return True  # nothing to gate: FA reads bf16/fp16 KV on any version
    head = cfg.model_config.get_head_size()
    return bool(flash_attn_supports_kv_cache_dtype(cache_dtype, head_size=head, supports_fa4_hd256=True))


def _resolve_fallback() -> str:
    name, explicit = _default_fallback()
    assert name not in ("CUSTOM", "MJOLNIR"), "MJOLNIR_ATTN_FALLBACK must name a stock backend"
    if name == "FLASH_ATTN" and not explicit and _fa4_serves_fp8_kv() is False:
        # The image default, but FA4's one-shot FP8-KV probe failed on this
        # device (or the layer shape is not FA4's): keep serving on FlashInfer.
        logger.warning(
            "Mjolnir: FlashAttention 4 cannot serve this fp8 KV cache here "
            "(see the 'THOR FA4 ... probe' log line); falling back to FLASHINFER"
        )
        return "FLASHINFER"
    return name


_FALLBACK = _resolve_fallback()


def _fallback_name() -> str:
    return _FALLBACK


def _base_backend():
    return AttentionBackendEnum[_FALLBACK].get_class()


_Base = _base_backend()
_BaseBuilder = _Base.get_builder_cls()
_BaseImpl = _Base.get_impl_cls()
_BASE_WRITES_KV_IN_FORWARD = getattr(_Base, "forward_includes_kv_cache_update", True)


def _block_size_ok(sizes, size: int) -> bool:
    """Does a backend's supported-size list (ints and MultipleOf) take ``size``?"""
    for s in sizes:
        base = getattr(s, "base", None)
        if (base is not None and size % base == 0) or s == size:
            return True
    return False


def _max_q_len(vllm_config) -> int:
    spec = getattr(vllm_config, "speculative_config", None)
    n = getattr(spec, "num_speculative_tokens", None) if spec is not None else None
    cap = int(env_flag("MJOLNIR_ATTN_MAX_QLEN", "0") or 0)
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
    if env_flag("MJOLNIR_ATTN_DISABLE") in ("1", "true", "yes"):
        return "MJOLNIR_ATTN_DISABLE set"
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
class MjolnirMetadata:
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
class MjolnirMetadataBuilder(_BaseBuilder):  # type: ignore[misc, valid-type]
    # Shared-metadata shortcut across groups would bypass our plan.
    supports_update_block_table: bool = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs)
        # Fused multi-step draft decode (MTP / EAGLE): the speculator builds the
        # draft metadata once and runs every draft step on it, advancing the
        # device seq_lens in place. A T5 plan stays valid across those steps if
        # it is made for the longest of them — the kernel masks with the live
        # seq_lens, so tiles past a sequence's end are empty — hence every plan
        # gets num_speculative_tokens of headroom, and update_draft_decode_metadata
        # has nothing to do for T5 steps (fallback steps defer to the base
        # builder). The capability is whatever the base builder grants.
        spec = getattr(vllm_config, "speculative_config", None)
        self._plan_margin = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec is not None else 0
        self._t5_reason = ineligible_reason(kv_cache_spec, layer_names, vllm_config)
        self.t5: T5DecodeRunner | None = None
        if self._t5_reason is not None:
            logger.warning(
                "Mjolnir decode disabled for %s: %s -> %s for all steps",
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
        q_mode = env_flag("MJOLNIR_ATTN_Q_DTYPE", "fp8").lower()
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
            ctas_per_sm=int(env_flag("MJOLNIR_ATTN_CTAS", "2")),
            max_splits=int(env_flag("MJOLNIR_ATTN_MAX_SPLITS", "16")),
            kv_stages=int(env_flag("MJOLNIR_ATTN_KV_STAGES", "0")),
            # EVICT_FIRST: default ON for the fully-fp8 path — the one change
            # that won in every Thor A/B (b=1 wall clock -8..-29% at 4K-16K,
            # ncu neutral at 64K-256K); bf16-Q long-L was mixed -> opt-in.
            # PDL: opt-in — its A/Bs flipped sign between runs (-17% .. +5%)
            # on a busy box; re-measure on an idle one before defaulting it.
            pdl=env_flag("MJOLNIR_ATTN_PDL", "0").lower() in ("1", "true", "yes"),
            l2_prefetch=int(env_flag("MJOLNIR_ATTN_L2_PREFETCH", "0")),
            kv_evict_first=env_flag("MJOLNIR_ATTN_KV_EVICT_FIRST", "1" if q_mode == "fp8" else "0").lower()
            in ("1", "true", "yes"),
            device=device,
        )
        pre = env_flag("MJOLNIR_ATTN_PRECOMPILE", "all").lower()
        if pre in ("all", "1", "true", "yes"):
            logger.info("Mjolnir: compiling decode kernels (q_len 1..%d, %s Q)", self.t5.max_q_len, q_mode)
            self.t5.precompile()
        logger.info(
            "Mjolnir decode enabled: h_q=%d h_k=%d d=%d page=%d q_len<=%d Q=%s "
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
            margin=self._plan_margin,
        )
        # Never compiled inside graph capture: build() runs outside it.
        t5.ensure_compiled(q_len, plan.with_reduction)
        return MjolnirMetadata(plan=plan, runner=t5, cam=cam)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build(0, common_attn_metadata)

    def update_draft_decode_metadata(self, metadata) -> None:
        """Fused draft decode: advance the metadata to the next draft step,
        capture-safe. A T5 plan already covers it (see __init__); other steps
        were built by the base builder, which updates its own metadata."""
        if getattr(metadata, "is_t5", False):
            return
        super().update_draft_decode_metadata(metadata)

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        # Cascade steps are built by the base builder (common_prefix_len > 0);
        # keep the base decision.
        return super().use_cascade_attention(*args, **kwargs)


# --------------------------------------------------------------------------
# Impl
# --------------------------------------------------------------------------
class MjolnirImpl(_BaseImpl):  # type: ignore[misc, valid-type]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # T5 quantizes Q itself (dynamic, per (sequence, KV head)); the
        # layer's static per-tensor quantization would change the input of
        # every step, so the layer keeps handing us bf16 Q. A fallback that
        # wants fp8 Q next to an fp8 KV cache (FlashAttention 4 on sm_110
        # needs Q, K and V in one fp8 dtype) gets it quantized here, on the
        # steps it runs — the same static quantization the layer would do.
        fp8_kv = str(getattr(self, "kv_cache_dtype", "auto")) in FP8_CACHE_DTYPES
        self._fallback_fp8_q = bool(self.supports_quant_query_input) and fp8_kv and _FALLBACK == "FLASH_ATTN"
        self.supports_quant_query_input = False
        self._kv_layout: str | None = None
        self._t5_reason = _layer_reason(self)

    def _quantize_q(self, layer, query: torch.Tensor) -> torch.Tensor:
        """Static fp8 Q for the fallback, as the attention layer would
        quantize it: Q / q_scale, saturated to e4m3, per tensor or per KV head."""
        scale = layer._q_scale
        n = query.shape[0]
        if scale.numel() == 1:
            from vllm import _custom_ops as ops

            q8, _ = ops.scaled_fp8_quant(query.reshape(n, -1), scale.reshape(1).float())
            return q8.view(query.shape)
        fp8 = torch.float8_e4m3fn
        lim = torch.finfo(fp8).max
        q = query.reshape(n, scale.numel(), -1).float() / scale.float().reshape(1, -1, 1)
        return q.clamp_(-lim, lim).to(fp8).view(query.shape)

    def fused_output_quant_supported(self, quant_key) -> bool:
        return False  # T5 writes bf16/fp16 output

    def _kv_views(self, kv_cache: torch.Tensor, runner: T5DecodeRunner):
        if self._kv_layout is None:
            self._kv_layout = detect_kv_layout(kv_cache.shape, runner.h_k, runner.page_size, runner.d)
            logger.info(
                "Mjolnir: KV cache %s %s -> layout %s", tuple(kv_cache.shape), tuple(kv_cache.stride()), self._kv_layout
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
            if self._fallback_fp8_q and attn_metadata is not None and query.dtype in (torch.bfloat16, torch.float16):
                query = self._quantize_q(layer, query)
                # The base forward passes the layer's q scale as q_descale
                # only when it believes the layer quantized Q.
                self.supports_quant_query_input = True
                try:
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
                finally:
                    self.supports_quant_query_input = False
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
            raise RuntimeError(f"Mjolnir got a layer it cannot run ({self._t5_reason}); set MJOLNIR_ATTN_DISABLE=1")
        assert output is not None and output_scale is None
        md: MjolnirMetadata = attn_metadata
        runner = md.runner
        k_cache, v_cache = self._kv_views(kv_cache, runner)
        if k_cache.shape[1] != runner.page_size:
            raise RuntimeError(
                f"Mjolnir: KV page size {k_cache.shape[1]} != planned {runner.page_size}; "
                "set MJOLNIR_ATTN_KV_LAYOUT or MJOLNIR_ATTN_DISABLE=1"
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
class MjolnirAttentionBackend(_Base):  # type: ignore[misc, valid-type]
    accept_output_buffer: bool = True
    base_backend: ClassVar[type] = _Base

    @staticmethod
    def get_name() -> str:
        # vLLM maps this name back through AttentionBackendEnum[...]; the
        # MJOLNIR member exists only in a patched vLLM (the Mjolnir image),
        # a stock vLLM reaches this class through CUSTOM.
        from mjolnir import BACKEND_NAME, has_native_member

        return BACKEND_NAME if has_native_member() else "CUSTOM"

    @classmethod
    def get_supported_kernel_block_sizes(cls, kv_cache_spec=None) -> list:
        """The Mjolnir kernel runs 16/32/64/128-token pages. On FlashInfer
        (which advertises 16/32/64 on sm_110) it adds 128 — the kernel's
        benchmark default and the best fit for head-major pages — since FI's
        prefill and KV-write paths take any page size. Any other fallback's
        constraint is kept: FA4's hd256 kernel needs exactly 128."""
        if _FALLBACK == "FLASHINFER":
            return list(SUPPORTED_PAGE_SIZES)
        try:
            base = _Base.get_supported_kernel_block_sizes(kv_cache_spec)
        except TypeError:  # older vLLM: no kv_cache_spec argument
            base = _Base.get_supported_kernel_block_sizes()
        ours = [s for s in SUPPORTED_PAGE_SIZES if _block_size_ok(base, s)]
        return ours or list(base)

    @staticmethod
    def get_impl_cls():
        return MjolnirImpl

    @staticmethod
    def get_builder_cls():
        return MjolnirMetadataBuilder

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # The uniform-decode check reads query_start_loc_cpu; it must be exact.
        return False
