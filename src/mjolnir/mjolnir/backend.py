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
    q_len <= 1 + num_speculative_tokens, causal or a parallel drafter's
    non-causal block, no cascade) gets a T5 plan
    (occupancy-pinned split-KV, written into persistent buffers, padded to a
    fixed grid -> FULL CUDA graphs replay it). Any other step is built by
    the base builder and run by the base impl, unchanged.
  * the impl: forward() runs the T5 kernel when it gets T5 metadata,
    otherwise it calls the base forward.

KV formats: fp8 (e4m3, vLLM's own cache write), and ``fp6_e2m3`` on the
image (``thor-fp6-e2m3-kv-cache.patch`` adds the dtype): packed 6-bit codes
with static per-(layer, KV head, channel) scales from a calibration file
(``mjolnir.kvscales``). Mjolnir writes that cache itself (``mjolnir.kvfp6``),
T5 reads it directly (the channel scales folded into Q and the output), and
the steps the FlashAttention 4 fallback runs see their pages expanded to an
fp8 scratch first (channel scales applied, one descale per head).

Select it with `--attention-backend MJOLNIR` (the Mjolnir image's vLLM has
the enum member) or `--attention-backend CUSTOM` (stock vLLM); the
`vllm.general_plugins` entry point registers the class at vLLM start-up.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .builder import MjolnirMetadata

import torch

from .platforms import unsupported_reason
from .runner import (
    SUPPORTED_PAGE_SIZES,
    detect_kv_layout,
    env_flag,
    split_kv_cache,
)

if TYPE_CHECKING:
    # The annotations name the Thor runner; a second family's runner duck-types
    # the same surface and arrives through its own arch subpackage. At runtime
    # the class is resolved via platforms.load_kernel — never by importing a
    # family module here (see mjolnir.arch).
    from .arch.thor import T5DecodeRunner

logger = logging.getLogger("vllm.mjolnir")  # under vLLM's handler

# --------------------------------------------------------------------------
# vLLM imports (paths moved between releases; try both)
# --------------------------------------------------------------------------
try:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
except ImportError:  # older layout
    from vllm.attention.backends.registry import AttentionBackendEnum

FP8_CACHE_DTYPES = ("fp8", "fp8_e4m3")
FP6_CACHE_DTYPES = ("fp6_e2m3",)
MJOLNIR_CACHE_DTYPES = FP8_CACHE_DTYPES + FP6_CACHE_DTYPES


def _kv_format(cache_dtype: str) -> str:
    return "e2m3" if str(cache_dtype) in FP6_CACHE_DTYPES else "fp8"


# Builder-level step counters (per process): smoke tests read these to make
# sure decode steps really ran on T5.
STATS = {"t5_steps": 0, "fallback_steps": 0, "t5_noncausal_steps": 0}
# Parallel drafters' non-causal block attention on T5 (MJOLNIR_ATTN_NONCAUSAL=0:
# those steps go to the fallback, as before)
_NONCAUSAL = env_flag("MJOLNIR_ATTN_NONCAUSAL", "1").lower() not in ("0", "false", "no")


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
    if cache_dtype in FP6_CACHE_DTYPES:
        # fp6: the prefill fallback reads the e4m3 scratch the pages expand
        # to, so gate the fallback on fp8 -- not on the config's own dtype
        # (an fp6 config must flip to FlashInfer on sm_87 exactly like an
        # fp8 one, or _Base binds to FA4 and the fp6 gate rejects).
        cache_dtype = "fp8"
    elif cache_dtype not in FP8_CACHE_DTYPES:
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


def _fp6_known_to_vllm() -> bool:
    """Does this vLLM have the fp6_e2m3 cache dtype (the image's patch)?"""
    try:
        from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

        return get_kv_quant_mode("fp6_e2m3") == getattr(KVQuantMode, "FP6_E2M3", None)
    except ImportError:
        return False


def _fallback_serves_fp8_scratch() -> bool:
    """Can the prefill fallback read the e4m3 scratch the fp6 pages expand
    to? FA4 (the Thor image default) or FlashInfer (the Orin — FA4 has no
    sm_87 kernels and its probe falls back; FlashInfer serves fp8 KV there,
    proven by the config-C serving)."""
    return _FALLBACK in ("FLASH_ATTN", "FLASHINFER")


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
    if cache_dtype not in MJOLNIR_CACHE_DTYPES:
        return f"kv cache dtype {cache_dtype!r} (T5 needs fp8_e4m3 or fp6_e2m3)"
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


def _unsupported_arch_reason(device) -> str | None:
    """None on a supported compute capability; else the platform registry's
    verdict (torch-free leaf), routed into the same t5=None fallback as every
    ineligible config."""
    return unsupported_reason(torch.cuda.get_device_capability(device))


# --------------------------------------------------------------------------
# fp6 (e2m3) KV: preconditions and per-channel scales
# --------------------------------------------------------------------------
def _check_fp6(t5_reason, layer_names) -> None:
    """fp6 caches have exactly two readers: the mma/T5 decode kernel and the
    fallback's fp8 scratch (any fallback that serves fp8 KV: FA4 on thor,
    FlashInfer on orin)."""
    if t5_reason is not None:
        raise ValueError(
            f"kv cache dtype fp6_e2m3 needs the Mjolnir decode kernel, but it is off for "
            f"{layer_names[:1]}: {t5_reason}. Serve fp8 KV for this model."
        )
    if not _fallback_serves_fp8_scratch():
        raise ValueError(
            f"kv cache dtype fp6_e2m3 needs a fallback that reads the fp8 scratch (FLASH_ATTN "
            f"or FLASHINFER); MJOLNIR_ATTN_FALLBACK is {_FALLBACK}"
        )
    from . import kvscales

    if kvscales.calibrating():
        raise ValueError("calibrate on fp8 KV (MJOLNIR_ATTN_KV_CALIBRATE), then serve fp6_e2m3 with the file")


_FP6_SCRATCH: dict = {}


def _fp6_scratch(pages: int, H: int, N: int, row: int, device) -> torch.Tensor:
    """A growing per-device (pages, H, N, 2d) uint8 buffer for the fallback's
    e4m3 pages. Layers run one after another, so one buffer serves them all;
    it lives outside vLLM's KV budget (a 200K-token context at H=4, d=256 is
    ~0.4 GB) and grows to the largest step seen."""
    key = (str(device), H, N, row)
    buf = _FP6_SCRATCH.get(key)
    need = max(1, pages)
    if buf is None or buf.shape[0] < need:
        cap = max(need, 2 * buf.shape[0] if buf is not None else need)
        buf = torch.empty(cap, H, N, row, dtype=torch.uint8, device=device)
        _FP6_SCRATCH[key] = buf
    return buf


def _set_scale(layer, name: str, value: torch.Tensor) -> None:
    """Replace a layer's k/v scale with ``value`` (any shape), keeping its
    kind: quantized checkpoints (e.g. ModelOpt NVFP4) register the scales as
    nn.Parameters, which a plain tensor cannot be assigned over."""
    cur = getattr(layer, name, None)
    if isinstance(cur, torch.nn.Parameter):
        setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))
    elif name in getattr(layer, "_buffers", {}):
        layer._buffers[name] = value
    else:
        setattr(layer, name, value)


def _install_fp6_scales(vllm_config, layer_names, h_k: int, d: int) -> None:
    """Per-(layer, KV head, channel) e2m3 scales from the calibration file.

    Channel scales s_c (h_k, d) quantize the cache write. Attention never
    sees them as descales — they fold into the math around it:

    * T5: K's into Q (q . k = sum_c (q_c s_c) code_c) and V's into the output
      (O_c = s_c sum_t p_t code_tc); the kernel's K/V descales are 1.
    * FA4 fallback: the scratch holds e4m3(code * s_c / S) with one
      S = max_c(s_c) * 7.5 / 448 per head, and S is the descale — so
      ``layer._k_scale`` / ``_v_scale`` are the per-head S.
    """
    from . import kvscales

    path = kvscales.scales_path()
    if not path:
        raise ValueError(
            "kv cache dtype fp6_e2m3 needs calibrated scales: set MJOLNIR_ATTN_KV_SCALES to a "
            "calibration file (`mjolnir calibrate`, or python -m mjolnir.calibrate)"
        )
    doc = kvscales.load(path)
    missing = kvscales.missing_layers(doc, layer_names)
    if missing:
        raise ValueError(f"{path} has no scales for {missing[:3]}{' ...' if len(missing) > 3 else ''}")
    layers = _layers(vllm_config, layer_names)
    for name, layer in layers.items():
        ks, vs = kvscales.e2m3_scales(doc, name, h_k, d)
        dev = layer._k_scale.device
        h_r = max(1, int(getattr(layer, "num_heads", h_k)) // h_k)
        state = {}
        for side, rows in (("k", ks), ("v", vs)):
            ch = torch.tensor(rows, dtype=torch.float32, device=dev).contiguous()  # (h_k, d)
            if _FALLBACK == "FLASH_ATTN":
                # FA4 reads one fp8 dtype with a per-head descale: the scratch
                # holds e4m3(code * s_c / S) and S (per head) rides
                # layer._k_scale / _v_scale.
                head = ch.amax(dim=1) * (kvscales.E2M3_MAX / 448.0)  # (h_k,) fallback descale
                state[f"{side}_ch"] = ch
                state[f"{side}_head"] = head
                state[f"{side}_ratio"] = (ch / head[:, None]).contiguous()
            else:
                # FlashInfer's native paged kernels take no KV descale: the
                # scratch holds the true code * s_c (<= the calibrated amax,
                # which must fit e4m3's 448) and the layer's scales stay 1.
                state[f"{side}_ch"] = ch
                state[f"{side}_head"] = torch.ones_like(ch[:, 0])
                state[f"{side}_ratio"] = ch.contiguous()
            state[f"{side}_mul"] = ch.repeat_interleave(h_r, dim=0).contiguous()  # (h_q, d)
        state["one"] = torch.ones(1, dtype=torch.float32, device=dev)
        layer._mjolnir_fp6 = state
        _set_scale(layer, "_k_scale", state["k_head"])
        _set_scale(layer, "_v_scale", state["v_head"])
    logger.info(
        "Mjolnir: fp6_e2m3 KV scales (%s) for %d layer(s) from %s",
        "per channel" if any(kvscales.per_channel(doc, n) for n in layers) else "per head",
        len(layers),
        path,
    )


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
        cache_dtype = str(getattr(self, "kv_cache_dtype", "auto"))
        # fp6: the fallback reads an fp8 scratch (see _forward_fp6_fallback)
        self._fp6 = cache_dtype in FP6_CACHE_DTYPES
        quantized_kv = cache_dtype in MJOLNIR_CACHE_DTYPES
        # fp8 Q only behind FA4: FlashInfer's maybe_quant_query wants bf16.
        self._fallback_fp8_q = (
            _FALLBACK == "FLASH_ATTN" and bool(self.supports_quant_query_input) and (quantized_kv or self._fp6)
        )
        if self._fp6:
            # The fp6 fallback hands the base a fresh scratch tensor every
            # step; captured graphs bake pointers at capture time and would
            # replay against the old allocation (the IMA). Fallback steps
            # run eager; T5 steps keep the runner's own graphs.
            self.enable_cuda_graph = False
        self.supports_quant_query_input = False
        self._kv_layout: str | None = None
        self._fp6_pool_layout: str | None = None  # MJOLNIR_FP6_DIRECT
        self._t5_reason = _layer_reason(self)

    # -- KV cache write ----------------------------------------------------
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """fp6: Mjolnir's own packed write; fp8: the base backend's. In
        calibration mode (MJOLNIR_ATTN_KV_CALIBRATE=1) also fold every
        written K/V into the layer's per-channel amax."""
        from . import kvscales

        if kvscales.calibrating():
            kvscales.observe(getattr(layer, "layer_name", str(id(layer))), key, value, slot_mapping)
        if not self._fp6:
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        if getattr(self, "kv_sharing_target_layer_name", None) is not None:
            return None
        from . import kvfp6

        st = layer._mjolnir_fp6
        kvfp6.write_kv(key, value, kv_cache, slot_mapping, st["k_ch"], st["v_ch"])
        return None

    def _forward_fp6_direct(self, layer, query, kv_cache, attn_metadata, output, *rest, **kwargs):
        """MJOLNIR_FP6_DIRECT: the scratchpad-free fp6 fallback. The patched
        FlashInfer Ampere prefill (``fi-fp6-kv-ampere-prefill.patch``) reads
        the packed pool directly — no per-step e4m3 scratch, no sync. K's
        per-channel scales fold into Q and V's post-multiply the output (the
        T5 numerics); the kernel's expand emits raw e4m3 codes."""
        st = layer._mjolnir_fp6
        wrapper = attn_metadata.prefill.wrapper
        t = query.shape[0]
        page_size = getattr(attn_metadata, "mjolnir_fp6_page", 16)
        q = query.reshape(t, self.num_heads, self.head_size)
        q = q * st["k_mul"].to(q.dtype)
        row = self.head_size * 6 // 8
        if self._fp6_pool_layout is None:
            self._fp6_pool_layout = detect_kv_layout(kv_cache.shape, self.num_kv_heads, page_size, row)
        k_cache, v_cache = split_kv_cache(kv_cache, self._fp6_pool_layout, row)
        out = wrapper.run(q, (k_cache, v_cache))
        out = out.reshape(t, self.num_heads, self.head_size) * st["v_mul"].to(out.dtype)
        out = out.reshape(query.shape).to(query.dtype)
        if output is not None:
            output.copy_(out)
            return output
        return out

    def _forward_fp6_fallback(self, layer, query, key, value, kv_cache, attn_metadata, *rest, **kwargs):
        """A fallback step on an fp6 cache: expand the pages this step reads
        to e4m3 as code * s_c / S, point a copy of the metadata at the
        scratch, and run the base forward on it."""
        from . import kvfp6

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Mjolnir: an fp6 fallback step inside CUDA-graph capture — its page expansion "
                "depends on per-step host data and would replay stale. Serve fp8 KV or report this shape."
            )
        pages = attn_metadata.mjolnir_fp6_pages
        _, H, N, _ = kv_cache.shape
        scratch = _fp6_scratch(pages["total"], H, N, 2 * self.head_size, kv_cache.device)
        st = layer._mjolnir_fp6
        # The warmup/dummy passes carry arbitrary block-table entries; clamp
        # the gather in-bounds (their output is unused). Real steps index the
        # pool, so this is a no-op for them.
        src = pages["src_pages"].clamp(0, kv_cache.shape[0] - 1)
        kvfp6.expand_pages_to_fp8(kv_cache, src, scratch, self.head_size, st["k_ratio"], st["v_ratio"])
        # The base's build planned this step against the scratch remap (the
        # builder swapped the block table for the plan and restored the cam),
        # so the wrapper's buffers already index the scratch: no metadata
        # surgery here. The scratch is presented as the fp8 tensor the plan's
        # dtype expects.
        return super().forward(
            layer,
            query,
            key,
            value,
            scratch[: max(1, pages["total"])].view(torch.float8_e4m3fn),
            attn_metadata,
            *rest,
            **kwargs,
        )

    def _quantize_q(self, layer, query: torch.Tensor) -> torch.Tensor:
        """Static fp8 Q for the fallback, as the layer would quantize it."""
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
        # fp6 rows are packed bytes: K|V split at d * 6 / 8 bytes, not d
        row = runner.kv_row_bytes
        if self._kv_layout is None:
            self._kv_layout = detect_kv_layout(kv_cache.shape, runner.h_k, runner.page_size, row)
        return split_kv_cache(kv_cache, self._kv_layout, row)

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
                # only when it believes the layer quantized Q. FA4's fp6
                # fallback rides this too (Q, K and V in one fp8 dtype);
                # FlashInfer's maybe_quant_query wants bf16, so its fp6
                # fallback arrives unquantized (_fallback_fp8_q is False).
                self.supports_quant_query_input = True
            try:
                if self._fp6 and getattr(attn_metadata, "mjolnir_fp6_direct", False):
                    # MJOLNIR_FP6_DIRECT: the patched Ampere prefill reads the
                    # packed pool directly (no scratch, no per-step expand).
                    return self._forward_fp6_direct(
                        layer,
                        query,
                        kv_cache,
                        attn_metadata,
                        output,
                        *args,
                        **kwargs,
                    )
                if self._fp6 and getattr(attn_metadata, "mjolnir_fp6_pages", None) is not None:
                    # The fallback reads the e4m3 scratch. Without the pages
                    # (the profile/dummy passes) the base forward's
                    # uint8->fp8 view handles the cache as-is.
                    return self._forward_fp6_fallback(
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
        if self._t5_reason is not None:
            raise RuntimeError(f"Mjolnir got a layer it cannot run ({self._t5_reason}); set MJOLNIR_ATTN_DISABLE=1")
        assert output is not None and output_scale is None
        md: MjolnirMetadata = attn_metadata  # type: ignore[invalid-annotation]
        runner = md.runner
        k_cache, v_cache = self._kv_views(kv_cache, runner)
        if k_cache.shape[1] != runner.page_size:
            raise RuntimeError(
                f"Mjolnir: KV page size {k_cache.shape[1]} != planned {runner.page_size}; "
                "set MJOLNIR_ATTN_KV_LAYOUT or MJOLNIR_ATTN_DISABLE=1"
            )
        self._write_kv(layer, key, value, kv_cache, k_cache, v_cache, md)
        if os.environ.get("MJOLNIR_ATTN_DEBUG") and not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()  # the IMA hunt: name the kernel that faults
            with open("/data/cache/mjolnir_fp6.log", "a") as f:
                f.write("write_kv ok\n")
        if os.environ.get("MJOLNIR_ATTN_DEBUG") and not torch.cuda.is_current_stream_capturing():
            # per-call dump -> a FILE (the EngineCore swallows raw stdout);
            # the draft-prefill-vs-verify identity diff, config C isolation
            n = md.plan.num_seqs
            with open("/tmp/mjolnir_dbg.log", "a") as f:
                f.write(
                    f"layer={getattr(layer, 'layer_name', '?')} ql={md.plan.q_len} b={n} "
                    f"lens={md.plan.seq_lens[:n].tolist()} "
                    f"slots={md.slot_mapping[:6].tolist()} "
                    f"run={id(runner) % 100000} "
                    f"qsum={query.float().sum().item():.4g} "
                    f"kvsum={k_cache[md.plan.block_table[0, :2].long()].float().sum().item():.6g}\n"
                )
        if query.dtype not in (torch.bfloat16, torch.float16):
            # pre-quantized fp8 query from a layer fusion: dequantize
            query = (query.float() * layer._q_scale).to(torch.bfloat16)
        if self._fp6:  # channel scales fold into Q and O; the cache codes have descale 1
            st = layer._mjolnir_fp6
            runner.run(
                md.plan,
                query,
                output,
                k_cache,
                v_cache,
                st["one"],
                st["one"],
                self.scale,
                q_mul=st["k_mul"],
                o_mul=st["v_mul"],
            )
            if os.environ.get("MJOLNIR_ATTN_DEBUG") and not torch.cuda.is_current_stream_capturing():
                torch.cuda.synchronize()
                with open("/data/cache/mjolnir_fp6.log", "a") as f:
                    f.write("mma fp6 ok\n")
            return output
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

    # -- KV cache dtypes --------------------------------------------------
    # The fallback's own list, plus fp6_e2m3 where vLLM knows the dtype (the
    # image's thor-fp6-e2m3-kv-cache patch) and the fallback can read the fp8
    # scratch (see _fallback_serves_fp8_scratch).
    supported_kv_cache_dtypes: ClassVar[list] = list(getattr(_Base, "supported_kv_cache_dtypes", [])) + (
        list(FP6_CACHE_DTYPES) if _fallback_serves_fp8_scratch() else []
    )

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        if kv_cache_dtype in FP6_CACHE_DTYPES:
            return _fp6_known_to_vllm() and _fallback_serves_fp8_scratch() and _Base.supports_kv_cache_dtype("fp8")
        return _Base.supports_kv_cache_dtype(kv_cache_dtype)

    @classmethod
    def supports_combination(cls, head_size, dtype, kv_cache_dtype, block_size, *args, **kwargs):
        # fp6: the fallback runs on an e4m3 scratch, so it must support fp8
        if kv_cache_dtype in FP6_CACHE_DTYPES:
            kv_cache_dtype = "fp8"
        return _Base.supports_combination(head_size, dtype, kv_cache_dtype, block_size, *args, **kwargs)

    @classmethod
    def customize_spec(cls, spec):
        """fp6_e2m3: one (head, token) cell is the packed K row then the
        packed V row, d * 6 / 8 bytes each."""
        mode = getattr(spec, "kv_quant_mode", None)
        if getattr(mode, "is_fp6", False) and getattr(spec, "state_content_bytes", None) is None:
            from dataclasses import replace

            from .kvfp6 import row_bytes

            return replace(spec, state_content_bytes=row_bytes(spec.head_size) + row_bytes(spec.head_size_v))
        return super().customize_spec(spec)

    @staticmethod
    def get_impl_cls():
        return MjolnirImpl

    @staticmethod
    def get_builder_cls():
        from .builder import MjolnirMetadataBuilder

        return MjolnirMetadataBuilder

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # The uniform-decode check reads query_start_loc_cpu; it must be exact.
        return False
