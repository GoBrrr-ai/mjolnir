"""The Mjolnir T5 metadata and the fallback-page builder.

Split from backend.py along its natural seam: the metadata/builder half
(the step plans, the fp6 page maps, the base-builder delegation) vs the
Impl/AttentionBackend half (the launches and the runtime flags). The
builder imports the backend's helpers; the backend reaches back only
through TYPE_CHECKING and a lazy get_builder_cls -- no import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjolnir.runner import StepPlan, env_flag

from . import banner
from .backend import STATS

from .backend import (
    _FALLBACK,
    ineligible_reason,
    _BaseBuilder,
    _check_fp6,
    _fallback_name,
    _gpu_sync_allowed,
    _install_fp6_scales,
    _kv_format,
    _max_q_len,
    _unsupported_arch_reason,
    _NONCAUSAL,
    logger,
)
from .platforms import detect, load_kernel

try:
    from vllm.v1.attention.backend import AttentionCGSupport
except ImportError:  # older vLLM
    from vllm.v1.attention.backends.utils import AttentionCGSupport

if TYPE_CHECKING:
    from .backend import T5DecodeRunner


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
        cache_config = vllm_config.cache_config
        saved_dtype = cache_config.cache_dtype
        fp6_scratch = _kv_format(saved_dtype) == "e2m3"
        if fp6_scratch and _FALLBACK in ("FLASHINFER", "FLASH_ATTN"):
            # The fallback builder maps the cache dtype for its wrappers and
            # raises on fp6_e2m3; it only ever reads the e4m3 scratch the fp6
            # pages expand to (kvfp6), so it sees fp8. Restored below.
            cache_config.cache_dtype = "fp8"
        try:
            super().__init__(kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs)
        finally:
            if fp6_scratch:
                cache_config.cache_dtype = saved_dtype
        if env_flag("MJOLNIR_ATTN_BANNER", "1").lower() not in ("0", "false", "no"):
            banner.emit()  # the hammer, once, into this log
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
        self._kv_format = _kv_format(getattr(vllm_config.cache_config, "cache_dtype", "auto"))
        self._fp6 = self._kv_format == "e2m3"
        self._page = kv_cache_spec.block_size
        self._t5_reason = ineligible_reason(kv_cache_spec, layer_names, vllm_config)
        if self._t5_reason is None:
            # platform guard (surgical): a GPU without a Mjolnir kernel
            # degrades exactly like any other ineligible config
            self._t5_reason = _unsupported_arch_reason(device)
        # Diagnosis knobs for the draft path (MTP / EAGLE drafter layers):
        # MJOLNIR_ATTN_DRAFT_T5=0 runs every drafter step on the fallback;
        # MJOLNIR_ATTN_FUSED_DRAFT=0 makes vLLM rebuild the draft metadata
        # between draft steps instead of fusing them into one graph.
        # GROUP-level, not per-layer: vLLM hands the builder one layer_names
        # list per KV-cache group, and the MTP layer shares the target's
        # full-attention group (the same spec) -- so _is_draft is true for
        # the TARGET's layers too. Harmless while MJOLNIR_ATTN_DRAFT_T5
        # defaults to 1 (the pin below is what made the mismark poison
        # config A: it force-fallbacked the whole group, kernels never
        # served); if you add draft-conditional behavior, gate it per LAYER
        # name, not on this flag. (orin1 notes, mma-kernel-design-3.md.)
        self._is_draft = any("mtp" in n or "draft" in n for n in layer_names)
        self._force_fallback = self._is_draft and env_flag("MJOLNIR_ATTN_DRAFT_T5", "1") in ("0", "false", "no")
        if env_flag("MJOLNIR_ATTN_FUSED_DRAFT", "1") in ("0", "false", "no"):
            self.supports_draft_decode_metadata_update = False
        if self._force_fallback and self._fp6:
            # the drafter's steps replay as FULL CUDA graphs, and an fp6 fallback
            # step (page expansion from a per-step page map) cannot be captured
            raise ValueError(
                "MJOLNIR_ATTN_DRAFT_T5=0 needs fp8 KV (e.g. NVFP4_MJOLNIR): on fp6 the drafter's "
                "fallback steps would be captured into CUDA graphs with a stale page map"
            )
        if self._force_fallback:
            logger.info("Mjolnir: drafter layers %s on the fallback (MJOLNIR_ATTN_DRAFT_T5=0)", layer_names[:1])
        if self._fp6:
            _check_fp6(self._t5_reason, layer_names)
            _install_fp6_scales(vllm_config, layer_names, kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
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
        if self._fp6 and q_mode != "fp8":
            raise ValueError("fp6_e2m3 KV runs on the fp8-Q path only: unset MJOLNIR_ATTN_Q_DTYPE")
        self.t5 = load_kernel(detect()).Runner(
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
            kv_format=self._kv_format,
            device=device,
        )
        pre = env_flag("MJOLNIR_ATTN_PRECOMPILE", "all").lower()
        if pre in ("all", "1", "true", "yes"):
            import time

            from . import runner as _runner

            before = dict(_runner.CACHE_STATS)
            t0 = time.perf_counter()
            self.t5.precompile()
            loaded = _runner.CACHE_STATS["loaded"] - before["loaded"]
            compiled = _runner.CACHE_STATS["compiled"] - before["compiled"]
            if loaded or compiled:
                logger.info(
                    "Mjolnir: decode kernels (q_len 1..%d, %s Q) ready in %.1f s: %d compiled, %d from %s",
                    self.t5.max_q_len,
                    q_mode,
                    time.perf_counter() - t0,
                    compiled,
                    loaded,
                    _runner.kernel_cache_dir() or "cache (off)",
                )
        logger.info(
            "Mjolnir decode enabled: h_q=%d h_k=%d d=%d page=%d q_len<=%d Q=%s "
            "ctas/SM=%d slots(q1)=%d pdl=%s l2_prefetch=%d "
            "kv_evict_first=%s kv=%s fallback=%s",
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
            self._kv_format,
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
        """q_len if this step is a uniform decode batch T5 can take: causal
        (decode, speculative verification), or non-causal (a parallel drafter's
        block — DSpark / DFlash — where every query token sees every key; off
        with MJOLNIR_ATTN_NONCAUSAL=0)."""
        t5 = self.t5
        if t5 is None:
            return None
        q_len = int(cam.max_query_len)
        if not 1 <= q_len <= t5.max_q_len or cam.num_reqs > t5.max_num_seqs:
            return None
        causal = getattr(cam, "causal", True)
        if causal is not True and not (causal is False and _NONCAUSAL):
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
            # An upper bound is fine for planning: every family's kernel
            # masks with the exact device seq_lens at launch (Thor reads
            # them directly; Orin derives its cu_seqlens from them on the
            # device inside the launch), so extra tiles / splits are empty.
            # Planning never syncs the GPU, and a plan stays valid while a
            # fused draft decode advances the seq_lens in place.
            return ub
        legacy = getattr(cam, "seq_lens_cpu", None)
        if legacy is not None:
            return legacy
        with _gpu_sync_allowed():
            return cam.seq_lens.cpu()

    def build(self, common_prefix_len: int, common_attn_metadata, fast_build: bool = False):
        cam = common_attn_metadata
        q_len = self._t5_q_len(cam) if common_prefix_len == 0 and not self._force_fallback else None
        if q_len is None:
            STATS["fallback_steps"] += 1
            md = super().build(common_prefix_len, cam, fast_build)
            if self._fp6:
                md.mjolnir_fp6_pages = self._fp6_page_map(cam)
            return md
        STATS["t5_steps"] += 1
        t5 = self.t5
        num_seqs = cam.num_reqs
        causal = getattr(cam, "causal", True) is not False
        if not causal:
            STATS["t5_noncausal_steps"] += 1
        plan = t5.plan(
            self._seq_lens_cpu(cam),
            num_seqs,
            q_len,
            cam.block_table_tensor,
            cam.seq_lens,
            margin=self._plan_margin,
            causal=causal,
        )
        # Never compiled inside graph capture: build() runs outside it.
        t5.ensure_compiled(q_len, plan.with_reduction, plan.causal)
        return MjolnirMetadata(plan=plan, runner=t5, cam=cam)

    def _fp6_page_map(self, cam):
        """For a step the fallback runs on an fp6 cache: which cache page
        becomes which scratch page, and the block table over the scratch.
        Scratch page offset_r + j holds request r's page j (no deduplication:
        a shared prefix page is expanded once per request that uses it)."""
        num_reqs = cam.num_reqs
        lens = [int(x) for x in self._seq_lens_cpu(cam)[:num_reqs]]
        width = cam.block_table_tensor.shape[1]
        # The lens may be an upper bound (seq_lens_cpu_upper_bound): fine for
        # T5 over-planning, but the page map indexes the block table with it
        # — cap at the table's width or the gather runs off the end (the IMA
        # the warmup hit). The extra pages expand garbage that no valid plan
        # reads.
        n_pages = [min(max(1, -(-L // self._page)) if L > 0 else 0, width) for L in lens]
        total = sum(n_pages)
        remap = torch.zeros(cam.block_table_tensor.shape[0], width, dtype=torch.int32)
        req_idx, page_idx, off = [], [], 0
        for r, n in enumerate(n_pages):
            if n:
                remap[r, :n] = torch.arange(off, off + n, dtype=torch.int32)
                req_idx += [r] * n
                page_idx += list(range(n))
            off += n
        dev = cam.block_table_tensor.device
        rows = torch.tensor(req_idx, dtype=torch.long).pin_memory().to(dev, non_blocking=True)
        cols = torch.tensor(page_idx, dtype=torch.long).pin_memory().to(dev, non_blocking=True)
        src = cam.block_table_tensor[rows, cols].to(torch.int32).contiguous()
        return {"src_pages": src, "block_table": remap.pin_memory().to(dev, non_blocking=True), "total": total}

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
        # keep the base decision — except on fp6, whose fallback path expands
        # per-request pages and has no shared-prefix variant.
        if self._fp6:
            return False
        return super().use_cascade_attention(*args, **kwargs)
