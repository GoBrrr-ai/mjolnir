"""The Thor (sm_110) runner: the T5 decode kernel's plan buffers, compile
cache and launch — the family-specific half of the old ``mjolnir.runner``
(now the shared machinery in :mod:`mjolnir.runner` plus this module).

Everything the vLLM backend needs that is not vLLM-specific:
  * the occupancy-pinned split plan (``plan.plan_split_kv``), written
    into FIXED-ADDRESS device buffers and padded to a FIXED grid size per
    q_len -> CUDA-graph safe
  * a process-wide cache of compiled kernel variants (+ precompile),
    persisted to the on-disk kernel cache under the ``thor`` family name
  * the launch: zero-copy Q/O views for q_len == 1, packing for q_len > 1,
    per-tensor / per-head / per-(seq, head) descales via strides, optional
    dynamic fp8 Q quantization

Nothing here imports vLLM, so tests/kernel/test_runner.py exercises it
directly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

from mjolnir.runner import (
    _COMPILED,
    _COMPILE_LOCK,
    CACHE_STATS,
    StepPlan,
    SUPPORTED_PAGE_SIZES,
    _disk_load,
    _disk_store,
    _ptr,
    _stream,
    env_flag,
)
from . import qpack
from . import decode as t5
from . import plan as t5p

FP8_MAX = 448.0
_log = logging.getLogger("vllm.mjolnir")


def _kv_cutlass_dtype(kv_format: str):
    import cutlass

    return {"fp8": cutlass.Float8E4M3FN, "e2m3": cutlass.Float6E2M3FN}[kv_format]


def tile_width(n_q: int, fp8_q: bool) -> int:
    """Rows per CTA tile: next power of two >= n_q, min 8 (bf16) / 16 (fp8), max 32."""
    base = 16 if fp8_q else 8
    return max(base, min(32, 1 << max(0, (n_q - 1).bit_length())))


@dataclass(frozen=True)
class VariantKey:
    q_dtype: torch.dtype  # bfloat16 | float8_e4m3fn
    out_dtype: torch.dtype
    head_size: int
    heads_q: int
    heads_k: int
    page_size: int
    q_len: int
    max_splits: int
    with_reduction: bool
    ctas_per_sm: int
    kv_stages: int  # 0 = kernel default
    pdl: bool
    l2_prefetch: int
    kv_evict_first: bool
    kv_format: str = "fp8"  # "fp8" | "e2m3" (packed fp6, read by kind::f8f6f4)
    causal: bool = True  # False: parallel-drafter blocks (every query sees every key)


# KV formats the vLLM path serves: e4m3 bytes, or e2m3 codes packed 4 per 3
# bytes (little-endian bitstream, mjolnir.quant.pack_subbyte).
RUNNER_KV_FORMATS = {"fp8": 8, "e2m3": 6}


class T5DecodeRunner:
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        page_size: int,
        out_dtype: torch.dtype,
        max_num_seqs: int,
        max_q_len: int = 1,
        q_mode: str = "bf16",  # "bf16" (Q as is) | "fp8" (dynamic per (seq, kv head))
        ctas_per_sm: int = 2,
        max_splits: int = 16,
        kv_stages: int = 0,
        pdl: bool = False,  # programmatic dependent launch (decode + reduction)
        l2_prefetch: int = 0,  # TMA L2 prefetch distance in 128-row KV tiles
        kv_evict_first: bool = False,  # EVICT_FIRST L2 policy on K/V TMA loads
        kv_format: str = "fp8",  # "fp8" | "e2m3" (fp8 Q only)
        device: torch.device | str = "cuda",
        sm_count: int | None = None,
    ):
        assert num_heads % num_kv_heads == 0
        assert head_size % 128 == 0, "T5 decode needs head_size % 128 == 0"
        assert page_size in SUPPORTED_PAGE_SIZES, f"page size {page_size} unsupported"
        assert q_mode in ("bf16", "fp8")
        assert out_dtype in (torch.bfloat16, torch.float16)
        self.h_q, self.h_k, self.d = num_heads, num_kv_heads, head_size
        self.h_r = num_heads // num_kv_heads
        self.page_size = page_size
        self.out_dtype = out_dtype
        self.max_num_seqs = max_num_seqs
        self.max_q_len = max_q_len
        self.fp8_q = q_mode == "fp8"
        self.ctas_per_sm = ctas_per_sm
        self.max_splits = max_splits
        self.kv_stages = kv_stages
        self.pdl = bool(pdl)
        self.l2_prefetch = int(l2_prefetch)
        self.kv_evict_first = bool(kv_evict_first)
        assert kv_format in RUNNER_KV_FORMATS, f"kv_format must be one of {tuple(RUNNER_KV_FORMATS)}"
        assert kv_format == "fp8" or q_mode == "fp8", f"{kv_format} KV runs on the fp8-Q path only"
        self.kv_format = kv_format
        self.kv_bits = RUNNER_KV_FORMATS[kv_format]
        self.kv_row_bytes = head_size * self.kv_bits // 8  # bytes of one K (or V) row
        self.device = torch.device(device)
        if sm_count is None:
            sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.sm_count = sm_count

        # Capacity per q_len: work items that fit one wave (slots) or one per
        # sequence, whichever is larger. Grid.x is always this -> fixed grid.
        self._slots = {ql: self.slots(ql) for ql in range(1, max_q_len + 1)}
        self._cap = {ql: max(self._slots[ql], max_num_seqs) for ql in self._slots}
        max_cap = max(self._cap.values())
        max_slots = max(self._slots.values())
        nq_max = max_q_len * self.h_r
        dev = self.device

        # Persistent (fixed-address) buffers
        self.work_buf = torch.zeros(max_cap, 4, dtype=torch.int32, device=dev)
        self.nsplit_buf = torch.ones(max_num_seqs, dtype=torch.int32, device=dev)
        self.lse_buf = torch.empty(max_num_seqs * self.h_k * nq_max, dtype=torch.float32, device=dev)
        # Partials are only written for sequences the plan split, and the plan
        # splits only when num_seqs < slots -> size them by slots, not batch.
        part = max_splits * max_slots * self.h_k * nq_max
        self.o_part = torch.empty(part * head_size, dtype=torch.float32, device=dev)
        self.m_part = torch.empty(part, dtype=torch.float32, device=dev)
        self.l_part = torch.empty(part, dtype=torch.float32, device=dev)
        # Pack buffers: used for q_len > 1, fp8 Q, or a Q/O whose head stride
        # is not d (always allocated so any vLLM view layout works).
        rows = max_num_seqs * self.h_k * nq_max * head_size
        q_pack_dtype = torch.float8_e4m3fn if self.fp8_q else torch.bfloat16
        self.q_pack = torch.empty(rows, dtype=q_pack_dtype, device=dev)
        self.o_pack = torch.empty(rows, dtype=out_dtype, device=dev)
        # placeholder for the kernel's nvfp4 scale pointers (the vLLM path
        # runs the fp8 KV format)
        self.dummy_scales = torch.zeros(16, dtype=torch.float8_e4m3fn, device=dev)
        self.q_descale = (
            torch.ones(max_num_seqs, self.h_k, dtype=torch.float32, device=dev)
            if self.fp8_q
            else torch.ones(1, dtype=torch.float32, device=dev)
        )
        # One fused Triton launch for the fp8 Q quantization (MJOLNIR_ATTN_FUSED_QPACK=0:
        # the PyTorch ops, ~10 launches per layer and step).
        self._fused_qpack = (
            self.fp8_q
            and self.device.type == "cuda"
            and qpack.available()
            and env_flag("MJOLNIR_ATTN_FUSED_QPACK", "1").lower() not in ("0", "false", "no")
        )

    # ------------------------------------------------------------------ plan
    def n_tiles(self, q_len: int) -> int:
        n_q = q_len * self.h_r
        return math.ceil(n_q / tile_width(n_q, self.fp8_q))

    def slots(self, q_len: int) -> int:
        return max(1, (self.sm_count * self.ctas_per_sm) // (self.h_k * self.n_tiles(q_len)))

    def plan(
        self,
        seq_lens_cpu,  # sequence of ints (len >= num_seqs)
        num_seqs: int,
        q_len: int,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        margin: int = 0,
        causal: bool = True,
    ) -> StepPlan:
        """CPU plan + async copy into the persistent device buffers.

        ``margin`` plans every sequence that many tokens longer: the kernel
        masks with the device ``seq_lens``, so the plan stays valid while
        they grow by up to ``margin`` (fused multi-step draft decode)."""
        assert 1 <= q_len <= self.max_q_len and num_seqs <= self.max_num_seqs
        lens = [int(x) + margin for x in seq_lens_cpu[:num_seqs]]
        cap = self._cap[q_len]
        p = t5p.plan_split_kv(
            lens,
            self.h_k,
            self.n_tiles(q_len),
            self.sm_count,
            ctas_per_sm=self.ctas_per_sm,
            max_splits=self.max_splits,
            pad_to=cap,
        )
        # Pinned staging from the caching host allocator: safe to drop after
        # a non_blocking copy (the allocator tracks the stream use).
        work_h = torch.tensor(p.work, dtype=torch.int32).pin_memory()
        ns_h = torch.tensor(p.n_splits, dtype=torch.int32).pin_memory()
        self.work_buf[:cap].copy_(work_h, non_blocking=True)
        self.nsplit_buf[:num_seqs].copy_(ns_h, non_blocking=True)
        # Reduction launch is decided per batch SIZE (deterministic for a
        # captured CUDA graph): a batch below one wave may split, a full
        # wave never does.
        with_reduction = self.max_splits > 1 and num_seqs < self._slots[q_len]
        return StepPlan(
            num_seqs=num_seqs,
            q_len=q_len,
            n_q=q_len * self.h_r,
            num_work=cap,
            with_reduction=with_reduction,
            block_table=block_table,
            seq_lens=seq_lens,
            work=self.work_buf[:cap],
            n_splits=self.nsplit_buf[:num_seqs],
            chunk_tiles=p.chunk_tiles,
            num_chunks=sum(1 for w in p.work if w[2] > w[1]),
            causal=causal or q_len == 1,  # one query token: the same kernel either way
        )

    # --------------------------------------------------------------- compile
    def _key(self, q_len: int, with_reduction: bool, causal: bool = True) -> VariantKey:
        return VariantKey(
            q_dtype=torch.float8_e4m3fn if self.fp8_q else torch.bfloat16,
            out_dtype=self.out_dtype,
            head_size=self.d,
            heads_q=self.h_q,
            heads_k=self.h_k,
            page_size=self.page_size,
            q_len=q_len,
            max_splits=self.max_splits,
            with_reduction=with_reduction,
            ctas_per_sm=self.ctas_per_sm,
            kv_stages=self.kv_stages,
            pdl=self.pdl,
            l2_prefetch=self.l2_prefetch,
            kv_evict_first=self.kv_evict_first,
            kv_format=self.kv_format,
            causal=causal or q_len == 1,
        )

    def _make_kernel(self, key: VariantKey):
        n_q = key.q_len * self.h_r
        fp8 = key.q_dtype == torch.float8_e4m3fn
        if fp8:
            cw = 0
        elif self.d > 128 and key.ctas_per_sm == 1:
            cw = 2
        else:
            cw = 1
        k = t5.MixedInputFusedMultiHeadAttentionDecode(
            headdim=self.d,
            grouped_head_tile=tile_width(n_q, fp8),
            convert_warpgroups=cw,
            page_size=key.page_size,
            q_len=key.q_len,
            max_splits=key.max_splits,
            ctas_per_sm=key.ctas_per_sm,
            max_kv_stages=key.kv_stages,
            with_reduction=key.with_reduction,
            pdl=key.pdl,
            l2_prefetch=key.l2_prefetch,
            kv_evict_first=key.kv_evict_first,
            kv_format=key.kv_format,
            causal=key.causal,
        )
        import cutlass

        k.can_implement(
            1,
            self.h_q,
            self.h_k,
            self.d,
            cutlass.Float8E4M3FN if fp8 else cutlass.BFloat16,
            _kv_cutlass_dtype(key.kv_format),
        )
        return k

    def _get_compiled(self, key: VariantKey, args):
        fn = _COMPILED.get(key)
        if fn is not None:
            return fn
        with _COMPILE_LOCK:
            fn = _COMPILED.get(key)
            if fn is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        f"T5 kernel variant {key} not compiled before CUDA-graph "
                        "capture; call precompile() (the vLLM backend does this)"
                    )
                fn = _disk_load(key, t5.__file__, "thor")
                if fn is None:
                    import cutlass.cute as cute

                    kernel = self._make_kernel(key)
                    fn = cute.compile(kernel, *args, options="--opt-level 2")
                    CACHE_STATS["compiled"] += 1
                    _disk_store(key, fn, t5.__file__, "thor")
                else:
                    CACHE_STATS["loaded"] += 1
                _COMPILED[key] = fn
        return fn

    def is_compiled(self, q_len: int, with_reduction: bool, causal: bool = True) -> bool:
        return self._key(q_len, with_reduction, causal) in _COMPILED

    def ensure_compiled(self, q_len: int, with_reduction: bool, causal: bool = True):
        """Compile one variant with tiny dummy tensors if it is missing.
        Runtime shapes/strides/pointers are dynamic, so one compile per
        (q_len, reduction) serves every batch and every layer. Must run
        outside CUDA-graph capture (the vLLM builder calls it from build())."""
        if self.is_compiled(q_len, with_reduction, causal):
            return
        dev = self.device
        PS, d = self.page_size, self.d
        n_q = q_len * self.h_r
        if self.kv_format == "fp8":
            kv = torch.zeros(1, PS, self.h_k, d, dtype=torch.float8_e4m3fn, device=dev)
        else:  # packed bytes, like vLLM's uint8 view
            kv = torch.zeros(1, PS, self.h_k, self.kv_row_bytes, dtype=torch.uint8, device=dev)
        one = torch.ones(1, dtype=torch.float32, device=dev)
        q = torch.zeros(1, self.h_k, n_q, d, device=dev, dtype=torch.float8_e4m3fn if self.fp8_q else torch.bfloat16)
        o = torch.empty(1, self.h_k, n_q, d, dtype=self.out_dtype, device=dev)
        plan = StepPlan(
            num_seqs=1,
            q_len=q_len,
            n_q=n_q,
            num_work=1,
            with_reduction=with_reduction,
            block_table=torch.zeros(1, 1, dtype=torch.int32, device=dev),
            seq_lens=torch.full((1,), PS, dtype=torch.int32, device=dev),
            work=torch.tensor([[0, 0, 1, 0]], dtype=torch.int32, device=dev),
            n_splits=torch.ones(1, dtype=torch.int32, device=dev),
            chunk_tiles=1,
            num_chunks=1,
            causal=causal or q_len == 1,
        )
        self._launch(plan, q, o, kv, kv, one, one, one, 1.0, compile_only=True)

    def precompile(self, q_lens=None):
        """Compile every variant this runner can launch."""
        for ql in q_lens or range(1, self.max_q_len + 1):
            for red in (False, True) if self.max_splits > 1 else (False,):
                self.ensure_compiled(ql, red)
        torch.cuda.synchronize(self.device)

    # ---------------------------------------------------------------- launch
    def _kv_ptr(self, t: torch.Tensor):
        """K/V pointer: e4m3 elements, or packed fp6 elements of a uint8 byte
        view (the kernel converts byte strides to element strides)."""
        if self.kv_format == "fp8":
            return _ptr(t)
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_ptr

        assert t.dtype == torch.uint8 and t.data_ptr() % 32 == 0, "packed fp6 KV needs a 32-byte aligned uint8 view"
        return make_ptr(_kv_cutlass_dtype(self.kv_format), t.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    def _launch(self, plan, q, o, k_cache, v_cache, qd, kd, vd, scale, compile_only=False):
        b, n_q = plan.num_seqs, plan.n_q
        strides = t5p.kernel_strides(q, o, k_cache, v_cache, qd, kd, vd, kv_bits=self.kv_bits)
        nqk = self.h_k * n_q
        args = (
            (b, self.h_k, n_q, self.d),
            plan.block_table.stride(0),  # block table row length (max_blocks)
            k_cache.shape[0],  # num_pages
            plan.num_work,
            strides,
            _ptr(q),
            self._kv_ptr(k_cache),
            self._kv_ptr(v_cache),
            _ptr(self.dummy_scales),
            _ptr(self.dummy_scales),  # nvfp4 scales (unused: fp8 KV)
            _ptr(plan.block_table),
            _ptr(plan.seq_lens),
            _ptr(plan.work),
            _ptr(plan.n_splits),
            _ptr(qd),
            _ptr(kd),
            _ptr(vd),
            _ptr(o),
            _ptr(self.lse_buf[: max(1, b * nqk)]),
            _ptr(self.o_part),
            _ptr(self.m_part),
            _ptr(self.l_part),
            float(scale),
            1.0,
            _stream(),
        )
        fn = self._get_compiled(self._key(plan.q_len, plan.with_reduction, plan.causal), args)
        if not compile_only:
            fn(*args)

    def _pack_q_fused(self, q3, b: int, ql: int, q_view, qd, q_mul=None) -> bool:
        """fp8 Q quantization + packing in one Triton launch (qpack). False
        when unavailable; a failure disables it for good, outside capture."""
        if not self._fused_qpack or q3.stride(-1) != 1:
            return False
        try:
            qpack.pack_q_fp8(q3, b, ql, self.h_k, self.h_r, q_view, qd, q_mul)
            return True
        except (
            Exception
        ) as e:  # re-raises under graph capture; otherwise falls back (e.g. a Triton build that cannot target this GPU)
            if torch.cuda.is_current_stream_capturing():
                raise
            self._fused_qpack = False
            _log.warning("Mjolnir: fused fp8 Q pack unavailable (%r); using the PyTorch ops", e)
            return False

    def run(
        self,
        plan: StepPlan,
        query: torch.Tensor,  # (>= b*q_len, h_q, d) bf16, any token stride
        out: torch.Tensor,  # (>= b*q_len, h_q, d) or (>= b*q_len, h_q*d)
        k_cache: torch.Tensor,  # (num_pages, page_size, h_k, d) logical view, fp8/uint8
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,  # fp32: 1 elem (per tensor), (h_k,), or (b, h_k)
        v_scale: torch.Tensor,
        softmax_scale: float,
        q_mul: torch.Tensor | None = None,  # (h_q, d) fp32: Q *= q_mul first (fp6 K channel scales)
        o_mul: torch.Tensor | None = None,  # (h_q, d) fp32: O *= o_mul last (fp6 V channel scales)
    ):
        b, ql, n_q = plan.num_seqs, plan.q_len, plan.n_q
        h_k, h_r, d = self.h_k, self.h_r, self.d
        tokens = b * ql
        if self.kv_format == "fp8":
            if k_cache.dtype == torch.uint8:
                k_cache = k_cache.view(torch.float8_e4m3fn)
            if v_cache.dtype == torch.uint8:
                v_cache = v_cache.view(torch.float8_e4m3fn)
        q3 = query[:tokens]
        if q3.dim() == 2:
            q3 = q3.view(tokens, self.h_q, d)
        o3 = out[:tokens]
        if o3.dim() == 2:
            o3 = o3.view(tokens, self.h_q, d)

        # ---- Q -------------------------------------------------------------
        qd = self.q_descale
        if self.fp8_q:
            # dynamic per-(seq, kv head) e4m3 quantization into the pack buffer
            qd = self.q_descale[:b]
            q_view = self.q_pack[: b * h_k * n_q * d].view(b, h_k, n_q, d)
            if not self._pack_q_fused(q3, b, ql, q_view, qd, q_mul):
                qg = q3.float()
                if q_mul is not None:
                    qg = qg * q_mul.view(self.h_q, d)
                qg = qg.reshape(b, ql, h_k, h_r, d)
                amax = qg.abs().amax(dim=(1, 3, 4)).clamp_min(1e-6)  # (b, h_k)
                scale_q = amax / FP8_MAX
                qd.copy_(scale_q)
                q_view.copy_(
                    (qg / scale_q[:, None, :, None, None])
                    .to(torch.float8_e4m3fn)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(b, h_k, n_q, d)
                )
        elif q_mul is not None:
            q_view = self.q_pack[: b * h_k * n_q * d].view(b, h_k, n_q, d)
            qm = (q3.float() * q_mul.view(self.h_q, d)).to(q_view.dtype)
            q_view.copy_(qm.reshape(b, ql, h_k, h_r, d).permute(0, 2, 1, 3, 4).reshape(b, h_k, n_q, d))
        elif ql == 1 and q3.stride(2) == 1 and q3.stride(1) == d and q3.data_ptr() % 16 == 0:
            q_view = q3.unflatten(1, (h_k, h_r))  # zero-copy (b, h_k, h_r, d)
        else:
            q_view = self.q_pack[: b * h_k * n_q * d].view(b, h_k, n_q, d)
            q_view.copy_(q3.reshape(b, ql, h_k, h_r, d).permute(0, 2, 1, 3, 4).reshape(b, h_k, n_q, d))

        # ---- O -------------------------------------------------------------
        direct_out = ql == 1 and o3.stride(2) == 1 and o3.stride(1) == d and o3.data_ptr() % 16 == 0
        o_view = o3.unflatten(1, (h_k, h_r)) if direct_out else self.o_pack[: b * h_k * n_q * d].view(b, h_k, n_q, d)

        self._launch(plan, q_view, o_view, k_cache, v_cache, qd, k_scale, v_scale, softmax_scale)

        if not direct_out:
            o3.copy_(o_view.reshape(b, h_k, ql, h_r, d).permute(0, 2, 1, 3, 4).reshape(tokens, self.h_q, d))
        if o_mul is not None:
            o3.mul_(o_mul.view(self.h_q, d))
        return out
