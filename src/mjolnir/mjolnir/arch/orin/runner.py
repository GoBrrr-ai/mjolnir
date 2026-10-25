"""The Orin (sm_87) runner: the GEMV decode kernel's plan buffers, compile
cache and launch — the family sibling of :mod:`mjolnir.arch.thor.runner`,
consuming the same shared machinery from :mod:`mjolnir.runner`.

Everything the vLLM backend needs that is not vLLM-specific:

* a FIXED split-KV grid (``grid = (h_k, ns_cap, batch)``): the capacity is
  the Alg. 1 plan at batch 1, so every capture size replays one binary;
  splits beyond a sequence's need take the kernel's ``kv_len == 0`` path
  (O = 0 / LSE = -inf) and drop out of the combine — CUDA-graph safe with
  no re-capture, exactly like the t5 family's fixed work grid;
* persistent device buffers: the ``cu_seqlens_k`` prefix sums (pinned
  staging + async copy, the t5 ``work_buf`` pattern), the (b, h_k)
  descales the kernel's 2D indexing needs (materialized from vLLM's
  per-tensor / per-head / per-(seq, head) scales by a captured broadcast
  copy), and the fp32 split partials sized by the capacity invariant;
* the launch: dense (b, M, h_q, d) Q/O views (vLLM's spec batches are
  uniform-q-len), the vLLM block table zero-copy as the page table, and
  the chunk-normalized LSE combine (torch, static shapes per capture —
  the known launch-floor debt; a fused merge is research lever #3).

Nothing here imports vLLM. The kernel is compiled per variant through the
shared on-disk cache under the ``orin`` family name.
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
)
from . import decode as gemv
from .combine import combine_launch
from .plan import BLOCKS_PER_SM, COMBINE_MAX_SPLITS

_log = logging.getLogger("vllm.mjolnir")


@dataclass(frozen=True)
class GEMVKey:
    """One compiled kernel variant: dtypes + the structural flags the kernel
    bakes at trace time (M, page size, split presence, causal). The impl
    (gemv | mma) selects the compiled kernel class."""

    q_dtype: torch.dtype  # bfloat16 | float8_e4m3fn
    out_dtype: torch.dtype
    heads_q: int
    heads_k: int
    page_size: int
    q_len: int  # M: the compile-time token count
    split: bool  # False: write-through, no partials/combine
    causal: bool
    q_descale: bool  # fp8 Q with per-(seq, head) descales
    kv_format: str = "fp8"  # fp8 | e2m3 (adds the code LUT arg) | bf16
    # The TVM-FFI signature bakes every tensor dim, so the shapes the launch
    # will use are part of the key: one binary per (pool size, batch, table
    # stride). vLLM has one pool + a handful of capture sizes; the disk cache
    # makes it a one-time cost per config.
    num_pages: int = 0
    batch: int = 0
    max_blocks: int = 0
    impl: str = "gemv"  # gemv | mma — the compiled kernel class


FP8_MAX = 448.0  # the e4m3 max — the dynamic Q quantization's scale divisor


class GEMVDecodeRunner:
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
        q_mode: str = "bf16",  # "bf16" | "fp8" (dynamic per (seq, kv head))
        max_splits: int = 0,  # 0 = the Alg. 1 capacity
        kv_stages: int = 0,  # ring depth override (0 = the kernel default)
        tile: int = 0,  # rows-per-(ty)-thread override (0 = the kernel default)
        kv_evict_first: bool = False,  # reserved: Ampere L2 policy (not wired yet)
        kv_format: str = "fp8",
        impl: str = "gemv",  # "gemv" | "mma" (the design: orin1/notes/mma-kernel-design.md)
        device: torch.device | str = "cuda",
        sm_count: int | None = None,
    ):
        self.impl = impl
        if impl == "mma":
            from mjolnir.arch.orin import mma as mma_kernel

            assert head_size == mma_kernel.MMADecodeKernel.HEAD_DIM, (
                f"the Orin MMA kernel serves head_size {mma_kernel.MMADecodeKernel.HEAD_DIM} only"
            )
            self._kernel_mod, self._kernel_cls = mma_kernel, mma_kernel.MMADecodeKernel
        else:
            assert impl == "gemv", f"unknown orin kernel impl {impl!r} (gemv | mma)"
            assert head_size == gemv.GEMVDecodeKernel.HEAD_DIM, (
                f"the Orin GEMV kernel serves head_size {gemv.GEMVDecodeKernel.HEAD_DIM} only"
            )
            self._kernel_mod, self._kernel_cls = gemv, gemv.GEMVDecodeKernel
        assert num_heads % num_kv_heads == 0
        assert page_size in SUPPORTED_PAGE_SIZES, f"page size {page_size} unsupported"
        assert q_mode in ("bf16", "fp8")
        assert out_dtype in (torch.bfloat16, torch.float16)
        assert max_q_len <= 8, "the GEMV kernel serves M <= 8 (the MTP verify shape)"
        assert kv_format in ("fp8", "e2m3", "bf16"), kv_format
        self.h_q, self.h_k, self.d = num_heads, num_kv_heads, head_size
        self.h_r = num_heads // num_kv_heads
        self.page_size = page_size
        self.out_dtype = out_dtype
        self.max_num_seqs = max_num_seqs
        self.max_q_len = max_q_len
        self.fp8_q = q_mode == "fp8"
        self.kv_format = kv_format
        self.kv_evict_first = bool(kv_evict_first)
        self.device = torch.device(device)
        # e2m3: the 64-entry code -> value table (exact for every code,
        # subnormals included), passed to the kernel as a gmem tensor.
        self._lut = None
        if kv_format == "e2m3":
            from mjolnir.kvfp6 import e2m3_values

            self._lut = e2m3_values(self.device)
        if sm_count is None:
            sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.sm_count = sm_count

        # Fixed split capacity: the Alg. 1 plan at batch 1 (its maximum), so
        # one grid serves every capture size; larger batches leave the extra
        # splits empty (kv_len == 0 -> O = 0 / LSE = -inf, dropped by the
        # combine). The kernel's empty-chunk path makes the over-plan free.
        # BLOCKS_PER_SM is the kernel's resident-CTA occupancy: the GEMV
        # CTA (~100 regs, ~6 KiB smem) fits 4; every MMA shape is pinned
        # at 2 (255 regs + 50-82 KiB smem both bind) — the measured
        # optimum across L (the 4-CTA capacity over-splits the 16-SM
        # grid and the extra partials only add combine traffic).
        blocks_per_sm = 2 if impl == "mma" else BLOCKS_PER_SM
        self.ns_cap = (
            max_splits
            if max_splits > 0
            else min(
                COMBINE_MAX_SPLITS,
                (blocks_per_sm * sm_count + self.h_k - 1) // self.h_k,
            )
        )
        self.split = self.ns_cap > 1
        self.kv_stages = kv_stages
        self.tile = tile

        dev = self.device
        rows_cap = max_num_seqs * max_q_len
        self._cu = torch.zeros(max_num_seqs + 1, dtype=torch.int32, device=dev)
        # (b, h_k) descales the kernel's 2D indexing needs; a captured
        # broadcast copy materializes them from vLLM's scale shapes per step.
        self._kd = torch.ones(max_num_seqs, self.h_k, dtype=torch.float32, device=dev)
        self._vd = torch.ones(max_num_seqs, self.h_k, dtype=torch.float32, device=dev)
        self._qd = torch.ones(max_num_seqs, self.h_k, dtype=torch.float32, device=dev) if self.fp8_q else None
        # fp8 Q: the dynamic per-(seq, kv head) quantization packs the bf16
        # query here (the Thor qpack semantics); a pre-quantized fp8 query
        # passes through and its descales land in _qd via run(q_scale=...).
        self.q_pack = (
            torch.empty(rows_cap * self.h_q * self.d, dtype=torch.float8_e4m3fn, device=dev) if self.fp8_q else None
        )
        # Split partials: (ns, rows, h_q, d) fp32 and (ns, h_q, rows) fp32,
        # allocated at the capacity. Realistic sizes: ns_cap=16, 512 rows,
        # h_q=24, d=256 -> 200 MB worst case, ~3 MB at the Orin 8-seq tuning.
        self._o_part = torch.empty(self.ns_cap * rows_cap * self.h_q * self.d, dtype=torch.float32, device=dev)
        self._lse_part = torch.empty(self.ns_cap * self.h_q * rows_cap, dtype=torch.float32, device=dev)
        self._rows_cap = rows_cap
        self._combine_fns: dict = {}  # (b, ql) -> the fused combine (combine.py)

    # ------------------------------------------------------------------ plan
    def n_tiles(self, q_len: int) -> int:
        """The t5 runner's tile count has no GEMV meaning; 1 CTA covers all
        M rows of a (kv_head, chunk). Kept for surface parity."""
        return 1

    def slots(self, q_len: int) -> int:
        """Sequences served by one wave at this q_len (the t5 invariant)."""
        return max(1, (self.sm_count * BLOCKS_PER_SM) // (self.h_k * self.n_tiles(q_len)))

    def plan(
        self,
        seq_lens_cpu,
        num_seqs: int,
        q_len: int,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        margin: int = 0,
        causal: bool = True,
    ) -> StepPlan:
        """CPU plan + async copy into the persistent device buffers.

        ``margin`` plans every sequence that many tokens longer (the fused
        multi-step draft decode); the kernel masks with the device
        ``cu_seqlens``, so the plan stays valid while they grow."""
        assert 1 <= q_len <= self.max_q_len and num_seqs <= self.max_num_seqs
        lens = [int(x) + margin for x in seq_lens_cpu[:num_seqs]]
        # cu_seqlens_k: prefix sums (device, graph-safe via pinned staging).
        cu = [0]
        for n in lens:
            cu.append(cu[-1] + max(n, 0))
        cu_h = torch.tensor(cu, dtype=torch.int32).pin_memory()
        self._cu[: num_seqs + 1].copy_(cu_h, non_blocking=True)
        return StepPlan(
            num_seqs=num_seqs,
            q_len=q_len,
            n_q=q_len * self.h_r,
            num_work=self.ns_cap,  # the fixed grid.y (split capacity)
            with_reduction=self.split,
            block_table=block_table,
            seq_lens=seq_lens,
            work=self._cu,  # repurposed: the device cu_seqlens_k
            n_splits=self._cu.new_zeros(num_seqs),  # unused by this family
            chunk_tiles=0,
            num_chunks=0,
            causal=causal or q_len == 1,
        )

    # ---------------------------------------------------------------- compile
    def _key(
        self,
        q_len: int,
        causal: bool = True,
        num_pages: int = 0,
        batch: int = 0,
        max_blocks: int = 0,
    ) -> GEMVKey:
        return GEMVKey(
            impl=self.impl,
            q_dtype=torch.float8_e4m3fn if self.fp8_q else torch.bfloat16,
            out_dtype=self.out_dtype,
            heads_q=self.h_q,
            heads_k=self.h_k,
            page_size=self.page_size,
            q_len=q_len,
            split=self.split,
            causal=causal,
            q_descale=self.fp8_q,
            kv_format=self.kv_format,
            num_pages=num_pages,
            batch=batch,
            max_blocks=max_blocks,
        )

    def _cute(self, t: torch.Tensor, assumed_align: int = 16):
        from cutlass.cute.runtime import from_dlpack

        src = t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t
        ct = from_dlpack(src, assumed_align=assumed_align)
        if t.dtype == torch.float8_e4m3fn:
            ct.element_type = gemv.cutlass.Float8E4M3FN
        return ct

    def _descales(self, b: int):
        qd_buf = self._qd  # local: ty narrows the Optional against the guard
        q = qd_buf[:b] if qd_buf is not None else None
        return gemv.DescaleTensors(
            q_descale=self._cute(q) if q is not None else None,
            k_descale=self._cute(self._kd[:b], assumed_align=4),
            v_descale=self._cute(self._vd[:b], assumed_align=4),
        )

    def _compile_args(self, key: GEMVKey, b: int, q, o, k_cache, v_cache, block_table):
        """The compile-time argument tuple: cute tensors (they bind dtypes,
        shapes and the Constexpr M) + the fake TVM-FFI env stream."""
        import cutlass.cute as cute

        rows = b * key.q_len
        ns = self.ns_cap if key.split else 1
        m_q = self._cute(q[:rows].view(b, key.q_len, self.h_q, self.d))
        m_o = self._cute(o[:rows].view(b, key.q_len, self.h_q, self.d))
        m_k = self._cute(k_cache)
        m_v = self._cute(v_cache)
        pt = self._cute(block_table)
        ds = self._descales(b)
        return (
            m_q,
            m_k,
            m_v,
            m_o,
            None,  # mLSE: vLLM decode needs no final LSE
            None,  # mCuSeqlensQ: dense (b, M, h_q, d) Q
            self._cute(self._cu[: b + 1], assumed_align=4),  # mCuSeqlensK
            ds,
            self._cute(self._lut, assumed_align=4) if self._lut is not None else None,
            self._cute(
                self._o_part[: ns * rows * self.h_q * self.d].view(ns, b, key.q_len, self.h_q, self.d),
                assumed_align=4,
            )
            if key.split
            else None,
            # The kernel's dense-Q split LSE coord is (split, batch, head, m)
            # -> the partial is (ns, b, h_q, M).
            self._cute(
                self._lse_part[: ns * b * self.h_q * key.q_len].view(ns, b, self.h_q, key.q_len),
                assumed_align=4,
            )
            if key.split
            else None,
            pt,
            key.q_len,  # seqlen_q (Constexpr — baked at compile)
            0,  # seqlen_k: unused on the varlen-k path (cu_seqlens_k rules)
            0.0,  # softmax_scale_log2 (runtime)
            ns,  # num_splits (runtime grid.y)
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        )

    def _torch_args(self, key: GEMVKey, b: int, q, o, k_cache, v_cache, block_table):
        """The launch-time tuple: RAW torch tensors (the TVM-FFI convention —
        fp8 views as uint8, the archived interface's workaround), python
        scalars, and no stream (the env stream is implicit)."""
        rows = b * key.q_len
        ns = self.ns_cap if key.split else 1
        q_call = q[:rows].view(b, key.q_len, self.h_q, self.d)
        o_call = o[:rows].view(b, key.q_len, self.h_q, self.d)
        k_call, v_call = k_cache, v_cache
        qd_buf = self._qd  # local: ty narrows the Optional against the guard
        ds = gemv.DescaleTensors(
            q_descale=qd_buf[:b] if qd_buf is not None else None,
            k_descale=self._kd[:b],
            v_descale=self._vd[:b],
        )
        return (
            q_call,
            k_call,
            v_call,
            o_call,
            None,  # mLSE
            None,  # mCuSeqlensQ
            self._cu[: b + 1],
            ds,
            self._lut,
            self._o_part[: ns * rows * self.h_q * self.d].view(ns, b, key.q_len, self.h_q, self.d)
            if key.split
            else None,
            self._lse_part[: ns * b * self.h_q * key.q_len].view(ns, b, self.h_q, key.q_len) if key.split else None,
            block_table,
            0,  # seqlen_k (unused on the varlen-k path)
        )

    def _get_compiled(self, key: GEMVKey, compile_args):
        import cutlass.cute as cute

        with _COMPILE_LOCK:
            fn = _COMPILED.get(key)
            if fn is not None:
                return fn
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"GEMV kernel variant {key} not compiled before CUDA-graph "
                    "capture; call precompile() (the vLLM backend does this)"
                )
            fn = _disk_load(key, self._kernel_mod.__file__, "orin")
            if fn is None:
                kernel = self._kernel_cls(
                    self.h_r,
                    tile_size_per_bdx=self.tile or 4,
                    page_size=self.page_size,
                    causal=key.causal,
                    kv_format=key.kv_format,
                )
                fn = cute.compile(kernel, *compile_args, options="--opt-level 2 --enable-tvm-ffi")
                CACHE_STATS["compiled"] += 1
                _disk_store(key, fn, self._kernel_mod.__file__, "orin")
            else:
                CACHE_STATS["loaded"] += 1
            _COMPILED[key] = fn
        return fn

    def is_compiled(self, q_len: int, causal: bool = True) -> bool:
        return self._key(q_len, causal) in _COMPILED

    def ensure_compiled(
        self,
        q_len: int,
        with_reduction: bool = False,  # thor-layout no-op here: the combine is always the fused kernel
        causal: bool = True,
        num_pages: int = 0,
        batch: int = 1,
        max_blocks: int = 1,
    ):
        """Compile one variant with dummy tensors if it is missing. The
        positional layout matches the Thor runner's (the shared backend's
        build() calls ``ensure_compiled(q_len, plan.with_reduction,
        plan.causal)`` blind) -- the args used to land shifted and the
        dummy-pool size became a bool.

        The TVM-FFI signature bakes tensor shapes, so the dummies must match
        the shapes the launch will use: pass the pool size, the batch (the
        graph-capture size) and the block-table stride. run() compiles on
        miss for eager calls; capture paths must precompile per shape."""
        key = self._key(q_len, causal, num_pages, batch, max_blocks)
        if key in _COMPILED:
            return
        dev = self.device
        b, ps = batch, self.page_size or 16
        np_ = num_pages or ps  # one page worth of pool at minimum
        q = torch.zeros(b * q_len, self.h_q, self.d, device=dev, dtype=key.q_dtype)
        o = torch.empty(b * q_len, self.h_q, self.d, device=dev, dtype=key.out_dtype)
        kv_dtype = {"fp8": torch.float8_e4m3fn, "e2m3": torch.uint8, "bf16": torch.bfloat16}[key.kv_format]
        kv = torch.zeros(np_, ps, self.h_k, self.d, device=dev, dtype=kv_dtype)
        bt = torch.zeros(b, max_blocks, dtype=torch.int32, device=dev)
        self._cu[1] = q_len  # one sequence of q_len tokens
        self._get_compiled(key, self._compile_args(key, b, q, o, kv, kv, bt))
        torch.cuda.synchronize(dev)

    def precompile(self, q_lens=None):
        """Compile every variant this runner can launch."""
        for ql in q_lens or range(1, self.max_q_len + 1):
            self.ensure_compiled(ql)
        torch.cuda.synchronize(self.device)

    # ---------------------------------------------------------------- launch
    def _scale_buf(self, buf: torch.Tensor, b: int, scale: torch.Tensor) -> torch.Tensor:
        """vLLM's (1,)/(h_k,)/(b, h_k) descale -> the kernel's (b, h_k) view,
        as one captured broadcast copy."""
        view = buf[:b]
        if scale.dim() == 0 or scale.numel() == 1:
            view.copy_(torch.ones_like(view) * scale.reshape(()))
        elif scale.dim() == 1 and scale.shape[0] == self.h_k:
            view.copy_(scale.unsqueeze(0).expand(b, self.h_k))
        else:
            view.copy_(scale.reshape(b, self.h_k))
        return view

    def _pack_q(self, q3: torch.Tensor, b: int, ql: int) -> torch.Tensor:
        """Dynamic per-(seq, kv head) e4m3 quantization of the bf16 query —
        the Thor qpack semantics: the group's amax over (ql, h_r, d) gives
        the scale, which lands in the descale buffer and divides the pack
        (the query's h_q axis is kv-head-major, so the 5D view regroups it).
        Returns the packed rows (flat)."""
        pack, qd = self.q_pack, self._qd  # locals: ty narrows against the guard
        assert pack is not None and qd is not None, "the fp8-Q buffers exist only on the fp8-Q runner"
        qg = q3.view(b, ql, self.h_k, self.h_r, self.d).float()
        scale = qg.abs().amax(dim=(1, 3, 4)).clamp_min(1e-6) / FP8_MAX  # (b, h_k)
        qd[:b] = scale
        packed = (qg / scale[:, None, :, None, None]).to(torch.float8_e4m3fn)
        rows = b * ql
        pack[: rows * self.h_q * self.d].copy_(packed.reshape(-1))
        return pack[: rows * self.h_q * self.d].view(rows, self.h_q, self.d).reshape(-1)

    def _combine(self, b: int, ql: int, out: torch.Tensor):
        """The chunk-normalized LSE combine: O = Σ_s exp(lse_s − L)·O_s
        with L = max + log Σ exp. One fused kernel (combine.py) — the
        eager torch form issued ~20 small kernels (~150 us of GPU per
        call, more than the decode kernel itself). Compiled once per
        capacity config; the shapes are runtime Int32s so one binary
        serves every capture size. Capture paths must warm up before
        capture (the vLLM backend does)."""
        rows = b * ql
        ns = self.ns_cap
        # raw torch tensors (the TVM-FFI convention, like _torch_args);
        # one compiled variant per (batch, q_len) — the shapes the FFI
        # bakes. The buffers are the capacity flats (fixed at init).
        args = (self._o_part, self._lse_part, out.view(-1), ns, rows, self.h_q, b, ql)
        key = (b, ql)
        fn = self._combine_fns.get(key)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("the combine kernel must be compiled before CUDA-graph capture")
            import cutlass.cute as cute
            from cutlass import Int32

            fn = cute.compile(
                combine_launch,
                self._cute(self._o_part, assumed_align=4),
                self._cute(self._lse_part, assumed_align=4),
                self._cute(out.view(-1), assumed_align=4),
                Int32(ns),  # max_ns: the unroll bound (= ns_cap, runner-fixed)
                Int32(ns),
                Int32(rows),
                Int32(self.h_q),
                Int32(b),
                Int32(ql),
                options="--opt-level 2 --enable-tvm-ffi",
            )
            self._combine_fns[key] = fn
        fn(*args)

    def run(
        self,
        plan: StepPlan,
        query: torch.Tensor,  # (>= b*q_len, h_q, d) bf16 (packed when fp8_q) or fp8 (+ q_scale)
        out: torch.Tensor,  # (>= b*q_len, h_q, d) or (>= b*q_len, h_q*d)
        k_cache: torch.Tensor,  # (num_pages, page_size, h_k, d) logical view
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,  # fp32: 1 elem (per tensor), (h_k,), or (b, h_k)
        v_scale: torch.Tensor,
        softmax_scale: float,
        q_mul: torch.Tensor | None = None,  # (h_q, d) fp32 — fp6 K scales (later)
        o_mul: torch.Tensor | None = None,  # (h_q, d) fp32 — fp6 V scales (later)
        q_scale: torch.Tensor | None = None,  # fp8 Q descales: (1,)/(h_k,)/(b, h_k)
    ):
        assert q_mul is None and o_mul is None, "fp6 channel scales land with the fp6 variant"
        b, ql = plan.num_seqs, plan.q_len
        tokens = b * ql
        q_pack = self.q_pack  # local: ty narrows the Optional against the fp8_q guard
        if self.fp8_q and query.dtype != torch.float8_e4m3fn:
            # bf16 query: the dynamic per-(seq, kv head) e4m3 pack writes the
            # descales itself; a pre-quantized fp8 query passes through and
            # brings its descales in q_scale (the harness/quantized path).
            q3 = query[:tokens]
            if q3.dim() == 2:
                q3 = q3.view(tokens, self.h_q, self.d)
            self._pack_q(q3, b, ql)
        elif self.fp8_q:
            assert q_scale is not None, "a pre-quantized fp8 Q needs its descales (q_scale)"
            self._scale_buf(self._qd, b, q_scale)
        if self.kv_format == "fp8":
            if k_cache.dtype == torch.uint8:
                k_cache = k_cache.view(torch.float8_e4m3fn)
            if v_cache.dtype == torch.uint8:
                v_cache = v_cache.view(torch.float8_e4m3fn)
        # Materialize the (b, h_k) descales (one captured broadcast each).
        self._scale_buf(self._kd, b, k_scale)
        self._scale_buf(self._vd, b, v_scale)
        key = self._key(ql, plan.causal, k_cache.shape[0], b, plan.block_table.shape[1])
        if self.fp8_q and query.dtype != torch.float8_e4m3fn:
            assert q_pack is not None, "the pack buffer exists only on the fp8-Q runner"
            q3 = q_pack[: tokens * self.h_q * self.d].view(tokens, self.h_q, self.d)
        else:
            q3 = query[:tokens]
            if q3.dim() == 2:
                q3 = q3.view(tokens, self.h_q, self.d)
        # Compile args are expensive (several from_dlpack round trips) --
        # build them only on a cache miss; the hot path packs raw tensors.
        fn = _COMPILED.get(key)
        if fn is None:
            fn = self._get_compiled(
                key,
                self._compile_args(key, b, q3, out, k_cache, v_cache, plan.block_table),
            )
        launch = self._torch_args(key, b, q3, out, k_cache, v_cache, plan.block_table)
        launch = (*launch, float(softmax_scale) * math.log2(math.e), self.ns_cap if key.split else 1)
        fn(*launch)
        if key.split:
            self._combine(b, ql, out)
