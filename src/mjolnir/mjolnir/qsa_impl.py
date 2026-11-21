"""Mjolnir decode for vLLM's ``qwen4_exp`` QSA layers — ``MJOLNIR_ATTN_QSA``.

The stock QSA decode is: indexer scores (Triton) -> top-k -> packed
token-index buffer -> ``qsa_sparse_paged_attention`` (Triton sparse GQA
over the selected rows). The Mjolnir path replaces ONLY the last stage:
the packed selection is gathered into a page-shaped scratch (one fused
Triton launch per K/V) and the existing dense tcgen05 kernel runs on the
scratch via ``effective_seq_lens`` plans — the scratch-gather P0 that
``verify qsagain`` measured at 13.5x/11.7x dense at 262K (2026-10-31).

Scope of v1 (deliberate):
- pure-decode steps only (one query token per request): verify batches
  (MTP) and mixed/prefill steps fall back to the stock Triton path —
  per-row selections of a verify step cannot share one scratch plan.
- fp8 e4m3 main caches only (the kernel reads fp8/fp6 KV); bf16 caches
  fall back to stock.
- eager mode: the host-side selection→plan work syncs, so capture is not
  attempted (graph support rides on the P2 in-loader gather).

Everything else (indexer scores, top-k, the K/V cache write, the output
gate application contract) stays stock: the impl subclasses the stock
impl and only overrides ``forward``.
"""

from __future__ import annotations

import os

import torch

from mjolnir.arch.thor import qsa
from mjolnir.arch.thor.runner import T5DecodeRunner

try:  # Triton ships with vLLM / torch on CUDA; the torch path is the fallback
    from mjolnir import qsa_gather
except ImportError:  # pragma: no cover
    qsa_gather = None

# The runner and the scratch workspace are layer-independent (every QSA
# layer of a model shares the geometry) — share them across the impl
# instances. Plan→run stays adjacent within one layer's forward, and
# layers run sequentially inside one engine step, so the persistent
# plan buffers are never overwritten mid-flight.
_RUNNERS: dict[tuple, T5DecodeRunner] = {}
_SCRATCHES: dict[tuple, torch.Tensor] = {}
_SCRATCH_SLACK_PAGES = 2  # scratch pages over the exact plan: headroom for the next step's growth
_ANNOUNCED = [False]  # one-time "the Mjolnir path is live" marker
_FALLBACKS: set[str] = set()


def _note_fallback(reason: str) -> None:
    if reason not in _FALLBACKS:
        _FALLBACKS.add(reason)
        print(f"MJOLNIR QSA decode: stock path ({reason})", flush=True)


def _geometry_key(layer, page_size: int, device: torch.device) -> tuple:
    return (
        layer.num_heads,
        layer.num_kv_heads,
        layer.head_dim,
        page_size,
        layer.kv_cache_dtype,
        str(device),
    )


_IMPL_CLASS: type | None = None


def _indexer_metadata(layer):
    """The QSAIndexerMetadata of this step, via the forward context.

    0.34 moved the per-request seq lens off the main QSA metadata onto the
    indexer's side metadata; the owner layer reads it under the indexer
    cache's key. Same lookup here, keyed off the layer's indexer. None on
    any surprise - the caller falls back to the stock path.
    """
    try:
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
        if isinstance(md, list):
            md = md[0]
        if not isinstance(md, dict):
            return None
        return md.get(layer.indexer.raw_key_cache.prefix)
    except Exception:  # noqa: BLE001 - any surprise degrades to stock
        return None


def build_impl_class() -> type:
    """Import + subclass the stock impl on first use.

    The stock module participates in a module-level circular import
    (qsa.py <-> model.py), so it is only safely importable once the engine
    has loaded the qwen4_exp package — guaranteed at layer-__init__ time,
    where the patch calls this."""
    global _IMPL_CLASS
    if _IMPL_CLASS is None:
        # model-first: qsa.py's own `from . import model` tolerates a partial
        # model.py, but not the reverse — importing model.py resolves the cycle
        import vllm.models.qwen4_exp.nvidia.model  # noqa: F401

        from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAImpl

        class Qwen4ExpQSAMjolnirImpl(Qwen4ExpQSAImpl):
            """Stock QSA impl with the Mjolnir scratch-gather decode."""

            def __init__(self, layer, *, max_num_seqs: int = 256) -> None:
                # 0.34 slimmed the impl ctor to (heads, head_size, scale,
                # kv_heads, cache dtype); alibi/window/softcap et al moved
                # into the layer.
                super().__init__(
                    layer.num_heads,
                    layer.head_dim,
                    layer.scaling,
                    layer.num_kv_heads,
                    layer.kv_cache_dtype,
                )
                self._spec = qsa.IndexerSpec()
                self._max_num_seqs = max_num_seqs
                self._scales: tuple[float, float] | None = None
                self._scale_bufs: tuple[torch.Tensor, torch.Tensor] | None = None
                # E4: packed-fp6 (e2m3) main caches — the dense path's scheme
                # (mjolnir.kvfp6) on the QSA pools: write via kvfp6.write_kv,
                # decode/verify read the packed rows through the same
                # byte-blind gather, the t5 kernel's e2m3 path consumes the
                # scratch. Enabled by kv-cache-dtype fp6_e2m3 (the stock ctor
                # accept-list is widened by the qwen4-qsa-fp6 patch); the
                # MJOLNIR_QSA_FP6 env is the legacy alias.
                self._name = getattr(layer, "layer_name", None) or ""
                self._fp6: dict | None = None
                dtype = self.kv_cache_dtype
                if dtype == "fp6_e2m3" or (os.environ.get("MJOLNIR_QSA_FP6") and dtype in ("fp8", "fp8_e4m3")):
                    self._fp6 = self._fp6_state(layer)

            # ---- helpers ----------------------------------------------------

            def _split_cache(self, kv_cache: torch.Tensor):
                """The stock derivation: the pool (HND: (pages, h_kv, PS, 2d))
                -> K, V views as (pages, PS, h_kv, d). NOTE the result is a
                TRANSPOSE VIEW, not a contiguous (pages, PS, h_kv, d) tensor:
                its stride(page) is h_kv * PS * stride(row), so a flat-row
                index (page * PS + row) must NOT be multiplied by stride(row)
                alone — pages >= 1 would read the wrong bytes when h_kv > 1
                (the day-0 gather bug). The gather kernel takes the page
                stride explicitly and decomposes; scratch_copy demands a
                contiguous pool (tests/reference only)."""
                split_at = self.head_size if self._fp6 is None else self.head_size * 6 // 8
                key_cache, value_cache = kv_cache.transpose(1, 2).split(split_at, dim=-1)
                return key_cache, value_cache

            def _scale_bufs_for(self, layer, device: torch.device):
                pair = (float(layer._k_scale_float), float(layer._v_scale_float))
                if self._scales != pair or self._scale_bufs is None or self._scale_bufs[0].device != device:
                    self._scales = pair
                    self._scale_bufs = (
                        torch.tensor([pair[0]], dtype=torch.float32, device=device),
                        torch.tensor([pair[1]], dtype=torch.float32, device=device),
                    )
                return self._scale_bufs

            # ---- E4: the fp6 pair -------------------------------------------

            def _fp6_state(self, layer) -> dict:
                """Per-layer e2m3 scales, the dense builder's scheme verbatim
                (backend._build_fp6_state): codes carry descale 1, K's channel
                scales fold into Q and V's into the output; the calibration
                doc is the same file the dense layers read."""
                from mjolnir import kvscales

                path = kvscales.scales_path()
                if not path:
                    raise ValueError(
                        "QSA fp6_e2m3 needs calibrated scales: set MJOLNIR_ATTN_KV_SCALES (`mjolnir calibrate`)"
                    )
                doc = kvscales.load(path)
                if self._name and self._name not in doc:
                    raise ValueError(f"{path} has no scales for {self._name!r}")
                ks, vs = kvscales.e2m3_scales(doc, self._name, self.num_kv_heads, self.head_size)
                dev = layer._k_scale.device
                h_r = max(1, self.num_heads // self.num_kv_heads)
                state: dict = {}
                for side, rows in (("k", ks), ("v", vs)):
                    ch = torch.tensor(rows, dtype=torch.float32, device=dev).contiguous()
                    state[f"{side}_ch"] = ch
                    state[f"{side}_mul"] = ch.repeat_interleave(h_r, dim=0).contiguous()
                state["one"] = torch.ones(1, dtype=torch.float32, device=dev)
                return state

            def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
                """fp6: the packed write into the HND pool; else stock. Under
                calibration every written K/V folds into the layer's amax
                (the same store the dense layers observe)."""
                from mjolnir import kvscales

                if kvscales.calibrating() and self._name:
                    kvscales.observe(self._name, key, value, slot_mapping)
                if self._fp6 is None:
                    return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
                from mjolnir import kvfp6

                kvfp6.write_kv(
                    key,
                    value,
                    kv_cache,
                    slot_mapping,
                    self._fp6["k_ch"],
                    self._fp6["v_ch"],
                )
                return None

            def _selection_rows(self, layer, num_tokens: int):
                """Packed token-index buffer -> (T, <=W) int32 block ids (host),
                ONE ROW PER QUERY TOKEN, sorted-unique. Returns None when a row
                is inert (count 0).

                The buffer rows hold request-relative TOKEN indices in score
                order (-1 padded, trailing count column); the block grid is
                token // ratio, deduped WITHIN the row — decode and verify
                alike: a verify step's rows carry independent selections, and
                the scratch layout is ours, so any order works as long as
                src/dst agree."""
                topk = layer.topk_indices_buffer[:num_tokens]
                tok_host = topk[:, :-1].cpu()
                counts = topk[:, -1].cpu().tolist()
                tok_rows = tok_host.tolist()
                ratio = self._spec.ratio
                width = 0
                rows: list[list[int]] = []
                for row in range(num_tokens):
                    cnt = int(counts[row])
                    blocks = sorted({t // ratio for t in tok_rows[row][:cnt] if t >= 0})
                    if not blocks:
                        return None
                    rows.append(blocks)
                    width = max(width, len(blocks))
                sel = torch.full((num_tokens, max(1, width)), -1, dtype=torch.int32)
                for row, blocks in enumerate(rows):
                    sel[row, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
                return sel

            # ---- the override ------------------------------------------------

            def forward(
                self,
                layer,
                query: torch.Tensor,
                kv_cache: torch.Tensor,
                attn_metadata,
                output: torch.Tensor,
                *,
                token_to_req: torch.Tensor,
                use_prefill_config: bool,
                output_gate: torch.Tensor,
            ) -> torch.Tensor:
                num_tokens = attn_metadata.num_actual_tokens
                # 0.34 main QSA metadata is slim (no seq_lens): the per-request
                # lens live on the indexer's side metadata, reachable through
                # the forward context under the indexer cache's key - the same
                # lookup the owner layer does in _run_qsa.
                indexer_md = _indexer_metadata(layer)
                if indexer_md is None:
                    _note_fallback("indexer metadata unavailable")
                    return super().forward(
                        layer,
                        query,
                        kv_cache,
                        attn_metadata,
                        output,
                        token_to_req=token_to_req,
                        use_prefill_config=use_prefill_config,
                        output_gate=output_gate,
                    )
                num_reqs = int(indexer_md.seq_lens.shape[0])
                # Guards → the stock Triton path. Prefill/verify/mixed batches
                # have per-row selections the single scratch plan can't express;
                # bf16 caches are not a kernel format.
                reason = None
                # (0.34 folded the fused output quantization into the impl
                # itself via layer._k_scale_float - no per-call scale args.)
                if use_prefill_config:
                    reason = "prefill/mixed step"
                elif self.kv_cache_dtype not in ("fp8", "fp8_e4m3") and self._fp6 is None:
                    reason = f"cache dtype {self.kv_cache_dtype}"
                elif query.dtype != torch.bfloat16:
                    reason = f"query dtype {query.dtype}"
                elif num_tokens > self._max_num_seqs:
                    # verify rows are scratch sequences too (one per query
                    # row); the runner's seq slots bound them
                    reason = f"verify batch {num_tokens} rows > {self._max_num_seqs} seq slots"
                elif num_tokens == 0:
                    reason = "empty batch"
                if reason is not None:
                    _note_fallback(reason)
                    return super().forward(
                        layer,
                        query,
                        kv_cache,
                        attn_metadata,
                        output,
                        token_to_req=token_to_req,
                        use_prefill_config=use_prefill_config,
                        output_gate=output_gate,
                    )
                key_cache, value_cache = self._split_cache(kv_cache)
                if key_cache.ndim != 4:
                    _note_fallback(f"unexpected cache layout {tuple(kv_cache.shape)}")
                    return super().forward(
                        layer,
                        query,
                        kv_cache,
                        attn_metadata,
                        output,
                        token_to_req=token_to_req,
                        use_prefill_config=use_prefill_config,
                        output_gate=output_gate,
                    )
                if self._fp6 is None and self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
                    key_cache = key_cache.view(torch.float8_e4m3fn)
                    value_cache = value_cache.view(torch.float8_e4m3fn)
                page_size = key_cache.shape[1]
                # the POOL page size is free (the gather reads it through its
                # own byte strides) — only ratio-divisibility matters, so a
                # selected block never crosses a page. A GDN-hybrid stack
                # forces this to the mamba page size (800 on the fixture),
                # NOT a kernel size. The scratch stays at the kernel's 128.
                if page_size % self._spec.ratio:
                    _note_fallback(
                        f"pool page size {page_size} not a multiple of the ratio"
                        f" (kv_cache {tuple(kv_cache.shape)}, key {tuple(key_cache.shape)})"
                    )
                    return super().forward(
                        layer,
                        query,
                        kv_cache,
                        attn_metadata,
                        output,
                        token_to_req=token_to_req,
                        use_prefill_config=use_prefill_config,
                        output_gate=output_gate,
                    )
                dev = key_cache.device
                spec = self._spec
                spec.check_main_page_size(page_size)

                # ---- selection + lens (host, one sync per step; eager v1) ----
                # the selection indices are host-derived per step and would
                # bake into a captured graph — this path is eager-only by design
                if torch.cuda.is_current_stream_capturing():
                    _note_fallback("cuda graph capture (per-step selection indices)")
                    return super().forward(
                        layer,
                        query,
                        kv_cache,
                        attn_metadata,
                        output,
                        token_to_req=token_to_req,
                        use_prefill_config=use_prefill_config,
                        output_gate=output_gate,
                    )
                sel = self._selection_rows(layer, num_tokens)
                # per-row sequence lens, the stock kernel's own expansion
                # (qsa.py: seq_lens.index_select(0, token_to_req)): a verify
                # row attends with its request's len; the eff/plan/bt below
                # are per-row from here on, one scratch seq per query row.
                seq_lens_cpu = indexer_md.seq_lens.cpu()[token_to_req[:num_tokens].cpu().to(torch.int64)]
                if sel is None:
                    _note_fallback("empty selection row")
                    return super().forward(
                        layer,
                        query,
                        kv_cache,
                        attn_metadata,
                        output,
                        token_to_req=token_to_req,
                        use_prefill_config=use_prefill_config,
                        output_gate=output_gate,
                    )
                eff = qsa.effective_seq_lens(sel, seq_lens_cpu, spec)
                if int(eff.min()) <= 0:
                    _note_fallback("empty effective selection")
                    return super().forward(
                        layer,
                        query,
                        kv_cache,
                        attn_metadata,
                        output,
                        token_to_req=token_to_req,
                        use_prefill_config=use_prefill_config,
                        output_gate=output_gate,
                    )

                # ---- runner + scratch (shared across layers, grow-on-demand) --
                # the SCRATCH and the runner live at the kernel's page size
                # (128); the POOL keeps its own (800 on a GDN hybrid)
                kernel_ps = 128
                fp6 = self._fp6 is not None
                gkey = (*_geometry_key(layer, kernel_ps, dev), fp6)
                runner = _RUNNERS.get(gkey)
                if runner is None:
                    runner = _RUNNERS[gkey] = T5DecodeRunner(
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_size=self.head_size,
                        page_size=kernel_ps,
                        out_dtype=torch.bfloat16,
                        max_num_seqs=self._max_num_seqs,
                        max_q_len=1,
                        # e2m3 rides the fp8-Q path (runner invariant)
                        q_mode="fp8" if fp6 else "bf16",
                        kv_format="e2m3" if fp6 else "fp8",
                        device=dev,
                    )
                scratch = _SCRATCHES.get(gkey)
                pages_needed = int(((eff.to(torch.int64) + kernel_ps - 1) // kernel_ps).sum()) + _SCRATCH_SLACK_PAGES
                if scratch is None or scratch.shape[0] < pages_needed:
                    # grow to the observed need (never shrinks); row width
                    # follows the mode: d bytes fp8, 3d/4 packed fp6
                    scratch = _SCRATCHES[gkey] = torch.empty(
                        pages_needed,
                        kernel_ps,
                        self.num_kv_heads,
                        self.head_size if not fp6 else self.head_size * 6 // 8,
                        dtype=torch.uint8 if fp6 else torch.float8_e4m3fn,
                        device=dev,
                    )

                # ---- plan the scratch layout and the decode -------------------
                # pairing: a pure-decode batch has one query row per request in
                # request order (the V1 decode contract), so scratch sequence i
                # serves query row i — the same order eff/selection are built in.
                # rows this step actually touches: sum(ceil(eff_i / kernel_ps)) pages
                pages_used = pages_needed - _SCRATCH_SLACK_PAGES
                scratch_k = scratch[:pages_used]
                vkey = ("v", *gkey)
                scratch_v = _SCRATCHES.get(vkey)
                if scratch_v is None or scratch_v.shape[0] < pages_needed:
                    scratch_v = _SCRATCHES[vkey] = torch.empty(
                        pages_needed,
                        kernel_ps,
                        self.num_kv_heads,
                        self.head_size if not fp6 else self.head_size * 6 // 8,
                        dtype=torch.uint8 if fp6 else torch.float8_e4m3fn,
                        device=dev,
                    )
                scratch_v = scratch_v[:pages_used]

                bt_host = attn_metadata.block_table[:num_reqs].cpu()
                bt_rows = bt_host[token_to_req[:num_tokens].cpu().to(torch.int64)]
                src, dst, sbt, _pages_max = qsa.scratch_plan(
                    sel, seq_lens_cpu, bt_rows, spec, page_size, scratch_page_size=kernel_ps
                )
                src_dev, dst_dev = src.to(dev, non_blocking=True), dst.to(dev, non_blocking=True)
                eff_dev = eff.to(dev, non_blocking=True)
                sbt_dev = torch.zeros(num_tokens, sbt.shape[1], dtype=torch.int32, device=dev)
                sbt_dev[:, : sbt.shape[1]] = sbt.to(dev, non_blocking=True)

                if qsa_gather is not None and qsa_gather.available():
                    if not _ANNOUNCED[0]:
                        _ANNOUNCED[0] = True
                        print(
                            f"MJOLNIR QSA decode: scratch-gather active "
                            f"({'e2m3' if fp6 else 'fp8'}, q_len=1, PS={page_size}, "
                            f"pool {tuple(key_cache.shape)}, "
                            f"sel width {sel.shape[1]}, {num_tokens} rows)",
                            flush=True,
                        )
                    kb = key_cache.view(torch.uint8)
                    vb = value_cache.view(torch.uint8)
                    qsa_gather.gather_rows(
                        kb,
                        scratch_k.view(torch.uint8),
                        src_dev,
                        dst_dev,
                        eff_dev,
                        page_size,
                        kb.stride(0),
                        kb.stride(1),
                        kb.stride(2),
                    )
                    qsa_gather.gather_rows(
                        vb,
                        scratch_v.view(torch.uint8),
                        src_dev,
                        dst_dev,
                        eff_dev,
                        page_size,
                        vb.stride(0),
                        vb.stride(1),
                        vb.stride(2),
                    )
                else:  # pragma: no cover - no triton: scratch_copy would need a
                    _note_fallback("no triton gather; stock path")  # contiguous pool copy we refuse to materialize
                    return super().forward(
                        layer,
                        query,
                        kv_cache,
                        attn_metadata,
                        output,
                        token_to_req=token_to_req,
                        use_prefill_config=use_prefill_config,
                        output_gate=output_gate,
                    )

                plan = runner.plan(eff.tolist(), num_tokens, 1, sbt_dev, eff_dev, causal=False)
                runner.ensure_compiled(1, plan.with_reduction, causal=False)
                if fp6:
                    # the dense contract: codes carry descale 1; the channel
                    # scales fold into Q / the output (runner q_mul / o_mul)
                    st = self._fp6
                    runner.run(
                        plan,
                        query[:num_tokens],
                        output,
                        scratch_k,
                        scratch_v,
                        st["one"],
                        st["one"],
                        self.scale,
                        q_mul=st["k_mul"],
                        o_mul=st["v_mul"],
                    )
                else:
                    k_buf, v_buf = self._scale_bufs_for(layer, dev)
                    runner.run(
                        plan,
                        query[:num_tokens],
                        output,
                        scratch_k,
                        scratch_v,
                        k_buf,
                        v_buf,
                        self.scale,
                    )

                # the stock Triton kernel applies the sigmoid output gate; the
                # dense kernel does not — apply it here (fp32, like the kernel)
                g = output_gate[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
                output[:num_tokens].mul_(torch.sigmoid(g.to(torch.float32)).to(output.dtype))
                return output

        _IMPL_CLASS = Qwen4ExpQSAMjolnirImpl
    return _IMPL_CLASS
