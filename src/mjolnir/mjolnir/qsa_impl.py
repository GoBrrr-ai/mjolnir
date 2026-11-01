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
impl and only overrides ``forward_qsa``.
"""

from __future__ import annotations

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

        from vllm.model_executor.layers.attention.attention import AttentionType
        from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionImpl

        class Qwen4ExpQSAMjolnirImpl(Qwen4ExpQSAFlashAttentionImpl):
            """Stock QSA impl with the Mjolnir scratch-gather decode."""

            def __init__(self, layer, *, max_num_seqs: int = 256) -> None:
                super().__init__(
                    layer.num_heads,
                    layer.head_dim,
                    layer.scaling,
                    layer.num_kv_heads,
                    None,  # alibi
                    None,  # sliding window
                    layer.kv_cache_dtype,
                    None,  # logits soft cap
                    AttentionType.DECODER,
                    None,  # kv sharing
                )
                self._spec = qsa.IndexerSpec()
                self._max_num_seqs = max_num_seqs
                self._scales: tuple[float, float] | None = None
                self._scale_bufs: tuple[torch.Tensor, torch.Tensor] | None = None

            # ---- helpers ----------------------------------------------------

            def _split_cache(self, kv_cache: torch.Tensor):
                """The stock derivation: (2, pages, PS, h_kv, 2d)-ish -> K, V
                as (pages, PS, h_kv, d). Whatever the upstream layout, this is
                the exact expression the stock Triton kernel consumes."""
                key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
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

            def _selection_blocks(self, layer, num_tokens: int, num_reqs: int, token_to_req: torch.Tensor):
                """Packed token-index buffer -> (b, <=W) int32 block ids (host),
                one row per request, sorted-unique. Returns None when a row is
                inert (count 0) — that cannot happen in a decode batch.

                The buffer rows hold request-relative TOKEN indices in score
                order (-1 padded, trailing count column); the block grid is
                token // ratio, deduped and sorted — the scratch layout is
                ours, so any order works as long as src/dst agree."""
                topk = layer.topk_indices_buffer[:num_tokens]
                tok_host = topk[:, :-1].cpu()
                counts = topk[:, -1].cpu().tolist()
                t2r = token_to_req[:num_tokens].cpu().tolist()
                tok_rows = tok_host.tolist()
                ratio = self._spec.ratio
                width = 0
                per_req: dict[int, list[int]] = {}
                for row in range(num_tokens):
                    cnt = int(counts[row])
                    blocks = sorted({t // ratio for t in tok_rows[row][:cnt] if t >= 0})
                    if not blocks:
                        return None
                    per_req[int(t2r[row])] = blocks
                    width = max(width, len(blocks))
                if len(per_req) != num_reqs:
                    return None  # a request scheduled without a selection row
                sel = torch.full((num_reqs, max(1, width)), -1, dtype=torch.int32)
                for req, blocks in per_req.items():
                    sel[req, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
                return sel

            # ---- the override ------------------------------------------------

            def forward_qsa(
                self,
                layer,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                kv_cache: torch.Tensor,
                attn_metadata,
                output: torch.Tensor,
                token_to_req: torch.Tensor,
                use_prefill_config: bool,
                output_gate: torch.Tensor,
                output_scale=None,
                output_block_scale=None,
            ) -> torch.Tensor:
                num_tokens = attn_metadata.num_actual_tokens
                # len(seq_lens) is the request count (the decode/prefill
                # counters are QSA-builder-specific and unset here)
                num_reqs = int(attn_metadata.seq_lens.shape[0])
                # Guards → the stock Triton path. Prefill/verify/mixed batches
                # have per-row selections the single scratch plan can't express;
                # bf16 caches are not a kernel format.
                reason = None
                if output_scale is not None or output_block_scale is not None:
                    reason = "fused output quantization"
                elif use_prefill_config:
                    reason = "prefill/mixed step"
                elif self.kv_cache_dtype not in ("fp8", "fp8_e4m3"):
                    reason = f"cache dtype {self.kv_cache_dtype}"
                elif query.dtype != torch.bfloat16:
                    reason = f"query dtype {query.dtype}"
                elif num_tokens != num_reqs:
                    reason = "verify/mixed batch (per-row selections)"
                elif num_tokens == 0:
                    reason = "empty batch"
                if reason is not None:
                    _note_fallback(reason)
                    return super().forward_qsa(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                        token_to_req,
                        use_prefill_config,
                        output_gate,
                        output_scale,
                        output_block_scale,
                    )
                key_cache, value_cache = self._split_cache(kv_cache)
                if key_cache.ndim != 4:
                    _note_fallback(f"unexpected cache layout {tuple(kv_cache.shape)}")
                    return super().forward_qsa(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                        token_to_req,
                        use_prefill_config,
                        output_gate,
                    )
                if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
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
                    return super().forward_qsa(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                        token_to_req,
                        use_prefill_config,
                        output_gate,
                    )
                dev = key_cache.device
                spec = self._spec
                spec.check_main_page_size(page_size)

                # ---- selection + lens (host, one sync per step; eager v1) ----
                # the selection indices are host-derived per step and would
                # bake into a captured graph — this path is eager-only by design
                if torch.cuda.is_current_stream_capturing():
                    _note_fallback("cuda graph capture (per-step selection indices)")
                    return super().forward_qsa(
                        layer, query, key, value, kv_cache, attn_metadata, output, token_to_req, use_prefill_config
                    )
                sel = self._selection_blocks(layer, num_tokens, num_reqs, token_to_req)
                seq_lens_cpu = attn_metadata.seq_lens.cpu()
                if sel is None:
                    _note_fallback("empty selection row")
                    return super().forward_qsa(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                        token_to_req,
                        use_prefill_config,
                        output_gate,
                    )
                eff = qsa.effective_seq_lens(sel, seq_lens_cpu, spec)
                if int(eff.min()) <= 0:
                    _note_fallback("empty effective selection")
                    return super().forward_qsa(
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                        token_to_req,
                        use_prefill_config,
                        output_gate,
                    )

                # ---- runner + scratch (shared across layers, grow-on-demand) --
                # the SCRATCH and the runner live at the kernel's page size
                # (128); the POOL keeps its own (800 on a GDN hybrid)
                kernel_ps = 128
                gkey = _geometry_key(layer, kernel_ps, dev)
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
                        q_mode="bf16",
                        kv_format="fp8",
                        device=dev,
                    )
                scratch = _SCRATCHES.get(gkey)
                pages_needed = int(((eff.to(torch.int64) + kernel_ps - 1) // kernel_ps).sum()) + _SCRATCH_SLACK_PAGES
                if scratch is None or scratch.shape[0] < pages_needed:
                    # grow to the observed need (never shrinks); fp8 rows
                    scratch = _SCRATCHES[gkey] = torch.empty(
                        pages_needed,
                        kernel_ps,
                        self.num_kv_heads,
                        self.head_size,
                        dtype=torch.float8_e4m3fn,
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
                        self.head_size,
                        dtype=torch.float8_e4m3fn,
                        device=dev,
                    )
                scratch_v = scratch_v[:pages_used]

                bt_host = attn_metadata.block_table[:num_reqs].cpu()
                src, dst, sbt, _pages_max = qsa.scratch_plan(
                    sel, seq_lens_cpu, bt_host, spec, page_size, scratch_page_size=kernel_ps
                )
                src_dev, dst_dev = src.to(dev, non_blocking=True), dst.to(dev, non_blocking=True)
                eff_dev = eff.to(dev, non_blocking=True)
                sbt_dev = torch.zeros(num_reqs, sbt.shape[1], dtype=torch.int32, device=dev)
                sbt_dev[:, : sbt.shape[1]] = sbt.to(dev, non_blocking=True)

                if qsa_gather is not None and qsa_gather.available():
                    if not _ANNOUNCED[0]:
                        _ANNOUNCED[0] = True
                        print(
                            f"MJOLNIR QSA decode: scratch-gather active "
                            f"(fp8, q_len=1, PS={page_size}, pool {tuple(key_cache.shape)}, "
                            f"sel width {sel.shape[1]}, {num_reqs} reqs)",
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
                    return super().forward_qsa(
                        layer, query, key, value, kv_cache, attn_metadata, output, token_to_req, use_prefill_config
                    )

                k_buf, v_buf = self._scale_bufs_for(layer, dev)
                plan = runner.plan(eff.tolist(), num_reqs, 1, sbt_dev, eff_dev, causal=False)
                runner.ensure_compiled(1, plan.with_reduction, causal=False)
                runner.run(
                    plan,
                    query[:num_tokens],
                    output,
                    scratch_k,
                    scratch_v,
                    k_buf,
                    v_buf,
                    self.head_size**-0.5,
                )

                # the stock Triton kernel applies the sigmoid output gate; the
                # dense kernel does not — apply it here (fp32, like the kernel)
                g = output_gate[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
                output[:num_tokens].mul_(torch.sigmoid(g.to(torch.float32)).to(output.dtype))
                return output

        _IMPL_CLASS = Qwen4ExpQSAMjolnirImpl
    return _IMPL_CLASS
