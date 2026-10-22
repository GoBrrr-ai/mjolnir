"""FlashAttention 4 hd256 paged fp8 decode driver for ncu — the patched-FA4 baseline.

The FA4 hd256 path the ``NVFP4_FA4hd256`` serving config runs on the Mjolnir
image: ``vllm.vllm_flash_attn.flash_attn_varlen_func(..., fa_version=4)`` →
the CuTe-DSL ``BlackwellFusedMultiHeadAttentionForward`` kernel
(``sm100_hd256_2cta_fmha_forward.py``), 1CTA decode carve-out (``max_seqlen_q
<= 8``, paged-128 TMA — the only legal hd256 paged mode; the 1CTA patch).
One sequence of BENCH_L tokens, a single decode query, BENCH_ITERS launches,
no warmup. No SplitKV — the hd256 kernel does not support it (vLLM pins
``num_splits=1``).

Deviation vs the bf16-Q fi/fa2 legs: with fp8 KV the SM100/SM110 FA4 kernel
requires **Q in fp8 e4m3 too** (vLLM quantizes Q in the attention layer) and
writes bf16 output — the honest production path, with identity descales and
the same N(0, 0.5) data as the other legs.

Env: BENCH_L, BENCH_ITERS, BENCH_HQ (24), BENCH_HK (4), BENCH_D (256),
     BENCH_PAGE (128), BENCH_CONTIG=1, BENCH_SEED.
"""

from __future__ import annotations

import math
import os

import torch


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(f"BENCH_{name}", default))


def main():
    L, iters = env_int("L", 8192), env_int("ITERS", 50)
    hq, hk, d, ps = env_int("HQ", 24), env_int("HK", 4), env_int("D", 256), env_int("PAGE", 128)
    assert ps == 128, "the FA4 hd256 kernel only supports paged-128 TMA (tile_n=128)"
    torch.manual_seed(env_int("SEED", 1111))
    dev = "cuda"

    from vllm.vllm_flash_attn import flash_attn_varlen_func

    pages = (L + ps - 1) // ps
    max_seqlen_k = pages * ps  # hd256: max_seqlen_k padded to the page multiple
    k_cache = torch.empty(pages, ps, hk, d, dtype=torch.float8_e4m3fn, device=dev)
    v_cache = torch.empty_like(k_cache)
    step = max(1, (64 << 20) // (ps * hk * d))
    for p0 in range(0, pages, step):
        p1 = min(pages, p0 + step)
        k_cache[p0:p1] = (torch.randn(p1 - p0, ps, hk, d, device=dev, dtype=torch.bfloat16) * 0.5).to(
            torch.float8_e4m3fn
        )
        v_cache[p0:p1] = (torch.randn(p1 - p0, ps, hk, d, device=dev, dtype=torch.bfloat16) * 0.5).to(
            torch.float8_e4m3fn
        )
    if os.environ.get("BENCH_CONTIG", "0") == "1":
        table = torch.arange(pages, dtype=torch.int32, device=dev)
    else:
        table = torch.randperm(pages, device=dev).to(torch.int32)
    block_table = table.view(1, pages)
    q = (torch.randn(1, hq, d, device=dev, dtype=torch.bfloat16) * 0.5).to(torch.float8_e4m3fn)
    out = torch.empty(1, hq, d, device=dev, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    seqused_k = torch.tensor([L], dtype=torch.int32, device=dev)
    # identity descales (batch, num_kv_heads) — the same "scale 1.0" convention
    # as the fi driver's k_scale/v_scale=1.0
    descale = torch.ones(1, hk, dtype=torch.float32, device=dev)
    kw = {
        "q": q,
        "k": k_cache,
        "v": v_cache,
        "out": out,
        "cu_seqlens_q": cu_q,
        "max_seqlen_q": 1,
        "seqused_k": seqused_k,
        "max_seqlen_k": max_seqlen_k,
        "softmax_scale": 1.0 / math.sqrt(d),
        "causal": True,
        "window_size": [-1, -1],
        "block_table": block_table,
        "fa_version": 4,
        "q_descale": descale,
        "k_descale": descale,
        "v_descale": descale,
        "num_splits": 1,  # hd256: SplitKV not supported
    }

    torch.cuda.synchronize()
    print(
        f"FA4: L={L} pages={pages}x{ps} h_q={hq} h_k={hk} d={d} fp8 KV iters={iters}",
        flush=True,
    )
    flash_attn_varlen_func(**kw)  # CuTe-DSL JIT compile — host-side, like fi's plan()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        flash_attn_varlen_func(**kw)
    ev1.record()
    torch.cuda.synchronize()
    print(f"FA4: mean {ev0.elapsed_time(ev1) * 1e3 / max(1, iters):.1f} us/iter (not ncu)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
