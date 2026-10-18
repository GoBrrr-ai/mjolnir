"""FlashAttention 2 paged decode driver for ncu — the stock-vLLM baseline.

What vLLM 0.32.0 runs on sm_110 out of the box for a head_dim=256 model:
``vllm.vllm_flash_attn.flash_attn_varlen_func(..., fa_version=2)`` over a
paged KV cache in NHD order ``(num_blocks, block_size, h_k, d)``. FA2 has no
fp8 KV path, so the cache is **bf16** — twice the bytes of the fp8 legs (the
GB/s column counts the real bytes). One sequence of BENCH_L tokens, a
single decode query, BENCH_ITERS launches, no warmup.

Env: BENCH_L, BENCH_ITERS, BENCH_HQ (24), BENCH_HK (4), BENCH_D (256),
     BENCH_PAGE (128), BENCH_CONTIG=1, BENCH_SEED, FA2_NUM_SPLITS (0 = FA2's
     own split heuristic, the vLLM default).
"""

from __future__ import annotations

import math
import os

import torch


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(f"BENCH_{name}", default))


def main():
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    L, iters = env_int("L", 8192), env_int("ITERS", 50)
    hq, hk, d, ps = env_int("HQ", 24), env_int("HK", 4), env_int("D", 256), env_int("PAGE", 128)
    splits = int(os.environ.get("FA2_NUM_SPLITS", "0"))
    torch.manual_seed(env_int("SEED", 1111))
    dev = "cuda"

    pages = (L + ps - 1) // ps
    k_cache = torch.empty(pages, ps, hk, d, dtype=torch.bfloat16, device=dev)
    v_cache = torch.empty_like(k_cache)
    step = max(1, (64 << 20) // (ps * hk * d * 2))
    for p0 in range(0, pages, step):
        p1 = min(pages, p0 + step)
        k_cache[p0:p1].normal_(0, 0.5)
        v_cache[p0:p1].normal_(0, 0.5)
    if os.environ.get("BENCH_CONTIG", "0") == "1":
        table = torch.arange(pages, dtype=torch.int32, device=dev)
    else:
        table = torch.randperm(pages, device=dev).to(torch.int32)
    block_table = table.view(1, pages)
    q = torch.randn(1, hq, d, device=dev, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    seqused_k = torch.tensor([L], dtype=torch.int32, device=dev)
    out = torch.empty_like(q)
    kw = {
        "q": q,
        "k": k_cache,
        "v": v_cache,
        "out": out,
        "cu_seqlens_q": cu_q,
        "max_seqlen_q": 1,
        "seqused_k": seqused_k,
        "max_seqlen_k": L,
        "softmax_scale": 1.0 / math.sqrt(d),
        "causal": True,
        "block_table": block_table,
        "fa_version": 2,
        "num_splits": splits,
    }

    def run():
        try:
            return flash_attn_varlen_func(**kw)
        except TypeError:  # older signature: no num_splits
            kw.pop("num_splits", None)
            return flash_attn_varlen_func(**kw)

    torch.cuda.synchronize()
    print(f"FA2: L={L} pages={pages}x{ps} h_q={hq} h_k={hk} d={d} bf16 KV iters={iters}", flush=True)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        run()
    ev1.record()
    torch.cuda.synchronize()
    print(f"FA2: mean {ev0.elapsed_time(ev1) * 1e3 / max(1, iters):.1f} us/iter (not ncu)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
