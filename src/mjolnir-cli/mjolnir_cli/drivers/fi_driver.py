"""FlashInfer paged fp8 decode driver for ncu (NHD or HND KV layout).

The FlashInfer leg of ``mjolnir verify``: BENCH_B sequences (default 1) of
BENCH_L tokens, bf16 Q, fp8 e4m3 KV, tensor-core decode
(``use_tensor_cores=True``, plan once, split-KV as FlashInfer chooses) — the
decode path vLLM's FlashInfer backend runs. BENCH_QLEN > 1 (speculative
verification, e.g. 4 for MTP=3) runs the paged *prefill* wrapper, causal,
BENCH_QLEN query tokens per sequence — how stock vLLM's FlashInfer backend
runs a multi-token verify batch. BENCH_ITERS launches, no warmup, so ncu sees exactly
BENCH_ITERS invocations. The KV data is generated once in NHD order; the
HND run uses the same values transposed, so NHD vs HND differ only in layout.

    FI_LAYOUT=HND BENCH_L=65536 BENCH_ITERS=50 python3 fi_driver.py

Env: FI_LAYOUT (NHD|HND), BENCH_L, BENCH_ITERS, BENCH_B (1), BENCH_QLEN (1), BENCH_HQ (24), BENCH_HK (4),
     BENCH_D (256), BENCH_PAGE (128), BENCH_CONTIG=1 (pages in order; default
     is a random page permutation, as the Mjolnir harness), FI_CHECK=1 (also
     run the other layout once and assert the outputs match), BENCH_SEED.
     The pre-rename FI_L / FI_ITERS / FI_HQ / FI_HK / FI_D / FI_PAGE names
     are still honored.
"""

from __future__ import annotations

import os

import torch


def env_int(name: str, default: int) -> int:
    for k in (f"BENCH_{name}", f"FI_{name}"):
        if k in os.environ:
            return int(os.environ[k])
    return default


def main():
    import flashinfer

    layout = os.environ.get("FI_LAYOUT", "NHD").upper()
    assert layout in ("NHD", "HND"), layout
    L, iters = env_int("L", 8192), env_int("ITERS", 50)
    hq, hk, d, ps = env_int("HQ", 24), env_int("HK", 4), env_int("D", 256), env_int("PAGE", 128)
    b, qlen = env_int("B", 1), env_int("QLEN", 1)
    check = os.environ.get("FI_CHECK", "0") == "1"
    torch.manual_seed(env_int("SEED", 1111))
    dev = "cuda"

    pages_seq = (L + ps - 1) // ps
    last = L - (pages_seq - 1) * ps
    pages = b * pages_seq
    # (pages, 2, PS, h_k, d) in fp8, filled in chunks to bound the temp size
    nhd = torch.empty(pages, 2, ps, hk, d, dtype=torch.float8_e4m3fn, device=dev)
    step = max(1, (64 << 20) // (2 * ps * hk * d))
    for p0 in range(0, pages, step):
        p1 = min(pages, p0 + step)
        nhd[p0:p1] = (torch.randn(p1 - p0, 2, ps, hk, d, device=dev, dtype=torch.bfloat16) * 0.5).to(
            torch.float8_e4m3fn
        )
    caches = {"NHD": None, "HND": None}

    def cache(lay):
        if caches[lay] is None:
            caches[lay] = nhd if lay == "NHD" else nhd.transpose(2, 3).contiguous()
        return caches[lay]

    if os.environ.get("BENCH_CONTIG", os.environ.get("FI_CONTIG", "0")) == "1":
        indices = torch.arange(pages, dtype=torch.int32, device=dev)
    else:
        indices = torch.randperm(pages, device=dev).to(torch.int32)
    indptr = torch.arange(b + 1, dtype=torch.int32, device=dev) * pages_seq
    last_len = torch.full((b,), last, dtype=torch.int32, device=dev)
    q = torch.randn(b * qlen, hq, d, device=dev, dtype=torch.bfloat16)

    def make(lay):
        ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
        if qlen > 1:
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout=lay)
            qo_indptr = torch.arange(b + 1, dtype=torch.int32, device=dev) * qlen
            w.plan(
                qo_indptr,
                indptr,
                indices,
                last_len,
                hq,
                hk,
                d,
                ps,
                causal=True,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.float8_e4m3fn,
            )
            return w
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout=lay, use_tensor_cores=True)
        kw = {"pos_encoding_mode": "NONE", "q_data_type": torch.bfloat16, "kv_data_type": torch.float8_e4m3fn}
        try:
            w.plan(indptr, indices, last_len, hq, hk, d, ps, **kw)
        except TypeError:  # older FlashInfer: data_type instead of kv_data_type
            kw["data_type"] = kw.pop("kv_data_type")
            w.plan(indptr, indices, last_len, hq, hk, d, ps, **kw)
        return w

    def run(w, lay):
        return w.run(q, cache(lay), k_scale=1.0, v_scale=1.0)

    w = make(layout)
    kv = cache(layout)
    warmup = env_int("WARMUP", 0)  # wall runs warm up (JIT amortized); ncu keeps 0 exact launches
    for _ in range(warmup):
        run(w, layout)
    torch.cuda.synchronize()
    print(
        f"FI {layout}: b={b} q_len={qlen} L={L} pages={pages}x{ps} h_q={hq} h_k={hk} d={d} "
        f"cache {tuple(kv.shape)} stride {kv.stride()} iters={iters}",
        flush=True,
    )

    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    out = None
    for _ in range(iters):
        out = run(w, layout)
    ev1.record()
    torch.cuda.synchronize()
    print(f"FI {layout}: mean {ev0.elapsed_time(ev1) * 1e3 / max(1, iters):.1f} us/iter (not ncu)", flush=True)

    if check:
        other = "HND" if layout == "NHD" else "NHD"
        ref = run(make(other), other)
        err = (out.float() - ref.float()).abs().max().item()
        print(f"FI check {layout} vs {other}: max abs diff {err:.3e}")
        assert err < 1e-2, "NHD and HND outputs differ"
    print("DONE")


if __name__ == "__main__":
    main()
