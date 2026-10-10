"""FlashInfer paged fp8 decode driver for ncu, NHD or HND KV layout.

Same problem as the T5 harness legs of ncu-t5-fp8-sweep.sh: one sequence of
FI_L tokens, bf16 Q, fp8 e4m3 KV, tensor-core decode (plan once, split-KV as
FlashInfer chooses), FI_ITERS launches and no warmup (so ncu sees exactly
FI_ITERS invocations). The KV data is generated once in NHD order; the HND
run uses the same values transposed, so NHD vs HND differ only in layout.

    FI_LAYOUT=HND FI_L=65536 FI_ITERS=50 python3 fi_layout_driver.py

Env: FI_LAYOUT (NHD|HND), FI_L, FI_ITERS, FI_HQ (24), FI_HK (4), FI_D (256),
     FI_PAGE (128, as the T5 harness), FI_CONTIG=1 (pages in order; default is
     a random page permutation, as the T5 harness), FI_CHECK=1 (also run the
     other layout once and assert the outputs match), FI_SEED.
"""

from __future__ import annotations

import os

import torch


def env_int(k, d):
    return int(os.environ.get(k, d))


def main():
    import flashinfer

    layout = os.environ.get("FI_LAYOUT", "NHD").upper()
    assert layout in ("NHD", "HND"), layout
    L, iters = env_int("FI_L", 8192), env_int("FI_ITERS", 50)
    hq, hk, d, ps = env_int("FI_HQ", 24), env_int("FI_HK", 4), env_int("FI_D", 256), env_int("FI_PAGE", 128)
    check = os.environ.get("FI_CHECK", "0") == "1"
    torch.manual_seed(env_int("FI_SEED", 1111))
    dev = "cuda"

    pages = (L + ps - 1) // ps
    last = L - (pages - 1) * ps
    # (pages, 2, PS, h_k, d) in fp8, filled page by page to bound the temp size
    nhd = torch.empty(pages, 2, ps, hk, d, dtype=torch.float8_e4m3fn, device=dev)
    step = max(1, (64 << 20) // (2 * ps * hk * d))
    for p0 in range(0, pages, step):
        p1 = min(pages, p0 + step)
        nhd[p0:p1] = (torch.randn(p1 - p0, 2, ps, hk, d, device=dev, dtype=torch.bfloat16) * 0.5
                      ).to(torch.float8_e4m3fn)
    caches = {"NHD": None, "HND": None}

    def cache(lay):
        if caches[lay] is None:
            caches[lay] = nhd if lay == "NHD" else nhd.transpose(2, 3).contiguous()
        return caches[lay]

    if os.environ.get("FI_CONTIG", "0") == "1":
        indices = torch.arange(pages, dtype=torch.int32, device=dev)
    else:
        indices = torch.randperm(pages, device=dev).to(torch.int32)
    indptr = torch.tensor([0, pages], dtype=torch.int32, device=dev)
    last_len = torch.tensor([last], dtype=torch.int32, device=dev)
    q = torch.randn(1, hq, d, device=dev, dtype=torch.bfloat16)
    k_scale = v_scale = 1.0

    def make(lay):
        ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout=lay, use_tensor_cores=True)
        kw = dict(pos_encoding_mode="NONE", q_data_type=torch.bfloat16,
                  kv_data_type=torch.float8_e4m3fn)
        try:
            w.plan(indptr, indices, last_len, hq, hk, d, ps, **kw)
        except TypeError:  # older FlashInfer: data_type instead of kv_data_type
            kw["data_type"] = kw.pop("kv_data_type")
            w.plan(indptr, indices, last_len, hq, hk, d, ps, **kw)
        return w

    def run(w, lay):
        return w.run(q, cache(lay), k_scale=k_scale, v_scale=v_scale)

    w = make(layout)
    kv = cache(layout)
    torch.cuda.synchronize()
    print(f"FI {layout}: L={L} pages={pages}x{ps} h_q={hq} h_k={hk} d={d} "
          f"cache {tuple(kv.shape)} stride {kv.stride()} iters={iters}", flush=True)

    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        out = run(w, layout)
    ev1.record()
    torch.cuda.synchronize()
    print(f"FI {layout}: mean {ev0.elapsed_time(ev1) * 1e3 / max(1, iters):.1f} us/iter "
          f"(warm-ish, not ncu)", flush=True)

    if check:
        other = "HND" if layout == "NHD" else "NHD"
        ref = run(make(other), other)
        err = (out.float() - ref.float()).abs().max().item()
        print(f"FI check {layout} vs {other}: max abs diff {err:.3e}")
        assert err < 1e-2, "NHD and HND outputs differ"
    print("DONE")


if __name__ == "__main__":
    main()
