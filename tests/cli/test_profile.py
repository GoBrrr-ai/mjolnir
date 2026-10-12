"""verify profile: the Kineto-trace summary (graph replays by correlation id,
eager kernels, GPU idle, categories)."""

from __future__ import annotations

import gzip
import json

from mjolnir_cli.verify import profile


def _k(name, ts, dur, corr):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": {"correlation": corr}}


def _rt(name, ts, corr):
    return {"ph": "X", "cat": "cuda_runtime", "name": name, "ts": ts, "dur": 5, "args": {"correlation": corr}}


def test_parse_trace(tmp_path):
    ev = []
    t = 0
    for step in range(3):
        # verify graph: 3 kernels; draft graph: 2 kernels; one eager kernel
        g1, g2, e = 100 + step * 10, 101 + step * 10, 102 + step * 10
        ev += [_rt("cudaGraphLaunch", t, g1)]
        ev += [_k("mjolnir_t5_decode", t + 10, 200, g1), _k("nvjet_gemm_bf16", t + 220, 3000, g1)]
        ev += [_k("rms_norm_kernel", t + 3230, 50, g1)]
        ev += [_rt("cudaGraphLaunch", t + 3300, g2)]
        ev += [_k("gemv_lm_head", t + 3310, 1000, g2), _k("argmax_kernel", t + 4320, 20, g2)]
        ev += [_rt("cudaLaunchKernel", t + 4400, e), _k("copy_kernel", t + 4410, 10, e)]
        t += 5000  # 580 µs idle per step
    p = tmp_path / "w.pt.trace.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump({"traceEvents": ev}, f)
    s = profile.parse_trace(p, tokens=9)
    assert s["kernels"] == 18 and s["graph_replays"] == 6 and s["eager_kernels"] == 3
    big, small = s["graphs"]
    assert (big["kernels"], big["replays"], small["kernels"]) == (3, 3, 2)
    assert abs(big["gpu_ms"] - 3.25) < 1e-6 and abs(big["attn_ms"] - 0.2) < 1e-6
    assert s["eager_kernels_per_step"] == 1.0
    assert s["by_category_ms"]["gemm"] == 12.0  # 3 x (3 ms + 1 ms)
    assert 0 < s["idle_pct"] < 20
    assert s["top_kernels"][0][0] == "nvjet_gemm_bf16"


def test_categories():
    assert profile.categorize("void mjolnir::MixedInputFMHA") == "attention"
    assert profile.categorize("fused_recurrent_gated_delta_rule_fwd") == "linear-attn"
    assert profile.categorize("sm100_xmma_gemm_e2m1") == "gemm"
    assert profile.categorize("gumbel_sample_kernel") == "sampling"
