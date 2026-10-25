"""Orin (sm_87) runner tests — the vLLM 0.32 serving contract, no vLLM needed.

The kernel's per-sequence ``kv_len`` is the device ``cu_seqlens_k``, so
these pin down where it comes from: the LIVE device ``seq_lens`` at launch
time, never the host plan. vLLM 0.32's V2 model runner relies on that in
three places the old host-staged cu got wrong:

* the planning lengths are an upper bound (``seq_lens_cpu_upper_bound``,
  + the fused-draft margin) — the kernel must not attend past the true end;
* the speculator replays its FULL CUDA graphs (the draft prefill and the
  draft decode steps) WITHOUT rebuilding the attention metadata — the
  graph was captured on dummy lengths and must read the current ones;
* the fused multi-step draft decode advances ``seq_lens`` in place between
  the steps of one metadata build.

Plus the family's other serving obligations: CUDA-graph padding rows
(seq_len 0) come out 0 — not NaN — after the split combine, the mma kernel
honors ``causal=False`` (the DSpark / DFlash drafter block), and the
backend-facing ``Runner`` picks the kernel and the split capacity.

    python3 tests/test_orin_runner.py        # in the image (verify vllm), or
    python -m pytest -q tests/kernel/test_orin_runner.py
"""

from __future__ import annotations

import math
import os
import sys

import torch


def _add_pkg_path():
    """The kernel package from an install, $MJOLNIR_SRC, or a parent checkout
    (the same lookup as test_runner.py, keyed on the orin family)."""
    import importlib.util

    marker = os.path.join("arch", "orin", "runner.py")
    spec = importlib.util.find_spec("mjolnir")
    if spec is not None and spec.origin and os.path.isfile(os.path.join(os.path.dirname(spec.origin), marker)):
        return
    env = os.environ.get("MJOLNIR_SRC")
    cands = [env] if env else []
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cands += [d, os.path.join(d, "src", "mjolnir")]
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "mjolnir", marker)):
            sys.path.insert(0, c)
            return
    raise SystemExit("the mjolnir kernel package was not found: set MJOLNIR_SRC to src/mjolnir")


_add_pkg_path()

from mjolnir.arch.orin import GEMVDecodeRunner, Runner, select_impl  # noqa: E402
from mjolnir.arch.orin.harness import make_case, ref_decode  # noqa: E402

H_Q, H_K, D, PS = 24, 4, 256, 16  # Qwen3.8-27B's attention: 24/4 heads, d 256
SCALE = 1.0 / math.sqrt(D)
TOL = 0.05  # the harness tolerance (fp8 KV, bf16 output)
IMPLS = ("mma", "gemv")


def _runner(impl: str, b: int, M: int, **kw) -> GEMVDecodeRunner:
    return GEMVDecodeRunner(
        num_heads=H_Q,
        num_kv_heads=H_K,
        head_size=D,
        page_size=PS,
        out_dtype=torch.bfloat16,
        max_num_seqs=b,
        max_q_len=M,
        q_mode="bf16",
        impl=impl,
        device="cuda",
        **kw,
    )


def _case(b: int, M: int, L: int, seed: int = 7):
    q, kv, table, seq_lens, _qd, kd, vd, _raw = make_case(
        b, M, L, H_Q, H_K, D, PS, torch.float8_e4m3fn, False, seed=seed, device="cuda"
    )
    return q, kv, table, seq_lens, kd, vd


def _ref(q, kv, table, lens, kd, vd, causal: bool = True) -> torch.Tensor:
    kv_c = kv.cpu()
    return ref_decode(q.float().cpu(), kv_c, kv_c, table.cpu(), lens.cpu(), None, kd.cpu(), vd.cpu(), SCALE, causal)


def _check(got: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    got = got.float().cpu()
    assert torch.isfinite(got).all(), f"{what}: non-finite output"
    rel = ((got - ref).norm() / ref.norm().clamp_min(1e-6)).item()
    assert rel < TOL, f"{what}: rel_err {rel:.4f} >= {TOL}"


def _run(runner, plan, q, out, kv, kd, vd):
    b, M = plan.num_seqs, plan.q_len
    runner.run(plan, q.view(b * M, H_Q, D), out, kv, kv, kd, vd, SCALE)


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
def test_kv_len_follows_the_live_seq_lens_not_the_plan():
    # the plan sees an optimistic upper bound + the fused-draft margin; the
    # device lengths change after plan(): the kernel must use the live ones
    for impl in IMPLS:
        b, M = 3, 1
        q, kv, table, seq_lens, kd, vd = _case(b, M, 3000, seed=11)
        live = seq_lens.clone()
        runner = _runner(impl, b, M)
        plan = runner.plan((live + 50).tolist(), b, M, table, live, margin=3)
        live[0] -= 100  # e.g. the draft-prefill rewind by the rejected tokens
        live[2] = M + 5  # a short request
        out = torch.empty(b * M, H_Q, D, dtype=torch.bfloat16, device="cuda")
        _run(runner, plan, q, out, kv, kd, vd)
        _check(out.view(b, M, H_Q, D), _ref(q, kv, table, live, kd, vd), f"{impl}: live seq_lens")


def test_graph_replay_reads_the_current_seq_lens():
    # the speculator's FULL graphs: captured on the dummy lengths vLLM's
    # make_dummy writes (num_tokens per request), replayed with the real
    # ones and without a metadata rebuild; then one fused draft step
    # advances the lengths in place and replays again
    for impl in IMPLS:
        b, M = 2, 1
        q, kv, table, seq_lens, kd, vd = _case(b, M, 2500, seed=5)
        runner = _runner(impl, b, M)
        seq_buf = torch.full((b,), M, dtype=torch.int32, device="cuda")  # the capture-time dummy
        q_buf = torch.zeros_like(q)
        out = torch.empty(b * M, H_Q, D, dtype=torch.bfloat16, device="cuda")
        plan = runner.plan(seq_buf.tolist(), b, M, table, seq_buf)
        _run(runner, plan, q_buf, out, kv, kd, vd)  # warm-up: compiles both kernels outside capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _run(runner, plan, q_buf, out, kv, kd, vd)
        q_buf.copy_(q)
        seq_buf.copy_(seq_lens)
        g.replay()
        torch.cuda.synchronize()
        _check(out.view(b, M, H_Q, D), _ref(q, kv, table, seq_lens, kd, vd), f"{impl}: replay")
        seq_buf += 1  # update_draft_inputs: the next draft step, same graph
        g.replay()
        torch.cuda.synchronize()
        _check(out.view(b, M, H_Q, D), _ref(q, kv, table, seq_lens + 1, kd, vd), f"{impl}: replay after advance")


def test_padding_rows_are_zero_not_nan():
    # CUDA-graph padding requests (seq_len 0): every split is empty, so the
    # combine sees lse = -inf everywhere — the output must be 0, not NaN
    for impl in IMPLS:
        for M in (1, 4):
            b = 2
            q, kv, table, seq_lens, kd, vd = _case(b, M, 1500, seed=3)
            seq_lens[1] = 0
            runner = _runner(impl, b, M)
            plan = runner.plan(seq_lens.tolist(), b, M, table, seq_lens)
            out = torch.full((b * M, H_Q, D), float("nan"), dtype=torch.bfloat16, device="cuda")
            _run(runner, plan, q, out, kv, kd, vd)
            o = out.view(b, M, H_Q, D)
            assert torch.isfinite(o[1]).all(), f"{impl} M={M}: NaN in a padding row"
            assert o[1].abs().max().item() == 0.0, f"{impl} M={M}: a padding row is not 0"
            ref = _ref(q[:1], kv, table[:1], seq_lens[:1], kd[:1], vd[:1])
            _check(o[:1], ref, f"{impl} M={M}: the real row beside the padding")


def test_mtp_verify_shape():
    # MTP K=3 verification: q_len 4, the per-token causal tail, varlen batch
    for impl in IMPLS:
        b, M = 4, 4
        q, kv, table, seq_lens, kd, vd = _case(b, M, 5000, seed=9)
        runner = _runner(impl, b, M)
        plan = runner.plan(seq_lens.tolist(), b, M, table, seq_lens)
        out = torch.empty(b * M, H_Q, D, dtype=torch.bfloat16, device="cuda")
        _run(runner, plan, q, out, kv, kd, vd)
        _check(out.view(b, M, H_Q, D), _ref(q, kv, table, seq_lens, kd, vd), f"{impl}: q_len 4 causal")


def test_non_causal_block():
    # a parallel drafter's block (DSpark / DFlash): every query token sees the
    # whole context — the mma kernel used to apply the causal tail regardless
    for impl in IMPLS:
        b, M = 2, 4
        q, kv, table, seq_lens, kd, vd = _case(b, M, 2000, seed=13)
        runner = _runner(impl, b, M)
        plan = runner.plan(seq_lens.tolist(), b, M, table, seq_lens, causal=False)
        assert plan.causal is False
        out = torch.empty(b * M, H_Q, D, dtype=torch.bfloat16, device="cuda")
        _run(runner, plan, q, out, kv, kd, vd)
        ref = _ref(q, kv, table, seq_lens, kd, vd, causal=False)
        _check(out.view(b, M, H_Q, D), ref, f"{impl}: non-causal")
        causal_ref = _ref(q, kv, table, seq_lens, kd, vd, causal=True)
        assert (ref - causal_ref).abs().max().item() > 1e-3, "the case must tell the two masks apart"


def test_backend_runner_picks_the_kernel_and_caps_the_splits():
    def backend_runner(kv_format: str, max_splits: int = 16) -> Runner:
        # the kwargs the vLLM backend passes (fp8 Q, MAX_SPLITS' default 16)
        return Runner(
            num_heads=H_Q,
            num_kv_heads=H_K,
            head_size=D,
            page_size=PS,
            out_dtype=torch.bfloat16,
            max_num_seqs=4,
            max_q_len=4,
            q_mode="fp8",
            max_splits=max_splits,
            kv_format=kv_format,
            device="cuda",
        )

    saved = os.environ.pop("MJOLNIR_ATTN_ORIN_IMPL", None)
    try:
        r = backend_runner("fp8")
        assert r.impl == "mma" and not r.fp8_q, "fp8 KV serves on the mma kernel, bf16 Q"
        auto = _runner("mma", 4, 4).ns_cap
        assert r.ns_cap == min(16, auto), f"MAX_SPLITS caps the auto capacity ({r.ns_cap} vs {auto})"
        assert backend_runner("fp8", max_splits=1).ns_cap == 1
        r6 = backend_runner("e2m3")
        assert r6.impl == "gemv" and r6.fp8_q, "fp6 KV stays on the GEMV (fp8 Q)"
        os.environ["MJOLNIR_ATTN_ORIN_IMPL"] = "gemv"
        assert backend_runner("fp8").impl == "gemv"
    finally:
        if saved is None:
            os.environ.pop("MJOLNIR_ATTN_ORIN_IMPL", None)
        else:
            os.environ["MJOLNIR_ATTN_ORIN_IMPL"] = saved
    assert select_impl(head_size=128, kv_format="fp8", max_q_len=4, requested="mma")[0] == "gemv"
    assert select_impl(head_size=256, kv_format="bf16", max_q_len=4, requested="mma")[0] == "gemv"
    assert select_impl(head_size=256, kv_format="fp8", max_q_len=8, requested="mma")[0] == "mma"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    only = sys.argv[1:]
    failed = 0
    for name, fn in tests:
        if only and not any(o in name for o in only):
            continue
        try:
            fn()
            print(f"  ✅ {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed" if not only else f"{failed} failed")
    sys.exit(1 if failed else 0)
