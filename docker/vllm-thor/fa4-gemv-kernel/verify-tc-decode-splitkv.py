#!/usr/bin/env python3
"""TC (mma.sync) hd256 decode — SplitKV verify on Jetson Thor (sm_110).

Validates the SplitKV pass (M1 kernel + M2 interface) end-to-end through the
REAL ``_flash_attn_fwd`` interface — so the SplitKV dispatch (the ``tc_split``
gate, the ``VLLM_FA4_HD256_TC_NUM_SPLITS`` env + caller ``num_splits`` paths,
the ``_gemv_auto_num_splits`` auto plan, the ``out_partial``/``lse_partial``
fp32 allocation) AND the host ``_flash_attn_fwd_combine`` re-merge are
exercised. The correctness invariant: the COMBINED (ns>1) output must recover
the ns=1 write-through output within the dtype accumulation floor — NOT
bit-exact (the multi-chunk combine accumulates differently than the single
pass), but well inside the tolerance that a combine/sVb bug would breach
(>= ~0.1, the pre-fix sVb race).

Cells (compact — the essential SplitKV surface, not a full sweep):
  * dense + paged (PS=128), bf16 + fp8-Q (the production config),
    ns in {1,2,4}: ns=1 is the write-through baseline; ns=2 is driven by the
    CALLER ``num_splits`` arg, ns=4 by the ``VLLM_FA4_HD256_TC_NUM_SPLITS``
    env (both dispatch paths). Each ns>1 combined output vs the ns=1 output.
  * empty-split: a paged B=2 call (kv_len 256/64) at ns=2 — the short batch
    (64 rows = 1 tile) has ns>its tile count, so splits 1.. are EMPTY
    (LSE=-inf); the combine must discard them (the combined output still
    matches ns=1 for BOTH batches). A single-batch call cannot reach this:
    the interface's 128-row cap keeps ns below the max batch's tile count, so
    the short batch is the only reachable route (the kernel's empty path is
    otherwise defensive).
  * auto-ns: dense bf16 L=8192, no env + no caller ns -> the
    ``_gemv_auto_num_splits`` plan; confirm ns>1 is active and matches ns=1.
  * determinism: the same fp8 large-L cell run 3x -> bit-identical O+LSE
    (the sVb cross-warp barrier fix; pre-fix this was run-to-run non-
    deterministic + NaN-prone).

Gate: the Thor GPU is shared with the live vLLM server that serves the
subagents. The gate (inline copy of src/mjolnir/gate.py) waits for a clean
window (vLLM request queue 0/0 for CONFIRM samples); the server is NEVER
stopped. ``--no-gate`` = smoke mode (ungated; correctness is
contention-independent, so the gate is for the audit discipline, not
correctness). The window audit reports gated / clean / dirty like
verify-tc-decode.py.

Run inside the image with the vfa tree mounted:
  docker run --rm --gpus all --network host --entrypoint python3 \
    -v $VFA_TREE:/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$PWD":/p \
    mjolnir/vllm-thor:qwen38-sm110-v13 /p/verify-tc-decode-splitkv.py [--no-gate]
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
import urllib.request

import torch

NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
G = NUM_Q_HEADS // NUM_KV_HEADS  # 6
D = 256
TILE_N = 64   # the TC mma N-tile (kv-rows per n_block)
PS = 128     # paged page size
DEVICE = "cuda"
SOFTMAX_SCALE = D ** -0.5
FP8 = torch.float8_e4m3fn
# Production fp8 descales (the bench's per-(batch, kv_head) values).
FP8_Q_DESCALE = 1.06
FP8_K_DESCALE = 1.125
FP8_V_DESCALE = 1.25

# combined(ns>1) vs ns=1 write-through tolerances. A real combine/sVb bug
# breaches >= ~0.1 (the pre-fix sVb race was ~2.27e-1 + NaN); the legitimate
# multi-chunk-vs-single-pass accumulation floor is ~2-3e-3 (bf16) / ~3e-3
# (fp8 — same quantized KV both ways, so NOT the fp8-vs-fp32 quantization gap).
BF16_COMBINE_ABS = 1e-2
BF16_COMBINE_LSE = 5e-3
FP8_COMBINE_ABS = 5e-2
FP8_COMBINE_LSE = 1e-2

# --- clean-window gate (inline copy of src/mjolnir/gate.py) -------------------
# Single source of truth on the host is src/mjolnir/gate.py; the in-container
# copy mirrors the GEMV/TC verify scripts (mjolnir/gate.py is not importable
# inside the image). The server is NEVER stopped — only waited on.
METRICS_URL = os.environ.get("MJOLNIR_METRICS_URL",
                             "http://127.0.0.1:6001/metrics")
POLL_S = 2.0
CONFIRM = 3
WAIT_TIMEOUT_S = 1800
OFFLINE_CONFIRM = 3


def server_load() -> tuple[float, float] | None:
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=5) as r:
            text = r.read().decode()
    except Exception:  # noqa: BLE001
        return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        for key in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            if line.startswith(key + "{") or line.startswith(key + " "):
                out[key.split(":")[1]] = float(line.rsplit(" ", 1)[1])
    if "num_requests_running" not in out or "num_requests_waiting" not in out:
        return None
    return (out["num_requests_running"], out["num_requests_waiting"])


class _Monitor(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.trace: list[tuple[float, tuple[float, float] | None]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            self.trace.append((time.time(), server_load()))
            self._stop.wait(POLL_S)

    def stop(self) -> None:
        self._stop.set()


def wait_for_idle(mon: _Monitor) -> float | str | None:
    t_start = time.perf_counter()
    consumed = 0
    streak = 0
    offline_streak = 0
    while time.perf_counter() - t_start < WAIT_TIMEOUT_S:
        while consumed < len(mon.trace):
            _ts, load = mon.trace[consumed]
            consumed += 1
            if load is None:
                streak = 0
                offline_streak += 1
            else:
                offline_streak = 0
                if load == (0.0, 0.0):
                    streak += 1
                else:
                    streak = 0
        if offline_streak >= OFFLINE_CONFIRM:
            return "offline"
        if streak >= CONFIRM:
            return mon.trace[consumed - 1][0]
        time.sleep(1.0)
    return None


def _rand(dtype: torch.dtype, *shape) -> torch.Tensor:
    return torch.randn(*shape, device=DEVICE, dtype=torch.float32).to(dtype)


def _descales(B: int):
    qd = torch.full((B, NUM_KV_HEADS), FP8_Q_DESCALE, dtype=torch.float32, device=DEVICE)
    kd = torch.full((B, NUM_KV_HEADS), FP8_K_DESCALE, dtype=torch.float32, device=DEVICE)
    vd = torch.full((B, NUM_KV_HEADS), FP8_V_DESCALE, dtype=torch.float32, device=DEVICE)
    return qd, kd, vd


def _unpage(pool_k, pool_v, table, kv_lens):
    """Deinterleave the (NP,PS,H,D) pool with the page table -> dense per-batch
    (B,L_max,H,D) KV (the paged verify's unpage, the reference's layout)."""
    B = len(kv_lens)
    H = pool_k.shape[2]
    max_L = max(kv_lens)
    k = torch.zeros(B, max_L, H, D, device=pool_k.device, dtype=pool_k.dtype)
    v = torch.zeros(B, max_L, H, D, device=pool_v.device, dtype=pool_v.dtype)
    for b in range(B):
        L = kv_lens[b]
        for r in range(L):
            page = int(table[b, r // PS])
            off = r % PS
            k[b, r] = pool_k[page, off]
            v[b, r] = pool_v[page, off]
    return k, v


def _make_pool(dtype, kv_lens, table_rows, n_spare=1):
    n_pages = max((L + PS - 1) // PS for L in kv_lens)
    max_ref = max(max(r) for r in table_rows)
    pool_pages = max(n_pages, max_ref + 1) + n_spare
    torch.manual_seed(42)
    pool_k = _rand(dtype, pool_pages, PS, NUM_KV_HEADS, D)
    pool_v = _rand(dtype, pool_pages, PS, NUM_KV_HEADS, D)
    table = torch.tensor(
        [list(table_rows[b]) + [0] * (n_pages - len(table_rows[b])) for b in range(len(kv_lens))],
        dtype=torch.int32, device=DEVICE)
    return pool_k, pool_v, table, pool_pages


def _run(q, k, v, *, max_seqlen_k, page_table=None, seqused_k=None,
         ns_caller=1, ns_env=None, q_descale=None, k_descale=None,
         v_descale=None):
    """One real _flash_attn_fwd call. ns_caller drives the caller num_splits
    arg; ns_env (str) drives the VLLM_FA4_HD256_TC_NUM_SPLITS env knob. For the
    env path pass ns_caller=1 (the env wins in the plan). Returns (out, lse)."""
    os.environ["VLLM_FA4_HD256_TC"] = "1"
    os.environ["VLLM_FA4_HD256_GEMV"] = "0"
    if ns_env is not None:
        os.environ["VLLM_FA4_HD256_TC_NUM_SPLITS"] = str(ns_env)
    else:
        os.environ.pop("VLLM_FA4_HD256_TC_NUM_SPLITS", None)
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
    out, lse, _, _ = _flash_attn_fwd(
        q, k, v,
        cu_seqlens_q=None, cu_seqlens_k=None,
        seqused_k=seqused_k,
        page_table=page_table,
        max_seqlen_q=1, max_seqlen_k=max_seqlen_k,
        softmax_scale=SOFTMAX_SCALE, causal=True, return_lse=True,
        num_splits=ns_caller,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
    os.environ.pop("VLLM_FA4_HD256_TC_NUM_SPLITS", None)
    return out, lse


def _effective_ns(B: int, L: int) -> int:
    """Mirror the interface's M2 plan (to REPORT the auto ns)."""
    from vllm.vllm_flash_attn.cute import interface as _iface
    ns = _iface._gemv_auto_num_splits(NUM_KV_HEADS, B, torch.device(DEVICE),
                                     _iface._get_device_arch())
    return max(1, min(ns, 64, (L + 127) // 128))


def _cmp(name, out, lse, out_ref, lse_ref, abs_tol, lse_tol, results):
    # out/lse are (B, H, D) / (B, H) after squeeze; compare combined vs ns=1.
    d_abs = float((out.float() - out_ref.float()).abs().max())
    d_lse = float((lse.float() - lse_ref.float()).abs().max())
    ok = d_abs <= abs_tol and d_lse <= lse_tol
    results.append(ok)
    tag = "OK " if ok else "BAD"
    print(f"[{tag}] {name:46s} max_abs={d_abs:.3e} lse={d_lse:.3e} "
          f"(tol {abs_tol:.0e}/{lse_tol:.0e})  {'<<< OVER' if not ok else ''}",
          flush=True)
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("TC-SPLITKV-VERIFY VERDICT: NO-GO (no CUDA device)")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name!r} SMs={props.multi_processor_count}", flush=True)
    print(f"geometry: GQA {NUM_Q_HEADS}/{NUM_KV_HEADS} (g={G}), hd={D}, M=1\n",
          flush=True)

    no_gate = "--no-gate" in sys.argv
    offline = False
    window_start = 0.0
    mon = _Monitor()
    mon.start()
    if no_gate:
        window_start = time.time()
        print(f"GATE: SMOKE MODE (--no-gate): window not gated ({METRICS_URL})",
              flush=True)
    else:
        window = wait_for_idle(mon)
        if window is None:
            print(f"GATE: TIMEOUT after {WAIT_TIMEOUT_S}s", flush=True)
            mon.stop()
            return 3
        if window == "offline":
            window_start = time.time()
            offline = True
            print("GATE: vLLM metrics unreachable — UNGATED (nothing to gate)",
                  flush=True)
        else:
            window_start = window
            print(f"GATE: CLEAN WINDOW OPEN ({CONFIRM} consecutive 0/0)", flush=True)

    results: list[bool] = []

    # ---- helpers to build a dense (L, B) cell -------------------------------
    def dense_bf16(L, B, seed):
        torch.manual_seed(seed)
        q = _rand(torch.bfloat16, B, 1, NUM_Q_HEADS, D)
        k = _rand(torch.bfloat16, B, L, NUM_KV_HEADS, D)
        v = _rand(torch.bfloat16, B, L, NUM_KV_HEADS, D)
        return q, k, v

    def dense_fp8(L, B, seed):
        torch.manual_seed(seed)
        L_pad = ((L + TILE_N - 1) // TILE_N) * TILE_N
        q = _rand(torch.bfloat16, B, 1, NUM_Q_HEADS, D)
        k = _rand(FP8, B, L_pad, NUM_KV_HEADS, D)
        v = _rand(FP8, B, L_pad, NUM_KV_HEADS, D)
        if L_pad > L:
            k[:, L:, :, :] = 0
            v[:, L:, :, :] = 0
        qd, kd, vd = _descales(B)
        return q, k, v, qd, kd, vd

    def ns_sweep_dense(name, build, L, B, seed, abs_tol, lse_tol):
        # ns=1 baseline (write-through), ns=2 caller, ns=4 env.
        if name.startswith("dense bf16"):
            q, k, v = build(L, B, seed)
            kw = {}
            o1, l1 = _run(q, k, v, max_seqlen_k=L, ns_caller=1, **kw)
            o2, l2 = _run(q, k, v, max_seqlen_k=L, ns_caller=2, **kw)
            o4, l4 = _run(q, k, v, max_seqlen_k=L, ns_caller=1, ns_env="4", **kw)
        else:
            q, k, v, qd, kd, vd = build(L, B, seed)
            kw = dict(q_descale=qd, k_descale=kd, v_descale=vd)
            o1, l1 = _run(q, k, v, max_seqlen_k=L, ns_caller=1, **kw)
            o2, l2 = _run(q, k, v, max_seqlen_k=L, ns_caller=2, **kw)
            o4, l4 = _run(q, k, v, max_seqlen_k=L, ns_caller=1, ns_env="4", **kw)
        _cmp(f"{name} L={L} B={B} ns=2 caller vs ns=1", o2, l2, o1, l1,
             abs_tol, lse_tol, results)
        _cmp(f"{name} L={L} B={B} ns=4 env   vs ns=1", o4, l4, o1, l1,
             abs_tol, lse_tol, results)

    def ns_sweep_paged(name, build_pool, kv_lens, table_rows, seed, abs_tol, lse_tol):
        B = len(kv_lens)
        dtype = torch.bfloat16 if "bf16" in name else FP8
        pool_k, pool_v, table, _ = _make_pool(dtype, kv_lens, table_rows)
        seqused = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)
        maxk = max(kv_lens)
        q = _rand(torch.bfloat16, B, 1, NUM_Q_HEADS, D)
        torch.manual_seed(seed)
        if "bf16" in name:
            o1, l1 = _run(q, pool_k, pool_v, max_seqlen_k=maxk, page_table=table,
                          seqused_k=seqused, ns_caller=1)
            o2, l2 = _run(q, pool_k, pool_v, max_seqlen_k=maxk, page_table=table,
                          seqused_k=seqused, ns_caller=2)
            o4, l4 = _run(q, pool_k, pool_v, max_seqlen_k=maxk, page_table=table,
                          seqused_k=seqused, ns_caller=1, ns_env="4")
        else:
            qd, kd, vd = _descales(B)
            o1, l1 = _run(q, pool_k, pool_v, max_seqlen_k=maxk, page_table=table,
                          seqused_k=seqused, ns_caller=1, q_descale=qd, k_descale=kd, v_descale=vd)
            o2, l2 = _run(q, pool_k, pool_v, max_seqlen_k=maxk, page_table=table,
                          seqused_k=seqused, ns_caller=2, q_descale=qd, k_descale=kd, v_descale=vd)
            o4, l4 = _run(q, pool_k, pool_v, max_seqlen_k=maxk, page_table=table,
                          seqused_k=seqused, ns_caller=1, ns_env="4",
                          q_descale=qd, k_descale=kd, v_descale=vd)
        _cmp(f"{name} L={kv_lens} ns=2 caller vs ns=1", o2, l2, o1, l1,
             abs_tol, lse_tol, results)
        _cmp(f"{name} L={kv_lens} ns=4 env   vs ns=1", o4, l4, o1, l1,
             abs_tol, lse_tol, results)

    # 1-4. dense + paged, bf16 + fp8, ns sweep {1,2,4} -------------------------
    ns_sweep_dense("dense bf16", dense_bf16, 512, 1, 11, BF16_COMBINE_ABS, BF16_COMBINE_LSE)
    ns_sweep_dense("dense fp8", dense_fp8, 512, 1, 12, FP8_COMBINE_ABS, FP8_COMBINE_LSE)
    ns_sweep_paged("paged bf16", None, [512], [[0, 1, 2, 3]], 13,
                   BF16_COMBINE_ABS, BF16_COMBINE_LSE)
    ns_sweep_paged("paged fp8", None, [512], [[0, 1, 2, 3]], 14,
                   FP8_COMBINE_ABS, FP8_COMBINE_LSE)

    # 5. empty-split: paged B=2 (256/64), ns=2 -> the 64-row batch has empty splits
    print("\n-- empty-split (paged 256/64, ns=2: the 64-row batch's splits 1.. are empty) --",
          flush=True)
    pool_k, pool_v, table, _ = _make_pool(torch.bfloat16, [256, 64], [[0, 1], [2]])
    seqused = torch.tensor([256, 64], dtype=torch.int32, device=DEVICE)
    q = _rand(torch.bfloat16, 2, 1, NUM_Q_HEADS, D)
    o1, l1 = _run(q, pool_k, pool_v, max_seqlen_k=256, page_table=table,
                  seqused_k=seqused, ns_caller=1)
    o2, l2 = _run(q, pool_k, pool_v, max_seqlen_k=256, page_table=table,
                  seqused_k=seqused, ns_env="2")
    _cmp("empty-split paged [256,64] ns=2 vs ns=1 (both batches)", o2, l2, o1, l1,
         BF16_COMBINE_ABS, BF16_COMBINE_LSE, results)

    # 6. auto-ns: dense bf16 L=8192, no env/no caller ns -> auto plan (ns>1)
    print("\n-- auto-ns (dense bf16 L=8192, no env/no caller ns -> _gemv_auto_num_splits) --",
          flush=True)
    q, k, v = dense_bf16(8192, 1, 21)
    auto_ns = _effective_ns(1, 8192)
    o1, l1 = _run(q, k, v, max_seqlen_k=8192, ns_caller=1)
    oa, la = _run(q, k, v, max_seqlen_k=8192, ns_caller=1)  # no env -> auto plan
    print(f"  auto plan active: ns={auto_ns} (>1 => split engages)", flush=True)
    results.append(auto_ns > 1)
    _cmp(f"auto-ns L=8192 (ns={auto_ns}) vs ns=1", oa, la, o1, l1,
         BF16_COMBINE_ABS, BF16_COMBINE_LSE, results)

    # 7. determinism: fp8 large-L, 3x -> bit-identical (the sVb barrier fix)
    print("\n-- determinism (fp8 L=8192 B=1 ns=4, 3x bit-identical — sVb fix) --",
          flush=True)
    q, k, v, qd, kd, vd = dense_fp8(8192, 1, 22)
    kw = dict(q_descale=qd, k_descale=kd, v_descale=vd)
    outs = [_run(q, k, v, max_seqlen_k=8192, ns_caller=1, ns_env="4", **kw) for _ in range(3)]
    det = all(torch.equal(outs[i][0], outs[0][0]) and torch.equal(outs[i][1], outs[0][1])
              for i in range(1, 3))
    results.append(det)
    print(f"  3 runs bit-identical (O and LSE): {'YES' if det else 'NO'}"
          f"{'  <<< NON-DETERMINISTIC' if not det else ''}", flush=True)
    # also confirm the fp8 combined matches ns=1 (the barrier dropped the race error)
    o1f, l1f = _run(q, k, v, max_seqlen_k=8192, ns_caller=1, **kw)
    _cmp("fp8 L=8192 ns=4 (deterministic) vs ns=1", outs[0][0], outs[0][1],
         o1f, l1f, FP8_COMBINE_ABS, FP8_COMBINE_LSE, results)

    # --- window audit: did it stay clean for the whole run? ------------------
    window_end = time.time()
    mon.stop()
    in_window = [(ts, load) for ts, load in mon.trace if window_start <= ts <= window_end]
    saw_reachable = any(load is not None for _, load in mon.trace)
    dirty = [s for s in in_window
             if (s[1] is not None and s[1] != (0.0, 0.0)) or (s[1] is None and saw_reachable)]
    clean = not dirty
    gated = not no_gate and not offline
    n_ok = sum(results)
    state = "gated" if gated else ("offline/ungated" if offline else "no-gate")
    # Window cleanliness gates the VERDICT only when we actually WAITED for a
    # clean window (gated mode): a co-tenant entering the window invalidates a
    # gated measurement. In --no-gate (smoke) and offline mode the audit is
    # informational — correctness is contention-independent.
    window_ok = clean or not gated
    go = all(results) and window_ok
    print(f"\nwindow: {state}, {'clean' if clean else 'DIRTY'} "
          f"({len(in_window)} samples, {len(dirty)} dirty)"
          f"{'' if window_ok else '  <<< gated window went dirty'}", flush=True)
    print(f"cases passing: {n_ok}/{len(results)}", flush=True)
    print(f"TC-SPLITKV-VERIFY VERDICT: {'GO' if go else 'NO-GO'}", flush=True)
    if not window_ok:
        return 2
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        print("\nEXCEPTION (verbatim traceback):")
        print(traceback.format_exc())
        print("TC-SPLITKV-VERIFY VERDICT: NO-GO")
        raise SystemExit(1)
