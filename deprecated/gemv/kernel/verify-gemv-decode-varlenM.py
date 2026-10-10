#!/usr/bin/env python3
"""GEMV hd256 decode — **varlen-M (M<=8)** verify on Jetson Thor (sm_110).

Validates the varlen-M extension of the GEMV decode kernel
(``sm100_hd256_decode_gemv``): the MTP-verify step (M query tokens, M<=8) runs
through the GEMV path with an **in-thread M-loop** (per-M register sizing,
``seqlen_q`` is a compile-time ``Constexpr``). M>1 is **dense-Q only**
(``cu_seqlens_q is None``), mirroring the TC varlen-M gate; each of the M tokens
does **full attention over the shared KV** (the causal verify where the M
tokens are appended to the L context, so each sees the whole past context).

Gold reference: for each (batch, token, head), softmax over the shared L KV
rows (no cross-token causal mask) — ``out[b,m,h] = softmax(q[b,m,h] @ K[b,:L,hkv]) @ V[b,:L,hkv]``,
``lse[b,h,m] = logsumexp(...)`` natural units.

Cells:
  * M sweep {1,2,4,8} x {dense, varlen-k} x {bf16, fp8-KV}: ns=1 (write-through)
    vs the gold reference — isolates the M correctness from the SplitKV combine.
  * multi-batch varlen-k at M=4 (distinct per-batch kv_len).
  * M=8 split: dense {bf16, fp8} at ns=4 (the SplitKV combine over the M
    write-through rows — the M dim is orthogonal to the split chunk).
  * gate reject: M=9 and M=16 with ``VLLM_FA4_HD256_GEMV=1`` — the GEMV gate must
    REJECT (no new ``gemv_compile_cache`` entry => the GEMV binary is not
    invoked; the call falls through to the FA4/FlashInfer path, finite output).
  * M=1 regression spot: a few M=1 cells (dense + varlen-k, bf16 + fp8, L=8192)
    — the full M=1 regression is the existing ``verify-gemv-decode.py`` (35),
    ``verify-gemv-decode-paged.py`` (8), ``verify-gemv-decode-splitkv.py`` (13);
    this spot confirms the M=1 codegen is unchanged (``seqlen_q``=1 Constexpr ->
    the M*VEC == VEC register tensors, byte-identical to before).

Gate: the Thor GPU is shared with the live vLLM server. The clean-window gate
(inline copy of src/mjolnir/gate.py) waits for a clean window; ``--no-gate`` =
smoke mode (correctness is contention-independent). The server is NEVER stopped.

Run inside the image with the vfa tree mounted:
  docker run --rm --gpus all --network host --entrypoint python3 \
    -v $VFA_TREE:/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$PWD":/p \
    narandill/mjolnir-thor:v14 /p/verify-gemv-decode-varlenM.py [--no-gate]
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
DEVICE = "cuda"
SOFTMAX_SCALE = D**-0.5
FP8 = torch.float8_e4m3fn
FP8_Q_DESCALE = 1.06
FP8_K_DESCALE = 1.125
FP8_V_DESCALE = 1.25

# Tolerances (match verify-gemv-decode.py): the M tokens are independent rows
# (no cross-token accumulation), so the per-(token, head) O/LSE error is the
# same floor as M=1.
MAX_ABS_TOL = 5e-2
MAX_REL_TOL = 5e-2
LSE_TOL = 2e-3
FP8_MAX_ABS_TOL = 5e-2
FP8_MAX_REL_TOL = 5e-2

# --- clean-window gate (inline copy of src/mjolnir/gate.py) -------------------
METRICS_URL = os.environ.get("MJOLNIR_METRICS_URL", "http://127.0.0.1:6001/metrics")
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


def _descales(B: int, fp8: bool):
    if not fp8:
        return None, None, None
    qd = torch.full((B, NUM_KV_HEADS), FP8_Q_DESCALE, dtype=torch.float32, device=DEVICE)
    kd = torch.full((B, NUM_KV_HEADS), FP8_K_DESCALE, dtype=torch.float32, device=DEVICE)
    vd = torch.full((B, NUM_KV_HEADS), FP8_V_DESCALE, dtype=torch.float32, device=DEVICE)
    return qd, kd, vd


def _ref(q, kv_k, kv_v, kv_lens, cu_k, fp8, qd, kd, vd):
    """Gold varlen-M reference. q: (B, M, H, D) dense-Q. kv_k/kv_v are either
    dense (B, L_max, Hkv, D) with cu_k=None, or the packed (total_k, Hkv, D)
    with cu_k set. Each of the M tokens attends its batch's L KV rows (full
    attention; no cross-token causal mask). Returns (out (B,M,H,D), lse (B,H,M)).
    For fp8 the reference re-casts q bf16->fp8 exactly as the interface pre-casts
    it, and dequantizes k/v with the per-(batch, kv_head) descales."""
    B, M = q.shape[0], q.shape[1]
    out = torch.empty(B, M, NUM_Q_HEADS, D, dtype=torch.float32, device=DEVICE)
    lse = torch.empty(B, NUM_Q_HEADS, M, dtype=torch.float32, device=DEVICE)
    for b in range(B):
        L = kv_lens[b]
        if cu_k is None:
            kb = kv_k[b, :L].float()
            vb = kv_v[b, :L].float()
        else:
            kb = kv_k[cu_k[b] : cu_k[b + 1], :L].float()
            vb = kv_v[cu_k[b] : cu_k[b + 1], :L].float()
        for m in range(M):
            for h in range(NUM_Q_HEADS):
                hkv = h // G
                if fp8:
                    # q stays bf16 (the GEMV kernel reads q bf16 even on the
                    # fp8-KV leg — only K/V are fp8, dequantized by the descales);
                    # pre-scale q by qd, k/v by their descales (matching the
                    # kernel's qk_descale=qd*kd on the score + vd on the PV).
                    q8 = q[b, m, h].float() * qd[b, hkv]
                    k8 = kb[:, hkv] * kd[b, hkv]
                    v8 = vb[:, hkv] * vd[b, hkv]
                else:
                    q8 = q[b, m, h].float()
                    k8 = kb[:, hkv]
                    v8 = vb[:, hkv]
                s = (q8 @ k8.T) * SOFTMAX_SCALE
                p = torch.softmax(s, dim=-1)
                out[b, m, h] = p @ v8
                lse[b, h, m] = torch.logsumexp(s, dim=-1)
    return out, lse


def _run(
    q, k, v, *, max_seqlen_k, max_seqlen_q, cu_k=None, q_descale=None, k_descale=None, v_descale=None, ns_env=None
):
    """One real _flash_attn_fwd call (dense-Q: cu_seqlens_q is always None here).
    ns_env (str) forces the split count via VLLM_FA4_HD256_GEMV_NUM_SPLITS
    (ns=1 = write-through); None = the interface's auto plan. Returns
    (out (B,M,H,D), lse (B,H,M))."""
    os.environ["VLLM_FA4_HD256_GEMV"] = "1"
    os.environ["VLLM_FA4_HD256_TC_SPARK"] = "0"
    if ns_env is not None:
        os.environ["VLLM_FA4_HD256_GEMV_NUM_SPLITS"] = str(ns_env)
    else:
        os.environ.pop("VLLM_FA4_HD256_GEMV_NUM_SPLITS", None)
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    out, lse, _, _ = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        return_lse=True,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    os.environ.pop("VLLM_FA4_HD256_GEMV_NUM_SPLITS", None)
    return out, lse


def _cmp(name, out, lse, ref, ref_lse, results, fp8=False):
    # out (B,M,H,D) vs ref (B,M,H,D); lse (B,H,M) vs ref_lse (B,H,M) directly.
    abs_tol = FP8_MAX_ABS_TOL if fp8 else MAX_ABS_TOL
    lse_tol = LSE_TOL
    d_abs = float((out.float() - ref).abs().max())
    ref_max = float(ref.abs().max())
    rel = d_abs / max(ref_max, 1e-9)
    d_lse = float((lse - ref_lse).abs().max())
    ok = d_abs <= abs_tol and d_lse <= lse_tol
    results.append(ok)
    tag = "OK " if ok else "BAD"
    over = "" if ok else "  <<< OVER"
    print(
        f"[{tag}] {name:44s} max_abs={d_abs:.4e} rel={rel:.4e} lse={d_lse:.4e} (tol {abs_tol:.0e}/{lse_tol:.0e}){over}",
        flush=True,
    )
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("GEMV-VARLENM-VERIFY VERDICT: NO-GO (no CUDA device)")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name!r} SMs={props.multi_processor_count}", flush=True)
    print(
        f"geometry: GQA {NUM_Q_HEADS}/{NUM_KV_HEADS} (g={G}), hd={D}, varlen-M (in-thread M-loop, M<=8)\n", flush=True
    )

    no_gate = "--no-gate" in sys.argv
    offline = False
    window_start = 0.0
    mon = _Monitor()
    mon.start()
    if no_gate:
        window_start = time.time()
        print(f"GATE: SMOKE MODE (--no-gate): window not gated ({METRICS_URL})", flush=True)
    else:
        window = wait_for_idle(mon)
        if window is None:
            print(f"GATE: TIMEOUT after {WAIT_TIMEOUT_S}s", flush=True)
            mon.stop()
            return 3
        if window == "offline":
            window_start = time.time()
            offline = True
            print("GATE: vLLM metrics unreachable — UNGATED (nothing to gate)", flush=True)
        else:
            window_start = window
            print(f"GATE: CLEAN WINDOW OPEN ({CONFIRM} consecutive 0/0)", flush=True)

    results: list[bool] = []

    # ---- 1. M sweep: dense + varlen-k, bf16 + fp8, ns=1 (write-through) ------
    def dense_cell(M, L, fp8, seed):
        B = 1
        torch.manual_seed(seed)
        q = _rand(torch.bfloat16, B, M, NUM_Q_HEADS, D)
        kv_dtype = FP8 if fp8 else torch.bfloat16
        k = _rand(kv_dtype, B, L, NUM_KV_HEADS, D)
        v = _rand(kv_dtype, B, L, NUM_KV_HEADS, D)
        qd, kd, vd = _descales(B, fp8)
        out, lse = _run(q, k, v, max_seqlen_k=L, max_seqlen_q=M, q_descale=qd, k_descale=kd, v_descale=vd, ns_env="1")
        ref, ref_lse = _ref(q, k, v, [L] * B, None, fp8, qd, kd, vd)
        return _cmp(f"dense {'fp8' if fp8 else 'bf16'} M={M} L={L} ns=1", out, lse, ref, ref_lse, results, fp8)

    def varlen_k_cell(M, kv_lens, fp8, seed):
        B = len(kv_lens)
        cu_k = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(kv_lens), 0).tolist()), dtype=torch.int32, device=DEVICE
        )
        total_k = int(cu_k[-1].item())
        torch.manual_seed(seed)
        kv_dtype = FP8 if fp8 else torch.bfloat16
        k = _rand(kv_dtype, total_k, NUM_KV_HEADS, D)
        v = _rand(kv_dtype, total_k, NUM_KV_HEADS, D)
        q = _rand(torch.bfloat16, B, M, NUM_Q_HEADS, D)
        qd, kd, vd = _descales(B, fp8)
        out, lse = _run(
            q,
            k,
            v,
            max_seqlen_k=max(kv_lens),
            max_seqlen_q=M,
            cu_k=cu_k,
            q_descale=qd,
            k_descale=kd,
            v_descale=vd,
            ns_env="1",
        )
        ref, ref_lse = _ref(q, k, v, kv_lens, cu_k, fp8, qd, kd, vd)
        return _cmp(f"varlenk {'fp8' if fp8 else 'bf16'} M={M} L={kv_lens} ns=1", out, lse, ref, ref_lse, results, fp8)

    print("-- M sweep (ns=1 write-through): dense + varlen-k, bf16 + fp8 --", flush=True)
    for M in (1, 2, 4, 8):
        dense_cell(M, 512, fp8=False, seed=200 + M)
        dense_cell(M, 512, fp8=True, seed=300 + M)
        varlen_k_cell(M, [512], fp8=False, seed=400 + M)
        varlen_k_cell(M, [512], fp8=True, seed=500 + M)

    # a multi-batch varlen-k cell at M=4 (distinct kv_len per batch)
    varlen_k_cell(4, [512, 97], fp8=False, seed=604)

    # ---- 2. M=8 split (ns=4): the SplitKV combine over the M rows -----------
    print("\n-- M=8 split (ns=4): dense bf16 + fp8 --", flush=True)
    for fp8, seed in ((False, 701), (True, 702)):
        M, L, B = 8, 8192, 1
        torch.manual_seed(seed)
        q = _rand(torch.bfloat16, B, M, NUM_Q_HEADS, D)
        kv_dtype = FP8 if fp8 else torch.bfloat16
        k = _rand(kv_dtype, B, L, NUM_KV_HEADS, D)
        v = _rand(kv_dtype, B, L, NUM_KV_HEADS, D)
        qd, kd, vd = _descales(B, fp8)
        out, lse = _run(q, k, v, max_seqlen_k=L, max_seqlen_q=M, q_descale=qd, k_descale=kd, v_descale=vd, ns_env="4")
        ref, ref_lse = _ref(q, k, v, [L] * B, None, fp8, qd, kd, vd)
        _cmp(f"dense {'fp8' if fp8 else 'bf16'} M=8 L=8192 ns=4", out, lse, ref, ref_lse, results, fp8)

    # ---- 3. gate reject: M=9 / M=16 must NOT hit the GEMV kernel -------------
    print("\n-- gate reject (M=9/16: no new gemv_compile_cache entry) --", flush=True)
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    if getattr(_flash_attn_fwd, "gemv_compile_cache", None) is None:
        _flash_attn_fwd.gemv_compile_cache = {}
    cache = _flash_attn_fwd.gemv_compile_cache

    def gate_rejected(M, seed):
        keys_before = set(cache.keys())
        L = 512
        torch.manual_seed(seed)
        q = _rand(torch.bfloat16, 1, M, NUM_Q_HEADS, D)
        k = _rand(torch.bfloat16, 1, L, NUM_KV_HEADS, D)
        v = _rand(torch.bfloat16, 1, L, NUM_KV_HEADS, D)
        out, _lse, _x, _y = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=M,
            max_seqlen_k=L,
            softmax_scale=SOFTMAX_SCALE,
            causal=True,
            return_lse=True,
        )
        keys_after = set(cache.keys())
        hit = len(keys_after - keys_before) > 0
        finite = bool(torch.isfinite(out).all())
        results.append((not hit) and finite)
        print(
            f"  M={M:2d}: GEMV-invoked={hit} output_finite={finite} "
            f"{'<<< GEMV GATE ADMISSION (should reject)' if hit else 'rejected (OK)'}",
            flush=True,
        )
        return (not hit) and finite

    gate_rejected(9, 909)
    gate_rejected(16, 916)

    # ---- 4. M=1 regression spot (the full 35/8/13 regression is separate) ----
    print("\n-- M=1 regression spot (dense + varlen-k, bf16 + fp8, L=8192) --", flush=True)
    dense_cell(1, 8192, fp8=False, seed=801)
    dense_cell(1, 8192, fp8=True, seed=802)
    varlen_k_cell(1, [8192], fp8=False, seed=803)
    varlen_k_cell(1, [8192], fp8=True, seed=804)

    # --- window audit ---
    window_end = time.time()
    mon.stop()
    in_window = [(ts, load) for ts, load in mon.trace if window_start <= ts <= window_end]
    saw_reachable = any(load is not None for _, load in mon.trace)
    dirty = [s for s in in_window if (s[1] is not None and s[1] != (0.0, 0.0)) or (s[1] is None and saw_reachable)]
    clean = not dirty
    gated = not no_gate and not offline
    n_ok = sum(results)
    state = "gated" if gated else ("offline/ungated" if offline else "no-gate")
    window_ok = clean or not gated
    go = all(results) and window_ok
    print(
        f"\nwindow: {state}, {'clean' if clean else 'DIRTY'} "
        f"({len(in_window)} samples, {len(dirty)} dirty)"
        f"{'' if window_ok else '  <<< gated window went dirty'}",
        flush=True,
    )
    print(f"cases passing: {n_ok}/{len(results)}", flush=True)
    print(f"GEMV-VARLENM-VERIFY VERDICT: {'GO' if go else 'NO-GO'}", flush=True)
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
        print("GEMV-VARLENM-VERIFY VERDICT: NO-GO")
        raise SystemExit(1)
