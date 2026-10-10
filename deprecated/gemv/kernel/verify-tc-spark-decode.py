#!/usr/bin/env python3
"""TC (mma.sync) hd256 decode — verify on Jetson Thor (sm_110).

Verifies the tensor-core decode kernel
(``vllm.vllm_flash_attn.cute.sm100_hd256_decode_tc_spark``) against an fp32 reference:
- the bf16 Q/KV leg (no descales) — the regression guard (dense + varlen);
- the fp8 Q-bf16/KV-e4m3 leg: the kernel's fp8 mma (MmaFP8Op) streams fp8 KV
  and descales fold in exactly as in the GEMV sibling (qk_descale into the
  score scale, v_descale in the epilogue). The reference dequantizes the
  actual fp8 tensors (q is cast bf16->fp8 in-kernel, so the reference casts
  it identically) and runs fp32 attention on the dequantized values.
Scope: dense + varlen, num_splits=1.

Run inside the image with the vfa tree mounted:
  docker run --rm --gpus all --network host --entrypoint python3 \
    -v $VFA_TREE:/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$PWD":/p \
    mjolnir/vllm-thor:qwen38-sm110-v13 /p/verify-tc-decode.py
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
TILE_N = 64  # matches BlackwellHd256DecodeTCSpark.TILE_N (kv-rows per mma n_block)
DEVICE = "cuda"
SOFTMAX_SCALE = D ** -0.5

MAX_ABS_TOL = 5e-2
MAX_REL_TOL = 5e-2  # relative to ref_max
# bf16 online-softmax: the LSE (log-sum-exp) carries ~1e-2 in the multi-block merge
# (the mma S/P are bf16), so a 2e-3 tol is tighter than the kernel's own precision.
LSE_TOL = 2e-2
# fp8 leg: the P (softmax weights) operand is cast to e4m3 for the PV mma —
# ~6% relative on the P values dominates the O error (the QK scores stay
# exact: fp8 mma on the dequantized fp8 values vs the same values in fp32 is
# an accumulation-order difference only). 5e-2 max_abs is loose enough for
# the P-cast tail but tight enough to catch a descale/logic error (those
# shift O by ~the descale mismatch, >= ~0.1).
FP8_MAX_ABS_TOL = 5e-2
FP8_MAX_REL_TOL = 5e-2
# Non-unity per-(batch, kv_head) descales (the bench's production values).
FP8_Q_DESCALE = 1.06
FP8_K_DESCALE = 1.125
FP8_V_DESCALE = 1.25
FP8 = torch.float8_e4m3fn

# --- clean-window gate (inline copy of src/mjolnir/gate.py) -------------------
# The Thor GPU is shared with the live vLLM server that serves the subagents.
# This verify runs inside the image (mjolnir/gate.py is not importable here), so
# it carries the same gate logic inline as the GEMV bench scripts (the single
# source of truth on the host is src/mjolnir/gate.py; these inline copies are
# required for in-container runs). The window is clean when the vLLM request
# queue (running+waiting) reads (0,0) for CONFIRM consecutive samples. The
# server is NEVER stopped — the gate only waits for a clean window to open.
METRICS_URL = os.environ.get("MJOLNIR_METRICS_URL",
                            "http://127.0.0.1:6001/metrics")
POLL_S = 2.0
CONFIRM = 3
WAIT_TIMEOUT_S = 1800
OFFLINE_CONFIRM = 3  # consecutive unreachable samples => server offline


def server_load() -> tuple[float, float] | None:
    """(num_running, num_waiting) from the co-located vLLM server, or None."""
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
    """Window-start ts; ``"offline"`` if the endpoint stays unreachable (the
    co-located server is down -> nothing to gate against); ``None`` on timeout."""
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


def _fp32_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_lens: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 M=1 GQA reference (bf16, no descale). q: (B,1,H,D) or (total_q,H,D),
    k/v: (B,L_max,H_kv,D) with per-batch length kv_lens[b]. Returns (out, lse)
    with out fp32 and lse: (B,H) fp32 natural units."""
    varlen = q.dim() == 3
    total_q = q.shape[0] if varlen else q.shape[0]
    out = torch.empty(total_q, NUM_Q_HEADS, D, dtype=torch.float32, device=DEVICE)
    lse = torch.empty(total_q, NUM_Q_HEADS, dtype=torch.float32, device=DEVICE)
    qrows = 0
    for b in range(len(kv_lens)):
        L = kv_lens[b]
        if varlen:
            q_b = q[qrows:qrows + 1]  # (1, H, D)
        else:
            q_b = q[b:b + 1]
        if L == 0:
            out[qrows:qrows + 1] = 0.0
            lse[qrows:qrows + 1] = float("-inf")
            qrows += 1
            continue
        if varlen:
            k_b = k[kv_offset[b]:kv_offset[b] + L]
            v_b = v[kv_offset[b]:kv_offset[b] + L]
        else:
            k_b = k[b, :L]
            v_b = v[b, :L]
        qs = q_b[0].squeeze(0).float()  # (H, D) (dense keeps a size-1 seqlen dim)
        ks = k_b.float()  # (L, Hkv, D)
        vs = v_b.float()
        for h in range(NUM_Q_HEADS):
            hkv = h // G
            s = (qs[h] @ ks[:, hkv].T) * SOFTMAX_SCALE  # (L,) natural units
            p = torch.softmax(s, dim=-1)
            out[qrows, h] = p @ vs[:, hkv]
            lse[qrows, h] = torch.logsumexp(s, dim=-1)
        qrows += 1
    return out, lse


def _fp32_ref_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qd: torch.Tensor,
    kd: torch.Tensor,
    vd: torch.Tensor,
    kv_lens: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 M=1 GQA reference for the fp8 KV leg (dequantized values).

    q: (B,1,H,D) bf16 — cast to e4m3 to mirror the kernel's in-reg bf16->fp8
    Q cast, then dequantized with qd. k/v: (B,L_max,H_kv,D) fp8, dequantized
    with kd/vd. qd/kd/vd: (B,H_kv) fp32. The math mirrors the kernel's
    descale fold exactly: S = (q8*qd) @ (k8*kd).T * scale, O = softmax(S) @
    (v8*vd), LSE = logsumexp(S). (The P weights the kernel casts to e4m3 for
    the PV mma stay fp32 here — that ~6%-relative P-cast error is the fp8
    leg's dominant diff, within the fp8 tolerances.)
    """
    B = q.shape[0]
    out = torch.empty(B, NUM_Q_HEADS, D, dtype=torch.float32, device=DEVICE)
    lse = torch.empty(B, NUM_Q_HEADS, dtype=torch.float32, device=DEVICE)
    for b in range(B):
        L = kv_lens[b]
        k_b = k[b, :L]
        v_b = v[b, :L]
        for h in range(NUM_Q_HEADS):
            hkv = h // G
            q8 = q[b, 0, h].to(FP8).float() * qd[b, hkv]
            k8 = k_b[:, hkv].float() * kd[b, hkv]
            v8 = v_b[:, hkv].float() * vd[b, hkv]
            s = (q8 @ k8.T) * SOFTMAX_SCALE  # (L,) natural units
            p = torch.softmax(s, dim=-1)
            out[b, h] = p @ v8
            lse[b, h] = torch.logsumexp(s, dim=-1)
    return out, lse


kv_offset: list[int] = []


def _run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor | None,
    cu_k: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    return_lse: bool = True,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    os.environ["VLLM_FA4_HD256_TC_SPARK"] = "1"
    os.environ["VLLM_FA4_HD256_GEMV"] = "0"
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    out, lse, _, _ = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        return_lse=return_lse,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    return out, lse


def _report_row(name: str, ok: bool, max_abs: float, rel: float,
                lse_err: float | None, tol: float = MAX_ABS_TOL,
                tol_rel: float = MAX_REL_TOL) -> None:
    tag = "OK " if ok else "BAD"
    lse_s = f" lse_err={lse_err:.4e}" if lse_err is not None else ""
    print(f"[{tag}] {name:38s} max_abs={max_abs:.4e} rel={rel:.4e}"
          f" (tol {tol:.0e}/{tol_rel:.0e}){lse_s}", flush=True)


def main() -> int:
    if not torch.cuda.is_available():
        print("TC-VERIFY VERDICT: NO-GO (no CUDA device)")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name!r} SMs={props.multi_processor_count}")
    print(f"geometry: GQA {NUM_Q_HEADS}/{NUM_KV_HEADS} (g={G}), hd={D}, M=1\n")
    print("legs: dense bf16 (regression) + dense fp8 KV + descales "
          f"({FP8_Q_DESCALE}/{FP8_K_DESCALE}/{FP8_V_DESCALE}) + varlen bf16\n")

    # --- clean-window gate ----------------------------------------------------
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
            print(f"GATE: TIMEOUT waiting for {CONFIRM} consecutive 0/0 samples "
                  f"after {WAIT_TIMEOUT_S}s", flush=True)
            mon.stop()
            return 3
        if window == "offline":
            window_start = time.time()
            offline = True
            print(f"GATE: vLLM metrics unreachable ({METRICS_URL}) — server "
                  f"offline; running UNGATED (nothing co-located to gate against)",
                  flush=True)
        else:
            window_start = window
            print(f"GATE: CLEAN WINDOW OPEN ({CONFIRM} consecutive 0/0 samples)",
                  flush=True)

    results: list[bool] = []

    # ---------------- dense bf16 cases --------------------------------------
    for L in (1, 3, 5, 64, 256, 8192, 32768, 65536):
        for batch in (1, 4):
            torch.manual_seed(777 + L * 13 + batch)
            # Pad the KV to a whole tile: the TC V load is un-predicated, so a
            # partial block reads gmem rows beyond kv_len (finite pool data in
            # production). An exactly sized KV would be true OOB -> pad to >= TILE_N.
            L_pad = ((L + TILE_N - 1) // TILE_N) * TILE_N
            q = _rand(torch.bfloat16, batch, 1, NUM_Q_HEADS, D)
            k = _rand(torch.bfloat16, batch, L_pad, NUM_KV_HEADS, D)
            v = _rand(torch.bfloat16, batch, L_pad, NUM_KV_HEADS, D)
            if L_pad > L:
                k[:, L:, :, :] = 0
                v[:, L:, :, :] = 0
            out, lse = _run(
                q, k, v, cu_q=None, cu_k=None,
                max_seqlen_q=1, max_seqlen_k=L,
            )
            out = out.squeeze(1)  # dense: (B, 1, H, D) -> (B, H, D) (L_q == 1 for TC decode)
            lse = lse.squeeze(-1)  # dense: (B, H, 1) -> (B, H)
            ref, ref_lse = _fp32_ref(q, k, v, [L] * batch)
            d_abs = float((out.float() - ref).abs().max())
            ref_max = float(ref.abs().max())
            rel = d_abs / ref_max
            lse_err = float((lse - ref_lse).abs().max()) if L > 0 else 0.0
            ok = d_abs <= MAX_ABS_TOL and rel <= MAX_REL_TOL and lse_err <= LSE_TOL
            results.append(ok)
            _report_row(f"dense bf16 L={L} B={batch}", ok, d_abs, rel, lse_err)

    # ---------------- dense fp16 (dtype parity) --------------------------------
    # O is fp16 (out_torch_dtype = q_dtype when not fp8). The mma is MmaF16BF16Op
    # (fp16) via DTYPE=mO.element_type=fp16. Small-L (deterministic, no co-tenancy).
    for L in (64, 256):
        for batch in (1, 4):
            torch.manual_seed(777 + L * 13 + batch + 500)
            L_pad = ((L + TILE_N - 1) // TILE_N) * TILE_N
            q = _rand(torch.float16, batch, 1, NUM_Q_HEADS, D)
            k = _rand(torch.float16, batch, L_pad, NUM_KV_HEADS, D)
            v = _rand(torch.float16, batch, L_pad, NUM_KV_HEADS, D)
            if L_pad > L:
                k[:, L:, :, :] = 0
                v[:, L:, :, :] = 0
            out, lse = _run(
                q, k, v, cu_q=None, cu_k=None,
                max_seqlen_q=1, max_seqlen_k=L,
            )
            out = out.squeeze(1)
            lse = lse.squeeze(-1)
            ref, ref_lse = _fp32_ref(q, k, v, [L] * batch)
            d_abs = float((out.float() - ref).abs().max())
            ref_max = float(ref.abs().max())
            rel = d_abs / ref_max
            lse_err = float((lse - ref_lse).abs().max())
            ok = d_abs <= MAX_ABS_TOL and rel <= MAX_REL_TOL and lse_err <= LSE_TOL
            results.append(ok)
            _report_row(f"dense fp16 L={L} B={batch}", ok, d_abs, rel, lse_err)

    # ---------------- dense fp8 KV + non-unity descales ----------------------
    # L>=64 only: the partial-tail OOB mask is a known TC limitation (served by
    # GEMV in production). The fp8 mma halves the KV bytes this leg streams.
    for L in (64, 256, 8192, 32768, 65536):
        for batch in (1, 4):
            torch.manual_seed(777 + L * 13 + batch + 100)
            L_pad = ((L + TILE_N - 1) // TILE_N) * TILE_N
            q = _rand(torch.bfloat16, batch, 1, NUM_Q_HEADS, D)
            k = _rand(FP8, batch, L_pad, NUM_KV_HEADS, D)
            v = _rand(FP8, batch, L_pad, NUM_KV_HEADS, D)
            if L_pad > L:
                k[:, L:, :, :] = 0
                v[:, L:, :, :] = 0
            qd = torch.full((batch, NUM_KV_HEADS), FP8_Q_DESCALE,
                            dtype=torch.float32, device=DEVICE)
            kd = torch.full((batch, NUM_KV_HEADS), FP8_K_DESCALE,
                            dtype=torch.float32, device=DEVICE)
            vd = torch.full((batch, NUM_KV_HEADS), FP8_V_DESCALE,
                            dtype=torch.float32, device=DEVICE)
            out, lse = _run(
                q, k, v, cu_q=None, cu_k=None,
                max_seqlen_q=1, max_seqlen_k=L,
                q_descale=qd, k_descale=kd, v_descale=vd,
            )
            out = out.squeeze(1)
            lse = lse.squeeze(-1)
            ref, ref_lse = _fp32_ref_fp8(q, k, v, qd, kd, vd, [L] * batch)
            d_abs = float((out.float() - ref).abs().max())
            ref_max = float(ref.abs().max())
            rel = d_abs / ref_max
            lse_err = float((lse - ref_lse).abs().max())
            ok = (d_abs <= FP8_MAX_ABS_TOL and rel <= FP8_MAX_REL_TOL
                  and lse_err <= LSE_TOL)
            results.append(ok)
            _report_row(f"dense fp8 L={L} B={batch}", ok, d_abs, rel, lse_err,
                        FP8_MAX_ABS_TOL, FP8_MAX_REL_TOL)

    # ---------------- dense fp8-Q (native fp8 Q; dtype parity) --------------
    # Q natively fp8 (no pre-cast — the interface's q.to(fp8) is a no-op). O is
    # bf16 (is_fp8 => out_torch_dtype=bf16). Same mma as the fp8-KV leg
    # (MmaFP8Op); the only difference is the Q arrives already fp8. Small-L.
    for L in (64, 256):
        for batch in (1, 4):
            torch.manual_seed(777 + L * 13 + batch + 1000)
            L_pad = ((L + TILE_N - 1) // TILE_N) * TILE_N
            q = _rand(FP8, batch, 1, NUM_Q_HEADS, D)  # native fp8-Q
            k = _rand(FP8, batch, L_pad, NUM_KV_HEADS, D)
            v = _rand(FP8, batch, L_pad, NUM_KV_HEADS, D)
            if L_pad > L:
                k[:, L:, :, :] = 0
                v[:, L:, :, :] = 0
            qd = torch.full((batch, NUM_KV_HEADS), 1.0, dtype=torch.float32, device=DEVICE)
            kd = torch.full((batch, NUM_KV_HEADS), 1.0, dtype=torch.float32, device=DEVICE)
            vd = torch.full((batch, NUM_KV_HEADS), 1.0, dtype=torch.float32, device=DEVICE)
            out, lse = _run(
                q, k, v, cu_q=None, cu_k=None,
                max_seqlen_q=1, max_seqlen_k=L,
                q_descale=qd, k_descale=kd, v_descale=vd,
            )
            out = out.squeeze(1)
            lse = lse.squeeze(-1)
            ref, ref_lse = _fp32_ref_fp8(q, k, v, qd, kd, vd, [L] * batch)
            d_abs = float((out.float() - ref).abs().max())
            ref_max = float(ref.abs().max())
            rel = d_abs / ref_max
            lse_err = float((lse - ref_lse).abs().max())
            ok = (d_abs <= FP8_MAX_ABS_TOL and rel <= FP8_MAX_REL_TOL
                  and lse_err <= LSE_TOL)
            results.append(ok)
            _report_row(f"dense fp8-Q L={L} B={batch}", ok, d_abs, rel, lse_err,
                        FP8_MAX_ABS_TOL, FP8_MAX_REL_TOL)

    # ---------------- varlen bf16 (incl. empty batch) -----------------------
    torch.manual_seed(4242)
    lens = [5, 1, 0]
    batch = len(lens)
    total_k = sum(lens)
    global kv_offset
    kv_offset = [0, 5, 6]
    q = _rand(torch.bfloat16, batch, NUM_Q_HEADS, D)  # (total_q=3, H, D)
    # Pad the flat KV pool to a whole tile beyond total_k: the TC V load is
    # un-predicated, so each batch's 64-row tile reads past its own (possibly
    # short/empty) span into the next batch's rows or OOB — those rows must be
    # FINITE (masked out by -inf scores, but 0 * NaN != 0). Production pools are
    # the full KV cache (always in-bounds); mirror that by zeroing the padding.
    pool = total_k + TILE_N
    k = _rand(torch.bfloat16, pool, NUM_KV_HEADS, D)
    v = _rand(torch.bfloat16, pool, NUM_KV_HEADS, D)
    k[total_k:] = 0
    v[total_k:] = 0
    cu_q = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=DEVICE)
    cu_k = torch.tensor([0, 5, 6, 6], dtype=torch.int32, device=DEVICE)
    out, lse = _run(
        q, k, v, cu_q=cu_q, cu_k=cu_k,
        max_seqlen_q=1, max_seqlen_k=max(lens),
    )
    lse = lse.T  # varlen: (H, total_q) -> (B, H)
    ref, ref_lse = _fp32_ref(q, k, v, lens)
    d_abs = float((out.float() - ref).abs().max())
    ref_max = float(ref.abs().max())
    rel = d_abs / ref_max
    finite = torch.isfinite(ref_lse)
    lse_err = float((lse - ref_lse)[finite].abs().max()) if finite.any() else 0.0
    lse_inf_ok = bool(torch.isneginf(lse[~finite]).all()) if not finite.all() else True
    ok = d_abs <= MAX_ABS_TOL and rel <= MAX_REL_TOL and lse_err <= LSE_TOL and lse_inf_ok
    results.append(ok)
    _report_row(f"varlen bf16 lens={lens}", ok, d_abs, rel, lse_err)

    # --- window audit: did it stay clean for the whole run? ------------------
    window_end = time.time()
    mon.stop()
    in_window = [(ts, load) for ts, load in mon.trace if window_start <= ts <= window_end]
    # Unreachable samples count as contamination only if the server WAS
    # reachable earlier in this run (a drop after being up may hide load);
    # if it is offline the whole time, "unreachable" == "nothing on the GPU".
    saw_reachable = any(load is not None for _, load in mon.trace)
    dirty = [
        s for s in in_window
        if (s[1] is not None and s[1] != (0.0, 0.0))
        or (s[1] is None and saw_reachable)
    ]
    clean = not dirty
    gated = not no_gate and not offline

    n_ok = sum(results)
    state = "gated" if gated else ("offline/ungated" if offline else "no-gate")
    print(f"\nwindow: {state}, {'clean' if clean else 'DIRTY'} "
          f"({len(in_window)} samples, {len(dirty)} dirty)")
    print(f"cases passing: {n_ok}/{len(results)}")
    go = all(results) and clean
    print(f"TC-VERIFY VERDICT: {'GO' if go else 'NO-GO'}")
    # Exit codes (the bench convention): 0 = GO (all pass + clean window);
    # 1 = NO-GO (a case failed in a clean window); 2 = dirty window (re-run);
    # 3 = timeout (handled at the gate open).
    if not clean:
        return 2
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — verbatim traceback, then NO-GO
        print("\nEXCEPTION (verbatim traceback):")
        print(traceback.format_exc())
        print("TC-VERIFY VERDICT: NO-GO")
        raise SystemExit(1)
