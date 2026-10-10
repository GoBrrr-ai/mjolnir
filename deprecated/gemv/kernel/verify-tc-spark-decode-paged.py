#!/usr/bin/env python3
"""TC (mma.sync) hd256 DECODE — PAGED-KV verify on Jetson Thor (sm_110).

Verifies the paged leg of the tensor-core decode kernel
(``vllm.vllm_flash_attn.cute.sm100_hd256_decode_tc_spark`` + the interface paged
dispatch): the kernel gathers each n_block's physical page from the page
table in-kernel (pidx/ktile, the R1-PROVEN runtime-gather idiom) and takes
the per-batch true KV length from ``seqused_k``.

Reference: apply the SAME page table to the physical KV pool to deinterleave
it into dense per-batch KV, then run the same fp32 M=1 GQA reference the
dense verify uses (verify-tc-decode.py). The cross-check cell additionally
asserts TC-paged == TC-dense (identical logical KV) on O and LSE.

Cells (PS=128 primary; PS is a compile-time constant — each PS is a separate
compile):
  1. identity table, L=256, bf16
  2. SHUFFLED table [3,1,0,2], L=512, bf16 (the real gather check)
  3. partial-page tail, L=97 (PS=128, 2 n_blocks), bf16
  4. large-L, L=8192 (64 pages, runtime page indices deep), bf16
  5. production fp8 leg: q bf16 + KV e4m3 + non-unity descales, L=256
  6. multi-batch B=2: distinct kv_len (256/97) + distinct per-batch tables
  7. dense-vs-paged cross-check: identical O+LSE (the paged leg must equal
     the dense leg on the same logical KV)

Run inside the image with the vfa tree mounted:
  docker run --rm --gpus all --network host --entrypoint python3 \
    -v $VFA_TREE:/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$PWD":/p \
    mjolnir/vllm-thor:qwen38-sm110-v13 /p/verify-tc-decode-paged.py
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
PS = 128     # page size (primary; the kernel requires PS % TILE_N == 0)
DEVICE = "cuda"
SOFTMAX_SCALE = D ** -0.5

MAX_ABS_TOL = 5e-2
MAX_REL_TOL = 5e-2  # relative to ref_max
LSE_TOL = 2e-2
# fp8 leg: the P (softmax weights) operand is cast to e4m3 for the PV mma —
# same tolerances as the dense verify's fp8 leg.
FP8_MAX_ABS_TOL = 5e-2
FP8_MAX_REL_TOL = 5e-2
FP8_Q_DESCALE = 1.06
FP8_K_DESCALE = 1.125
FP8_V_DESCALE = 1.25
# cross-check leg-vs-leg (TC-paged vs TC-dense, identical data): same kernel
# math, so the diff is LSE-store/mma tie-break rounding only.
XCHK_MAX_ABS_TOL = 2e-3
XCHK_LSE_TOL = 1e-3
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
    co-located server is down -> nothing to gate against); ``None" on timeout."""
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
    """fp32 M=1 GQA reference (bf16, no descale). q: (B,1,H,D), k/v:
    (B,L_max,H_kv,D) with per-batch length kv_lens[b]. Returns (out, lse)
    with out fp32 (B,H,D) and lse (B,H) fp32 natural units."""
    B = q.shape[0]
    out = torch.empty(B, NUM_Q_HEADS, D, dtype=torch.float32, device=DEVICE)
    lse = torch.empty(B, NUM_Q_HEADS, dtype=torch.float32, device=DEVICE)
    for b in range(B):
        L = kv_lens[b]
        q_b = q[b, 0].float()  # (H, D)
        if L == 0:
            out[b] = 0.0
            lse[b] = float("-inf")
            continue
        ks = k[b, :L].float()  # (L, Hkv, D)
        vs = v[b, :L].float()
        for h in range(NUM_Q_HEADS):
            hkv = h // G
            s = (q_b[h] @ ks[:, hkv].T) * SOFTMAX_SCALE  # (L,) natural units
            p = torch.softmax(s, dim=-1)
            out[b, h] = p @ vs[:, hkv]
            lse[b, h] = torch.logsumexp(s, dim=-1)
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
    """fp32 M=1 GQA reference for the fp8 KV leg (dequantized values) — same
    math as verify-tc-decode.py's _fp32_ref_fp8 (q cast bf16->fp8 identically
    to the kernel, descales folded as in the kernel)."""
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


def _unpage(
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    table: torch.Tensor,
    kv_lens: list[int],
    ps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the page table to the physical pool -> dense (B, L_max, H_kv, D)
    KV (the torch reference the paged kernel must match). Logical row r of
    batch b comes from physical page table[b, r // ps], offset r % ps. Rows
    beyond kv_lens[b] are zeroed (finite; mirrors the padded-pool tail)."""
    B = table.shape[0]
    L_max = max(kv_lens)
    dt = pool_k.dtype
    k = torch.zeros(B, L_max, NUM_KV_HEADS, D, dtype=dt, device=DEVICE)
    v = torch.zeros(B, L_max, NUM_KV_HEADS, D, dtype=dt, device=DEVICE)
    for b in range(B):
        L = kv_lens[b]
        if L == 0:
            continue
        r = torch.arange(L, dtype=torch.int32, device=DEVICE)
        pg = table[b, r // ps].long()   # (L,) physical page ids
        off = r % ps                     # (L,) in-page offsets
        k[b, :L] = pool_k[pg, off]       # (L, H_kv, D) advanced indexing
        v[b, :L] = pool_v[pg, off]
    return k, v


def _run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_table: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    max_seqlen_k: int,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    os.environ["VLLM_FA4_HD256_TC_SPARK"] = "1"
    os.environ["VLLM_FA4_HD256_GEMV"] = "0"
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    out, lse, _, _ = _flash_attn_fwd(
        q,
        k,
        v,
        page_table=page_table,
        seqused_k=seqused_k,
        max_seqlen_q=1,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        return_lse=True,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    return out.squeeze(1), lse.squeeze(-1)  # dense q: (B,1,H,D)->(B,H,D)


def _report_row(name: str, ok: bool, max_abs: float, rel: float,
                lse_err: float | None, tol: float = MAX_ABS_TOL,
                tol_rel: float = MAX_REL_TOL,
                lse_tol: float = LSE_TOL) -> None:
    tag = "OK " if ok else "BAD"
    lse_s = f" lse_err={lse_err:.4e}" if lse_err is not None else ""
    print(f"[{tag}] {name:44s} max_abs={max_abs:.4e} rel={rel:.4e}"
          f" (tol {tol:.0e}/{tol_rel:.0e}, lse {lse_tol:.0e}){lse_s}", flush=True)


def main() -> int:
    if not torch.cuda.is_available():
        print("TC-PAGED-VERIFY VERDICT: NO-GO (no CUDA device)")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name!r} SMs={props.multi_processor_count}")
    print(f"geometry: GQA {NUM_Q_HEADS}/{NUM_KV_HEADS} (g={G}), hd={D}, M=1, "
          f"PS={PS}\n")

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

    def _cell_pool(
        name: str,
        dtype: torch.dtype,
        kv_lens: list[int],
        table_rows: list[list[int]],
        n_spare_pages: int = 1,
        fp8_descales: bool = False,
        xcheck_dense: bool = False,
    ) -> None:
        """One paged cell: build pool + tables, unpage -> torch ref, run the
        paged kernel, compare O + LSE; optional dense-leg cross-check."""
        B = len(kv_lens)
        # n_pages = the page-table width (max over batches of the pages a
        # single batch's kv_len needs). pool_pages = every physical page any
        # table row may reference (a shorter batch's table can point at a
        # HIGHER physical page id than its own width) + spares.
        n_pages = max((L + PS - 1) // PS for L in kv_lens)
        max_ref = max(max(row) for row in table_rows)
        pool_pages = max(n_pages, max_ref + 1) + n_spare_pages
        torch.manual_seed(9000 + hash(name) % 1000)
        pool_k = _rand(dtype, pool_pages, PS, NUM_KV_HEADS, D)
        pool_v = _rand(dtype, pool_pages, PS, NUM_KV_HEADS, D)
        assert all(p < pool_pages for row in table_rows for p in row), (
            "table references a page id outside the pool")
        table = torch.tensor(
            [table_rows[b] + [0] * (n_pages - len(table_rows[b])) for b in range(B)],
            dtype=torch.int32, device=DEVICE,
        )
        seqused = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)
        q = _rand(torch.bfloat16, B, 1, NUM_Q_HEADS, D)
        # The torch reference: deinterleave the pool with the SAME table.
        dk, dv = _unpage(pool_k, pool_v, table, kv_lens, PS)
        if fp8_descales:
            qd = torch.full((B, NUM_KV_HEADS), FP8_Q_DESCALE,
                            dtype=torch.float32, device=DEVICE)
            kd = torch.full((B, NUM_KV_HEADS), FP8_K_DESCALE,
                            dtype=torch.float32, device=DEVICE)
            vd = torch.full((B, NUM_KV_HEADS), FP8_V_DESCALE,
                            dtype=torch.float32, device=DEVICE)
            ref, ref_lse = _fp32_ref_fp8(q, dk, dv, qd, kd, vd, kv_lens)
            out, lse = _run(q, pool_k, pool_v, table, seqused, max(kv_lens),
                            qd, kd, vd)
            tol, tol_rel = FP8_MAX_ABS_TOL, FP8_MAX_REL_TOL
        else:
            ref, ref_lse = _fp32_ref(q, dk, dv, kv_lens)
            out, lse = _run(q, pool_k, pool_v, table, seqused, max(kv_lens))
            tol, tol_rel = MAX_ABS_TOL, MAX_REL_TOL
        d_abs = float((out.float() - ref).abs().max())
        ref_max = float(ref.abs().max())
        rel = d_abs / ref_max
        lse_err = float((lse - ref_lse).abs().max())
        ok = d_abs <= tol and rel <= tol_rel and lse_err <= LSE_TOL
        results.append(ok)
        _report_row(name, ok, d_abs, rel, lse_err, tol, tol_rel)
        if not xcheck_dense:
            return
        # Cross-check: the TC DENSE leg on the same logical (unpaged) KV must
        # match the paged leg on O and LSE (identical math, same data).
        dk_pad = dk.clone()
        dv_pad = dv.clone()
        # dense buffers must be a whole tile (the V load is un-predicated) —
        # L_max is a multiple of PS=128 >= TILE_N by construction for every
        # cell using this check; pad if a kv_len below L_max leaves a partial.
        L_max_pad = ((max(kv_lens) + TILE_N - 1) // TILE_N) * TILE_N
        if L_max_pad > max(kv_lens):
            pad_k = torch.zeros(B, L_max_pad, NUM_KV_HEADS, D, dtype=dtype,
                                device=DEVICE)
            pad_v = torch.zeros(B, L_max_pad, NUM_KV_HEADS, D, dtype=dtype,
                                device=DEVICE)
            pad_k[:, :max(kv_lens)] = dk
            pad_v[:, :max(kv_lens)] = dv
            dk_pad, dv_pad = pad_k, pad_v
        out_d, lse_d = _run(q, dk_pad, dv_pad, None, None, max(kv_lens))
        x_abs = float((out.float() - out_d.float()).abs().max())
        x_lse = float((lse - lse_d).abs().max())
        xok = x_abs <= XCHK_MAX_ABS_TOL and x_lse <= XCHK_LSE_TOL
        results.append(xok)
        _report_row(name + "  [xcheck dense-vs-paged]", xok, x_abs,
                    x_abs / max(ref_max, 1e-6), x_lse,
                    XCHK_MAX_ABS_TOL, XCHK_MAX_ABS_TOL, XCHK_LSE_TOL)

    # 1. identity table, L=256, bf16 ------------------------------------------
    _cell_pool("paged identity L=256 bf16", torch.bfloat16,
               [256], [[0, 1]])

    # 2. SHUFFLED table [3,1,0,2], L=512, bf16 (the real gather check) --------
    _cell_pool("paged SHUFFLED L=512 bf16", torch.bfloat16,
               [512], [[3, 1, 0, 2]])

    # 3. partial-page tail, L=97 (PS=128, 2 n_blocks, 97/128 of page 0) ------
    _cell_pool("paged partial-tail L=97 bf16", torch.bfloat16,
               [97], [[0]], n_spare_pages=1)

    # 4. large-L, L=8192 (64 pages, deep runtime page indices), bf16 ---------
    _cell_pool("paged large-L L=8192 bf16", torch.bfloat16,
               [8192], [list(range(64))])

    # 5. production fp8 leg: q bf16 + KV e4m3 + non-unity descales ----------
    _cell_pool("paged fp8-Q L=256 (q bf16, kv fp8)", FP8,
               [256], [[0, 1]], fp8_descales=True)

    # 6. multi-batch B=2: distinct kv_len (256/97) + distinct tables ----------
    # batch 0: pages [1,0] (shuffled 2 pages); batch 1: page [2] only.
    _cell_pool("paged multi-batch B=2 (256/97)", torch.bfloat16,
               [256, 97], [[1, 0], [2]], n_spare_pages=0)

    # 7. dense-vs-paged cross-check (the paged leg must equal the dense leg) --
    _cell_pool("paged cross-check L=256 bf16", torch.bfloat16,
               [256], [[0, 1]], xcheck_dense=True)

    # --- window audit: did it stay clean for the whole run? ------------------
    window_end = time.time()
    mon.stop()
    in_window = [(ts, load) for ts, load in mon.trace if window_start <= ts <= window_end]
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
    print(f"gated: {'true' if gated else 'false'}")
    print(f"cases passing: {n_ok}/{len(results)}")
    go = all(results) and clean
    print(f"TC-PAGED-VERIFY VERDICT: {'GO' if go else 'NO-GO'}")
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
        print("TC-PAGED-VERIFY VERDICT: NO-GO")
        raise SystemExit(1)
