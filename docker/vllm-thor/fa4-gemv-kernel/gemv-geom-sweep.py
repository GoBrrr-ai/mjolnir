#!/usr/bin/env python3
"""GEMV decode B3 re-tile — geometry SWEEP (L=8192, dense fp8, GQA 24/4, hd=256).

The single clean-window artifact that ranks the (in-flight footprint) x
(CTA count) geometry space so the B3 re-tile picks its shape by measurement
instead of guesswork (the LeanTile sweep, arxiv-gemv-decode.md P5).

Axes
----
- footprint = stages * bdy * tile * D * 2 bytes (in-flight KV/CTA). bdy=g=6,
  D=256, fp8, stages fixed at 16 -> tile in {1, 2, 4} gives 48 / 96 / 192 KiB
  (192 KiB == FlashInfer FA2-tc's 384-row tile footprint).
- ns (num_splits) -> grid CTA count = ns * num_kv_heads (4). ns in
  {4, 8, 10, 16, 20, 32} -> 16 / 32 / 40 / 64 / 80 / 128 CTAs (40 == FI).

Geometry is selected by env (read in the kernel __init__ at compile time):
  VLLM_FA4_HD256_GEMV_TILE, VLLM_FA4_HD256_GEMV_STAGES (kernel), and
  VLLM_FA4_HD256_GEMV_NUM_SPLITS (interface, runtime). The compile cache is
  cleared once per (tile, stages) geometry; ns is a runtime grid value so all
  ns for a tile share one compile.

Baselines (same window, for the head-to-head): flashinfer (FA2-tc, the
~202us target to beat) and fa4_1cta (the FA4 1CTA carve-out).

Protocol: gated clean window (same Monitor/wait_for_idle as
gemv-decode-bench.py; an unreachable metrics endpoint => UNGATED, the JSON
marks it ``"gated": false``). WARMUP warm + ITERS pooled CUDA-event iters per
leg (median / p95 / min / mean).

Exit codes: 0 = written (clean or ungated); 2 = dirty window;
            3 = idle-wait timeout.

Run (one docker; under ``mjolnir bench kernel gemv-sweep`` the host's
benchmarks/raw is mounted at /raw and a bare run writes
/raw/gemv-geom-sweep.json):
  docker run --rm --gpus all --network host --entrypoint python3 \
    -v $VFA_TREE:/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$PKG":/p -e TMPDIR=/tmp/gemv-sweep \
    mjolnir/vllm-thor:<img> /p/gemv-geom-sweep.py --layout dense
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import threading
import time
import traceback
import urllib.request

import torch

NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
DEVICE = "cuda"
SOFTMAX_SCALE = HEAD_DIM ** -0.5
FP8 = torch.float8_e4m3fn

TILES = (1, 2, 4)            # footprint axis: 48 / 96 / 192 KiB in-flight KV/CTA
STAGES = 16                  # ring depth fixed for this sweep
NS_LIST = (4, 8, 10, 16, 20, 32)   # CTA-count axis: 16/32/40/64/80/128 CTAs
WARMUP = 15
ITERS = 300
ROOFLINE_BW = 273e9  # B/s (Thor LPDDR5X)

METRICS_URL = os.environ.get("MJOLNIR_METRICS_URL",
                              "http://127.0.0.1:6001/metrics")
POLL_S = 2.0
CONFIRM = 3
WAIT_TIMEOUT_S = 1800
OFFLINE_CONFIRM = 3  # consecutive unreachable samples => server offline


def server_load() -> tuple[float, float] | None:
    """(num_running, num_waiting) from the co-located vllm server, or None."""
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


class Monitor(threading.Thread):
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


def wait_for_idle(mon: Monitor) -> float | str | None:
    """Window-start ts; ``"offline"`` if the endpoint stays unreachable;
    ``None`` on timeout."""
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


class Cases:
    """Shared KV dataset (identical values for every leg)."""

    def __init__(self, L: int) -> None:
        torch.manual_seed(20260925)
        self.q = torch.randn(NUM_Q_HEADS, HEAD_DIM, device=DEVICE,
                              dtype=torch.float32).to(torch.bfloat16)
        self.k = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, device=DEVICE,
                             dtype=torch.float32).to(FP8)
        self.v = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, device=DEVICE,
                             dtype=torch.float32).to(FP8)
        self.qd = torch.full((1, NUM_KV_HEADS), 1.06, dtype=torch.float32,
                             device=DEVICE)
        self.kd = torch.full((1, NUM_KV_HEADS), 1.125, dtype=torch.float32,
                             device=DEVICE)
        self.vd = torch.full((1, NUM_KV_HEADS), 1.25, dtype=torch.float32,
                             device=DEVICE)
        self.f16 = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8,
                               device=DEVICE)


def gemv_leg(cases: Cases, L: int, tile: int, stages: int, ns: int,
             page_size: int):
    """One GEMV (tile, stages, ns) callable for sequence length L; sets the
    env knobs per call."""
    import vllm.vllm_flash_attn.cute.interface as iface
    k = cases.k[:L]
    v = cases.v[:L]
    if page_size == 0:
        k_in, v_in, pt = (k.unsqueeze(0).contiguous(),
                          v.unsqueeze(0).contiguous(), None)
    else:
        npg = L // page_size
        k_in = k.view(npg, page_size, NUM_KV_HEADS, HEAD_DIM).contiguous()
        v_in = v.view(npg, page_size, NUM_KV_HEADS, HEAD_DIM).contiguous()
        pt = torch.arange(npg, dtype=torch.int32, device=DEVICE).unsqueeze(0)
    q_in = cases.q.unsqueeze(0).unsqueeze(0)  # (1, 1, H, D)

    def fn() -> None:
        os.environ["VLLM_FA4_HD256_GEMV"] = "1"
        os.environ["VLLM_FA4_HD256_TC_SPARK"] = "0"  # reset the TC leg's env (TC gate is checked first)
        os.environ["VLLM_FA4_HD256_GEMV_TILE"] = str(tile)
        os.environ["VLLM_FA4_HD256_GEMV_STAGES"] = str(stages)
        os.environ["VLLM_FA4_HD256_GEMV_NUM_SPLITS"] = str(ns)
        iface._flash_attn_fwd(
            q_in, k_in, v_in, page_table=pt,
            max_seqlen_q=1, max_seqlen_k=L,
            softmax_scale=SOFTMAX_SCALE, causal=True,
            q_descale=cases.qd, k_descale=cases.kd, v_descale=cases.vd,
        )

    return fn


def tc_leg(cases: Cases, L: int, ns: int, page_size: int):
    """One TC (mma.sync, ``VLLM_FA4_HD256_TC_SPARK=1``) callable at ns. The TC tile
    geometry (TILE_N=64) is fixed (no GEMV tile/stages knobs), so only the
    SplitKV ns axis is swept; GEMV is force-off so the TC path is taken."""
    import vllm.vllm_flash_attn.cute.interface as iface
    k = cases.k[:L]
    v = cases.v[:L]
    if page_size == 0:
        k_in, v_in, pt, sk = (k.unsqueeze(0).contiguous(),
                              v.unsqueeze(0).contiguous(), None, None)
    else:
        npg = L // page_size
        k_in = k.view(npg, page_size, NUM_KV_HEADS, HEAD_DIM).contiguous()
        v_in = v.view(npg, page_size, NUM_KV_HEADS, HEAD_DIM).contiguous()
        pt = torch.arange(npg, dtype=torch.int32, device=DEVICE).unsqueeze(0)
        sk = torch.tensor([L], dtype=torch.int32, device=DEVICE)
    q_in = cases.q.unsqueeze(0).unsqueeze(0)  # (1, 1, H, D)

    def fn() -> None:
        os.environ["VLLM_FA4_HD256_TC_SPARK"] = "1"
        os.environ["VLLM_FA4_HD256_GEMV"] = "0"
        os.environ["VLLM_FA4_HD256_TC_SPARK_NUM_SPLITS"] = str(ns)
        iface._flash_attn_fwd(
            q_in, k_in, v_in, page_table=pt, seqused_k=sk,
            max_seqlen_q=1, max_seqlen_k=L,
            softmax_scale=SOFTMAX_SCALE, causal=True,
            q_descale=cases.qd, k_descale=cases.kd, v_descale=cases.vd,
        )

    return fn


def clear_gemv_cache() -> None:
    """Drop the GEMV compile cache so the next call recompiles with the
    (tile, stages) geometry now in env (ns is a runtime value and needs no
    recompile)."""
    import vllm.vllm_flash_attn.cute.interface as iface
    cache = getattr(iface._flash_attn_fwd, "gemv_compile_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def time_iters(fn, warmup: int = WARMUP, iters: int = ITERS) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        times.append(ev0.elapsed_time(ev1) * 1000.0)
    return times


def summarize(times: list[float]) -> dict:
    st = statistics
    return {
        "median": round(st.median(times), 2),
        "p95": round(sorted(times)[int(0.95 * (len(times) - 1))], 2),
        "min": round(min(times), 2),
        "mean": round(st.mean(times), 2),
        "iters": len(times),
    }


class FiLeg:
    """FlashInfer FA2-tc paged decode (the target to beat, ~202us @L=8192)."""

    def __init__(self, cases: Cases, L: int) -> None:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        self._cls = BatchDecodeWithPagedKVCacheWrapper
        self.wrapper = None
        self.cases = cases
        self.L = L
        self.out = None

    def setup(self) -> None:
        L = self.L
        self.wrapper = self._cls(
            self.cases.f16, kv_layout="NHD", use_cuda_graph=False,
            use_tensor_cores=True, backend="auto",
        )
        nblk = L // 16
        k = self.cases.k[:L]
        v = self.cases.v[:L]
        self.kv = (k.view(nblk, 16, NUM_KV_HEADS, HEAD_DIM).contiguous(),
                   v.view(nblk, 16, NUM_KV_HEADS, HEAD_DIM).contiguous())
        self.out = torch.empty(1, NUM_Q_HEADS, HEAD_DIM,
                               dtype=torch.bfloat16, device=DEVICE)

    def __call__(self) -> None:
        assert self.wrapper is not None
        L = self.L
        nblk = L // 16
        self.wrapper.plan(
            torch.tensor([0, nblk], dtype=torch.int32, device=DEVICE),
            torch.arange(nblk, dtype=torch.int32, device=DEVICE),
            torch.full((1,), 16, dtype=torch.int32, device=DEVICE),
            num_qo_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM, page_size=16,
            q_data_type=torch.bfloat16, kv_data_type=FP8,
            o_data_type=torch.bfloat16, sm_scale=SOFTMAX_SCALE, q_len_per_req=1,
        )
        self.wrapper.run(self.cases.q.unsqueeze(0), self.kv, q_scale=1.0,
                         k_scale=1.0, v_scale=1.0, out=self.out)


class Fa4Leg:
    """FA4 hd256 1CTA carve-out (GEMV off) — the pre-GEMV baseline."""

    def __init__(self, cases: Cases, L: int) -> None:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        self._call = flash_attn_varlen_func
        npg = L // 128
        self.args = dict(
            q=cases.q.to(FP8).unsqueeze(0),
            k=cases.k[:L].view(npg, 128, NUM_KV_HEADS, HEAD_DIM).contiguous(),
            v=cases.v[:L].view(npg, 128, NUM_KV_HEADS, HEAD_DIM).contiguous(),
            max_seqlen_q=1,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
            max_seqlen_k=L,
            seqused_k=torch.tensor([L], dtype=torch.int32, device=DEVICE),
            block_table=torch.arange(npg, dtype=torch.int32,
                                     device=DEVICE).unsqueeze(0),
            softmax_scale=SOFTMAX_SCALE, causal=True, fa_version=4,
            q_descale=cases.qd, k_descale=cases.kd, v_descale=cases.vd,
        )

    def __call__(self) -> None:
        os.environ.pop("VLLM_FA4_HD256_GEMV", None)  # force the FA4 path
        os.environ["VLLM_FA4_HD256_TC_SPARK"] = "0"        # (the TC gate is checked before FA4)
        self._call(**self.args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=("dense", "paged"), default="dense")
    ap.add_argument("--L", default="8192,32768,65536",
                    help="comma-separated sequence lengths to sweep "
                         "(default: 8192,32768,65536)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()
    L_LIST = [int(x) for x in args.L.split(",") if x.strip()]
    if args.out is None:
        base = "/raw" if os.path.isdir("/raw") else "/tmp"
        args.out = f"{base}/gemv-geom-sweep.json"
    page_size = 128 if args.layout == "paged" else 0

    import vllm.vllm_flash_attn.cute.interface as iface  # noqa: F401

    props = torch.cuda.get_device_properties(0)
    env = {
        "layout": args.layout, "L_list": L_LIST, "stages_fixed": STAGES,
        "tiles": list(TILES), "ns_list": list(NS_LIST),
        "device": str(props.name),
        "capability": list(torch.cuda.get_device_capability()),
        "sms": int(props.multi_processor_count),
        "torch": torch.__version__,
        "interface_sha256": hashlib.sha256(
            open(iface.__file__, "rb").read()).hexdigest(),
        "warmup": WARMUP, "iters": ITERS,
        "roofline_bw_bytes_per_s": ROOFLINE_BW,
        "descales": {"q": 1.06, "k": 1.125, "v": 1.25},
    }

    mon = Monitor()
    mon.start()

    # --- prep: per-L data + allocation (outside the window) -------------------
    # Build the shared KV data + FlashInfer/FA4 legs for each L, warming the
    # FA4 main path (its own compile cache) outside the window. GEMV
    # footprints are compiled lazily inside the window (first call per tile).
    print(f"[sweep] prep: building cases for L in {L_LIST}...", flush=True)
    per_l: dict[int, dict] = {}
    try:
        for L in L_LIST:
            cases = Cases(L)
            fi = FiLeg(cases, L)
            fi.setup()
            fa4 = Fa4Leg(cases, L)
            fa4()  # compile the FA4 main path outside the window
            torch.cuda.synchronize()
            per_l[L] = {"cases": cases, "fi": fi, "fa4": fa4}
            print(f"[sweep]   prepped L={L} (flashinfer + fa4_1cta)",
                  flush=True)
    except Exception:  # noqa: BLE001
        print("PREP EXCEPTION (verbatim):")
        print(traceback.format_exc())
        mon.stop()
        return 1

    load0 = server_load()
    print(f"[sweep] prep done; server load now: {load0}", flush=True)

    # --- wait for a clean window ----------------------------------------------
    offline = False
    if args.no_gate:
        window_start = time.time()
        print("[sweep] SMOKE MODE (--no-gate): window not gated")
    else:
        window = wait_for_idle(mon)
        if window is None:
            print(f"[sweep] TIMEOUT waiting for 0/0 window after {WAIT_TIMEOUT_S}s")
            mon.stop()
            return 3
        if window == "offline":
            window_start = time.time()
            offline = True
            print(f"[sweep] vLLM metrics unreachable ({METRICS_URL}) — server "
                  f"offline; running UNGATED", flush=True)
        else:
            window_start = window
            print("[sweep] CLEAN WINDOW OPEN (3 consecutive 0/0 samples)",
                  flush=True)

    # --- measurement window (all L in one clean window) ------------------------
    # One fresh GEMV compile per (tile, stages) footprint: the interface's
    # gemv_key omits tile/stages, so a new footprint requires clearing the
    # cache; ns is a runtime grid value, so every ns of a tile reuses that
    # tile's single binary (and the binary is L-agnostic, so it also carries
    # across the L values). The clear happens at each tile boundary, right
    # before the tile's first (compiling) call.
    results: dict[str, dict] = {}

    def _time(tag: str, fn) -> None:
        t = time_iters(fn)
        results[tag] = summarize(t)
        print(f"[sweep] {tag:22s} med={results[tag]['median']:9.2f} us "
              f"p95={results[tag]['p95']:9.2f} us "
              f"min={results[tag]['min']:9.2f} us", flush=True)

    try:
        for L in L_LIST:
            cases, fi, fa4 = (per_l[L]["cases"], per_l[L]["fi"],
                              per_l[L]["fa4"])
            print(f"[sweep] ── L={L} " + "─" * 40, flush=True)
            prev_tile = None
            for tile in TILES:
                for ns in NS_LIST:
                    if tile != prev_tile:
                        clear_gemv_cache()   # force this footprint's recompile
                        prev_tile = tile
                    _time(f"L{L}|tile{tile}|ns{ns}",
                          gemv_leg(cases, L, tile, STAGES, ns, page_size))
            # TC (mma.sync) SplitKV ns sweep (TILE_N fixed at 64) — the mma
            # streamer is memory-bound, so more CTAs (ns) is its lever too.
            for ns in NS_LIST:
                _time(f"L{L}|tc|ns{ns}", tc_leg(cases, L, ns, page_size))
            _time(f"L{L}|flashinfer", fi)
            _time(f"L{L}|fa4_1cta", fa4)
            # Free this L's KV/leg tensors before the next (larger) L.
            del cases, fi, fa4, per_l[L]
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        print("EXCEPTION (verbatim):")
        print(traceback.format_exc())
        mon.stop()
        return 1

    window_end = time.time()
    load_end = server_load()
    mon.stop()
    in_window = [(ts, load) for ts, load in mon.trace
                 if window_start <= ts <= window_end]
    saw_reachable = any(load is not None for _, load in mon.trace)
    dirty = []
    for s in in_window:
        load = s[1]
        if load is None:
            if saw_reachable:
                dirty.append(s)
        elif load != (0.0, 0.0):
            dirty.append(s)
    clean = not dirty

    roofline = {str(L): round(L * NUM_KV_HEADS * HEAD_DIM * 2
                              / ROOFLINE_BW * 1e6, 2) for L in L_LIST}
    report = {
        "env": env,
        "window": {
            "start": window_start, "end": window_end,
            "gated": not args.no_gate and not offline,
            "clean": clean, "num_samples": len(in_window),
            "dirty_samples": dirty,
            "server_load_before_wait": load0, "server_load_at_end": load_end,
        },
        "roofline_us": roofline,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[sweep] wrote {args.out}")

    # Per-L best GEMV geometry, vs the flashinfer + fa4_1cta targets.
    for L in L_LIST:
        gemv = {k: v for k, v in results.items()
                if k.startswith(f"L{L}|tile")}
        tc = {k: v for k, v in results.items()
              if k.startswith(f"L{L}|tc")}
        if not gemv:
            continue
        best_g = min(gemv, key=lambda k: gemv[k]["median"])
        best_g_med = gemv[best_g]["median"]
        best_t = (min(tc, key=lambda k: tc[k]["median"]) if tc else None)
        best_t_med = tc[best_t]["median"] if best_t else float("nan")
        fi_med = results.get(f"L{L}|flashinfer", {}).get("median",
                                                          float("nan"))
        fa4_med = results.get(f"L{L}|fa4_1cta", {}).get("median",
                                                         float("nan"))
        print(f"[sweep] L={L} BEST GEMV {best_g.split('|', 1)[1]}="
              f"{best_g_med} us | BEST TC {best_t.split('|', 1)[1] if best_t else '-'}="
              f"{best_t_med} us | flashinfer {fi_med} us | fa4_1cta {fa4_med} us",
              flush=True)
    verdict = "CLEAN" if clean else "DIRTY"
    if args.no_gate or offline:
        verdict += " (ungated: " + ("--no-gate" if args.no_gate
                                     else "vLLM offline") + ")"
    print(f"[sweep] VERDICT: {verdict}")
    return 0 if clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
