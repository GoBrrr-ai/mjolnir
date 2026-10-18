"""``mjolnir verify profile`` — where a decode step's GPU time goes.

Serves each leg with vLLM's torch profiler on (``--profiler-config``), sends
a fixed decode workload between ``/start_profile`` and ``/stop_profile``,
and reads the trace (PyTorch / Kineto chrome-trace JSON) the worker writes:

* **CUDA-graph replays**: every ``cudaGraphLaunch`` and the kernels it ran
  (they share its CUPTI correlation id), grouped by signature (kernel count)
  — the target's verify graph, the drafter's prefill graph, the fused draft
  decode graph — with GPU time per replay;
* **eager kernels** (not in a graph) per decode step — work that pays a CPU
  launch each time;
* GPU busy vs idle time over the window (idle = the CPU did not keep up);
* kernel time by category (attention, GEMM, linear attention, sampling,
  copies, …) per generated token, and the top kernels.

Two legs side by side answer "is the drafter graphed, and which kernels make
its step slower". The raw traces stay in the run's log dir.
"""

from __future__ import annotations

import gzip
import json
import re
import threading
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from mjolnir_cli import dockerctl
from mjolnir_cli.config import DEFAULT_IMAGE, STOCK_IMAGE, resolve
from mjolnir_cli.legs import verify_legs

DEFAULT_LEGS = verify_legs("verify.profile")  # mjolnir_cli.legs

CATEGORIES = (  # first match wins: GEMMs before attention (FlashInfer's NVFP4 GEMMs are "flashinfer…gemm")
    ("gemm", r"gemm|gemv|nvjet|xmma|matmul|cublas|splitK|scaled_mm|_fp8_head"),
    ("linear-attn", r"gated_delta|delta_rule|fused_recurrent|gdn|mamba|conv1d|causal_conv"),
    ("attention", r"mjolnir|t5_decode|fmha|flash_attn|flash_fwd|attention|BatchDecode|BatchPrefill|merge_attn|xqa"),
    ("sampling", r"sample|topk|top_k|argmax|gumbel|softmax|rejection|logits|penalt|exponential"),
    ("norm/act", r"norm|rms|silu|gelu|swiglu|act_and_mul|rotary|rope"),
    ("copy", r"copy|memcpy|memset|cat_|index|gather|scatter|fill|Memcpy|Memset"),
)


def categorize(name: str) -> str:
    for cat, pat in CATEGORIES:
        if re.search(pat, name, re.IGNORECASE):
            return cat
    return "other"


def _load_trace(path: Path) -> dict:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def _union(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def parse_trace(path: Path, tokens: int | None = None, top: int = 15) -> dict:
    """Summarize one Kineto trace (times in ms)."""
    events = _load_trace(path).get("traceEvents", [])
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    runtime = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                runtime[c] = e
    if not kernels:
        return {"kernels": 0}
    t0 = min(k["ts"] for k in kernels)
    t1 = max(k["ts"] + k.get("dur", 0) for k in kernels)
    window = (t1 - t0) / 1000
    busy = _union([(k["ts"], k["ts"] + k.get("dur", 0)) for k in kernels]) / 1000

    by_cat: dict[str, float] = defaultdict(float)
    by_name: dict[str, list[float]] = defaultdict(list)
    graph_kernels: dict[int, list[dict]] = defaultdict(list)
    eager = []
    for k in kernels:
        d = k.get("dur", 0) / 1000
        by_cat[categorize(k.get("name", ""))] += d
        by_name[k.get("name", "")].append(d)
        corr = (k.get("args") or {}).get("correlation")
        launcher = runtime.get(corr, {}).get("name", "")
        if corr is not None and "GraphLaunch" in launcher:
            graph_kernels[corr].append(k)
        else:
            eager.append(k)

    # one entry per graph replay: kernel count, GPU span, summed kernel time
    replays: list[dict[str, Any]] = []
    for corr, ks in graph_kernels.items():
        start = min(k["ts"] for k in ks)
        end = max(k["ts"] + k.get("dur", 0) for k in ks)
        replays.append(
            {
                "corr": corr,
                "ts": start,
                "n": len(ks),
                "span_ms": (end - start) / 1000,
                "gpu_ms": sum(k.get("dur", 0) for k in ks) / 1000,
                "attn_ms": sum(k.get("dur", 0) for k in ks if categorize(k.get("name", "")) == "attention") / 1000,
                "names": defaultdict(float),
            }
        )
        for k in ks:
            replays[-1]["names"][k.get("name", "")] += k.get("dur", 0) / 1000
    sigs: dict[int, list[dict]] = defaultdict(list)
    for r in replays:
        sigs[r["n"]].append(r)
    graphs: list[dict[str, Any]] = []
    for n, rs in sorted(sigs.items(), key=lambda kv: -kv[0]):
        names: dict[str, float] = defaultdict(float)
        for r in rs:
            for nm, v in r["names"].items():
                names[nm] += v
        graphs.append(
            {
                "kernels": n,
                "replays": len(rs),
                "span_ms": round(sum(r["span_ms"] for r in rs) / len(rs), 3),
                "gpu_ms": round(sum(r["gpu_ms"] for r in rs) / len(rs), 3),
                "attn_ms": round(sum(r["attn_ms"] for r in rs) / len(rs), 3),
                "top": [[nm[:90], round(v / len(rs), 4)] for nm, v in sorted(names.items(), key=lambda kv: -kv[1])[:5]],
            }
        )
    # eager kernels between graph replays, per replay of the largest graph
    n_steps = max((g["replays"] for g in graphs[:1]), default=0) or 1
    eager_ms = sum(k.get("dur", 0) for k in eager) / 1000
    eager_names: dict[str, list[float]] = defaultdict(list)
    for k in eager:
        eager_names[k.get("name", "")].append(k.get("dur", 0) / 1000)

    def tops(d: dict[str, list[float]]) -> list[list]:
        rows = sorted(d.items(), key=lambda kv: -sum(kv[1]))[:top]
        return [[nm[:90], len(v), round(sum(v), 3), round(sum(v) / len(v) * 1000, 1)] for nm, v in rows]

    per_tok = (lambda v: round(v / tokens, 4)) if tokens else (lambda v: None)
    return {
        "kernels": len(kernels),
        "window_ms": round(window, 2),
        "busy_ms": round(busy, 2),
        "idle_pct": round((1 - busy / window) * 100, 1) if window > 0 else None,
        "tokens": tokens,
        "ms_per_token": per_tok(window),
        "by_category_ms": {k: round(v, 3) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        "by_category_ms_per_token": {k: per_tok(v) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        "graphs": graphs,
        "graph_replays": len(replays),
        "eager_kernels": len(eager),
        "eager_ms": round(eager_ms, 3),
        "eager_kernels_per_step": round(len(eager) / n_steps, 1),
        "eager_kernels_per_token": round(len(eager) / tokens, 1) if tokens else None,
        "graph_replays_per_token": round(len(replays) / tokens, 2) if tokens else None,
        "top_kernels": tops(by_name),  # [name, count, total ms, mean µs]
        "top_eager": tops(eager_names),
    }


# ── the stage ───────────────────────────────────────────────────────────────


def _post(url: str, timeout: float = 600.0) -> None:
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()


def _context_prompt(words: int) -> str:
    from mjolnir_cli.verify.spec import long_prompt

    return long_prompt(words) if words > 0 else "Write a detailed essay about the history of computing."


def _newest_trace(d: Path, since: float) -> Path | None:
    cands = [p for p in d.rglob("*.pt.trace.json*") if p.stat().st_mtime >= since - 1]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def stage_profile(
    ctx,
    legs: str = DEFAULT_LEGS,
    drafts: int = 3,
    context_words: int = 6000,
    concurrency: int = 1,
    requests: int = 2,
    max_tokens: int = 128,
    model: str | None = None,
    restart: bool = False,
    timeout: float = 1800.0,
) -> dict:
    from mjolnir_cli.verify.kvcache import _legs
    from mjolnir_cli.verify.spec import spec_override, stream_one

    params = {
        "legs": legs,
        "drafts": drafts,
        "context_words": context_words,
        "concurrency": concurrency,
        "requests": requests,
        "max_tokens": max_tokens,
    }
    rows, cases = [], []
    if dockerctl.container_running() and not restart and not ctx.runner.dry_run:
        ctx.out(
            "[red]   a vLLM server is running — this stage restarts the server for every leg. "
            "Stop it in a window you own (mjolnir serve down), or pass --restart.[/red]"
        )
        return {"rows": [], "cases": [{"label": "server busy (not restarted)", "passed": False}], "params": params}
    prompt = _context_prompt(context_words)
    mj_image = ctx.runner.image or DEFAULT_IMAGE
    for side, quant in _legs(legs):
        s = resolve(model, quant, STOCK_IMAGE if side == "official" else mj_image, None)
        point = f"{side}-{quant}-k{drafts}"
        sub = f"profiles/{ctx.run_id}/{point}"
        host_dir = s.data_dir / "cache" / sub
        prof = json.dumps(
            {
                "profiler": "torch",
                "torch_profiler_dir": f"/data/cache/{sub}",
                "torch_profiler_with_stack": False,
                "torch_profiler_use_gzip": True,
                "torch_profiler_dump_cuda_time_total": False,
                "ignore_frontend": True,
            }
        )
        serve_args: tuple[str, ...] = ("--profiler-config", prof)
        override = spec_override(s.config_path, drafts) if drafts > 0 else None
        if override:
            serve_args += ("--speculative-config", override)
        ctx.out(f"   ▸ {point}  ({s.image})")
        if ctx.runner.dry_run:
            dockerctl.serve_up(s, wait=False, dry_run=True, serve_args=serve_args)
            continue
        log = ctx.logdir / "profile" / f"{point}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        err, summary = None, None
        try:
            host_dir.mkdir(parents=True, exist_ok=True)
            dockerctl.serve_up(s, wait=True, wait_timeout_s=timeout, serve_args=serve_args)
            root = s.base_url.rsplit("/v1", 1)[0]
            for _ in range(2):  # warm-up at the same shape: compiles, graphs, allocator
                stream_one(s.base_url, s.served_model, prompt, 32)
            t_start = time.time()
            _post(f"{root}/start_profile")
            gen = [0]
            lock = threading.Lock()

            # bind the per-point values: the threads are joined before the next point rebinds them
            def work(s=s, gen=gen, lock=lock):
                r = stream_one(s.base_url, s.served_model, prompt, max_tokens)
                with lock:
                    gen[0] += r.get("tokens") or 0

            for _ in range(requests):
                ts = [threading.Thread(target=work) for _ in range(concurrency)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
            _post(f"{root}/stop_profile")
            trace = None
            for _ in range(120):  # the worker writes the trace after stop
                trace = _newest_trace(host_dir, t_start)
                if trace is not None:
                    size = trace.stat().st_size
                    time.sleep(2)
                    if trace.stat().st_size == size and size > 0:
                        break
                time.sleep(2)
            if trace is None:
                err = f"no trace in {host_dir}"
            else:
                summary = parse_trace(trace, tokens=gen[0])
                summary["trace"] = str(trace)
        except (dockerctl.DockerError, TimeoutError, OSError, ValueError) as e:
            err = str(e)
        finally:
            dockerctl.dump_server_log(None, log)
            dockerctl.serve_down(s)
            dockerctl._run_quiet([dockerctl._docker(), "rm", "-f", dockerctl.CONTAINER_NAME], ok_codes=(0, 1))
        ok = err is None and summary is not None and summary.get("kernels")
        cases.append({"label": point, "passed": bool(ok), **({"error": err} if err else {}), "log": str(log)})
        if not ok:
            ctx.out(f"     ❌ {err or 'empty trace'} — {log}")
            continue
        rows.append({"id": point, "side": side, "config": quant, "image": s.image, "k": drafts, **summary})
        ctx.out(
            f"     {summary['tokens']} tokens · {summary['ms_per_token']} ms/token · GPU idle "
            f"{summary['idle_pct']}% · {summary['graph_replays']} graph replays · "
            f"{summary['eager_kernels_per_token']} eager kernels/token"
        )
        for g in summary["graphs"][:4]:
            ctx.out(
                f"       graph ×{g['replays']:<4} {g['kernels']:>5} kernels  {g['span_ms']:8.3f} ms span  "
                f"{g['gpu_ms']:8.3f} ms GPU  (attention {g['attn_ms']:.3f})  top: {g['top'][0][0][:60]}"
            )
    return {"rows": rows, "cases": cases, "params": params}
