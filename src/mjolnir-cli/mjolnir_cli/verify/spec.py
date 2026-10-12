"""``mjolnir verify spec`` — speculative decoding per config and draft count.

For every leg (``side:config``) and every number of draft tokens k, the
stage serves the config with ``--speculative-config`` overridden to k
(the server restarts per point, like ``kv-cache-size``), sends a fixed
prompt set at temperature 0 — the same text for every point — at each
concurrency, and records:

* tokens per verify step (1 + accepted / drafts) and per-position
  acceptance, from vLLM's spec-decode counters before / after the block;
* decode tokens/s per request (streamed: tokens after the first ÷ time
  from the first to the last token) and in aggregate, and TTFT.

Decode t/s = tokens per step ÷ step time, and every extra draft adds a
draft forward to the step — the best k is the one with the highest decode
t/s, which the stage reports per config and concurrency.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import yaml

from mjolnir_cli import dockerctl
from mjolnir_cli.config import DEFAULT_IMAGE, FAST_QUANT, STOCK_IMAGE, resolve
from mjolnir_cli.gate import server_counters, spec_delta

DEFAULT_LEGS = f"official:NVFP4,mjolnir:NVFP4_MJOLNIR,mjolnir:NVFP4_MJOLNIR_E2M3,mjolnir:{FAST_QUANT}"
DEFAULT_DRAFTS = "1 2 3 4"
DEFAULT_CONCURRENCY = "1 4"
DEFAULT_MAX_TOKENS = 256

# A fixed, varied prompt set (acceptance depends on the text: code and
# lists accept more, prose less). Every point of the stage sends exactly these.
PROMPTS = [
    "Explain how a transformer's attention mechanism works, step by step, with a small numeric example.",
    "Write a Python function that parses an ISO-8601 timestamp without external libraries, then test it.",
    "def quicksort(xs):\n    # complete this implementation and explain its complexity\n",
    "Summarize the causes and consequences of the 1929 stock market crash in five paragraphs.",
    "Solve: a train leaves at 9:40 at 72 km/h, another at 10:05 at 90 km/h on the same track. When do they meet?",
    "Write a SQL query that returns the top three customers by revenue per region, with window functions.",
    "List the differences between TCP and UDP and when you would use each, as a table.",
    "Rust: implement a thread-safe bounded queue with condition variables; explain the invariants.",
    "Translate into German, French and Polish: 'The quick brown fox jumps over the lazy dog near the river bank.'",
    "Write a bash script that finds the ten largest files under a directory and prints them in a table.",
    "Prove that the square root of two is irrational.",
    "Compose a short story about a lighthouse keeper who finds a message in a bottle.",
]
LONG_SOURCE = (
    "In the long history of computing, memory bandwidth has repeatedly set the pace of progress. Each "
    "generation of accelerators added arithmetic faster than it added bytes per second, and every workload "
    "that streams data once - decoding a language model token by token is one - became a question of how "
    "few bytes it could read. "
)


def long_prompt(words: int) -> str:
    reps = max(1, words // len(LONG_SOURCE.split()))
    body = "".join(f"[{i}] {LONG_SOURCE}" for i in range(reps))
    return body + "\nSummarize the argument above in five sentences and name its weakest step."


def prompt_set(long_words: int) -> list[str]:
    return PROMPTS + ([long_prompt(long_words)] if long_words > 0 else [])


def _ints(v: str) -> list[int]:
    return [int(x) for x in str(v).replace(",", " ").split() if x.strip()]


def _legs(spec: str) -> list[tuple[str, str]]:
    from mjolnir_cli.verify.kvcache import _legs as legs

    return legs(spec)


def spec_override(config_path: Path, k: int) -> str | None:
    """The config's speculative-config with num_speculative_tokens = k, as
    the JSON ``--speculative-config`` takes (None: the config has none)."""
    try:
        data = yaml.safe_load(Path(config_path).read_text()) or {}
    except Exception:  # noqa: BLE001
        return None
    sp = data.get("speculative-config")
    if not isinstance(sp, dict):
        return None
    return json.dumps({**sp, "num_speculative_tokens": int(k)})


def stream_one(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float = 600.0) -> dict:
    """One streamed chat completion at temperature 0: TTFT, decode t/s, tokens."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    first = last = None
    chunks = 0
    tokens = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                tokens = ev["usage"].get("completion_tokens")
            for ch in ev.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    now = time.perf_counter()
                    first = first if first is not None else now
                    last = now
                    chunks += 1
    tokens = tokens if tokens is not None else chunks
    out = {"tokens": tokens, "chunks": chunks, "ttft_ms": None, "decode_tps": None}
    if first is not None:
        out["ttft_ms"] = (first - t0) * 1000
        if last is not None and last > first and tokens > 1:
            out["decode_tps"] = (tokens - 1) / (last - first)
            out["tokens_per_chunk"] = tokens / max(1, chunks)
    return out


def run_block(s, prompts: list[str], conc: int, max_tokens: int) -> dict:
    """Every prompt once, ``conc`` in flight; spec counters around it."""
    before = server_counters(s.metrics_url)
    results: list[dict] = []
    errors: list[str] = []
    lock = threading.Lock()
    queue = list(prompts)

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                p = queue.pop(0)
            try:
                r = stream_one(s.base_url, s.served_model, p, max_tokens)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(repr(e))
                continue
            with lock:
                results.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    stats = spec_delta(before, server_counters(s.metrics_url)) or {}
    tps = [r["decode_tps"] for r in results if r.get("decode_tps")]
    ttft = [r["ttft_ms"] for r in results if r.get("ttft_ms") is not None]
    tokens = sum(r["tokens"] or 0 for r in results)
    return {
        "concurrency": conc,
        "requests": len(results),
        "errors": errors[:3],
        "tokens": tokens,
        "wall_s": round(wall, 2),
        "decode_tps": round(sum(tps) / len(tps), 3) if tps else None,  # per request, mean
        "agg_tps": round(tokens / wall, 3) if wall > 0 else None,  # all requests, incl. prefill time
        "ttft_ms": round(sum(ttft) / len(ttft), 1) if ttft else None,
        # per request, in completion order: the mean hides a few slow first steps
        "ttft_p50_ms": round(sorted(ttft)[len(ttft) // 2], 1) if ttft else None,
        "ttft_list_ms": [round(x, 1) for x in ttft],
        **{k: stats.get(k) for k in ("acceptance_length", "acceptance_rate", "per_pos", "drafts", "preemptions")},
    }


def best_k(rows: list[dict]) -> dict:
    """{(config, concurrency): row with the highest per-request decode t/s}."""
    out: dict = {}
    for r in rows:
        if r.get("decode_tps") is None:
            continue
        key = (r["config"], r["concurrency"])
        if key not in out or r["decode_tps"] > out[key]["decode_tps"]:
            out[key] = r
    return out


def stage_spec(
    ctx,
    legs: str = DEFAULT_LEGS,
    drafts: str = DEFAULT_DRAFTS,
    concurrency: str = DEFAULT_CONCURRENCY,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    long_words: int = 6000,
    model: str | None = None,
    restart: bool = False,
    timeout: float = 1800.0,
) -> dict:
    """Serve every leg at every draft count; one block per concurrency."""
    rows, cases = [], []
    mj_image = ctx.runner.image or DEFAULT_IMAGE
    params = {
        "legs": legs,
        "drafts": drafts,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "long_words": long_words,
        "prompts": len(prompt_set(long_words)),
    }
    if dockerctl.container_running() and not restart and not ctx.runner.dry_run:
        ctx.out(
            "[red]   a vLLM server is running — this stage restarts the server for every point. "
            "Stop it in a window you own (mjolnir serve down), or pass --restart.[/red]"
        )
        return {"rows": [], "cases": [{"label": "server busy (not restarted)", "passed": False}], "params": params}
    prompts = prompt_set(long_words)
    logdir = ctx.logdir / "spec"
    logdir.mkdir(parents=True, exist_ok=True)
    for side, quant in _legs(legs):
        s = resolve(model, quant, STOCK_IMAGE if side == "official" else mj_image, None)
        for k in _ints(drafts):
            point = f"{side}-{quant}-k{k}"
            override = spec_override(s.config_path, k)
            ctx.out(f"   ▸ {point}  ({s.image})")
            if override is None:
                cases.append({"label": point, "passed": False, "error": "config has no speculative-config"})
                continue
            serve_args = ("--speculative-config", override)
            if ctx.runner.dry_run:
                dockerctl.serve_up(s, wait=False, dry_run=True, serve_args=serve_args)
                continue
            err = None
            log = logdir / f"{point}.log"
            try:
                dockerctl.serve_up(s, wait=True, wait_timeout_s=timeout, serve_args=serve_args)
                for _ in range(2):  # warm-up: graphs, caches, allocator
                    stream_one(s.base_url, s.served_model, PROMPTS[0], 32)
                for conc in _ints(concurrency):
                    b = run_block(s, prompts, conc, max_tokens)
                    rows.append({"id": point, "side": side, "config": quant, "image": s.image, "k": k, **b})
                    ctx.out(
                        f"     c={conc}: {b['acceptance_length'] or 0:.2f} tok/step · decode "
                        f"{b['decode_tps'] or 0:.1f} t/s/req · agg {b['agg_tps'] or 0:.1f} t/s · per pos "
                        f"{' '.join(f'{p:.2f}' for p in (b['per_pos'] or []))}"
                        + (f" · {len(b['errors'])} errors" if b["errors"] else "")
                    )
            except (dockerctl.DockerError, TimeoutError, OSError) as e:
                err = str(e)
            finally:
                dockerctl.dump_server_log(None, log)
                dockerctl.serve_down(s)
                dockerctl._run_quiet([dockerctl._docker(), "rm", "-f", dockerctl.CONTAINER_NAME], ok_codes=(0, 1))
            ok = err is None and any(r["id"] == point and r.get("decode_tps") for r in rows)
            cases.append({"label": point, "passed": ok, **({"error": err} if err else {}), "log": str(log)})
            if not ok:
                ctx.out(f"     ❌ {err or 'no decode measured'} — {log}")
    for (cfg, conc), r in sorted(best_k(rows).items()):
        ctx.out(f"   best k for {cfg} at c={conc}: k={r['k']} ({r['decode_tps']:.1f} t/s/req)")
    return {"rows": rows, "cases": cases, "params": params}
