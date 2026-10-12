"""End-to-end vLLM smoke test: MJOLNIR vs the stock backend.

Runs the same greedy prompts twice, each in its own process — once with
the stock backend (FLASHINFER) and once with MJOLNIR — with the top-k
logprobs of every decode step, then compares the two. fp8 KV cache on both
sides, so the only difference is the decode kernel.

Why not "the greedy tokens must match": greedy decoding of a small model on
an fp8 KV cache is chaotic. Two near-equal candidates (a fluent "Rome" vs a
fluent "also") flip on any rounding change — FlashInfer's own output differs
between CUDA-graph and eager mode — and after one flip the two texts share
nothing, so the matching prefix measures luck, not the kernel. What *is* a
property of the kernel:

* on the shared prefix (identical context on both sides) the chosen token
  gets the same logprob from both backends, within fp8 rounding;
* where the outputs first diverge, it is a near-tie: each backend's token is
  in the other's top-k and loses by at most ``--tie_nats``.

A broken kernel fails both — wrong attention moves logprobs by whole nats and
diverges onto a token the reference never considered. The matching prefix is
still printed, as information.

    python tests/smoke_vllm.py                         # Qwen3-0.6B, CUDA graphs
    python tests/smoke_vllm.py --eager --model Qwen/Qwen3-1.7B
    python tests/smoke_vllm.py --spec 2                # ngram spec decode, q_len 3
    python tests/smoke_vllm.py --kv fp6_e2m3           # Mjolnir on a 6-bit e2m3 KV cache
                                                       # (calibrates its scales first)

Works without `pip install` of the plugin: the child then registers the
backend itself and runs vLLM in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# Used only when a run returns no logprobs.
LEGACY_MIN_PREFIX, LEGACY_MIN_MATCH = 8, 0.6

# Default limits per Mjolnir KV format (flags override). fp8 from the first
# Thor runs (Qwen3-0.6B, vLLM 0.31.0): mean |Δ| 0.018 graph / 0.024 eager,
# max |Δ| 0.27, divergence gaps 0.12-0.62 nats (0.125 = one bf16 step of a
# logit near 16-32). fp6 (e2m3) adds its own rounding on top of the fp8
# reference's (harness: 1.15x fp8's relative L2 error) — twice the room until
# the first fp6 runs say better.
LIMITS = {
    "fp8": {"tie_nats": 1.0, "max_mean_dlp": 0.06, "max_dlp": 0.5},
    "fp6": {"tie_nats": 1.5, "max_mean_dlp": 0.12, "max_dlp": 1.0},
}

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean:",
    "def fibonacci(n):",
    "Explain why the sky is blue in two sentences.",
    "List the first ten prime numbers:",
    "Once upon a time, in a small village by the mountains,",
    "The three laws of thermodynamics are",
    "Translate to German: 'The weather is nice today.'",
]


def _add_pkg_path():
    """Make the kernel package `mjolnir` importable from wherever the tests
    live: an installed mjolnir-kernel, $MJOLNIR_SRC, or the nearest parent
    directory that holds src/mjolnir/mjolnir/ (a checkout)."""
    import importlib.util

    spec = importlib.util.find_spec("mjolnir")
    if spec is not None and spec.origin and os.path.isfile(os.path.join(os.path.dirname(spec.origin), "t5_decode.py")):
        return
    env = os.environ.get("MJOLNIR_SRC") or os.environ.get("THOR_T5_SRC")
    cands = [env] if env else []
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cands += [d, os.path.join(d, "src", "mjolnir")]
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "mjolnir", "t5_decode.py")):
            sys.path.insert(0, c)
            return
    raise SystemExit("the mjolnir kernel package was not found: set MJOLNIR_SRC to src/mjolnir")


def _top_logprobs(steps) -> list[dict[str, float]] | None:
    """vLLM's per-step {token id: Logprob} (list or FlatLogprobs) as plain
    JSON: [{"<id>": logprob, ...}, ...]."""
    if steps is None:
        return None
    return [{str(int(t)): float(getattr(lp, "logprob", lp)) for t, lp in step.items()} for step in steps]


def compare_prompt(ref_tok, t5_tok, ref_lp, t5_lp, tie_nats: float) -> dict:
    """One prompt: the matching prefix, |Δ logprob| of the chosen token on it,
    and — if the outputs diverge — whether the divergence is a near-tie.

    ``*_lp`` are per-step {token id: logprob} dicts (keys int or str), or None
    when the run had no logprobs."""
    n = next((k for k, (x, y) in enumerate(zip(ref_tok, t5_tok)) if x != y), min(len(ref_tok), len(t5_tok)))
    out: dict = {"prefix": n, "len": len(ref_tok), "deltas": [], "diverged": n < len(ref_tok)}
    if ref_lp is None or t5_lp is None:
        return out

    def lp(steps, k, tok):
        d = steps[k] if k < len(steps) else {}
        v = d.get(str(tok), d.get(tok)) if isinstance(d, dict) else None
        return None if v is None else float(v)

    for k in range(n):
        a, b = lp(ref_lp, k, ref_tok[k]), lp(t5_lp, k, ref_tok[k])
        if a is not None and b is not None:
            out["deltas"].append(abs(a - b))
    if out["diverged"] and n < len(t5_tok):
        a, b = ref_tok[n], t5_tok[n]
        ra, rb = lp(ref_lp, n, a), lp(ref_lp, n, b)  # the reference's view of both tokens
        ta, tb = lp(t5_lp, n, a), lp(t5_lp, n, b)  # Mjolnir's view
        gaps = [g for g in ((ra - rb) if None not in (ra, rb) else None, (tb - ta) if None not in (ta, tb) else None)]
        out["step"] = n
        out["gap"] = max((g for g in gaps if g is not None), default=None)
        out["tie"] = None not in gaps and max(gaps) <= tie_nats
    return out


def judge(results: list[dict], stats: dict, args) -> tuple[bool, dict]:
    """Overall verdict from the per-prompt comparisons (see the module doc)."""
    deltas = [d for r in results for d in r["deltas"]]
    div = [r for r in results if r["diverged"]]
    have_lp = any("tie" in r for r in div) or bool(deltas)
    summary = {
        "mean_match": sum(r["prefix"] / max(1, r["len"]) for r in results) / max(1, len(results)),
        "worst_prefix": min((r["prefix"] for r in results), default=0),
        "mean_dlp": sum(deltas) / len(deltas) if deltas else None,
        "max_dlp": max(deltas) if deltas else None,
        "diverged": len(div),
        "ties": sum(1 for r in div if r.get("tie")),
        "worst_gap": max((r["gap"] for r in div if r.get("gap") is not None), default=None),
        "logprobs": have_lp,
    }
    why = []
    if stats.get("t5_steps", 1) == 0:
        why.append("Mjolnir never ran a decode step (check the 'Mjolnir' log lines)")
    if summary["worst_prefix"] < args.min_prefix:
        why.append(f"worst prefix {summary['worst_prefix']} < --min_prefix {args.min_prefix}")
    if summary["mean_match"] < args.min_match:
        why.append(f"mean match {summary['mean_match']:.2f} < --min_match {args.min_match}")
    if have_lp:
        if summary["mean_dlp"] is not None and summary["mean_dlp"] > args.max_mean_dlp:
            why.append(f"mean |Δlogprob| {summary['mean_dlp']:.3f} > --max_mean_dlp {args.max_mean_dlp}")
        if summary["max_dlp"] is not None and summary["max_dlp"] > args.max_dlp:
            why.append(f"max |Δlogprob| {summary['max_dlp']:.3f} > --max_dlp {args.max_dlp}")
        if summary["ties"] < summary["diverged"]:
            why.append(
                f"{summary['diverged'] - summary['ties']} divergence(s) are not near-ties (≤ {args.tie_nats} nats)"
            )
    elif summary["worst_prefix"] < LEGACY_MIN_PREFIX or summary["mean_match"] < LEGACY_MIN_MATCH:
        # no logprobs (a config that cannot return them): the old, stricter token criterion
        why.append(
            f"no logprobs; token criterion: worst prefix {summary['worst_prefix']} (≥ {LEGACY_MIN_PREFIX}), "
            f"mean match {summary['mean_match']:.2f} (≥ {LEGACY_MIN_MATCH})"
        )
    summary["why"] = why
    return not why, summary


def child(args) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _add_pkg_path()
    if args.backend in ("MJOLNIR", "CUSTOM"):
        import mjolnir

        mjolnir.register()  # idempotent if the entry point already ran
        if args.backend == "MJOLNIR" and not mjolnir.has_native_member():
            args.backend = "CUSTOM"  # stock vLLM (no MJOLNIR enum member): same class via CUSTOM
    from vllm import LLM, SamplingParams

    kw = {
        "model": args.model,
        "kv_cache_dtype": args.kv if args.backend in ("MJOLNIR", "CUSTOM") else "fp8",
        "dtype": "bfloat16",
        "enforce_eager": args.eager,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem,
        "max_num_seqs": max(16, len(PROMPTS)),
        "seed": 0,
    }
    if args.spec:
        kw["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": args.spec,
            "prompt_lookup_max": 4,
            "prompt_lookup_min": 2,
        }
    try:
        llm = LLM(attention_config={"backend": args.backend}, **kw)
    except TypeError:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.backend
        llm = LLM(**kw)

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True, logprobs=args.topk)
    try:
        llm.generate(PROMPTS[:2], sp)  # warm-up
    except (ValueError, NotImplementedError) as e:  # a config that cannot return logprobs
        print(f"logprobs unavailable ({e}); comparing tokens only", flush=True)
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
        llm.generate(PROMPTS[:2], sp)
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp)
    dt = time.perf_counter() - t0
    res = {
        "backend": args.backend,
        "tokens": [list(o.outputs[0].token_ids) for o in outs],
        "text": [o.outputs[0].text for o in outs],
        # per prompt, per step: {token id (str, JSON keys): logprob}
        "logprobs": [_top_logprobs(o.outputs[0].logprobs) for o in outs],
        "seconds": dt,
    }
    if args.backend in ("MJOLNIR", "CUSTOM"):
        try:
            from mjolnir.backend import STATS

            res["stats"] = dict(STATS)
        except Exception as e:  # noqa: BLE001
            res["stats"] = {"error": repr(e)}
    with open(args.out, "w") as f:
        json.dump(res, f)


def calibrate(args, out: str) -> str:
    """Per-channel e2m3 scales for the smoke model (mjolnir.calibrate, own
    process). The full default prompt set: per-channel amax needs coverage —
    a channel seen too little saturates later."""
    cmd = [
        sys.executable,
        "-m",
        "mjolnir.calibrate",
        "--model",
        args.model,
        "--out",
        out,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-mem",
        str(args.gpu_mem),
        "--prompts",
        "48",
        "--long",
        "4",
        "--long-tokens",
        "1500",
        "--max-tokens",
        "96",
    ]
    print("\n=== calibrate (fp8 KV, e2m3 scales) ===", flush=True)
    env = dict(os.environ)
    _add_pkg_path()
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p and os.path.isdir(os.path.join(p, "mjolnir"))][:1] + [env.get("PYTHONPATH", "")]
    )
    if subprocess.run(cmd, check=False, env=env).returncode != 0:
        sys.exit("calibration failed")
    return out


def run_child(args, backend, out):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--child",
        "--backend",
        backend,
        "--out",
        out,
        "--model",
        args.model,
        "--max_tokens",
        str(args.max_tokens),
        "--max_model_len",
        str(args.max_model_len),
        "--gpu_mem",
        str(args.gpu_mem),
        "--spec",
        str(args.spec),
    ] + (["--eager"] if args.eager else [])
    if backend in ("MJOLNIR", "CUSTOM"):
        cmd += ["--kv", args.kv]
    env = dict(os.environ)
    env.pop("MJOLNIR_LM_HEAD", None)
    if backend in ("MJOLNIR", "CUSTOM") and args.lm_head != "bf16":  # the reference keeps the bf16 head
        env["MJOLNIR_LM_HEAD"] = args.lm_head
    print(f"\n=== {backend}{'' if backend not in ('MJOLNIR', 'CUSTOM') else ' (' + args.kv + ' KV)'} ===", flush=True)
    r = subprocess.run(cmd, check=False, env=env)
    if r.returncode != 0:
        sys.exit(f"{backend} run failed (exit {r.returncode})")
    with open(out) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--reference", default="FLASHINFER")
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem", type=float, default=0.15)  # Thor: 0.15 x 122 GiB ~ 18 GiB
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--spec", type=int, default=0, help="ngram speculative tokens (0 = off)")
    ap.add_argument("--kv", default="fp8", help="Mjolnir's KV cache dtype: fp8 | fp6_e2m3 (the reference stays fp8)")
    ap.add_argument(
        "--lm-head", dest="lm_head", default="bf16", help="Mjolnir side's lm_head: bf16 | fp8 (MJOLNIR_LM_HEAD)"
    )
    ap.add_argument("--topk", type=int, default=5, help="logprobs returned per step")
    ap.add_argument(
        "--tie_nats", type=float, default=None, help="a divergence passes if the chosen tokens differ by <= this"
    )
    ap.add_argument(
        "--max_mean_dlp", type=float, default=None, help="mean |Δ logprob| of the chosen token on shared prefixes"
    )
    ap.add_argument(
        "--max_dlp", type=float, default=None, help="max |Δ logprob| of the chosen token on shared prefixes"
    )
    ap.add_argument("--min_prefix", type=int, default=0, help="optional: every prompt must match this many tokens")
    ap.add_argument("--min_match", type=float, default=0.0, help="optional: mean fraction of matching leading tokens")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--backend", default="MJOLNIR")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.child:
        return child(args)
    # an fp8 lm_head rounds every logit: the fp6 room until runs say better
    limits = LIMITS["fp6" if args.kv.startswith("fp6") or args.lm_head == "fp8" else "fp8"]
    for k, v in limits.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    tmp = os.environ.get("TMPDIR", "/tmp")
    if args.kv.startswith("fp6") and not os.environ.get("MJOLNIR_ATTN_KV_SCALES"):
        os.environ["MJOLNIR_ATTN_KV_SCALES"] = calibrate(args, os.path.join(tmp, "t5_smoke_kv_scales.json"))
    ref = run_child(args, args.reference, os.path.join(tmp, "t5_smoke_ref.json"))
    t5 = run_child(args, "MJOLNIR", os.path.join(tmp, "t5_smoke_t5.json"))

    print("\n=== comparison (shared prefix, |Δ logprob| on it, first divergence) ===")
    results = []
    for i, (a, b) in enumerate(zip(ref["tokens"], t5["tokens"])):
        rl, tl = (ref.get("logprobs") or [None] * len(PROMPTS))[i], (t5.get("logprobs") or [None] * len(PROMPTS))[i]
        r = compare_prompt(a, b, rl, tl, args.tie_nats)
        results.append(r)
        dl = f"  max |Δlp| {max(r['deltas']):.3f}" if r["deltas"] else ""
        if not r["diverged"]:
            print(f"  ✅ prompt {i}: {r['prefix']:4d}/{r['len']}{dl}  {PROMPTS[i][:40]!r}")
            continue
        tie = r.get("tie")
        mark = "\U0001f7e1" if tie else "❌"
        gap = (
            f"near-tie, gap {r['gap']:.2f} nats"
            if tie
            else (f"gap {r['gap']:.2f} nats" if r.get("gap") is not None else "not in the other's top-k")
        )
        print(f"  {mark} prompt {i}: {r['prefix']:4d}/{r['len']}{dl}  diverges: {gap}  {PROMPTS[i][:40]!r}")
        print(f"       ref: {ref['text'][i][:90]!r}\n       t5 : {t5['text'][i][:90]!r}")
    stats = t5.get("stats", {})
    ok, s = judge(results, stats, args)
    print(
        f"\n  mean match {s['mean_match']:.2f}, worst prefix {s['worst_prefix']}, "
        f"time ref {ref['seconds']:.2f}s / T5 {t5['seconds']:.2f}s, T5 stats {stats}"
    )
    if s["logprobs"]:

        def fmt(v):
            return "n/a" if v is None else f"{v:.3f}"

        print(
            f"  logprob agreement: mean |Δ| {fmt(s['mean_dlp'])}, max |Δ| {fmt(s['max_dlp'])}, "
            f"near-ties {s['ties']}/{s['diverged']}, worst gap {fmt(s['worst_gap'])}"
        )
    for w in s["why"]:
        print(f"  ❌ {w}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
