"""End-to-end vLLM smoke test: MJOLNIR vs the stock backend.

Runs the same greedy prompts twice, each in its own process — once with
the stock backend (FLASHINFER) and once with MJOLNIR — then compares the
generated tokens. fp8 KV cache on both sides, so the only difference is the
decode kernel (T5 uses fp8 P and, by default, fp8 Q, so the first few tokens
must match and the rest should mostly match).

    python tests/smoke_vllm.py                         # Qwen3-0.6B, CUDA graphs
    python tests/smoke_vllm.py --eager --model Qwen/Qwen3-1.7B
    python tests/smoke_vllm.py --spec 2                # ngram spec decode, q_len 3

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


def child(args) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _add_pkg_path()
    if args.backend in ("MJOLNIR", "CUSTOM"):
        import mjolnir

        mjolnir.register()  # idempotent if the entry point already ran
        if args.backend == "MJOLNIR" and not mjolnir.has_native_member():
            args.backend = "CUSTOM"  # stock vLLM (no MJOLNIR enum member): same class via CUSTOM
    from vllm import LLM, SamplingParams

    kw = dict(
        model=args.model,
        kv_cache_dtype="fp8",
        dtype="bfloat16",
        enforce_eager=args.eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        max_num_seqs=max(16, len(PROMPTS)),
        seed=0,
    )
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

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    llm.generate(PROMPTS[:2], sp)  # warm-up
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp)
    dt = time.perf_counter() - t0
    res = {
        "backend": args.backend,
        "tokens": [list(o.outputs[0].token_ids) for o in outs],
        "text": [o.outputs[0].text for o in outs],
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
    print(f"\n=== {backend} ===", flush=True)
    r = subprocess.run(cmd)
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
    ap.add_argument(
        "--min_prefix", type=int, default=8, help="every prompt must match the reference for this many tokens"
    )
    ap.add_argument("--min_match", type=float, default=0.6, help="mean fraction of identical leading tokens required")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--backend", default="MJOLNIR")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.child:
        return child(args)

    tmp = os.environ.get("TMPDIR", "/tmp")
    ref = run_child(args, args.reference, os.path.join(tmp, "t5_smoke_ref.json"))
    t5 = run_child(args, "MJOLNIR", os.path.join(tmp, "t5_smoke_t5.json"))

    print("\n=== comparison (leading identical tokens) ===")
    fracs, worst = [], 10**9
    for i, (a, b) in enumerate(zip(ref["tokens"], t5["tokens"])):
        n = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        fracs.append(n / max(1, len(a)))
        worst = min(worst, n)
        mark = "✅" if n == len(a) else ("\U0001f7e1" if n >= args.min_prefix else "❌")
        print(f"  {mark} prompt {i}: {n:4d}/{len(a)}  {PROMPTS[i][:40]!r}")
        if n < len(a):
            print(f"       ref: {ref['text'][i][:90]!r}\n       t5 : {t5['text'][i][:90]!r}")
    mean = sum(fracs) / len(fracs)
    stats = t5.get("stats", {})
    print(
        f"\n  mean match {mean:.2f}, worst prefix {worst}, "
        f"time ref {ref['seconds']:.2f}s / T5 {t5['seconds']:.2f}s, T5 stats {stats}"
    )
    ok = worst >= args.min_prefix and mean >= args.min_match
    if stats.get("t5_steps", 1) == 0:
        print("  ❌ T5 never ran a decode step (check the 'Mjolnir' log lines)")
        ok = False
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
