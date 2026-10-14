"""Calibrate per-(layer, KV head, channel) K/V scales for the fp6 (e2m3) KV cache.

Runs the model in-process on vLLM with the Mjolnir backend on **fp8** KV and
``MJOLNIR_ATTN_KV_CALIBRATE=1``: every K/V row the backend writes to the
cache is folded into a running per-channel amax (``mjolnir.kvscales``). After
the engine's own warm-up and profiling runs the accumulators are zeroed, a
varied prompt set is generated (prose, code, math, multilingual, long
contexts), and the result is written as a ``mjolnir.kv_scales/2`` file —
what ``MJOLNIR_ATTN_KV_SCALES`` points ``--kv-cache-dtype fp6_e2m3`` at.

    python -m mjolnir.calibrate --config /configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR.yaml \\
        --out /out/kv_scales.json --download-dir /data/models/huggingface
    python -m mjolnir.calibrate --model Qwen/Qwen3-0.6B --out /tmp/scales.json   # small models / tests

The scales cover every attention layer that runs on the Mjolnir backend,
drafters included (drafting writes its KV too): the built-in MTP head, and an
external drafter whose ``speculative-config.attention_backend`` is MJOLNIR
(DSpark: its 5 layers come out as layers 64-68). ``mjolnir calibrate`` runs this in the image and writes a config
with its own drafter to ``<config>.kv_scales.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

TOPICS = [
    "Explain how a transformer's attention mechanism works, step by step, with a small numeric example.",
    "Write a Python function that parses an ISO-8601 timestamp without external libraries, then test it.",
    "def quicksort(xs):\n    # complete this implementation and explain its complexity\n",
    "Summarize the causes and consequences of the 1929 stock market crash in five paragraphs.",
    "Solve: a train leaves at 9:40 at 72 km/h, another at 10:05 at 90 km/h on the same track. When do they meet?",
    "Translate into German, French and Polish: 'The quick brown fox jumps over the lazy dog near the river bank.'",
    "Write a SQL query that returns the top three customers by revenue per region, with window functions.",
    "Compose a short story about a lighthouse keeper who finds a message in a bottle.",
    "List the differences between TCP and UDP and when you would use each, as a table.",
    "Prove that the square root of two is irrational.",
    "Describe the architecture of the NVIDIA Blackwell GPU memory hierarchy for a systems engineer.",
    "Rust: implement a thread-safe bounded queue with condition variables; explain the invariants.",
    "What are the main ideas of Kant's Critique of Pure Reason? Answer for a first-year student.",
    "日本の四季について、それぞれの季節の特徴を説明してください。",
    "Напиши краткое руководство по выращиванию томатов на балконе.",
    'Given the JSON {"a": [1, 2, {"b": null}], "c": "x"}, write a jq expression that extracts b and explain it.',
]

LONG_SOURCE = (
    "In the long history of computing, memory bandwidth has repeatedly set the pace of progress. "
    "Each generation of accelerators added arithmetic faster than it added bytes per second, and "
    "every workload that streams data once — decoding a language model token by token is one — "
    "became a question of how few bytes it could read. Caches help only when data is reused; a "
    "key-value cache of a long conversation is read in full for every new token, so its format "
    "decides the speed. "
)


def long_prompt(tokens: int, i: int) -> str:
    """A long context (~`tokens` words) ending in a question about it."""
    words_per_rep = len(LONG_SOURCE.split())
    reps = max(1, tokens // words_per_rep)
    body = "".join(f"[section {k + i}] {LONG_SOURCE}" for k in range(reps))
    return body + "\nQuestion: summarize the argument above in three sentences and name its weakest step."


def _engine_parser():
    from vllm.engine.arg_utils import EngineArgs

    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:  # older vLLM
        from vllm.utils import FlexibleArgumentParser
    return EngineArgs.add_cli_args(FlexibleArgumentParser()), EngineArgs


def engine_kwargs_from_config(path: str) -> dict:
    """A `vllm serve --config` YAML as LLM(...) kwargs, parsed by vLLM's own
    argument parser — so every value gets the type `vllm serve` gives it
    (``16K`` -> 16384, ``kv-cache-dtype-skip-layers: sliding_window`` -> a
    list, nested dicts -> configs). Serve-only keys (port, served-model-name,
    tool parsers, ...) are dropped; only values that differ from the engine
    defaults are returned."""
    import tempfile

    import yaml

    parser, engine_args = _engine_parser()
    data = yaml.safe_load(open(path)) or {}
    known = {k: v for k, v in data.items() if "--" + str(k).replace("_", "-") in parser._option_string_actions}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(known, f)
    try:
        ns = vars(parser.parse_args(["--config", f.name]))
    finally:
        os.unlink(f.name)
    base = vars(parser.parse_args([]))
    fields = {fld.name for fld in dataclasses.fields(engine_args)}
    kw = {}
    for k, v in ns.items():
        if k not in fields:
            continue
        try:
            same = v == base.get(k)
        except Exception:  # noqa: BLE001 - uncomparable config objects: keep the value
            same = False
        if not same:
            kw[k] = v
    return kw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="vllm serve YAML (the serving config to calibrate)")
    src.add_argument("--model", help="a model id (no config: defaults for everything else)")
    ap.add_argument("--out", required=True, help="where to write the mjolnir.kv_scales/2 JSON")
    ap.add_argument("--download-dir", default=None)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-mem", type=float, default=None, help="gpu_memory_utilization (default: the config's)")
    ap.add_argument("--prompts", type=int, default=48, help="short prompts (cycled through the topic list)")
    ap.add_argument("--long", type=int, default=6, help="long-context prompts")
    ap.add_argument("--long-tokens", type=int, default=6000, help="approx. words per long prompt")
    ap.add_argument("--max-tokens", type=int, default=96, help="tokens generated per prompt")
    args = ap.parse_args(argv)

    os.environ["MJOLNIR_ATTN_KV_CALIBRATE"] = "1"
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # the accumulators live in this process
    import mjolnir
    from mjolnir import kvscales

    mjolnir.register()
    from vllm import LLM, SamplingParams

    kw = engine_kwargs_from_config(args.config) if args.config else {"model": args.model, "dtype": "bfloat16"}
    for k in ("served_model_name",):
        kw.pop(k, None)
    kw.update(
        kv_cache_dtype="fp8",  # calibrate on the fp8 path; the K/V are observed before quantization
        enforce_eager=True,
        max_model_len=min(int(kw.get("max_model_len") or args.max_model_len), args.max_model_len),
        seed=0,
    )
    kw["attention_backend"] = "MJOLNIR" if mjolnir.has_native_member() else "CUSTOM"
    kw.pop("attention_config", None)
    if args.gpu_mem is not None:
        kw["gpu_memory_utilization"] = args.gpu_mem
    if args.download_dir:
        kw["download_dir"] = args.download_dir
    print(f"calibrate: LLM({', '.join(f'{k}={v!r}' for k, v in sorted(kw.items()))})", flush=True)
    try:
        llm = LLM(**kw)
    except TypeError:
        backend = kw.pop("attention_backend")
        llm = LLM(attention_config={"backend": backend}, **kw)

    kvscales.reset()  # drop what the warm-up / profiling runs wrote
    prompts = [TOPICS[i % len(TOPICS)] for i in range(args.prompts)]
    budget = max(256, kw["max_model_len"] - args.max_tokens - 64)
    prompts += [long_prompt(min(args.long_tokens, budget * 3 // 4), i) for i in range(args.long)]
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=args.max_tokens, seed=0)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    toks = sum(len(o.prompt_token_ids) + len(o.outputs[0].token_ids) for o in outs)
    doc = kvscales.dump(args.out, model=str(kw.get("model", "")))
    if not doc["amax"]:
        print("calibrate: no K/V was observed — did the Mjolnir backend run?", file=sys.stderr)
        return 1
    print(f"calibrate: {len(prompts)} prompts, {toks} tokens in {time.time() - t0:.0f}s", flush=True)
    for name, e in doc["amax"].items():
        k_max, v_max = max(max(r) for r in e["k"]), max(max(r) for r in e["v"])
        k_med = sorted(x for r in e["k"] for x in r)[len(e["k"]) * len(e["k"][0]) // 2]
        print(
            f"  {name}: K amax {k_max:8.3f} (median channel {k_med:7.3f})  V amax {v_max:8.3f}"
            f"  ({len(e['k'])} heads x {len(e['k'][0])} channels)"
        )
    print(f"wrote {args.out}  ({len(doc['amax'])} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
