"""Count which tokens the target model generates — the drafter's vocabulary
for VocabTrim (``MJOLNIR_DRAFT_VOCAB``, see ``mjolnir.lmhead``).

Runs the serving config in-process (vLLM ``LLM``), samples a varied prompt
set (several completions per prompt, temperature 0.7), counts the generated
token ids and writes them most frequent first, with the coverage — the share
of generated tokens inside the top-K — for a few K. VocabTrim found the
target's own generations the best source (arXiv 2506.22694).

    python -m mjolnir.draftvocab --config /configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3.yaml \\
        --out /out/draft_vocab.json --download-dir /data/models/huggingface

File: ``{"format": "mjolnir.draft_vocab/1", "model": ..., "generated_tokens": N,
"coverage": {"32768": 0.997, ...}, "tokens": [ids, most frequent first]}``.
Special tokens (EOS, the chat / think markers) are always included.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

FORMAT = "mjolnir.draft_vocab/1"
COVERAGE_K = (4096, 8192, 16384, 32768, 49152, 65536)

EXTRA_TOPICS = [
    "Write a detailed README for a command-line tool that benchmarks GPUs.",
    "Explain gradient descent to a high-school student, then give the math.",
    "Write TypeScript code for a debounced search box in React, with tests.",
    "Summarize the plot of a classic novel of your choice and analyze its themes.",
    "Give a step-by-step recipe for sourdough bread with timings.",
    "Compare PostgreSQL and SQLite for an embedded device; answer as a table and a recommendation.",
    "Write a formal email declining a meeting and proposing two alternative times.",
    "Debug this Python: `for i in range(10): print(i/0)` — explain the error and fix it.",
    "Describe the water cycle in Spanish, then in English.",
    "Write a C function that reverses a linked list in place; explain each line.",
    "List ten interview questions for a senior backend engineer with model answers.",
    "Explain how TLS 1.3 handshakes work, with a sequence of messages.",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="vllm serve YAML")
    src.add_argument("--model", help="a model id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--download-dir", default=None)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--prompts", type=int, default=96)
    ap.add_argument("--samples", type=int, default=2, help="completions per prompt")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--long", type=int, default=4, help="long-context prompts")
    args = ap.parse_args(argv)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    for k in ("MJOLNIR_DRAFT_VOCAB", "MJOLNIR_LM_HEAD"):  # count with the plain heads
        os.environ.pop(k, None)
    import mjolnir
    from mjolnir.calibrate import TOPICS, engine_kwargs_from_config, long_prompt

    mjolnir.register()
    from vllm import LLM, SamplingParams

    kw = engine_kwargs_from_config(args.config) if args.config else {"model": args.model, "dtype": "bfloat16"}
    kw.pop("served_model_name", None)
    kw["max_model_len"] = min(int(kw.get("max_model_len") or args.max_model_len), args.max_model_len)
    if args.download_dir:
        kw["download_dir"] = args.download_dir
    print(f"draftvocab: LLM({', '.join(f'{k}={v!r}' for k, v in sorted(kw.items()))})", flush=True)
    llm = LLM(**kw)
    topics = TOPICS + EXTRA_TOPICS
    prompts = [topics[i % len(topics)] for i in range(args.prompts)]
    prompts += [long_prompt(min(4000, kw["max_model_len"] // 2), i) for i in range(args.long)]
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=args.max_tokens, n=args.samples, seed=0)
    t0 = time.time()
    outs = llm.chat(msgs, sp)
    counts: Counter = Counter()
    for o in outs:
        for c in o.outputs:
            counts.update(c.token_ids)
    total = sum(counts.values())
    if not total:
        print("draftvocab: nothing generated", file=sys.stderr)
        return 1
    tok = llm.get_tokenizer()
    special = set(getattr(tok, "all_special_ids", []) or [])
    try:
        special |= set(tok.get_added_vocab().values())
    except Exception:  # noqa: BLE001
        pass
    ranked = [t for t in sorted(special) if t not in counts] + [t for t, _ in counts.most_common()]
    ranked_counts = [counts.get(t, 0) for t in ranked]
    coverage, acc = {}, 0
    for i, c in enumerate(ranked_counts, 1):
        acc += c
        for k in COVERAGE_K:
            if i == k:
                coverage[str(k)] = round(acc / total, 5)
    for k in COVERAGE_K:
        coverage.setdefault(str(k), 1.0)
    doc = {
        "format": FORMAT,
        "model": str(kw.get("model", "")),
        "generated_tokens": total,
        "distinct_tokens": len(counts),
        "coverage": coverage,
        "tokens": ranked,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f)
    print(
        f"draftvocab: {len(prompts)} prompts x {args.samples}, {total} tokens generated in {time.time() - t0:.0f}s, "
        f"{len(counts)} distinct"
    )
    for k in COVERAGE_K:
        print(f"  top {k:>6}: {coverage[str(k)] * 100:6.2f}% of generated tokens")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
