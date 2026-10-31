"""Release-day shape lock for the Qwen4-27B checkpoint.

Run against the released ``config.json`` (a local path, or a HF repo id
fetched from the hub) the day weights land; prints the checklist from
``docs/faq/qwen4-27b.md`` and exits non-zero when a load-blocking field is
missing or wrong. Stdlib only — no torch/transformers, so it runs anywhere
(the transformers-level validation lives in the research dumps and is
re-runnable with ``uv run --with transformers``).

    python scripts/qwen4_shape_lock.py Qwen/Qwen4-27B
    python scripts/qwen4_shape_lock.py /path/to/config.json
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

# (json path in the text config, question, required — absence fails)
CHECKS = [
    ("model_type", "qwen4_exp -> the dense patch applies; a NEW type -> registration needed first", True),
    (
        "num_hidden_layers",
        "layer count — the KV/param ledger scales with it (research: unknown, arithmetic favors 64)",
        True,
    ),
    ("hidden_size", "hidden size (leak: 5120)", True),
    ("num_attention_heads", "main GQA Q heads (leak: 24) — the Mjolnir kernel shape", True),
    ("num_key_value_heads", "main GQA KV heads (leak: 4) — the Mjolnir kernel shape", True),
    ("head_dim", "head dim (leak: 256) — must be a multiple of 128", True),
    ("hc_count", "hyper-connection streams (leak: 4) — REQUIRED by the family model code", True),
    ("hc_lowrank", "HC low rank (leak: 640 TP1 / 320 TP2 — community tables under-report by the TP factor)", True),
    ("mtp_num_hidden_layers", "MTP drafter depth (family: 1; geometry unknown)", True),
    ("hidden_act", "must be silu for the dense Qwen3NextMLP path", True),
    ("vocab_size", "tokenizer size (Qwen3.8-family: 248320)", True),
    # dense-specific: required when the checkpoint is dense (num_experts > 0
    # + mlp_only_layers); the verdict line calls this out
    ("intermediate_size", "dense FFN width (leak: 14336) — required by thor-qwen4-dense-model.patch", False),
    ("mlp_only_layers", "dense marker: every layer listed -> all-dense under the patch", False),
    ("num_experts", ">0 keeps transformers' validation happy even for dense; 0 is rejected upstream", False),
    ("indexer_n_heads", "QSA indexer present? absent/null -> plain full attention (use_qsa=False); 4 -> QSA", False),
    ("indexer_head_dim", "indexer head dim (family: 128)", False),
    ("indexer_budget", "selection budget in tokens (family: 2048 = 512 blocks)", False),
    ("indexer_compress_ratio", "tokens per compressed slot (family: 4 — the MTP draft cap)", False),
    ("layer_types", "the GDN/full schedule (3:1 assumed, evidenced nowhere)", False),
    ("full_attention_interval", "the 1-in-4 full-attention cadence fallback", False),
    ("decoder_sparse_step", "MoE cadence for hybrid dense/MoE (patch defaults it to 1)", False),
    ("ple_layer_ids", "PLE n-gram layers: non-empty -> ~95 GiB host table + offload planning", False),
    ("ple_embed_dim", "PLE width (family: 2560)", False),
    ("max_position_embeddings", "native context (family: 262144)", False),
    ("quantization_config", "quant format of the release — memory ledger + backend selection", False),
]

# fields whose ABSENCE is meaningful but not a failure (the patch/code defaults them)
ABSENT_OK = {
    "indexer_n_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
    "layer_types",
    "full_attention_interval",
    "decoder_sparse_step",
    "ple_layer_ids",
    "ple_embed_dim",
    "max_position_embeddings",
    "quantization_config",
    "mlp_only_layers",
}


def load_config(src: str) -> tuple[dict, str]:
    p = Path(src)
    if p.is_file():
        return json.loads(p.read_text()), str(p)
    url = f"https://huggingface.co/{src.rstrip('/')}/raw/main/config.json"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read()), url


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    cfg, src = load_config(sys.argv[1])
    text = cfg.get("text_config", cfg)  # nested (qwen4_exp) or flat
    print(f"# shape lock: {src}\n")
    failed = False
    for path, question, required in CHECKS:
        val = text.get(path, cfg.get(path) if path == "model_type" else None)
        if val is None:
            if required and path not in ABSENT_OK:
                print(f"  [MISSING] {path:<26} = —                     {question}")
                failed = True
            else:
                print(f"  [absent ] {path:<26} = —                     {question}")
            continue
        state = val if isinstance(val, str) else json.dumps(val)
        mark = "ok"
        if path == "head_dim" and (not isinstance(val, int) or val % 128 != 0):
            mark, failed = "BAD", True
        if path == "hidden_act" and val != "silu":
            mark, failed = "BAD", True
        if path == "num_key_value_heads" and isinstance(val, int):
            qh = text.get("num_attention_heads")
            if isinstance(qh, int) and (qh % val != 0 or text.get("head_dim", 256) % 128 != 0):
                mark, failed = "BAD", True
        print(f"  [{mark:>7}] {path:<26} = {state:<24} {question}")
    dense = text.get("mlp_only_layers") is not None and text.get("intermediate_size") is not None
    print(
        "\nverdict:",
        (
            "FAIL — fix the MISSING/BAD fields before serving"
            if failed
            else "dense-ready — fill the memory ledger + serving config (docs/faq/qwen4-27b.md)"
            if dense
            else "fields present; DENSE markers (mlp_only_layers/intermediate_size) absent — "
            "check whether the release is MoE or ships them differently"
        ),
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
