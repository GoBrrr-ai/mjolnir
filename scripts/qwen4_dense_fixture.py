"""Generate a tiny DENSE qwen4_exp fixture checkpoint (random weights).

The serve-smoke fixture for ``thor-qwen4-dense-model.patch``: a real
qwen4_exp checkpoint — GDN + QSA layers, hyper-connections, the indexer
cache — with every MoE block replaced by the dense ``mlp.gate_up_proj`` /
``mlp.down_proj`` pair the patched vLLM builds (the patch's
``mlp_only_layers`` rule). It exercises the whole serving path on sm_110
WITHOUT the released model: config validation, the dense branch, the
indexer pool allocation, the stock Triton QSA kernels (the canary), and
the ``hc_down_silu`` CuTe JIT.

Requires torch + transformers (>=5.16, qwen4_exp) — run inside the vLLM
image or any torch env; needs network only for ``--tokenizer``.

    python scripts/qwen4_dense_fixture.py --out /tmp/qwen4-fixture
    # then: vllm serve /tmp/qwen4-fixture --enforce-eager -tp 1 ...

Weights are random — the smoke asserts finite logits + greedy
determinism, never quality. NOTE: HF has no dense decoder, so the MLP
tensor swap + the config.json rewrite are the load-bearing steps; if the
first run trips a weight-name assert, diff the loader's expected names
against the state dict and adjust the two marked blocks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen4ExpForCausalLM, Qwen4ExpTextConfig

# layer schedule: GDN, QSA, GDN, QSA (the family's 1-in-4 cadence, tiny)
LAYER_TYPES = ["linear_attention", "indexed_attention", "linear_attention", "indexed_attention"]


def build_config(args: argparse.Namespace) -> dict:
    return {
        "model_type": "qwen4_exp_text",
        "hidden_size": args.hidden,
        "num_hidden_layers": args.layers,
        "layer_types": (LAYER_TYPES * ((args.layers + 3) // 4))[: args.layers],
        "head_dim": 256,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "num_experts": 8,  # >0: transformers rejects 0; unused (all layers dense)
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 128,
        "mlp_only_layers": list(range(args.layers)),  # THE dense marker
        "intermediate_size": args.ffn,  # THE dense FFN width
        "decoder_sparse_step": 4,
        "hc_count": 4,
        "hc_lowrank": max(16, args.hidden // 4),
        "ple_layer_ids": [],
        "mtp_num_hidden_layers": 1,
        "output_gate_type": "sigmoid",
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "max_position_embeddings": 8192,
        "vocab_size": 248320,
        "hidden_act": "silu",
    }


def dense_swap(state_dict: dict, layers: int, hidden: int, ffn: int) -> dict:
    """Replace every layer's MoE tensors with the dense pair vLLM builds.
    MARKED BLOCK: if the loader trips on a name here, the qwen4_exp HF
    naming drifted from Qwen3NextMLP's — adjust prefixes only."""
    drop = []
    for i in range(layers):
        pre = f"model.layers.{i}.mlp."
        drop += [
            k
            for k in state_dict
            if k.startswith((f"{pre}experts", f"{pre}shared_expert")) or k in (f"{pre}gate", f"{pre}gate.weight")
        ]
    for k in drop:
        del state_dict[k]
    g = torch.Generator().manual_seed(0)
    for i in range(layers):
        pre = f"model.layers.{i}.mlp."
        state_dict[f"{pre}gate_up_proj.weight"] = (torch.randn(2 * ffn, hidden, generator=g) * 0.02).to(torch.bfloat16)
        state_dict[f"{pre}down_proj.weight"] = (torch.randn(hidden, ffn, generator=g) * 0.02).to(torch.bfloat16)
    return state_dict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B", help="any Qwen3.8-family tokenizer (vocab 248320)")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--ffn", type=int, default=1024)
    args = p.parse_args()

    text = build_config(args)
    # transformers 5.19: the composite Qwen4ExpConfig does NOT forward attrs
    # to text_config (no heterogeneity proxy) — the causal LM must be built
    # from the TEXT config; the composite shape goes only into config.json
    # (vLLM reads hf_text_config = hf_config.text_config).
    cfg_text = Qwen4ExpTextConfig(**text)  # validates: dense scenario C
    print("config validation OK; initializing (random, CPU)…")
    model = Qwen4ExpForCausalLM(cfg_text)
    model = model.to(torch.bfloat16)
    model.eval()

    sd = dense_swap(dict(model.state_dict()), args.layers, args.hidden, args.ffn)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, state_dict=sd, safe_serialization=True)

    # pin config.json to the COMPOSITE shape vLLM resolves (hf_text_config =
    # text_config) with the dense markers guaranteed present
    cfg_dict = {
        "model_type": "qwen4_exp",
        "text_config": text,
        "architectures": ["Qwen4ExpForCausalLM"],
    }
    (args.out / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    tok_dir = Path(args.tokenizer)
    if tok_dir.is_dir():
        shutil.copytree(
        tok_dir,
        args.out,
        dirs_exist_ok=True,
        # config.json: the composite shape written above is load-bearing — a
        # full checkout passed as --tokenizer must not clobber it
        ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "config.json"),
    )
    else:
        AutoTokenizer.from_pretrained(args.tokenizer).save_pretrained(args.out)
    n = sum(1 for _ in args.out.rglob("*.safetensors"))
    print(f"fixture written: {args.out} ({n} safetensors shards) — serve it with the patched image")


if __name__ == "__main__":
    main()
