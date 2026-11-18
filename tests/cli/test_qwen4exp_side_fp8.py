"""Online FP8 for the Qwen4Exp target's BF16 side layers
(``mjolnir.qwen4exp.side_fp8``), against stand-ins for torch and vLLM.

* the target regex covers the seven side-layer kinds of ``model.layers.N``
  (the QSA indexer included, so vLLM 0.34 keeps it fused into qkv_proj as on
  0.33; never the drafter ``mtp.layers.N``, the router, the HC mixers or the
  experts), minus ``QWEN4EXP_SIDE_FP8_SKIP``: skipping qkv_proj also keeps the
  indexer BF16, skipping only the indexer keeps just it BF16 (C1); an unknown
  kind refuses;
* ``QWEN4EXP_SIDE_FP8`` composes the online target onto a ModelOpt
  MIXED_PRECISION config that leaves the side layers BF16, is a no-op for
  other configs or a checkpoint that already quantizes them (FP4-Local), and
  refuses next to ``--quantization-config``.
"""

from __future__ import annotations

import re
import types

import pytest
from qwen4exp_stubs import fresh, stand_ins

ENVS = ("QWEN4EXP_SIDE_FP8", "QWEN4EXP_SIDE_FP8_SKIP", "QWEN4EXP_FP8_FORMAT")


@pytest.fixture
def stubs(monkeypatch):
    yield from stand_ins(monkeypatch)


@pytest.fixture
def sf(stubs, monkeypatch):
    for name in ENVS:
        monkeypatch.delenv(name, raising=False)
    fresh("mjolnir.qwen4exp.draft_head")
    return fresh("mjolnir.qwen4exp.side_fp8")


def _hits(sf, name: str) -> bool:
    return bool(re.fullmatch(sf.side_fp8_target().removeprefix("re:"), name))


@pytest.mark.parametrize(
    ("name", "hit"),
    [
        ("model.layers.0.linear_attn.in_proj_qkvz", True),
        ("model.layers.0.linear_attn.out_proj", True),
        ("language_model.model.layers.3.self_attn.qkv_proj", True),
        ("model.layers.3.self_attn.o_proj", True),
        ("model.layers.7.mlp.shared_expert.gate_up_proj", True),
        ("model.layers.7.mlp.shared_expert.down_proj", True),
        ("model.layers.0.linear_attn.in_proj_ba", False),
        ("model.layers.7.mlp.gate", False),
        ("model.layers.7.mlp.experts", False),
        ("model.layers.7.attn_hyper_connection.input_mix_weight_up", False),
        ("mtp.layers.48.self_attn.qkv_proj", False),
        ("model.layers.3.self_attn.indexer.index_qk_proj", True),
        ("mtp.layers.48.self_attn.indexer.index_qk_proj", False),
    ],
)
def test_target_kinds(sf, name, hit):
    assert _hits(sf, name) is hit


def test_skip(sf, monkeypatch):
    # N3: the QSA qkv GEMM stays BF16, with the indexer it carried on 0.33
    monkeypatch.setenv("QWEN4EXP_SIDE_FP8_SKIP", "self_attn.qkv_proj")
    assert not _hits(sf, "model.layers.3.self_attn.qkv_proj")
    assert not _hits(sf, "model.layers.3.self_attn.indexer.index_qk_proj")
    assert _hits(sf, "model.layers.3.self_attn.o_proj")
    # C1: only the indexer stays BF16 (vLLM 0.34 then leaves it unfused)
    monkeypatch.setenv("QWEN4EXP_SIDE_FP8_SKIP", "self_attn.indexer.index_qk_proj")
    assert _hits(sf, "model.layers.3.self_attn.qkv_proj")
    assert not _hits(sf, "model.layers.3.self_attn.indexer.index_qk_proj")


def test_bad_skip_refuses(sf, monkeypatch):
    monkeypatch.setenv("QWEN4EXP_SIDE_FP8_SKIP", "self_attn.q_proj")
    with pytest.raises(ValueError, match="unknown kinds"):
        sf.side_fp8_target()


def _cfg(name="modelopt_mixed", quantized=None, online=None):
    return types.SimpleNamespace(
        get_name=lambda: name,
        quantized_layers=quantized or {"model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"}},
        online_quantization_config=online,
    )


def test_side_fp8_off_by_default(sf):
    cfg = _cfg()
    assert sf.maybe_quantize_side_layers(cfg) is False
    assert cfg.online_quantization_config is None


def test_side_fp8_composes_the_online_target(sf, monkeypatch):
    monkeypatch.setenv("QWEN4EXP_SIDE_FP8", "1")
    cfg = _cfg()
    assert sf.maybe_quantize_side_layers(cfg) is True
    (args,) = cfg.online_quantization_config.args
    assert args.kwargs == {"targets": {sf.side_fp8_target(): "fp8_per_block"}}


@pytest.mark.parametrize(
    "cfg",
    [
        _cfg(name="modelopt"),
        _cfg(quantized={"model.language_model.layers.0.linear_attn.out_proj": {"quant_algo": "MXFP8"}}),
    ],
)
def test_side_fp8_noop_cases(sf, monkeypatch, cfg):
    monkeypatch.setenv("QWEN4EXP_SIDE_FP8", "1")
    assert sf.maybe_quantize_side_layers(cfg) is False
    assert cfg.online_quantization_config is None


def test_side_fp8_refuses_next_to_quantization_config(sf, monkeypatch):
    monkeypatch.setenv("QWEN4EXP_SIDE_FP8", "1")
    with pytest.raises(ValueError, match="--quantization-config"):
        sf.maybe_quantize_side_layers(_cfg(online=object()))


@pytest.mark.parametrize(
    ("fmt", "n", "k", "want"), [("fp8_per_block", 256, 256, 65536 + 16), ("mxfp8", 64, 64, 4096 + 128)]
)
def test_fp8_bytes(sf, fmt, n, k, want):
    assert sf._fp8_bytes(fmt, n, k) == want
