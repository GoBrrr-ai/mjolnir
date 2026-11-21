"""The Qwen4Exp reduced draft vocabulary and 8-bit drafter switches
(``mjolnir.qwen4exp.draft_head``), against stand-ins for torch and vLLM.

* the vocabulary comes from the ``mjolnir.draft_vocab/1`` file named by
  ``MJOLNIR_DRAFT_VOCAB`` (``mjolnir.lmhead.load_vocab``, so
  ``MJOLNIR_DRAFT_VOCAB_SIZE`` keeps the first N), as sorted unique ids topped
  up with the lowest unused ids to a multiple of 128; unset means no
  reduced head;
* ``QWEN4EXP_MTP_DENSE_FP8`` composes one online target onto the drafter's
  quant config, and its regex matches the drafter's dense projections only;
* ``mjolnir.lmhead`` leaves a drafter that already carries its own reduced
  head alone (it would otherwise trim it a second time).
"""

from __future__ import annotations

import json
import re
import types

import numpy as np
import pytest
from qwen4exp_stubs import fresh, stand_ins

VOCAB = 248_077


@pytest.fixture
def stubs(monkeypatch):
    yield from stand_ins(monkeypatch)


@pytest.fixture
def dh(stubs, monkeypatch):
    torch = stubs["torch"]
    torch.as_tensor = lambda ids, device=None: np.asarray(ids)
    torch.get_default_device = lambda: "cpu"
    for name in ("MJOLNIR_DRAFT_VOCAB", "MJOLNIR_DRAFT_VOCAB_SIZE", "QWEN4EXP_MTP_DENSE_FP8", "QWEN4EXP_FP8_FORMAT"):
        monkeypatch.delenv(name, raising=False)
    return fresh("mjolnir.qwen4exp.draft_head")


def _vocab_file(tmp_path, tokens) -> str:
    path = tmp_path / "draft_vocab.json"
    path.write_text(json.dumps({"format": "mjolnir.draft_vocab/1", "tokens": [int(t) for t in tokens]}))
    return str(path)


def test_pad_tops_up_to_a_multiple_of_128_with_the_lowest_unused_ids(dh):
    ids = np.array([7, 3, 3, 1000, 5, 2])
    out = dh.pad_draft_vocab(ids, VOCAB)
    assert len(out) == 128
    assert list(out) == sorted(set(out))
    assert {2, 3, 5, 7, 1000} <= set(out.tolist())
    assert [i for i in out.tolist() if i not in (2, 3, 5, 7, 1000)] == [i for i in range(200) if i not in (2, 3, 5, 7)][
        :123
    ]


def test_an_aligned_set_is_kept_as_is(dh):
    ids = np.arange(256)[::-1]
    assert dh.pad_draft_vocab(ids, VOCAB).tolist() == list(range(256))


def test_no_file_no_reduced_head(dh):
    assert dh.load_draft_vocab(VOCAB) is None


def test_the_draft_vocab_file_and_size(dh, tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    tokens = rng.choice(VOCAB, 1000, replace=False)
    monkeypatch.setenv("MJOLNIR_DRAFT_VOCAB", _vocab_file(tmp_path, tokens))
    full = dh.load_draft_vocab(VOCAB)
    assert len(full) == 1024 and set(tokens.tolist()) <= set(full.tolist())
    monkeypatch.setenv("MJOLNIR_DRAFT_VOCAB_SIZE", "500")  # the file's first 500, padded
    part = dh.load_draft_vocab(VOCAB)
    assert len(part) == 512 and set(tokens[:500].tolist()) <= set(part.tolist())


def test_the_shipped_vocabulary_is_the_measured_47232_rows(dh, monkeypatch):
    """configs/nvidia/Qwen3.8-Flash-Next/draft_vocab.json holds the 47,149-id
    coding vocabulary; padded it is the 47,232-row draft head measured."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "configs/nvidia/Qwen3.8-Flash-Next/draft_vocab.json"
    if not path.exists():
        pytest.skip("no Qwen3.8-Flash-Next draft vocabulary in this checkout")
    monkeypatch.setenv("MJOLNIR_DRAFT_VOCAB", str(path))
    ids = dh.load_draft_vocab(VOCAB)
    assert len(ids) == 47_232
    tokens = json.loads(path.read_text())["tokens"]
    assert len(tokens) == 47_149 == len(set(tokens))
    added = sorted(set(ids.tolist()) - set(tokens))
    assert (added[0], added[-1]) == (36454, 36564)
    kept = set(tokens)
    assert added == [i for i in range(VOCAB) if i not in kept][:83]


def test_mtp_dense_fp8_composes_one_online_target(dh, monkeypatch):
    cfg = types.SimpleNamespace(online_quantization_config=None)
    dh.maybe_quantize_mtp_dense(cfg)
    assert cfg.online_quantization_config is None  # off by default
    monkeypatch.setenv("QWEN4EXP_MTP_DENSE_FP8", "1")
    dh.maybe_quantize_mtp_dense(cfg)
    (args,) = cfg.online_quantization_config.args
    assert args.kwargs == {"targets": {dh.MTP_DENSE_TARGET: "fp8_per_block"}}


def test_unknown_fp8_format_refuses(dh, monkeypatch):
    monkeypatch.setenv("QWEN4EXP_MTP_DENSE_FP8", "1")
    monkeypatch.setenv("QWEN4EXP_FP8_FORMAT", "fp4")
    with pytest.raises(ValueError, match="QWEN4EXP_FP8_FORMAT"):
        dh.maybe_quantize_mtp_dense(types.SimpleNamespace(online_quantization_config=None))


@pytest.mark.parametrize(
    ("name", "hit"),
    [
        ("mtp.fc_embedding", True),
        ("mtp.fc_hidden", True),
        ("mtp.layers.48.self_attn.qkv_proj", True),
        ("mtp.layers.48.self_attn.indexer.index_qk_proj", True),
        ("mtp.layers.48.self_attn.o_proj", True),
        ("mtp.layers.48.mlp.shared_expert.gate_up_proj", True),
        ("mtp.layers.48.mlp.shared_expert.down_proj", True),
        ("mtp.layers.48.mlp.gate", False),
        ("mtp.layers.48.mlp.experts", False),
        ("model.layers.3.self_attn.qkv_proj", False),
        ("mtp.hyper_connection_mixer.input_mix_weight_up", False),
    ],
)
def test_mtp_dense_target(dh, name, hit):
    assert bool(re.fullmatch(dh.MTP_DENSE_TARGET.removeprefix("re:"), name)) is hit


def test_lmhead_leaves_a_self_trimmed_drafter_alone(stubs, monkeypatch, tmp_path):
    stubs["torch"].cuda = types.SimpleNamespace(is_available=lambda: False)
    lmhead = fresh("mjolnir.lmhead")
    monkeypatch.setenv("MJOLNIR_DRAFT_VOCAB", _vocab_file(tmp_path, range(10)))
    for name in ("MJOLNIR_LM_HEAD", "MJOLNIR_MTP_FP8", "MJOLNIR_MTP_LAYER"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    monkeypatch.setattr(lmhead, "install_draft_vocab", lambda *a, **k: calls.append(a))
    own = types.SimpleNamespace(draft_vocab_ids=np.arange(128))
    lmhead.after_load(types.SimpleNamespace(model=None, speculator=types.SimpleNamespace(model=own)))
    assert calls == []
    plain = types.SimpleNamespace(draft_vocab_ids=None)
    lmhead.after_load(types.SimpleNamespace(model=None, speculator=types.SimpleNamespace(model=plain)))
    assert len(calls) == 1
