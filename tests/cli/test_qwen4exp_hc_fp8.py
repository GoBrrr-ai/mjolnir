"""Online 8-bit hyperconnection GEMMs (``mjolnir.qwen4exp.hc_fp8``), against
stand-ins for torch and vLLM.

* off unless ``QWEN4EXP_HC_FP8`` is set: every HC Linear keeps
  ``quant_config=None`` (stock BF16);
* on, the target's down(+inject) and up projections and the final mixer get
  one shared online config with the per-kind formats; the drafter's mixers
  (``mtp.*``) and anything else stay ``None``;
* ``QWEN4EXP_HC_FP8_FORMAT`` takes ``<down>[,<up>]`` and refuses anything else.
"""

from __future__ import annotations

import pytest
from qwen4exp_stubs import fresh, stand_ins

DOWN = "language_model.model.layers.7.attn_hyper_connection.input_mix_weight_down_block_inject"
UP = "language_model.model.layers.7.mlp_hyper_connection.input_mix_weight_up"
MIXER = "language_model.model.hyper_connection_mixer.input_mix_weight_down"


@pytest.fixture
def stubs(monkeypatch):
    yield from stand_ins(monkeypatch)


@pytest.fixture
def hc(stubs, monkeypatch):
    for name in ("QWEN4EXP_HC_FP8", "QWEN4EXP_HC_FP8_FORMAT"):
        monkeypatch.delenv(name, raising=False)
    fresh("mjolnir.qwen4exp.draft_head")
    fresh("mjolnir.qwen4exp.side_fp8")
    return fresh("mjolnir.qwen4exp.hc_fp8")


def test_off_by_default(hc):
    assert hc.hc_fp8_quant_config(DOWN) is None


def test_targets_and_formats(hc, monkeypatch):
    monkeypatch.setenv("QWEN4EXP_HC_FP8", "1")
    cfg = hc.hc_fp8_quant_config(DOWN)
    assert cfg is not None and hc.hc_fp8_quant_config(UP) is cfg and hc.hc_fp8_quant_config(MIXER) is cfg
    (args,) = cfg.args
    assert args.kwargs == {"targets": {hc.HC_DOWN_TARGET: "fp8_per_block", hc.HC_UP_TARGET: "mxfp8"}}


@pytest.mark.parametrize(
    "prefix",
    [
        "mtp.layers.48.attn_hyper_connection.input_mix_weight_up",
        "mtp.hyper_connection_mixer.input_mix_weight_down",
        "language_model.model.layers.7.mlp.shared_expert.down_proj",
    ],
)
def test_drafter_and_others_stay_bf16(hc, monkeypatch, prefix):
    monkeypatch.setenv("QWEN4EXP_HC_FP8", "1")
    assert hc.hc_fp8_quant_config(prefix) is None


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("mxfp8", ("mxfp8", "mxfp8")),
        ("fp8_per_block, mxfp8", ("fp8_per_block", "mxfp8")),
        ("", ("fp8_per_block", "mxfp8")),
    ],
)
def test_format_parsing(hc, monkeypatch, raw, want):
    monkeypatch.setenv("QWEN4EXP_HC_FP8_FORMAT", raw)
    assert hc._hc_formats() == want


@pytest.mark.parametrize("raw", ["fp4", "mxfp8,mxfp8,mxfp8"])
def test_bad_format_refuses(hc, monkeypatch, raw):
    monkeypatch.setenv("QWEN4EXP_HC_FP8_FORMAT", raw)
    with pytest.raises(ValueError, match="QWEN4EXP_HC_FP8_FORMAT"):
        hc._hc_formats()
