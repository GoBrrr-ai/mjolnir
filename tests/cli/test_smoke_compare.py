"""The vLLM smoke test's verdict logic (tests/kernel/smoke_vllm.py) and the
`verify vllm` stage's parsing of its log — on synthetic runs, no GPU."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mjolnir_cli.verify.stages import parse_smoke, smoke_summary

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("smoke_vllm", ROOT / "tests" / "kernel" / "smoke_vllm.py")
assert _spec is not None and _spec.loader is not None
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)

ARGS = SimpleNamespace(tie_nats=1.0, max_mean_dlp=0.15, max_dlp=1.0, min_prefix=0, min_match=0.0)


def lp(**probs) -> dict[str, float]:
    """{"<id>": logprob} from token=prob keywords (t7=0.6 → {"7": ln 0.6})."""
    return {k[1:]: math.log(v) for k, v in probs.items()}


def test_identical_runs_pass():
    toks = [1, 2, 3]
    steps = [lp(t1=0.9, t5=0.1), lp(t2=0.8, t6=0.2), lp(t3=0.7, t4=0.3)]
    r = smoke.compare_prompt(toks, toks, steps, steps, 1.0)
    assert r == {"prefix": 3, "len": 3, "deltas": [0.0, 0.0, 0.0], "diverged": False}
    ok, s = smoke.judge([r], {"t5_steps": 10}, ARGS)
    assert ok and s["mean_match"] == 1.0 and s["diverged"] == 0


def test_near_tie_divergence_passes_despite_short_prefix():
    # step 1: the reference picks 2 (0.51) over 6 (0.49); Mjolnir the other way
    ref = [lp(t1=0.9, t5=0.1), lp(t2=0.51, t6=0.49)]
    t5 = [lp(t1=0.89, t5=0.11), lp(t2=0.49, t6=0.51)]
    r = smoke.compare_prompt([1, 2], [1, 6], ref, t5, 1.0)
    assert r["prefix"] == 1 and r["diverged"] and r["tie"] and r["gap"] < 0.1
    ok, s = smoke.judge([r], {"t5_steps": 10}, ARGS)
    assert ok, s["why"]
    assert s["mean_match"] == 0.5 and s["ties"] == 1


def test_divergence_onto_an_unconsidered_token_fails():
    ref = [lp(t1=0.99, t5=0.01)]
    t5 = [lp(t9=0.99, t5=0.01)]  # Mjolnir picks 9, which the reference never ranked
    r = smoke.compare_prompt([1], [9], ref, t5, 1.0)
    assert r["diverged"] and r["tie"] is False
    ok, s = smoke.judge([r], {"t5_steps": 10}, ARGS)
    assert not ok and "not near-ties" in s["why"][0]


def test_clear_preference_flip_fails():
    ref = [lp(t1=0.95, t2=0.05)]  # 2.9 nats apart
    t5 = [lp(t1=0.05, t2=0.95)]
    r = smoke.compare_prompt([1], [2], ref, t5, 1.0)
    assert r["tie"] is False and r["gap"] > 2.5


def test_logprob_drift_on_shared_prefix_fails():
    ref = [lp(t1=0.9, t2=0.1)] * 4
    t5 = [lp(t1=0.4, t2=0.35)] * 4  # same argmax, 0.81 nats lower every step
    r = smoke.compare_prompt([1] * 4, [1] * 4, ref, t5, 1.0)
    ok, s = smoke.judge([r], {"t5_steps": 10}, ARGS)
    assert not ok and "mean |Δlogprob|" in s["why"][0]


def test_no_decode_steps_fails():
    r = smoke.compare_prompt([1], [1], [lp(t1=1.0)], [lp(t1=1.0)], 1.0)
    ok, s = smoke.judge([r], {"t5_steps": 0}, ARGS)
    assert not ok and "never ran" in s["why"][0]


def test_without_logprobs_the_token_criterion_applies():
    short = smoke.compare_prompt([1, 2, 3], [1, 9, 3], None, None, 1.0)
    ok, s = smoke.judge([short], {"t5_steps": 10}, ARGS)
    assert not ok and not s["logprobs"] and "token criterion" in s["why"][0]
    full = smoke.compare_prompt(list(range(10)), list(range(10)), None, None, 1.0)
    assert smoke.judge([full], {"t5_steps": 10}, ARGS)[0]


def test_main_output_round_trips_through_the_stage_parser(monkeypatch, capsys):
    n = len(smoke.PROMPTS)
    steps = [lp(t1=0.6, t2=0.4), lp(t3=0.55, t4=0.45)]
    ref = {"tokens": [[1, 3]] * n, "text": ["a b"] * n, "logprobs": [steps] * n, "seconds": 1.0}
    t5 = dict(ref, tokens=[[1, 4]] + [[1, 3]] * (n - 1), stats={"t5_steps": 5, "fallback_steps": 1})
    t5["logprobs"] = [[lp(t1=0.6, t2=0.4), lp(t3=0.45, t4=0.55)]] + [steps] * (n - 1)
    outs = iter([ref, t5])
    monkeypatch.setattr(smoke, "run_child", lambda *a: next(outs))
    monkeypatch.setattr(sys, "argv", ["smoke_vllm.py"])
    with pytest.raises(SystemExit) as e:
        smoke.main()
    text = capsys.readouterr().out
    assert e.value.code == 0 and text.rstrip().endswith("PASS")
    c = parse_smoke(text)
    assert c["divergences"] == 1 and c["near_ties"] == 1
    assert c["mean_dlp"] == 0.0 and c["worst_prefix"] == 1
    assert c["mean_match"] == pytest.approx((0.5 + n - 1) / n, abs=0.01)
    assert "near-ties 1/1" in smoke_summary(c)


def test_parse_old_log_format():
    c = parse_smoke("  mean match 0.43, worst prefix 5, time ref 1.0s\nFAIL\n")
    assert c["mean_match"] == 0.43 and c["worst_prefix"] == 5 and c["mean_dlp"] is None
    assert smoke_summary(c) == "  match 0.43"
