"""The screened target lm_head's switches and per-batch decision
(``mjolnir.qwen4exp.screened_head``), against stand-ins for torch and vLLM.

* off unless ``QWEN4EXP_LM_HEAD_SCREENED`` is set; refuses next to
  ``MJOLNIR_LM_HEAD=fp8`` (it replaces the BF16 head the full-head path
  reads) and outside 1..1024 for ``QWEN4EXP_LM_HEAD_SCREEN_K``;
* rows per call = ``max_num_seqs * (1 + K)``, capped at 64;
* a batch is screened only when every request is greedy or samples with
  ``top_k <= SCREEN_K`` and nothing else reads the logits (logits
  processors, logprobs, min_p); everything else takes the full head;
* the ``Sampler.add_request`` replacement keeps vLLM 0.34's
  ``needs_logits_processing`` / ``may_mask_logits`` and adds only the
  logits-processor split.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest
from qwen4exp_stubs import fresh, stand_ins


@pytest.fixture
def stubs(monkeypatch):
    yield from stand_ins(monkeypatch)


@pytest.fixture
def sh(stubs, monkeypatch):
    stubs["torch"].cuda = types.SimpleNamespace(is_available=lambda: True)
    stubs["vllm.envs"].VLLM_BATCH_INVARIANT = False
    for name in ("QWEN4EXP_LM_HEAD_SCREENED", "MJOLNIR_LM_HEAD", "QWEN4EXP_LM_HEAD_SCREEN_K"):
        monkeypatch.delenv(name, raising=False)
    return fresh("mjolnir.qwen4exp.screened_head")


def _config(seqs=1, k=3, lora=None, tp=1):
    return types.SimpleNamespace(
        lora_config=lora,
        parallel_config=types.SimpleNamespace(tensor_parallel_size=tp),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=k),
        scheduler_config=types.SimpleNamespace(max_num_seqs=seqs),
    )


def test_off_by_default(sh):
    assert sh.screened_head_settings(_config()) is None


@pytest.mark.parametrize(("seqs", "k", "rows"), [(1, 3, 4), (6, 3, 24), (20, 3, 64)])
def test_rows_per_call(sh, monkeypatch, seqs, k, rows):
    monkeypatch.setenv("QWEN4EXP_LM_HEAD_SCREENED", "1")
    assert sh.screened_head_settings(_config(seqs, k)) == (rows, 32)


def test_refuses_next_to_the_fp8_lm_head(sh, monkeypatch):
    monkeypatch.setenv("QWEN4EXP_LM_HEAD_SCREENED", "1")
    monkeypatch.setenv("MJOLNIR_LM_HEAD", "fp8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        sh.screened_head_settings(_config())
    monkeypatch.setenv("MJOLNIR_LM_HEAD", "bf16")  # nothing mjolnir.lmhead acts on
    assert sh.screened_head_settings(_config()) == (4, 32)


@pytest.mark.parametrize("k", ["0", "1025"])
def test_screen_k_range(sh, monkeypatch, k):
    monkeypatch.setenv("QWEN4EXP_LM_HEAD_SCREENED", "1")
    monkeypatch.setenv("QWEN4EXP_LM_HEAD_SCREEN_K", k)
    with pytest.raises(ValueError, match=r"1\.\.1024"):
        sh.screened_head_settings(_config())


def test_refuses_lora_and_tp(sh, monkeypatch):
    monkeypatch.setenv("QWEN4EXP_LM_HEAD_SCREENED", "1")
    with pytest.raises(ValueError, match="no LoRA"):
        sh.screened_head_settings(_config(lora=object()))
    with pytest.raises(ValueError, match="TP sharding"):
        sh.screened_head_settings(_config(tp=2))


def _arr(values):
    return types.SimpleNamespace(np=np.asarray(values))


def _sampler(temps, top_ks, min_ps=None, uses=None, logprobs=None):
    n = len(temps)
    return types.SimpleNamespace(
        sampling_states=types.SimpleNamespace(
            temperature=_arr(temps),
            top_k=_arr(top_ks),
            min_p=_arr(min_ps or [0.0] * n),
            vocab_size=248_320,
        ),
        uses_logits_processors=np.asarray(uses or [False] * n),
        get_logprobs_dims=lambda idx: logprobs,
    )


def _head(sh):
    head = object.__new__(sh.ScreenedLMHead)
    head.max_num_tokens, head.max_top_k, head.min_top_k = 8, 32, 1
    return head


@pytest.mark.parametrize(
    ("sampler", "rows", "want"),
    [
        (_sampler([0.0, 0.0], [248_320, 248_320]), 8, (1, None)),
        (_sampler([0.7, 0.0], [20, 248_320]), 8, (20, None)),
        (_sampler([0.7], [64]), 4, (None, "top_k_over_max")),
        (_sampler([0.7], [248_320]), 4, (None, "no_top_k")),
        (_sampler([0.7], [20], min_ps=[0.05]), 4, (None, "min_p")),
        (_sampler([0.0], [1], uses=[True]), 4, (None, "logits_processor")),
        (_sampler([0.0], [1], logprobs=5), 4, (None, "logprobs")),
        (_sampler([0.0], [1]), 9, (None, "rows_over_max")),
        (_sampler([0.0], [1]), 0, (None, "no_rows")),
    ],
)
def test_screen_decision(sh, sampler, rows, want):
    idx = np.arange(len(sampler.sampling_states.temperature.np))
    assert _head(sh).screen_decision(sampler, idx, rows) == want


def test_processor_reasons(sh):
    def sp(**kw):
        base = {
            "allowed_token_ids": None,
            "logit_bias": None,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
        }
        return types.SimpleNamespace(**(base | kw))

    def proc(name):
        return type(name, (), {})()

    assert sh._processor_reason(proc("LogitBiasState"), sp(allowed_token_ids=[1])) == "allowed_token_ids"
    assert sh._processor_reason(proc("LogitBiasState"), sp()) == "min_tokens"
    assert sh._processor_reason(proc("PenaltiesState"), sp(repetition_penalty=1.1)) == "repetition_penalty"
    assert sh._processor_reason(proc("PenaltiesState"), sp()) == "presence_penalty"
    assert sh._processor_reason(proc("BadWordsState"), sp()) == "bad_words"
    assert sh._processor_reason(proc("Other"), sp()) == "custom_processor"


def test_add_request_keeps_upstream_flags(sh, stubs):
    class Sampler:
        pass

    class GPUModelRunner:
        def sample(self, hidden_states, input_batch, grammar_output):
            return None

    stubs["vllm.v1.worker.gpu.sample.sampler"].Sampler = Sampler
    stubs["vllm.v1.worker.gpu.model_runner"].GPUModelRunner = GPUModelRunner
    sh._install_hooks()
    # per request: (sampling states need processing, top-k/top-p/min-p can mask, thinking budget)
    cases = [(True, False, False), (True, True, False), (False, False, True), (False, False, False)]
    s: Any = Sampler()
    s.needs_logits_processing = np.zeros(len(cases), dtype=bool)
    s.may_mask_logits = np.zeros(len(cases), dtype=bool)
    s.sampling_states = types.SimpleNamespace(add_request=lambda i, p: cases[i][0], masks_logits=lambda i: cases[i][1])
    s.thinking_budget_state = types.SimpleNamespace(add_request=lambda i, p: cases[i][2])
    s.logits_processors = []
    s.logprob_token_ids_state = types.SimpleNamespace(add_request=lambda i, p: None)
    s.trace_replay_state = None
    for i in range(len(cases)):
        s.add_request(i, None)
    # as vLLM 0.34's Sampler.add_request: a request may mask when top-k / top-p /
    # min-p, the thinking budget or a processor can; masking implies processing
    assert s.needs_logits_processing.tolist() == [True, True, True, False]
    assert s.may_mask_logits.tolist() == [False, True, True, False]
    # the #60566 split: only thinking budget / processors block the screen
    assert s.uses_logits_processors.tolist() == [False, False, True, False]
