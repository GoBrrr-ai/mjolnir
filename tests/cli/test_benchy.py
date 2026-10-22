"""The llama-benchy invocation: ``--warmup-runs`` must map onto what
llama-benchy 0.4.0 actually does (one discarded warmup run per cell unless
``--no-warmup``), so the history record's ``warmup_runs`` is true."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from mjolnir_cli import benchy
from mjolnir_cli.config import Settings


def _cmd(warmup_runs: int, runs: int = 6) -> list[str]:
    return benchy.build_command(
        ["llama-benchy"], Settings(), runs, warmup_runs, True, [0, 4096], [1, 2], 2048, 128, Path("out.json"), None
    )


def _runs(cmd: list[str]) -> int:
    return int(cmd[cmd.index("--runs") + 1])


def test_zero_warmups_passes_no_warmup():
    cmd = _cmd(0)
    assert "--no-warmup" in cmd
    assert _runs(cmd) == 6


def test_one_warmup_is_llama_benchys_own():
    cmd = _cmd(1)
    assert "--no-warmup" not in cmd
    assert _runs(cmd) == 6


@pytest.mark.parametrize("w", [0, 1])
def test_no_nonexistent_flag(w):
    assert "--warmup-runs" not in _cmd(w)  # llama-benchy 0.4.0 has no such flag


@pytest.mark.parametrize("w", [2, -1])
def test_unsupported_warmup_count_refused(w):
    with pytest.raises(benchy.BenchError, match="warmup"):
        _cmd(w)


def test_default_warmups_match_what_runs():
    assert inspect.signature(benchy.run_perf).parameters["warmup_runs"].default == 1
