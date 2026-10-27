"""The health wait is progress-aware: a boot that is visibly advancing
(cold inductor/CUDA-graph compiles ran 34 minutes on orin2's first boot —
past the default window) keeps the wait alive, while a silent or exited
container fails fast."""

import urllib.error

import pytest

from mjolnir_cli import dockerctl
from mjolnir_cli.config import Settings


def _settings() -> Settings:
    return Settings(port=6001)


class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_clock(monkeypatch):
    t = [0.0]
    monkeypatch.setattr(dockerctl.time, "time", lambda: t[0])
    monkeypatch.setattr(dockerctl.time, "sleep", lambda s: t.__setitem__(0, t[0] + s))
    return t


def _patch_board(monkeypatch, size: list[int | None], healthy_at: float, die_at: float | None = None):
    """size: the current log size (mutated by the test); healthy_at: the fake
    clock time the server answers; die_at: when the container exits."""
    t = _patch_clock(monkeypatch)

    def urlopen(url, timeout=5):
        if t[0] >= healthy_at:
            return _Resp()
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(dockerctl.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dockerctl, "_log_size", lambda: size[0])
    monkeypatch.setattr(dockerctl, "container_exists", lambda: True)
    monkeypatch.setattr(dockerctl, "container_running", lambda: die_at is None or t[0] < die_at)
    return t, size


def test_progressing_boot_survives_past_the_window(monkeypatch):
    # the log advances every poll: the wait lives to 1.5x the timeout
    t, size = _patch_board(monkeypatch, size=[100], healthy_at=150.0)

    def grow():
        size[0] = size[0] + 1000 if t[0] < 150 else size[0]

    monkeypatch.setattr(dockerctl, "_log_size", lambda: (grow(), size[0])[1])
    dockerctl.wait_healthy(_settings(), timeout_s=100.0)  # no raise: healthy at 150s


def test_silent_boot_fails_at_the_window(monkeypatch):
    t, _ = _patch_board(monkeypatch, size=[100], healthy_at=10_000.0)  # never healthy
    with pytest.raises(TimeoutError, match="no log progress"):
        dockerctl.wait_healthy(_settings(), timeout_s=100.0)
    assert 100 <= t[0] <= 130  # dies at the window, not 4x


def test_progressing_boot_is_capped_at_4x(monkeypatch):
    t, size = _patch_board(monkeypatch, size=[100], healthy_at=10_000.0)
    monkeypatch.setattr(dockerctl, "_log_size", lambda: size[0] + int(t[0]))  # always advancing
    with pytest.raises(TimeoutError):
        dockerctl.wait_healthy(_settings(), timeout_s=100.0)
    assert t[0] < 500  # the 4x cap, not forever


def test_exited_container_fails_immediately(monkeypatch):
    t, _ = _patch_board(monkeypatch, size=[100], healthy_at=10_000.0, die_at=15.0)
    with pytest.raises(RuntimeError, match="exited before becoming healthy"):
        dockerctl.wait_healthy(_settings(), timeout_s=1800.0)
    assert t[0] <= 30
