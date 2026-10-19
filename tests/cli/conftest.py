"""Shared CLI-suite configuration.

The CLI is detection-first: with no ``MJOLNIR_PLATFORM`` every command
probes the host (nvidia-smi, device-tree, DMI). CI has no GPU and unit
tests must never touch real ``/proc`` / ``/sys`` / ``nvidia-smi`` — so the
suite default pins an explicit platform, keeping every test hermetic. A
test that exercises the detection path opts out: clear the env var (or
pass ``env={"MJOLNIR_PLATFORM": None}`` to ``CliRunner.invoke``) and
install the fakes from :mod:`tests.cli.platform_fixtures`.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _pinned_platform_env(monkeypatch):
    monkeypatch.setenv("MJOLNIR_PLATFORM", "thor")
