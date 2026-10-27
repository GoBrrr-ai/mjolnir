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

import os

import pytest

from mjolnir_cli import docker_version


@pytest.fixture(autouse=True)
def _pinned_platform_env(monkeypatch):
    monkeypatch.setenv("MJOLNIR_PLATFORM", "thor")


@pytest.fixture(autouse=True)
def _skip_docker_version_check(monkeypatch):
    # No existing test may touch the real docker daemon or apt: skip the pin
    # check entirely and reset the once-per-process guard between tests. A
    # test that exercises the check itself sets the env var explicitly
    # (monkeypatch.setenv overrides this one).
    if os.environ.get("MJOLNIR_SKIP_DOCKER_CHECK") is None:
        monkeypatch.setenv("MJOLNIR_SKIP_DOCKER_CHECK", "1")
    docker_version.reset_check_for_tests()
    yield
    docker_version.reset_check_for_tests()
