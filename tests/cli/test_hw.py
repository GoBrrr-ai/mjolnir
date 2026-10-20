"""``mjolnir hw`` dispatch (cycle 2): the detected platform decides what
"hardware setup" even means.

The spec (plan §0.6, detection injected so tests never touch real probes):

* thor detected -> the existing ``scripts/hw/setup-thor.sh`` path runs
  (unchanged behaviour, now behind the detection);
* orin detected -> the same script runs, with an explicit "untested on
  this platform" note first;
* spark detected -> exit 2, naming what exists instead (the thor setup) —
  research platforms get an explicit refusal, not the wrong script;
* an unknown host -> the probe instructions, like every other command.

All tests fake the platform (env pin or ``platform_fixtures.install``):
no real ``/proc``, ``/sys`` or ``nvidia-smi`` contact, and
``thor.run_setup`` is monkeypatched so nothing ever executes the script.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from mjolnir_cli import cli
from platform_fixtures import install


@pytest.fixture
def no_script(monkeypatch):
    """Never run the real setup script; record the calls instead."""
    calls: list[dict] = []
    monkeypatch.setattr("mjolnir_cli.thor.run_setup", lambda **kw: calls.append(kw) or 0)
    return calls


def test_hw_setup_on_a_detected_thor_runs_the_thor_script(no_script):
    r = CliRunner().invoke(cli.app, ["hw", "setup", "--dry-run"], env={"MJOLNIR_PLATFORM": "thor"})
    assert r.exit_code == 0, r.output
    assert len(no_script) == 1  # the thor path, unchanged


def test_hw_setup_on_a_detected_spark_refuses_and_names_what_exists(no_script):
    r = CliRunner().invoke(cli.app, ["hw", "setup", "--dry-run"], env={"MJOLNIR_PLATFORM": "spark"})
    assert r.exit_code == 2, r.output
    assert no_script == []  # the thor script must not run on a spark
    assert "thor" in r.output  # the message lists what does exist


def test_hw_setup_on_a_detected_orin_runs_with_an_untested_note(no_script):
    r = CliRunner().invoke(cli.app, ["hw", "setup", "--dry-run"], env={"MJOLNIR_PLATFORM": "orin"})
    assert r.exit_code == 0, r.output
    assert len(no_script) == 1  # still runnable: the thor script is the jetson path
    assert "untested" in r.output.lower()


def test_hw_setup_on_an_unknown_host_fails_with_the_probe_instructions(no_script, monkeypatch):
    install(monkeypatch, smi=None, jetson=None, dmi=None)
    r = CliRunner().invoke(cli.app, ["hw", "setup", "--dry-run"], env={"MJOLNIR_PLATFORM": None})
    assert r.exit_code != 0, r.output
    assert no_script == []
    assert "nvidia-smi" in r.output and "MJOLNIR_PLATFORM" in r.output


def test_hw_setup_dispatches_on_detection_not_on_the_env_alone(no_script, monkeypatch):
    # the env is just the top of the precedence: with no env at all, the
    # detected host decides (here: a spark, refused)
    install(monkeypatch, smi="NVIDIA GB10, 12.1")
    r = CliRunner().invoke(cli.app, ["hw", "setup", "--dry-run"], env={"MJOLNIR_PLATFORM": None})
    assert r.exit_code == 2, r.output
    assert no_script == []
