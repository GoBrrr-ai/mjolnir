"""``mjolnir hw`` dispatch (cycle 2): the detected platform decides what
"hardware setup" even means.

The spec (plan §0.6, detection injected so tests never touch real probes):

* thor detected -> the existing ``scripts/hw/setup-jetson.sh`` path runs
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


def test_hw_setup_on_a_detected_orin_runs_like_thor(no_script):
    # the setup is the Jetson family path: it detects the box (the banner
    # reports SoM/Memory/Model from the device-tree) and runs the same script
    r = CliRunner().invoke(cli.app, ["hw", "setup", "--dry-run"], env={"MJOLNIR_PLATFORM": "orin"})
    assert r.exit_code == 0, r.output
    assert len(no_script) == 1  # still runnable: the thor script is the jetson path


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


def test_hw_status_reads_the_fan_conf(tmp_path, monkeypatch):
    conf = tmp_path / "nvfancontrol.conf"
    conf.write_text(
        "\tFAN_PROFILE quiet\n\tFAN_PROFILE recommended\n\tFAN_PROFILE max\n\tFAN_DEFAULT_PROFILE recommended\n"
    )
    monkeypatch.setenv("MJOLNIR_NVFANCONF", str(conf))
    r = CliRunner().invoke(cli.app, ["hw", "status"])
    assert r.exit_code == 0, r.output
    assert "quiet, recommended, max" in r.output
    assert "default:  recommended" in r.output
    assert conf.name + ".bck (missing)" in r.output


def test_hw_status_names_a_missing_conf(tmp_path, monkeypatch):
    monkeypatch.setenv("MJOLNIR_NVFANCONF", str(tmp_path / "absent.conf"))
    r = CliRunner().invoke(cli.app, ["hw", "status"])
    assert r.exit_code == 4 and "no conf at" in r.output


def test_hw_setup_reports_an_unknown_platform_with_exit_2(monkeypatch):
    from mjolnir_cli.platforms import UnknownPlatform

    def boom(*a, **kw):
        raise UnknownPlatform("could not detect the platform")

    monkeypatch.setattr("mjolnir_cli.thor.dispatch", boom)
    r = CliRunner().invoke(cli.app, ["hw", "setup"])
    assert r.exit_code == 2 and "could not detect the platform" in r.output


def test_hw_setup_surfaces_the_script_failure(monkeypatch):
    def boom(*a, **kw):
        raise ValueError("bad --fan-profile")

    monkeypatch.setattr("mjolnir_cli.thor.dispatch", boom)
    r = CliRunner().invoke(cli.app, ["hw", "setup"])
    assert r.exit_code == 4 and "bad --fan-profile" in r.output


# --- the Docker pin offer (V8) ---------------------------------------------
# The check is recorded, not run: below-pin behaviour lives in
# test_docker_version.py; here only the wiring (kwargs, skip handling).


@pytest.fixture
def _record_check(monkeypatch):
    seen: list[dict] = []
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    monkeypatch.setattr("mjolnir_cli.docker_version.ensure_docker_version", lambda **kw: seen.append(kw))
    return seen


def test_hw_setup_skips_the_check_when_the_docker_step_is_skipped(no_script, _record_check):
    r = CliRunner().invoke(cli.app, ["hw", "setup", "--skip", "docker"])
    assert r.exit_code == 0, r.output
    assert _record_check == []  # not called with --skip docker
    assert no_script[0]["skip"] == "docker"


def test_hw_setup_passes_yes_and_dry_run_to_the_check(no_script, _record_check):
    r = CliRunner().invoke(cli.app, ["hw", "setup", "--dry-run", "--yes"])
    assert r.exit_code == 0, r.output
    assert _record_check == [{"assume_yes": True, "dry_run": True}]


def test_hw_setup_unknown_skip_name_exits_4_without_running_the_check(no_script, _record_check, monkeypatch, capsys):
    # exact match, like thor.build_cmd: "DOCKER" is not a step name
    r = CliRunner().invoke(cli.app, ["hw", "setup", "--skip", "DOCKER"])
    assert r.exit_code == 4, r.output
    assert _record_check == []  # no check run
    assert no_script == []  # the script never runs either
    assert "unknown --skip step(s)" in r.output
