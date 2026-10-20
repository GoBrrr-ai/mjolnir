"""The ``mjolnir platform`` command group (``cli_platform.py``, mounted in
``cli.py``) — read-only in cycle 2: ``list`` and ``show``.

The spec (plan §0.1 / §3.5, detection-first — no remembered state), pinned
as tests:

* ``list`` prints the registry (all three keys, both statuses) — unchanged;
* ``show`` resolves flag/env/detection and prints the platform, its
  confidence and one provenance line per source (source name + raw value);
  on an unknown host it fails naming the probe instructions, and a
  remembered platform in ``$MJOLNIR_STATE`` is ignored — there is no state;
* ``set`` is gone: a usage error (exit 2), no state file, ever;
* :func:`mjolnir_cli.cli_shared._settings` lost its ``platform`` slot —
  the remembered-platform threading is dead; resolution happens in
  ``resolve()`` (flag > env > detection).
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from mjolnir_cli import cli, cli_shared
from platform_fixtures import install


@pytest.fixture(autouse=True)
def _clean_mjolnir_env(monkeypatch):
    for var in (
        "MJOLNIR_STATE",
        "MJOLNIR_MODEL",
        "MJOLNIR_QUANT",
        "MJOLNIR_IMAGE",
        "MJOLNIR_PORT",
    ):
        monkeypatch.delenv(var, raising=False)
    # MJOLNIR_PLATFORM stays at the suite's thor pin; detection tests opt out


def test_platform_list_prints_the_registry():
    r = CliRunner().invoke(cli.app, ["platform", "list"])
    assert r.exit_code == 0, r.output
    assert all(k in r.output for k in ("thor", "spark", "orin"))
    assert "supported" in r.output and "research" in r.output


def test_platform_show_prints_the_platform_confidence_and_provenance(monkeypatch):
    # two source classes agree: confirmed, with one provenance line per source
    install(monkeypatch, smi="NVIDIA Thor, 11.0", jetson="nvidia,p4071-0000 nvidia,tegra264")
    r = CliRunner().invoke(cli.app, ["platform", "show"])
    assert r.exit_code == 0, r.output
    assert "thor" in r.output
    assert "confirmed" in r.output
    assert "nvidia-smi" in r.output and "NVIDIA Thor, 11.0" in r.output
    assert "jetson-files" in r.output and "tegra264" in r.output


def test_platform_show_honors_the_env_platform():
    r = CliRunner().invoke(cli.app, ["platform", "show"], env={"MJOLNIR_PLATFORM": "spark"})
    assert r.exit_code == 0, r.output
    assert "spark" in r.output


def test_platform_show_on_an_unknown_host_fails_with_the_probe_instructions(monkeypatch):
    # no env, no probes: an explicit error, never a silent thor guess
    install(monkeypatch, smi=None, jetson=None, dmi=None)
    r = CliRunner().invoke(cli.app, ["platform", "show"], env={"MJOLNIR_PLATFORM": None})
    assert r.exit_code != 0, r.output
    assert "nvidia-smi" in r.output
    assert "MJOLNIR_PLATFORM" in r.output  # the escape hatch is named


def test_platform_set_is_gone(tmp_path):
    state = tmp_path / "state.json"
    r = CliRunner().invoke(cli.app, ["platform", "set", "spark"], env={"MJOLNIR_STATE": str(state)})
    assert r.exit_code == 2, r.output  # command not found — a usage error, not a handler
    assert "No such command" in r.output
    assert not state.exists()  # no state round-trip: the command never existed to write one


def test_platform_show_ignores_a_remembered_platform(monkeypatch, tmp_path):
    # a state file left by an older CLI must not steer the detection-first one
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"platform": "spark"}))
    install(monkeypatch, smi="NVIDIA Thor, 11.0")
    r = CliRunner().invoke(cli.app, ["platform", "show"], env={"MJOLNIR_STATE": str(state), "MJOLNIR_PLATFORM": None})
    assert r.exit_code == 0, r.output
    assert "thor" in r.output
    assert "spark" not in r.output  # the remembered key is dead


def test_settings_ignores_a_remembered_platform(monkeypatch, tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"platform": "spark"}))
    monkeypatch.setenv("MJOLNIR_STATE", str(state))
    install(monkeypatch, smi="NVIDIA Thor, 11.0")  # detection: thor; no env
    assert cli_shared._settings(None, None, None, None).platform == "thor"


def test_settings_has_no_platform_parameter():
    # the state threading is gone: _settings is back to four slots
    with pytest.raises(TypeError):
        cli_shared._settings(None, None, None, None, platform="spark")  # ty: ignore[unknown-argument]
