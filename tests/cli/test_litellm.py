"""`mjolnir litellm` — the command bodies in ``commands/litellm.py``. The
stack driver is faked at the ``mjolnir_cli.litellm`` module boundary; the
error exits and the option threading are what's pinned."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import mjolnir_cli.litellm as litellm_mod
from mjolnir_cli import cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "state.json"))
    return tmp_path


def test_litellm_up_threads_the_options(monkeypatch):
    seen = {}

    def up(s, **kw):
        seen.update(**kw)

    monkeypatch.setattr(litellm_mod, "up", up)
    r = runner.invoke(cli.app, ["litellm", "up", "--port", "4001", "--no-wait", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert seen == {"proxy_port": 4001, "wait": False, "wait_timeout_s": 600.0, "dry_run": True}
    assert "proxy:  mjolnir litellm status" in r.output


def test_litellm_up_failure_names_the_error_and_picks_the_exit(monkeypatch):
    def not_found(*a, **kw):
        raise FileNotFoundError("template not found")

    def compose_failed(*a, **kw):
        raise litellm_mod.LitellmError("compose failed")

    monkeypatch.setattr(litellm_mod, "up", not_found)
    r = runner.invoke(cli.app, ["litellm", "up"])
    assert r.exit_code == 4 and "template not found" in r.output

    monkeypatch.setattr(litellm_mod, "up", compose_failed)
    r = runner.invoke(cli.app, ["litellm", "up"])
    assert r.exit_code == 1 and "compose failed" in r.output


def test_litellm_down_threads_purge_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(litellm_mod, "down", lambda **kw: seen.update(**kw) or 0)
    r = runner.invoke(cli.app, ["litellm", "down", "--purge", "--dry-run"])
    assert r.exit_code == 0 and seen == {"purge": True, "dry_run": True}


def test_litellm_status_prints_json_and_reports_missing_data(monkeypatch):
    monkeypatch.setattr(litellm_mod, "status", lambda s, port=None: {"containers": {}, "proxy": "live"})
    r = runner.invoke(cli.app, ["litellm", "status"])
    assert r.exit_code == 0 and '"proxy": "live"' in r.output

    monkeypatch.setattr(
        litellm_mod, "status", lambda s, port=None: (_ for _ in ()).throw(FileNotFoundError("no litellm data dir"))
    )
    r = runner.invoke(cli.app, ["litellm", "status"])
    assert r.exit_code == 4 and "no litellm data dir" in r.output


def test_litellm_logs_threads_follow_and_tail(monkeypatch):
    seen = {}
    monkeypatch.setattr(litellm_mod, "logs", lambda **kw: seen.update(**kw) or 0)
    r = runner.invoke(cli.app, ["litellm", "logs", "--follow", "--tail", "5"])
    assert r.exit_code == 0 and seen == {"follow": True, "tail": 5}


def test_litellm_config_renders_and_validates(tmp_path, monkeypatch):
    p = tmp_path / "rendered.yaml"
    p.write_text("model: x")
    monkeypatch.setattr(litellm_mod, "render_config", lambda s: p)
    r = runner.invoke(cli.app, ["litellm", "config"])
    assert r.exit_code == 0 and "rendered ok" in r.output
    r = runner.invoke(cli.app, ["litellm", "config", "--show"])
    assert r.exit_code == 0 and "model: x" in r.output

    monkeypatch.setattr(
        litellm_mod, "render_config", lambda s: (_ for _ in ()).throw(litellm_mod.LitellmError("bad yaml"))
    )
    r = runner.invoke(cli.app, ["litellm", "config"])
    assert r.exit_code == 1 and "bad yaml" in r.output
