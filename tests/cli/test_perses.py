"""`mjolnir perses` — the command bodies in ``commands/perses.py``. The stack
driver is faked at the ``mjolnir_cli.perses`` module boundary."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import mjolnir_cli.perses as perses_mod
from mjolnir_cli import cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "state.json"))
    return tmp_path


def test_perses_up_threads_the_options(monkeypatch):
    seen = {}

    def up(s, **kw):
        seen.update(**kw)

    monkeypatch.setattr(perses_mod, "up", up)
    r = runner.invoke(cli.app, ["perses", "up", "--port", "3456", "--no-wait", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert seen == {"port": 3456, "wait": False, "wait_timeout_s": 300.0, "dry_run": True}
    assert "dashboards:  mjolnir perses status" in r.output


def test_perses_up_failure_names_the_error_and_picks_the_exit(monkeypatch):
    def not_found(*a, **kw):
        raise FileNotFoundError("env file not found")

    def compose_down(*a, **kw):
        raise perses_mod.PersesError("compose down")

    monkeypatch.setattr(perses_mod, "up", not_found)
    r = runner.invoke(cli.app, ["perses", "up"])
    assert r.exit_code == 4 and "env file not found" in r.output

    monkeypatch.setattr(perses_mod, "up", compose_down)
    r = runner.invoke(cli.app, ["perses", "up"])
    assert r.exit_code == 1 and "compose down" in r.output


def test_perses_down_threads_purge_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(perses_mod, "down", lambda **kw: seen.update(**kw) or 0)
    r = runner.invoke(cli.app, ["perses", "down", "--purge", "--dry-run"])
    assert r.exit_code == 0 and seen == {"purge": True, "dry_run": True}


def test_perses_status_prints_json(monkeypatch):
    monkeypatch.setattr(perses_mod, "status", lambda s, port=None: {"services": {"perses": "running"}})
    r = runner.invoke(cli.app, ["perses", "status"])
    assert r.exit_code == 0 and '"perses": "running"' in r.output


def test_perses_logs_threads_the_service_pick(monkeypatch):
    seen = {}
    monkeypatch.setattr(perses_mod, "logs", lambda **kw: seen.update(**kw) or 0)
    r = runner.invoke(cli.app, ["perses", "logs", "--service", "prometheus", "--tail", "9"])
    assert r.exit_code == 0 and seen == {"follow": False, "tail": 9, "service": "prometheus"}
