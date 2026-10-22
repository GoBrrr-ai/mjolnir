"""The measurement-window root commands (`gate`, `grant-permissions`,
`drop-caches`) — the command bodies in ``commands/window.py``. The host
effects are faked at the `tasks` / `cachectl` boundaries."""

from __future__ import annotations

from typer.testing import CliRunner

from mjolnir_cli import cachectl, cli, tasks

runner = CliRunner()


def test_gate_threads_the_flags_into_run_gate(monkeypatch):
    seen = {}

    def run_gate(s, script_args, dry_run):
        seen.update(args=script_args, dry_run=dry_run)
        return 0

    monkeypatch.setattr(tasks, "run_gate", run_gate)
    r = runner.invoke(cli.app, ["gate", "--once"])
    assert r.exit_code == 0 and seen == {"args": ["--once"], "dry_run": False}
    r = runner.invoke(cli.app, ["gate", "--confirm", "3", "--timeout", "60.0"])
    assert r.exit_code == 0 and seen == {"args": ["--confirm", "3", "--timeout", "60.0"], "dry_run": False}


def test_gate_exits_with_the_gate_result(monkeypatch):
    monkeypatch.setattr(tasks, "run_gate", lambda s, args, dry_run: 7)
    r = runner.invoke(cli.app, ["gate", "--once"])
    assert r.exit_code == 7


def test_grant_permissions_reports_the_install_result(monkeypatch):
    seen = {}
    monkeypatch.setattr(cachectl, "grant_drop_caches", lambda assume_yes: seen.update(assume_yes=assume_yes) or 0)
    r = runner.invoke(cli.app, ["grant-permissions"])
    assert r.exit_code == 0 and seen == {"assume_yes": False}
    r = runner.invoke(cli.app, ["grant-permissions", "--yes"])
    assert r.exit_code == 0 and seen == {"assume_yes": True}


def test_drop_caches_needs_the_grant(monkeypatch):
    monkeypatch.setattr(cachectl, "drop_caches_granted", lambda: False)
    r = runner.invoke(cli.app, ["drop-caches"])
    assert r.exit_code == 4 and "grant missing" in r.output


def test_drop_caches_clears_and_reports(monkeypatch):
    monkeypatch.setattr(cachectl, "drop_caches_granted", lambda: True)
    monkeypatch.setattr(cachectl, "drop_caches", lambda: 0)
    r = runner.invoke(cli.app, ["drop-caches"])
    assert r.exit_code == 0 and "page cache cleared" in r.output
    monkeypatch.setattr(cachectl, "drop_caches", lambda: 9)
    r = runner.invoke(cli.app, ["drop-caches"])
    assert r.exit_code == 9 and "failed (exit 9)" in r.output
