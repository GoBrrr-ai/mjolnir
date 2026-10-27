"""The pinned minimum Docker version and the interactive update offer.

Nothing here calls real docker or apt: ``subprocess.run`` and
``sys.stdin.isatty`` are monkeypatched, and the conftest autouse fixture keeps
the env skip on for every other test in the suite.
"""

from __future__ import annotations

import subprocess

import pytest
import typer

from mjolnir_cli import docker_version


class FakeRun:
    """Records calls; returns a configurable CompletedProcess."""

    def __init__(self, stdout="29.1.3", returncode=0):
        self.calls: list[list[str]] = []
        self.stdout = stdout
        self.returncode = returncode

    def __call__(self, args, **kw):
        self.calls.append(args)
        return subprocess.CompletedProcess(args, self.returncode, stdout=self.stdout, stderr="")


@pytest.fixture
def fake_docker(monkeypatch):
    """Patch installed_docker_version + upgrade_commands so no real docker/apt runs."""
    monkeypatch.setattr(docker_version, "installed_docker_version", lambda: "29.1.3")
    monkeypatch.setattr(
        docker_version,
        "upgrade_commands",
        lambda: [["sudo", "apt-get", "install", "-y", "docker-ce"]],
    )
    return True


def _record_upgrades(monkeypatch, probe):
    """Wrap the version-probe fake so upgrade commands are recorded, not run."""
    calls: list[list[str]] = []

    def run(cmd, **kw):
        if cmd[:1] == ["docker"]:
            return probe(cmd, **kw)
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_version.subprocess, "run", run)
    return calls


# --- parsing ---------------------------------------------------------------


def test_installed_version_strips_the_ubuntu_build_suffix(monkeypatch):
    fake = FakeRun(stdout="29.1.3-0ubuntu1\n")
    monkeypatch.setattr(docker_version.subprocess, "run", fake)
    assert docker_version.installed_docker_version() == "29.1.3"


def test_installed_version_falls_back_to_the_client_when_the_server_is_down(monkeypatch):
    def run(args, **kw):
        if "{{.Server.Version}}" in args:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="cannot connect")
        return subprocess.CompletedProcess(args, 0, stdout="28.3.0\n")

    monkeypatch.setattr(docker_version.subprocess, "run", run)
    assert docker_version.installed_docker_version() == "28.3.0"


def test_installed_version_is_none_when_docker_is_missing(monkeypatch):
    def boom(*a, **kw):
        raise OSError("no docker binary")

    monkeypatch.setattr(docker_version.subprocess, "run", boom)
    assert docker_version.installed_docker_version() is None


# --- comparison ------------------------------------------------------------


def test_numeric_tuple_comparison_orders_29_10_above_29_9():
    assert docker_version.is_below_pin("29.9.2") is False
    assert docker_version.is_below_pin("29.10.0") is False  # 29.10.0 > 29.9.2
    assert docker_version.is_below_pin("29.9.1") is True
    assert docker_version.is_below_pin("28.0.0") is True


def test_upgrade_commands_are_the_apt_steps():
    cmds = docker_version.upgrade_commands()
    flat = [" ".join(c) for c in cmds]
    assert any(
        "apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin" in f
        for f in flat
    )
    assert any(f.startswith("sudo systemctl restart docker") for f in flat)


def test_repo_setup_writes_a_resolved_docker_list(monkeypatch):
    # no keyring yet -> the repo is added; the line must be fully resolved
    # (argv has no shell, so a literal $(...) would be written verbatim)
    monkeypatch.setattr(docker_version.os.path, "exists", lambda _p: False)
    monkeypatch.setattr(docker_version, "_ubuntu_codename", lambda *_a: "noble")
    monkeypatch.setattr(docker_version, "_dpkg_arch", lambda: "arm64")
    tee = next(c for c in docker_version.upgrade_commands() if c[1] == "tee")
    line = tee[3]
    assert "$(" not in line
    assert line == (
        "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable"
    )


def test_ubuntu_codename_prefers_ubuntu_codename(tmp_path):
    f = tmp_path / "os-release"
    f.write_text('UBUNTU_CODENAME="noble"\nVERSION_CODENAME="bionic"\n')
    assert docker_version._ubuntu_codename(str(f)) == "noble"


# --- ensure_docker_version behaviour ----------------------------------------
# Each test sets MJOLNIR_SKIP_DOCKER_CHECK explicitly so it overrides the
# conftest autouse fixture (which only sets it when unset).


def test_at_or_above_the_pin_is_a_noop(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    monkeypatch.setattr(docker_version, "installed_docker_version", lambda: "29.9.2")
    docker_version.ensure_docker_version()
    out = capsys.readouterr().out
    assert out == ""


def test_below_the_pin_tty_yes_runs_the_commands(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    fake = FakeRun(stdout="29.1.3\n")
    monkeypatch.setattr(docker_version.subprocess, "run", fake)
    monkeypatch.setattr(docker_version.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: True)
    calls = _record_upgrades(monkeypatch, fake)
    docker_version.ensure_docker_version()
    assert "Docker 29.1.3 is below mjolnir's pinned 29.9.2." in capsys.readouterr().out
    assert len(fake.calls) == 1  # the version probe only
    assert len(calls) > 0 and all(c[0] == "sudo" for c in calls)  # the upgrade commands ran


def test_below_the_pin_tty_no_runs_nothing(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    fake = FakeRun(stdout="29.1.3\n")
    monkeypatch.setattr(docker_version.subprocess, "run", fake)
    monkeypatch.setattr(docker_version.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: False)
    calls = _record_upgrades(monkeypatch, fake)
    docker_version.ensure_docker_version()
    assert len(fake.calls) == 1
    assert calls == []  # nothing ran
    assert "Continuing with the installed Docker" in capsys.readouterr().out


def test_below_the_pin_non_tty_never_prompts_and_runs_nothing(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    fake = FakeRun(stdout="29.1.3\n")
    monkeypatch.setattr(docker_version.subprocess, "run", fake)
    monkeypatch.setattr(docker_version.sys.stdin, "isatty", lambda: False)
    called = []
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: called.append(a) or True)
    calls = _record_upgrades(monkeypatch, fake)
    docker_version.ensure_docker_version()
    assert called == []  # never prompted
    assert calls == []  # never blocked, never ran
    captured = capsys.readouterr()
    assert "Non-interactive session: not updating" in captured.err
    assert "sudo apt-get install -y docker-ce" in captured.out  # the commands were printed


def test_dry_run_prints_only(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    fake = FakeRun(stdout="29.1.3\n")
    monkeypatch.setattr(docker_version.subprocess, "run", fake)
    calls = _record_upgrades(monkeypatch, fake)
    docker_version.ensure_docker_version(dry_run=True)
    assert len(fake.calls) == 1  # just the version probe
    assert calls == []  # nothing executed
    assert "sudo apt-get install -y docker-ce" in capsys.readouterr().out


def test_assume_yes_runs_without_prompting(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    fake = FakeRun(stdout="29.1.3\n")
    monkeypatch.setattr(docker_version.subprocess, "run", fake)
    called = []
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: called.append(a) or True)
    calls = _record_upgrades(monkeypatch, fake)
    docker_version.ensure_docker_version(assume_yes=True)
    assert called == []  # no prompt
    assert len(calls) > 0  # the upgrade commands ran


def test_env_skip_disables_everything(monkeypatch, capsys):
    fake = FakeRun(stdout="29.1.3\n")
    monkeypatch.setattr(docker_version.subprocess, "run", fake)
    monkeypatch.setenv("MJOLNIR_SKIP_DOCKER_CHECK", "1")
    docker_version.ensure_docker_version()
    assert fake.calls == []  # not even the version probe
    assert capsys.readouterr().out == ""


def test_runs_at_most_once_per_process(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    fake = FakeRun(stdout="29.1.3\n")
    monkeypatch.setattr(docker_version.subprocess, "run", fake)
    monkeypatch.setattr(docker_version.sys.stdin, "isatty", lambda: False)
    calls = _record_upgrades(monkeypatch, fake)
    docker_version.ensure_docker_version()
    docker_version.ensure_docker_version()
    assert len(fake.calls) == 1  # probed once, the guard holds the second call
    assert calls == []


def test_failed_upgrade_command_exits_with_code_4(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    fake = FakeRun(stdout="29.1.3\n")
    monkeypatch.setattr(docker_version.subprocess, "run", fake)
    monkeypatch.setattr(docker_version.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: True)

    def failing(cmd, **kw):
        if cmd[:1] == ["docker"]:
            return fake(cmd, **kw)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(docker_version.subprocess, "run", failing)
    with pytest.raises(typer.Exit) as e:
        docker_version.ensure_docker_version(assume_yes=True)
    assert e.value.exit_code == 4
