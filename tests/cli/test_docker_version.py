"""The pinned minimum Docker version: the offer (hw setup / image build) and
the notice (every dockerctl-backed command, the verify preflights).

Nothing here calls real docker or apt: ``subprocess.run``, ``sys.stdin.isatty``
and the os-release/keyring probes are monkeypatched, and the conftest autouse
fixture keeps ``MJOLNIR_SKIP_DOCKER_CHECK=1`` on for every other test in the
suite — a test that exercises the check removes it with ``monkeypatch.delenv``.
"""

from __future__ import annotations

import subprocess

import pytest
import typer
from typer.testing import CliRunner

from mjolnir_cli import cli, docker_version, dockerctl

BELOW = "29.1.3"


class FakeRun:
    """Records calls; returns a configurable CompletedProcess."""

    def __init__(self, stdout="29.1.3", returncode=0):
        self.calls: list[list[str]] = []
        self.stdout = stdout
        self.returncode = returncode

    def __call__(self, args, **kw):
        self.calls.append(args)
        return subprocess.CompletedProcess(args, self.returncode, stdout=self.stdout, stderr="")


def _hermetic_below_pin(monkeypatch, tty: bool | None = None):
    """A simulated below-pin host: no real docker probe, no real
    /etc/os-release or keyring read. Returns (probe, run) — ``run`` records
    every non-docker subprocess call (the upgrade commands)."""
    probe = FakeRun(stdout=f"{BELOW}\n")
    monkeypatch.setattr(docker_version.subprocess, "run", probe)
    if tty is not None:
        monkeypatch.setattr(docker_version.sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr(docker_version.os.path, "exists", lambda _p: True)  # keyring present
    monkeypatch.setattr(docker_version, "_ubuntu_codename", lambda *_a: "noble")
    return probe


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


def test_upgrade_commands_are_the_apt_steps(monkeypatch):
    monkeypatch.setattr(docker_version.os.path, "exists", lambda _p: True)  # keyring present
    cmds = docker_version.upgrade_commands()
    flat = [" ".join(c) for c in cmds]
    assert any(
        "apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin "
        "nvidia-container-toolkit docker-compose-plugin" in f
        for f in flat
    )
    assert any(f.startswith("sudo systemctl restart docker") for f in flat)


def test_upgrade_packages_match_the_setup_script(monkeypatch):
    # parity: every package the script's install line carries must be in
    # upgrade_commands()' install list (scripts/hw/setup-jetson-steps.sh)
    from mjolnir_cli.config import resource_path

    text = resource_path("scripts/hw/setup-jetson-steps.sh").read_text()
    line = next(ln for ln in text.splitlines() if "apt install -y docker-ce" in ln)
    pkgs = [t for t in line.split()[3:] if not t.startswith("-")]  # after `sudo apt install -y`
    monkeypatch.setattr(docker_version.os.path, "exists", lambda _p: True)
    installs = [c for c in docker_version.upgrade_commands() if len(c) > 4 and c[1:4] == ["apt-get", "install", "-y"]]
    assert len(installs) == 1
    got = installs[0][4:]
    assert set(pkgs) <= set(got), f"missing from the install list: {set(pkgs) - set(got)}"


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


def test_ubuntu_codename_is_none_without_the_codename_fields(tmp_path):
    f = tmp_path / "os-release"
    f.write_text('NAME="Ubuntu"\nVERSION="24.04.2 LTS (Noble Numbat)"\n')
    assert docker_version._ubuntu_codename(str(f)) is None


def test_ubuntu_codename_is_none_for_a_missing_file():
    assert docker_version._ubuntu_codename("/nonexistent/os-release") is None


# --- ensure_docker_version behaviour ----------------------------------------
# Each test removes MJOLNIR_SKIP_DOCKER_CHECK (the conftest autouse fixture
# sets it for every other test) and fakes the probes, so nothing touches the
# real docker daemon, apt, /etc/os-release or keyring.


def test_at_or_above_the_pin_is_a_noop(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    monkeypatch.setattr(docker_version, "installed_docker_version", lambda: "29.9.2")
    docker_version.ensure_docker_version()
    out = capsys.readouterr().out
    assert out == ""


def test_below_the_pin_tty_yes_runs_the_commands(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch, tty=True)
    upgrades: list[list[str]] = []

    def run(cmd, **kw):
        if cmd[:1] == ["docker"]:
            return probe(cmd, **kw)
        upgrades.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_version.subprocess, "run", run)
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: True)
    docker_version.ensure_docker_version()
    assert f"Docker {BELOW} is below mjolnir's pinned 29.9.2." in capsys.readouterr().out
    assert len(probe.calls) == 1  # the version probe only
    assert len(upgrades) > 0 and all(c[0] == "sudo" for c in upgrades)  # the upgrade commands ran


def test_below_the_pin_tty_no_runs_nothing(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch, tty=True)
    upgrades: list[list[str]] = []

    def run(cmd, **kw):
        if cmd[:1] == ["docker"]:
            return probe(cmd, **kw)
        upgrades.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_version.subprocess, "run", run)
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: False)
    docker_version.ensure_docker_version()
    assert len(probe.calls) == 1
    assert upgrades == []  # nothing ran
    out = capsys.readouterr().out
    assert "Continuing with the installed Docker" in out
    assert "`mjolnir hw setup --yes` updates it." in out  # the hint must point at the path that has --yes


def test_below_the_pin_non_tty_never_prompts_and_runs_nothing(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch, tty=False)
    called = []
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: called.append(a) or True)
    docker_version.ensure_docker_version()
    assert called == []  # never prompted
    assert len(probe.calls) == 1  # just the version probe
    captured = capsys.readouterr()
    assert "Non-interactive session: not updating" in captured.err
    assert "sudo apt-get install -y docker-ce" in captured.out  # the commands were printed


def test_dry_run_prints_only(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch)
    docker_version.ensure_docker_version(dry_run=True)
    assert len(probe.calls) == 1  # just the version probe
    assert "sudo apt-get install -y docker-ce" in capsys.readouterr().out


def test_assume_yes_runs_without_prompting(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch)
    called = []
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: called.append(a) or True)
    docker_version.ensure_docker_version(assume_yes=True)
    assert called == []  # no prompt
    assert len(probe.calls) > 1  # the version probe + the upgrade commands


def test_env_skip_disables_everything(monkeypatch, capsys):
    fake = FakeRun(stdout=f"{BELOW}\n")
    monkeypatch.setattr(docker_version.subprocess, "run", fake)
    monkeypatch.setenv("MJOLNIR_SKIP_DOCKER_CHECK", "1")
    docker_version.ensure_docker_version()
    assert fake.calls == []  # not even the version probe
    assert capsys.readouterr().out == ""


def test_runs_at_most_once_per_process(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch, tty=False)
    docker_version.ensure_docker_version()
    docker_version.ensure_docker_version()
    assert len(probe.calls) == 1  # probed once, the guard holds the second call


def test_failed_upgrade_command_exits_with_code_4(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch)

    def failing(cmd, **kw):
        if cmd[:1] == ["docker"]:
            return probe(cmd, **kw)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(docker_version.subprocess, "run", failing)
    with pytest.raises(typer.Exit) as e:
        docker_version.ensure_docker_version(assume_yes=True)
    assert e.value.exit_code == 4


# --- the notice path (offer=False) -----------------------------------------


def test_notice_below_the_pin_prints_one_stderr_line_and_runs_nothing(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch, tty=True)
    called = []
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: called.append(a) or True)
    docker_version.ensure_docker_version(offer=False)
    err = capsys.readouterr().err
    assert (
        err
        == f"Docker {BELOW} is below mjolnir's pinned 29.9.2 — run `mjolnir hw setup` or `mjolnir image build` to update.\n"
    )
    assert called == []  # no prompt
    assert len(probe.calls) == 1  # the version probe only — no commands run


def test_notice_does_not_suppress_a_later_offer(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch, tty=True)
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: False)
    docker_version.ensure_docker_version(offer=False)
    capsys.readouterr()
    docker_version.ensure_docker_version()  # the offer still runs after the notice
    assert len(probe.calls) == 2  # probed twice: the guards are separate
    assert "Continuing with the installed Docker" in capsys.readouterr().out


def test_notice_is_silent_at_or_above_the_pin(monkeypatch, capsys):
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    monkeypatch.setattr(docker_version, "installed_docker_version", lambda: "29.9.2")
    docker_version.ensure_docker_version(offer=False)
    assert capsys.readouterr().out == "" and capsys.readouterr().err == ""


def test_missing_docker_is_silent_on_both_paths(monkeypatch, capsys):
    # V10: installed_docker_version -> None: silent, raises nothing, runs nothing
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch, tty=True)
    monkeypatch.setattr(docker_version, "installed_docker_version", lambda: None)
    docker_version.ensure_docker_version(offer=False)
    docker_version.ensure_docker_version()
    assert probe.calls == []  # nothing ran at all
    out, err = capsys.readouterr().out, capsys.readouterr().err
    assert out == "" and err == ""


def test_non_ubuntu_cannot_add_the_repo_and_updates_manually(monkeypatch, capsys):
    # V3: keyring missing + codename None -> upgrade_commands cannot add the
    # repo: the manual-update message, no offer, no exception
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch, tty=True)
    monkeypatch.setattr(docker_version.os.path, "exists", lambda _p: False)  # keyring missing
    monkeypatch.setattr(docker_version, "_ubuntu_codename", lambda *_a: None)
    called = []
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: called.append(a) or True)
    docker_version.ensure_docker_version()
    assert called == []
    assert len(probe.calls) == 1  # the version probe only
    err = capsys.readouterr().err
    assert "automatic Docker update supports Ubuntu (apt) only" in err
    assert f">= {docker_version.PINNED_DOCKER_VERSION}" in err


# --- wiring ------------------------------------------------------------------


def test_docker_helper_calls_the_check_once_with_the_notice(monkeypatch, capsys):
    # V9: dockerctl._docker() -> exactly one ensure_docker_version(offer=False)
    # (the real guard does the once-per-process part; here the fake records it)
    seen: list[dict] = []
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    monkeypatch.setattr(dockerctl.shutil, "which", lambda _n: "/usr/bin/docker")
    monkeypatch.setattr(docker_version, "installed_docker_version", lambda: BELOW)
    monkeypatch.setattr(docker_version, "ensure_docker_version", lambda **kw: seen.append(kw) or None)
    dockerctl._docker()
    assert seen == [{"offer": False}]  # called once, notice path
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_serve_status_on_a_below_pin_host_neither_confirms_nor_upgrades(monkeypatch):
    # V1: read-only paths get the notice only — no confirm, no upgrade command
    runner = CliRunner()
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch, tty=True)
    monkeypatch.setattr(dockerctl.shutil, "which", lambda _n: "/usr/bin/docker")
    confirmed = []
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: confirmed.append(a) or True)
    monkeypatch.setattr(dockerctl, "_run_quiet", lambda args, ok_codes=(0, 1): (1, ""))  # no container: quiet failure
    r = runner.invoke(cli.app, ["serve", "status"])
    assert r.exit_code == 0, r.output
    assert confirmed == []  # no confirm
    upgrades = [c for c in probe.calls if c[:1] != ["docker"]]
    assert upgrades == []  # no upgrade command ran
    assert f"Docker {BELOW} is below mjolnir's pinned 29.9.2 —" in r.stderr  # the notice on stderr


def test_serve_up_dry_run_on_a_below_pin_host_neither_confirms_nor_upgrades(monkeypatch):
    # V1: --dry-run is a read-only path too — the notice, never the offer
    runner = CliRunner()
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch, tty=True)
    monkeypatch.setattr(dockerctl.shutil, "which", lambda _n: "/usr/bin/docker")
    confirmed = []
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: confirmed.append(a) or True)
    monkeypatch.setattr(dockerctl, "_run_quiet", lambda args, ok_codes=(0, 1): (1, ""))  # no container: quiet failure
    r = runner.invoke(cli.app, ["serve", "up", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "docker run" in r.output  # the dry-run plan was printed
    assert confirmed == []
    upgrades = [c for c in probe.calls if c[:1] != ["docker"]]
    assert upgrades == []  # no upgrade command ran
    assert f"Docker {BELOW} is below mjolnir's pinned 29.9.2 —" in r.stderr  # the notice on stderr


def test_image_build_dry_run_offers_with_dry_run(monkeypatch, capsys):
    # V1: image build is an explicit mutating entry point — the offer path
    # (here with dry_run=True: the steps print, nothing runs)
    runner = CliRunner()
    monkeypatch.delenv("MJOLNIR_SKIP_DOCKER_CHECK", raising=False)
    probe = _hermetic_below_pin(monkeypatch, tty=True)
    monkeypatch.setattr(typer, "confirm", lambda *a, **kw: True)
    r = runner.invoke(cli.app, ["image", "build", "--tag", "mine:v1", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "docker build -f" in r.output
    assert f"Docker {BELOW} is below mjolnir's pinned 29.9.2." in r.output  # the offer banner
    assert "sudo apt-get install -y docker-ce" in r.output  # the steps printed
    upgrades = [c for c in probe.calls if c[:1] not in (["docker"], ["git"])]
    assert upgrades == []  # no upgrade command ran (the git rev-parse is image build's own)
    assert sum(1 for c in probe.calls if c[:1] == ["docker"]) == 1  # the version probe only
