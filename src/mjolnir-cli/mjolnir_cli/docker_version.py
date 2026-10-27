"""The pinned minimum Docker version and the interactive update offer.

mjolnir's container glue (the serve image, the verify stages, the one-shot
calibrate/gate containers) is written against modern Docker CLI/daemon
behaviour; a distro ``docker.io`` from an old base breaks it. The pin is one
constant here — the single source of truth. ``ensure_docker_version`` runs at
most once per process: below the pin it offers to upgrade (a TTY gets a
``[Y/n]`` prompt, everything else never blocks), mirroring the apt steps in
``scripts/hw/setup-jetson-steps.sh``.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys

import typer

# Latest docker-ce stable for Ubuntu 24.04 noble arm64.
PINNED_DOCKER_VERSION = "29.9.2"

_SKIP_ENV = "MJOLNIR_SKIP_DOCKER_CHECK"

# Set True after the first check so later calls in the same process no-op.
_checked = False


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split(".") if p.isdigit())


def installed_docker_version() -> str | None:
    """The running daemon's version (client as fallback); None when docker is
    missing or the call fails. Build suffixes are stripped:
    ``29.1.3-0ubuntu1`` -> ``29.1.3``."""
    for fmt in ("{{.Server.Version}}", "{{.Client.Version}}"):
        try:
            out = subprocess.run(["docker", "version", "--format", fmt], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0:
            v = out.stdout.strip().split("-")[0].strip()
            if v and _version_tuple(v):
                return ".".join(p for p in v.split(".") if p.isdigit())
    return None


def is_below_pin(installed: str, pinned: str = PINNED_DOCKER_VERSION) -> bool:
    """Numeric tuple comparison, so 29.10.0 > 29.9.2."""
    return _version_tuple(installed) < _version_tuple(pinned)


def _dpkg_arch() -> str:
    machine = platform.machine()
    return {"aarch64": "arm64", "x86_64": "amd64"}.get(machine, machine)


def _ubuntu_codename(os_release: str = "/etc/os-release") -> str:
    fields = {}
    with open(os_release) as f:
        for line in f:
            key, sep, value = line.strip().partition("=")
            if sep:
                fields[key] = value.strip('"')
    return fields.get("UBUNTU_CODENAME") or fields["VERSION_CODENAME"]


def _docker_repo_line(keyring: str) -> str:
    """The line for /etc/apt/sources.list.d/docker.list, fully resolved.
    The steps are argv (no shell), so the ``$(...)`` the Docker docs leave
    to bash is substituted here before the line is written."""
    return (
        f"deb [arch={_dpkg_arch()} signed-by={keyring}] "
        "https://download.docker.com/linux/ubuntu "
        f"{_ubuntu_codename()} stable"
    )


def upgrade_commands() -> list[list[str]]:
    """The apt upgrade steps, mirroring ensure_docker_repo + install_docker in
    scripts/hw/setup-jetson-steps.sh (install, not --only-upgrade, so a distro
    docker.io is replaced too)."""
    keyring = "/etc/apt/keyrings/docker.asc"
    cmds: list[list[str]] = []
    if not os.path.exists(keyring):
        cmds += [
            ["sudo", "apt-get", "update"],
            ["sudo", "apt-get", "install", "-y", "ca-certificates", "curl"],
            ["sudo", "install", "-m", "0755", "-d", "/etc/apt/keyrings"],
            [
                "sudo",
                "curl",
                "-fsSL",
                "https://download.docker.com/linux/ubuntu/gpg",
                "-o",
                keyring,
            ],
            ["sudo", "chmod", "a+r", keyring],
            ["sudo", "tee", "/etc/apt/sources.list.d/docker.list", _docker_repo_line(keyring)],
        ]
    cmds += [
        ["sudo", "apt-get", "update"],
        [
            "sudo",
            "apt-get",
            "install",
            "-y",
            "docker-ce",
            "docker-ce-cli",
            "containerd.io",
            "docker-buildx-plugin",
            "docker-compose-plugin",
        ],
        ["sudo", "systemctl", "restart", "docker"],
    ]
    return cmds


def _run_upgrade(commands: list[list[str]]) -> None:
    for cmd in commands:
        print("$ " + " ".join(cmd))
        p = subprocess.run(cmd)
        if p.returncode != 0:
            typer.secho(f"Docker upgrade failed ({p.returncode}): {' '.join(cmd[:3])}…", fg=typer.colors.RED, err=True)
            raise typer.Exit(4)


def ensure_docker_version(*, assume_yes: bool = False, dry_run: bool = False) -> None:
    """Offer to bring an old Docker up to the pin. At most once per process;
    skipped entirely when ``MJOLNIR_SKIP_DOCKER_CHECK=1``. Never blocks on a
    non-TTY stdin."""
    global _checked
    if _checked:
        return
    _checked = True
    if os.environ.get(_SKIP_ENV) == "1":
        return
    installed = installed_docker_version()
    if installed is None or not is_below_pin(installed):
        return
    print(f"Docker {installed} is below mjolnir's pinned {PINNED_DOCKER_VERSION}.")
    typer.secho(
        "Warning: the upgrade restarts the Docker daemon, which restarts every running container on this box.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    commands = upgrade_commands()
    if dry_run:
        for cmd in commands:
            print("$ " + " ".join(cmd))
        return
    if assume_yes:
        _run_upgrade(commands)
        return
    if sys.stdin.isatty():
        if typer.confirm("Update Docker now?", default=True):
            _run_upgrade(commands)
        else:
            typer.secho("Continuing with the installed Docker — re-run with --yes to update.", fg=typer.colors.YELLOW)
    else:
        typer.secho(
            "Non-interactive session: not updating. To update manually:",
            fg=typer.colors.YELLOW,
            err=True,
        )
        for cmd in commands:
            print("$ " + " ".join(cmd))


def reset_check_for_tests() -> None:
    """Clear the once-per-process guard (test hook only)."""
    global _checked
    _checked = False
