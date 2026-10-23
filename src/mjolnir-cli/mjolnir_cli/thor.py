"""``mjolnir hw setup`` — Jetson AGX Thor host pre-configuration.

The ported Thor HW setup (``scripts/hw/setup-thor.sh``, from the ~/thor
scripts) brings a box to the serving baseline the rest of the CLI
assumes: headless boot target, apt upgrade, Docker with the NVIDIA
runtime as default, pip, jtop, a 32 GB swap file (zRAM off), the fan
profiles (``recommended`` + ``max`` installed, default selected — the
default profile is ``recommended``), trimmed host services, locked max
clocks, and the MAXN power mode. Every step is idempotent.

This module only locates and runs the script: flags map 1:1 to its
options, and ``--dry-run`` previews the plan without touching the box.
The script needs sudo — it asks on the first privileged call.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from mjolnir_cli.config import resource_path
from mjolnir_cli.platforms import DEFAULT_PLATFORM

# The --skip names, in run order (mirror of the script's steps).
STEPS: tuple[str, ...] = (
    "gui",
    "upgrade",
    "pip",
    "docker",
    "jtop",
    "nsight",
    "memory",
    "fan",
    "host",
    "clocks",
    "power",
)

FAN_PROFILES: tuple[str, ...] = ("recommended", "max", "cool", "quiet")
DEFAULT_FAN_PROFILE = "recommended"


def setup_script() -> Path:
    """The ported setup script (the repo's ``scripts/hw/setup-thor.sh``)."""
    p = resource_path("scripts/hw/setup-thor.sh")
    if not p.is_file():
        raise FileNotFoundError(f"setup script not found: {p}")
    return p


def build_cmd(
    yes: bool = False,
    upgrade: bool = True,
    keep_gui: bool = False,
    fan_profile: str = DEFAULT_FAN_PROFILE,
    swap_size: int = 32,
    skip: str = "",
    reboot: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """The ``bash scripts/hw/setup-thor.sh ...`` argv for the given flags."""
    if fan_profile not in FAN_PROFILES:
        raise ValueError(f"bad --fan-profile '{fan_profile}' (want one of: {', '.join(FAN_PROFILES)})")
    names = [s.strip() for s in skip.split(",") if s.strip()]
    bad = sorted(set(names) - set(STEPS))
    if bad:
        raise ValueError(f"unknown --skip step(s): {', '.join(bad)} (valid: {', '.join(STEPS)})")

    cmd = ["bash", str(setup_script())]
    if yes:
        cmd.append("--yes")
    if not upgrade:
        cmd.append("--no-upgrade")
    if keep_gui:
        cmd.append("--keep-gui")
    cmd += ["--fan-profile", fan_profile]
    cmd += ["--swap-size", str(swap_size)]
    if names:
        cmd += ["--skip", ",".join(names)]
    if reboot:
        cmd.append("--reboot")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run_setup(
    *,
    yes: bool = False,
    upgrade: bool = True,
    keep_gui: bool = False,
    fan_profile: str = DEFAULT_FAN_PROFILE,
    swap_size: int = 32,
    skip: str = "",
    reboot: bool = False,
    dry_run: bool = False,
) -> int:
    """Run the Thor setup script; return its exit code (0 = done)."""
    return subprocess.call(
        build_cmd(
            yes=yes,
            upgrade=upgrade,
            keep_gui=keep_gui,
            fan_profile=fan_profile,
            swap_size=swap_size,
            skip=skip,
            reboot=reboot,
            dry_run=dry_run,
        )
    )


# ── Nsight Compute (ncu): the verify stages profile with it ────────────────


def ncu_status() -> tuple[bool, str]:
    """(installed, ncu-path) — the resolved home only counts when the binary
    is actually there (the resolver falls back to the Thor default path when
    nothing is installed)."""
    from mjolnir_cli.verify.docker import resolve_ncu_home

    home = resolve_ncu_home()
    if (Path(home) / "ncu").is_file():
        return True, f"{home}/ncu"
    return False, ""


def ncu_package() -> str:
    """The nsight-compute apt package to install: the cuda-version meta
    matching the box's ``/usr/local/cuda-X.Y``, else the newest plain
    ``nsight-compute-*`` the configured repos carry. Empty when nothing
    matches (no JetPack repo)."""
    import glob
    import re
    import subprocess

    metas = sorted(glob.glob("/usr/local/cuda-*/bin")) if glob.glob("/usr/local/cuda-*/bin") else []
    if metas:
        m = re.search(r"cuda-(\d+)\.(\d+)$", str(Path(metas[-1]).parent))
        if m:
            meta = f"cuda-nsight-compute-{m.group(1)}-{m.group(2)}"
            r = subprocess.run(["apt-cache", "show", meta], capture_output=True)
            if r.returncode == 0 and r.stdout.strip():
                return meta
    r = subprocess.run(
        ["bash", "-c", "apt-cache search --names-only '^nsight-compute-[0-9]' | awk '{print $1}' | sort -V | tail -1"],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


def install_ncu(dry_run: bool = False) -> int:
    """Ensure ncu is present; install the package via interactive sudo when
    missing. Returns the process exit code."""
    ok, path = ncu_status()
    if ok:
        typer.echo(f"✅ ncu found: {path}")
        return 0
    pkg = ncu_package()
    if not pkg:
        typer.secho(
            "no nsight-compute package found — is the NVIDIA JetPack apt repo configured "
            "(/etc/apt/sources.list.d/nvidia-l4t-apt-source.list)?",
            fg=typer.colors.RED,
            err=True,
        )
        return 4
    if not yes_confirm(f"install {pkg} via sudo apt-get?"):
        return 3
    if dry_run:
        typer.echo(f"(dry run) sudo apt-get install -y {pkg}")
        return 0
    rc = subprocess.call(["sudo", "apt-get", "install", "-y", pkg])
    if rc != 0:
        return rc
    ok, path = ncu_status()
    if ok:
        typer.echo(f"✅ ncu installed: {path}")
        return 0
    typer.secho("installed, but no ncu binary under /opt/nvidia/nsight-compute/ — check the package layout", fg=typer.colors.RED, err=True)
    return 4


def yes_confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


# The platforms this Jetson setup covers (the script is the Jetson path —
# thor today, orin the same family). A new platform rows in here or gets its
# own setup; anything else is refused below, never run with the wrong script.
_JETSON_FAMILY = ("thor", "orin")


def dispatch(platform: str, **kw) -> int:
    """Run the HW setup for the detected platform: the Jetson family gets the
    existing script (orin with an explicit untested note); a platform outside
    the family is refused — naming what does exist — rather than run the
    wrong script."""
    if platform not in _JETSON_FAMILY:
        typer.secho(
            f"mjolnir hw setup is not implemented for {platform} yet — what exists today is the "
            f"Jetson setup ({', '.join(_JETSON_FAMILY)}; scripts/hw/setup-thor.sh).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2) from None
    if platform != DEFAULT_PLATFORM:
        typer.secho(
            f"note: {platform} is untested with this setup — it targets Jetson AGX Thor.",
            fg=typer.colors.YELLOW,
        )
    return run_setup(**kw)
