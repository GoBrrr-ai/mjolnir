"""Shared detection fakes for the cycle-2 (detection-first) tests.

The CLI probes the host (nvidia-smi, device-tree files, DMI) when no
``MJOLNIR_PLATFORM`` is set. Unit tests must never touch real ``/proc``,
``/sys`` or ``nvidia-smi`` (the CI host has no GPU), so every detection
test installs these fakes instead — either by passing fake sources
straight into :func:`mjolnir_cli.platforms.detect` or by replacing the
module-level ``DEFAULT_SOURCES`` the CLI resolves through.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from mjolnir_cli import platforms

# The research probe files (task/research/detection.md): what the file-reading
# source classes are expected to ask the filesystem for. The world-writable
# /tmp model copy is its own source (the cycle-2 security audit LOW) — it
# corroborates only, so it is not part of the jetson-files evidence class.
JETSON_PATHS = (
    "/proc/device-tree/compatible",
    "/proc/device-tree/model",
    "/etc/nv_tegra_release",
)
MODEL_FILE_PATHS = ("/tmp/nv_jetson_model",)
DMI_PATHS = ("/sys/class/dmi/id/product_name", "/etc/dgx-release")


class FakeSource:
    """A ``PlatformSource`` fake: a canned raw value plus a probe counter."""

    def __init__(self, name: str, raw: str | None, corroboration_required: bool = False) -> None:
        self.name = name
        self._raw = raw
        self.calls = 0
        self.corroboration_required = corroboration_required

    def probe(self) -> str | None:
        self.calls += 1
        return self._raw


def fake_sources(
    smi: str | None = "NVIDIA Thor, 11.0",
    jetson: str | None = None,
    dmi: str | None = None,
    model_file: str | None = None,
):
    """The four source classes as fakes; ``None`` raw = the probe finds nothing.

    The ``nv_jetson_model`` fake is last (probed last) and carries
    ``corroboration_required`` like the real class."""
    return (
        FakeSource("nvidia-smi", smi),
        FakeSource("jetson-files", jetson),
        FakeSource("dmi", dmi),
        FakeSource("nv_jetson_model", model_file, corroboration_required=True),
    )


def install(
    monkeypatch,
    smi: str | None = None,
    jetson: str | None = None,
    dmi: str | None = None,
    model_file: str | None = None,
):
    """Detection answers only from these fakes: no env var, no real probes.

    ``raising=False`` so the RED suite fails on behaviour, not on the
    missing module attribute."""
    monkeypatch.delenv("MJOLNIR_PLATFORM", raising=False)
    srcs = fake_sources(smi=smi, jetson=jetson, dmi=dmi, model_file=model_file)
    monkeypatch.setattr(platforms, "DEFAULT_SOURCES", srcs, raising=False)
    return srcs


def fake_smi(monkeypatch, stdout: str = "NVIDIA Thor, 11.0\n") -> None:
    """Point every nvidia-smi subprocess call at a canned csv answer."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, stdout=stdout))


def fake_reader(files: dict[str, str], asked: list[str] | None = None) -> Callable[[Path], str | None]:
    """A file-reader fake for the file-reading source classes: paths -> contents."""
    seen = asked if asked is not None else []

    def read(path: Path) -> str | None:
        seen.append(str(path))
        return files.get(str(path))

    return read
