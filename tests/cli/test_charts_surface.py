"""Pins the public/internal surface of mjolnir_cli.charts and its host-side
import hygiene, so that splitting charts.py into a package cannot silently drop
a name other modules rely on or pull matplotlib / the GPU stack in at import."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

SURFACE = (
    "render_all",
    "load_verify",
    "e2e_rows",
    "HERO_BACKGROUND",
    "render_hero",
    "_logotype_box",
    "multiple",
    "chart_spec",
    "chart_kvcache",
    "chart_autoresearch",
    "_plt",
    "_save",
)

HEAVY = ("matplotlib", "torch", "vllm", "cutlass", "flashinfer", "triton")


@pytest.mark.parametrize("name", SURFACE)
def test_charts_exports_name_when_imported(name):
    charts = importlib.import_module("mjolnir_cli.charts")
    assert hasattr(charts, name), f"mjolnir_cli.charts lost {name}"


def test_charts_import_loads_no_matplotlib_or_gpu_stack():
    code = (
        "import sys, mjolnir_cli.charts;"
        f"bad = [m for m in sys.modules if m.split('.')[0] in {HEAVY!r}];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
