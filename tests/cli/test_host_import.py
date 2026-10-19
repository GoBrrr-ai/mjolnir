"""The CLI runs on the host; torch / CUDA / the CuTe DSL only ever run inside
the vLLM container. Importing every CLI module must not pull them in."""

from __future__ import annotations

import subprocess
import sys

GPU_STACK = ("torch", "vllm", "cutlass", "flashinfer", "triton")


def test_cli_import_is_gpu_free():
    code = (
        "import sys, mjolnir_cli.cli, mjolnir_cli.verify.cli, mjolnir_cli.autoresearch,"
        " mjolnir_cli.platforms, mjolnir_cli.cli_platform, mjolnir_cli.availability;"
        # the new platform leaves stay torch-lazy too; availability (cycle 3a)
        # is a stdlib-only data leaf every renderer reads
        f"bad = [m for m in sys.modules if m.split('.')[0] in {GPU_STACK!r}];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_availability_import_is_gpu_free():
    """The availability tables are the one module serve-validation, leg
    filtering and every cycle-3 renderer read — they must import on the host
    (and anywhere else) without dragging in torch/vllm/cutlass."""
    code = (
        "import sys, mjolnir_cli.availability;"
        f"bad = [m for m in sys.modules if m.split('.')[0] in {GPU_STACK!r}];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
