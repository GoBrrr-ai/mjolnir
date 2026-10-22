"""The kernel-side platform leaf (``mjolnir.platforms``) — arch detection and
kernel resolution that never touches the GPU stack at import time.

The spec (plan §1), pinned as tests:

* importing the module leaves torch / cutlass / vllm out of ``sys.modules``;
* ``detect()`` — the compute capability, with the ``MJOLNIR_ARCH`` override
  (``"11.0"`` / ``"sm_110"``); without torch it fails with a clean message,
  not a traceback from a missing module;
* ``kernel_for()`` — ``(11, 0)`` resolves the dotted path of the Thor
  implementation; anything else raises ``PlatformNotSupported`` naming the
  requested sm *and* the shipped platforms;
* ``unsupported_reason()`` — the pure string the vLLM builder routes into its
  ``ineligible_reason`` degrade path (``None`` when supported, mentions the
  fallback otherwise);
* ``load_kernel()`` and ``mjolnir.arch.thor`` expose exactly the kernel
  surface ``backend.py`` consumes (guarded: they pull the CUTLASS DSL, so
  they only run where it is installed);
* the kernel↔CLI registry mirror: every shipped platform's ``sm`` row agrees
  between ``mjolnir.platforms._SHIPPED`` and ``mjolnir_cli.platforms.REGISTRY``
  (the packages cannot import each other — the pin is the sync).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import mjolnir.platforms as kp
import mjolnir_cli.platforms as cp

GPU_STACK = ("torch", "vllm", "cutlass", "flashinfer", "triton")

# The kernel surface backend.py consumes — mjolnir.arch.thor must re-export
# exactly these names, as the same objects. The runner class itself is
# resolved through the uniform `Runner` alias (pinned below).
T5_SURFACE = ("T5DecodeRunner", "StepPlan", "SUPPORTED_PAGE_SIZES", "detect_kv_layout", "env_flag", "split_kv_cache")


@pytest.fixture(autouse=True)
def _clean_arch_env(monkeypatch):
    monkeypatch.delenv("MJOLNIR_ARCH", raising=False)
    monkeypatch.delenv("MJOLNIR_PLATFORM", raising=False)


def test_importing_the_module_is_gpu_stack_free():
    code = (
        "import sys, mjolnir.platforms;"
        f"bad = [m for m in sys.modules if m.split('.')[0] in {GPU_STACK!r}];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_kernel_for_resolves_thor_and_load_kernel_exposes_the_t5_surface():
    pytest.importorskip("cutlass")
    assert kp.kernel_for((11, 0)) == "mjolnir.arch.thor"
    mod = kp.load_kernel((11, 0))
    assert mod.__name__ == "mjolnir.arch.thor"
    for name in T5_SURFACE:
        assert hasattr(mod, name), name


@pytest.mark.parametrize("cc", [(8, 7), (12, 1)])
def test_kernel_for_unsupported_arch_names_the_sm_and_the_shipped_platforms(cc):
    with pytest.raises(kp.PlatformNotSupported) as ei:
        kp.kernel_for(cc)
    msg = str(ei.value)
    assert f"sm_{cc[0]}{cc[1]}" in msg  # sm_87 / sm_121
    assert all(k in msg for k in ("thor", "spark", "orin"))  # the shipped platforms


@pytest.mark.parametrize(("arch", "want"), [("11.0", (11, 0)), ("sm_110", (11, 0)), ("8.7", (8, 7))])
def test_detect_parses_the_mjolnir_arch_override(monkeypatch, arch, want):
    monkeypatch.setenv("MJOLNIR_ARCH", arch)
    assert kp.detect() == want


def test_detect_without_torch_fails_with_a_clean_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` raises ImportError
    with pytest.raises(RuntimeError, match="torch"):
        kp.detect()


def test_unsupported_reason_is_none_for_thor_and_names_the_fallback_otherwise():
    assert kp.unsupported_reason((11, 0)) is None
    for cc in ((8, 7), (12, 1)):
        reason = kp.unsupported_reason(cc)
        assert reason, cc
        assert f"sm_{cc[0]}{cc[1]}" in reason
        assert "fallback" in reason.lower()  # the degrade path it routes into


def test_arch_thor_reexports_the_kernel_surface_backend_consumes():
    pytest.importorskip("cutlass")
    from mjolnir import runner
    from mjolnir.arch import thor

    for name in T5_SURFACE:
        assert getattr(thor, name) is getattr(runner, name), name


def test_arch_thor_exposes_the_uniform_runner_name_the_backend_resolves():
    """``Runner`` is the one name a family subpackage must define itself —
    backend.py resolves its runner class through it (load_kernel(cc).Runner),
    so landing a family edits nothing but platforms._SHIPPED."""
    pytest.importorskip("cutlass")
    from mjolnir.arch import thor

    assert thor.Runner is thor.T5DecodeRunner


# ── the kernel↔CLI registry mirror (review IMPROVEMENT #2) ─────────────────
# _SHIPPED mirrors cp.REGISTRY because the kernel package cannot import the
# CLI — this pin is the only thing keeping the duplicated sm rows in sync.

_KERNEL_SM = {name: sm for name, sm, _ in kp._SHIPPED}
_CLI_SM = {key: p.sm for key, p in cp.REGISTRY.items()}


@pytest.mark.parametrize("key", sorted(_KERNEL_SM.keys() | _CLI_SM.keys()))
def test_kernel_and_cli_registries_agree_on_the_sm_rows(key):
    """Every shipped platform has the same ``sm`` row on both sides of the
    mirror. A platform added to one package only — or with a drifted sm —
    fails here, naming the missing side. Regression guard, like
    ``test_thor_legs_are_unchanged``: green against today's data by design."""
    in_kernel, in_cli = key in _KERNEL_SM, key in _CLI_SM
    assert in_kernel and in_cli, (
        f"platform {key!r} exists on one side of the mirror only "
        f"(mjolnir._SHIPPED: {in_kernel}, mjolnir_cli.REGISTRY: {in_cli}) — add it to both"
    )
    assert _KERNEL_SM[key] == _CLI_SM[key], f"{key}: sm {_KERNEL_SM[key]} != {_CLI_SM[key]}"
