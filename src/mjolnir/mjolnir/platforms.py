"""Kernel-side platform leaf: which architecture is this GPU, and which kernel
implements it.

Import-safe everywhere (torch is imported inside :func:`detect` only, never at
module level), so the vLLM backend can consult it and any host can read the
registry without pulling the GPU stack.

``kernel_for`` resolves a compute capability to a dotted module path under
:mod:`mjolnir.arch`; :func:`load_kernel` imports it — heavy (the CUTLASS DSL
comes with it), so only call it when a device is real.
"""

from __future__ import annotations

import os
from types import ModuleType

# The platforms Mjolnir ships support for: (name, sm, implemented). A row
# without a kernel (spark / orin) exists so the unsupported-arch error can
# name the whole matrix; its kernel arrives as mjolnir/arch/<name>.py. The
# kernel package cannot import the CLI's registry, so the sm rows are
# mirrored here — the two sides are pinned in sync by
# tests/cli/test_kernel_platforms.py::test_kernel_and_cli_registries_agree_on_the_sm_rows.
_SHIPPED: tuple[tuple[str, tuple[int, int], bool], ...] = (
    ("thor", (11, 0), True),
    ("spark", (12, 1), False),
    ("orin", (8, 7), False),
)


class PlatformNotSupported(RuntimeError):
    """The GPU's compute capability has no Mjolnir kernel."""


def detect() -> tuple[int, int]:
    """The device's compute capability: ``$MJOLNIR_ARCH`` (``sm_110`` or
    ``11.0``) wins, else torch reads it off the CUDA device."""
    raw = os.environ.get("MJOLNIR_ARCH", "").strip()
    if raw:
        return _parse_arch(raw)
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "Mjolnir cannot detect the GPU compute capability (torch is not importable): "
            "set MJOLNIR_ARCH, e.g. MJOLNIR_ARCH=sm_110"
        ) from e
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Mjolnir cannot detect the GPU compute capability (no CUDA device): "
            "set MJOLNIR_ARCH, e.g. MJOLNIR_ARCH=sm_110"
        )
    cap = torch.cuda.get_device_capability()
    return int(cap[0]), int(cap[1])


def _parse_arch(raw: str) -> tuple[int, int]:
    """``sm_110`` / ``11.0`` / ``8.7`` -> (major, minor)."""
    s = raw.strip().lower().removeprefix("sm_")
    if s.isdigit():  # the sm_ form packs the minor into the last digit: sm_110 = 11.0
        return int(s[:-1]), int(s[-1])
    major, sep, minor = s.partition(".")
    if sep and major.isdigit() and minor.isdigit():
        return int(major), int(minor)
    raise RuntimeError(f"MJOLNIR_ARCH={raw!r}: expected 'sm_<major><minor>' (e.g. sm_110) or '<major>.<minor>'")


def kernel_for(cc: tuple[int, int]) -> str:
    """The dotted module path of the kernel implementing ``cc`` — currently
    only ``mjolnir.arch.thor``; anything else raises :class:`PlatformNotSupported`."""
    for name, sm, implemented in _SHIPPED:
        if sm == tuple(cc):
            if implemented:
                return f"mjolnir.arch.{name}"
            break  # shipped, but the kernel has not landed: unsupported for now
    raise PlatformNotSupported(_unsupported_msg(cc))


def load_kernel(cc: tuple[int, int]) -> ModuleType:
    """Import and return the kernel module for ``cc`` (heavy: pulls the
    CUTLASS DSL — only call this when a device is real)."""
    import importlib

    return importlib.import_module(kernel_for(cc))


def unsupported_reason(cc: tuple[int, int]) -> str | None:
    """None when ``cc`` has a kernel; else the reason the vLLM backend routes
    into its ineligible-config degrade path (names the requested sm and the
    fallback that takes over). Pure — no device access."""
    for _name, sm, implemented in _SHIPPED:
        if sm == tuple(cc) and implemented:
            return None
    return _unsupported_msg(cc) + " Decode steps run on the fallback backend."


def _unsupported_msg(cc: tuple[int, int]) -> str:
    impl = [f"{n} (sm_{maj}{minr})" for n, (maj, minr), ok in _SHIPPED if ok]
    research = [f"{n} (sm_{maj}{minr})" for n, (maj, minr), ok in _SHIPPED if not ok]
    msg = f"Mjolnir has no kernel for sm_{cc[0]}{cc[1]} — implemented: {', '.join(impl)}"
    if research:
        msg += f"; research (no kernel yet): {', '.join(research)}"
    return msg
