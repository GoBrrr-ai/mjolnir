"""``mjolnir.builder``'s host-side contracts — the pieces the image runs for
both families, pinned where they can import (the Mjolnir image; CI has no
torch/vllm and skips this file via the in-test guards).

* ``_retarget_flashinfer_arch`` — the FLASHINFER_CUDA_ARCH_LIST decision:
  a bake that already targets the serving device must survive *in full*
  (any entry matching suffices — the canonical FlashInfer form is
  ``;``-separated, e.g. ``10.0;11.0a``); a bake that does not target the
  device is retargeted to it; no bake and no CUDA are no-ops. The Thor
  image ships ``11.0a``: on the Thor every canonical form is a no-op, on
  an Orin the same bake retargets to the Orin's cc.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm")

from mjolnir.builder import _retarget_flashinfer_arch  # noqa: E402


def _sim(monkeypatch, cap, baked):
    """Set the device + bake, run the retarget, return the resulting env value."""
    if baked is None:
        monkeypatch.delenv("FLASHINFER_CUDA_ARCH_LIST", raising=False)
    else:
        monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", baked)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cap)
    _retarget_flashinfer_arch()
    return os.environ.get("FLASHINFER_CUDA_ARCH_LIST", "<unset>")


@pytest.mark.parametrize(
    ("cap", "baked", "want"),
    [
        # the Thor image's bake targets sm_110: every canonical form is a no-op
        ((11, 0), "11.0a", "11.0a"),
        ((11, 0), "11.0", "11.0"),  # the plain form
        ((11, 0), "11.0f", "11.0f"),  # the f-suffix form
        ((11, 0), "10.0;11.0a", "10.0;11.0a"),  # multi-arch, ;-separated
        ((11, 0), "10.0,11.0a", "10.0,11.0a"),  # the , variant
        # the same bake on an Orin poisons FI there: retarget to the device
        ((8, 7), "11.0a", "8.7"),
        ((8, 7), "10.0;11.0a", "8.7"),  # no 8.7 entry -> the whole bake is replaced
        ((8, 7), "10.0;8.7;11.0a", "10.0;8.7;11.0a"),  # ...but a bake CONTAINING 8.7 stays
        ((8, 7), "8.7", "8.7"),
        # a device the bake does not mention, the same rule
        ((12, 0), "11.0a", "12.0"),
    ],
)
def test_retarget_flashinfer_arch_decides_per_device(monkeypatch, cap, baked, want):
    assert _sim(monkeypatch, cap, baked) == want


def test_retarget_flashinfer_arch_noop_without_cuda(monkeypatch):
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "11.0a")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _retarget_flashinfer_arch()
    assert os.environ["FLASHINFER_CUDA_ARCH_LIST"] == "11.0a"


def test_retarget_flashinfer_arch_noop_without_a_bake(monkeypatch):
    monkeypatch.delenv("FLASHINFER_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 7))
    _retarget_flashinfer_arch()
    assert "FLASHINFER_CUDA_ARCH_LIST" not in os.environ
