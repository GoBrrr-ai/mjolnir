"""The license files shipped inside each package stay identical to the repo's."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_package_license_copies_match_the_repo():
    for pkg in ("src/mjolnir", "src/mjolnir-cli"):
        for name in ("LICENSE", "NOTICE"):
            assert (ROOT / pkg / name).read_text() == (ROOT / name).read_text(), (
                f"{pkg}/{name} is stale: cp {name} {pkg}/"
            )


def test_kernel_keeps_the_upstream_bsd_notice():
    head = (ROOT / "src/mjolnir/mjolnir/arch/thor/decode.py").read_text()[:3000]
    assert "Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved." in head
    assert "SPDX-License-Identifier: BSD-3-Clause" in head
    assert "Redistributions of source code must retain the above copyright notice" in head
