#!/usr/bin/env python3
"""Plan a Mjolnir release from the git tag history (used by .github/workflows/release.yml).

Three artifacts version independently:

* ``mjolnir-kernel`` — tags ``mjolnir-kernel-vX.Y.Z`` (source: ``src/mjolnir/``);
* ``mjolnir-cli``    — tags ``mjolnir-cli-vX.Y.Z``    (source: ``src/mjolnir-cli/`` + what
  ``hatch_build.py`` bundles into the wheel: ``configs/``, ``docker/litellm``,
  ``docker/perses``, ``scripts/hw``);
* the Docker image   — tagged after the **pinned upstream vLLM version**
  (``VLLM_VERSION`` in ``mjolnir_cli/config.py``, which must equal the
  Dockerfile's ``BASE_IMAGE``), never after a package version.

A release is one run: each selected package gets its next version — the
latest tag bumped by ``--bump`` (patch / minor / major), or the version in its
``__init__.py`` when that is already higher (a hand bump) or no tag exists yet
(the first release) — plus one umbrella tag ``release-YYYY.MM.DD[.N]`` that
carries the GitHub Release (notes: ``scripts/release_notes.py``).

``--packages changed`` (the default) releases a package only when its source
changed since its last tag; ``both`` / ``cli`` / ``kernel`` / ``none`` force it.

Prints a JSON plan, and with ``--github-output`` also writes ``key=value``
lines for the workflow.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PACKAGES = {
    "kernel": {
        "dist": "mjolnir-kernel",
        "prefix": "mjolnir-kernel-v",
        "init": "src/mjolnir/mjolnir/__init__.py",
        "paths": ["src/mjolnir"],
    },
    "cli": {
        "dist": "mjolnir-cli",
        "prefix": "mjolnir-cli-v",
        "init": "src/mjolnir-cli/mjolnir_cli/__init__.py",
        # the wheel: mjolnir_cli + exactly hatch_build.py's BUNDLED map
        "paths": ["src/mjolnir-cli", "configs", "docker/litellm", "docker/perses", "scripts/hw"],
    },
}
# What the Docker image ships from this repo (the Dockerfile's COPY lines) —
# not the docs next to it (README.md, PATCHES.md are rendered, not installed).
# The kernel (src/mjolnir) is in the image too; the notes reference it.
IMAGE_PATHS = [
    "docker/vllm-thor/Dockerfile",
    "docker/vllm-thor/patches",
    "docker/vllm-thor/apply_patches.py",
    "docker/vllm-thor/verify_patches.py",
    "docker/vllm-thor/functional_check_55519.py",
    "docker/vllm-thor/kernel_test_fi_update.py",
    "docker/vllm-thor/kernel_test_dspark_draft_nonncausal.py",
    "LICENSE",
    "NOTICE",
]
UMBRELLA = "release-"
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def parse(v: str) -> tuple[int, int, int] | None:
    m = SEMVER.match(v.strip())
    return tuple(int(x) for x in m.groups()) if m else None  # type: ignore[return-value]


def fmt(v: tuple[int, int, int]) -> str:
    return ".".join(map(str, v))


def bump(v: tuple[int, int, int], level: str) -> tuple[int, int, int]:
    major, minor, patch = v
    if level == "major":
        return (major + 1, 0, 0)
    if level == "minor":
        return (major, minor + 1, 0)
    return (major, minor, patch + 1)


def last_tag(repo: Path, prefix: str) -> tuple[str, tuple[int, int, int]] | None:
    """The highest final-release tag with ``prefix`` (pre-releases ignored)."""
    best = None
    for tag in git(repo, "tag", "--list", f"{prefix}*").splitlines():
        v = parse(tag[len(prefix) :])
        if v and (best is None or v > best[1]):
            best = (tag, v)
    return best


def file_version(repo: Path, init: str) -> tuple[int, int, int]:
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', (repo / init).read_text(), re.MULTILINE)
    v = parse(m.group(1)) if m else None
    if v is None:
        raise SystemExit(f"{init}: no X.Y.Z __version__")
    return v


def changed_since(repo: Path, tag: str, paths: list[str]) -> bool:
    return bool(git(repo, "diff", "--name-only", f"{tag}..HEAD", "--", *paths))


def vllm_version(repo: Path) -> str:
    cfg = re.search(
        r'^VLLM_VERSION\s*=\s*"([^"]+)"', (repo / "src/mjolnir-cli/mjolnir_cli/config.py").read_text(), re.MULTILINE
    )
    dock = re.search(
        r"^ARG BASE_IMAGE=vllm/vllm-openai:v([0-9.]+)", (repo / "docker/vllm-thor/Dockerfile").read_text(), re.MULTILINE
    )
    if not cfg or not dock:
        raise SystemExit("cannot read VLLM_VERSION (config.py) or BASE_IMAGE (Dockerfile)")
    if cfg.group(1) != dock.group(1):
        raise SystemExit(f"config.py VLLM_VERSION {cfg.group(1)} != Dockerfile BASE_IMAGE {dock.group(1)}: bump both")
    return cfg.group(1)


def previous_umbrella(repo: Path) -> str | None:
    """The latest umbrella tag (release-YYYY.MM.DD[.N]), by date then N."""

    def key(tag: str):
        parts = tag[len(UMBRELLA) :].split(".")
        return tuple(int(x) for x in parts if x.isdigit()) + ((1,) if len(parts) == 3 else ())

    tags = [t for t in git(repo, "tag", "--list", f"{UMBRELLA}*").splitlines() if t]
    return max(tags, key=key) if tags else None


def title(date: str, packages: dict, vllm: str, image: bool) -> str:
    """``Mjolnir 2026.10.20 — mjolnir-kernel 0.1.1 · image v0.31.0``: only what this release ships."""
    parts = [f"{v['dist']} {v['version']}" for v in packages.values() if v["release"]]
    if image:
        parts.append(f"image v{vllm}")
    return f"Mjolnir {date} — " + (" · ".join(parts) if parts else "no package or image changes")


def umbrella_tag(repo: Path, today: dt.date) -> str:
    base = f"{UMBRELLA}{today:%Y.%m.%d}"
    taken = set(git(repo, "tag", "--list", f"{base}*").splitlines())
    if base not in taken:
        return base
    n = 2
    while f"{base}.{n}" in taken:
        n += 1
    return f"{base}.{n}"


def plan(repo: Path, level: str, packages: str, today: dt.date | None = None, image: bool = True) -> dict:
    out: dict = {
        "vllm": vllm_version(repo),
        "image": image,
        "previous_umbrella": previous_umbrella(repo),
        "packages": {},
    }
    for key, p in PACKAGES.items():
        prev = last_tag(repo, p["prefix"])
        have = file_version(repo, p["init"])
        if packages == "none" or (packages in PACKAGES and packages != key):
            selected, why = False, "not selected"
        elif packages == "changed" and prev and not changed_since(repo, prev[0], p["paths"]):
            selected, why = False, f"unchanged since {prev[0]}"
        else:
            selected, why = True, "first release" if prev is None else f"changed since {prev[0]}"
        nxt = have if prev is None else max(bump(prev[1], level), have)
        out["packages"][key] = {
            "dist": p["dist"],
            "release": selected,
            "reason": why,
            "previous": prev[0] if prev else None,
            "version": fmt(nxt) if selected else None,
            "tag": f"{p['prefix']}{fmt(nxt)}" if selected else None,
        }
    # The release happens in the operator's local day; an aware local date
    # (not the naive UTC date) names the umbrella tag.
    out["umbrella"] = umbrella_tag(repo, today or dt.datetime.now().astimezone().date())
    out["title"] = title(out["umbrella"][len(UMBRELLA) :], out["packages"], out["vllm"], image)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--bump", choices=("patch", "minor", "major"), default="patch")
    ap.add_argument("--packages", choices=("changed", "both", "cli", "kernel", "none"), default="changed")
    ap.add_argument("--image", choices=("true", "false"), default="true", help="is the image rebuilt (the title)")
    ap.add_argument("--github-output", action="store_true", help="also append key=value lines to $GITHUB_OUTPUT")
    args = ap.parse_args(argv)
    p = plan(Path(args.repo), args.bump, args.packages, image=args.image == "true")
    print(json.dumps(p, indent=2, ensure_ascii=False))
    if args.github_output and os.environ.get("GITHUB_OUTPUT"):
        k, c = p["packages"]["kernel"], p["packages"]["cli"]
        lines = {
            "vllm": p["vllm"],
            "umbrella": p["umbrella"],
            "title": p["title"],
            "kernel_release": str(k["release"]).lower(),
            "kernel_version": k["version"] or "",
            "kernel_tag": k["tag"] or "",
            "cli_release": str(c["release"]).lower(),
            "cli_version": c["version"] or "",
            "cli_tag": c["tag"] or "",
            "kernel_previous": k["previous"] or "",
            "cli_previous": c["previous"] or "",
            "previous_umbrella": p["previous_umbrella"] or "",
        }
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.writelines(f"{key}={val}\n" for key, val in lines.items())
    return 0


if __name__ == "__main__":
    sys.exit(main())
