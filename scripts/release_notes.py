#!/usr/bin/env python3
"""The GitHub Release body of an umbrella release (used by .github/workflows/release.yml).

One section per artifact the release ships, each sliced from git history by
the paths that artifact is built from (``release_plan.PACKAGES`` /
``IMAGE_PATHS``) since that artifact's own previous release:

* ``mjolnir-kernel`` — ``src/mjolnir`` since its last ``mjolnir-kernel-v*`` tag;
* ``mjolnir-cli``    — ``src/mjolnir-cli`` + the bundled resources since its last
  ``mjolnir-cli-v*`` tag;
* the image          — the files the Dockerfile copies since the previous
  umbrella tag, plus where its base and kernel come from (the kernel's changes
  are listed once, under the kernel).

An entry is a pull request (``#N`` + its title, from the merge commit on
main's first-parent line) or a commit pushed to main directly; Dependabot
updates and other merges are left out. Nothing is written to the repository —
the body goes to stdout / ``--out``.

    python3 scripts/release_notes.py --umbrella release-2026.10.20 \\
        --kernel-version 0.1.1 --kernel-previous mjolnir-kernel-v0.1.0 \\
        --cli-previous mjolnir-cli-v0.2.0 --previous-umbrella release-2026.10.15 \\
        --vllm 0.31.0 --image true
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("release_plan", Path(__file__).with_name("release_plan.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)  # type: ignore[union-attr]

PR_MERGE = re.compile(r"^Merge pull request #(\d+) from (\S+)")
SEP, END = "\x1f", "\x1e"
MAX_UNTAGGED = 30  # a first release (no previous tag): the newest entries only


def entries(repo: Path, base: str | None, paths: list[str]) -> list[str]:
    """``- <PR title> (#N)`` / ``- <subject>`` bullets for main's first-parent
    history since ``base`` that touched ``paths``, newest first."""
    rng = [f"{base}..HEAD"] if base else ["HEAD"]
    raw = rp.git(
        repo,
        "log",
        "--first-parent",
        f"--format=%P{SEP}%an{SEP}%s{SEP}%b{END}",
        *rng,
        "--",
        *paths,
    )
    out: list[str] = []
    for rec in raw.split(END):
        rec = rec.strip("\n")
        if not rec:
            continue
        parents, author, subject, body = (rec.split(SEP) + ["", "", "", ""])[:4]
        merge = len(parents.split()) > 1
        if merge:
            m = PR_MERGE.match(subject)
            if not m or "dependabot/" in m.group(2):
                continue  # a non-PR merge, or a Dependabot PR
            title = next((ln.strip() for ln in body.splitlines() if ln.strip()), m.group(2))
            out.append(f"- {title} (#{m.group(1)})")
        else:
            if "dependabot" in author.lower():
                continue
            out.append(f"- {subject}")
    return out if base else out[:MAX_UNTAGGED]


def base_vllm(repo: Path, ref: str | None) -> str | None:
    """The upstream vLLM version the Dockerfile at ``ref`` builds on."""
    if not ref:
        return None
    try:
        text = rp.git(repo, "show", f"{ref}:docker/vllm-thor/Dockerfile")
    except Exception:  # noqa: BLE001 — the file did not exist there
        return None
    m = re.search(r"^ARG BASE_IMAGE=vllm/vllm-openai:v([0-9.]+)", text, re.M)
    return m.group(1) if m else None


def _version_of(tag: str | None, prefix: str) -> str | None:
    return tag[len(prefix) :] if tag else None


def body(
    repo: Path,
    *,
    vllm: str,
    image: bool,
    previous_umbrella: str | None,
    kernel: dict,
    cli: dict,
) -> str:
    """``kernel`` / ``cli``: {"version": "0.1.1" or None (not released), "previous": tag or None}."""
    kp, cp = rp.PACKAGES["kernel"], rp.PACKAGES["cli"]
    k_prev = _version_of(kernel.get("previous"), kp["prefix"])
    c_prev = _version_of(cli.get("previous"), cp["prefix"])
    k_new, c_new = kernel.get("version"), cli.get("version")

    lines = ["## Install", "", "```bash"]
    if c_new:
        lines.append(f"uv tool install mjolnir-cli=={c_new}")
    if k_new:
        lines.append(f"pip install mjolnir-kernel=={k_new}   # stock vLLM {vllm}: --attention-backend CUSTOM")
    if image:
        lines.append(f"docker pull narandill/mjolnir-vllm:v{vllm}")
    if not (c_new or k_new or image):
        lines.append("# no package or image in this release")
    lines += ["```", ""]

    def cell(new: str | None, prev: str | None) -> str:
        return f"**{new}**" if new else f"unchanged ({prev or '—'})"

    lines += [
        "| artifact | version |",
        "|---|---|",
        f"| [mjolnir-kernel](https://pypi.org/project/mjolnir-kernel/) | {cell(k_new, k_prev)} |",
        f"| [mjolnir-cli](https://pypi.org/project/mjolnir-cli/) | {cell(c_new, c_prev)} |",
        "| [narandill/mjolnir-vllm](https://hub.docker.com/r/narandill/mjolnir-vllm) | "
        + (f"**v{vllm}** rebuilt (vLLM {vllm})" if image else f"unchanged (v{vllm}, vLLM {vllm})")
        + " |",
        "",
    ]

    def section(head: str, items: list[str], empty: str) -> None:
        lines.extend([f"## {head}", *(items or [f"- {empty}"]), ""])

    if k_new:
        section(
            f"mjolnir-kernel {k_new}",
            entries(repo, kernel.get("previous"), kp["paths"]),
            "no changes since the last release (re-released)",
        )
    if c_new:
        section(
            f"mjolnir-cli {c_new}",
            entries(repo, cli.get("previous"), cp["paths"]),
            "no changes since the last release (re-released)",
        )
    if image:
        was = base_vllm(repo, previous_umbrella)
        items = []
        if was and was != vllm:
            items.append(f"- base bumped to vLLM {vllm} (from {was}); patch stack re-verified at build")
        else:
            items.append(f"- rebuilt on vLLM {vllm}")
        if k_new:
            items.append(f"- ships mjolnir-kernel {k_new} (see above)")
        elif k_prev and rp.changed_since(repo, kernel["previous"], kp["paths"]):
            items.append(
                f"- ships the kernel source of this commit, stamped {k_prev}: it has changes not released "
                "to PyPI (release the kernel too, or rebuild later)"
            )
        else:
            items.append(f"- ships mjolnir-kernel {k_prev or '—'} (unchanged)")
        items += entries(repo, previous_umbrella, rp.IMAGE_PATHS)
        section(f"image v{vllm}", items, "")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--image", choices=("true", "false"), default="true")
    ap.add_argument("--previous-umbrella", default="")
    ap.add_argument("--kernel-version", default="", help="empty: the kernel is not released")
    ap.add_argument("--kernel-previous", default="")
    ap.add_argument("--cli-version", default="", help="empty: the CLI is not released")
    ap.add_argument("--cli-previous", default="")
    ap.add_argument("--out", default="", help="write the body here (default: stdout)")
    a = ap.parse_args(argv)
    text = body(
        Path(a.repo),
        vllm=a.vllm,
        image=a.image == "true",
        previous_umbrella=a.previous_umbrella or None,
        kernel={"version": a.kernel_version or None, "previous": a.kernel_previous or None},
        cli={"version": a.cli_version or None, "previous": a.cli_previous or None},
    )
    if a.out:
        Path(a.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
