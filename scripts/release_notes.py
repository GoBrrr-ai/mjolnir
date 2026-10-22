#!/usr/bin/env python3
"""The GitHub Release body of an umbrella release (used by .github/workflows/release.yml).

One deduplicated story: every pull request appears exactly once, under a
group named after the set of artifacts it ships in. Artifacts are sliced from
git history by the paths each is built from (``release_plan.PACKAGES`` /
``IMAGE_PATHS``), each since its own previous release:

* ``mjolnir-kernel`` — ``src/mjolnir`` since its last ``mjolnir-kernel-v*`` tag;
* ``mjolnir-cli``    — ``src/mjolnir-cli`` + the bundled resources since its
  last ``mjolnir-cli-v*`` tag;
* the image          — the files the Dockerfile copies since the previous
  umbrella tag.

A PR that touched several artifacts' paths lands once, in a group whose
header names the set (``mjolnir-kernel 0.1.1 + mjolnir-cli 0.2.1``) instead
of repeating it under each artifact. PRs in the umbrella range that touch no
artifact's paths (docs, research, CI, the release scripts themselves) are
grouped under ``Repo, docs & tooling`` instead of being dropped. The image
group carries the vLLM base / shipped-kernel lines, and a trailing
``Platform notes`` matrix maps each PR to Thor / Orin / Spark from its
changed paths (explicit signals only — the platform overlays and docs,
``scripts/hw``, ``AWQ_FLASHINFER``; plus the kernel package itself, which
serves Thor's sm_110 only; the image is the shared artifact and maps to no
platform).

An entry is a pull request (``#N`` + its title, from the merge commit on
main's first-parent line) or a commit pushed to main directly; Dependabot
updates and other merges are left out. Nothing is written to the repository —
the body goes to stdout / ``--out``.

    python3 scripts/release_notes.py --vllm 0.31.0 \\
        --kernel-version 0.1.1 --kernel-previous mjolnir-kernel-v0.1.0 \\
        --cli-version 0.2.1 --cli-previous mjolnir-cli-v0.2.0 \\
        --previous-umbrella release-2026.10.15 --image true
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_spec = importlib.util.spec_from_file_location("release_plan", Path(__file__).with_name("release_plan.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)  # type: ignore[union-attr]

PR_MERGE = re.compile(r"^Merge pull request #(\d+) from (\S+)")
SEP, END = "\x1f", "\x1e"
MAX_UNTAGGED = 30  # a first release (no previous tag): the newest entries only

# Artifact -> the paths it is built from (release_plan is the single source).
SLICES: dict[str, list[str]] = {
    "kernel": rp.PACKAGES["kernel"]["paths"],
    "cli": rp.PACKAGES["cli"]["paths"],
    "image": rp.IMAGE_PATHS,
}
ORDER = ("kernel", "cli", "image")
PLATFORM_ROWS = ("thor", "orin", "spark")


def _platforms(paths: frozenset[str]) -> frozenset[str]:
    """Explicit platform signals only, so the matrix stays meaningful: a PR
    touching only shared surfaces (the image, the CLI, CI) maps to no
    platform — the image is the artifact every platform shares."""
    out: set[str] = set()
    for p in paths:
        if p.startswith(("src/mjolnir/", "scripts/hw/", "docs/platforms/thor/")) or (
            p.startswith("configs/") and "/platforms/thor" in p
        ):
            out.add("thor")  # the Mjolnir kernel serves sm_110 (Thor) only
        if p.startswith("docs/platforms/orin/") or (
            p.startswith("configs/") and ("/platforms/orin" in p or "/AWQ_FLASHINFER/" in p)
        ):
            out.add("orin")  # AWQ_FLASHINFER is the Orin serving config
        if p.startswith("docs/platforms/spark/") or (p.startswith("configs/") and "/platforms/spark" in p):
            out.add("spark")
    return frozenset(out)


@dataclass
class Entry:
    """One pooled change: a PR (key ``#N``) or a direct push to main (key = sha)."""

    key: str
    sha: str
    title: str
    date: int
    files: frozenset[str] = field(default_factory=frozenset)
    artifacts: set[str] = field(default_factory=set)
    platforms: frozenset[str] = field(default_factory=frozenset)

    @property
    def ref(self) -> str:
        """``#N`` for a PR, the abbreviated sha for a direct push."""
        return self.key if self.key.startswith("#") else self.sha[:7]

    def bullet(self) -> str:
        return f"- {self.title} ({self.key})" if self.key.startswith("#") else f"- {self.title}"


def _log(repo: Path, base: str | None) -> list[tuple[str, list[str], str | None, str, int]]:
    """``(sha, parents, '#N' | None, title, date)`` for main's first-parent
    history since ``base`` (``None``: everything), newest first — Dependabot
    and non-PR merges left out, like the old slicing. The caller applies the
    first-release cap after its own slice filtering."""
    rng = [f"{base}..HEAD"] if base else ["HEAD"]
    raw = rp.git(repo, "log", "--first-parent", f"--format=%P{SEP}%H{SEP}%ct{SEP}%an{SEP}%s{SEP}%b{END}", *rng)
    out: list[tuple[str, list[str], str | None, str, int]] = []
    for rec in raw.split(END):
        rec = rec.strip("\n")
        if not rec:
            continue
        parents, sha, date, author, subject, body = (*rec.split(SEP), "", "", "", "", "", "")[:6]
        plist = parents.split()
        if len(plist) > 1:  # a merge commit
            m = PR_MERGE.match(subject)
            if not m or "dependabot/" in m.group(2):
                continue  # a non-PR merge, or a Dependabot PR
            title = next((ln.strip() for ln in body.splitlines() if ln.strip()), m.group(2))
            out.append((sha, plist, f"#{m.group(1)}", title, int(date)))
        else:
            if "dependabot" in author.lower():
                continue
            out.append((sha, plist, None, subject, int(date)))
    return out


def _files(repo: Path, sha: str, parents: list[str]) -> frozenset[str]:
    """Every path a commit changed — a merge against its FIRST parent only,
    i.e. what the merged branch actually touched: diffing against all parents
    would absorb changes main gained from other commits since the branch
    point (the old pathspec slicing never saw those)."""
    if parents:
        raw = rp.git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", parents[0], sha)
    else:  # a root commit: the whole tree
        raw = rp.git(repo, "diff-tree", "--no-commit-id", "--name-only", "--root", "-r", sha)
    return frozenset(ln for ln in raw.splitlines() if ln.strip())


def _in_slice(files: frozenset[str], slice_paths: list[str]) -> bool:
    """Git-pathspec semantics: a directory path matches everything under it."""
    return any(f == p or f.startswith(p.rstrip("/") + "/") for p in slice_paths for f in files)


def collect(
    repo: Path, kernel_base: str | None, cli_base: str | None, umbrella_base: str | None
) -> tuple[dict[str, Entry], list[str], list[str]]:
    """The deduplicated entry pool with its first-seen key order, plus the
    keys of the umbrella-range entries that touch no artifact's paths (the
    Repo/docs group). An entry's artifacts are the slices it touched within
    each slice's own range — "new for that artifact's version"."""
    pool: dict[str, Entry] = {}
    order: list[str] = []
    cache: dict[str, frozenset[str]] = {}

    def files(sha: str, parents: list[str]) -> frozenset[str]:
        if sha not in cache:
            cache[sha] = _files(repo, sha, parents)
        return cache[sha]

    for name, base in (("kernel", kernel_base), ("cli", cli_base), ("image", umbrella_base)):
        matched = 0
        for sha, parents, key, title, date in _log(repo, base):
            fs = files(sha, parents)
            if not _in_slice(fs, SLICES[name]):
                continue
            k = key or sha
            e = pool.get(k)
            if e is None:
                e = pool[k] = Entry(key=k, sha=sha, title=title, date=date, files=fs, platforms=_platforms(fs))
                order.append(k)
            e.artifacts.add(name)
            matched += 1
            if base is None and matched >= MAX_UNTAGGED:
                break  # a first release: the newest matching entries only

    catchall: list[str] = []
    for sha, parents, key, title, date in _log(repo, umbrella_base):
        k = key or sha
        if k in pool:
            continue
        fs = files(sha, parents)
        if any(_in_slice(fs, paths) for paths in SLICES.values()):
            continue
        pool[k] = Entry(key=k, sha=sha, title=title, date=date, files=fs, platforms=_platforms(fs))
        order.append(k)
        catchall.append(k)
        if umbrella_base is None and len(catchall) >= MAX_UNTAGGED:
            break
    return pool, order, catchall


def base_vllm(repo: Path, ref: str | None) -> str | None:
    """The upstream vLLM version the Dockerfile at ``ref`` builds on."""
    if not ref:
        return None
    try:
        text = rp.git(repo, "show", f"{ref}:docker/vllm-thor/Dockerfile")
    except Exception:  # noqa: BLE001 — the file did not exist there
        return None
    m = re.search(r"^ARG BASE_IMAGE=vllm/vllm-openai:v([0-9.]+)", text, re.MULTILINE)
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
    shipped = {"kernel": bool(k_new), "cli": bool(c_new), "image": image}
    versions = {"kernel": k_new, "cli": c_new, "image": f"v{vllm}"}
    labels = {"kernel": "mjolnir-kernel", "cli": "mjolnir-cli", "image": "image"}

    pool, order, catchall = collect(repo, kernel.get("previous"), cli.get("previous"), previous_umbrella or None)

    def rendered(e: Entry) -> tuple[str, ...]:
        """The artifact set a bullet ships in — only what this release ships."""
        return tuple(a for a in ORDER if a in e.artifacts and shipped[a])

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

    def cell(name: str) -> str:
        if not shipped[name]:
            prev = {"kernel": k_prev, "cli": c_prev}.get(name)
            return f"unchanged ({prev or '—'})"
        n = sum(1 for e in pool.values() if name in e.artifacts)
        return f"**{versions[name]}**" + ("" if n else " (re-released, no changes)")

    lines += [
        "| artifact | version |",
        "|---|---|",
        f"| [mjolnir-kernel](https://pypi.org/project/mjolnir-kernel/) | {cell('kernel')} |",
        f"| [mjolnir-cli](https://pypi.org/project/mjolnir-cli/) | {cell('cli')} |",
        "| [narandill/mjolnir-vllm](https://hub.docker.com/r/narandill/mjolnir-vllm) | "
        + (f"**v{vllm}** rebuilt (vLLM {vllm})" if image else f"unchanged (v{vllm}, vLLM {vllm})")
        + " |",
        "",
    ]

    def group_key(g: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
        return (-len(g), g)  # release-wide sets lead, then kernel < cli < image

    groups: dict[tuple[str, ...], list[Entry]] = {}
    for k in order:
        r = rendered(pool[k])
        if r:
            groups.setdefault(r, []).append(pool[k])

    syn: dict[tuple[str, ...], list[str]] = {}
    if image:  # the image facts (base, shipped kernel) lead the first image group
        was = base_vllm(repo, previous_umbrella)
        items = []
        if was and was != vllm:
            items.append(f"- base bumped to vLLM {vllm} (from {was}); patch stack re-verified at build")
        else:
            items.append(f"- rebuilt on vLLM {vllm}")
        if k_new:
            items.append(f"- ships mjolnir-kernel {k_new}")
        elif k_prev and rp.changed_since(repo, kernel["previous"], kp["paths"]):
            items.append(
                f"- ships the kernel source of this commit, stamped {k_prev}: it has changes not released "
                "to PyPI (release the kernel too, or rebuild later)"
            )
        else:
            items.append(f"- ships mjolnir-kernel {k_prev or '—'} (unchanged)")
        target = next((g for g in sorted(groups, key=group_key) if "image" in g), ("image",))
        syn[target] = items
        groups.setdefault(target, [])

    lines += ["## What's changed", ""]
    if not groups and not catchall:
        lines += ["no changes since the last release.", ""]
    for g in sorted(groups, key=group_key):
        lines.append("### " + " + ".join(f"{labels[a]} {versions[a]}" for a in g))
        lines += syn.get(g, [])
        lines += [e.bullet() for e in sorted(groups[g], key=lambda e: -e.date)]
        lines.append("")
    if catchall:
        lines.append("### Repo, docs & tooling")
        lines += [pool[k].bullet() for k in catchall]
        lines.append("")

    lines += ["## Platform notes", "", "| platform | what's new |", "|---|---|"]
    visible = {*catchall, *(e.key for entries in groups.values() for e in entries)}
    for plat in PLATFORM_ROWS:
        refs = [pool[k].ref for k in order if plat in pool[k].platforms and k in visible]
        lines.append(f"| **{plat}** | {' '.join(refs) if refs else '—'} |")
    lines += [
        "",
        "_Platform attribution comes from the changed paths (the kernel ⇒ Thor's sm_110; the image is shared by "
        "all platforms and maps to none)._",
    ]
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
