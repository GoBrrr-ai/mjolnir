"""The raw-prune audit: the verify raw JSONs the render never reads.

The render (``readme_targets.pair_docs`` — every chart, README table and
docs page) consumes, per ``(stage, platform, model)`` group, exactly ONE
document: the newest, preferring default parameters. The ``_UNCHARTED``
stages (timing/profile/vllm) are never rendered at all. Every other raw in
``benchmarks/raw/verify/`` is dead weight — it never reaches README.md,
``docs/platforms/**``, a chart PNG or the site.

``mjolnir plot --prune`` runs :func:`prune`: it git rms the tracked dead
raws (recoverable from history), reports the untracked ones (left on disk —
no git backup, not ours to delete) and warns about the reverse error — a
raw the render WOULD read that is not committed (the fresh clone's render
would drift from the committed docs, which fails CI).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from mjolnir_cli import readme_targets
from mjolnir_cli.verify import results
from mjolnir_cli.verify.stages import STAGES

from mjolnir_cli.config import DEFAULT_MODEL, RepoLayout, model_of
from mjolnir_cli.platforms import platform_of


@dataclass
class RawPrune:
    """The audit result: what the render reads vs what it never will."""

    used: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)  # the tracked unused — git rm'd
    untracked: list[str] = field(default_factory=list)  # the unused on disk, not in git
    missing: list[str] = field(default_factory=list)  # the used but untracked — would break CI

    @property
    def ok(self) -> bool:
        return not self.removed and not self.missing


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def _tracked(root: Path, verify_dir: Path) -> set[str]:
    """The verify raws git knows about, as paths relative to the repo root."""
    rel = verify_dir.relative_to(root)
    out = _git(root, "ls-files", "-z", "--", str(rel))
    return {p for p in out.split("\0") if p}


def _rel(root: Path, p: str) -> str | None:
    try:
        return str(Path(p).relative_to(root))
    except ValueError:  # the raw lives outside the checkout — nothing to git rm
        return None


def audit(layout: RepoLayout) -> RawPrune:
    """Classify every ``benchmarks/raw/verify/*.json`` as render-read or dead."""
    verify_dir = layout.verify_dir
    by_group: dict[tuple[str, str, str], list[dict]] = {}
    loaded: set[str] = set()
    for stage in STAGES:
        for doc in results.runs(verify_dir, stage):
            loaded.add(doc["_path"])
            key = (stage, platform_of(doc), model_of(doc) or DEFAULT_MODEL)
            by_group.setdefault(key, []).append(doc)

    report = RawPrune()
    for (stage, _platform, _model), group in sorted(by_group.items()):
        if stage in readme_targets._UNCHARTED:
            continue  # never rendered — every raw of the stage is dead weight
        newest = readme_targets._newest(group)
        if newest is not None:
            report.used.append(newest["_path"])

    # the files the render's own loader never sees: the stage dirs outside
    # STAGES, and jsons that fail the schema check inside them
    on_disk = (
        {str(p) for d in verify_dir.iterdir() if d.is_dir() for p in d.glob("*.json")} if verify_dir.is_dir() else set()
    )

    root = layout.repo_root
    tracked = _tracked(root, verify_dir) if root else set()
    dead = sorted((on_disk | loaded) - set(report.used))
    for p in dead:
        rel = _rel(root, p) if root else None
        if rel and rel in tracked:
            report.removed.append(p)
        else:
            report.untracked.append(p)
    if root:
        report.missing = sorted(p for p in report.used if (rel := _rel(root, p)) and rel not in tracked)
    return report


def prune(layout: RepoLayout) -> RawPrune:
    """Audit, then ``git rm`` the tracked dead raws (recoverable from history)."""
    report = audit(layout)
    if report.removed and layout.repo_root:
        root = layout.repo_root
        rels = [r for p in report.removed if (r := _rel(root, p))]
        # -f: a dead raw staged but never committed has no HEAD copy to
        # protect — the audit only removes what the render never reads
        _git(root, "rm", "-f", "--quiet", "--", *rels)
    return report
