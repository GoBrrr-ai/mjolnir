"""Automatic retention for the untracked exhaust — ``benchmarks/staging/``
and ``benchmarks/logs/``.

The tracked evidence never needs a GC: the writers replace it in place (one
document per verify stage + platform + model, one history record per serving
key). The untracked side does grow — every experiment run, perf raw dir and
verify log lands there and nothing removed it. :func:`sweep` deletes what is
both (a) older than the grace period and (b) not cited by anything:

- ``staging/verify/**``: experiment run documents. Nothing tracked cites them
  (they are untracked by definition), so age is the only rule; the grace
  period is what keeps live work — an autoresearch loop, a mid-flight A/B —
  safe. The autoresearch ledger rows carry their own numbers, so expiring a
  staged document never breaks a row.
- ``staging/perf-<stamp>/``: kept while a history record names it (the
  record's ``raw`` / ``merged_runs`` list the dirs whose cells it carries).
- ``staging/history-extras.jsonl``: superseded-record log — append-only and
  tiny; never pruned. Anything else in ``staging/`` is left alone (only
  recognized exhaust is ever deleted).
- ``logs/verify-<run_id>/``: kept while that run is the tracked evidence of
  its (stage, platform, model) — the log behind a committed claim; every
  other run's logs expire.

:func:`sweep` runs automatically at the end of every ``mjolnir verify`` stage
and ``mjolnir bench`` sweep (silent unless something was freed) and on demand
via ``mjolnir gc``. ``$MJOLNIR_RETENTION_DAYS`` tunes the grace period
(default 14; ``0`` disables retention entirely).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from mjolnir_cli.config import RepoLayout
from mjolnir_cli.history import load_records
from mjolnir_cli.verify import results

DEFAULT_DAYS = 14


def grace_days() -> int:
    """The grace period in days: ``$MJOLNIR_RETENTION_DAYS``, else 14. An
    unset or invalid value keeps the default — a typo must never silently
    disable retention (the disk would fill) nor wipe fresh work."""
    raw = (os.environ.get("MJOLNIR_RETENTION_DAYS") or "").strip()
    try:
        return int(raw) if raw else DEFAULT_DAYS
    except ValueError:
        return DEFAULT_DAYS


@dataclass
class Report:
    """What a sweep removed and why it kept the rest."""

    days: int
    removed: list[str] = field(default_factory=list)  # paths relative to benchmarks_dir
    freed_bytes: int = 0
    kept_cited: int = 0  # past the grace period but still cited — kept

    def summary(self) -> str:
        if not self.removed:
            return "nothing to free"
        return f"freed {len(self.removed)} item(s), {self.freed_bytes / 1e6:.1f} MB (retention {self.days}d)"


def _cited(layout: RepoLayout) -> set[str]:
    """Staging dirs the history records still name (benchy writes the perf
    dir's benchmarks-relative path into the record's ``raw``; merges
    accumulate them in ``merged_runs``)."""
    out: set[str] = set()
    if layout.history_file.exists():
        for rec in load_records(layout.history_file):
            if rec.get("raw"):
                out.add(str(rec["raw"]).rstrip("/"))
            for r in rec.get("merged_runs") or []:
                if r:
                    out.add(str(r).rstrip("/"))
    return out


def _evidence_run_ids(layout: RepoLayout) -> set[str]:
    """Run ids of the tracked verify evidence — every raw the render reads."""
    out: set[str] = set()
    vd = layout.verify_dir
    if vd.is_dir():
        for stage_dir in vd.iterdir():
            if stage_dir.is_dir():
                out.update(doc["run_id"] for doc in results.runs(vd, stage_dir.name))
    return out


def sweep(layout: RepoLayout, *, days: int | None = None, dry_run: bool = False) -> Report:
    """Free staging/log entries that are past the grace period and cited by
    nothing. Never touches ``raw/`` or ``history.jsonl`` — and within
    staging, only the recognized exhaust (verify run docs, ``perf-*`` dirs):
    unknown content is left for its owner."""
    rep = Report(days=grace_days() if days is None else days)
    if rep.days <= 0 or not layout.benchmarks_dir.is_dir():
        return rep
    cutoff = time.time() - rep.days * 86400
    cited = _cited(layout)
    evidence = _evidence_run_ids(layout)

    def old(p: Path) -> bool:
        try:
            return p.stat().st_mtime < cutoff
        except OSError:  # vanished under us — nothing to retain
            return False

    def remove(p: Path) -> None:
        rel = str(p.relative_to(layout.benchmarks_dir))
        with contextlib.suppress(OSError):
            rep.freed_bytes += (
                sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
            )
        rep.removed.append(rel)
        if dry_run:
            return
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                p.unlink()

    staging = layout.staging_dir

    # 1. experiment run documents — nothing tracked cites them: age only
    sverify = staging / "verify"
    if sverify.is_dir():
        for stage_dir in sorted(p for p in sverify.iterdir() if p.is_dir()):
            for p in sorted(stage_dir.glob("*.json")):
                if old(p):
                    remove(p)
            if not dry_run:
                try:
                    if not any(stage_dir.iterdir()):
                        stage_dir.rmdir()
                except OSError:
                    pass

    # 2. perf raw dirs — exhaust, but a history record may still name one
    if staging.is_dir():
        for d in sorted(p for p in staging.glob("perf-*") if p.is_dir()):
            if not old(d):
                continue
            if str(d.relative_to(layout.benchmarks_dir)) in cited:
                rep.kept_cited += 1
                continue
            remove(d)

    # 3. verify logs — kept while the run is the tracked evidence of its group
    logs = layout.benchmarks_dir / "logs"
    if logs.is_dir():
        for d in sorted(p for p in logs.glob("verify-*") if p.is_dir()):
            if not old(d):
                continue
            if d.name[len("verify-") :] in evidence:
                rep.kept_cited += 1
                continue
            remove(d)
    return rep
