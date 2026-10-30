"""Retention: the staging/logs GC — what the writers prune automatically at
the end of every run and ``mjolnir gc`` prunes on demand. Old exhaust is
freed unless something still cites it."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from typer.testing import CliRunner

from mjolnir_cli import cli, retention
from mjolnir_cli.config import RepoLayout
from mjolnir_cli.verify import results

runner = CliRunner()


def _layout(tmp_path: Path) -> RepoLayout:
    return RepoLayout(
        repo_root=tmp_path,
        data_dir=tmp_path,
        state_file=tmp_path / "s.json",
        benchmarks_dir=tmp_path / "benchmarks",
        history_file=tmp_path / "benchmarks" / "history.jsonl",
    )


def _age(p: Path, days: float) -> None:
    old = time.time() - days * 86400
    os.utime(p, (old, old))


def _evidence_doc(verify_dir: Path) -> None:
    """One tracked evidence run, written through the writer (placement rule)."""
    results.save(
        verify_dir,
        results.header("sweep", "20261001-000000", image="img", default=True, params={}, kernel={}, gate={}),
    )


def test_old_uncited_exhaust_is_freed_fresh_survives(tmp_path):
    layout = _layout(tmp_path)
    old_json = layout.staging_dir / "verify" / "sweep" / "20261001-000001.json"
    old_json.parent.mkdir(parents=True)
    old_json.write_text("{}")
    old_perf = layout.staging_dir / "perf-A"
    old_perf.mkdir()
    (old_perf / "raw.json").write_text("{}")
    old_log = layout.benchmarks_dir / "logs" / "verify-20261001-000002"
    old_log.mkdir(parents=True)
    for p in (old_json, old_perf, old_log):
        _age(p, 30)
    fresh = layout.staging_dir / "verify" / "sweep" / "20261030-000000.json"
    fresh.write_text("{}")  # now — a live experiment
    fresh_perf = layout.staging_dir / "perf-B"
    fresh_perf.mkdir()

    rep = retention.sweep(layout)

    assert not old_json.exists() and not old_perf.exists() and not old_log.exists()
    assert fresh.exists() and fresh_perf.exists()
    assert (
        rep.removed
        == [
            "staging/verify/sweep/20261001-000001.json",
            "staging/perf-A",
            "logs/verify-20261001-000002",
        ]
        and rep.freed_bytes > 0
    )


def test_cited_perf_dir_survives(tmp_path):
    layout = _layout(tmp_path)
    cited = layout.staging_dir / "perf-CITED"
    cited.mkdir(parents=True)
    _age(cited, 30)
    uncited = layout.staging_dir / "perf-OLD"
    uncited.mkdir()
    _age(uncited, 30)
    (layout.benchmarks_dir / "history.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (layout.benchmarks_dir / "history.jsonl").write_text(
        json.dumps({"raw": "staging/perf-CITED", "merged_runs": ["staging/perf-CITED"], "cells": []}) + "\n"
    )

    rep = retention.sweep(layout)

    assert cited.exists() and not uncited.exists()
    assert rep.kept_cited == 1


def test_evidence_log_survives_other_logs_expire(tmp_path):
    layout = _layout(tmp_path)
    _evidence_doc(layout.verify_dir)
    evidence_log = layout.benchmarks_dir / "logs" / "verify-20261001-000000"
    evidence_log.mkdir(parents=True)
    _age(evidence_log, 30)
    dead_log = layout.benchmarks_dir / "logs" / "verify-20260901-000000"
    dead_log.mkdir(parents=True)
    _age(dead_log, 30)

    rep = retention.sweep(layout)

    assert evidence_log.exists() and not dead_log.exists()
    assert rep.kept_cited == 1


def test_zero_days_and_bad_env_disable_nothing_gets_deleted(tmp_path, monkeypatch):
    layout = _layout(tmp_path)
    old = layout.staging_dir / "perf-OLD"
    old.mkdir(parents=True)
    _age(old, 30)

    assert retention.sweep(layout, days=0).removed == []
    assert old.exists()

    monkeypatch.setenv("MJOLNIR_RETENTION_DAYS", "0")
    assert retention.sweep(layout).removed == []
    assert old.exists()

    monkeypatch.setenv("MJOLNIR_RETENTION_DAYS", "banana")
    assert retention.grace_days() == retention.DEFAULT_DAYS  # a typo never wipes fresh work


def test_dry_run_reports_without_deleting(tmp_path):
    layout = _layout(tmp_path)
    old = layout.staging_dir / "verify" / "sweep" / "old.json"
    old.parent.mkdir(parents=True)
    old.write_text("{}")
    _age(old, 30)

    rep = retention.sweep(layout, dry_run=True)

    assert rep.removed == [str(old.relative_to(layout.benchmarks_dir))]
    assert old.exists()


def test_gc_command_dry_run(tmp_path, monkeypatch):
    _layout(tmp_path)
    monkeypatch.setenv("MJOLNIR_REPO", str(tmp_path))
    r = runner.invoke(cli.app, ["gc", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "(dry run)" in r.output
    assert "nothing to free" in r.output
