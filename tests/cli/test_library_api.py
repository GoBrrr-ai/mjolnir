"""The declared public library surface (issue #41): the ``__all__`` pins, the
``schema_version`` field on history rows, and the import-light guarantee
(stdlib-only import chain — no CLI deps at import time)."""

from __future__ import annotations

import json
import subprocess
import sys

from mjolnir_cli import gate, history


def test_gate_all_is_the_declared_surface():
    assert set(gate.__all__) == {
        "DEFAULT_CONFIRM",
        "DEFAULT_POLL_S",
        "DEFAULT_TIMEOUT_S",
        "CleanWindow",
        "PreflightError",
        "server_counters",
        "server_load",
        "spec_delta",
        "wait_for_idle",
    }
    for name in gate.__all__:
        assert hasattr(gate, name)


def test_history_all_is_the_declared_surface():
    assert set(history.__all__) == {"SCHEMA_VERSION", "append_record", "load_records"}
    for name in history.__all__:
        assert hasattr(history, name)


def test_make_record_carries_schema_version():
    # a resolved Settings and an explicit gpu: the record is written without
    # probing the host (hermetic — CI has no GPU), as in test_history.py
    from mjolnir_cli.config import Settings
    from mjolnir_cli.history import make_record

    assert history.SCHEMA_VERSION == 1  # bumping the schema must be deliberate
    s = Settings(image="img:v1", model="m", quant="c", platform="thor")
    rec = make_record(
        s, gate_summary=None, runs=1, warmup_runs=0, exact_tg=True, cells=[], raw_relpath="raw/x/", gpu=""
    )
    assert rec["schema_version"] == history.SCHEMA_VERSION
    assert next(iter(rec)) == "schema_version"  # first key of the written JSON line


def test_load_records_returns_rows_raw(tmp_path):
    # legacy rows (written before schema_version existed) pass through
    # untouched — absence is the legacy format, no normalization
    f = tmp_path / "history.jsonl"
    f.write_text(json.dumps({"ts": "2026-01-01T00:00:00+00:00"}) + "\n")
    (rec,) = history.load_records(f)
    assert "schema_version" not in rec


def test_library_import_pulls_no_cli_dependencies():
    code = (
        "import sys; import mjolnir_cli.gate, mjolnir_cli.history; "
        "heavy = sorted(m.split('.')[0] for m in sys.modules if m.split('.')[0] in "
        "{'typer', 'rich', 'yaml', 'jinja2', 'click', 'pygments', 'markdown_it'}); "
        "print(heavy)"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert out.stdout.strip() == "[]"
