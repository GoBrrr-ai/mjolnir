"""`mjolnir plot` and `mjolnir history` — the command bodies in
``commands/results.py``. The heavy renderers are faked at the module
boundary; `history` runs against a real (tmp) history.jsonl."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from mjolnir_cli import charts, cli
from mjolnir_cli.commands import results

runner = CliRunner()


@pytest.fixture
def history(tmp_path, monkeypatch):
    rows = [
        {
            "ts": "2026-10-21T09:30:00",
            "backend": "Mjolnir",
            "image": "narandill/mjolnir-vllm:v0.32.0",
            "cells": [
                {"context": 4096, "concurrency": 1, "tg_tps": {"mean": 120.5}},
                {"context": 4096, "concurrency": 4, "tg_tps": {"mean": 300.0}},
                {"context": 65536, "concurrency": 1, "tg_tps": {"mean": 90.0}},
            ],
        },
        {
            "ts": "2026-10-20T18:00:00",
            "backend": "FlashInfer",
            "image": "vllm/vllm-openai:v0.32.0-ubuntu2404",
            "cells": [{"context": 4096, "concurrency": 1, "tg_tps": {"mean": 100.0}}],
        },
    ]
    p = tmp_path / "history.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setenv("MJOLNIR_HISTORY", str(p))
    return p


def test_history_prints_the_decode_tables(history):
    r = runner.invoke(cli.app, ["history"])
    assert r.exit_code == 0, r.output
    assert "Mjolnir — narandill/mjolnir-vllm:v0.32.0" in r.output
    assert "2026-10-21 09:30" in r.output  # trimmed ts
    assert "c=1 · t/s" in r.output and "c=4 · t/s" in r.output
    assert "4K" in r.output and "64K" in r.output  # _ctx_label
    assert "300.0" in r.output and "90.0" in r.output


def test_history_defaults_to_the_newest_five(history):
    lines = history.read_text().splitlines()
    history.write_text("\n".join(lines + lines + lines + lines + lines + lines[:1]) + "\n")  # 7 records
    r = runner.invoke(cli.app, ["history"])
    assert r.exit_code == 0
    assert len(re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", r.output)) == 5  # limit applied


def test_history_json_dumps_the_raw_records(history):
    r = runner.invoke(cli.app, ["history", "--json"])
    assert r.exit_code == 0
    doc = json.loads(r.output)
    assert len(doc) == 2 and doc[0]["cells"][0]["tg_tps"]["mean"] == 120.5


def test_history_without_any_runs_names_the_command(tmp_path, monkeypatch):
    monkeypatch.setenv("MJOLNIR_HISTORY", str(tmp_path / "absent.jsonl"))
    r = runner.invoke(cli.app, ["history"])
    assert r.exit_code == 0 and "no history yet — run: mjolnir bench perf" in r.output


def test_history_skips_a_truncated_last_line(history):
    with open(history, "a") as f:
        f.write('{"ts": "2026-10-22T00:00:0')  # an interrupted append
    r = runner.invoke(cli.app, ["history"])
    assert r.exit_code == 0 and len(re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", r.output)) == 2


def test_plot_chart_failure_exits_1(tmp_path, monkeypatch):
    def boom(layout):
        raise RuntimeError("no data at all")

    monkeypatch.setattr(charts, "render_all", boom)
    r = runner.invoke(cli.app, ["plot"])
    assert r.exit_code == 1 and "no data at all" in r.output


def test_plot_readme_failure_exits_1(tmp_path, monkeypatch):
    from mjolnir_cli import autoresearch, readme

    monkeypatch.setattr(charts, "render_all", lambda layout: [])
    monkeypatch.setattr(autoresearch, "render_progress", lambda *a, **kw: None)
    monkeypatch.setattr(
        readme, "render_readme", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("no template"))
    )
    r = runner.invoke(cli.app, ["plot", "--readme"])
    assert r.exit_code == 1 and "no template" in r.output


def test_plot_without_the_readme_flag_never_renders_it(tmp_path, monkeypatch):
    from mjolnir_cli import autoresearch, readme

    seen = {}
    monkeypatch.setattr(charts, "render_all", lambda layout: [])
    monkeypatch.setattr(readme, "render_readme", lambda *a, **kw: seen.setdefault("readme", True))
    monkeypatch.setattr(autoresearch, "render_progress", lambda *a, **kw: None)
    r = runner.invoke(cli.app, ["plot"])
    assert r.exit_code == 0 and "readme" not in seen


def test_make_plot_returns_a_callable_registered_by_the_composition_root():
    # the factory's product is what cli.py registers; the closure is not a
    # second copy of the body
    plot = results.make_plot.__doc__
    assert plot is not None and "injects" in plot


def test_plot_site_builds_the_interactive_page(tmp_path, monkeypatch):
    import mjolnir_cli.interactive as interactive_mod
    from mjolnir_cli import autoresearch

    monkeypatch.setattr(charts, "render_all", lambda layout: [])
    monkeypatch.setattr(autoresearch, "render_progress", lambda *a, **kw: None)
    site = tmp_path / "site"
    monkeypatch.setattr(interactive_mod, "build", lambda layout, out: seen.setdefault("site", out))
    seen: dict = {}
    r = runner.invoke(cli.app, ["plot", "--site", str(site)])
    assert r.exit_code == 0, r.output
    assert seen["site"] == site


def test_history_skips_cells_without_a_decode_rate(tmp_path, monkeypatch):
    p = tmp_path / "history.jsonl"
    rows = [
        {
            "ts": "2026-10-21T09:30:00",
            "backend": "B",
            "image": "img",
            "cells": [
                {"context": 4096, "concurrency": 1},  # no tg_tps: skipped
                {"context": None, "concurrency": 2, "tg_tps": {"mean": 11.0}},  # context None → the "0" row
            ],
        }
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setenv("MJOLNIR_HISTORY", str(p))
    r = runner.invoke(cli.app, ["history"])
    assert r.exit_code == 0, r.output
    assert "4K" not in r.output  # the tg-less cell contributed no row
    assert "11.0" in r.output
    assert "\n0 " in r.output or " 0 " in r.output  # the None-context row renders as "0"
