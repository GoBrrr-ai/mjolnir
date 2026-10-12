"""``mjolnir bench accuracy``: tool-eval-bench's SQLite -> accuracy.json ->
docs table + charts, and the container cleanup helpers it relies on."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from mjolnir_cli import accuracy, charts, dockerctl, readme
from mjolnir_cli.config import load_layout

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = "vllm/vllm-openai:v0.31.0-ubuntu2404"
MJ = "narandill/mjolnir-vllm:v0.31.0"


def _db(path: Path, runs: list[tuple[str, dict]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE scenario_runs (run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL, "
        "model TEXT NOT NULL, config_json TEXT NOT NULL, scores_json TEXT, metadata_json TEXT, "
        "run_type TEXT NOT NULL DEFAULT 'tool_eval', report_path TEXT)"
    )
    for i, (run_type, scores) in enumerate(runs):
        con.execute(
            "INSERT INTO scenario_runs VALUES (?, ?, 'completed', 'm', '{}', ?, '{}', ?, NULL)",
            (f"r{i}", f"2026-10-12T10:0{i}:00", json.dumps(scores), run_type),
        )
    con.commit()
    con.close()
    return path


def _tool_scores(a_earned: int, p_earned: int) -> dict:
    return {
        "final_score": 80,
        "category_scores": [
            {"category": "A", "label": "Tool selection", "earned": a_earned, "max": 10, "pass_count": 4},
            {"category": "B", "label": "Parameters", "earned": 8, "max": 10, "pass_count": 4},
            {"category": "P", "label": "Hard Mode", "earned": p_earned, "max": 8, "pass_count": 2},
        ],
        "scenario_results": [
            {"scenario_id": "TC-01", "points": 2},
            {"scenario_id": "TC-02", "points": a_earned // 5},
        ],
    }


def test_summarize_db_splits_standard_and_hard_mode(tmp_path):
    db = _db(
        tmp_path / "data" / "benchmarks.sqlite",
        [("tool_eval", _tool_scores(10, 4)), ("gsm8k", {"accuracy": 91.5, "correct": 183, "total": 200})],
    )
    s = accuracy.summarize_db(db)
    assert s["suites"]["tools"]["score"] == 90.0  # (10 + 8) / 20
    assert s["suites"]["hardmode"]["score"] == 50.0
    assert s["suites"]["gsm8k"]["score"] == 91.5 and s["suites"]["gsm8k"]["total"] == 200
    assert s["categories"]["A"]["percent"] == 100.0 and s["scenarios"]["TC-02"] == 2.0


def test_summarize_db_averages_trials(tmp_path):
    db = _db(tmp_path / "b.sqlite", [("tool_eval", _tool_scores(10, 4)), ("tool_eval", _tool_scores(6, 8))])
    t = accuracy.summarize_db(db)["suites"]
    assert t["tools"]["n"] == 2 and t["tools"]["scores"] == [90.0, 70.0] and t["tools"]["score"] == 80.0
    assert accuracy.summarize_db(tmp_path / "missing.sqlite")["suites"] == {}


def test_tool_args_default_and_passthrough():
    assert accuracy.tool_args([], "http://h:8000/v1") == [*accuracy.DEFAULT_ARGS, "--base-url", "http://h:8000/v1"]
    fwd = accuracy.tool_args(["--gsm8k-only", "--gsm8k-limit", "50"], "http://h/v1")
    assert fwd[:3] == ["--gsm8k-only", "--gsm8k-limit", "50"] and fwd[-2:] == ["--base-url", "http://h/v1"]
    assert accuracy.tool_args(["run", "--base-url=http://x"], "http://h/v1") == ["run", "--base-url=http://x"]


def _leg(image: str, config: str, tools: float, hard: float, cats: dict, extra: dict | None = None) -> dict:
    rec = {"id": config, "model": "Qwen/Qwen3.8-27B", "image": image, "config": config}
    rec["label"] = accuracy.leg_label(rec)
    rec["suites"] = {"tools": {"score": tools, "n": 1, "completion_rate": 100.0}, "hardmode": {"score": hard, "n": 1}}
    rec["suites"].update(extra or {})
    rec["categories"] = {c: {"label": f"cat {c}", "percent": v} for c, v in cats.items()}
    rec["scenarios"] = {"TC-01": 2.0, "TC-02": 1.0 if config != "NVFP4" else 2.0}
    return rec


def _write_runs(raw: Path) -> None:
    first = {
        "format": accuracy.FORMAT,
        "run_id": "20261012-100000",
        "args": list(accuracy.DEFAULT_ARGS),
        "legs": [
            _leg(OFFICIAL, "NVFP4", 88.0, 60.0, {"A": 90, "P": 60}),
            _leg(MJ, "NVFP4_MJOLNIR_E2M3", 86.5, 62.0, {"A": 85, "P": 62}),
        ],
    }
    later = {
        "format": accuracy.FORMAT,
        "run_id": "20261013-100000",
        "args": ["--gsm8k-only"],
        "legs": [
            {
                **_leg(MJ, "NVFP4_MJOLNIR_E2M3", 0, 0, {}),
                "suites": {"gsm8k": {"score": 90.0, "n": 1}},
                "categories": {},
            }
        ],
    }
    for d in (first, later):
        p = raw / "accuracy" / d["run_id"] / "accuracy.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(d))


def test_latest_legs_merge_suites_across_runs(tmp_path):
    _write_runs(tmp_path)
    rows = accuracy.latest_legs(accuracy.load_runs(SimpleNamespace(raw_dir=tmp_path)))
    assert [r["config"] for r in rows] == ["NVFP4", "NVFP4_MJOLNIR_E2M3"]
    mj = rows[1]
    assert set(mj["suites"]) == {"tools", "hardmode", "gsm8k"}  # the GSM8K run joins the A/B row
    assert mj["categories"]["A"]["percent"] == 85  # categories stay from the tool-call run
    assert accuracy.baseline(rows)["config"] == "NVFP4"
    assert accuracy.suites_in(rows) == ["tools", "hardmode", "gsm8k"]
    assert rows[0]["label"].startswith("official vLLM")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "docker" / "vllm-thor").mkdir(parents=True)
    shutil.copy(REPO / "docker" / "vllm-thor" / "Dockerfile", root / "docker" / "vllm-thor" / "Dockerfile")
    shutil.copy(REPO / "README.md.j2", root / "README.md.j2")
    shutil.copytree(REPO / "docs" / "kernel", root / "docs" / "kernel", ignore=shutil.ignore_patterns("*.md"))
    for pkg in ("mjolnir/mjolnir", "mjolnir-cli/mjolnir_cli"):
        (root / "src" / pkg).mkdir(parents=True)
        shutil.copy(REPO / "src" / pkg / "__init__.py", root / "src" / pkg / "__init__.py")
    (root / "benchmarks").mkdir()
    (root / "benchmarks" / "history.jsonl").write_text("")
    monkeypatch.setenv("MJOLNIR_REPO", str(root))
    for v in ("MJOLNIR_BENCHMARKS", "MJOLNIR_HISTORY"):
        monkeypatch.delenv(v, raising=False)
    return root


def test_accuracy_page_and_charts_render(repo):
    pytest.importorskip("matplotlib")
    layout = load_layout()
    _write_runs(layout.raw_dir)
    manifest = charts.render_all(layout)
    for name in ("accuracy-suites", "accuracy-categories"):
        assert name in manifest and (repo / manifest[name]["dark"]).exists()
    readme.render_readme(layout)
    page = (repo / "docs" / "kernel" / "accuracy.md").read_text()
    assert "{{" not in page and "{%" not in page
    assert "| **official vLLM · FlashInfer** |" in page
    assert "86.5 (-1.5)" in page and "62.0 (+2.0)" in page and "90.0" in page
    assert "1 scenarios scored lower, 0 higher of 2" in page
    assert "accuracy-suites-dark.png" in page and "20261013-100000/accuracy.json" in page
    assert "[Accuracy](docs/kernel/accuracy.md)" in (repo / "README.md").read_text()


def test_accuracy_page_pending_without_runs(repo):
    layout = load_layout()
    readme.render_readme(layout)
    assert "Pending — run `mjolnir bench accuracy`" in (repo / "docs" / "kernel" / "accuracy.md").read_text()


def test_infographic_multiple():
    assert charts.multiple(200) == "3×" and charts.multiple(150) == "2.5×" and charts.multiple(100.4) == "2×"


def test_container_output_is_handed_to_the_host_user(monkeypatch):
    monkeypatch.setattr(dockerctl.os, "getuid", lambda: 1000)
    monkeypatch.setattr(dockerctl.os, "getgid", lambda: 1001)
    tail = dockerctl.owner_suffix(["/out/draft_vocab.json"])
    assert tail == "; rc=$?; chown -R 1000:1001 /out/draft_vocab.json 2>/dev/null; exit $rc"
    assert dockerctl.owner_suffix([]) == ""
