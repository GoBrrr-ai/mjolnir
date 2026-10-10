"""Charts + README + docs render from raw verify data — and degrade to
"pending" notes when a stage has no run (the state of a fresh checkout)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mjolnir_cli import charts, interactive, readme
from mjolnir_cli.config import load_layout

from verify_fixtures import write_all

REPO = Path(__file__).resolve().parents[2]

pytest.importorskip("matplotlib")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A minimal checkout: the templates + Dockerfile + the real history."""
    root = tmp_path / "repo"
    (root / "docker" / "vllm-thor").mkdir(parents=True)
    shutil.copy(REPO / "docker" / "vllm-thor" / "Dockerfile", root / "docker" / "vllm-thor" / "Dockerfile")
    shutil.copy(REPO / "README.md.j2", root / "README.md.j2")
    shutil.copytree(REPO / "docs" / "kernel", root / "docs" / "kernel", ignore=shutil.ignore_patterns("*.md"))
    for pkg, init in (("mjolnir/mjolnir", "__init__.py"), ("mjolnir-cli/mjolnir_cli", "__init__.py")):
        (root / "src" / pkg).mkdir(parents=True)
        shutil.copy(REPO / "src" / pkg / init, root / "src" / pkg / init)
    (root / "benchmarks").mkdir()
    shutil.copy(REPO / "benchmarks" / "history.jsonl", root / "benchmarks" / "history.jsonl")
    monkeypatch.setenv("MJOLNIR_REPO", str(root))
    for v in ("MJOLNIR_BENCHMARKS", "MJOLNIR_HISTORY"):
        monkeypatch.delenv(v, raising=False)
    return root


def test_full_render_from_fixture_data(repo):
    layout = load_layout()
    write_all(layout.verify_dir)
    manifest = charts.render_all(layout)
    for name in (
        "kernel-speedup",
        "kernel-latency",
        "hnd-layout",
        "kvfmt-heatmap",
        "ring-heatmap",
        "prefetch-heatmap",
        "ab-heatmap",
        "pdl-delta",
        "quant-accuracy",
        "fp6prof-bytes-vs-time",
    ):
        assert name in manifest, name
        assert (repo / manifest[name]["light"]).exists() and (repo / manifest[name]["dark"]).exists()
    readme.render_readme(layout)
    text = (repo / "README.md").read_text()
    assert "{{" not in text and "{%" not in text
    assert "1.26× faster" in text  # geomean vs FlashInfer from the fixture sweep
    assert 'srcset="assets/benchmarks/kernel-speedup-dark.png"' in text
    page = (repo / "docs" / "kernel" / "ring.md").read_text()
    assert "../../assets/benchmarks/ring-heatmap.png" in page  # paths are relative to the page
    assert "Fastest variant at the most context lengths" in page
    site = interactive.build(layout, repo / "site")
    data = json.loads((repo / "site" / "data.json").read_text())
    assert "sweep" in data["stages"] and site.read_text().count("cdn.jsdelivr.net/npm/plotly.js-dist-min@") == 1


def test_render_without_verify_data_is_pending(repo):
    layout = load_layout()
    charts.render_all(layout)  # e2e charts only (history), no kernel charts
    readme.render_readme(layout)
    text = (repo / "README.md").read_text()
    assert "has not been recorded in this checkout yet" in text
    assert "Pending: `mjolnir verify sweep`" in text
    assert "Pending — run `mjolnir verify ring`" in (repo / "docs" / "kernel" / "ring.md").read_text()


def test_render_is_idempotent(repo):
    layout = load_layout()
    write_all(layout.verify_dir)
    charts.render_all(layout)
    readme.render_readme(layout)
    first = (repo / "README.md").read_text()
    readme.render_readme(layout)
    assert (repo / "README.md").read_text() == first
