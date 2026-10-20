"""Cycle 3d RED — site v2 (task/plan.md §3): the interactive GitHub-Pages
site becomes a static shell + small manifest + one lazy payload per MEASURED
(platform, model) pair, emitted by the new ``mjolnir_cli.site`` package with
real ``app.js``/``app.css`` files (no inline mega-string, no node build).

The data contracts are pinned as JSON here: measured-pairs-only payloads, slim
stage docs (no ``kernels``/``details``/``pipes``), compare-ready e2e cell keys
across platforms, a <256KB manifest. The JS behavior is pinned by string
contracts only (no node in CI): the hash-state keys, the filter element ids
and the no-hash hero default (thor, Qwen3.8-27B).

Fixture: :mod:`render_targets_fixtures` — two platforms (thor + spark) with
data for Qwen3.8-27B and an unmeasured Qwen3.6-35B-A3B (its config tree
exists, no (platform, model) data anywhere)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from mjolnir_cli.config import load_layout
from mjolnir_cli.config import RepoLayout

import render_targets_fixtures as fx
from render_targets_fixtures import M36, M38, build_repo

# the hash-state keys (plan §1: #platform&model&config&stage&compare) — the
# exact line app.js must carry so the URL state contract survives edits
HASH_KEYS = 'const HASH_KEYS = ["platform", "model", "config", "stage", "compare"];'

# the filter controls of spec §9.2 (platform tabs, searchable model dropdown,
# config chips, backend-family toggle, stage nav, metric, host/run-age,
# compare switch, global search)
FILTER_IDS = (
    "platform",
    "model",
    "config",
    "family",
    "stage",
    "metric",
    "host",
    "age",
    "compare",
    "search",
)


def _emit(tmp_path: Path, monkeypatch: pytest.FixtureRequest) -> tuple[Path, RepoLayout]:
    """The fixture repo with the site emitted into ``<repo>/_site``."""
    root = build_repo(tmp_path, monkeypatch)
    from mjolnir_cli import interactive

    site = interactive.build(load_layout(), root / "_site")
    assert site == root / "_site" / "index.html"
    return root / "_site", load_layout()


def _manifest(out: Path) -> dict:
    return cast(dict, json.loads((out / "manifest.json").read_text()))


def _payload(out: Path, platform: str, short: str) -> dict:
    return cast(dict, json.loads((out / "data" / platform / f"{short}.json").read_text()))


def _cell_keys(cells: list[dict]) -> set[tuple[int, int]]:
    return {(c["concurrency"], c["context"]) for c in cells}


# ── 1. the manifest ─────────────────────────────────────────────────────────


def test_manifest_lists_measured_pairs_with_stages_charts_and_sizes(
    tmp_path: Path, monkeypatch: pytest.FixtureRequest
) -> None:
    out, _ = _emit(tmp_path, monkeypatch)
    m = _manifest(out)
    # cycle 3f re-pin: the fixture's solo NVFP4_FA2 history record measures
    # (thor, M36) — an e2e-only pair, exactly like the real tree's
    pairs = {(p["platform"], p["model"]) for p in m["pairs"]}
    assert pairs == {("thor", M38), ("spark", M38), ("thor", M36)}
    m36 = next(p for p in m["pairs"] if (p["platform"], p["model"]) == ("thor", M36))
    assert m36["stages"] == [] and m36["charts"] == [], "history-only pair: e2e payload only"
    thor = next(p for p in m["pairs"] if p["platform"] == "thor" and p["model"] == M38)
    spark = next(p for p in m["pairs"] if p["platform"] == "spark")
    assert thor["short"] == "Qwen3.8-27B" and spark["short"] == "Qwen3.8-27B"
    assert "sweep" in thor["stages"] and "spec" in thor["stages"]
    assert spark["stages"] == [], "spark has no verify raws — e2e history only"
    assert "kernel-speedup" in thor["charts"] and spark["charts"] == []
    assert thor["size"] > 0 and spark["size"] > 0
    # the stages index: one entry per verify stage with its title/summary/kind
    assert m["stages"]["sweep"]["title"] == "Headline L-sweep"
    assert m["stages"]["spec"]["summary"] and m["stages"]["sweep"]["kind"] == "ncu"


def test_manifest_stays_small(tmp_path: Path, monkeypatch: pytest.FixtureRequest) -> None:
    out, _ = _emit(tmp_path, monkeypatch)
    assert (out / "manifest.json").stat().st_size < 256 * 1024


def test_manifest_carries_the_noise_band(tmp_path: Path, monkeypatch: pytest.FixtureRequest) -> None:
    from mjolnir_cli.verify.stages import NOISE_PCT

    out, _ = _emit(tmp_path, monkeypatch)
    assert _manifest(out)["noise_pct"] == NOISE_PCT


# ── 2. the payloads ─────────────────────────────────────────────────────────


def _assert_no_forbidden_keys(node: object, path: str = "payload") -> None:
    if isinstance(node, dict):
        bad = {"kernels", "details", "pipes"} & set(node)
        assert not bad, f"{path} carries slim-rule keys {sorted(bad)}"
        for k, v in node.items():
            _assert_no_forbidden_keys(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _assert_no_forbidden_keys(v, f"{path}[{i}]")


def test_payload_only_for_measured_pairs(tmp_path: Path, monkeypatch: pytest.FixtureRequest) -> None:
    out, _ = _emit(tmp_path, monkeypatch)
    assert (out / "data" / "thor" / "Qwen3.8-27B.json").is_file()
    assert (out / "data" / "spark" / "Qwen3.8-27B.json").is_file()
    # cycle 3f re-pin: the fixture's solo FA2 history record measures
    # (thor, M36) — it gets its (e2e-only) payload like the real tree's
    assert (out / "data" / "thor" / "Qwen3.6-35B-A3B.json").is_file()
    assert not (out / "data" / "orin").exists()  # an unmeasured pair gets no payload


def test_payload_is_slim_and_carries_stages_e2e_configs(tmp_path: Path, monkeypatch: pytest.FixtureRequest) -> None:
    out, _ = _emit(tmp_path, monkeypatch)
    p = _payload(out, "thor", "Qwen3.8-27B")
    _assert_no_forbidden_keys(p)
    assert p["platform"] == "thor" and p["model"] == M38 and p["short"] == "Qwen3.8-27B"
    # per-stage series: the pair's verify docs, slimmed
    assert "sweep" in p["stages"] and "fp6prof" in p["stages"] and "spec" in p["stages"]
    sweep = p["stages"]["sweep"]
    assert all("kernels" not in pt for s in sweep["series"] for pt in s["points"])
    # e2e cells: the pair's A/B grid — tg for decode, pp only where context exists
    assert set(p["e2e"]) == {"NVFP4", fx.FP6_QUANT}
    cells = p["e2e"]["NVFP4"]["cells"]
    assert all({"concurrency", "context", "tg", "pp"} <= set(c) for c in cells)
    assert next(c for c in cells if not c["context"])["pp"] is None
    assert next(c for c in cells if c["concurrency"] == 1 and c["context"] == 0)["tg"] == pytest.approx(40.3)
    # the pair's config list with the availability cell for THIS platform
    assert {c["quant"] for c in p["configs"]} == {fx.FP6_QUANT, "NVFP4"}
    fp6 = next(c for c in p["configs"] if c["quant"] == fx.FP6_QUANT)
    assert fp6["state"] == "native" and fp6["why"]
    spark = _payload(out, "spark", "Qwen3.8-27B")
    assert set(spark["e2e"]) == {"NVFP4", "NVFP4_MJOLNIR"}
    assert {c["quant"] for c in spark["configs"]} == {"NVFP4"}
    assert spark["stages"] == {}


# ── 3. emission shape ───────────────────────────────────────────────────────


def test_emission_shape_with_real_assets_and_no_v1_data_json(
    tmp_path: Path, monkeypatch: pytest.FixtureRequest
) -> None:
    out, _ = _emit(tmp_path, monkeypatch)
    idx = (out / "index.html").read_text()
    js = (out / "app.js").read_text()
    css = (out / "app.css").read_text()
    assert len(js) > 2000 and len(css) > 500, "real files, not stubs"
    assert not (out / "data.json").exists(), "the v1 single-page payload is gone"
    assert idx.count("cdn.jsdelivr.net/npm/plotly.js-dist-min@") == 1, "plotly is a CDN reference"
    for ref in ('src="app.js"', 'href="app.css"', "manifest.json"):
        assert ref in idx


def test_emission_is_idempotent(tmp_path: Path, monkeypatch: pytest.FixtureRequest) -> None:
    out, layout = _emit(tmp_path, monkeypatch)
    first = {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    from mjolnir_cli import interactive

    interactive.build(layout, out)
    second = {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    assert first == second


# ── 4. hash state + filter ids (string contracts) ───────────────────────────


def test_app_js_carries_the_hash_state_keys(tmp_path: Path, monkeypatch: pytest.FixtureRequest) -> None:
    out, _ = _emit(tmp_path, monkeypatch)
    assert HASH_KEYS in (out / "app.js").read_text()


def test_index_html_carries_every_filter_id_and_the_app_wires_them(
    tmp_path: Path, monkeypatch: pytest.FixtureRequest
) -> None:
    out, _ = _emit(tmp_path, monkeypatch)
    idx = (out / "index.html").read_text()
    js = (out / "app.js").read_text()
    for fid in FILTER_IDS:
        assert f'id="filter-{fid}"' in idx, f"index.html lacks the filter-{fid} control"
        assert f"filter-{fid}" in js, f"app.js never wires filter-{fid}"


# ── 5. the no-hash hero default ─────────────────────────────────────────────


def test_default_hero_is_the_pinned_thor_pair(tmp_path: Path, monkeypatch: pytest.FixtureRequest) -> None:
    out, _ = _emit(tmp_path, monkeypatch)
    assert _manifest(out)["default"] == {"platform": "thor", "model": M38}
    js = (out / "app.js").read_text()
    assert 'platform: "thor"' in js and "Qwen/Qwen3.8-27B" in js, "the no-hash fallback pair"


# ── 6. compare readiness (same (model, config) across platforms) ────────────


def test_compare_payloads_share_the_e2e_cell_grid(tmp_path: Path, monkeypatch: pytest.FixtureRequest) -> None:
    out, _ = _emit(tmp_path, monkeypatch)
    thor = _payload(out, "thor", "Qwen3.8-27B")["e2e"]["NVFP4"]["cells"]
    spark = _payload(out, "spark", "Qwen3.8-27B")["e2e"]["NVFP4"]["cells"]
    assert _cell_keys(thor) & _cell_keys(spark), "compare needs a shared (concurrency, context) grid"
    fields = {frozenset(c) for c in thor} | {frozenset(c) for c in spark}
    assert len(fields) == 1, "one cell schema across platforms"


# ── 7. hash-derived strings are escaped before any innerHTML sink ───────────


def test_app_js_escapes_hash_derived_state_before_inner_html(
    tmp_path: Path, monkeypatch: pytest.FixtureRequest
) -> None:
    """``state.stage``/``state.config`` come from ``location.hash``
    (``parseHash()``) — attacker-controlled — and reach the ``el()`` →
    ``innerHTML`` sink via the empty-state message and the compare section
    titles. The emitted app.js must run them through an entity-escaping
    helper, and the raw concatenations must be gone (review CRITICAL 1)."""
    out, _ = _emit(tmp_path, monkeypatch)
    js = (out / "app.js").read_text()
    assert "&lt;" in js and "&amp;" in js, "an entity-escaping esc() helper exists"
    assert "esc(state.stage)" in js, "the empty-state stage name is escaped"
    assert "esc(state.config)" in js, "the compare section titles escape the config"
    assert '"No " + state.stage' not in js, "raw hash stage reaches innerHTML"
    assert '" · " + state.config' not in js, "raw hash config reaches innerHTML"
