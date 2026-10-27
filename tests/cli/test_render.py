"""Charts + README + docs render from raw verify data — and degrade to
"pending" notes when a stage has no run (the state of a fresh checkout).

Cycle 3b: the render loop is ``readme.render_all`` (plan §1) over the spec's
target set; the kernel chapters render under ``docs/platforms/thor/kernel/``
from the moved ``docs/templates/kernel/`` templates and chart paths gain the
``assets/benchmarks/<platform>/<model-short>/`` identity. The rewritten tests
are listed in task/tests/red_state.md under "Spec changes" — the generated
outputs legitimately move/change (spec §Directory Layout, §Naming).

Spec changes: static charts render the dark palette (the Pages site is
interactive Plotly): ``chart()`` emits a plain ``<img src="<id>.png">`` (no
``<picture>``/srcset), ``render_all`` writes ``<id>.png`` files and manifest
v2 entries carry a single ``dark`` path."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mjolnir_cli import charts, interactive, readme
from mjolnir_cli.config import DEFAULT_MODEL, load_layout

from verify_fixtures import write_all

REPO = Path(__file__).resolve().parents[2]

pytest.importorskip("matplotlib")


def _stamp(verify_dir: Path) -> None:
    """verify_fixtures' docs predate the platform/model fields: stamp the
    (thor, Qwen3.8-27B) identity the v2 chart identity is derived from."""
    for p in verify_dir.glob("*/*.json"):
        doc = json.loads(p.read_text())
        doc["platform"] = "thor"
        doc["model"] = DEFAULT_MODEL
        p.write_text(json.dumps(doc, indent=1) + "\n")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A minimal checkout: the templates (their cycle-3 home, guarded so the
    RED state builds a template-less repo) + Dockerfile + a sandboxed configs
    dir + the real history."""
    root = tmp_path / "repo"
    (root / "docker" / "vllm-thor").mkdir(parents=True)
    shutil.copy(REPO / "docker" / "vllm-thor" / "Dockerfile", root / "docker" / "vllm-thor" / "Dockerfile")
    shutil.copy(REPO / "docker" / "vllm-thor" / "PATCHES.md", root / "docker" / "vllm-thor" / "PATCHES.md")
    shutil.copy(REPO / "docker" / "vllm-thor" / "README.md.j2", root / "docker" / "vllm-thor" / "README.md.j2")
    if (REPO / "docs" / "templates").is_dir():  # the cycle-3 template home
        shutil.copytree(REPO / "docs" / "templates", root / "docs" / "templates", ignore=shutil.ignore_patterns("*.md"))
    if (REPO / "README.md.j2").is_file():  # the root README template (repo root)
        shutil.copy(REPO / "README.md.j2", root / "README.md.j2")
    (root / "configs").mkdir()  # sandboxed: render_all must never write the real tree
    for pkg, init in (("mjolnir/mjolnir", "__init__.py"), ("mjolnir-cli/mjolnir_cli", "__init__.py")):
        (root / "src" / pkg).mkdir(parents=True)
        shutil.copy(REPO / "src" / pkg / init, root / "src" / pkg / init)
    (root / "benchmarks").mkdir()
    shutil.copy(REPO / "benchmarks" / "history.jsonl", root / "benchmarks" / "history.jsonl")
    monkeypatch.setenv("MJOLNIR_REPO", str(root))
    monkeypatch.setenv("MJOLNIR_CONFIGS_DIR", str(tmp_path / "local-configs"))
    (tmp_path / "local-configs").mkdir(exist_ok=True)
    for v in ("MJOLNIR_BENCHMARKS", "MJOLNIR_HISTORY"):
        monkeypatch.delenv(v, raising=False)
    return root


def test_full_render_from_fixture_data(repo):
    layout = load_layout()
    write_all(layout.verify_dir)
    _stamp(layout.verify_dir)
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
        entry = manifest[name][0] if isinstance(manifest[name], list) else manifest[name]
        assert (repo / entry["dark"]).is_file()
    # no legacy ``-dark`` suffix anywhere under assets/benchmarks
    pngs = list((repo / "assets" / "benchmarks").rglob("*.png"))
    assert pngs and all("-dark" not in p.name for p in pngs)
    readme.render_all(layout)
    text = (repo / "README.md").read_text()
    assert "{{" not in text and "{%" not in text
    # the Docker Hub description: rendered too, absolute URLs only (Docker Hub
    # has no repo file hosting to resolve relatives against)
    dh = (repo / "docker" / "vllm-thor" / "README.md").read_text()
    assert "{{" not in dh and "<img" not in dh
    assert "https://raw.githubusercontent.com/thor-arena/mjolnir/main/assets/hero.png" in dh
    # the patch inventory re-emitted verbatim from docker/vllm-thor/PATCHES.md
    assert "| File | PR | What it fixes |" in dh and "thor-fp6-e2m3-kv-cache.patch" in dh
    # the kernel index lives on the platform page's `## Kernel` section (cycle
    # 3k): the headline finding is reused there, the kernel README is gone
    thor_page = (repo / "docs" / "platforms" / "thor" / "README.md").read_text()
    assert "**Where it stands:** 1.26× FlashInfer" in thor_page
    assert not (repo / "docs" / "platforms" / "thor" / "kernel" / "README.md").exists()
    # the kernel benchmark charts moved off the root README to the Thor page
    # (the root keeps only the hero + the static assets)
    assert '<img src="../../../assets/benchmarks/thor/Qwen3.8-27B/kernel-latency.png"' in thor_page
    assert "assets/benchmarks/thor" not in text
    assert "<picture>" not in thor_page and "srcset" not in thor_page and "prefers-color-scheme" not in thor_page
    page = (repo / "docs" / "platforms" / "thor" / "kernel" / "ring.md").read_text()
    assert "../../../assets/benchmarks/thor/Qwen3.8-27B/ring-heatmap.png" in page  # relative to the page
    assert "Fastest variant at the most context lengths" in page
    # site keeps building (v2: the lazy per-pair payload replaces the v1
    # single-page data.json); the CLI reference is the plot command's write
    # (test_plot_readme_writes_the_cli_reference), not the render loop's
    site = interactive.build(layout, repo / "site")
    payload = json.loads((repo / "site" / "data" / "thor" / "Qwen3.8-27B.json").read_text())
    assert "sweep" in payload["stages"] and site.read_text().count("cdn.jsdelivr.net/npm/plotly.js-dist-min@") == 1


def test_plot_readme_writes_the_cli_reference(repo):
    """The render loop takes no command tree — the ``plot`` command does the
    write, because it is the one that holds the root app (injected by the
    composition root: no command module imports ``cli``)."""
    pytest.importorskip("matplotlib")
    from typer.testing import CliRunner

    from mjolnir_cli import cli

    layout = load_layout()
    write_all(layout.verify_dir)
    _stamp(layout.verify_dir)
    r = CliRunner().invoke(cli.app, ["plot", "--readme"])
    assert r.exit_code == 0, r.output
    cli_md = repo / "docs" / "cli.md"
    assert cli_md.is_file()
    assert f"[mjolnir] wrote {cli_md}" in r.output  # among the render's wrote lines


def test_render_without_verify_data_is_pending(tmp_path, monkeypatch):
    """Sparseness (spec §Render Targets): with no data at all every PAGE
    still renders (placeholders) — while the raws-gated kernel tree does not
    exist and every config README's Benchmarks section shows its honest
    no-data state (cycle 3f: BENCHMARKS.md is folded into the README)."""
    from render_targets_fixtures import FP6_QUANT, M38, build_repo

    root = build_repo(tmp_path, monkeypatch, with_data=False)
    readme.render_all(load_layout())
    assert (root / "README.md").is_file()
    assert (root / "docs" / "platforms" / "thor" / "README.md").is_file()
    assert not (root / "docs" / "platforms" / "thor" / "kernel").exists()
    assert not list(root.glob("configs/**/BENCHMARKS.md"))
    page = (root / "configs" / M38 / FP6_QUANT / "README.md").read_text()
    assert "no data yet" in page.split("<details>")[0]


def test_render_is_idempotent(repo):
    layout = load_layout()
    write_all(layout.verify_dir)
    _stamp(layout.verify_dir)
    charts.render_all(layout)
    readme.render_all(layout)
    first = (repo / "README.md").read_bytes()
    ring = (repo / "docs" / "platforms" / "thor" / "kernel" / "ring.md").read_bytes()
    readme.render_all(layout)
    assert (repo / "README.md").read_bytes() == first
    assert (repo / "docs" / "platforms" / "thor" / "kernel" / "ring.md").read_bytes() == ring


def test_kernel_chapters_render_under_docs_platforms_thor_kernel(repo):
    """The moved templates render the thor case study to its new home —
    content parity (spec #4): the same fixture data produces the same
    findings, how panels, provenance and chart paths; the old home is gone."""
    layout = load_layout()
    write_all(layout.verify_dir)
    _stamp(layout.verify_dir)
    charts.render_all(layout)
    readme.render_all(layout)
    for chapter in (
        "sweep",
        "hndsweep",
        "kvfmt",
        "quant",
        "fp6prof",
        "ring",
        "prefetch",
        "pdl",
        "ab",
        "correctness",
        "spec",
        "kv-cache-size",
    ):
        assert (repo / "docs" / "platforms" / "thor" / "kernel" / f"{chapter}.md").is_file(), chapter
    page = (repo / "docs" / "platforms" / "thor" / "kernel" / "ring.md").read_text()
    assert "Fastest variant at the most context lengths" in page  # the fixture sweep's finding
    assert "How the % and × here are calculated" in page  # the how("variant") panel survives
    assert "benchmarks/raw/verify/ring/20261010-120000.json" in page  # raw provenance
    assert "srcset" not in page and "<picture>" not in page
    assert '<img src="../../../../assets/benchmarks/thor/Qwen3.8-27B/ring-heatmap.png"' in page
    assert not (repo / "docs" / "kernel" / "ring.md").exists()  # the old home is gone


def test_chart_manifest_v2_writes_the_pair_identity_and_nested_paths(repo):
    """Manifest v2 (spec #8): entries carry the (platform, model) identity and
    resolve to the asset assets/benchmarks/<platform>/<model-short>/<id>.png
    — a single ``dark`` path, no ``light`` key; pairs without data render no
    charts (no orin anything)."""
    layout = load_layout()
    write_all(layout.verify_dir)
    _stamp(layout.verify_dir)
    manifest = charts.render_all(layout)
    # v2 manifests are list-valued: one entry per (chart, measured pair)
    entry = next(e for e in manifest["kernel-speedup"] if e["platform"] == "thor")
    assert entry["model"] == DEFAULT_MODEL
    assert entry["dark"] == "assets/benchmarks/thor/Qwen3.8-27B/kernel-speedup.png"
    assert (repo / entry["dark"]).is_file()
    for name, es in manifest.items():
        for e in es:
            assert "light" not in e, name
            assert (repo / e["dark"]).is_file(), name
            if name in ("hero", "opening-infographic"):  # the root-page one-offs stay flat
                continue
            assert "platform" in e and "model" in e, name
    # pairs without data render no charts (the multi-pair manifest carries
    # one entry per measured pair -- exercised on the real orin data)
    assert not any(e.get("platform") == "spark" for es in manifest.values() for e in es)


def test_chart_manifest_v2_legacy_entries_still_resolve(repo):
    """The reader falls back: committed flat entries (no identity fields) keep
    resolving through chart() — old manifests never break a render. The
    fixture entry carries a flat ``dark`` path and the page quotes it as a
    plain ``<img>``."""
    layout = load_layout()
    charts_dir = layout.charts_dir
    charts_dir.mkdir(parents=True)
    (charts_dir / "ring-heatmap.png").write_bytes(b"\x89PNG legacy")
    (charts_dir / "manifest.json").write_text(
        json.dumps({"ring-heatmap": {"stage": "ring", "dark": "assets/benchmarks/ring-heatmap.png"}})
    )
    write_all(layout.verify_dir)
    _stamp(layout.verify_dir)
    readme.render_all(layout)
    page = (repo / "docs" / "platforms" / "thor" / "kernel" / "ring.md").read_text()
    assert '<img src="../../../../assets/benchmarks/ring-heatmap.png"' in page


def test_hero_overlays_the_background_under_the_logotype(tmp_path):
    from PIL import Image

    from mjolnir_cli.config import RepoLayout, find_repo_root

    root = find_repo_root()
    (tmp_path / "assets").mkdir()
    shutil.copy(root / "assets" / charts.HERO_BACKGROUND, tmp_path / "assets" / charts.HERO_BACKGROUND)
    layout = RepoLayout(
        repo_root=tmp_path,
        data_dir=tmp_path,
        state_file=tmp_path / "s.json",
        benchmarks_dir=tmp_path / "benchmarks",
        history_file=tmp_path / "h.jsonl",
    )
    out = charts.render_hero(layout)
    assert out is not None
    bg = Image.open(tmp_path / "assets" / charts.HERO_BACKGROUND)
    hero = Image.open(out)
    assert out.name == "hero.png" and hero.size == bg.size
    left, top, right, bottom = charts._logotype_box(bg)
    assert left > bg.size[0] * 0.6 and bottom < bg.size[1] * 0.8  # the MJOLNIR logotype, right side
    # the logotype is untouched; the area under it now carries the slogan and pills
    assert (
        hero.crop((left, top, right, bottom)).tobytes() == bg.convert("RGB").crop((left, top, right, bottom)).tobytes()
    )
    under = (left, bottom + 20, right, min(bg.size[1], bottom + 200))
    assert hero.crop(under).tobytes() != bg.convert("RGB").crop(under).tobytes()


def test_readme_uses_the_hero():
    from mjolnir_cli.config import find_repo_root

    tpl = (find_repo_root() / "README.md.j2").read_text()
    assert 'src="assets/hero.png"' in tpl and "header.png" not in tpl
