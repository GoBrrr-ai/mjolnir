"""Cycle-3b RED: render targets — discovery, sparseness, template contracts.

Plan §2.1 scenarios; spec of record:
task/interviewer/spec-docs-benchmarks.md (§Render Targets, §Data→Page
Mapping, §Directory Layout, §Cross-Linking, §Naming). TDD contract: the
planned interfaces do not exist yet, and each failure below is the required
RED state:

* ``mjolnir_cli.readme_targets`` (the Target hierarchy, ``discover_targets``,
  per-target ``build_context``) — imported *inside* the discovery tests so the
  rest of the module reports its own RED signal instead of hiding behind one
  collection error; there the ImportError IS the RED state.
* ``readme.render_all(layout)`` — the template loop (plan §1) — every render
  contract fails on AttributeError until it exists.
* ``docs/templates/pages/*.j2`` + ``docs/templates/kernel/*.md.j2`` — pinned
  directly against the real tree (one existence test); fixture repos copy the
  templates guarded, so RED builds template-less repos.

Fixture repos: :mod:`render_targets_fixtures` — two models × five configs with
platform overlays, verify_fixtures-style raws stamped (thor, Qwen3.8-27B), a
spark e2e pair the root headline must ignore, and a seeded v2 chart manifest.
The real ``docs/`` tree is never written to; matplotlib is never needed here
(chart references resolve from the seeded manifest).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mjolnir_cli import availability
from mjolnir_cli.config import REGISTRY, load_layout

import render_targets_fixtures as fx
from render_targets_fixtures import FP6_QUANT, M36, M38, build_repo

REPO = Path(__file__).resolve().parents[2]
TEMPLATE_HOME = REPO / "docs" / "templates"

# the spec §Directory Layout chapter set (docs/templates/kernel/)
CHAPTERS = (
    "README",
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
)
PAGE_TEMPLATES = (
    "PLATFORMS_INDEX.md.j2",
    "PLATFORM_README.md.j2",
    "MODEL_README.md.j2",
    "CONFIG_README.md.j2",
    "CONFIG_BENCHMARKS.md.j2",
)

SYMBOL = {
    availability.NATIVE: "✅",
    availability.FALLBACK: "⚠️",
    availability.UNAVAILABLE: "❌",
}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    return build_repo(tmp_path, monkeypatch)


@pytest.fixture
def layout(repo):
    return load_layout()


def _discover(layout):
    """Planned interface (plan §1): readme_targets.discover_targets."""
    from mjolnir_cli import readme_targets

    return readme_targets.discover_targets(layout)


def _ident(t) -> tuple:
    """The target's identity: (kind, payload…) — only fields the plan names."""
    from mjolnir_cli import readme_targets as rt

    if isinstance(t, rt.Root):
        return ("root",)
    if isinstance(t, rt.PlatformsIndex):
        return ("index",)
    if isinstance(t, rt.Platform):
        return ("platform", t.key, tuple(sorted(getattr(t, "chapters", ()))))
    if isinstance(t, rt.Model):
        return ("model", t.model)
    if isinstance(t, rt.Config):
        return ("config", t.model, t.quant, t.include_benchmarks)
    raise AssertionError(f"unexpected target type: {t!r}")


def _expected_idents(thor_chapters: tuple = CHAPTERS) -> set[tuple]:
    return {
        ("root",),
        ("index",),
        ("platform", "thor", tuple(sorted(thor_chapters))),
        ("platform", "spark", ()),
        ("platform", "orin", ()),
        ("model", M38),
        ("model", M36),
        ("config", M38, "NVFP4_MJOLNIR_E2M3_LMH8_VT64K", True),
        ("config", M38, "NVFP4", True),
        ("config", M38, "AWQ_FLASHINFER", True),
        ("config", M36, "NVFP4_MJOLNIR_E2M3", True),
        ("config", M36, "NVFP4_FA2", True),
    }


def _render_all():
    """Planned interface (plan §1): readme.render_all — the template loop."""
    from mjolnir_cli import readme

    return readme.render_all(load_layout())


def _line(text: str, *needles: str, exact: tuple = ()) -> str:
    """The one line carrying all needles ('' when none) — for matrix cells.
    ``exact`` needles must match as a whole token (NVFP4 vs NVFP4_FA2)."""
    for ln in text.splitlines():
        if all(n in ln for n in needles) and all(re.search(rf"(?<![\w-]){re.escape(e)}(?![\w-])", ln) for e in exact):
            return ln
    return ""


def _sections(text: str) -> dict[str, str]:
    """A page split into its '## ' sections, keyed by heading text."""
    parts = re.split(r"(?m)^## (.+)$", text)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def _section(text: str, *needles: str) -> str:
    """The body of the '## ' section whose heading carries all needles."""
    for head, body in _sections(text).items():
        if all(n in head for n in needles):
            return body
    raise AssertionError(f"no '## ' section matching {needles} in:\n{text[:2000]}")


# ── discovery + sparseness (plan §2.1) ──────────────────────────────────────


def test_discover_targets_yields_the_spec_target_set(layout):
    """2 models × 3 + 2 configs (overlays decide), 3 registry platforms, the
    root + index targets — and kernel chapters only for (thor, headline)."""
    idents = {_ident(t) for t in _discover(layout)}
    assert idents == _expected_idents()


@pytest.mark.parametrize(("model", "quant"), [(m, q) for m, qs in fx.CONFIGS.items() for q in qs], ids=lambda v: str(v))
def test_config_targets_always_include_benchmarks(repo, model, quant):
    """Spec target #7: BENCHMARKS.md is ALWAYS generated — include_benchmarks
    is True on every Config target, data or not."""
    layout = load_layout()
    cfgs = [t for t in _discover(layout) if _ident(t)[:3] == ("config", model, quant)]
    assert len(cfgs) == 1
    assert cfgs[0].include_benchmarks is True


def test_kernel_chapters_follow_the_platforms_headline_model_raws(layout):
    """Spec #4: chapters render for (platform, headline model of platform) —
    thor's headline is Qwen3.8-27B and the fixture has its verify raws, so
    the full §Directory Layout chapter set is discovered."""
    thor = next(t for t in _discover(layout) if _ident(t)[:2] == ("platform", "thor"))
    assert tuple(sorted(thor.chapters)) == tuple(sorted(CHAPTERS))


@pytest.mark.parametrize("platform", ["spark", "orin"])
def test_research_platforms_get_no_kernel_chapters(repo, platform):
    """No spark/orin verify raws exist ⇒ no kernel chapters, though spark has
    e2e history (kernel targets come from verify raws, not history)."""
    layout = load_layout()
    t = next(t for t in _discover(layout) if _ident(t)[:2] == ("platform", platform))
    assert t.chapters == ()


def test_raws_for_a_non_headline_model_create_no_chapters(tmp_path, monkeypatch):
    """A stage run of another model must not create thor kernel chapters —
    the case study is pinned to the platform's headline model (spec #12)."""
    from mjolnir_cli.verify import results

    root = build_repo(tmp_path, monkeypatch, with_data=False)
    d = results.header(
        "sweep",
        "20261013-000000",
        image="fixture",
        default=True,
        params={},
        kernel={},
        gate={},
        platform="thor",
    )
    d["model"] = M36
    results.save(root / "benchmarks" / "raw" / "verify", d)
    thor = next(t for t in _discover(load_layout()) if _ident(t)[:2] == ("platform", "thor"))
    assert thor.chapters == ()


def test_discovery_on_an_empty_repo_yields_every_page_target_and_no_chapters(tmp_path, monkeypatch):
    """Sparseness (spec table): root + index + per-platform + per-model +
    per-config pages are ALWAYS discovered — placeholders, not omission —
    while the kernel tree needs raws."""
    build_repo(tmp_path, monkeypatch, with_data=False)
    idents = {_ident(t) for t in _discover(load_layout())}
    assert idents == _expected_idents(thor_chapters=())


@pytest.mark.parametrize("platform", ["thor", "spark", "orin"])
def test_platform_targets_cover_exactly_the_registry(repo, platform):
    """One Platform target per shipped registry record — spark-cluster is
    Phase 3 and must not appear (spec §Open Items 4)."""
    layout = load_layout()
    keys = [t.key for t in _discover(layout) if _ident(t)[0] == "platform"]
    assert sorted(keys) == sorted(REGISTRY)
    assert platform in keys
    assert "spark-cluster" not in keys


def test_every_discovered_target_builds_a_context(repo):
    """Plan §1: per-target build_context(...) — every target builds a context
    from the fixture repo without touching the GPU stack."""
    layout = load_layout()
    for t in _discover(layout):
        assert t.build_context(layout) is not None, _ident(t)


# ── the template home (spec §Naming) ────────────────────────────────────────


@pytest.mark.parametrize(
    "tpl",
    [f"pages/{p}" for p in PAGE_TEMPLATES] + [f"kernel/{c}.md.j2" for c in CHAPTERS],
    ids=lambda v: v,
)
def test_the_template_home_holds_every_page_and_kernel_template(tpl):
    """docs/templates/ is the single template home (spec §Directory Layout):
    the five page templates and the moved kernel chapters live there."""
    assert (TEMPLATE_HOME / tpl).is_file(), f"missing docs/templates/{tpl}"


# ── platforms index + platform pages (spec targets #2, #3) ──────────────────


def test_platforms_index_lists_every_platform_with_status_and_links(repo):
    _render_all()
    page = (repo / "docs" / "platforms" / "README.md").read_text()
    for name in ("Jetson AGX Thor", "DGX Spark", "Jetson Orin"):
        assert name in page, name
    assert "supported" in page and "research" in page
    for p in REGISTRY:
        assert f"{p}/README.md" in page  # cross-linking map: index → platform pages


def test_platform_readme_shows_the_models_quants_matrix(repo):
    """The matrix is config-scan × availability: on the platform ⇒ the
    availability symbol; not on the platform ⇒ '—' (spec §Data→Page Mapping)."""
    _render_all()
    thor = (repo / "docs" / "platforms" / "thor" / "README.md").read_text()
    spark = (repo / "docs" / "platforms" / "spark" / "README.md").read_text()
    assert "✅" in _line(thor, FP6_QUANT)  # exists on thor, fp6+MJOLNIR native
    assert "—" in _line(thor, "AWQ_FLASHINFER")  # orin-only: not on thor
    assert "—" in _line(spark, "AWQ_FLASHINFER")  # orin-only: not on spark
    assert "✅" in _line(spark, "NVFP4", exact=("NVFP4",))  # fp8+FLASHINFER native on spark
    assert "❌" in _line(spark, FP6_QUANT)  # exists on spark, combo unavailable there


def test_availability_cells_render_the_table_states_and_whys(repo):
    """AC3: availability cells come ONLY from availability.py — every rendered
    cell shows the module's symbol and why for that (dtype/backend/platform),
    never a duplicated table (states pulled from the module, not typed here)."""
    _render_all()
    thor = (repo / "docs" / "platforms" / "thor" / "README.md").read_text()
    spark = (repo / "docs" / "platforms" / "spark" / "README.md").read_text()
    orin = (repo / "docs" / "platforms" / "orin" / "README.md").read_text()
    m36 = (repo / "configs" / M36 / "README.md").read_text()
    cells = [
        (thor, FP6_QUANT, availability.kv_state("fp6_e2m3", "MJOLNIR", "thor")),
        (spark, FP6_QUANT, availability.kv_state("fp6_e2m3", "MJOLNIR", "spark")),
        (spark, "NVFP4", availability.kv_state("fp8", "FLASHINFER", "spark")),
        (m36, "NVFP4_FA2", availability.kv_state("nvfp4", "FA2", "thor")),
    ]
    for page, row, cell in cells:
        assert SYMBOL[cell.state] in _line(page, row), f"{row}: cell symbol missing"
        assert cell.why in page, f"{row}: the table's why is not rendered"
    # weights-driven cell: the AWQ name maps to int4-w4a16 (spec §Directory
    # Layout: 'AWQ_FLASHINFER — int4 W4A16 weights')
    cell = availability.weights_state("int4-w4a16", "orin")
    assert SYMBOL[cell.state] in _line(orin, "AWQ_FLASHINFER")
    assert cell.why in orin


@pytest.mark.parametrize(
    ("platform", "req"),
    [
        ("thor", "JetPack 7.x"),
        ("spark", "DGX OS + CUDA 13"),
        ("orin", "JetPack 7.2.1+ (L4T 39.2.1, CUDA 13.2)"),
    ],
)
def test_platform_readme_carries_host_requirements(repo, platform, req):
    """The registry's host_requirements reach the platform page verbatim."""
    _render_all()
    assert req in (repo / "docs" / "platforms" / platform / "README.md").read_text()


def test_platform_readme_marks_measured_rows(repo):
    """Availability ≠ measured (decision 9): rows carry 'measured: yes/no'
    from history.jsonl — the spark NVFP4 pair is measured, the fp6 config on
    spark has no data, the orin-only config has none anywhere."""
    _render_all()
    spark = (repo / "docs" / "platforms" / "spark" / "README.md").read_text()
    orin = (repo / "docs" / "platforms" / "orin" / "README.md").read_text()
    assert "measured: yes" in _line(spark, "NVFP4", exact=("NVFP4",))
    assert "measured: no" in _line(spark, FP6_QUANT)
    assert "measured: no" in _line(orin, "AWQ_FLASHINFER")


@pytest.mark.parametrize(
    ("platform", "linked"),
    [("thor", True), ("spark", False)],
)
def test_platform_readme_links_the_kernel_case_study_only_with_raws(repo, platform, linked):
    """Spec #3/#4: the kernel link appears only when the platform has raws;
    spark gets the research-status note instead."""
    _render_all()
    page = (repo / "docs" / "platforms" / platform / "README.md").read_text()
    assert ("kernel/README.md" in page) is linked
    if not linked:
        assert "research" in page


def test_platform_readme_inlines_the_hand_written_overview(repo):
    """The overview.md escape hatch (spec §Directory Layout) is inlined
    verbatim by the platform template."""
    _render_all()
    page = (repo / "docs" / "platforms" / "thor" / "README.md").read_text()
    assert fx.OVERVIEW.strip() in page


# ── model pages (spec target #5) ────────────────────────────────────────────


def test_model_readme_renders_the_derived_intro_and_configs_table(repo):
    """Derived, not hand-written (decision 7): served name + the model's
    quants, each row naming its backend labels from the config scan."""
    _render_all()
    page = (repo / "configs" / M38 / "README.md").read_text()
    assert "Qwen/Qwen3.8-27B" in page  # the served-model-name from the scan
    for quant in fx.CONFIGS[M38]:
        assert quant in page
    assert "Mjolnir" in page and "FlashInfer" in page  # derived backend labels


def test_model_readme_shows_the_per_platform_e2e_ab_rows(repo):
    """One A/B row per platform (latest_pair(platform, model)) + the how()
    panel: thor +30.8% (fixture pair), spark +28.9% — each from that
    platform's own history."""
    _render_all()
    page = (repo / "configs" / M38 / "README.md").read_text()
    assert fx.THOR_GAIN in page and fx.SPARK_GAIN in page
    assert "How the % and × here are calculated" in page  # the how("e2e") panel


def test_model_readme_quickstart_argv_follows_the_platform(repo):
    """Per-platform quickstart (spec §Data→Page Mapping): the argv names the
    platform and a config that exists there — spark's is the FlashInfer one."""
    _render_all()
    page = (repo / "configs" / M38 / "README.md").read_text()
    thor_ln = _line(page, "--platform thor")
    spark_ln = _line(page, "--platform spark")
    assert "--config NVFP4_MJOLNIR_E2M3_LMH8_VT64K" in thor_ln
    assert "--config NVFP4 " in spark_ln or spark_ln.rstrip().endswith("--config NVFP4")


def test_model_readme_stays_lean(repo):
    """Audience rule (spec §Audience): no kernel internals on model pages."""
    _render_all()
    page = (repo / "configs" / M38 / "README.md").read_text()
    assert "kernel-speedup" not in page and "L-sweep" not in page


# ── config pages (spec targets #6, #7) ──────────────────────────────────────


def test_config_readme_renders_static_facts_and_env_keys(repo):
    """Static facts from config.yaml + the .env sidecar keys (spec row 6)."""
    _render_all()
    page = (repo / "configs" / M38 / FP6_QUANT / "README.md").read_text()
    assert "fp6_e2m3" in page and "MJOLNIR" in page and "256K" in page
    assert "MJOLNIR_FP8_LM_HEAD" in page and "MJOLNIR_DRAFT_VOCAB" in page


def test_config_readme_shows_overlay_deltas_as_columns(repo):
    """Spec change (cycle 3e, documented): the per-platform delta dump
    (key | value | from) is gone — the overlay delta surfaces as a column of
    the serving-profile table and the file attribution is the intro's
    platforms/ link. The base value stays visible (identical-across-platform
    values repeat per column)."""
    _render_all()
    page = (repo / "configs" / M38 / "NVFP4" / "README.md").read_text()
    assert "spark.yaml" not in page  # the dump's attribution is gone
    assert "| parameter | Thor | Spark |" in page  # one column per platform
    assert "| Max context | 200K | 128K |" in page  # the delta, as a column


def test_config_readme_renders_the_serve_argv_per_platform(repo):
    """`mjolnir serve --platform <p>` rendered argv per platform (spec row 6):
    one line per platform the config exists on, naming the config."""
    _render_all()
    fp6 = (repo / "configs" / M38 / FP6_QUANT / "README.md").read_text()
    ln = _line(fp6, "mjolnir serve", "--platform thor")
    assert f"--config {FP6_QUANT}" in ln
    nvfp4 = (repo / "configs" / M38 / "NVFP4" / "README.md").read_text()
    for platform in ("thor", "spark"):
        assert "--config NVFP4" in _line(nvfp4, "mjolnir serve", f"--platform {platform}")


def test_config_readme_badges_come_from_the_availability_table(repo):
    """The badges are the table's states + whys for the platforms the config
    exists on — the nvfp4×FA2 config on thor is exists-but-unavailable ❌."""
    _render_all()
    fp6 = (repo / "configs" / M38 / FP6_QUANT / "README.md").read_text()
    cell = availability.kv_state("fp6_e2m3", "MJOLNIR", "thor")
    assert SYMBOL[cell.state] in fp6 and cell.why in fp6
    fa2 = (repo / "configs" / M36 / "NVFP4_FA2" / "README.md").read_text()
    cell = availability.kv_state("nvfp4", "FA2", "thor")
    assert SYMBOL[cell.state] in fa2 and cell.why in fa2


# ── BENCHMARKS.md (spec target #7) ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "quant", "platforms", "no_data"),
    [
        (M38, FP6_QUANT, ("Thor",), 0),
        (M38, "NVFP4", ("Thor", "DGX Spark"), 0),
        (M38, "AWQ_FLASHINFER", ("Jetson Orin",), 1),
        (M36, "NVFP4_MJOLNIR_E2M3", ("Thor",), 1),
    ],
)
def test_benchmarks_md_has_a_section_per_platform_the_config_exists_on(repo, model, quant, platforms, no_data):
    """Sections follow the overlay rule (a config exists where its overlays
    put it); a section with no (platform, model, config) data shows the
    'no data yet' placeholder (spec #7 sparseness rule)."""
    _render_all()
    page = (repo / "configs" / model / quant / "BENCHMARKS.md").read_text()
    for name in platforms:
        assert name in page, name
    assert page.count("no data yet") == no_data


def test_benchmarks_md_empty_cell_shows_no_data_yet(repo):
    """A config whose cell has no rows: the placeholder in its platform
    section — while the platform-level kernel stages still render (they come
    from the headline model's raws, spec row 7)."""
    _render_all()
    page = (repo / "configs" / M36 / "NVFP4_MJOLNIR_E2M3" / "BENCHMARKS.md").read_text()
    thor_sec = _section(page, "Thor")
    assert "no data yet" in thor_sec
    assert "1.26" in thor_sec  # the fixture sweep's geomean — kernel stages are platform-level


def test_benchmarks_md_fills_the_thor_cell_from_the_raws(repo):
    """The measured cell: e2e decode from history (52.7), spec acceptance from
    the spec raw (46.9), KV tokens from the kv-cache-size raw (410,000),
    accuracy suite rows, kernel stages — every number from raw data."""
    _render_all()
    page = (repo / "configs" / M38 / FP6_QUANT / "BENCHMARKS.md").read_text()
    thor_sec = _section(page, "Thor")
    assert "52.7" in thor_sec  # history.jsonl decode t/s at 1 request
    assert "46.9" in thor_sec  # spec raw: best k=4 decode t/s for this config
    assert re.search(r"410[,\s]?000", thor_sec)  # kv-cache-size raw: fp6 tokens
    assert "Tool calls" in thor_sec  # accuracy.json suite row
    assert "1.26" in thor_sec  # sweep geomean (kernel stage summary)
    assert "benchmarks/raw/verify/" in page  # per-stage raw provenance (cross-linking map)


def test_benchmarks_md_accuracy_rows_follow_their_platform(repo):
    """Accuracy is platform-scoped (spec #7: every number from raw data *for
    this cell*): a multi-platform config whose only accuracy legs ran on thor
    shows no accuracy table under its spark section — thor-measured scores are
    never presented as another platform's data, and a section whose only data
    would have been leaked accuracy counts as empty (`has_data`)."""
    _render_all()
    page = (repo / "configs" / M38 / "NVFP4" / "BENCHMARKS.md").read_text()
    spark_sec = _section(page, "DGX Spark")
    assert "Model quality" not in spark_sec  # the fixture's accuracy legs are thor legs
    assert "Tool calls" not in spark_sec
    thor_sec = _section(page, "Thor")
    assert "Tool calls" in thor_sec  # the thor cell keeps its suite rows


def test_benchmarks_md_references_the_v2_chart_paths(repo):
    """Charts live under assets/benchmarks/<platform>/<model-short>/ and the
    config page's references resolve from three directories deep."""
    _render_all()
    page = (repo / "configs" / M38 / FP6_QUANT / "BENCHMARKS.md").read_text()
    assert "../../../assets/benchmarks/thor/Qwen3.8-27B/" in page


# ── cross-linking (spec §Cross-Linking Map) + root README (spec target #1) ──


def _md_links(text: str) -> list[str]:
    """Relative markdown link/image targets (external + anchors skipped)."""
    out = []
    for target in (
        re.findall(r"\]\(([^)\s]+)\)", text)
        + re.findall(r'<img src="([^"\s]+)"', text)  # chart() emits plain dark <img>s
    ):
        if target.startswith(("http", "#", "mailto:")) or "{{" in target:
            continue
        out.append(target.split("#")[0])
    return out


def test_every_relative_link_in_every_rendered_page_resolves(repo):
    """Every relative link/image on every generated page names a file that
    exists in the rendered tree (spec §Cross-Linking Map, all arrows)."""
    _render_all()
    pages = [repo / "README.md", repo / "docs" / "platforms" / "README.md"]
    pages += sorted((repo / "docs" / "platforms").glob("*/README.md"))
    pages += (
        sorted((repo / "docs" / "platforms" / "thor" / "kernel").glob("*.md"))
        if (repo / "docs" / "platforms" / "thor" / "kernel").is_dir()
        else []
    )
    pages += sorted((repo / "configs").glob("*/*/README.md")) + sorted((repo / "configs").glob("*/*/BENCHMARKS.md"))
    config_pages = sorted((repo / "configs").glob("*/*/*/README.md")) + sorted(
        (repo / "configs").glob("*/*/*/BENCHMARKS.md")
    )
    assert len(config_pages) == 10  # every fixture config's README + BENCHMARKS link-checked
    pages += config_pages
    assert len(pages) > 10  # the full target set rendered
    broken = [
        f"{page.relative_to(repo)} -> {target}"
        for page in pages
        for target in _md_links(page.read_text())
        if not (page.parent / target).exists()
    ]
    assert not broken, broken


def test_root_readme_renders_the_platform_matrix(repo):
    """Root = status matrix + links (decision 3): every platform rows-linked
    into docs/platforms/."""
    _render_all()
    text = (repo / "README.md").read_text()
    assert "{{" not in text and "{%" not in text
    for p in REGISTRY:
        assert f"docs/platforms/{p}/README.md" in text
    assert "research" in text  # spark/orin rows carry their status


def test_root_readme_renders_the_measured_pair_index(repo):
    """Measured-pair index (spec row 1): links into the measured model's
    config-tree page."""
    _render_all()
    text = (repo / "README.md").read_text()
    assert f"configs/{M38}/README.md" in text


def test_root_headline_stays_pinned_to_thor(repo):
    """Decision 3: the headline pair is (thor, Qwen3.8-27B) — the root shows
    the thor pair's gain and never the spark pair's (+28.9% / +29%)."""
    _render_all()
    text = (repo / "README.md").read_text()
    assert "+30.8%" in text or "+31%" in text, "the pinned (thor, Qwen3.8-27B) gain must show"
    assert "+28.9%" not in text and "+29%" not in text


def test_no_generated_page_carries_trailing_whitespace(repo):
    """Render output must be commit-clean: the pre-commit trailing-whitespace
    hook strips trailing spaces at commit time while CI's reproduce gate
    re-renders and diffs — a template that emits one (the kernel chapters
    line's trailing ' · ') would make every render differ from HEAD."""
    _render_all()
    pages = [
        repo / "README.md",
        repo / "docker" / "vllm-thor" / "README.md",
        *sorted((repo / "docs" / "platforms").rglob("*.md")),
        *sorted((repo / "configs").rglob("*.md")),
    ]
    dirty = [
        str(p.relative_to(repo))
        for p in pages
        if p.is_file() and any(ln.rstrip("\n") != ln.rstrip() for ln in p.read_text().splitlines())
    ]
    assert not dirty, dirty


# ── idempotency (CI reproduce gate, spec §Render Triggers) ──────────────────


def test_render_all_is_idempotent_across_the_target_set(repo):
    """Render twice → byte-identical outputs across every target kind (the
    CI gate re-renders and fails on diff)."""
    _render_all()
    sample = [
        repo / "README.md",
        repo / "docs" / "platforms" / "README.md",
        repo / "docs" / "platforms" / "thor" / "README.md",
        repo / "docs" / "platforms" / "thor" / "kernel" / "ring.md",
        repo / "configs" / M38 / "README.md",
        repo / "configs" / M38 / FP6_QUANT / "README.md",
        repo / "configs" / M38 / FP6_QUANT / "BENCHMARKS.md",
    ]
    first = [p.read_bytes() for p in sample]
    _render_all()
    assert [p.read_bytes() for p in sample] == first
