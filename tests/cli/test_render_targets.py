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

Fixture repos: :mod:`render_targets_fixtures` — two models × six configs with
platform overlays, verify_fixtures-style raws stamped (thor, Qwen3.8-27B), a
spark e2e pair the root headline must ignore, and a seeded v2 chart manifest.
The real ``docs/`` tree is never written to; matplotlib is never needed here
(chart references resolve from the seeded manifest).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mjolnir_cli import availability, platforms
from mjolnir_cli.config import REGISTRY, load_layout

import render_targets_fixtures as fx
from render_targets_fixtures import FP6_QUANT, M36, M38, build_repo

REPO = Path(__file__).resolve().parents[2]
TEMPLATE_HOME = REPO / "docs" / "templates"

# the spec §Directory Layout chapter set (docs/templates/kernel/) — cycle 3k:
# the case-study README is no longer a chapter (the index lives on the
# platform page's `## Kernel` section, built from kernel/_index.md.j2)
CHAPTERS = (
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
        return ("config", t.model, t.quant)
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
        ("config", M38, "NVFP4_MJOLNIR_E2M3_LMH8_VT64K"),
        ("config", M38, "NVFP4"),
        ("config", M38, "AWQ_FLASHINFER"),
        ("config", M36, "NVFP4_MJOLNIR_E2M3"),
        ("config", M36, "NVFP4_FA2"),
        ("config", M36, "NVFP4_FA4"),
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
def test_config_targets_emit_a_single_page(repo, model, quant):
    """Spec change (cycle 3f): BENCHMARKS.md is folded into the config
    README — each Config target emits exactly one page (the README)."""
    layout = load_layout()
    cfgs = [t for t in _discover(layout) if _ident(t)[:3] == ("config", model, quant)]
    assert len(cfgs) == 1
    outs = cfgs[0].outputs(repo, layout)
    assert len(outs) == 1 and outs[0][1].name == "README.md"


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
    [f"pages/{p}" for p in PAGE_TEMPLATES] + [f"kernel/{c}.md.j2" for c in CHAPTERS] + ["kernel/_index.md.j2"],
    ids=lambda v: v,
)
def test_the_template_home_holds_every_page_and_kernel_template(tpl):
    """docs/templates/ is the single template home (spec §Directory Layout):
    the five page templates, the kernel chapters and the shared kernel-index
    partial live there — the root README template lives at the repo root
    (``README.md.j2``)."""
    assert (TEMPLATE_HOME / tpl).is_file(), f"missing docs/templates/{tpl}"


def test_the_root_readme_template_lives_at_the_repo_root():
    assert (REPO / "README.md.j2").is_file()


# ── platforms index + platform pages (spec targets #2, #3) ──────────────────


def test_platforms_index_lists_every_platform_with_status_and_links(repo):
    _render_all()
    page = (repo / "docs" / "platforms" / "README.md").read_text()
    for name in ("Jetson AGX Thor", "DGX Spark", "Jetson Orin"):
        assert name in page, name
    assert "supported" in page and "research" in page
    for p in REGISTRY:
        assert f"{p}/README.md" in page  # cross-linking map: index → platform pages


def test_platform_readme_has_one_row_per_model_with_a_default_config(repo):
    """Cycle 3k §1: the models table is ONE row per `<org>/<model>` (linked to
    the model README) — never per config — with a default-config column
    (linked) and a measured ✅/—. The default-config rule (documented on the
    builder): no registry record carries a per-platform default config yet, so
    it is the model page's quickstart rule — DEFAULT_QUANT when the platform's
    overlays allow it, else the first config they allow (scan order)."""
    _render_all()
    thor = (repo / "docs" / "platforms" / "thor" / "README.md").read_text()
    serving = _section(thor, "Serving")
    rows = [ln for ln in serving.splitlines() if ln.startswith("| [`")]
    assert len(rows) == 2, f"one row per model, got {len(rows)}: {rows}"
    m38 = _line(serving, f"configs/{M38}/README.md")
    assert f"[`{FP6_QUANT}`](../../../configs/{M38}/{FP6_QUANT}/README.md)" in m38
    m36 = _line(serving, f"configs/{M36}/README.md")
    # no M36 config is DEFAULT_QUANT, so the rule falls to the scan-first
    # config the overlays allow (NVFP4_FA2, a legacy thor-only dir)
    assert f"[`NVFP4_FA2`](../../../configs/{M36}/NVFP4_FA2/README.md)" in m36
    # the availability-prose legend is gone: per-config availability lives on
    # the model page (cycle 3k §1)
    assert f"`{FP6_QUANT}` on Jetson AGX Thor:" not in thor
    spark = (repo / "docs" / "platforms" / "spark" / "README.md").read_text()
    spark_rows = [ln for ln in _section(spark, "Serving").splitlines() if ln.startswith("| [`")]
    assert len(spark_rows) == 2
    # spark's overlays serve NVFP4 first for M38; M36 has no spark config
    assert "[`NVFP4`](../../../configs/Qwen/Qwen3.8-27B/NVFP4/README.md)" in _line(spark, f"configs/{M38}/README.md")
    assert "| — | — |" in _line(spark, f"configs/{M36}/README.md")


def test_platform_readme_measured_column_follows_the_pair_history(repo):
    """The measured column is ✅ when the (platform, model) pair has e2e
    history, — otherwise (availability ≠ measured, decision 9)."""
    _render_all()
    thor = (repo / "docs" / "platforms" / "thor" / "README.md").read_text()
    assert "✅" in _line(_section(thor, "Serving"), f"configs/{M38}/README.md")
    assert "✅" in _line(_section(thor, "Serving"), f"configs/{M36}/README.md")
    spark = (repo / "docs" / "platforms" / "spark" / "README.md").read_text()
    assert "✅" in _line(_section(spark, "Serving"), f"configs/{M38}/README.md")
    assert "| — |" in _line(_section(spark, "Serving"), f"configs/{M36}/README.md")


@pytest.mark.parametrize(
    ("platform", "req"),
    [
        ("thor", "JetPack 7.x"),
        ("spark", "DGX OS + CUDA 13"),
        ("orin", "JetPack 7.2.1+"),
    ],
)
def test_platform_readme_carries_host_requirements(repo, platform, req):
    """The registry's host_requirements reach the platform page verbatim."""
    _render_all()
    assert req in (repo / "docs" / "platforms" / platform / "README.md").read_text()


@pytest.mark.parametrize(
    ("platform", "indexed"),
    [("thor", True), ("spark", False)],
)
def test_platform_readme_carries_the_kernel_index_only_with_raws(repo, platform, indexed):
    """Cycle 3k §1: `## Kernel` embeds the old case-study index (headline
    findings, chart, chapter table) only when the platform has raws; a rawless
    platform renders no kernel section at all — and the kernel README is never
    emitted (its content moved into the section)."""
    _render_all()
    page = (repo / "docs" / "platforms" / platform / "README.md").read_text()
    assert ("## Kernel" in page) is indexed
    if indexed:
        assert "Where it stands" in page  # the headline finding, reused
        assert "](kernel/sweep.md)" in page  # the chapter table links the chapters
        assert "<img" in page  # the pair's kernel-speedup chart
    assert "kernel/README.md" not in page  # the index is the section, not a page


def test_the_kernel_readme_is_never_rendered(repo):
    """Cycle 3k §1: docs/platforms/thor/kernel/README.md is deleted — the
    chapter pages stay, the README does not come back."""
    _render_all()
    assert not (repo / "docs" / "platforms" / "thor" / "kernel" / "README.md").exists()


def test_platform_readme_renders_the_default_model_e2e_only_with_data(repo):
    """Cycle 3k §1: the registry's headline model + a measured pair → an
    `## End to end` section with the model-page A/B row (compact table) and
    the pair's e2e charts; no headline model (spark) → no section."""
    _render_all()
    thor = (repo / "docs" / "platforms" / "thor" / "README.md").read_text()
    e2e = _section(thor, "End to end")
    assert f"`{FP6_QUANT}`" in e2e and "`NVFP4`" in e2e  # both sides named
    assert fx.THOR_GAIN in e2e  # the same pair data the model page renders
    assert "e2e-decode.png" in e2e  # the pair-scoped chart
    spark = (repo / "docs" / "platforms" / "spark" / "README.md").read_text()
    assert "## End to end" not in spark  # no headline model pinned yet


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
    """Per-platform quickstart (spec §Data→Page Mapping). Spec change
    (cycle 3h — detection-first): no ``--platform`` on the serve line; each
    line names a config that exists on that platform — spark's is the
    FlashInfer one."""
    _render_all()
    page = (repo / "configs" / M38 / "README.md").read_text()
    thor_ln = _line(page, "--config NVFP4_MJOLNIR_E2M3_LMH8_VT64K")
    spark_ln = _line(page, "--config NVFP4", exact=("NVFP4",))
    for ln in (thor_ln, spark_ln):
        assert "mjolnir serve up --config" in ln
        assert "--platform" not in ln
    assert "--platform" not in page


def test_model_readme_stays_lean(repo):
    """Audience rule (spec §Audience): no kernel internals on model pages."""
    _render_all()
    page = (repo / "configs" / M38 / "README.md").read_text()
    assert "kernel-speedup" not in page and "L-sweep" not in page


# ── config pages (spec targets #6, #7) ──────────────────────────────────────


def test_config_readme_renders_static_facts(repo):
    """Static facts from config.yaml (spec row 6). Spec change (cycle 3e
    amendment 2026-10-20): the `.env` sidecar's keys are no longer rendered —
    the README points at the sidecar, it does not quote it."""
    _render_all()
    page = (repo / "configs" / M38 / FP6_QUANT / "README.md").read_text()
    assert "fp6_e2m3" in page and "MJOLNIR" in page and "256K" in page


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
    """`mjolnir serve up` rendered per platform (spec row 6). Spec change
    (cycle 3h — detection-first): the flagless line, one serve block per
    platform the config exists on, naming the config."""
    _render_all()
    fp6 = (repo / "configs" / M38 / FP6_QUANT / "README.md").read_text()
    ln = _line(fp6, "mjolnir serve up")
    assert f"--config {FP6_QUANT}" in ln
    assert "--platform" not in ln
    nvfp4 = (repo / "configs" / M38 / "NVFP4" / "README.md").read_text()
    assert nvfp4.count("mjolnir serve up") == 2  # one block per platform (thor, spark)
    assert "--platform" not in nvfp4


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


# ── the merged Benchmarks section (cycle 3f: BENCHMARKS.md folded into the
#    config README; the old BENCHMARKS.md pins re-pointed to the new shape) ──


@pytest.mark.parametrize(
    ("model", "quant", "state"),
    [(M38, FP6_QUANT, "pair"), (M36, "NVFP4_FA2", "solo"), (M38, "AWQ_FLASHINFER", "none")],
)
def test_benchmarks_open_section_covers_pair_solo_and_no_data(repo, model, quant, state):
    """The open end-to-end state per config cell (cycle 3f §1): the A/B table
    + chart when the config is a side of the (platform, model) pair, its own
    rows when it has history but no partner, the honest one-liner otherwise."""
    _render_all()
    page = (repo / "configs" / model / quant / "README.md").read_text()
    fold = page.split("<details>")[0]
    if state == "pair":
        assert "### End to end (" in fold and "| change |" in fold
    elif state == "solo":
        assert "### End to end" in fold and "| metric | `NVFP4_FA2` |" in fold
        assert "| change |" not in fold
    else:
        assert "no data yet" in fold and "### End to end" not in fold


def test_benchmarks_data_lives_in_the_merged_readme(repo):
    """The (platform, model, config) cell renders in the README's Benchmarks
    section — every number from raw data. The kernel stages are NOT on the
    page (Simplify-1 §3): they are the 13 chapter pages' content — the case
    study is one link away instead."""
    _render_all()
    page = (repo / "configs" / M38 / FP6_QUANT / "README.md").read_text()
    bench = _section(page, "Benchmarks")
    assert "52.7" in bench  # history.jsonl decode t/s at 1 request
    assert "46.9" in bench  # spec raw: best k=4 decode t/s for this config
    assert re.search(r"410[,\s]?000", bench)  # kv-cache-size raw: fp6 tokens
    assert "Tool calls" in bench  # accuracy.json suite row
    assert "1.26" not in bench  # no inline sweep summary
    assert "benchmarks/raw/verify/" not in bench  # no per-stage provenance
    assert (
        "[Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#kernel)" in bench
    )  # the one replacement line


def test_benchmarks_multi_platform_sections_carry_platform_headers(repo):
    """A config serving (and measured) on several platforms: the Benchmarks
    section's open sub-blocks carry per-platform headings, and the accuracy
    rows follow their platform (the fixture's accuracy legs are thor legs)."""
    _render_all()
    page = (repo / "configs" / M38 / "NVFP4" / "README.md").read_text()
    bench = _section(page, "Benchmarks")
    assert "### End to end — Jetson AGX Thor (NVFP4 vs" in bench
    assert "### End to end — DGX Spark (NVFP4 vs NVFP4_MJOLNIR," in bench
    acc = page.split("<summary>Model quality (per suite)</summary>")[1].split("<details>")[0]
    assert "Tool calls" in acc  # the thor cell keeps its suite rows
    assert "DGX Spark" not in acc  # never presented as another platform's data


def test_benchmarks_no_data_line_plus_the_kernel_case_study_link(repo):
    """A config cell with no data: the honest line stays, and the kernel
    stages are still never inline — the pair-level case study is one link
    away (it comes from the headline model's raws, not this config's)."""
    _render_all()
    page = (repo / "configs" / M36 / "NVFP4_MJOLNIR_E2M3" / "README.md").read_text()
    assert "no data yet" in page.split("<details>")[0]
    assert "1.26" not in page
    assert "[Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#kernel)" in page


def test_benchmarks_chart_references_resolve_from_the_config_dir(repo):
    """Charts live under assets/benchmarks/<platform>/<model-short>/ and the
    config page's references resolve from four directories deep."""
    _render_all()
    page = (repo / "configs" / M38 / FP6_QUANT / "README.md").read_text()
    assert "../../../../assets/benchmarks/thor/Qwen3.8-27B/e2e-decode.png" in page


# ── cross-linking (spec §Cross-Linking Map) + root README (spec target #1) ──


def _md_links(text: str) -> list[str]:
    """Relative markdown link/image targets, `#anchor` kept (external skipped)."""
    out = []
    for target in (
        re.findall(r"\]\(([^)\s]+)\)", text)
        + re.findall(r'<img src="([^"\s]+)"', text)  # chart() emits plain dark <img>s
    ):
        if target.startswith(("http", "mailto:")) or "{{" in target:
            continue
        out.append(target)
    return out


def _gh_anchors(text: str) -> set[str]:
    """GitHub-style slugs of every heading (fenced code blocks carry none)."""
    body = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return {
        re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-").strip()
        for h in re.findall(r"^#{1,6} (.+)$", body, flags=re.MULTILINE)
    }


def test_every_relative_link_in_every_rendered_page_resolves(repo):
    """Every relative link/image on every generated page names a file that
    exists in the rendered tree, and every `#anchor` names a real heading of
    its target file (spec §Cross-Linking Map, all arrows)."""
    _render_all()
    # docs/cli.md is written by the plot command (it holds the root app), not
    # the render loop — the README links to it, so write it as plot would
    from mjolnir_cli import cli, clidoc

    clidoc.write_reference(repo, cli.app)
    pages = [repo / "README.md", repo / "docs" / "platforms" / "README.md"]
    pages += sorted((repo / "docs" / "platforms").glob("*/README.md"))
    pages += (
        sorted((repo / "docs" / "platforms" / "thor" / "kernel").glob("*.md"))
        if (repo / "docs" / "platforms" / "thor" / "kernel").is_dir()
        else []
    )
    pages += sorted((repo / "configs").glob("*/*/README.md"))
    config_pages = sorted((repo / "configs").glob("*/*/*/README.md"))
    assert len(config_pages) == 6  # every fixture config's README — the only config page (cycle 3f)
    pages += config_pages
    assert len(pages) > 10  # the full target set rendered
    broken = []
    for page in pages:
        text = page.read_text()
        for target in _md_links(text):
            path, _, anchor = target.partition("#")
            if not anchor:
                ok = (page.parent / path).exists()
            elif not path:  # same-file anchor
                ok = anchor in _gh_anchors(text)
            else:
                f = page.parent / path
                ok = f.exists() and anchor in _gh_anchors(f.read_text())
            if not ok:
                broken.append(f"{page.relative_to(repo)} -> {target}")
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


def test_model_and_platform_pages_link_only_the_config_readme(repo):
    """Re-pin ripple (cycle 3f §2): the model table's config cell and the
    platform matrix keep ONE link per config — the README; the BENCHMARKS
    links are gone."""
    _render_all()
    model = (repo / "configs" / M38 / "README.md").read_text()
    assert "BENCHMARKS" not in model
    assert f"]({FP6_QUANT}/README.md)" in model
    thor = (repo / "docs" / "platforms" / "thor" / "README.md").read_text()
    assert "BENCHMARKS" not in thor
    assert f"../../../configs/{M38}/{FP6_QUANT}/README.md" in thor


def test_root_headline_stays_pinned_to_thor(repo):
    """Decision 3: the headline pair is (thor, Qwen3.8-27B) — the root shows
    the thor pair's gain and never the spark pair's (+28.9% / +29%)."""
    _render_all()
    text = (repo / "README.md").read_text()
    assert "+30.8%" in text or "+31%" in text, "the pinned (thor, Qwen3.8-27B) gain must show"
    assert "+28.9%" not in text and "+29%" not in text


# ── the root README's hardware + models tables (cycle 3g) ───────────────────


def test_hw_table_renders_icon_word_status_and_bold_linked_names(repo):
    """Plan §1: the supported-hardware table is the transposed platform grid —
    platforms as columns, each header cell the emoji art over the bold linked
    name (centered columns), a left key column, and the status / kernel /
    host-requirements values stacked under each one. Raw HTML: GitHub's
    animated-image player never upgrades `<br>`-shaped pipe-table headers, so
    the table ships as explicit HTML like the features grid."""
    _render_all()
    hw = _section((repo / "README.md").read_text(), "Supported Hardware")
    # the header row: emoji art over the bold linked name, one cell per platform
    assert 'width="72"><br><strong>Jetson AGX Thor</strong></a>' in hw
    assert "assets/emoji/mjolnir_thor_emoji.png" in hw
    for p in platforms.REGISTRY.values():
        assert hw.count(f"<strong>{p.name}</strong></a>") == 1
    # the key rows: status (icon + word, registry-derived), kernel (the
    # data-driven case-study link — the fixture has thor raws only), host
    # requirements — one row per key, values in registry order
    assert (
        '<td align="center"><strong>status</strong></td><td align="center">✅ supported</td>'
        '<td align="center">⏳ research</td><td align="center">✅ supported</td></tr>' in hw
    )
    assert (
        '<td align="center"><strong>kernel</strong></td>'
        '<td align="center"><a href="docs/platforms/thor/README.md#kernel">case study</a></td>'
        '<td align="center">—</td><td align="center">—</td></tr>' in hw
    )
    assert (
        '<td align="center"><strong>host requirements</strong></td><td align="center">JetPack 7.x</td>'
        '<td align="center">DGX OS + CUDA 13</td><td align="center">JetPack 7.2.1+</td></tr>'
        in hw
    )


def test_status_icon_map_is_data_driven(monkeypatch):
    """A registry platform with status ``not_supported`` renders ❌ — the icon
    is map data keyed by status, never a platform-name branch."""
    from mjolnir_cli import platforms, readme_targets

    retired = platforms.Platform(
        key="retired",
        name="Retired Board",
        sm=(9, 9),
        sm_family="sm_99",
        image_repo="x/y",
        default_image="x:y",
        stock_image=None,
        vllm_version=None,
        status="not_supported",
    )
    monkeypatch.setitem(platforms.REGISTRY, "retired", retired)
    rows = {r.key: r for r in readme_targets.platform_rows()}
    assert rows["retired"].status_icon == "❌"
    assert rows["thor"].status_icon == "✅"
    assert rows["spark"].status_icon == "⏳"


def test_models_table_cells_native_with_history_native_without_and_no_configs(repo):
    """Plan §2 cell semantics on the fixture: (a) thor has native configs and
    (1, 0) history → `✅ <t/s> t/s`, fastest config wins (the fp6 config's
    52.7 beats NVFP4's 40.3, its name in the cell title); (b) spark/orin have
    a native config but no (1, 0) history → the bare ✅; (d) Qwen3.6 targets
    only thor (all its configs are thor-only) → — on spark/orin."""
    _render_all()
    models = _section((repo / "README.md").read_text(), "Supported Models")
    m38 = _line(models, "configs/Qwen/Qwen3.8-27B/README.md")
    assert '<span title="NVFP4_MJOLNIR_E2M3_LMH8_VT64K">✅ 52.7 t/s</span>' in m38
    assert m38.endswith(
        '<td align="center">✅</td><td align="center">✅</td></tr>'
    )  # spark + orin: native, no (1, 0) data
    m36 = _line(models, "configs/nvidia/Qwen3.6-35B-A3B/README.md")
    assert (
        '<td align="center">✅</td><td align="center">—</td><td align="center">—</td></tr>' in m36
    )  # thor native; spark/orin: no config targets them


def test_models_table_marks_every_config_unavailable(repo):
    """(c) every config of the model targeting the platform is explicitly
    unavailable there → ❌. Test-local tweak: the fp6 config gains a spark
    overlay (the Mjolnir kernel is the only fp6 reader — thor only), making
    it Qwen3.6's only spark-targeting config."""
    d = repo / "configs" / M36 / "NVFP4_MJOLNIR_E2M3" / "platforms"
    d.mkdir(exist_ok=True)
    (d / "spark.yaml").write_text("{}\n")
    _render_all()
    models = _section((repo / "README.md").read_text(), "Supported Models")
    assert '<td align="center">❌</td>' in _line(models, "configs/nvidia/Qwen3.6-35B-A3B/README.md")


def test_models_table_fastest_config_tie_breaks_on_scan_order(repo):
    """Fastest config wins; an exact tie resolves deterministically to the
    scan-first config (configs/ glob order), pinned via the cell title."""
    import json

    layout = load_layout()
    recs = fx.history_records()
    for r in recs:
        if r["platform"] == "thor" and r["model"] == M38:
            for c in r["cells"]:
                if c["concurrency"] == 1 and (c.get("context") or 0) == 0:
                    c["tg_tps"]["mean"] = 50.0
    layout.history_file.write_text("".join(json.dumps(r) + "\n" for r in recs))
    _render_all()
    models = _section((repo / "README.md").read_text(), "Supported Models")
    assert '<span title="NVFP4">✅ 50.0 t/s</span>' in _line(models, "configs/Qwen/Qwen3.8-27B/README.md")


def test_models_table_header_and_row_links_and_footnote(repo):
    """Plan §2/§3: one linked column per registry platform, one linked row per
    scanned model, and the footnote's cell definition + history provenance."""
    _render_all()
    models = _section((repo / "README.md").read_text(), "Supported Models")
    for p in REGISTRY:
        assert f'href="docs/platforms/{p}/README.md"' in models
    assert 'href="configs/Qwen/Qwen3.8-27B/README.md"' in models
    assert 'href="configs/nvidia/Qwen3.6-35B-A3B/README.md"' in models
    assert "1 request" in models and "0 context" in models
    assert "[benchmarks/history.jsonl](benchmarks/history.jsonl)" in models


def test_both_measured_pairs_lists_are_gone(repo):
    """Plan §3: no `Measured (platform, model) pairs` list anywhere — the
    models table replaces both copies. The kernel benchmark sections moved
    off the root README to the Thor platform page; the rule is checked there
    too."""
    _render_all()
    text = (repo / "README.md").read_text()
    assert "Measured (platform, model) pairs" not in text
    assert "- [" not in _section(text, "Supported Models")
    thor = (repo / "docs" / "platforms" / "thor" / "README.md").read_text()
    assert "Measured (platform, model) pairs" not in thor


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
    ]
    first = [p.read_bytes() for p in sample]
    _render_all()
    assert [p.read_bytes() for p in sample] == first
