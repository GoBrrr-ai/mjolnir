"""Cycle-3e RED: the config-README redesign (task/plan.md §1–§3).

The per-config ``README.md`` must *interpret* config.yaml, not mirror it:
the H1 is the model, the profile is secondary, ONE curated meta-table
carries a column per serving platform (overlay deltas surface as columns),
and the config.yaml / .env / delta dumps are gone. Serve lines and
availability are untouched; §9 folds BENCHMARKS.md into the page.

Fixture: :mod:`render_targets_fixtures` — the FP6 config carries the full
curated key set (incl. dict-valued keys that must never appear); NVFP4
serves thor + spark with a max-model-len overlay delta, the columns case.
The last tests are the repo-tree ratchets (plan §4) on the committed pages.
§9 is cycle 3f: BENCHMARKS.md merged into the README (open end-to-end,
collapsed rest, no second file).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mjolnir_cli.config import load_layout

import render_targets_fixtures as fx
from render_targets_fixtures import FP6_QUANT, M38, build_repo

REPO = Path(__file__).resolve().parents[2]
CONFIG_PAGES = [
    (M38, FP6_QUANT),
    (M38, "NVFP4"),
    (M38, "AWQ_FLASHINFER"),
    (fx.M36, "NVFP4_MJOLNIR_E2M3"),
    (fx.M36, "NVFP4_FA2"),
    (fx.M36, "NVFP4_FA4"),
]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    return build_repo(tmp_path, monkeypatch)


def _render_all() -> None:
    from mjolnir_cli.readme import render_all

    render_all(load_layout())


def _page(repo: Path, model: str, quant: str) -> str:
    return (repo / "configs" / model / quant / "README.md").read_text()


def _profile(page: str) -> tuple[list[str], dict[str, list[str]]]:
    """The '## Serving profile' table: (column headers, {row label: cells})."""
    body = re.split(r"(?m)^## Serving profile$", page)[1].split("\n## ")[0]
    header: list[str] = []
    rows: dict[str, list[str]] = {}
    for ln in body.splitlines():
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if set(cells[0]) <= {"-", " "}:
            continue  # the |---| separator
        if cells[0] == "parameter":
            header = cells[1:]
        else:
            rows[cells[0]] = cells[1:]
    return header, rows


# ── 1. the title (plan §3.1) ────────────────────────────────────────────────


def test_h1_is_the_model_and_the_profile_is_secondary(repo):
    _render_all()
    for model, quant in CONFIG_PAGES:
        page = _page(repo, model, quant)
        assert page.splitlines()[0] == f"# {model}", page.splitlines()[0]
        # cycle 3f re-pin: the merged Benchmarks section's e2e headings name
        # the pair's quants; everywhere else the quant is never a heading
        head, _bench = page.split("## Benchmarks", 1)
        headings = [ln for ln in head.splitlines() if ln.startswith("#")]
        assert all(quant not in ln for ln in headings), (quant, headings)
        assert page.splitlines()[2].startswith(f"`{quant}`")  # the profile, secondary


# ── 2. no dump tables (plan §3.2) ───────────────────────────────────────────


def test_no_key_value_dump_table_in_any_config_readme(repo):
    _render_all()
    for model, quant in CONFIG_PAGES:
        assert "| key | value |" not in _page(repo, model, quant), (model, quant)


def test_real_tree_config_readmes_carry_no_dump_fit_the_budget_and_head_with_the_model():
    """The ratchet (plan §4) on the repo's own committed pages. Budget
    amended cycle 3f: the page total is uncapped (collapsed sections), the
    above-the-fold — everything before the first ``<details>`` — is ≤ 55."""
    pages = sorted(REPO.glob("configs/*/*/*/README.md"))
    assert pages, "the repo's generated config pages are missing"
    for p in pages:
        text = p.read_text()
        assert "| key | value |" not in text, p
        assert len(text.split("<details>")[0].splitlines()) <= 55, p
        model = p.parent.parent.relative_to(REPO / "configs").as_posix()
        assert text.splitlines()[0] == f"# {model}", p


# ── 3. the curated rows, summarized (plan §3.3) ─────────────────────────────


def test_meta_rows_present_with_summarized_values(repo):
    _render_all()
    header, rows = _profile(_page(repo, M38, FP6_QUANT))
    assert header == ["Thor"]  # the registry short label
    assert rows["Checkpoint"] == ["`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`"]
    assert rows["Format"] == ["Mjolnir fp6 KV + fp8 lm_head + 64K draft vocab"]
    assert rows["Max context"] == ["256K"]
    assert rows["Max concurrency"] == ["8"]
    assert rows["Prefill chunk"] == ["16K"]
    assert rows["KV cache"] == ["fp6_e2m3 · skip: sliding_window"]
    assert rows["Attention backend"] == ["MJOLNIR"]
    assert rows["Spec decode"] == ["MTP ×5"]
    assert rows["GPU memory util"] == ["0.80"]
    assert rows["Parsing"] == ["qwen3_coder / qwen3"]
    assert "Served as" not in rows  # served-model-name == the H1 model → omitted


# ── 4. overlay deltas surface as columns (plan §3.4) ────────────────────────


def test_overlay_deltas_surface_as_columns(repo):
    _render_all()
    page = _page(repo, M38, "NVFP4")
    header, rows = _profile(page)
    assert header == ["Thor", "Spark"]  # one column per serving platform
    assert rows["Max context"] == ["200K", "128K"]  # the spark overlay's delta, as a column
    assert rows["GPU memory util"] == ["0.80", "0.85"]
    assert "spark.yaml" not in page  # the per-platform delta dump is gone


# ── 5. dict-valued keys never appear (plan §3.5) ────────────────────────────


def test_dict_valued_config_keys_never_appear(repo):
    _render_all()
    page = _page(repo, M38, FP6_QUANT)
    for key in (
        "compilation-config",
        "override-generation-config",
        "default-chat-template-kwargs",
        "chat-template",
    ):
        assert key not in page, key


# ── 6. no .env contents — the pointer only (plan §1, amended 2026-10-20) ────

_ENV_PAIR = re.compile(r"`(?:MJOLNIR|VLLM)_[A-Z_]+=")


def test_no_env_contents_on_any_config_readme(repo):
    """The `.env` compact line is DROPPED (user directive 2026-10-20): no
    KEY=VALUE contents on any generated config README — the intro keeps only
    the pointer to the sidecar file, and the sidecars themselves are
    untouched."""
    _render_all()
    for model, quant in CONFIG_PAGES:
        page = _page(repo, model, quant)
        assert not [ln for ln in page.splitlines() if ln.startswith(".env:")], (model, quant)
        assert not _ENV_PAIR.search(page), (model, quant)
    fp6 = _page(repo, M38, FP6_QUANT)  # the fixture WITH a .env sidecar
    assert "`.env`" in fp6  # the intro's pointer to the file stays


def test_real_tree_config_readmes_carry_no_env_contents():
    """The .env half of the ratchet (plan §4, amendment 2026-10-20) on the
    repo's own committed pages: no KEY=VALUE contents, and the intro's
    pointer appears exactly when the config has a sidecar."""
    pages = sorted(REPO.glob("configs/*/*/*/README.md"))
    assert pages, "the repo's generated config pages are missing"
    for p in pages:
        text = p.read_text()
        assert not [ln for ln in text.splitlines() if ln.startswith(".env:")], p
        assert not _ENV_PAIR.search(text), p
        assert ("`.env`" in text) == (p.parent / ".env").is_file(), p


# ── 7. the line budget (plan §3.7, amended cycle 3f: above-the-fold) ────────


def _above_fold(page: str) -> str:
    """Everything before the first ``<details>`` (cycle 3f §1)."""
    return page.split("<details>")[0]


def test_every_config_readme_above_fold_is_at_most_55_lines(repo):
    _render_all()
    for model, quant in CONFIG_PAGES:
        fold = _above_fold(_page(repo, model, quant))
        assert len(fold.splitlines()) <= 55, (model, quant, len(fold.splitlines()))


# ── 8. the M36 fixture: int labels, omitted rows, Served as, backend ────────


def test_m36_fixture_pins_int_context_omitted_rows_served_as_and_backend(repo):
    """The nvidia/Qwen3.6-35B-A3B fixture pages: a plain-int ``max-model-len``
    renders the repo's L label (262144 → 256K — the real Qwen3.6 production
    shape, review Flag 3), keys absent from every platform's config omit
    their row, a ``served-model-name`` ≠ the page model shows as ``Served
    as``, and the attention-backend row is the canonical
    ``overlays.backend_class`` derivation (the FA2 fixture's
    ``flash_attn_version: 2`` stays plain FLASH_ATTN — @4 is the only
    special case)."""
    _render_all()
    header, rows = _profile(_page(repo, fx.M36, "NVFP4_MJOLNIR_E2M3"))
    assert header == ["Thor"]
    assert rows["Max context"] == ["256K"]  # the int → L_label path, not "262144"
    for absent in ("Max concurrency", "Prefill chunk", "GPU memory util", "Spec decode", "Parsing"):
        assert absent not in rows, absent  # absent everywhere → row omitted
    assert rows["Served as"] == ["`nvidia/Qwen3.6-35B-A3B-NVFP4`"]
    _header, fa2_rows = _profile(_page(repo, fx.M36, "NVFP4_FA2"))
    assert fa2_rows["Attention backend"] == ["FLASH_ATTN"]


def test_fa4_fixture_pins_the_distinguishing_flash_attn_4_page_backend(repo):
    """The @4 half of the backend pin (review cycle-3e round 2, IMPROVEMENT 1):
    NVFP4_FA4 — NVFP4_FA2's sibling — carries ``attention-backend:
    FLASH_ATTN`` + ``flash_attn_version: 4``, the ONLY input on which the old
    inline derivation (plain ``FLASH_ATTN``) and the canonical
    :func:`mjolnir_cli.overlays.backend_class` (``FLASH_ATTN@4``) disagree.
    The page renders through the real builder, so reverting
    ``config_profile.backend()`` to the old inline two-liner fails here — the
    FA2 pin above cannot see that regression (the derivations agree on
    version 2)."""
    _render_all()
    _header, rows = _profile(_page(repo, fx.M36, "NVFP4_FA4"))
    assert _header == ["Thor"]  # legacy fixture config: thor-only, like FA2
    assert rows["Attention backend"] == ["FLASH_ATTN@4"]


# ── 9. cycle 3f: BENCHMARKS.md merged into the README (plan §1–§4) ──────────


def test_no_benchmarks_file_is_emitted_and_config_targets_are_single_page(repo):
    """Scenario 1: BENCHMARKS.md is gone — one README per config, and each
    Config target emits exactly one page (the single-target registry)."""
    from mjolnir_cli import readme_targets

    _render_all()
    assert not list(repo.glob("configs/**/BENCHMARKS.md"))
    layout = load_layout()
    cfgs = [t for t in readme_targets.discover_targets(layout) if isinstance(t, readme_targets.Config)]
    assert len(cfgs) == 6
    for t in cfgs:
        outs = t.outputs(repo, layout)
        assert len(outs) == 1 and outs[0][1].name == "README.md", [str(o[1]) for o in outs]


def test_e2e_pair_cell_renders_open_with_table_and_chart(repo):
    """Scenario 2 (LMH8-style pair data): the A/B table + the e2e-decode chart
    sit above the first ``<details>`` — open, not collapsed."""
    _render_all()
    fold = _above_fold(_page(repo, M38, FP6_QUANT))
    assert "### End to end (NVFP4 vs NVFP4_MJOLNIR_E2M3_LMH8_VT64K, 1 request · 8K context)" in fold
    assert "| metric | official `NVFP4` | Mjolnir `NVFP4_MJOLNIR_E2M3_LMH8_VT64K` | change |" in fold
    assert "+30.8%" in fold  # the fixture thor decode gain, in the open table
    assert "e2e-decode.png" in fold  # the open A/B chart


def test_e2e_solo_cell_renders_own_rows_without_a_delta_column(repo):
    """Scenario 3: own history rows, no official-image partner — the config's
    own metrics table, no change column, no chart, no how panel."""
    _render_all()
    page = _page(repo, fx.M36, "NVFP4_FA2")
    fold = _above_fold(page)
    assert "### End to end (NVFP4_FA2, 1 request · 8K context)" in fold
    assert "| metric | `NVFP4_FA2` |" in fold
    assert "44.0" in fold  # the solo record's decode t/s
    assert "| change |" not in fold and "e2e-decode" not in fold
    assert "How the % and × here are calculated" not in fold  # nothing to explain


def test_no_data_state_shows_the_one_liner_with_no_empty_details(repo):
    """Scenario 4: nothing anywhere → the honest one-liner open, and no empty
    ``<details>`` husks for the absent sections."""
    _render_all()
    page = _page(repo, M38, "AWQ_FLASHINFER")
    fold = _above_fold(page)
    assert "no data yet" in fold
    assert "`Qwen/Qwen3.8-27B` / `AWQ_FLASHINFER` on Jetson Orin so far." in fold
    for summary in (
        "Speculative decoding",
        "KV capacity",
        "Model quality (per suite)",
        "Kernel stages (case-study scope)",
        "Raw provenance",
    ):
        assert f"<summary>{summary}</summary>" not in page, summary


def test_non_e2e_sections_are_collapsed_with_their_charts_and_how_inside(repo):
    """Scenario 5: spec / KV / quality / kernel wrapped in
    ``<details><summary>…</summary>``; the how-panels and chart imgs sit
    inside their blocks."""
    _render_all()
    page = _page(repo, M38, FP6_QUANT)
    for summary in (
        "Speculative decoding",
        "KV capacity",
        "Model quality (per suite)",
        "Kernel stages (case-study scope)",
        "Raw provenance",
    ):
        assert f"<details>\n<summary>{summary}</summary>" in page, summary
    spec = page.split("<summary>Speculative decoding</summary>")[1].split("<summary>KV capacity</summary>")[0]
    assert "46.9" in spec and "How the % and × here are calculated" in spec
    kernel = page.split("<summary>Kernel stages (case-study scope)</summary>")[1].split(
        "<summary>Raw provenance</summary>"
    )[0]
    assert "1.26" in kernel and "kernel-speedup.png" in kernel
    assert "How the % and × here are calculated" in kernel


def test_intro_derives_from_raw_data_without_a_benchmarks_link(repo):
    """The intro names the raw data (the old BENCHMARKS.md provenance line)
    and links the model README — no BENCHMARKS link anywhere."""
    _render_all()
    for model, quant in CONFIG_PAGES:
        page = _page(repo, model, quant)
        assert "BENCHMARKS" not in page, (model, quant)
        assert "derives from the committed raw data" in page, (model, quant)
        assert "benchmarks/history.jsonl" in page and "benchmarks/raw/" in page, (model, quant)
        assert "](../README.md)" in page, (model, quant)


def test_real_tree_config_readmes_have_no_benchmarks_file_or_reference():
    """Scenario 6, tree half: zero BENCHMARKS.md under configs/ and zero
    BENCHMARKS strings in any committed config README."""
    assert not list(REPO.glob("configs/**/BENCHMARKS.md"))
    for p in sorted(REPO.glob("configs/*/*/*/README.md")):
        assert "BENCHMARKS" not in p.read_text(), p


def test_site_payload_and_manifest_carry_no_benchmarks_refs(repo):
    """Scenario 7: the site payloads + manifest never reference
    BENCHMARKS.md; the config deep-link stays the payload's quant hash."""
    from mjolnir_cli import interactive

    out = interactive.build(load_layout(), repo / "_site")
    for p in sorted(out.parent.rglob("*.json")):
        assert "BENCHMARKS" not in p.read_text(), p
    manifest = json.loads((out.parent / "manifest.json").read_text())
    assert {p["platform"] for p in manifest["pairs"]} == {"thor", "spark"}  # hash targets unchanged
