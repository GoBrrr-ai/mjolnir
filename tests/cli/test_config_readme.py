"""Cycle-3e RED: the config-README redesign (task/plan.md §1–§3).

The per-config ``README.md`` must *interpret* config.yaml, not mirror it:
the H1 is the model, the profile is secondary, ONE curated meta-table
carries a column per serving platform (overlay deltas surface as columns),
and the config.yaml / .env / delta dumps are gone. Serve lines, availability,
the links and BENCHMARKS.md are untouched.

Fixture: :mod:`render_targets_fixtures` — the FP6 config carries the full
curated key set (incl. dict-valued keys that must never appear); NVFP4
serves thor + spark with a max-model-len overlay delta, the columns case.
The last test is the repo-tree ratchet (plan §4) on the committed pages.
"""

from __future__ import annotations

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
        headings = [ln for ln in page.splitlines() if ln.startswith("#")]
        assert all(quant not in ln for ln in headings), (quant, headings)
        assert page.splitlines()[2].startswith(f"`{quant}`")  # the profile, secondary


# ── 2. no dump tables (plan §3.2) ───────────────────────────────────────────


def test_no_key_value_dump_table_in_any_config_readme(repo):
    _render_all()
    for model, quant in CONFIG_PAGES:
        assert "| key | value |" not in _page(repo, model, quant), (model, quant)


def test_real_tree_config_readmes_carry_no_dump_fit_the_budget_and_head_with_the_model():
    """The ratchet (plan §4) on the repo's own committed pages."""
    pages = sorted(REPO.glob("configs/*/*/*/README.md"))
    assert pages, "the repo's generated config pages are missing"
    for p in pages:
        text = p.read_text()
        assert "| key | value |" not in text, p
        assert len(text.splitlines()) <= 45, p
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


# ── 6. the .env compact line (plan §3.6) ────────────────────────────────────


def test_env_sidecar_becomes_one_compact_line(repo):
    _render_all()
    fp6 = _page(repo, M38, FP6_QUANT)
    lines = [ln for ln in fp6.splitlines() if ln.startswith(".env:")]
    assert lines == [".env: `MJOLNIR_FP8_LM_HEAD=1` · `MJOLNIR_DRAFT_VOCAB=65536`"]
    nvfp4 = _page(repo, M38, "NVFP4")  # no sidecar → no line
    assert [ln for ln in nvfp4.splitlines() if ln.startswith(".env:")] == []


# ── 7. the line budget (plan §3.7) ──────────────────────────────────────────


def test_every_config_readme_is_at_most_45_lines(repo):
    _render_all()
    for model, quant in CONFIG_PAGES:
        assert len(_page(repo, model, quant).splitlines()) <= 45, (model, quant)


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
