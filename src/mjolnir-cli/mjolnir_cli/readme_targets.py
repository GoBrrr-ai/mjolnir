"""Render targets (cycle 3b): what ``mjolnir plot --readme`` renders, and the
per-target contexts the page templates consume.

``discover_targets(layout)`` walks the registry, the config tree and the raw
data and yields one :class:`Target` per spec render target
(``task/interviewer/spec-docs-benchmarks.md`` §Render Targets 1–8):

* :class:`Root` — the root ``README.md`` (status matrix, measured-pair index,
  the headline pinned to (thor, the default model) — decision 3);
* :class:`PlatformsIndex` — ``docs/platforms/README.md``;
* :class:`Platform` — ``docs/platforms/<p>/README.md`` — cycle 3k: one row
  per model, the kernel index in its ``## Kernel`` section and the
  headline model's A/B in ``## End to end`` — plus the kernel chapters
  under ``kernel/`` when the platform has verify raws for its headline
  model (``chapters`` — spec #4; never for a research platform);
* :class:`Model` — ``configs/<vendor>/<model>/README.md``;
* :class:`Config` — ``configs/…/<quant>/README.md`` (cycle 3f: the old
  ``BENCHMARKS.md`` is folded into it as the Benchmarks section — the
  per-platform section builders live in :mod:`mjolnir_cli.config_bench`).

Every target renders through :meth:`Target.outputs` — the
``(template, output, context)`` triples ``readme.render_all`` writes — and
exposes :meth:`build_context`, the pinned per-target context signature.
Root + index + platform + model + config pages are always discovered
(sparseness shows as placeholders, never as missing pages); the kernel tree
needs raws.

Two data rules live here. :func:`pair_docs` — a model-less verify document
belongs to its platform's headline model (the repo's committed history is the
headline model's data; an explicit ``model`` field marks another model's run,
which never matches). And :func:`config_cell` — the availability cell for a
config on a platform, always read from :mod:`mjolnir_cli.availability` (the
one source of truth; renderers never duplicate it).

Stdlib + yaml only — a host-side data leaf (import purity is pinned in
``tests/cli/test_host_import.py``); no torch/vllm, no matplotlib, no jinja2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mjolnir_cli import availability
from mjolnir_cli.config import (
    BACKEND_LABELS,
    DEFAULT_MODEL,
    DEFAULT_QUANT,
    RepoLayout,
    config_attention_backend,
    config_field,
    model_of,
)
from mjolnir_cli.config_bench import (
    BenchContext,
    _decode_pair,
    _e2e_noise_pct,
    accuracy_pair_platform,
    bench_context,
)
from mjolnir_cli.config_profile import ProfileRow, _backend_label, _profile_rows
from mjolnir_cli.history import load_records
from mjolnir_cli.overlays import load_overlay, merge_config, overlay_platforms
from mjolnir_cli.platforms import DEFAULT_PLATFORM, REGISTRY, STATUS_ICON, platform_of
from mjolnir_cli.verify import results
from mjolnir_cli.verify.stages import STAGES

# The spec §Directory Layout chapter set (docs/templates/kernel/): one
# chapter per verify stage. Cycle 3k: the case-study README is gone — the
# index (headline findings, chart, chapter table) is the platform page's
# `## Kernel` section, built from the shared kernel/_index.md.j2 partial.
# Membership is what discovery pins; order is the reading order.
CASE_STUDY_CHAPTERS: tuple[str, ...] = (
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
STAGE_CHAPTERS = CASE_STUDY_CHAPTERS


def chapters_for(platform: str) -> tuple[str, ...]:
    """The chapters a platform's kernel tree can carry: the ncu-profiled
    stages drop on platforms without Nsight Compute (there they are N/A, not
    pending -- the events-timed stages carry the headline instead)."""
    if REGISTRY[platform].ncu:
        return CASE_STUDY_CHAPTERS
    return tuple(c for c in CASE_STUDY_CHAPTERS if STAGES[c].kind != "ncu")
# verify stages that never carry a case-study chapter
_UNCHARTED = ("timing", "profile", "vllm")

# availability symbols (the table's three states, as the pages render them)
SYMBOLS = {availability.NATIVE: "✅", availability.FALLBACK: "⚠️", availability.UNAVAILABLE: "❌"}

# quant-dir-name prefixes that encode a *weights* format (spec §Directory
# Layout: ``AWQ_FLASHINFER`` is int4 W4A16 weights). Everything else is read
# as kv-cache-dtype × backend class.
_WEIGHT_FORMATS: tuple[tuple[str, str], ...] = (
    ("AWQ_", "int4-w4a16"),
    ("GPTQ_", "int4-w4a16"),
    ("INT4_", "int4-w4a16"),
    ("INT8_", "int8-w8a8"),
)


def weights_format(quant: str) -> str | None:
    """The checkpoint format a config name encodes (None = read the KV dtype)."""
    return next((fmt for prefix, fmt in _WEIGHT_FORMATS if quant.startswith(prefix)), None)


def config_cell(config_dir: Path, quant: str, platform: str) -> availability.Availability:
    """The availability cell for serving ``quant`` on ``platform`` — always
    from :mod:`mjolnir_cli.availability` (AC: no duplicated table).

    A name-encoded weights format goes to :func:`availability.weights_state`;
    everything else through ``kv-cache-dtype`` × the config's backend class.
    ``auto`` KV means the unquantized baseline (bf16)."""
    fmt = weights_format(quant)
    if fmt is not None:
        return availability.weights_state(fmt, platform)
    dtype = config_field(config_dir / "config.yaml", "kv-cache-dtype") or "auto"
    if dtype == "auto":
        dtype = "bf16"
    backend = availability.backend_class(config_attention_backend(config_dir / "config.yaml"), platform)
    return availability.kv_state(dtype, backend, platform)


# ── the (platform, model) data scope ────────────────────────────────────────


def _matches(doc: dict, platform: str, model: str) -> bool:
    """Does a raw document belong to the (platform, model) pair? A model-less
    document belongs to the platform's headline model (the default model —
    the repo's committed data); an explicit other model never matches."""
    if platform_of(doc) != platform:
        return False
    m = model_of(doc)
    return m == model or (m is None and model == DEFAULT_MODEL)


def _newest(rs: list[dict]) -> dict | None:
    dflt = [r for r in rs if r.get("default")]
    return (dflt or rs)[-1] if rs else None


def _stage_doc(layout: RepoLayout, stage: str, platform: str, model: str) -> dict | None:
    return _newest([d for d in results.runs(layout.verify_dir, stage) if _matches(d, platform, model)])


def pair_docs(layout: RepoLayout, platform: str, model: str) -> dict[str, dict]:
    """The newest default run of every verify stage for one (platform, model)
    pair — the case-study data scope (spec #4) and the config pages'
    kernel-section scope (spec #7: platform-level, headline model)."""
    out: dict[str, dict] = {}
    for stage in STAGES:
        if stage in _UNCHARTED:
            continue
        doc = _stage_doc(layout, stage, platform, model)
        if doc:
            out[stage] = doc
    return out


def has_kernel_raws(layout: RepoLayout, platform: str, model: str = DEFAULT_MODEL) -> bool:
    """Does the platform have verify raws for its headline model? (spec #4's
    render gate — the kernel tree exists only then.)"""
    return any(_stage_doc(layout, stage, platform, model) for stage in STAGE_CHAPTERS)


def measured_pairs(layout: RepoLayout) -> list[tuple[str, str]]:
    """Every (platform, model) pair that has data — verify raws (a model-less
    doc is its platform's headline model's), history rows and accuracy legs —
    sorted, with the default pair last (it owns the flat chart ids)."""
    pairs: set[tuple[str, str]] = set()
    for stage in STAGES:
        if stage in _UNCHARTED:
            continue
        for doc in results.runs(layout.verify_dir, stage):
            pairs.add((platform_of(doc), model_of(doc) or DEFAULT_MODEL))
    for rec in load_records(layout.history_file):
        m = model_of(rec)
        if m and rec.get("cells"):
            pairs.add((platform_of(rec), m))
    from mjolnir_cli import accuracy

    by_model: dict[str, list[dict]] = {}
    for leg in accuracy.latest_legs(accuracy.load_runs(layout)):
        m = model_of(leg)
        if m:
            by_model.setdefault(m, []).append(leg)
    for model, legs in by_model.items():
        pairs.add((accuracy_pair_platform(legs), model))
    default = (DEFAULT_PLATFORM, DEFAULT_MODEL)
    return sorted(pairs, key=lambda p: (p == default, p))


# ── the config tree ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConfigDir:
    """One ``configs/<vendor>/<model>/<quant>/`` directory of the checkout."""

    model: str
    quant: str
    path: Path


def scan_configs(layout: RepoLayout) -> list[ConfigDir]:
    """The checkout's config tree (the pages document the repo's own configs;
    the wheel's bundled copies and user-local dirs are serving concerns)."""
    root = layout.repo_root
    if root is None:
        return []
    base = root / "configs"
    if not base.is_dir():
        return []
    return [
        ConfigDir(model=p.parent.parent.relative_to(base).as_posix(), quant=p.parent.name, path=p.parent)
        for p in sorted(base.glob("*/*/*/config.yaml"))
    ]


def _merged_config(config_dir: Path, platform: str) -> dict:
    """The base config.yaml deep-merged with the platform overlay (the base
    as-is when the overlay is empty or the config is unreadable)."""
    try:
        raw = yaml.safe_load((config_dir / "config.yaml").read_text()) or {}
    except OSError:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    overlay = load_overlay(config_dir, platform)
    return merge_config(raw, overlay) if overlay else raw


def _has_env_sidecar(config_dir: Path) -> bool:
    """Does the config dir carry a `.env` sidecar with KEY=VALUE lines?
    (The README points at the sidecar; it never quotes it.)"""
    side = config_dir / ".env"
    if not side.is_file():
        return False
    return any("=" in ln.split("#", 1)[0] for ln in side.read_text().splitlines())


def exists_on(config_dir: Path, platform: str) -> bool:
    """Does the config serve on the platform? (the overlay rule: explicit
    overlays when the dir has a ``platforms/``, else the legacy thor-only.)"""
    return platform in (overlay_platforms(config_dir) or (DEFAULT_PLATFORM,))


# ── shared context rows ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlatformRow:
    """One registry record as the pages render it."""

    key: str
    name: str
    short: str
    status: str
    sm_family: str
    host_requirements: str
    # Renderer-facing data (cycle 3g): the status icon of the root README's
    # hardware table (platforms.STATUS_ICON) — icon + word in the status cell.
    status_icon: str = ""


def platform_rows() -> list[PlatformRow]:
    return [
        PlatformRow(
            key=p.key,
            name=p.name,
            short=p.short or p.name,
            status=p.status,
            sm_family=p.sm_family,
            host_requirements=p.host_requirements,
            status_icon=STATUS_ICON[p.status],
        )
        for p in REGISTRY.values()
    ]


@dataclass(frozen=True)
class Cell:
    """One availability cell as a page renders it: the symbol plus the why —
    both straight from :mod:`mjolnir_cli.availability`."""

    symbol: str
    why: str


def _cell_of(config_dir: Path, quant: str, platform: str) -> Cell:
    a = config_cell(config_dir, quant, platform)
    return Cell(SYMBOLS[a.state], a.why)


# ── the targets ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Root:
    """The root ``README.md`` — status matrix, measured-pair index, headline."""

    def build_context(self, layout: RepoLayout) -> Any:
        from mjolnir_cli import readme  # lazy: readme's loop imports this module

        return readme.collect_context(layout)

    def outputs(self, root: Path, layout: RepoLayout) -> list[tuple[Path, Path, Any]]:
        ctx = self.build_context(layout)
        return [(root / "README.md.j2", root / "README.md", ctx)]


@dataclass(frozen=True)
class PlatformsIndex:
    """``docs/platforms/README.md`` — the platform list + status (spec #2)."""

    def build_context(self, layout: RepoLayout) -> PlatformsIndexContext:
        return PlatformsIndexContext(platforms=platform_rows())

    def outputs(self, root: Path, layout: RepoLayout) -> list[tuple[Path, Path, Any]]:
        tpl = root / "docs" / "templates" / "pages" / "PLATFORMS_INDEX.md.j2"
        return [(tpl, root / "docs" / "platforms" / "README.md", self.build_context(layout))]


@dataclass(frozen=True)
class Platform:
    """``docs/platforms/<p>/README.md`` + the platform's kernel case study
    (spec #3/#4): the chapter set follows the platform's raws. Cycle 3k: the
    case-study README is gone — the index lives on this page's ``## Kernel``
    section, and the config matrix became one row per model."""

    key: str
    chapters: tuple[str, ...] = ()

    def build_context(self, layout: RepoLayout) -> PlatformContext:
        rec = REGISTRY[self.key]
        model = rec.headline_model or DEFAULT_MODEL
        rows = []
        history = load_records(layout.history_file) if layout.history_file.exists() else []
        by_model: dict[str, list[ConfigDir]] = {}
        for c in scan_configs(layout):
            by_model.setdefault(c.model, []).append(c)
        for m, dirs in by_model.items():
            # the default config rule (cycle 3k §1): no registry record
            # carries a per-platform default config yet, so it is the model
            # page's quickstart rule — DEFAULT_QUANT when the platform's
            # overlays allow it, else the first config they allow (scan order).
            serving = [c for c in dirs if exists_on(c.path, self.key)]
            dflt = next((c for c in serving if c.quant == DEFAULT_QUANT), serving[0] if serving else None)
            measured = any(platform_of(r) == self.key and model_of(r) == m for r in history)
            rows.append(PlatformModel(model=m, default_config=dflt.quant if dflt else "", measured=measured))
        overview = Path("")
        if layout.repo_root:
            overview = layout.repo_root / "docs" / "platforms" / self.key / "overview.md"
        kernel = None
        if self.chapters:
            from mjolnir_cli import readme  # lazy: the pair-scoped kernel context

            kernel = readme.collect_context(layout, platform=self.key, model=model)
        e2e = _ab_row(layout, self.key, model, rec.name) if rec.headline_model else None
        return PlatformContext(
            key=rec.key,
            name=rec.name,
            status=rec.status,
            sm_family=rec.sm_family,
            host_requirements=rec.host_requirements,
            rows=rows,
            overview=overview.read_text() if overview.is_file() else "",
            chapters=self.chapters,
            kernel=kernel,
            e2e=e2e,
            chart_pair=(self.key, model) if (kernel or e2e) else None,
        )

    def outputs(self, root: Path, layout: RepoLayout) -> list[tuple[Path, Path, Any]]:
        ctx = self.build_context(layout)
        pages = root / "docs" / "templates" / "pages"
        out = [(pages / "PLATFORM_README.md.j2", root / "docs" / "platforms" / self.key / "README.md", ctx)]
        if ctx.kernel is None:
            return out
        tpls = root / "docs" / "templates" / "kernel"
        kdir = root / "docs" / "platforms" / self.key / "kernel"
        out += [(tpls / f"{c}.md.j2", kdir / f"{c}.md", ctx.kernel) for c in self.chapters]
        # chapters this platform cannot carry must not linger from an older render
        if kdir.is_dir():
            wanted = {f"{c}.md" for c in self.chapters} | {"accuracy.md"}
            for stale in kdir.glob("*.md"):
                if stale.name not in wanted:
                    stale.unlink()
        # the accuracy chapter renders with the case study (its "Pending" note
        # is the no-data state; test_accuracy_page_pending_without_runs pins it)
        out.append((tpls / "accuracy.md.j2", kdir / "accuracy.md", ctx.kernel))
        return out


@dataclass(frozen=True)
class Model:
    """``configs/<vendor>/<model>/README.md`` (spec #5)."""

    model: str

    def build_context(self, layout: RepoLayout) -> ModelContext:
        dirs = [c for c in scan_configs(layout) if c.model == self.model]
        rows = []
        for c in dirs:
            cells = {
                p.key: _cell_of(c.path, c.quant, p.key) if exists_on(c.path, p.key) else Cell("—", "")
                for p in platform_rows()
            }
            rows.append(ModelConfigRow(quant=c.quant, label=BACKEND_LABELS.get(c.quant, c.quant), cells=cells))
        per_platform = []
        for p in platform_rows():
            existing = [c for c in dirs if exists_on(c.path, p.key)]
            if not existing:
                continue
            prefer = next((c for c in existing if c.quant == DEFAULT_QUANT), existing[0])
            per_platform.append(
                PlatformModelRow(
                    key=p.key, name=p.name, quickstart=prefer.quant, ab=_ab_row(layout, p.key, self.model, p.name)
                )
            )
        served = (config_field(dirs[0].path / "config.yaml", "served-model-name") or self.model) if dirs else self.model
        return ModelContext(
            model=self.model,
            short=self.model.split("/")[-1],
            served=served,
            rows=rows,
            platforms=per_platform,
        )

    def outputs(self, root: Path, layout: RepoLayout) -> list[tuple[Path, Path, Any]]:
        tpl = root / "docs" / "templates" / "pages" / "MODEL_README.md.j2"
        return [(tpl, root / "configs" / self.model / "README.md", self.build_context(layout))]


@dataclass(frozen=True)
class Config:
    """``configs/…/<quant>/README.md`` (spec #6) — cycle 3f: the old
    ``BENCHMARKS.md`` is folded in as the page's Benchmarks section; the
    per-platform data comes from :mod:`mjolnir_cli.config_bench`."""

    model: str
    quant: str

    def build_context(self, layout: RepoLayout) -> ConfigContext:
        """The config README (cycles 3e+3f): the model is the page and the
        profile secondary; ONE curated meta-table with a column per serving
        platform — overlay deltas surface as columns (plan §1) — and the
        merged Benchmarks section (open end-to-end, collapsed rest). No
        config.yaml / .env / delta dumps."""
        c = self._dir(layout)
        plats = [p for p in platform_rows() if exists_on(c.path, p.key)]
        merged = [_merged_config(c.path, p.key) for p in plats]
        return ConfigContext(
            model=self.model,
            quant=self.quant,
            label=_backend_label(self.quant, merged),
            profile=_profile_rows(self.model, self.quant, merged),
            has_overlays=bool(overlay_platforms(c.path)),
            has_env=_has_env_sidecar(c.path),
            platforms=[
                ConfigPlatform(
                    name=p.name,
                    short=p.short,
                    argv=f"mjolnir serve up --config {self.quant}",
                    cell=_cell_of(c.path, self.quant, p.key),
                )
                for p in plats
            ],
            bench=bench_context(layout, self.model, self.quant, plats),
            e2e_noise_pct=_e2e_noise_pct(),
        )

    def outputs(self, root: Path, layout: RepoLayout) -> list[tuple[Path, Path, Any]]:
        pages = root / "docs" / "templates" / "pages"
        out_dir = root / "configs" / self.model / self.quant
        return [(pages / "CONFIG_README.md.j2", out_dir / "README.md", self.build_context(layout))]

    def _dir(self, layout: RepoLayout) -> ConfigDir:
        return next(c for c in scan_configs(layout) if c.model == self.model and c.quant == self.quant)


def discover_targets(layout: RepoLayout) -> list[Root | PlatformsIndex | Platform | Model | Config]:
    """Every render target of the checkout, in render order. Root + index +
    one Platform per registry record (with ``chapters`` when its headline
    model has verify raws) + one Model per config-tree model + one Config per
    config dir (one page each — the merged Benchmarks section rides it)."""
    targets: list[Root | PlatformsIndex | Platform | Model | Config] = [Root(), PlatformsIndex()]
    targets += [
        Platform(
            key=p.key,
            chapters=chapters_for(p.key) if has_kernel_raws(layout, p.key, p.headline_model or DEFAULT_MODEL) else (),
        )
        for p in REGISTRY.values()
    ]
    models = sorted({c.model for c in scan_configs(layout)})
    targets += [Model(model=m) for m in models]
    targets += [Config(model=c.model, quant=c.quant) for c in scan_configs(layout)]
    return targets


# ── the page contexts (templates read data, never compute) ──────────────────


@dataclass(frozen=True)
class PlatformsIndexContext:
    platforms: list[PlatformRow]


@dataclass(frozen=True)
class PlatformModel:
    """One row of a platform page's models table (cycle 3k §1): the model,
    its default config on the platform and whether the (platform, model)
    pair has end-to-end data. The default-config rule lives on
    :meth:`Platform.build_context`."""

    model: str
    default_config: str
    measured: bool


@dataclass(frozen=True)
class PlatformContext:
    key: str
    name: str
    status: str
    sm_family: str
    host_requirements: str
    rows: list[PlatformModel]
    overview: str
    chapters: tuple[str, ...]
    kernel: Any | None  # the pair-scoped ReadmeContext when the case study renders
    # cycle 3k §1: the headline model's A/B row (None without a pinned
    # headline model or without pair data) and the pair the page's charts
    # resolve for (None when neither section renders)
    e2e: ABRow | None = None
    chart_pair: tuple[str, str] | None = None


@dataclass(frozen=True)
class ModelConfigRow:
    quant: str
    label: str
    cells: dict[str, Cell]  # platform key -> cell ("—" Cell when not on it)


@dataclass(frozen=True)
class ABRow:
    """One platform's end-to-end A/B (official vLLM image vs Mjolnir)."""

    platform: str
    official: str
    mjolnir: str
    official_decode: str
    mjolnir_decode: str
    change: str
    ctx: str


@dataclass(frozen=True)
class PlatformModelRow:
    key: str
    name: str
    quickstart: str
    ab: ABRow | None


@dataclass(frozen=True)
class ModelContext:
    model: str
    short: str
    served: str
    rows: list[ModelConfigRow]
    platforms: list[PlatformModelRow]
    e2e_noise_pct: str = "3"


@dataclass(frozen=True)
class ConfigPlatform:
    """One serving platform of a config README: the serve line, the table's
    column label and the availability cell."""

    name: str
    short: str
    argv: str
    cell: Cell


@dataclass(frozen=True)
class ConfigContext:
    """The config README's context (cycles 3e+3f): the model is the H1, the
    profile secondary; ``profile`` is the curated meta-table (one cell per
    platform); ``has_env`` gates the intro's pointer to the `.env` sidecar
    (whose contents are never rendered); ``bench`` is the merged Benchmarks
    section (:mod:`mjolnir_cli.config_bench`)."""

    model: str
    quant: str
    label: str
    profile: list[ProfileRow]
    has_overlays: bool
    has_env: bool
    platforms: list[ConfigPlatform]
    bench: BenchContext
    e2e_noise_pct: str = "3"


def _ab_row(layout: RepoLayout, platform: str, model: str, name: str) -> ABRow | None:
    """One platform's e2e A/B row (``latest_pair(platform, model)``): None
    when the pair has no data."""
    from mjolnir_cli import e2e

    off, mj = e2e.latest_pair(layout.history_file, model=model, platform=platform)
    if not off or not mj:
        return None
    pair = _decode_pair(off, mj)
    if pair is None:
        return None
    ctx, a, b = pair
    return ABRow(
        platform=name,
        official=str(off.get("config", "")),
        mjolnir=str(mj.get("config", "")),
        official_decode=f"{a:.1f}",
        mjolnir_decode=f"{b:.1f}",
        change=e2e.pct(e2e.change_pct(a, b)),
        ctx=ctx,
    )
