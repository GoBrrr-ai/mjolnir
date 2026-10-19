"""Render targets (cycle 3b): what ``mjolnir plot --readme`` renders, and the
per-target contexts the page templates consume.

``discover_targets(layout)`` walks the registry, the config tree and the raw
data and yields one :class:`Target` per spec render target
(``task/interviewer/spec-docs-benchmarks.md`` §Render Targets 1–8):

* :class:`Root` — the root ``README.md`` (status matrix, measured-pair index,
  the headline pinned to (thor, the default model) — decision 3);
* :class:`PlatformsIndex` — ``docs/platforms/README.md``;
* :class:`Platform` — ``docs/platforms/<p>/README.md`` plus, when the platform
  has verify raws for its headline model, its kernel case study under
  ``kernel/`` (``chapters`` — spec #4; never for a research platform);
* :class:`Model` — ``configs/<vendor>/<model>/README.md``;
* :class:`Config` — ``configs/…/<quant>/README.md`` **and**
  ``BENCHMARKS.md`` (``include_benchmarks`` is always True: both files are
  always generated, with placeholders where a cell has no data — spec #6/#7,
  decision 13).

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
from mjolnir_cli.history import load_records
from mjolnir_cli.overlays import load_overlay, merge_config, overlay_platforms
from mjolnir_cli.platforms import DEFAULT_PLATFORM, REGISTRY, platform_of
from mjolnir_cli.verify import results
from mjolnir_cli.verify.stages import STAGES

# The spec §Directory Layout chapter set (docs/templates/kernel/): the case
# study's README plus one chapter per verify stage. Membership is what
# discovery pins; order is the reading order.
CASE_STUDY_CHAPTERS: tuple[str, ...] = (
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
STAGE_CHAPTERS = tuple(c for c in CASE_STUDY_CHAPTERS if c != "README")
# verify stages that never carry a case-study chapter
_UNCHARTED = ("timing", "profile", "vllm")
# chart ids per stage (the read side of charts.KERNEL_CHARTS)
STAGE_CHARTS: dict[str, tuple[str, ...]] = {
    "sweep": ("kernel-speedup", "kernel-latency", "kernel-bandwidth"),
    "hndsweep": ("hnd-layout",),
    "kvfmt": ("kvfmt-heatmap", "kvfmt-tradeoff"),
    "quant": ("quant-accuracy",),
    "ring": ("ring-heatmap",),
    "prefetch": ("prefetch-heatmap",),
    "ab": ("ab-heatmap",),
    "pdl": ("pdl-delta",),
    "fp6prof": ("fp6prof-bytes-vs-time",),
    "kv-cache-size": ("kv-cache-capacity",),
    "spec": ("spec-drafts",),
}
# the "how is this calculated" panel kind per stage (None: no % or × in its findings)
_STAGE_HOW: dict[str, str | None] = {
    "sweep": "kernel",
    "hndsweep": "kernel",
    "kvfmt": "variant",
    "ring": "variant",
    "prefetch": "variant",
    "ab": "variant",
    "fp6prof": "fp6prof",
    "pdl": "pdl",
    "kv-cache-size": "kvcache",
    "spec": "spec",
    "quant": "kernel",
    "correctness": None,
}

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
    pair — the case-study data scope (spec #4) and the BENCHMARKS kernel
    scope (spec #7: platform-level, headline model)."""
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


def leg_platform(leg: dict) -> str:
    """The platform an accuracy leg ran on: its image's registry namespace."""
    image = str(leg.get("image", ""))
    return next((p.key for p in REGISTRY.values() if image.startswith(p.image_repo)), DEFAULT_PLATFORM)


def accuracy_pair_platform(rows: list[dict]) -> str:
    """One accuracy row set's platform: its Mjolnir-image legs' platform (a
    run serves its configs on one platform), else the default."""
    repos = tuple(p.image_repo for p in REGISTRY.values())
    mine = [leg_platform(r) for r in rows if str(r.get("image", "")).startswith(repos)]
    return mine[0] if mine else DEFAULT_PLATFORM


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


def _scalars(data: dict) -> list[tuple[str, str]]:
    """A config mapping as (key, rendered value) rows — nested structures as
    flow yaml so a delta table can quote them."""
    return [
        (
            str(k),
            yaml.safe_dump(v, default_flow_style=True).strip()
            if isinstance(v, dict | list)
            else ("" if v is None else str(v)),
        )
        for k, v in data.items()
    ]


def _config_scalars(path: Path) -> list[tuple[str, str]]:
    """The config yaml's top-level entries."""
    try:
        data = yaml.safe_load(path.read_text())
    except Exception:  # noqa: BLE001 — a malformed config renders as empty
        return []
    return _scalars(data) if isinstance(data, dict) else []


def _env_sidecar(config_dir: Path) -> dict[str, str]:
    """The ``.env`` sidecar's KEY=VALUE lines (comments stripped)."""
    side = config_dir / ".env"
    if not side.is_file():
        return {}
    out: dict[str, str] = {}
    for line in side.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _overlay_deltas(config_dir: Path, platform: str) -> list[tuple[str, str, str]]:
    """(key, value, contributing file) for every value the platform overlay
    changes against the base config.yaml (spec row 6: which file contributed
    each delta)."""
    raw = yaml.safe_load((config_dir / "config.yaml").read_text()) or {}
    overlay = load_overlay(config_dir, platform) or {}
    merged = _scalars(merge_config(raw, overlay) if overlay else raw)
    base = dict(_scalars(raw))
    return [(k, v, f"platforms/{platform}.yaml") for k, v in merged if base.get(k) != v]


def exists_on(config_dir: Path, platform: str) -> bool:
    """Does the config serve on the platform? (the overlay rule: explicit
    overlays when the dir has a ``platforms/``, else the legacy thor-only.)"""
    return platform in (overlay_platforms(config_dir) or (DEFAULT_PLATFORM,))


def matrix_on(config_dir: Path, platform: str) -> bool:
    """The platform pages' matrix rule: a config with explicit overlays
    appears only on its listed platforms (an em-dash elsewhere); a legacy
    config — no ``platforms/`` dir, the pre-overlay layout — is scanned on
    every platform and the availability cell says whether it can serve
    there (the fp6 scheme is ❌ everywhere but thor)."""
    plats = overlay_platforms(config_dir)
    return platform in plats if plats else True


# ── shared context rows ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlatformRow:
    """One registry record as the pages render it."""

    key: str
    name: str
    status: str
    sm_family: str
    host_requirements: str


def platform_rows() -> list[PlatformRow]:
    return [
        PlatformRow(
            key=p.key, name=p.name, status=p.status, sm_family=p.sm_family, host_requirements=p.host_requirements
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
    (spec #3/#4): the chapter set follows the platform's raws."""

    key: str
    chapters: tuple[str, ...] = ()

    def build_context(self, layout: RepoLayout) -> PlatformContext:
        rec = REGISTRY[self.key]
        rows = []
        history = load_records(layout.history_file) if layout.history_file.exists() else []
        for c in scan_configs(layout):
            on = matrix_on(c.path, self.key)
            measured = any(
                platform_of(r) == self.key and model_of(r) == c.model and r.get("config") == c.quant for r in history
            )
            rows.append(
                MatrixRow(
                    model=c.model,
                    quant=c.quant,
                    exists=on,
                    cell=_cell_of(c.path, c.quant, self.key) if on else None,
                    measured=measured,
                )
            )
        overview = Path("")
        if layout.repo_root:
            overview = layout.repo_root / "docs" / "platforms" / self.key / "overview.md"
        kernel = None
        if self.chapters:
            from mjolnir_cli import readme  # lazy: the pair-scoped kernel context

            kernel = readme.collect_context(layout, platform=self.key, model=DEFAULT_MODEL)
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
    """``configs/…/<quant>/README.md`` + ``BENCHMARKS.md`` (spec #6/#7) —
    both always generated (``include_benchmarks`` pinned True)."""

    model: str
    quant: str
    include_benchmarks: bool = True

    def build_context(self, layout: RepoLayout) -> ConfigContext:
        c = self._dir(layout)
        sections = []
        for p in platform_rows():
            if not exists_on(c.path, p.key):
                continue
            sections.append(
                ConfigPlatform(
                    key=p.key,
                    name=p.name,
                    deltas=_overlay_deltas(c.path, p.key),
                    argv=f"mjolnir serve up --platform {p.key} --config {self.quant}",
                    cell=_cell_of(c.path, self.quant, p.key),
                )
            )
        return ConfigContext(
            model=self.model,
            quant=self.quant,
            fields=_config_scalars(c.path / "config.yaml"),
            env=_env_sidecar(c.path),
            platforms=sections,
            e2e_noise_pct=_e2e_noise_pct(),
        )

    def build_benchmarks(self, layout: RepoLayout) -> BenchmarksContext:
        """The BENCHMARKS.md context: one section per platform the config
        exists on, each with the (platform, model, config) cell — or the
        single "no data yet" placeholder — plus the platform-level kernel
        stages (the headline model's raws, spec #7)."""
        c = self._dir(layout)
        sections = [
            _bench_section(p, self.model, self.quant, layout) for p in platform_rows() if exists_on(c.path, p.key)
        ]
        return BenchmarksContext(model=self.model, quant=self.quant, sections=sections, e2e_noise_pct=_e2e_noise_pct())

    def outputs(self, root: Path, layout: RepoLayout) -> list[tuple[Path, Path, Any]]:
        pages = root / "docs" / "templates" / "pages"
        out_dir = root / "configs" / self.model / self.quant
        return [
            (pages / "CONFIG_README.md.j2", out_dir / "README.md", self.build_context(layout)),
            (pages / "CONFIG_BENCHMARKS.md.j2", out_dir / "BENCHMARKS.md", self.build_benchmarks(layout)),
        ]

    def _dir(self, layout: RepoLayout) -> ConfigDir:
        return next(c for c in scan_configs(layout) if c.model == self.model and c.quant == self.quant)


def discover_targets(layout: RepoLayout) -> list[Root | PlatformsIndex | Platform | Model | Config]:
    """Every render target of the checkout, in render order. Root + index +
    one Platform per registry record (with ``chapters`` when its headline
    model has verify raws) + one Model per config-tree model + one Config per
    config dir (``include_benchmarks`` always True)."""
    targets: list[Root | PlatformsIndex | Platform | Model | Config] = [Root(), PlatformsIndex()]
    targets += [
        Platform(key=p.key, chapters=CASE_STUDY_CHAPTERS if has_kernel_raws(layout, p.key) else ())
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
class MatrixRow:
    """One config row of a platform page's matrix: on the platform ⇒ the
    availability cell; not on it ⇒ the em-dash."""

    model: str
    quant: str
    exists: bool
    cell: Cell | None
    measured: bool


@dataclass(frozen=True)
class PlatformContext:
    key: str
    name: str
    status: str
    sm_family: str
    host_requirements: str
    rows: list[MatrixRow]
    overview: str
    chapters: tuple[str, ...]
    kernel: Any | None  # the pair-scoped ReadmeContext when the case study renders


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
    key: str
    name: str
    deltas: list[tuple[str, str, str]]
    argv: str
    cell: Cell


@dataclass(frozen=True)
class ConfigContext:
    model: str
    quant: str
    fields: list[tuple[str, str]]
    env: dict[str, str]
    platforms: list[ConfigPlatform]
    e2e_noise_pct: str = "3"


@dataclass(frozen=True)
class PlatformBench:
    """One platform section of a BENCHMARKS.md: the (platform, model, config)
    cell — or the placeholder — plus the platform-level kernel stages."""

    key: str
    name: str
    headline: str
    has_data: bool
    e2e: dict | None
    spec: str | None
    kv: str | None
    accuracy: list[tuple[str, str]]
    stages: dict[str, dict]
    chart_pair: tuple[str, str] | None


@dataclass(frozen=True)
class BenchmarksContext:
    model: str
    quant: str
    sections: list[PlatformBench]
    e2e_noise_pct: str = "3"


# ── cell builders (spec #7: every number from raw data) ─────────────────────


def _e2e_noise_pct() -> str:
    from mjolnir_cli.e2e import E2E_NOISE_PCT

    return f"{E2E_NOISE_PCT:.0f}"


def _decode_pair(off: dict, mj: dict) -> tuple[str, float, float] | None:
    """(context label, official, mjolnir) decode t/s at 1 request, at the
    longest shared context — the A/B row's numbers."""
    from mjolnir_cli import e2e

    best = None
    for conc, ctx in e2e.shared_cells(off, mj):
        if conc != 1:
            continue
        a = e2e.value(e2e.cell(off, 1, ctx), "tg_tps")
        b = e2e.value(e2e.cell(mj, 1, ctx), "tg_tps")
        if a and b:
            best = (ctx, a, b)  # shared_cells ascends: the last hit is the longest
    return None if best is None else (e2e.ctx_label(best[0]), best[1], best[2])


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


def _bench_section(p: PlatformRow, model: str, quant: str, layout: RepoLayout) -> PlatformBench:
    from mjolnir_cli import readme  # the findings live beside the tables

    docs = pair_docs(layout, p.key, DEFAULT_MODEL) if has_kernel_raws(layout, p.key) else {}
    stages = {
        s: {
            "title": STAGES[s].title,
            "findings": readme._findings(s, docs[s]),
            "how": _STAGE_HOW.get(s),
            "charts": STAGE_CHARTS.get(s, ()),
            "run": docs[s].get("run_id", ""),
        }
        for s in STAGE_CHARTS
        if s in docs
    }
    e2e_cell = _e2e_cell(layout, p.key, model, quant)
    spec = _spec_cell(docs, quant)
    kv = _kv_cell(docs, quant)
    acc = _accuracy_cell(layout, p.key, model, quant)
    has_data = e2e_cell is not None or spec is not None or kv is not None or bool(acc)
    return PlatformBench(
        key=p.key,
        name=p.name,
        headline=DEFAULT_MODEL,
        has_data=has_data,
        e2e=e2e_cell,
        spec=spec,
        kv=kv,
        accuracy=acc,
        stages=stages,
        chart_pair=(p.key, DEFAULT_MODEL) if stages else None,
    )


def _e2e_cell(layout: RepoLayout, platform: str, model: str, quant: str) -> dict | None:
    """The (platform, model, config) end-to-end cell: the pair's A/B when the
    config is one of its sides, else the config's own records; None when the
    cell has no data at all."""
    from mjolnir_cli import e2e

    off, mj = e2e.latest_pair(layout.history_file, model=model, platform=platform)
    if off and mj and quant in (str(off.get("config", "")), str(mj.get("config", ""))):
        pass  # this config is a side of the pair: the A/B is its cell
    else:
        recs = [
            r
            for r in load_records(layout.history_file)
            if platform_of(r) == platform and model_of(r) == model and r.get("config") == quant
        ]
        off = e2e._merge([r for r in recs if e2e.is_official(r)])
        mj = e2e._merge([r for r in recs if e2e.is_mjolnir(r)])
        if not off and not mj:
            return None
    if not off or not mj:
        return {
            "ctx": "",
            "official": str(off.get("config", "")) if off else "",
            "mjolnir": str(mj.get("config", "")) if mj else "",
            "rows": [],
        }
    pair = _decode_pair(off, mj)
    if pair is None:
        return {"ctx": "", "official": str(off.get("config", "")), "mjolnir": str(mj.get("config", "")), "rows": []}
    ctx, _a, _b = pair
    conc = 1
    at = max((x for c, x in e2e.shared_cells(off, mj) if c == conc), default=0)
    rows = []
    for m in e2e.METRICS:
        va, vb = e2e.value(e2e.cell(off, conc, at), m.key), e2e.value(e2e.cell(mj, conc, at), m.key)
        rows.append(
            {
                "metric": m.label,
                "unit": m.unit,
                "official": "—" if va is None else m.fmt.format(va),
                "mjolnir": "—" if vb is None else m.fmt.format(vb),
                "change": "—" if va is None or vb is None else e2e.pct(e2e.change_pct(va, vb)),
            }
        )
    return {"ctx": ctx, "official": str(off.get("config", "")), "mjolnir": str(mj.get("config", "")), "rows": rows}


def _spec_cell(docs: dict[str, dict], quant: str) -> str | None:
    """This config's best speculative-decoding point from the pair's spec raw."""
    doc = docs.get("spec")
    if not doc:
        return None
    from mjolnir_cli.readme_tables import spec_label
    from mjolnir_cli.verify.spec import best_k

    rows = [r for r in doc.get("rows", []) if r.get("decode_tps") and r.get("config") == quant]
    parts = [
        f"{spec_label(cfg)}, {conc} request{'s' if conc > 1 else ''}: best with **{b['k']} draft"
        f"{'s' if b['k'] > 1 else ''}** — {b['decode_tps']:.1f} t/s per request, "
        f"{b['acceptance_length']:.2f} tokens per step."
        for (cfg, conc), b in sorted(best_k(rows).items())
    ]
    return " ".join(parts) or None


def _kv_cell(docs: dict[str, dict], quant: str) -> str | None:
    """This config's KV-capacity row from the pair's kv-cache-size raw."""
    doc = docs.get("kv-cache-size")
    if not doc:
        return None
    from mjolnir_cli.charts_base import L_label
    from mjolnir_cli.readme_tables import kvcache_label

    out = []
    for r in doc.get("rows", []):
        if not r.get("kv_tokens") or r.get("config") != quant:
            continue
        line = f"**{kvcache_label(r)}**: **{r['kv_tokens']:,} tokens**"
        if r.get("max_concurrency"):
            line += (
                f", {r['max_concurrency']:.1f} requests at the full {L_label(r.get('max_model_len') or 0)} context"
                " (vLLM's own estimate)."
            )
        out.append(line)
    return " ".join(out) or None


def _accuracy_cell(layout: RepoLayout, platform: str, model: str, quant: str) -> list[tuple[str, str]]:
    """This (platform, model, config) cell's accuracy suite scores: (suite
    label, score cell) — a leg measured on another platform never shows here."""
    from mjolnir_cli import accuracy

    rows = [
        r
        for r in accuracy.latest_legs(accuracy.load_runs(layout))
        if leg_platform(r) == platform and model_of(r) == model and r.get("config") == quant
    ]
    if not rows:
        return []
    base = accuracy.baseline(rows)
    out: list[tuple[str, str]] = []
    for r in rows:
        for n in accuracy.suites_in(rows):
            v = (r["suites"].get(n) or {}).get("score")
            if v is None:
                continue
            out.append((accuracy.suite_label(n), f"{v:.1f}{accuracy.fmt_delta(accuracy.delta(r, base, n))}"))
    return out
