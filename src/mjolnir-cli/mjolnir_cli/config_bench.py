"""The config README's Benchmarks section (cycle 3f): the per-platform
section builders behind :attr:`mjolnir_cli.readme_targets.ConfigContext.bench`
— BENCHMARKS.md folded into the config README, one page per config.

:func:`bench_context` builds the whole section: one :class:`PlatformBench`
per serving platform (the same per-platform builders the old BENCHMARKS.md
used, verbatim) plus the honest no-data one-liner for a config with no data
at all. The end-to-end cell (:func:`_e2e_cell`) carries the open state: the
pair's A/B when the config is one of its sides, else the config's own rows
(no partner to compare against), else nothing.

Host-side data leaf like its only consumer: stdlib + the repo's own data
modules, no torch/vllm, no matplotlib, no jinja2 (import purity is pinned in
``tests/cli/test_host_import.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mjolnir_cli.config import DEFAULT_MODEL, RepoLayout, model_of
from mjolnir_cli.history import load_records
from mjolnir_cli.platforms import DEFAULT_PLATFORM, REGISTRY, platform_of

if TYPE_CHECKING:
    from mjolnir_cli.readme_targets import PlatformRow


def _e2e_noise_pct() -> str:
    from mjolnir_cli.e2e import E2E_NOISE_PCT

    return f"{E2E_NOISE_PCT:.0f}"


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


@dataclass(frozen=True)
class PlatformBench:
    """One platform's benchmarks data for the merged config README: the
    (platform, model, config) cell — or the placeholder — plus whether the
    platform has kernel raws (the case-study link renders; the per-stage
    summaries never do — they are the 13 chapter pages' content, byte-identical
    across a model's sibling config pages otherwise)."""

    key: str
    name: str
    headline: str
    has_data: bool
    e2e: dict | None
    spec: str | None
    kv: str | None
    accuracy: list[tuple[str, str]]
    kernel_study: bool


@dataclass(frozen=True)
class BenchContext:
    """The merged Benchmarks section (cycle 3f): one :class:`PlatformBench`
    per serving platform + the no-data one-liner when no platform has any
    cell data. The template renders; this module computes."""

    sections: list[PlatformBench]
    no_data: str | None


def bench_context(layout: RepoLayout, model: str, quant: str, plats: list[PlatformRow]) -> BenchContext:
    """One PlatformBench per serving platform (the overlay rule, applied by
    the caller) plus the no-data line — the old BENCHMARKS.md placeholder,
    kept verbatim except that it names every serving platform at once."""
    sections = [_bench_section(p, model, quant, layout) for p in plats]
    if all(not s.has_data for s in sections):
        names = ", ".join(p.name for p in plats)
        no_data = (
            "**no data yet** — no end-to-end, speculative-decoding, KV-capacity or accuracy "
            f"run for `{model}` / `{quant}` on {names} so far."
        )
    else:
        no_data = None
    return BenchContext(sections=sections, no_data=no_data)


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


def _bench_section(p: PlatformRow, model: str, quant: str, layout: RepoLayout) -> PlatformBench:
    from mjolnir_cli import readme_targets  # the (platform, model) data scope

    docs = (
        readme_targets.pair_docs(layout, p.key, DEFAULT_MODEL) if readme_targets.has_kernel_raws(layout, p.key) else {}
    )
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
        kernel_study=bool(docs),
    )


def _e2e_cell(layout: RepoLayout, platform: str, model: str, quant: str) -> dict | None:
    """The (platform, model, config) end-to-end cell, as the open section
    renders it: the pair's A/B when the config is one of its sides, else the
    config's own runs (no partner to compare against); None when the cell has
    no data at all."""
    from mjolnir_cli import e2e

    off, mj = e2e.latest_pair(layout.history_file, model=model, platform=platform)
    if off and mj and quant in (str(off.get("config", "")), str(mj.get("config", ""))):
        pair = _decode_pair(off, mj)
        if pair is None:
            return None
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
        label = f"{off.get('config', '')} vs {mj.get('config', '')}, 1 request"
        return {
            "kind": "pair",
            "label": label + (f" · {ctx} context" if ctx else ""),
            "official": str(off.get("config", "")),
            "mjolnir": str(mj.get("config", "")),
            "rows": rows,
            "chart_pair": (platform, model),
        }
    # no partner: the config's own runs, one value column
    recs = [
        r
        for r in load_records(layout.history_file)
        if platform_of(r) == platform and model_of(r) == model and r.get("config") == quant
    ]
    mine = e2e._merge(recs)
    if mine is None:
        return None
    cells = [(c.get("concurrency"), c.get("context") or 0) for c in mine.get("cells", []) if not c.get("prefill_phase")]
    at = max((x for c, x in cells if c == 1), default=None)
    if at is None:
        return None
    rows = []
    for m in e2e.METRICS:
        v = e2e.value(e2e.cell(mine, 1, at), m.key)
        rows.append({"metric": m.label, "unit": m.unit, "value": "—" if v is None else m.fmt.format(v)})
    if all(r["value"] == "—" for r in rows):
        return None
    label = f"{quant}, 1 request" + (f" · {e2e.ctx_label(at)} context" if at else "")
    return {"kind": "solo", "label": label, "config": quant, "rows": rows}


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
