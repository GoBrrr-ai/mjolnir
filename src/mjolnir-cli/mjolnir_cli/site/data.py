"""The site data layer: the manifest + one lazy payload per measured pair.

Emission shape (``task/plan.md`` §1 / design §9):

    _site/
      index.html
      manifest.json                      # small: pairs, stages index, default
      app.js, app.css                    # real files (page.py copies them)
      data/<platform>/<model-short>.json # one payload per MEASURED pair

Every builder reads the SAME loaders as the static renderers
(``readme_targets.pair_docs``/``measured_pairs``/``config_cell``,
``charts_e2e.e2e_rows``, ``mjolnir_cli.availability``) — no duplicated data
logic. Payloads are slim (the ``_slim`` rules: no ``kernels``/``details``/
``pipes``); the manifest carries no data at all, just the pair index the
lazy loader needs.
"""

from __future__ import annotations

import json

from mjolnir_cli.config import BACKEND_LABELS, DEFAULT_MODEL, RepoLayout, model_short
from mjolnir_cli.verify.stages import NOISE_PCT, STAGES


def _slim(doc: dict) -> dict:
    """Drop per-kernel detail and full ncu detail pages to keep payloads small."""
    d = {k: v for k, v in doc.items() if not k.startswith("_")}
    if "series" in d:
        d["series"] = [
            {
                **s,
                "points": [{k: v for k, v in p.items() if k != "kernels"} for p in s.get("points", [])],
            }
            for s in d["series"]
        ]
    if "profiles" in d:
        d["profiles"] = [{k: v for k, v in p.items() if k not in ("details", "pipes")} for p in d["profiles"]]
    if "cases" in d:
        d["cases"] = [{k: c.get(k) for k in ("label", "q", "passed", "check")} for c in d["cases"]]
    return d


def _e2e_cells(rec: dict) -> dict:
    """One e2e record as the payload's slim entry: the compare contract is the
    (concurrency, context) cell grid — identical field names on every platform."""
    return {
        "config": rec.get("config"),
        "image": rec.get("image"),
        "cells": [
            {
                "concurrency": c.get("concurrency"),
                "context": c.get("context") or 0,
                "tg": (c.get("tg_tps") or {}).get("mean"),
                "pp": (c.get("pp_tps") or {}).get("mean") if c.get("context") else None,  # no ctx: no prefill
            }
            for c in rec.get("cells", [])
        ],
    }


def _pair_configs(layout: RepoLayout, platform: str, model: str) -> list[dict]:
    """The model's configs that exist on the platform (the overlay rule), each
    with its availability cell — always from :mod:`mjolnir_cli.availability`
    via ``readme_targets.config_cell`` (no duplicated table)."""
    from mjolnir_cli import overlays, readme_targets

    out = []
    for cd in readme_targets.scan_configs(layout):
        if cd.model != model or not overlays.config_exists_on(cd.path, platform):
            continue
        cell = readme_targets.config_cell(cd.path, cd.quant, platform)
        out.append(
            {
                "quant": cd.quant,
                "label": BACKEND_LABELS.get(cd.quant, cd.quant),
                "state": cell.state,
                "why": cell.why,
            }
        )
    return out


def pair_payload(layout: RepoLayout, platform: str, model: str) -> dict:
    """One measured pair's payload: its verify stage docs (slim), its e2e cell
    grid and its config list with the platform's availability cells."""
    from mjolnir_cli import readme_targets

    docs = readme_targets.pair_docs(layout, platform, model)
    return {
        "platform": platform,
        "model": model,
        "short": model_short({"model": model}),
        "stages": {stage: _slim(doc) for stage, doc in docs.items()},
        "e2e": {cfg: _e2e_cells(rec) for cfg, rec in _e2e_rows(layout, model, platform).items()},
        "configs": _pair_configs(layout, platform, model),
    }


def _e2e_rows(layout: RepoLayout, model: str, platform: str) -> dict[str, dict]:
    """``charts_e2e.e2e_rows`` scoped to the pair (the platform hook)."""
    from mjolnir_cli.charts_e2e import e2e_rows

    return e2e_rows(layout, model=model, platform=platform)


def _chart_ids(layout: RepoLayout, platform: str, model: str) -> list[str]:
    """The pair's chart ids from the v2 chart manifest (written by
    ``charts.render_all``; entries carry the platform/model identity)."""
    p = layout.charts_dir / "manifest.json"
    if not p.is_file():
        return []
    manifest = json.loads(p.read_text())
    # v2 manifests are list-valued (one entry per measured pair); legacy
    # single-entry dicts still resolve.
    ids = []
    for name, e in manifest.items():
        entries = e if isinstance(e, list) else [e]
        if any(x.get("platform") == platform and x.get("model") == model for x in entries):
            ids.append(name)
    return sorted(ids)


def build_site(layout: RepoLayout) -> tuple[dict, list[tuple[str, str]]]:
    """The manifest plus every payload, in one pass —
    ``(manifest_dict, [(site-relative path, body), …])``; ``page.emit_site``
    writes them. One payload per MEASURED pair, none for unmeasured ones.

    The manifest is the small lazy-load index: each measured pair with its
    measured stages, chart ids and payload size; the verify-stage index
    (title/summary/kind); the noise band; the no-hash hero default (thor, the
    default model)."""
    from mjolnir_cli import readme_targets

    pairs = []
    files = []
    for platform, model in readme_targets.measured_pairs(layout):
        payload = pair_payload(layout, platform, model)
        body = json.dumps(payload, separators=(",", ":"))
        pairs.append(
            {
                "platform": platform,
                "model": model,
                "short": model_short({"model": model}),
                "stages": sorted(payload["stages"]),
                "charts": _chart_ids(layout, platform, model),
                "size": len(body.encode()),
            }
        )
        files.append((f"data/{platform}/{model_short({'model': model})}.json", body))
    m = {
        "default": {"platform": "thor", "model": DEFAULT_MODEL},
        "stages": {s.name: {"title": s.title, "summary": s.summary, "kind": s.kind} for s in STAGES.values()},
        "noise_pct": NOISE_PCT,
        "pairs": pairs,
    }
    return m, files


__all__ = ["build_site", "pair_payload"]
