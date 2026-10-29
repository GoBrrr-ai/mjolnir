"""Structured verify results — the raw substrate for every chart and table.

Each ``mjolnir verify <stage>`` run writes ONE JSON document:

    benchmarks/raw/verify/<stage>/<run_id>.json

Schema ``mjolnir.verify/1`` (all fields optional except the header):

    {
      "schema": "mjolnir.verify/1",
      "stage": "sweep",                    # stage name (see stages.STAGES)
      "run_id": "20261010-131259",         # UTC timestamp, sortable
      "ts": "2026-10-10T13:12:59+00:00",
      "default": true,                     # ran with the stage's default parameters
      "host": "jetsonthor", "image": "narandill/mjolnir-vllm:v0.32.0",
      "kernel": {"version": "0.1.0", "git": "68ef2b3", "dirty": false},
      "gate": {"gated": true, "clean": true, "server": "up"},
      "params": {...},                     # every knob the stage ran with
      "series": [                          # ncu stages: one entry per leg / variant
        {"id": "fi", "label": "FlashInfer", "kind": "fi", "role": "baseline",
         "flags": "", "kv_bits": 8,
         "points": [{"L": 8192, "total_us": 96.1, "main_us": 89.2,
                     "merge_us": 6.9, "l2_sectors": 1.1e6, "ctas": 40,
                     "invocations": 50, "gbps": 174.8}]}
      ],
      "rows": [...],                       # harness stages: one dict per measurement
      "cases": [...],                      # correctness: one dict per case
      "profiles": [...],                   # fp6prof: full-ncu metric dicts
      "summary": {...}                     # headline numbers computed at save time
    }

Readers (charts, README, docs) pick the latest *default* run of a stage, so
an ad-hoc A/B with custom flags never replaces a headline chart by accident
(``latest(stage, default_only=False)`` returns any run). The writer enforces
the same rule at save time (:func:`save`): default runs of rendered stages
replace the tracked evidence for their (stage, platform, model); everything
else — ad-hoc A/Bs, uncharted stages — lands in the gitignored
``benchmarks/staging/`` instead, so the tracked raws stay exactly what the
render reads no matter how many experiments run.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path
from typing import Any

import platform as host_platform

from mjolnir_cli.config import DEFAULT_MODEL, model_of
from mjolnir_cli.platforms import platform_of

SCHEMA = "mjolnir.verify/1"

# Stages whose runs are never rendered (no chart, no docs chapter — see
# ``readme_targets``). Their runs always land in the untracked staging area,
# default parameters included, so the tracked raws are exactly what the render
# reads. Mirrored by ``rawprune`` (the audit).
UNCHARTED = ("timing", "profile", "vllm")


def new_run_id() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")


def git_info(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {}
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--", "src/mjolnir"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"git": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def header(
    stage: str,
    run_id: str,
    *,
    image: str,
    default: bool,
    params: dict,
    kernel: dict,
    gate: dict,
    platform: str | None = None,
    gpu: str | None = None,
) -> dict:
    doc = {
        "schema": SCHEMA,
        "stage": stage,
        "run_id": run_id,
        "ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "default": default,
        "host": host_platform.node(),
        "image": image,
        "kernel": kernel,
        "gate": gate,
        "params": params,
    }
    if platform is not None:
        doc["platform"] = platform
    if gpu is not None:
        doc["gpu"] = gpu
    return doc


def stage_dir(verify_dir: Path, stage: str) -> Path:
    return verify_dir / stage


def staging_dir(verify_dir: Path) -> Path:
    """The untracked landing zone (``benchmarks/staging/``) — derived from the
    verify dir (``benchmarks/raw/verify`` → ``benchmarks/staging``), the same
    path :class:`~mjolnir_cli.config.RepoLayout.staging_dir` builds."""
    return verify_dir.parent.parent / "staging"


def save(verify_dir: Path, doc: dict) -> Path:
    """File a run's document.

    A default-parameter run of a rendered stage IS the tracked evidence for
    its (stage, platform, model): it lands in ``raw/verify/<stage>/`` and
    replaces the previous default run of that group — the newest-default-wins
    rule the render applies, applied at write time (git history keeps the
    retired file). Everything else — an ad-hoc A/B (custom parameters) or a
    run of an uncharted stage — lands in ``benchmarks/staging/`` (gitignored):
    visible locally, never shipped, never in a chart by accident."""
    tracked = bool(doc.get("default")) and doc["stage"] not in UNCHARTED
    d = stage_dir(verify_dir, doc["stage"]) if tracked else staging_dir(verify_dir) / "verify" / doc["stage"]
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{doc['run_id']}.json"
    p.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    if tracked:
        _retire_superseded(verify_dir, doc)
    return p


def _group(doc: dict) -> tuple[str, str]:
    """The evidence group a document belongs to — mirrors ``readme_targets``
    (``_matches``): a model-less document belongs to the platform's headline
    model."""
    return (platform_of(doc), model_of(doc) or DEFAULT_MODEL)


def _retire_superseded(verify_dir: Path, doc: dict) -> None:
    """Delete the previous default run of the same (stage, platform, model):
    the render would never read it (``_newest`` prefers the newest default),
    and keeping it would grow the repo with every re-run."""
    for old in runs(verify_dir, doc["stage"]):
        if old["run_id"] == doc["run_id"] or not old.get("default"):
            continue
        if _group(old) == _group(doc):
            Path(old["_path"]).unlink(missing_ok=True)


def load(path: Path) -> dict | None:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return None
    return doc


def runs(verify_dir: Path, stage: str) -> list[dict]:
    """Every saved run of a stage, oldest first."""
    d = stage_dir(verify_dir, stage)
    out = []
    for p in sorted(d.glob("*.json")):
        doc = load(p)
        if doc:
            doc["_path"] = str(p)
            out.append(doc)
    return out


def latest(
    verify_dir: Path,
    stage: str,
    default_only: bool = True,
    platform: str | None = None,
    model: str | None = None,
) -> dict | None:
    """The newest run of a stage — the newest *default-parameter* run unless
    ``default_only`` is False (then any run). Falls back to any run when the
    stage has no default run yet. ``platform`` resolves platform-less (legacy)
    documents as thor; ``model`` skips model-less documents (mirrors
    :func:`model_of` — a guessed model is how two models' runs get compared)."""
    rs = runs(verify_dir, stage)
    if platform is not None:
        rs = [r for r in rs if platform_of(r) == platform]
    if model is not None:
        rs = [r for r in rs if (r.get("model") or None) == model]
    if not rs:
        return None
    if default_only:
        dflt = [r for r in rs if r.get("default")]
        if dflt:
            return dflt[-1]
    return rs[-1]


# ── series helpers (shared by charts / README / docs) ───────────────────────


def series_by_id(doc: dict) -> dict[str, dict]:
    return {s["id"]: s for s in doc.get("series", [])}


def point_map(series: dict, field: str = "total_us") -> dict[int, float]:
    return {int(p["L"]): float(p[field]) for p in series.get("points", []) if p.get(field) is not None}


def speedup(base: dict, other: dict) -> dict[int, float]:
    """base time / other time per L (>1 = ``other`` faster)."""
    b, o = point_map(base), point_map(other)
    return {L: b[L] / o[L] for L in sorted(b) if L in o and o[L] > 0}


def geomean(xs) -> float | None:
    import math

    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None
