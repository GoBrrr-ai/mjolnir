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
(``latest(stage, default_only=False)`` returns any run).
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "mjolnir.verify/1"


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


def header(stage: str, run_id: str, *, image: str, default: bool, params: dict, kernel: dict, gate: dict) -> dict:
    return {
        "schema": SCHEMA,
        "stage": stage,
        "run_id": run_id,
        "ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "default": default,
        "host": platform.node(),
        "image": image,
        "kernel": kernel,
        "gate": gate,
        "params": params,
    }


def stage_dir(verify_dir: Path, stage: str) -> Path:
    return verify_dir / stage


def save(verify_dir: Path, doc: dict) -> Path:
    d = stage_dir(verify_dir, doc["stage"])
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{doc['run_id']}.json"
    p.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    return p


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


def latest(verify_dir: Path, stage: str, default_only: bool = True) -> dict | None:
    """The newest run of a stage — the newest *default-parameter* run unless
    ``default_only`` is False (then any run). Falls back to any run when the
    stage has no default run yet."""
    rs = runs(verify_dir, stage)
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
