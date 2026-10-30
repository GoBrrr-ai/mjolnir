"""The benchmark history (``benchmarks/history.jsonl``).

One record per serving key (platform, model, image, config):

    {
      "schema_version": 1,
      "ts": "2026-09-25T14:03:11+00:00", "epoch": 1758801791.123,
      "host": "thor-01", "image": "mjolnir/vllm-thor:qwen38-sm110-v11",
      "model": "Qwen/Qwen3.8-27B", "config": "NVFP4_MJOLNIR",
      "backend": "Mjolnir",
      "gate": {"clean": true, "confirm": 3, "open_ts": 1758801600.0},
      "runs": 5, "warmup_runs": 1, "exact_tg": true,
      "cells": [
        {"concurrency": 1, "context": 8192, "prompt": 2048, "gen": 128,
         "tg_tps": {"mean": 41.2, "std": 0.7, "values": [...]},   # decode t/s (all requests)
         "tg_req_tps": {...},                                      # decode t/s per request
         "pp_tps": {"mean": 9830.0, "std": 120.0, "values": [...]},  # prefill t/s
         "ttfr_ms": {...}, "ttft_ms": {...},                       # first response / e2e TTFT
         "itl_ms": {...}},                                         # inter-token latency = 1000 / tg_req
        ...
      ]
    }

The file is the raw, git-trackable substrate for the charts and for
"progress over time". :func:`append_record` files a run under its serving
key — a re-run merges its cells into the stored record (each cell kept from
the newest run that measured it, so a ``--depths`` run adds cells instead of
replacing the table) instead of appending another line, and a run with a
different speculative setup replaces the record outright (the superseded one
moves to the gitignored ``benchmarks/staging/history-extras.jsonl``). Git
history keeps every superseded version.

Records written before ``schema_version`` existed lack the field — readers treat
absence as the legacy (pre-0.3) format. :func:`load_records` returns rows raw,
never normalized.

Declared a public, stable library surface — see "Library use" in
``src/mjolnir-cli/README.md``.
"""

from __future__ import annotations

import json
import platform as host_platform
import sys
import time
from pathlib import Path
from typing import Any

from mjolnir_cli.platforms import detect, gpu_name, platform_of

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION", "append_record", "load_records"]


def serving_key(rec: dict[str, Any]) -> tuple[str, Any, Any, Any] | None:
    """The serving key a perf record files under: (platform, model, image,
    config). A record without any of model/image/config cannot be classified —
    :func:`append_record` appends it verbatim."""
    if not (rec.get("model") or rec.get("config") or rec.get("image")):
        return None
    return (platform_of(rec), rec.get("model"), rec.get("image"), rec.get("config"))


def spec_of(rec: dict[str, Any]) -> dict | None:
    """The speculative decoding a record ran with: stored in the record since
    the field exists, else read from the config it names."""
    if rec.get("spec") is not None:
        return rec["spec"] or None
    from mjolnir_cli.config import config_speculative, configs_root

    path = configs_root() / str(rec.get("model", "")) / str(rec.get("config", "")) / "config.yaml"
    return config_speculative(path) if path.exists() else None


def append_record(history_file: Path, record: dict[str, Any]) -> None:
    """File one perf record under its serving key (see the module docstring):
    a new key appends a line, a known key merges into its stored record and
    the file is rewritten (the merged record — the newest measurement — takes
    the file's end, keeping the chronological order ``mjolnir history`` and
    ``[-limit:]`` readers expect)."""
    history_file.parent.mkdir(parents=True, exist_ok=True)
    recs = load_records(history_file)
    key = serving_key(record)
    idx = [i for i, r in enumerate(recs) if key is not None and serving_key(r) == key] if key else []
    if not idx:
        _append_line(history_file, record)  # a new serving key (or unclassifiable)
        return
    merged, superseded = _merge_records([record, *(recs[i] for i in idx)])
    kept = [r for i, r in enumerate(recs) if i not in idx]
    kept.append(merged)
    superseded.extend(recs[i] for i in idx)
    _rewrite(history_file, kept)
    if superseded:
        _stage_extras(history_file.parent / "staging" / "history-extras.jsonl", superseded)


def _merge_records(recs: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Newest record first → (the merged record, the superseded ones). Every
    (concurrency, context, prefill_phase) cell is kept from the newest run
    that measured it; runs with a different speculative setup than the newest
    are superseded (a different experiment, not evidence); ``merged_runs``
    lists the raw dirs that contributed cells — the reader-side rule
    (``e2e._merge``) applied at write time."""
    rs = sorted(recs, key=lambda r: r.get("epoch", 0), reverse=True)
    sp = spec_of(rs[0])
    cells: dict = {}
    used: list[str] = []
    superseded: list[dict[str, Any]] = []
    for r in rs:
        if spec_of(r) != sp:
            superseded.append(r)
            continue
        took = False
        for c in r.get("cells", []):
            ck = (c.get("concurrency"), c.get("context") or 0, bool(c.get("prefill_phase")))
            if ck not in cells:
                cells[ck] = c
                took = True
        if took:
            used.append(str(r.get("raw", "")))
    out = dict(rs[0])
    out["cells"] = sorted(cells.values(), key=lambda c: (c.get("concurrency") or 0, c.get("context") or 0))
    out["merged_runs"] = used
    return out, superseded


def _append_line(history_file: Path, record: dict[str, Any]) -> None:
    # a hand-edited file may lack the final newline: never glue a record onto it
    lead = ""
    if history_file.exists() and history_file.stat().st_size:
        with history_file.open("rb") as f:
            f.seek(-1, 2)
            lead = "" if f.read(1) == b"\n" else "\n"
    with history_file.open("a") as f:
        f.write(lead + json.dumps(record) + "\n")


def _rewrite(history_file: Path, recs: list[dict[str, Any]]) -> None:
    tmp = history_file.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    tmp.replace(history_file)


def _stage_extras(path: Path, recs: list[dict[str, Any]]) -> None:
    """Park superseded records in the gitignored staging area (provenance is
    on disk until the next repo clean; git history keeps them regardless)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for r in recs:
        _append_line(path, r)


def load_records(history_file: Path) -> list[dict[str, Any]]:
    """Every record, one JSON object per line — tolerant of objects glued
    onto one line (an append after a hand edit that dropped the newline) and
    of a truncated last line (an interrupted write), which is skipped with a
    warning instead of failing every reader."""
    if not history_file.exists():
        return []
    out = []
    dec = json.JSONDecoder()
    with history_file.open() as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            i = 0
            while i < len(line):
                try:
                    rec, i = dec.raw_decode(line, i)
                except json.JSONDecodeError as e:
                    print(f"[mjolnir] {history_file}:{n}: skipped unreadable data ({e})", file=sys.stderr)
                    break
                out.append(rec)
                while i < len(line) and line[i].isspace():
                    i += 1
    return out


def make_record(
    s,
    gate_summary: dict | None,
    runs: int,
    warmup_runs: int,
    exact_tg: bool,
    cells: list[dict[str, Any]],
    raw_relpath: str,
    platform: str | None = None,
    gpu: str | None = None,
) -> dict[str, Any]:
    """Best-effort row assembly from a CLI ``Settings`` — not part of the
    promised library surface (the Settings object is the CLI's internal type)."""
    plat = platform if platform is not None else s.platform
    if plat is None:  # an unresolved Settings: the record names the *detected* platform
        plat = detect().platform
    return {
        "schema_version": SCHEMA_VERSION,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime()).replace("+0000", "+00:00"),
        "epoch": time.time(),
        "host": host_platform.node() or "unknown",
        "gpu": gpu if gpu is not None else gpu_name(),
        # `is not None` on both: an explicit value (even "") wins over detection.
        "platform": plat,
        "image": s.image,
        "model": s.model,
        "config": s.quant,
        "backend": s.backend_label,
        "spec": _spec(s),
        "gate": gate_summary,
        "runs": runs,
        "warmup_runs": warmup_runs,
        "exact_tg": exact_tg,
        "cells": cells,
        "raw": raw_relpath,
    }


def _spec(s) -> dict | None:
    """Speculative decoding the served config used (e2e numbers depend on it)."""
    try:
        from mjolnir_cli.config import config_speculative

        return config_speculative(s.config_path)
    except Exception:  # noqa: BLE001
        return None


def parse_benchy_json(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """llama-benchy JSON → normalized per-cell records with the metrics we
    chart (tg / pp generation + prefill throughput, each with mean/std/values
    from the per-run samples). Best-effort — it mirrors llama-benchy's
    third-party JSON, so it is not part of the promised library surface."""
    cells = []
    for b in raw.get("benchmarks", []):

        def metric(rec: dict, name: str) -> dict | None:
            m = rec.get(name)
            if not isinstance(m, dict):
                return None
            vals = m.get("values")
            out = {"mean": m.get("mean"), "std": m.get("std")}
            if vals is not None:
                out["values"] = [float(v) for v in vals]
            return out

        cell = {
            "concurrency": b.get("concurrency"),
            "context": b.get("context_size"),
            "prompt": b.get("prompt_size"),
            "gen": b.get("response_size"),
            "prefill_phase": bool(b.get("is_context_prefill_phase", False)),
        }
        for src, dst in (
            ("tg_throughput", "tg_tps"),
            ("tg_req_throughput", "tg_req_tps"),
            ("pp_throughput", "pp_tps"),
            ("peak_throughput", "peak_tps"),
            ("ttfr", "ttfr_ms"),
            ("e2e_ttft", "ttft_ms"),
        ):
            m = metric(b, src)
            if m is not None:
                cell[dst] = m
        # inter-token latency: the per-request decode rate inverted, per run
        # (at c=1 tg_req == tg). llama-benchy reports rates, not ITL itself.
        if (b.get("concurrency") or 1) == 1:
            rate = cell.get("tg_req_tps") or cell.get("tg_tps")
        else:
            rate = cell.get("tg_req_tps")
        if rate and rate.get("values"):
            itl = [1000.0 / v for v in rate["values"] if v and v > 0]
            if itl:
                mean = sum(itl) / len(itl)
                std = (sum((v - mean) ** 2 for v in itl) / (len(itl) - 1)) ** 0.5 if len(itl) > 1 else 0.0
                cell["itl_ms"] = {"mean": mean, "std": std, "values": itl}
        cells.append(cell)
    return cells
