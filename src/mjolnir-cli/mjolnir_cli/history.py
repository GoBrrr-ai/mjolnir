"""Append-only benchmark history (``benchmarks/history.jsonl``).

One line per perf sweep repeat:

    {
      "ts": "2026-09-25T14:03:11+00:00", "epoch": 1758801791.123,
      "host": "thor-01", "image": "mjolnir/vllm-thor:qwen38-sm110-v11",
      "model": "Qwen/Qwen3.8-27B", "config": "NVFP4_MJOLNIR",
      "backend": "Mjolnir",
      "gate": {"clean": true, "confirm": 3, "open_ts": 1758801600.0},
      "runs": 5, "warmup_runs": 2, "exact_tg": true,
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
"progress over time" — every nightly bump / kernel change adds a row.
"""

from __future__ import annotations

import json
import sys
import platform
import time
from pathlib import Path
from typing import Any


def append_record(history_file: Path, record: dict[str, Any]) -> None:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    # a hand-edited file may lack the final newline: never glue a record onto it
    lead = ""
    if history_file.exists() and history_file.stat().st_size:
        with history_file.open("rb") as f:
            f.seek(-1, 2)
            lead = "" if f.read(1) == b"\n" else "\n"
    with history_file.open("a") as f:
        f.write(lead + json.dumps(record) + "\n")


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
) -> dict[str, Any]:
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime()).replace("+0000", "+00:00"),
        "epoch": time.time(),
        "host": platform.node() or "unknown",
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
    from the per-run samples)."""
    cells = []
    for b in raw.get("benchmarks", []):

        def metric(name: str) -> dict | None:
            m = b.get(name)
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
            m = metric(src)
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
