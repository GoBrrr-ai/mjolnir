"""The history substrate: serving-key upsert round-trip, llama-benchy raw
parsing, and record assembly (what the charts + the CI substrate gate consume)."""

from __future__ import annotations

import json
import subprocess

import pytest

from mjolnir_cli.config import Settings
from mjolnir_cli.history import append_record, load_records, make_record, parse_benchy_json


def test_append_then_load_roundtrip(tmp_path):
    f = tmp_path / "history.jsonl"
    append_record(f, {"a": 1})
    append_record(f, {"a": 2})
    assert load_records(f) == [{"a": 1}, {"a": 2}]  # no serving key → verbatim


def _rec(epoch: int, *, spec=None, cells=None, model="Qwen/Qwen3.8-27B", config="NVFP4", image="img:v1") -> dict:
    return {
        "schema_version": 1,
        "epoch": float(epoch),
        "platform": "thor",
        "model": model,
        "config": config,
        "image": image,
        "spec": spec,
        "cells": cells or [{"concurrency": 1, "context": 0, "tg_tps": {"mean": 41.2}}],
        "raw": f"staging/perf-{epoch}/",
    }


def test_upsert_merges_cells_into_the_stored_record(tmp_path):
    f = tmp_path / "history.jsonl"
    append_record(f, _rec(1, cells=[{"concurrency": 1, "context": 0, "tg_tps": {"mean": 41.2}}]))
    append_record(f, _rec(2, cells=[{"concurrency": 1, "context": 65536, "tg_tps": {"mean": 30.0}}]))
    recs = load_records(f)
    assert len(recs) == 1  # one record per serving key — a re-run never adds a line
    r = recs[0]
    assert [(c["concurrency"], c["context"]) for c in r["cells"]] == [(1, 0), (1, 65536)]
    assert r["epoch"] == 2.0  # the newest measurement is the head
    assert r["merged_runs"] == ["staging/perf-2/", "staging/perf-1/"]  # newest contribution first


def test_upsert_reuses_a_cell_from_the_newest_run_that_measured_it(tmp_path):
    f = tmp_path / "history.jsonl"
    append_record(f, _rec(1, cells=[{"concurrency": 1, "context": 0, "tg_tps": {"mean": 41.2}}]))
    append_record(f, _rec(2, cells=[{"concurrency": 1, "context": 0, "tg_tps": {"mean": 43.0}}]))
    (rec,) = load_records(f)
    assert rec["cells"][0]["tg_tps"]["mean"] == 43.0
    assert rec["merged_runs"] == ["staging/perf-2/"]


def test_different_spec_replaces_the_record_and_stages_the_old_one(tmp_path):
    f = tmp_path / "history.jsonl"
    append_record(f, _rec(1, spec={"method": "mtp", "tokens": 3}))
    append_record(f, _rec(2, spec=None))
    (rec,) = load_records(f)
    assert rec["epoch"] == 2.0 and rec["spec"] is None
    extras = [json.loads(line) for line in (tmp_path / "staging" / "history-extras.jsonl").read_text().splitlines()]
    assert [e["epoch"] for e in extras] == [1.0] and extras[0]["spec"] == {"method": "mtp", "tokens": 3}


def test_load_missing_file_is_empty(tmp_path):
    assert load_records(tmp_path / "nope.jsonl") == []


def test_parse_benchy_json_maps_cells():
    raw = {
        "benchmarks": [
            {
                "concurrency": 1,
                "context_size": 8192,
                "prompt_size": 2048,
                "response_size": 128,
                "is_context_prefill_phase": False,
                "tg_throughput": {"mean": 41.2, "std": 0.7, "values": [40.0, 42.4]},
                "pp_throughput": {"mean": 9830.0, "std": 120.0, "values": [9800.0, 9860.0]},
            },
            {
                "concurrency": 4,
                "context_size": 0,
                "prompt_size": 2048,
                "response_size": 128,
                "is_context_prefill_phase": True,
                "tg_throughput": {"mean": 1.0, "std": 0.0},  # no 'values' -> omitted
            },
        ]
    }
    cells = parse_benchy_json(raw)
    assert len(cells) == 2
    c0, c1 = cells
    assert (c0["concurrency"], c0["context"], c0["prompt"], c0["gen"]) == (1, 8192, 2048, 128)
    assert c0["prefill_phase"] is False
    assert c0["tg_tps"] == {"mean": 41.2, "std": 0.7, "values": [40.0, 42.4]}
    assert c0["pp_tps"]["mean"] == 9830.0
    assert c1["prefill_phase"] is True
    assert c1["tg_tps"] == {"mean": 1.0, "std": 0.0}  # no 'values' key at all
    assert "pp_tps" not in c1  # absent source metric -> absent cell field


def test_parse_benchy_json_empty():
    assert parse_benchy_json({"benchmarks": []}) == []
    assert parse_benchy_json({}) == []


def test_make_record_maps_settings():
    # a resolved Settings (platform set) and an explicit gpu: the record is
    # written without probing the host (hermetic — CI has no GPU)
    s = Settings(image="img:v1", model="Qwen/Qwen3.8-27B", quant="NVFP4", platform="thor")
    rec = make_record(
        s,
        gate_summary={"clean": True, "confirm": 3},
        runs=5,
        warmup_runs=2,
        exact_tg=True,
        cells=[{"concurrency": 1, "context": 0, "prompt": 2048, "gen": 128}],
        raw_relpath="raw/perf-x/",
        gpu="",
    )
    assert rec["image"] == "img:v1"
    assert rec["model"] == "Qwen/Qwen3.8-27B"
    assert rec["config"] == "NVFP4"
    assert rec["backend"] == "FlashInfer"  # backend_label via BACKEND_LABELS
    assert rec["gate"] == {"clean": True, "confirm": 3}
    assert rec["runs"] == 5 and rec["warmup_runs"] == 2 and rec["exact_tg"] is True
    assert rec["raw"] == "raw/perf-x/"
    assert "ts" in rec and "epoch" in rec and "host" in rec


# ── the platform + gpu record fields (cycle 2: detection-first) ─────────────


def _fake_smi(monkeypatch, stdout="Fake GPU, 11.0\n"):
    """Point the nvidia-smi subprocess at a canned csv answer."""
    from platform_fixtures import fake_smi

    fake_smi(monkeypatch, stdout)


def _record(s, **kw):
    return make_record(s, gate_summary=None, runs=1, warmup_runs=0, exact_tg=True, cells=[], raw_relpath="raw/x/", **kw)


def test_make_record_platform_comes_from_detection(monkeypatch):
    # no explicit platform, no env: the record's platform is the detected one
    # (spark here), and the gpu line comes from the same nvidia-smi answer
    from platform_fixtures import install

    _fake_smi(monkeypatch, "DGX Spark, 12.1\n")
    install(monkeypatch, smi="DGX Spark, 12.1\n")
    s = Settings(image="img:v1", model="Qwen/Qwen3.8-27B", quant="NVFP4")
    rec = _record(s)
    assert rec["platform"] == "spark"
    assert rec["gpu"] == "DGX Spark"


def test_make_record_explicit_platform_and_gpu_win_over_detection(monkeypatch):
    _fake_smi(monkeypatch, stdout="Should Not Be Consulted, 11.0\n")
    s = Settings(image="img:v1", model="m", quant="c")
    rec = _record(s, platform="spark", gpu="Other GPU")
    assert rec["platform"] == "spark" and rec["gpu"] == "Other GPU"


def test_make_record_uses_the_resolved_settings_platform_without_detecting(monkeypatch):
    # a resolved Settings (s.platform set) is authoritative: no detection probe
    from platform_fixtures import install

    _fake_smi(monkeypatch)  # the gpu line still parses nvidia-smi — faked, never the host
    install(monkeypatch, smi=None, jetson=None, dmi=None)  # detection would fail
    s = Settings(image="img:v1", model="m", quant="c", platform="orin")
    rec = _record(s)
    assert rec["platform"] == "orin"


def test_make_record_detection_failure_is_explicit(monkeypatch):
    # no GPU, no board files: the record cannot be written — the error names
    # the probes instead of guessing a platform
    import mjolnir_cli.platforms as platforms
    from platform_fixtures import install

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", boom)
    install(monkeypatch, smi=None, jetson=None, dmi=None)
    s = Settings(image="img:v1", model="m", quant="c")
    with pytest.raises(platforms.UnknownPlatform) as ei:
        _record(s)
    assert "nvidia-smi" in str(ei.value) and "MJOLNIR_PLATFORM" in str(ei.value)
