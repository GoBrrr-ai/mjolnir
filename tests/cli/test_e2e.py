"""Official vLLM image vs Mjolnir image: pairing, gains, ITL/TTFT fallbacks."""

from __future__ import annotations

import json

from mjolnir_cli import e2e
from mjolnir_cli.history import parse_benchy_json


def _cell(conc, ctx, tg, pp, ttfr):
    return {
        "concurrency": conc,
        "context": ctx,
        "tg_tps": {"mean": tg},
        "pp_tps": {"mean": pp},
        "ttfr_ms": {"mean": ttfr},
    }


def _rec(image, config, epoch, cells):
    return {"image": image, "config": config, "epoch": epoch, "backend": "x", "cells": cells}


def test_pair_prefers_mjolnir_backend_and_latest(tmp_path):
    h = tmp_path / "history.jsonl"
    rows = [
        _rec("vllm/vllm-openai:v0.30.0-ubuntu2404", "NVFP4", 1, [_cell(1, 8192, 20, 2000, 4000)]),
        _rec("vllm/vllm-openai:v0.30.0-ubuntu2404", "NVFP4", 3, [_cell(1, 8192, 22, 2100, 3900)]),
        _rec("narandill/mjolnir-vllm:v0.30.0", "NVFP4", 5, [_cell(1, 8192, 23, 2500, 3500)]),
        _rec("narandill/mjolnir-vllm:v0.30.0", "NVFP4_MJOLNIR", 4, [_cell(1, 8192, 30, 2600, 3200)]),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h)
    assert off["epoch"] == 3 and mj["config"] == "NVFP4_MJOLNIR"  # backend config beats a newer FI run
    hero = e2e.hero(off, mj)
    assert hero["ctx"] == 8192
    assert round(hero["tg_tps"][2], 1) == round((30 / 22 - 1) * 100, 1)
    # lower-is-better metrics: positive gain = Mjolnir lower; TTFT falls back to ttfr, ITL to 1000/tg at c=1
    assert round(hero["ttft_ms"][2], 1) == round((3900 / 3200 - 1) * 100, 1)
    assert round(hero["itl_ms"][1], 3) == round(1000 / 30, 3)


def test_benchy_cells_carry_ttft_and_itl():
    raw = {
        "benchmarks": [
            {
                "concurrency": 2,
                "context_size": 4096,
                "prompt_size": 2048,
                "response_size": 128,
                "tg_throughput": {"mean": 50, "std": 1, "values": [49, 51]},
                "tg_req_throughput": {"mean": 25, "std": 1, "values": [20, 25]},
                "e2e_ttft": {"mean": 900, "std": 5, "values": [895, 905]},
            }
        ]
    }
    c = parse_benchy_json(raw)[0]
    assert c["ttft_ms"]["mean"] == 900
    assert c["itl_ms"]["values"] == [50.0, 40.0]  # per request: 1000 / tg_req
    assert e2e.value(c, "itl_ms") == 45.0


def test_native_backends_on_the_mjolnir_image_never_count(tmp_path):
    """FlashInfer measured on the Mjolnir image is not native vLLM: it must
    neither stand in for the official side nor for Mjolnir."""
    h = tmp_path / "history.jsonl"
    rows = [
        _rec("vllm/vllm-openai:v0.30.0-ubuntu2404", "NVFP4", 1, [_cell(1, 8192, 20, 2000, 4000)]),
        _rec("narandill/mjolnir-vllm:v14", "NVFP4", 2, [_cell(1, 8192, 25, 2900, 3500)]),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h)
    assert off["image"].startswith("vllm/") and mj is None


def test_bench_e2e_only_runs_one_leg(monkeypatch):
    from typer.testing import CliRunner

    from mjolnir_cli import cli, dockerctl

    served = []
    monkeypatch.setattr(dockerctl, "serve_up", lambda s, **kw: served.append((s.image, s.quant)))
    runner = CliRunner()
    for side, want in (("mjolnir", "NVFP4_MJOLNIR"), ("official", "NVFP4")):
        served.clear()
        r = runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--only", side])
        assert r.exit_code == 0, r.output
        assert [q for _, q in served] == [want]
    served.clear()
    assert runner.invoke(cli.app, ["bench", "e2e", "--dry-run"]).exit_code == 0
    assert len(served) == 2
    assert runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--only", "fi"]).exit_code == 2
