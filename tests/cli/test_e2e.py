"""Official vLLM image vs Mjolnir image: pairing, gains, ITL/TTFT fallbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import mjolnir_cli.benchy as benchy
from mjolnir_cli import cli, dockerctl, e2e
from mjolnir_cli.config import ConfigEntry
from mjolnir_cli.history import parse_benchy_json

runner = CliRunner()


def _cell(conc, ctx, tg, pp, ttfr):
    return {
        "concurrency": conc,
        "context": ctx,
        "tg_tps": {"mean": tg},
        "pp_tps": {"mean": pp},
        "ttfr_ms": {"mean": ttfr},
    }


def _rec(image, config, epoch, cells, model="Qwen/Qwen3.8-27B", platform=None):
    rec = {"model": model, "image": image, "config": config, "epoch": epoch, "backend": "x", "cells": cells}
    if platform:
        rec["platform"] = platform
    return rec


def test_pair_prefers_mjolnir_backend_and_latest(tmp_path):
    h = tmp_path / "history.jsonl"
    rows = [
        _rec("vllm/vllm-openai:v0.30.0-ubuntu2404", "NVFP4", 1, [_cell(1, 8192, 20, 2000, 4000)]),
        _rec("vllm/vllm-openai:v0.30.0-ubuntu2404", "NVFP4", 3, [_cell(1, 8192, 22, 2100, 3900)]),
        _rec("narandill/mjolnir-vllm:v0.30.0", "NVFP4", 5, [_cell(1, 8192, 23, 2500, 3500)]),
        _rec("narandill/mjolnir-vllm:v0.30.0", "NVFP4_MJOLNIR", 4, [_cell(1, 8192, 30, 2600, 3200)]),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR")
    assert off and mj
    assert off["epoch"] == 3 and mj["config"] == "NVFP4_MJOLNIR"  # backend config beats a newer FI run
    hero = e2e.hero(off, mj)
    assert hero
    assert hero["ctx"] == 8192
    assert round(hero["tg_tps"][2], 1) == round((30 / 22 - 1) * 100, 1)
    # lower-is-better metrics: positive gain = the latency REDUCTION (official − Mjolnir) ÷ official,
    # not the reciprocal speedup official ÷ Mjolnir − 1; TTFT falls back to ttfr, ITL to 1000/tg at c=1
    assert round(hero["ttft_ms"][2], 1) == round((3900 - 3200) / 3900 * 100, 1) == 17.9
    assert round(e2e.change_pct(3900, 3200), 1) == -17.9
    assert round(hero["itl_ms"][1], 3) == round(1000 / 30, 3)


def test_infographic_row_is_the_one_smallest_measured_c1_context(tmp_path):
    """The infographic reads ONE row — C=1 at the smallest context both sides
    measured — for all four metrics: never the best-gain cell, never C=2 or
    the ctx=0 decode-only cell."""
    h = tmp_path / "history.jsonl"
    rows = [
        _rec(
            "vllm/vllm-openai:v0.32.0",
            "NVFP4",
            1,
            [
                _cell(1, 0, 20, 0, 800),
                _cell(1, 4096, 18, 2000, 4000),
                _cell(1, 8192, 15, 1900, 5000),  # higher decode gain than 4K — still not the row
                _cell(2, 8192, 90, 9000, 1000),
            ],
        ),
        _rec(
            "narandill/mjolnir-vllm:v0.32.0",
            "NVFP4_MJOLNIR",
            2,
            [
                _cell(1, 0, 30, 0, 600),
                _cell(1, 4096, 25, 5000, 3000),
                _cell(1, 8192, 25, 6000, 3600),
                _cell(2, 8192, 99, 9900, 900),
            ],
        ),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR")
    assert off and mj
    row = e2e.infographic_row(off, mj)
    assert row
    assert row["ctx"] == 4096
    assert row["tg_tps"][:2] == (18.0, 25.0)  # values from the 4K row, not 8K or ctx=0
    assert round(row["tg_tps"][2], 1) == round((25 / 18 - 1) * 100, 1)
    assert row["pp_tps"][:2] == (2000.0, 5000.0)
    assert round(row["pp_tps"][2], 1) == 150.0
    assert round(row["ttft_ms"][1], 3) == 3000.0  # TTFT falls back to ttfr like hero
    assert round(row["ttft_ms"][2], 1) == 25.0
    assert round(row["itl_ms"][1], 3) == round(1000 / 25, 3)  # ITL falls back to 1000/tg at c=1
    assert round(row["itl_ms"][2], 1) == 28.0


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
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR")
    assert off is not None and mj is None
    assert off["image"].startswith("vllm/")


def test_bench_e2e_only_runs_one_leg(monkeypatch):
    from mjolnir_cli import cli, dockerctl
    from mjolnir_cli import legs
    from typer.testing import CliRunner

    served = []
    monkeypatch.setattr(dockerctl, "serve_up", lambda s, **kw: served.append((s.image, s.quant)))
    runner = CliRunner()
    mj, off = legs.side_configs("bench.e2e", legs.MJOLNIR), legs.side_configs("bench.e2e", legs.OFFICIAL)
    for side, want in (("mjolnir", mj), ("official", off[:1])):
        served.clear()
        r = runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--only", side])
        assert r.exit_code == 0, r.output
        assert [q for _, q in served] == want
    served.clear()
    assert runner.invoke(cli.app, ["bench", "e2e", "--dry-run"]).exit_code == 0
    assert [q for _, q in served] == [off[0], *mj]  # the pair, then --also
    served.clear()
    assert runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--also", ""]).exit_code == 0
    assert len(served) == 2
    assert runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--only", "fi"]).exit_code == 2


def test_spec_label_from_records_and_configs():
    mtp3 = {"spec": {"method": "mtp", "tokens": 3}}
    assert e2e.spec_label(mtp3, dict(mtp3)) == "MTP, 3 speculative tokens"
    assert e2e.spec_label(mtp3, {"spec": {}}) == "MTP, 3 speculative tokens vs no speculative decoding"
    # older records carry no "spec": it is read from the config they name
    old = {"model": "Qwen/Qwen3.8-27B", "config": "NVFP4_MJOLNIR"}
    assert e2e.spec_label(old) == "MTP, 3 speculative tokens"


def test_e2e_cells_are_colored_by_the_band():
    from mjolnir_cli import readme

    assert e2e.E2E_NOISE_PCT == 3.0
    assert "#16a34a" in readme.ratio_pct(3.5) and "+3.5" in readme.ratio_pct(3.5)
    assert "grey" in readme.ratio_pct(2.9) and "grey" in readme.ratio_pct(-2.9)
    assert "#dc2626" in readme.ratio_pct(-3.1) and "−3.1" in readme.ratio_pct(-3.1)
    assert not any(ch in readme.ratio_pct(10) for ch in "✅🟰🔻")


def test_each_leg_stops_its_server(monkeypatch):
    from mjolnir_cli import benchy, dockerctl
    from mjolnir_cli.commands import bench
    from mjolnir_cli.commands._common import settings

    calls = []
    monkeypatch.setattr(dockerctl, "serve_down", lambda s, **kw: calls.append(("down", s.quant)))
    monkeypatch.setattr(dockerctl, "serve_up", lambda s, **kw: calls.append(("up", s.quant)))
    monkeypatch.setattr(benchy, "run_perf", lambda s, layout, **kw: calls.append(("bench", s.quant)))
    s = settings(None, None, None, None)
    bench.run_legs(s, None, [(s.model, "a", "NVFP4", None), (s.model, "b", "NVFP4_MJOLNIR", None)])
    assert calls == [
        ("down", "NVFP4"),
        ("up", "NVFP4"),
        ("bench", "NVFP4"),
        ("down", "NVFP4"),
        ("down", "NVFP4_MJOLNIR"),
        ("up", "NVFP4_MJOLNIR"),
        ("bench", "NVFP4_MJOLNIR"),
        ("down", "NVFP4_MJOLNIR"),
    ]

    def boom(s, layout, **kw):
        raise benchy.BenchError("llama-benchy failed")

    calls.clear()
    monkeypatch.setattr(benchy, "run_perf", boom)
    import pytest
    import typer

    with pytest.raises(typer.Exit):
        bench.run_legs(s, None, [(s.model, "a", "NVFP4", None)])
    assert calls[-1] == ("down", "NVFP4")  # stopped even when the bench fails


def test_runs_merge_cell_by_cell(tmp_path):
    """A later run at extra depths adds rows; the newest run wins shared cells."""
    h = tmp_path / "history.jsonl"
    off = "vllm/vllm-openai:v0.31.0-ubuntu2404"
    mjim = "narandill/mjolnir-vllm:v0.31.0"
    rows = [
        _rec(off, "NVFP4", 1, [_cell(1, 8192, 20, 2000, 4000), _cell(1, 32768, 18, 1700, 9000)]),
        _rec(mjim, "NVFP4_MJOLNIR", 2, [_cell(1, 8192, 22, 4000, 3000), _cell(1, 32768, 19, 3400, 5000)]),
        _rec(off, "NVFP4", 3, [_cell(1, 131072, 12, 900, 90000)]),
        _rec(mjim, "NVFP4_MJOLNIR", 4, [_cell(1, 131072, 14, 1800, 50000), _cell(1, 8192, 23, 4100, 2900)]),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    o, m = e2e.latest_pair(h, "NVFP4_MJOLNIR")
    assert o and m
    c8, c32 = e2e.cell(m, 1, 8192), e2e.cell(m, 1, 32768)
    hero = e2e.hero(o, m)
    assert c8 and c32 and hero
    assert sorted(c["context"] for c in o["cells"]) == [8192, 32768, 131072]
    assert c8["tg_tps"]["mean"] == 23  # newest run wins
    assert c32["tg_tps"]["mean"] == 19  # kept from the older run
    # the hero: 1 request, up to MAX_CTX, the cell with the highest decode gain (128K is past the cap)
    assert hero["ctx"] == 8192
    hero_all = e2e.hero(o, m, max_ctx=None)
    assert hero_all and hero_all["ctx"] in (8192, 131072)


def test_cells_only_one_side_measured_are_kept_and_flagged():
    def rec(ctxs):
        return {"cells": [{"concurrency": 1, "context": c, "tg_tps": {"mean": 20.0}} for c in ctxs]}

    off, mj = rec([0, 8192, 131072]), rec([0, 8192])
    assert [r["ctx"] for r in e2e.compare(off, mj)] == [0, 8192]  # the comparison: shared cells only
    full = e2e.compare(off, mj, every=True)
    assert [r["ctx"] for r in full] == [0, 8192]  # 128K is past MAX_CTX: not in the table
    assert e2e.missing_contexts(off, mj) == {"official": [], "mjolnir": []}
    full = e2e.compare(off, mj, every=True, max_ctx=None)
    assert [r["ctx"] for r in full] == [0, 8192, 131072] and full[-1]["tg_tps"][1] is None
    assert e2e.missing_contexts(off, mj, max_ctx=None) == {"official": [], "mjolnir": [131072]}


def test_history_survives_a_missing_final_newline(tmp_path):
    from mjolnir_cli import history

    h = tmp_path / "history.jsonl"
    h.write_text(json.dumps({"a": 1}))  # hand-edited: no final newline
    history.append_record(h, {"b": 2})
    assert h.read_text().count("\n") == 2
    h.write_text(json.dumps({"a": 1}) + json.dumps({"b": 2}) + "\n" + '{"c": ')  # glued + truncated
    assert history.load_records(h) == [{"a": 1}, {"b": 2}]


def test_no_context_has_no_prefill():
    from mjolnir_cli import e2e

    assert e2e.value({"context": 0, "pp_tps": {"mean": 2929.0}}, "pp_tps") is None
    assert e2e.value({"context": 4096, "pp_tps": {"mean": 2929.0}}, "pp_tps") == 2929.0
    assert e2e.value({"context": 0, "tg_tps": {"mean": 22.5}}, "tg_tps") == 22.5


def test_optional_legs_run_only_when_asked(monkeypatch):
    from mjolnir_cli import cli, dockerctl
    from mjolnir_cli import legs
    from mjolnir_cli.config import STOCK_IMAGE
    from typer.testing import CliRunner

    served = []
    monkeypatch.setattr(dockerctl, "serve_up", lambda s, **kw: served.append((s.image, s.quant)))
    runner = CliRunner()
    assert runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--also", ""]).exit_code == 0
    assert len(served) == 2  # no optional leg by default
    served.clear()
    r = runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--also", "", "--optional"])
    assert r.exit_code == 0, r.output
    opt = legs.OPTIONAL["bench.e2e"]
    assert [q for _, q in served[2:]] == [cfg for _, cfg in opt]
    for (img, _), (side, _) in zip(served[2:], opt, strict=True):
        assert (img == STOCK_IMAGE) == (side == legs.OFFICIAL)  # official legs stay on the upstream image
    served.clear()
    r = runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--only", "official", "--optional", "--also", ""])
    assert [q for _, q in served] == ["NVFP4", "NVFP4_DSPARK", "NVFP4_DFLASH2"]
    served.clear()
    r = runner.invoke(cli.app, ["bench", "e2e", "--dry-run", "--also", "official:NVFP4_DSPARK"])
    assert served[-1] == (STOCK_IMAGE, "NVFP4_DSPARK")


def test_optional_configs_exist_and_carry_their_drafter():
    from mjolnir_cli import legs
    from mjolnir_cli.config import config_speculative, configs_root

    want = {"DSPARK": "dspark", "DFLASH2": "dflash"}
    for _, cfg in legs.OPTIONAL["bench.e2e"]:
        path = configs_root() / "Qwen" / "Qwen3.8-27B" / cfg / "config.yaml"
        assert path.exists(), cfg
        assert config_speculative(path) == {"method": want[cfg.rsplit("_", 1)[-1]], "tokens": 7}
        assert not (path.parent / ".env").exists()  # DFlash2 shares the target's lm_head: no LMH8 / VT knobs
    assert legs.parse_leg("official:NVFP4_DSPARK") == ("official", "NVFP4_DSPARK")
    assert legs.parse_leg("NVFP4_MJOLNIR_E2M3_DSPARK") == ("mjolnir", "NVFP4_MJOLNIR_E2M3_DSPARK")


def test_a_newer_optional_official_leg_never_replaces_the_baseline(tmp_path):
    h = tmp_path / "history.jsonl"
    rows = [
        _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4", 1, [_cell(1, 8192, 20, 2000, 4000)]),
        _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4_DSPARK", 5, [_cell(1, 8192, 35, 2000, 4000)]),
        _rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR", 4, [_cell(1, 8192, 30, 2600, 3200)]),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, _ = e2e.latest_pair(h, "NVFP4_MJOLNIR")
    assert off and off["config"] == "NVFP4"
    off, _ = e2e.latest_pair(h, "NVFP4_MJOLNIR", "NVFP4_DSPARK")
    assert off and off["config"] == "NVFP4_DSPARK"


def test_pair_and_site_rows_never_mix_models(tmp_path):
    """Another model's run under a Mjolnir config name (Qwen3.6's NVFP4_MJOLNIR_E2M3)
    never lands next to the default model's runs: not in the README pair, not
    on the interactive site."""
    from types import SimpleNamespace
    from typing import cast

    from mjolnir_cli import charts
    from mjolnir_cli.config import RepoLayout

    h = tmp_path / "history.jsonl"
    q36 = {"model": "nvidia/Qwen3.6-35B-A3B"}
    rows = [
        _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4", 1, [_cell(1, 0, 20, 0, 100)]),
        _rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 2, [_cell(1, 0, 30, 0, 90)]),
        {**_rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4", 3, [_cell(1, 0, 80, 0, 50)]), **q36},
        {**_rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 4, [_cell(1, 0, 130, 0, 40)]), **q36},
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR_E2M3")
    assert off and mj
    assert off["epoch"] == 1 and mj["epoch"] == 2  # the default model's runs only
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR_E2M3", model="nvidia/Qwen3.6-35B-A3B")
    assert off and mj and off["epoch"] == 3 and mj["epoch"] == 4
    # the duck-typed layout stub e2e_rows only reads history_file
    site = charts.e2e_rows(cast(RepoLayout, SimpleNamespace(history_file=h)))
    assert {k: v["epoch"] for k, v in site.items()} == {"NVFP4": 1, "NVFP4_MJOLNIR_E2M3": 2}


def test_compare_pairs_the_requested_models_runs(tmp_path, monkeypatch, capsys):
    """`bench compare --model` (and `bench e2e`, which passes its own model) show
    that model's pair, not the default model's."""
    from mjolnir_cli.commands import bench_report

    h = tmp_path / "history.jsonl"
    q36 = {"model": "nvidia/Qwen3.6-35B-A3B"}
    rows = [
        _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4", 1, [_cell(1, 0, 20, 0, 100)]),
        _rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 2, [_cell(1, 0, 30, 0, 90)]),
        {**_rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4_DFLASH", 3, [_cell(1, 0, 80, 0, 50)]), **q36},
        {**_rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 4, [_cell(1, 0, 130, 0, 40)]), **q36},
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    layout = type("L", (), {"history_file": h})()
    assert bench_report.print_compare(layout, "NVFP4_MJOLNIR_E2M3", True, "NVFP4_DFLASH", q36["model"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cells"][0]["tg_tps"][:2] == [80.0, 130.0]  # Qwen3.6's pair, not Qwen3.8's 20 vs 30
    assert "NVFP4_DFLASH" in out["official"]


def test_a_run_without_model_is_never_the_default_models(tmp_path):
    """No default model: a history record without `model` pairs with nothing."""
    h = tmp_path / "history.jsonl"
    rows = [
        _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4", 1, [_cell(1, 0, 20, 0, 100)]),
        _rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 2, [_cell(1, 0, 30, 0, 90)], model=None),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR_E2M3")
    assert off is not None and mj is None
    assert off["epoch"] == 1


def test_compare_defaults_to_the_models_own_mjolnir_config(tmp_path):
    """Without --config, a model that has no DEFAULT_QUANT runs pairs its newest
    Mjolnir-image config instead of reporting no Mjolnir run."""
    h = tmp_path / "history.jsonl"
    q36 = "nvidia/Qwen3.6-35B-A3B"
    rows = [
        _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4_DFLASH", 1, [_cell(1, 0, 80, 0, 50)], model=q36),
        _rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 2, [_cell(1, 0, 130, 0, 40)], model=q36),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h, model=q36)
    assert off and mj
    assert off["config"] == "NVFP4_DFLASH" and mj["config"] == "NVFP4_MJOLNIR_E2M3"


# ── the platform dimension (plan §2 scenarios 34-38) ────────────────────────


def test_latest_pair_platform_filter_ignores_other_platforms_rows(tmp_path):
    h = tmp_path / "history.jsonl"
    rows = [
        _rec("vllm/vllm-openai:v0.32.0-ubuntu2404", "NVFP4", 1, [_cell(1, 8192, 20, 2000, 4000)], platform="spark"),
        _rec("narandill/mjolnir-vllm:v0.32.0", "NVFP4_MJOLNIR", 2, [_cell(1, 8192, 30, 2600, 3200)], platform="spark"),
        _rec("vllm/vllm-openai:v0.32.0-ubuntu2404", "NVFP4", 3, [_cell(1, 8192, 22, 2100, 3900)]),  # thor
        _rec("narandill/mjolnir-vllm:v0.32.0", "NVFP4_MJOLNIR", 4, [_cell(1, 8192, 32, 2700, 3100)]),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR", platform="thor")
    assert off and mj
    assert off["epoch"] == 3 and mj["epoch"] == 4  # the spark rows with the same config are ignored


def test_latest_pair_never_pairs_across_platform_boundaries(tmp_path):
    h = tmp_path / "history.jsonl"
    rows = [
        _rec("vllm/vllm-openai:v0.32.0-ubuntu2404", "NVFP4", 1, [_cell(1, 8192, 20, 2000, 4000)], platform="spark"),
        _rec("narandill/mjolnir-vllm:v0.32.0", "NVFP4_MJOLNIR", 2, [_cell(1, 8192, 30, 2600, 3200)], platform="spark"),
        _rec("vllm/vllm-openai:v0.32.0-ubuntu2404", "NVFP4", 3, [_cell(1, 8192, 22, 2100, 3900)]),  # thor
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR", platform="thor")
    assert off is not None and off["epoch"] == 3
    assert mj is None  # the spark Mjolnir row never stands in for a thor one
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR", platform="spark")
    assert off is not None and off["epoch"] == 1
    assert mj is not None and mj["epoch"] == 2  # and the thor rows are invisible to spark


def test_legacy_platform_less_rows_pair_as_thor(tmp_path):
    # the repo's committed history has no platform field: it pairs as thor
    h = tmp_path / "history.jsonl"
    rows = [
        _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4", 1, [_cell(1, 8192, 20, 2000, 4000)]),
        _rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR", 2, [_cell(1, 8192, 30, 2600, 3200)]),
    ]
    h.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off, mj = e2e.latest_pair(h, "NVFP4_MJOLNIR", platform="thor")
    assert off and mj and off["epoch"] == 1 and mj["epoch"] == 2


def test_is_mjolnir_recognizes_registry_image_repos_and_legacy_names():
    import mjolnir_cli.platforms as platforms

    for key in ("thor", "spark", "orin"):
        repo = platforms.REGISTRY[key].image_repo
        assert e2e.is_mjolnir({"image": f"{repo}:v1"}), key
    assert e2e.is_mjolnir({"image": "mjolnir/vllm-thor:old"})  # the legacy names keep classifying
    assert e2e.is_mjolnir({"image": "narandill/mjolnir-vllm:v0.32.0"})


def test_is_official_is_true_for_a_stock_image_on_any_platform():
    import mjolnir_cli.platforms as platforms

    for key in ("thor", "spark", "orin"):
        stock = platforms.REGISTRY[key].stock_image
        assert stock and e2e.is_official({"image": stock}), key  # official = the stock image, any platform
    assert not e2e.is_official({"image": "narandill/mjolnir-vllm:v0.32.0"})


# ── the bench command bodies (perf / ab / compare) and the report helpers ────


@pytest.fixture
def _no_serving(monkeypatch, tmp_path):
    """The leg machinery, faked at the docker/benchy boundary: nothing ever
    touches docker, nothing measures, the re-render is observed not run."""
    import mjolnir_cli.readme as readme_mod
    from mjolnir_cli import charts

    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "state.json"))  # the baked-in defaults
    seen: dict[str, Any] = {"perf": [], "up": [], "down": [], "readme": 0, "charts": 0}
    monkeypatch.setattr(dockerctl, "stale_image", lambda image: None)
    monkeypatch.setattr(dockerctl, "serve_down", lambda s, **kw: seen["down"].append(s.quant))
    monkeypatch.setattr(dockerctl, "serve_up", lambda s, **kw: seen["up"].append(s.quant))
    monkeypatch.setattr(dockerctl, "container_running", lambda: False)

    def fake_perf(s, layout, **kw):
        seen["perf"].append(s.quant)

    monkeypatch.setattr(benchy, "run_perf", fake_perf)

    def fake_charts(layout):
        seen["charts"] += 1

    monkeypatch.setattr(charts, "render_all", fake_charts)
    monkeypatch.setattr(readme_mod, "render_readme", lambda *a, **kw: seen.__setitem__("readme", seen["readme"] + 1))
    return seen


def test_bench_perf_runs_and_rerenders(_no_serving):
    r = runner.invoke(cli.app, ["bench", "perf", "--depths", "4096", "--concurrency", "1", "--repeat", "2"])
    assert r.exit_code == 0, r.output
    assert _no_serving["perf"] == ["NVFP4_MJOLNIR_E2M3_LMH8_VT64K"]  # the baked-in default config
    assert _no_serving["readme"] == 1 and _no_serving["charts"] == 1


def test_bench_perf_without_plot_touches_neither_renderer(_no_serving):
    r = runner.invoke(cli.app, ["bench", "perf", "--no-plot"])
    assert r.exit_code == 0 and _no_serving["readme"] == 0 and _no_serving["charts"] == 0


def test_bench_perf_names_the_bench_error(monkeypatch):
    def boom(s, layout, **kw):
        raise benchy.BenchError("the server never went healthy")

    monkeypatch.setattr(benchy, "run_perf", boom)
    r = runner.invoke(cli.app, ["bench", "perf"])
    assert r.exit_code == 1 and "the server never went healthy" in r.output


def test_bench_ab_serves_each_leg_and_stops_it(_no_serving):
    r = runner.invoke(cli.app, ["bench", "ab", "--backends", "NVFP4,NVFP4_MJOLNIR_E2M3"])
    assert r.exit_code == 0, r.output
    assert _no_serving["perf"] == ["NVFP4", "NVFP4_MJOLNIR_E2M3"]
    # every leg was stopped after its bench (run_legs' finally)
    assert _no_serving["down"] == ["NVFP4", "NVFP4", "NVFP4_MJOLNIR_E2M3", "NVFP4_MJOLNIR_E2M3"]
    assert _no_serving["readme"] == 1  # the post-run re-render


def test_bench_ab_rejects_bad_legs(_no_serving):
    for bad in (":", "NOPE", "vendor/Missing/NOPE", ""):
        r = runner.invoke(cli.app, ["bench", "ab", "--backends", bad])
        assert r.exit_code == 4, (bad, r.output)
        assert "mjolnir model list" in r.output or "bad leg" in r.output or "--backends is empty" in r.output


def test_ab_legs_error_messages():
    from mjolnir_cli.commands.bench import _ab_legs

    entries = [
        ConfigEntry(model="vendor/M1", quant="A", path=Path("/x")),
        ConfigEntry(model="vendor/M2", quant="B", path=Path("/x")),
    ]
    with pytest.raises(ValueError, match="--backends is empty"):
        _ab_legs("", "vendor/M1", "img", entries)
    with pytest.raises(ValueError, match="unknown config 'NOPE' for model vendor/M1"):
        _ab_legs("NOPE", "vendor/M1", "img", entries)
    with pytest.raises(ValueError, match="bad leg 'M1/NOPE'"):  # the model part needs vendor/Model
        _ab_legs("M1/NOPE", "vendor/M1", "img", entries)
    with pytest.raises(ValueError, match="unknown model config 'vendor/Missing/NOPE'"):
        _ab_legs("vendor/Missing/NOPE", "vendor/M1", "img", entries)
    with pytest.raises(ValueError, match="bad leg 'a/b/c/d'"):  # two slashes in the model part
        _ab_legs("a/b/c/d", "vendor/M1", "img", entries)
    # the happy shapes
    assert _ab_legs("A,img:A,vendor/M2/B,img:vendor/M2/B", "vendor/M1", "img", entries) == [
        ("vendor/M1", "img", "A"),
        ("vendor/M1", "img", "A"),
        ("vendor/M2", "img", "B"),
        ("vendor/M2", "img", "B"),
    ]


def test_bench_compare_wires_the_history_pair(monkeypatch, tmp_path):
    from mjolnir_cli.commands import bench

    seen = {}
    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "state.json"))
    monkeypatch.setattr(bench, "load_layout", lambda: object())

    def fake_compare(layout, config=None, as_json=False, official_config=None, model=None):
        seen.update(config=config, as_json=as_json, official_config=official_config)
        return 0

    monkeypatch.setattr(bench, "print_compare", fake_compare)
    r = runner.invoke(cli.app, ["bench", "compare", "--config", "C1", "--official-config", "C0", "--json"])
    assert r.exit_code == 0 and seen == {"config": "C1", "as_json": True, "official_config": "C0"}

    monkeypatch.setattr(bench, "print_compare", lambda *a, **kw: 4)
    r = runner.invoke(cli.app, ["bench", "compare"])
    assert r.exit_code == 4  # print_compare's "no run" verdict becomes the exit


def test_print_compare_renders_the_table_from_a_real_history(tmp_path, capsys):
    from mjolnir_cli.commands.bench_report import print_compare

    h = tmp_path / "history.jsonl"
    off = _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4", 1, [_cell(1, 0, 20, 0, 100)])
    mj = _rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 2, [_cell(1, 0, 30, 0, 90)])
    h.write_text("".join(json.dumps(r) + "\n" for r in [off, mj]))
    layout = type("L", (), {"history_file": h})()
    assert print_compare(layout) == 0
    out = capsys.readouterr().out
    assert "official vLLM image → Mjolnir image" in out
    assert "20.0 → 30.0" in out  # the decode cell, gain colored in


def test_print_compare_with_no_pair_names_the_missing_side(tmp_path, capsys):
    from mjolnir_cli.commands.bench_report import print_compare

    h = tmp_path / "history.jsonl"
    h.write_text("")
    layout = type("L", (), {"history_file": h})()
    assert print_compare(layout) == 4
    assert "no run on the official vLLM image" in capsys.readouterr().out


def test_rerender_swallows_renderer_failures(monkeypatch, capsys):
    from mjolnir_cli import charts
    import mjolnir_cli.readme as readme_mod
    from mjolnir_cli.commands.bench_report import rerender

    def charts_boom(layout):
        raise RuntimeError("no data")

    def readme_boom(*a, **kw):
        raise FileNotFoundError("no template")

    monkeypatch.setattr(charts, "render_all", charts_boom)
    monkeypatch.setattr(readme_mod, "render_readme", readme_boom)
    rerender(type("L", (), {})())
    err = capsys.readouterr().err
    assert "(plots skipped: no data)" in err and "(readme skipped: no template)" in err


def test_bench_perf_reports_skipped_renderers(monkeypatch, tmp_path):
    from mjolnir_cli import charts
    import mjolnir_cli.readme as readme_mod

    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "state.json"))
    monkeypatch.setattr(benchy, "run_perf", lambda s, layout, **kw: None)
    monkeypatch.setattr(charts, "render_all", lambda layout: (_ for _ in ()).throw(RuntimeError("no data")))
    monkeypatch.setattr(
        readme_mod, "render_readme", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("no template"))
    )
    r = runner.invoke(cli.app, ["bench", "perf"])
    assert r.exit_code == 0, r.output
    assert "(plots skipped: no data)" in r.output
    assert "(readme skipped: no template)" in r.output


def _fake_accuracy_doc(tmp_path: Path) -> Path:
    out = tmp_path / "accuracy.json"
    out.write_text(
        json.dumps(
            {
                "run_id": "r1",
                "args": ["run"],
                "legs": [{"id": "off", "label": "off", "suites": {"tools": {"score": 90.0}}}],
            }
        )
    )
    return out


def test_bench_accuracy_runs_prints_and_rerenders(monkeypatch, tmp_path, _no_serving):
    import mjolnir_cli.accuracy as accuracy_mod

    out = _fake_accuracy_doc(tmp_path)
    seen = {}
    monkeypatch.setattr(
        accuracy_mod, "run_accuracy", lambda s, layout, legs, passthrough, restart: seen.update(legs=legs) or out
    )
    r = runner.invoke(cli.app, ["bench", "accuracy", "--backends", "NVFP4"])
    assert r.exit_code == 0, r.output
    assert "restarts the vLLM server for each leg" in r.output  # the deliberate-restart warning
    assert "wrote" in r.output and _no_serving["readme"] == 1


def test_bench_accuracy_dry_run_prints_the_tool_argv(monkeypatch, _no_serving):
    import mjolnir_cli.accuracy as accuracy_mod

    monkeypatch.setattr(accuracy_mod, "find_tool", lambda: ["tool-eval-bench"])
    r = runner.invoke(cli.app, ["bench", "accuracy", "--backends", "NVFP4", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "tool-eval-bench run --hardmode --seed 42 --timeout 600" in r.output


def test_bench_accuracy_dry_run_names_a_missing_tool(monkeypatch, _no_serving):
    import mjolnir_cli.accuracy as accuracy_mod

    monkeypatch.setattr(
        accuracy_mod, "find_tool", lambda: (_ for _ in ()).throw(accuracy_mod.AccuracyError("uvx not found"))
    )
    r = runner.invoke(cli.app, ["bench", "accuracy", "--backends", "NVFP4", "--dry-run"])
    assert r.exit_code == 0 and "<uvx not found>" in r.output


def test_bench_accuracy_no_restart_needs_a_running_server(monkeypatch, tmp_path, _no_serving):
    r = runner.invoke(cli.app, ["bench", "accuracy", "--backends", "NVFP4", "--no-restart"])
    assert r.exit_code == 3 and "no vLLM server is running" in r.output


def test_bench_accuracy_names_the_accuracy_error(monkeypatch, tmp_path, _no_serving):
    import mjolnir_cli.accuracy as accuracy_mod

    def boom(s, layout, legs, passthrough, restart):
        raise accuracy_mod.AccuracyError("tool crashed")

    monkeypatch.setattr(accuracy_mod, "run_accuracy", boom)
    r = runner.invoke(cli.app, ["bench", "accuracy", "--backends", "NVFP4"])
    assert r.exit_code == 1 and "tool crashed" in r.output


def test_bench_e2e_full_run_serves_compares_and_rerenders(monkeypatch, tmp_path):
    from mjolnir_cli.commands import bench

    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "state.json"))
    seen: dict[str, Any] = {"up": [], "down": [], "perf": [], "readme": 0}
    monkeypatch.setattr(dockerctl, "stale_image", lambda image: None)
    monkeypatch.setattr(dockerctl, "serve_down", lambda s, **kw: seen["down"].append(s.quant))
    monkeypatch.setattr(dockerctl, "serve_up", lambda s, **kw: seen["up"].append(s.quant))

    def fake_perf(s, layout, **kw):
        seen["perf"].append(s.quant)

    monkeypatch.setattr(benchy, "run_perf", fake_perf)

    def fake_compare(layout, config=None, as_json=False, official_config=None, model=None):
        seen["compared"] = (config, official_config)
        return 0

    monkeypatch.setattr(bench, "print_compare", fake_compare)
    import mjolnir_cli.readme as readme_mod

    monkeypatch.setattr(readme_mod, "render_readme", lambda *a, **kw: seen.__setitem__("readme", seen["readme"] + 1))
    r = runner.invoke(cli.app, ["bench", "e2e", "--depths", "4096", "--concurrency", "1"])
    assert r.exit_code == 0, r.output
    assert "restarts the vLLM server for each image" in r.output
    assert seen["perf"] and seen["readme"] == 1
    assert seen["compared"][0] is not None  # the Mjolnir-side config reached the comparison


def test_bench_e2e_refuses_a_stale_image_unless_allowed(monkeypatch, tmp_path):
    from mjolnir_cli.commands import bench

    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "state.json"))
    monkeypatch.setattr(dockerctl, "stale_image", lambda image: "image is behind the checkout")
    monkeypatch.setattr(bench, "print_compare", lambda *a, **kw: 0)
    r = runner.invoke(cli.app, ["bench", "e2e"])
    assert r.exit_code == 2 and "(or --allow-stale-image)" in r.output
    # with the flag: the warning is printed and the run proceeds
    import mjolnir_cli.readme as readme_mod

    monkeypatch.setattr(dockerctl, "serve_down", lambda s, **kw: None)
    monkeypatch.setattr(dockerctl, "serve_up", lambda s, **kw: None)
    monkeypatch.setattr(benchy, "run_perf", lambda s, layout, **kw: None)
    monkeypatch.setattr(readme_mod, "render_readme", lambda *a, **kw: None)
    r = runner.invoke(cli.app, ["bench", "e2e", "--allow-stale-image"])
    assert r.exit_code == 0 and "warning: image is behind the checkout" in r.output


def test_print_compare_prints_the_hero_line(tmp_path, capsys):
    from mjolnir_cli.commands.bench_report import print_compare

    h = tmp_path / "history.jsonl"
    cells = [_cell(1, 4096, 20, 100, 100), _cell(4, 4096, 60, 200, 90)]
    off = _rec("vllm/vllm-openai:v0.31.0-ubuntu2404", "NVFP4", 1, cells)
    mj = _rec("narandill/mjolnir-vllm:v0.31.0", "NVFP4_MJOLNIR_E2M3", 2, cells)
    h.write_text("".join(json.dumps(r) + "\n" for r in [off, mj]))
    layout = type("L", (), {"history_file": h})()
    assert print_compare(layout) == 0
    out = capsys.readouterr().out
    assert "1 request @ 4K:" in out  # the infographic line (both sides measured c=1 decode+prefill)
    assert "decode" in out and "TTFT" in out


def test_bench_accuracy_rejects_bad_legs(_no_serving):
    r = runner.invoke(cli.app, ["bench", "accuracy", "--backends", "NOPE", "--dry-run"])
    assert r.exit_code == 4 and "unknown config 'NOPE'" in r.output


def test_bench_accuracy_no_restart_takes_exactly_one_leg(_no_serving):
    r = runner.invoke(cli.app, ["bench", "accuracy", "--backends", "NVFP4,NVFP4_FA4hd256", "--no-restart"])
    assert r.exit_code == 2 and "exactly one leg" in r.output


def test_bench_e2e_suggests_how_to_compare_the_optional_legs(monkeypatch, tmp_path):
    from mjolnir_cli.commands import bench

    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "state.json"))
    monkeypatch.setattr(dockerctl, "stale_image", lambda image: None)
    monkeypatch.setattr(dockerctl, "serve_down", lambda s, **kw: None)
    monkeypatch.setattr(dockerctl, "serve_up", lambda s, **kw: None)
    monkeypatch.setattr(benchy, "run_perf", lambda s, layout, **kw: None)
    monkeypatch.setattr(bench, "print_compare", lambda *a, **kw: 0)
    import mjolnir_cli.readme as readme_mod

    monkeypatch.setattr(readme_mod, "render_readme", lambda *a, **kw: None)
    r = runner.invoke(cli.app, ["bench", "e2e", "--also", "NVFP4_MJOLNIR"])
    assert r.exit_code == 0, r.output
    assert "optional legs — compare them with:" in r.output
    assert "mjolnir bench compare --model" in r.output
