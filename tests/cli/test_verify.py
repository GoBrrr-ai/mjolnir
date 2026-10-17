"""`mjolnir verify` plumbing: ncu log parsing, legs, result storage and the
headline-run selection the charts depend on — no GPU, no docker."""

from __future__ import annotations

import json

import pytest
from mjolnir_cli.verify import legs, ncu, results, stages

NCU_LOG = """\
==PROF== Disconnected from process 1
  MixedInputFusedMultiHeadAttentionDecode_0x1 (40, 1, 1)x(384, 1, 1), Context 1, Stream 7, Device 0, CC 11.0, Invocations 50
    Section: Command line profiler metrics
    ---------------------------------------- ----------- ------------ ------------ ------------
    Metric Name                              Metric Unit Minimum      Maximum      Average
    gpu__time_duration.sum                   usecond     600.10       700.20       640.50
    lts__t_sectors.sum                       sector      3,100,000    3,200,000    3,171,000
    sm__ctas_launched.sum                    block       40           40           40
  kernel_cutlass_reduction_0x2 (24, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 11.0, Invocations 50
    Section: Command line profiler metrics
    gpu__time_duration.sum                   nsecond     6,900        7,800        7,330
    lts__t_sectors.sum                       sector      1,000        1,000        1,000
    sm__ctas_launched.sum                    block       24           24           24
"""


def test_parse_summary_reads_units_and_thousands():
    rows = ncu.parse_summary(NCU_LOG)
    assert [r["inv"] for r in rows] == [50, 50]
    assert rows[0]["avg_us"] == pytest.approx(640.5)
    assert rows[1]["avg_us"] == pytest.approx(7.33)  # ns -> µs
    assert rows[0]["sectors"] == 3_171_000


def test_point_splits_decode_and_merge_and_counts_real_bytes():
    L = 65536
    leg = legs.MJOLNIR  # e2m3: 6 bits/element
    kv_bytes = L * 4 * 256 * 2 * leg.kv_bits / 8
    pt = ncu.point(ncu.parse_summary(NCU_LOG), legs.MAIN["t5"], legs.MERGE["t5"], L, kv_bytes)
    assert pt["main_us"] == pytest.approx(640.5)
    assert pt["merge_us"] == pytest.approx(7.33)
    assert pt["total_us"] == pytest.approx(647.83)
    assert pt["ctas"] == 40
    assert pt["gbps"] == pytest.approx(kv_bytes / (647.83 * 1e3), rel=1e-3)


def test_point_without_main_kernel_is_none():
    assert ncu.point(ncu.parse_summary(NCU_LOG), ("BatchDecode",), ("MergeStates",), 8192, 1.0) is None


def test_legs_kv_format_follows_the_last_flag_and_harness_default():
    assert legs.MJOLNIR.kv_format == "e2m3" and legs.MJOLNIR.kv_layout == "hnd"
    assert legs.MJOLNIR_FP8.kv_format == "fp8" and legs.MJOLNIR_FP8.kv_bits == 8
    v = legs.t5_variant("x", "--kv_format fp8 --kv_format e2m1 --kv_layout nhd")
    assert (v.kv_format, v.kv_layout, v.kv_bits) == ("e2m1", "nhd", 4.0)
    assert legs.t5_variant("bf16q", "", q="bf16").kv_format == "fp8"  # narrow formats need fp8 Q
    assert legs.FA2.kv_bits == 16 and legs.FI_HND.kv_layout == "hnd"


def test_parse_variants_and_unknown_legs():
    vs = legs.parse_variants("a:--kv_stages 3| b:--pdl |")
    assert [(v.id, v.flags) for v in vs] == [("t5:a", "--kv_stages 3"), ("t5:b", "--pdl")]
    with pytest.raises(ValueError, match="unknown leg"):
        legs.legs_from_ids("fi,nope")


def _doc(stage, run_id, default, series=None, **kw):
    d = results.header(stage, run_id, image="img", default=default, params={}, kernel={}, gate={})
    d["series"] = series or []
    d.update(kw)
    return d


def test_latest_prefers_default_runs(tmp_path):
    results.save(tmp_path, _doc("sweep", "20261010-100000", True))
    results.save(tmp_path, _doc("sweep", "20261010-110000", False))
    assert results.latest(tmp_path, "sweep")["run_id"] == "20261010-100000"
    assert results.latest(tmp_path, "sweep", default_only=False)["run_id"] == "20261010-110000"
    assert results.latest(tmp_path, "ring") is None


def test_foreign_json_is_ignored(tmp_path):
    (tmp_path / "sweep").mkdir()
    (tmp_path / "sweep" / "x.json").write_text(json.dumps({"schema": "other"}))
    assert results.runs(tmp_path, "sweep") == []


def _series(sid, pts):
    return {"id": sid, "label": sid, "kind": "t5", "points": [{"L": L, "total_us": t} for L, t in pts.items()]}


def test_summarize_speedups_vs_each_baseline():
    doc = _doc(
        "sweep",
        "r",
        True,
        series=[
            _series("fi", {8192: 100.0, 65536: 800.0}),
            _series("fa2", {8192: 200.0, 65536: 1600.0}),
            _series("mjolnir", {8192: 80.0, 65536: 640.0}),
        ],
    )
    s = stages.summarize("sweep", doc)
    assert s["vs_fi"]["geomean"] == pytest.approx(1.25)
    assert s["vs_fa2"]["min"] == pytest.approx(2.5)
    assert s["vs_fi"]["per_L"] == {"8192": 1.25, "65536": 1.25}


def test_correctness_cases_pin_the_vllm_config_unless_default():
    pinned = stages.case_args("--seqlen 8192 --kv_layout hnd")
    assert pinned.startswith("--kv_format fp8 --kv_layout nhd") and pinned.endswith("--kv_layout hnd")
    assert stages.case_args("@default --seqlen 8192") == "--seqlen 8192"
    labels = [c[0] for c in stages.CASES]
    assert len(labels) == len(set(labels)), "case labels must be unique (log file names)"


def test_every_stage_defaults_to_the_full_L_axis():
    assert stages.ALL_LS == (128, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
    assert set(stages.ALL_ORDER) <= set(stages.STAGES)


def test_raw_csv_with_text_only_columns_does_not_crash():
    from mjolnir_cli.verify import ncu

    text = (
        '"ID","Kernel Name","gpu__time_duration.sum","note"\n'
        '"","","usecond",""\n'
        '"0","decode","12.5","n/a"\n'
        '"1","decode","13.5","n/a"\n'
    )
    mean, units, best_id = ncu.parse_raw_csv(text)
    # identical launches flip up to ~25% under replay: the fastest launch is
    # the reported snapshot, not the mean
    assert mean["gpu__time_duration.sum"] == 12500.0  # µs -> ns, fastest launch
    assert best_id == 0
    assert "note" not in mean and "Kernel Name" not in mean


def test_details_csv_keeps_only_the_fastest_launch():
    from mjolnir_cli.verify import ncu

    text = (
        '"ID","Section Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","Warp State Statistics","Stall Long Scoreboard","cycle/instruction","10"\n'
        '"1","Warp State Statistics","Stall Long Scoreboard","cycle/instruction","40"\n'
    )
    d = ncu.parse_details_csv(text, best_id=0)
    assert d["Warp State Statistics"]["Stall Long Scoreboard (cycle/instruction)"] == 10.0
    assert ncu.parse_details_csv(text)["Warp State Statistics"]["Stall Long Scoreboard (cycle/instruction)"] == 25.0


def test_sweep_batch_and_q_len_reach_both_drivers(tmp_path):
    from mjolnir_cli.verify.docker import Runner

    r = Runner(image="img", logdir=tmp_path, batch=4, q_len=4)
    cmd, _ = r.leg_command(legs.legs_from_ids("mjolnir-fp8")[0], 8192, 10)
    assert "--batches 4" in cmd and "--q_len 4" in cmd
    _, env = r.leg_command(legs.legs_from_ids("fi")[0], 8192, 10)
    assert "BENCH_B=4" in env and "BENCH_QLEN=4" in env
    one = Runner(image="img", logdir=tmp_path)
    cmd, _ = one.leg_command(legs.legs_from_ids("mjolnir-fp8")[0], 8192, 10)
    assert "--batches" not in cmd and "--q_len" not in cmd  # the default command is unchanged


def test_sweep_rejects_fa2_for_batched_shapes(tmp_path):
    from types import SimpleNamespace

    ctx = SimpleNamespace(runner=SimpleNamespace(batch=1, q_len=1))
    with pytest.raises(ValueError, match="fa2"):
        stages.stage_sweep(ctx, legs="fi,fa2", batch=4, q_len=4)
