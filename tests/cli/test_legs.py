"""mjolnir_cli.legs is the single source of every default leg."""

from __future__ import annotations

import inspect

from mjolnir_cli import accuracy, legs
from mjolnir_cli.config import STOCK_IMAGE
from mjolnir_cli.verify import kvcache, profile, spec, stages


def test_forms():
    legs.SERVING["_t"] = (("official", "NVFP4"), ("mjolnir", "NVFP4_MJOLNIR_E2M3"))
    try:
        assert legs.verify_legs("_t") == "official:NVFP4,mjolnir:NVFP4_MJOLNIR_E2M3"
        assert legs.bench_legs("_t") == f"{STOCK_IMAGE}:NVFP4,NVFP4_MJOLNIR_E2M3"
        assert legs.side_configs("_t", legs.MJOLNIR) == ["NVFP4_MJOLNIR_E2M3"]
    finally:
        del legs.SERVING["_t"]


def test_no_duplicate_legs():
    for name, ls in legs.SERVING.items():
        assert len(ls) == len(set(ls)), name
        assert all(side in (legs.OFFICIAL, legs.MJOLNIR) for side, _ in ls), name


def test_stages_and_commands_read_the_table():
    assert legs.verify_legs("verify.spec") == spec.DEFAULT_LEGS
    assert legs.verify_legs("verify.kv-cache-size") == kvcache.DEFAULT_LEGS
    assert legs.verify_legs("verify.profile") == profile.DEFAULT_LEGS
    assert legs.KERNEL["verify.sweep"] == stages.SWEEP_LEGS
    assert inspect.signature(stages.stage_hndsweep).parameters["legs"].default == legs.KERNEL["verify.hndsweep"]
    assert inspect.signature(stages.stage_vllm).parameters["smoke_modes"].default == legs.SMOKE_MODES
    assert not hasattr(accuracy, "DEFAULT_LEGS")


def test_bench_defaults_follow_the_table():
    from typer.testing import CliRunner

    from mjolnir_cli import cli

    for cmd, name in (("ab", "bench.ab"), ("accuracy", "bench.accuracy")):
        out = CliRunner().invoke(cli.app, ["bench", cmd, "--help"], env={"COLUMNS": "400"}).output
        assert legs.bench_legs(name) in out, cmd
