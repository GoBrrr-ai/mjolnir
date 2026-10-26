"""mjolnir_cli.legs is the single source of every default leg.

The flat ``SERVING``/``OPTIONAL``/``KERNEL`` tables stay the *default
platform* (thor) contract — pinned at import time by
``test_stages_and_commands_read_the_table`` below. The per-platform layer is
resolved at call time beside them: ``legs_for_stage(stage, platform)``,
``kernel_legs(stage, platform)`` and ``available_serving_legs(...)``, which
since cycle 2 is overlay-aware: a config exists on a platform only when its
``platforms/<platform>.yaml`` overlay is there (no ``platforms/`` dir =
legacy thor-only), and ``bench e2e`` / ``verify`` re-resolve their default
legs on a non-default platform the way ``bench ab``/``accuracy`` always did.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from mjolnir_cli import accuracy, legs
from mjolnir_cli.config import DEFAULT_IMAGE, STOCK_IMAGE
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


# ── per-platform resolution (plan §2 scenarios 29-33) ───────────────────────

# today's thor legs, pinned as literal strings: the per-platform layer must
# never change what the default platform measures
THOR_SPEC_LEGS = "official:NVFP4,mjolnir:NVFP4_MJOLNIR,mjolnir:NVFP4_MJOLNIR_E2M3,mjolnir:NVFP4_MJOLNIR_E2M3_LMH8_VT64K"
THOR_AB_LEGS = f"{STOCK_IMAGE}:NVFP4,NVFP4_MJOLNIR_E2M3_LMH8_VT64K"


def test_thor_legs_are_unchanged():
    assert legs.verify_legs("verify.spec") == THOR_SPEC_LEGS
    assert legs.bench_legs("bench.ab") == THOR_AB_LEGS
    assert legs.side_configs("bench.e2e", legs.OFFICIAL) == ["NVFP4"]
    assert legs.side_configs("bench.e2e", legs.MJOLNIR) == ["NVFP4_MJOLNIR_E2M3_LMH8_VT64K"]


def test_legs_for_stage_defaults_to_the_thor_table():
    for name in legs.SERVING:
        assert legs.legs_for_stage(name) == legs.SERVING[name], name


@pytest.mark.parametrize(
    ("platform", "baseline"),
    [("spark", "NVFP4")],
)
def test_legs_for_stage_research_platforms_measure_their_own_baseline(platform, baseline):
    # research platforms measure baselines only until their kernels land:
    # the official side only — exactly one leg, the platform's own baseline
    # config (spark's is the thor table's NVFP4 baseline). The orin names its
    # own serving table instead (it serves the AWQ family), so it is not here.
    for name in legs.SERVING:
        assert legs.legs_for_stage(name, platform) == ((legs.OFFICIAL, baseline),), name


def test_legs_for_stage_orin_serves_its_own_family():
    # the orin's override table: the AWQ family for every stage, with the
    # native-FlashInfer config as the stock-backend baseline
    for name in legs.SERVING:
        table = legs.legs_for_stage(name, "orin")
        assert table, name
        assert all(cfg.startswith("AWQ_") for _, cfg in table), (name, table)
    spec = legs.legs_for_stage("verify.spec", "orin")
    assert spec[0] == (legs.OFFICIAL, "AWQ_FLASHINFER")
    assert (legs.MJOLNIR, "AWQ_MJOLNIR") in spec
    assert (legs.MJOLNIR, "AWQ_MJOLNIR_E2M3") in spec
    assert (legs.MJOLNIR, "AWQ_MJOLNIR_E2M3_VT64K") in spec


def test_available_serving_legs_resolve_the_orin_baseline_from_the_repo_configs():
    # the real tree: the orin table's configs exist for the model (their
    # overlays), so they survive the existence filter; a thor-only config
    # (NVFP4_*) would be filtered out on orin
    got = legs.available_serving_legs("bench.ab", "orin", "Qwen/Qwen3.8-27B")
    assert got == legs.legs_for_stage("bench.ab", "orin")
    assert all("NVFP4" not in cfg for _, cfg in got)


def test_bench_and_verify_legs_render_the_orin_family():
    # the official leg pins the platform's stock image; the Mjolnir legs are bare
    assert legs.bench_legs("bench.ab", "orin") == (
        f"{STOCK_IMAGE}:AWQ_FLASHINFER,AWQ_MJOLNIR_E2M3,AWQ_MJOLNIR_E2M3_VT64K"
    )
    assert legs.verify_legs("verify.spec", "orin") == (
        "official:AWQ_FLASHINFER,mjolnir:AWQ_MJOLNIR,mjolnir:AWQ_MJOLNIR_E2M3,mjolnir:AWQ_MJOLNIR_E2M3_VT64K"
    )


def _fake_configs(root: Path, entries: dict[str, tuple[str, ...]]) -> Path:
    """A configs root (<root>/configs/<model>/<quant>/config.yaml) for the filter."""
    for model, cfgs in entries.items():
        for cfg in cfgs:
            d = root / "configs" / model / cfg
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.yaml").write_text("model: x\n")
    return root / "configs"


def test_available_serving_legs_drop_configs_missing_for_the_model(tmp_path):
    cfgs = _fake_configs(
        tmp_path,
        {
            "Qwen/Qwen3.8-27B": ("NVFP4", "NVFP4_MJOLNIR_E2M3_LMH8_VT64K"),
            "nvidia/Qwen3.6-35B-A3B": ("NVFP4_DFLASH",),
            "acme/Partial": ("NVFP4",),
        },
    )
    assert legs.available_serving_legs("bench.ab", "thor", "Qwen/Qwen3.8-27B", cfgs) == legs.SERVING["bench.ab"]
    # a model with only the baseline config keeps only that leg
    assert legs.available_serving_legs("bench.ab", "thor", "acme/Partial", cfgs) == ((legs.OFFICIAL, "NVFP4"),)
    # Qwen3.6 has none of bench.ab's configs: nothing to measure
    assert legs.available_serving_legs("bench.ab", "thor", "nvidia/Qwen3.6-35B-A3B", cfgs) == ()


def test_available_serving_legs_default_to_the_repo_configs():
    # in a checkout, the default model keeps the full thor set
    assert legs.available_serving_legs("bench.ab", "thor", "Qwen/Qwen3.8-27B") == legs.SERVING["bench.ab"]


def test_explicit_backends_flag_beats_the_platform_body_resolution(tmp_path):
    # On a research platform the body re-resolves the legs (sentinel: an
    # unset flag means "resolve for my platform"); an explicit --backends
    # always wins over that.
    from typer.testing import CliRunner

    from mjolnir_cli import cli

    env = {"MJOLNIR_STATE": str(tmp_path / "state.json"), "MJOLNIR_PLATFORM": "spark"}
    runner = CliRunner()
    r = runner.invoke(cli.app, ["bench", "accuracy", "--dry-run"], env=env)
    assert r.exit_code == 0, r.output
    assert "/NVFP4 on " in r.output  # the official baseline leg survives on spark
    assert "/NVFP4_MJOLNIR" not in r.output  # no Mjolnir leg unless asked: the body re-resolved
    r = runner.invoke(cli.app, ["bench", "accuracy", "--dry-run", "--backends", "NVFP4_MJOLNIR"], env=env)
    assert r.exit_code == 0, r.output
    assert "/NVFP4_MJOLNIR" in r.output  # the explicit flag wins


@pytest.mark.parametrize(
    ("stage", "platform", "want"),
    [
        ("verify.sweep", "thor", "fi,fa4,mjolnir-fp8,mjolnir"),
        ("verify.hndsweep", "thor", "fi,fi-hnd,mjolnir-fp8"),
        ("verify.sweep", "spark", "fi,fa2"),
        ("verify.hndsweep", "orin", "fi,fi-hnd,mjolnir-fp8"),
    ],
)
def test_kernel_legs_per_platform(stage, platform, want):
    assert legs.kernel_legs(stage, platform) == want
    assert legs.kernel_legs(stage) == legs.KERNEL[stage]  # the default stays the flat table


# ── cycle 2: overlay-aware availability + e2e/verify re-resolution ──────────

MODEL = "Qwen/Qwen3.8-27B"


def _overlay_tree(tmp_path: Path, overlays: dict[str, tuple[str, ...]]) -> Path:
    """A configs root where each quant carries its ``platforms/`` overlays.

    A quant with no ``platforms/`` dir is a legacy thor-only config; a dir
    with ``spark.yaml`` means the config exists on spark (and only there)."""
    base = tmp_path / "configs" / "Qwen" / "Qwen3.8-27B"
    for quant, platforms_ in overlays.items():
        d = base / quant
        d.mkdir(parents=True)
        (d / "config.yaml").write_text("model: x\n")
        for p in platforms_:
            (d / "platforms").mkdir(exist_ok=True)
            (d / "platforms" / f"{p}.yaml").write_text("gpu-memory-utilization: 0.6\n")
    return tmp_path / "configs"


def test_available_serving_legs_respect_the_platform_overlays(tmp_path):
    cfgs = _overlay_tree(
        tmp_path,
        {
            "NVFP4": (),  # legacy: no platforms/ dir -> thor-only
            "NVFP4_MJOLNIR": ("thor",),  # explicit thor overlay
            "NVFP4_MJOLNIR_E2M3": ("spark",),  # spark overlay: not a thor config
        },
    )
    assert legs.available_serving_legs("bench.accuracy", "thor", MODEL, cfgs) == (
        (legs.OFFICIAL, "NVFP4"),
        (legs.MJOLNIR, "NVFP4_MJOLNIR"),
    )
    # NVFP4 is legacy thor-only, so the spark leg list is empty until a
    # config carries a spark overlay — legs skip it, they never guess
    assert legs.available_serving_legs("bench.accuracy", "spark", MODEL, cfgs) == ()


def _spark_repo(tmp_path: Path) -> Path:
    """A checkout whose only config (NVFP4) carries a spark overlay."""
    repo = tmp_path / "repo"
    (repo / "docker" / "vllm-thor").mkdir(parents=True)
    (repo / "docker" / "vllm-thor" / "Dockerfile").write_text("# base")
    d = repo / "configs" / "Qwen" / "Qwen3.8-27B" / "NVFP4"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text("model: Qwen/Qwen3.8-27B\n")
    (d / "platforms").mkdir()
    (d / "platforms" / "spark.yaml").write_text("gpu-memory-utilization: 0.6\n")
    return repo


def _serving(monkeypatch):
    import mjolnir_cli.dockerctl as dockerctl

    served: list[tuple[str, str]] = []
    monkeypatch.setattr(dockerctl, "serve_up", lambda s, **kw: served.append((s.image, s.quant)))
    return served


def test_bench_e2e_re_resolves_the_default_legs_on_a_research_platform(tmp_path, monkeypatch):
    # sentinel semantics: the import-time typer defaults are the thor table,
    # so on spark an unset flag re-resolves via the platform-aware helpers
    from typer.testing import CliRunner

    from mjolnir_cli import cli

    env = {"MJOLNIR_REPO": str(_spark_repo(tmp_path)), "MJOLNIR_PLATFORM": "spark"}
    served = _serving(monkeypatch)
    r = CliRunner().invoke(cli.app, ["bench", "e2e", "--dry-run"], env=env)
    assert r.exit_code == 0, r.output
    assert served == [(STOCK_IMAGE, "NVFP4")]  # the official baseline only: no Mjolnir legs on spark


def test_bench_e2e_explicit_flags_win_on_a_research_platform(tmp_path, monkeypatch):
    # an explicit --config is served as given — even a thor leg on spark —
    # while the unset official side still re-resolves for the platform
    from typer.testing import CliRunner

    from mjolnir_cli import cli

    env = {"MJOLNIR_REPO": str(_spark_repo(tmp_path)), "MJOLNIR_PLATFORM": "spark"}
    served = _serving(monkeypatch)
    r = CliRunner().invoke(cli.app, ["bench", "e2e", "--dry-run", "--config", "NVFP4_MJOLNIR_E2M3_LMH8_VT64K"], env=env)
    assert r.exit_code == 0, r.output
    assert served == [(STOCK_IMAGE, "NVFP4"), (DEFAULT_IMAGE, "NVFP4_MJOLNIR_E2M3_LMH8_VT64K")]


def test_verify_re_resolves_the_default_legs_on_a_research_platform(tmp_path):
    from typer.testing import CliRunner

    from mjolnir_cli import cli

    env = {"MJOLNIR_REPO": str(_spark_repo(tmp_path)), "MJOLNIR_PLATFORM": "spark"}
    out = CliRunner().invoke(cli.app, ["verify", "kv-cache-size", "--dry-run"], env=env)
    assert out.exit_code == 0, out.output
    assert "official-NVFP4" in out.output
    assert "mjolnir-NVFP4" not in out.output  # the thor table is not measured on spark by default


# ── fix round 1: the verify sentinel is the flag source, not string equality ─
# (review cycle 2, IMPROVEMENT 1: an explicit --legs that spells the default
# string must win, exactly like bench ab/accuracy/e2e's ParameterSource rule)


@pytest.mark.parametrize(
    ("stage", "default_legs", "spark_legs"),
    [
        ("sweep", legs.KERNEL["verify.sweep"], legs.kernel_legs("verify.sweep", "spark")),  # an ncu stage
        ("kv-cache-size", kvcache.DEFAULT_LEGS, legs.verify_legs("verify.kv-cache-size", "spark")),  # a serving stage
    ],
)
def test_verify_explicit_legs_equal_to_the_default_string_are_not_re_resolved(stage, default_legs, spark_legs):
    from mjolnir_cli.verify.cli import _platform_legs

    # explicitly passed (the typer ParameterSource is not DEFAULT): even when it
    # spells the default-platform table — e.g. copied from --help — the flag
    # wins; nothing is rewritten, so the run is not silently narrowed (and not
    # recorded default:true for legs nobody asked to measure)
    p, d = _platform_legs(stage, {"legs": default_legs}, {"legs": default_legs}, "spark", legs_unset=False)
    assert p["legs"] == default_legs
    assert d["legs"] == default_legs
    # unset: re-resolves for the platform — and defaults follow so is_default
    # keeps meaning "the stage's defaults for this platform" (plan §0.5)
    p, d = _platform_legs(stage, {"legs": default_legs}, {"legs": default_legs}, "spark", legs_unset=True)
    assert p["legs"] == spark_legs
    assert d["legs"] == p["legs"]


def test_verify_explicit_default_string_serving_legs_are_served_on_spark(tmp_path):
    # the review's reproduction: `MJOLNIR_PLATFORM=spark mjolnir verify
    # kv-cache-size --legs <the --help default>` measures exactly those legs
    from typer.testing import CliRunner

    from mjolnir_cli import cli

    env = {
        "MJOLNIR_REPO": str(_spark_repo(tmp_path)),
        "MJOLNIR_STATE": str(tmp_path / "state.json"),
        "MJOLNIR_PLATFORM": "spark",
    }
    out = CliRunner().invoke(cli.app, ["verify", "kv-cache-size", "--dry-run", "--legs", kvcache.DEFAULT_LEGS], env=env)
    assert out.exit_code == 0, out.output
    assert "mjolnir-NVFP4_MJOLNIR" in out.output  # the explicit legs ran, unchanged


def test_verify_explicit_default_string_ncu_legs_run_on_spark(tmp_path):
    from typer.testing import CliRunner

    from mjolnir_cli import cli

    env = {
        "MJOLNIR_REPO": str(_spark_repo(tmp_path)),
        "MJOLNIR_STATE": str(tmp_path / "state.json"),
        "MJOLNIR_PLATFORM": "spark",
    }
    out = CliRunner().invoke(cli.app, ["verify", "sweep", "--dry-run", "--legs", legs.KERNEL["verify.sweep"]], env=env)
    assert out.exit_code == 0, out.output
    assert "mjolnir.arch.thor.harness" in out.output  # the mjolnir legs are profiled as asked, not dropped
