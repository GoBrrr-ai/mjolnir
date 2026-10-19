"""fp6 (e2m3) KV on the host side: scales env for serving, `mjolnir calibrate`."""

from __future__ import annotations

from mjolnir_cli import cli, dockerctl
from mjolnir_cli.config import BACKEND_LABELS, Settings, config_field
from typer.testing import CliRunner


def _s(quant: str) -> Settings:
    return Settings(model="Qwen/Qwen3.8-27B", quant=quant)


def test_e2m3_config_is_the_mjolnir_config_with_fp6_kv():
    fp8, fp6 = _s("NVFP4_MJOLNIR"), _s("NVFP4_MJOLNIR_E2M3")
    assert config_field(fp6.config_path, "kv-cache-dtype") == "fp6_e2m3"
    assert config_field(fp8.config_path, "kv-cache-dtype") == "fp8"
    for key in ("model", "attention-backend", "max-model-len", "gpu-memory-utilization", "max-num-seqs"):
        assert config_field(fp6.config_path, key) == config_field(fp8.config_path, key), key


def test_serve_env_points_fp6_configs_at_their_scales(monkeypatch):
    monkeypatch.delenv("MJOLNIR_ATTN_KV_SCALES", raising=False)
    env = dockerctl.backend_env(_s("NVFP4_MJOLNIR_E2M3"))
    assert "MJOLNIR_ATTN_KV_SCALES=/configs/Qwen/Qwen3.8-27B/kv_scales.json" in env
    assert not any(e.startswith("MJOLNIR_ATTN_KV_SCALES") for e in dockerctl.backend_env(_s("NVFP4_MJOLNIR")))
    monkeypatch.setenv("MJOLNIR_ATTN_KV_SCALES", "/data/mine.json")  # the host's choice wins
    env = dockerctl.backend_env(_s("NVFP4_MJOLNIR_E2M3"))
    assert env.count("MJOLNIR_ATTN_KV_SCALES=/data/mine.json") == 1


def test_serve_env_names_the_config_for_the_banner():
    assert "MJOLNIR_CONFIG=NVFP4_MJOLNIR_E2M3" in dockerctl.backend_env(_s("NVFP4_MJOLNIR_E2M3"))
    assert not any(e.startswith("MJOLNIR_CONFIG") for e in dockerctl.backend_env(_s("NVFP4")))


def test_calibrate_dry_run_builds_a_foreground_container(tmp_path):
    out = tmp_path / "scales.json"
    r = CliRunner().invoke(cli.app, ["calibrate", "--config", "NVFP4_MJOLNIR", "--out", str(out), "--dry-run"])
    assert r.exit_code == 0, r.output
    cmd = r.output
    assert "--rm" in cmd and " -d " not in cmd and "-p " not in cmd  # foreground, no port
    assert f"{tmp_path.resolve()}:/out" in cmd
    assert "python3 -m mjolnir.calibrate --config /configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR/config.yaml" in cmd
    assert "--out /out/scales.json" in cmd


def test_calibrate_refuses_while_the_server_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(dockerctl, "container_running", lambda: True)
    r = CliRunner().invoke(cli.app, ["calibrate", "--out", str(tmp_path / "x.json")])
    assert r.exit_code == 3 and "server is running" in r.output


def test_removed_variant_configs_are_absent():
    """Spec change: the VT32K/LMH8 variants were removed entirely — superseded by
    NVFP4_MJOLNIR_E2M3_LMH8_VT64K (the default quant, which carries both knobs)."""
    for name in (
        "NVFP4_MJOLNIR_E2M3_LMH8",
        "NVFP4_MJOLNIR_E2M3_VT32K",
        "NVFP4_MJOLNIR_E2M3_LMH8_VT32K",
    ):
        assert not _s(name).config_path.parent.exists(), name
        assert name not in BACKEND_LABELS, name


def test_experiment_configs_are_e2m3_plus_env_sidecars():
    base = _s("NVFP4_MJOLNIR_E2M3")
    fast = _s("NVFP4_MJOLNIR_E2M3_LMH8_VT64K")
    for key in ("model", "attention-backend", "kv-cache-dtype", "max-model-len", "max-num-seqs"):
        assert config_field(fast.config_path, key) == config_field(base.config_path, key), key
    env = dockerctl.backend_env(fast)
    assert "MJOLNIR_LM_HEAD=fp8" in env
    assert "MJOLNIR_DRAFT_VOCAB=/configs/Qwen/Qwen3.8-27B/draft_vocab.json" in env
    assert "MJOLNIR_DRAFT_VOCAB_SIZE=65536" in env
    assert not any(e.startswith("MJOLNIR_LM_HEAD") for e in dockerctl.backend_env(base))


def test_vocab_sidecar_env_with_configs_expansion(tmp_path, monkeypatch):
    model, quant = "Vend/Test", "TEST"
    cfg = tmp_path / "local" / model / quant
    cfg.mkdir(parents=True)
    (cfg / "config.yaml").write_text("model: x\n")
    monkeypatch.setenv("MJOLNIR_CONFIGS_DIR", str(tmp_path / "local"))
    # {configs} expands to the model's dir (the sidecar's parent in the container)
    (cfg / ".env").write_text("MJOLNIR_DRAFT_VOCAB={configs}/draft_vocab.json\nMJOLNIR_DRAFT_VOCAB_SIZE=32768\n")
    env = dockerctl.config_env(Settings(model=model, quant=quant))
    assert "MJOLNIR_DRAFT_VOCAB=/configs-local/Vend/Test/draft_vocab.json" in env
    assert "MJOLNIR_DRAFT_VOCAB_SIZE=32768" in env


def test_calibrate_draft_vocab_dry_run(tmp_path):
    r = CliRunner().invoke(
        cli.app,
        [
            "calibrate",
            "--config",
            "NVFP4_MJOLNIR_E2M3",
            "--draft-vocab",
            "--out",
            str(tmp_path / "v.json"),
            "--dry-run",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "python3 -m mjolnir.draftvocab --config /configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3/config.yaml" in r.output


def test_a_drafter_on_mjolnir_gets_its_own_scales(monkeypatch):
    from mjolnir_cli.config import config_drafter_on_mjolnir

    monkeypatch.delenv("MJOLNIR_ATTN_KV_SCALES", raising=False)
    dspark, dflash2 = _s("NVFP4_MJOLNIR_E2M3_DSPARK"), _s("NVFP4_MJOLNIR_E2M3_DFLASH2")
    assert config_drafter_on_mjolnir(dspark.config_path)  # DSpark's drafter on MJOLNIR: fp6 + its own calibration
    import yaml

    spec = yaml.safe_load(dspark.config_path.read_text())["speculative-config"]
    assert spec["attention_backend"] == "MJOLNIR"  # vLLM never hands a DSpark drafter the target's backend
    assert not config_drafter_on_mjolnir(dflash2.config_path)  # DFlash2's sliding window stays on FlashInfer
    assert not config_drafter_on_mjolnir(_s("NVFP4_MJOLNIR_E2M3").config_path)  # MTP is part of the model
    assert not config_drafter_on_mjolnir(_s("NVFP4_DSPARK").config_path)  # official image: no Mjolnir at all
    # the drafter's scales live inside its config dir; the rest share the model's file
    assert dockerctl.kv_scales_host_path(dspark) == dspark.config_path.parent / "kv_scales.json"
    assert dockerctl.kv_scales_host_path(dflash2) == dflash2.config_path.parent.parent / "kv_scales.json"
    env = dockerctl.backend_env(dspark)
    assert "MJOLNIR_ATTN_KV_SCALES=/configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3_DSPARK/kv_scales.json" in env
    r = CliRunner().invoke(
        cli.app, ["calibrate", "--config", "NVFP4_MJOLNIR_E2M3_DSPARK", "--dry-run"]
    )  # default output: the config's own file
    assert r.exit_code == 0, r.output
    assert "--out /out/kv_scales.json" in r.output


def test_serve_refuses_an_fp6_config_without_scales(monkeypatch):
    monkeypatch.delenv("MJOLNIR_ATTN_KV_SCALES", raising=False)
    s = _s("NVFP4_MJOLNIR_E2M3_DSPARK")
    if dockerctl.kv_scales_host_path(s).exists():  # calibrated in this checkout: nothing to refuse
        return
    why = dockerctl.missing_kv_scales(s)
    assert why and "mjolnir calibrate --config NVFP4_MJOLNIR_E2M3_DSPARK" in why
    assert dockerctl.missing_kv_scales(_s("NVFP4_MJOLNIR")) is None  # fp8: no scales needed
    monkeypatch.setenv("MJOLNIR_ATTN_KV_SCALES", "/data/mine.json")
    assert dockerctl.missing_kv_scales(s) is None


def test_quick_correctness_covers_the_non_causal_drafter_block():
    from mjolnir_cli.verify import stages

    label = "non-causal q_len=8 (drafter block)"
    args = {lab: a for lab, a, _ in stages.CASES}[label]
    assert label in stages.QUICK_CASES and "--non_causal" in args and "--headdim 128" in args
