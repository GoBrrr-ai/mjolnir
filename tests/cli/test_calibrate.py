"""fp6 (e2m3) KV on the host side: scales env for serving, `mjolnir calibrate`."""

from __future__ import annotations

from mjolnir_cli import cli, dockerctl
from mjolnir_cli.config import Settings, config_field
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


def test_calibrate_dry_run_builds_a_foreground_container(tmp_path):
    out = tmp_path / "scales.json"
    r = CliRunner().invoke(cli.app, ["calibrate", "--config", "NVFP4_MJOLNIR", "--out", str(out), "--dry-run"])
    assert r.exit_code == 0, r.output
    cmd = r.output
    assert "--rm" in cmd and " -d " not in cmd and "-p " not in cmd  # foreground, no port
    assert f"{tmp_path.resolve()}:/out" in cmd
    assert "python3 -m mjolnir.calibrate --config /configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR.yaml" in cmd
    assert "--out /out/scales.json" in cmd


def test_calibrate_refuses_while_the_server_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(dockerctl, "container_running", lambda: True)
    r = CliRunner().invoke(cli.app, ["calibrate", "--out", str(tmp_path / "x.json")])
    assert r.exit_code == 3 and "server is running" in r.output
