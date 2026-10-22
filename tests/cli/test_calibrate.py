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
    assert "MJOLNIR_ATTN_KV_SCALES=/configs/Qwen/Qwen3.8-27B/kv_scales/e2m3.json" in env
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
    # the drafter's scales live in the model's kv_scales/ dir, named for the
    # drafter's method; the rest share the model's per-format file
    assert (
        dockerctl.kv_scales_host_path(dspark)
        == dspark.config_path.parent.parent / "kv_scales" / "dspark.kv_scales.json"
    )
    assert dockerctl.kv_scales_host_path(dflash2) == dflash2.config_path.parent.parent / "kv_scales" / "e2m3.json"
    env = dockerctl.backend_env(dspark)
    assert "MJOLNIR_ATTN_KV_SCALES=/configs/Qwen/Qwen3.8-27B/kv_scales/dspark.kv_scales.json" in env
    r = CliRunner().invoke(
        cli.app, ["calibrate", "--config", "NVFP4_MJOLNIR_E2M3_DSPARK", "--dry-run"]
    )  # default output: the drafter's file in the model's kv_scales/ dir
    assert r.exit_code == 0, r.output
    assert "--out /out/dspark.kv_scales.json" in r.output


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


# ── kv_scales/<format>.json — the only layout ────────────────────────────────


def _local_tree(tmp_path, monkeypatch, dtype: str = "fp6_e2m3", drafter: bool = False):
    """A user-local config tree: Vend/Test/TEST with one config whose
    ``kv-cache-dtype`` (and optional Mjolnir drafter) the test controls."""
    model, quant = "Vend/Test", "TEST"
    cfg = tmp_path / "local" / model / quant
    cfg.mkdir(parents=True, exist_ok=True)
    body = f"model: x\nkv-cache-dtype: {dtype}\nattention-backend: MJOLNIR\n"
    if drafter:
        body += (
            "speculative-config:\n  method: dspark\n  model: v/d\n"
            "  num_speculative_tokens: 7\n  attention_backend: MJOLNIR\n"
        )
    (cfg / "config.yaml").write_text(body)
    monkeypatch.setenv("MJOLNIR_CONFIGS_DIR", str(tmp_path / "local"))
    return Settings(model=model, quant=quant)


def test_loader_reads_the_per_format_file(tmp_path, monkeypatch):
    """kv_scales/<format>.json beside the config dirs is the only layout
    `mjolnir serve` reads (and the one `mjolnir calibrate` writes)."""
    s = _local_tree(tmp_path, monkeypatch)
    mdir = s.config_path.parent.parent
    (mdir / "kv_scales").mkdir()
    (mdir / "kv_scales" / "e2m3.json").write_text("{}")
    assert dockerctl.kv_scales_host_path(s) == mdir / "kv_scales" / "e2m3.json"


def test_loader_defaults_to_the_per_format_path(tmp_path, monkeypatch):
    """No file present yet: the canonical path calibrate writes (what
    missing_kv_scales names)."""
    s = _local_tree(tmp_path, monkeypatch)
    assert dockerctl.kv_scales_host_path(s) == s.config_path.parent.parent / "kv_scales" / "e2m3.json"


def test_the_scale_file_is_named_after_the_kv_format(tmp_path, monkeypatch):
    for dtype, stem in (("fp8_e5m2", "e5m2"), ("fp8", "fp8"), ("fp6_e2m3", "e2m3")):
        s = _local_tree(tmp_path, monkeypatch, dtype=dtype)
        assert dockerctl.kv_scales_host_path(s).name == f"{stem}.json"


def test_a_drafter_calibrates_into_the_model_kv_scales_dir(tmp_path, monkeypatch):
    """A Mjolnir drafter's scales are kv_scales/<method>.kv_scales.json in
    the model's kv_scales/ dir; every other config shares the model's
    per-format file."""
    s = _local_tree(tmp_path, monkeypatch, drafter=True)
    cdir = s.config_path.parent
    mdir = cdir.parent
    canonical = mdir / "kv_scales" / "dspark.kv_scales.json"
    assert dockerctl.kv_scales_host_path(s) == canonical  # nothing yet: canonical
    (mdir / "kv_scales").mkdir()
    canonical.write_text("{}")
    assert dockerctl.kv_scales_host_path(s) == canonical


def test_calibrate_writes_one_file(tmp_path, monkeypatch):
    """One run writes one file: the per-format kv_scales/<format>.json (an
    explicit --out or --draft-vocab names that one file)."""
    s = _local_tree(tmp_path, monkeypatch)
    mdir = s.config_path.parent.parent
    target = dockerctl.calibrate_targets(s, None, False)
    assert target == mdir / "kv_scales" / "e2m3.json"
    assert dockerctl.calibrate_targets(s, tmp_path / "x.json", False) == tmp_path / "x.json"
    assert dockerctl.calibrate_targets(s, None, True) == mdir / "kv_scales" / "draft_vocab.json"


def test_a_drafter_calibration_writes_its_own_file(tmp_path, monkeypatch):
    s = _local_tree(tmp_path, monkeypatch, drafter=True)
    target = dockerctl.calibrate_targets(s, None, False)
    assert target == s.config_path.parent.parent / "kv_scales" / "dspark.kv_scales.json"


def test_calibrate_default_out_is_the_per_format_file():
    r = CliRunner().invoke(cli.app, ["calibrate", "--config", "NVFP4_MJOLNIR_E2M3", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "--out /out/e2m3.json" in r.output
    assert "/configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3/config.yaml" in r.output


def test_the_user_dir_fallback_keeps_the_kv_scales_subdir(tmp_path, monkeypatch):
    """An unwritable configs dir redirects the whole model-relative layout
    (kv_scales/e2m3.json, not a flat e2m3.json) under the user configs dir —
    the wheel-installed-CLI case (bundled configs are read-only)."""
    s = _local_tree(tmp_path, monkeypatch)
    mdir = s.config_path.parent.parent
    (mdir / "kv_scales").mkdir()
    (mdir / "kv_scales").chmod(0o500)  # exists, unwritable -> the fallback
    try:
        monkeypatch.setattr(dockerctl, "user_configs_dir", lambda: tmp_path / "user")
        target = dockerctl.calibrate_targets(s, None, False)
        assert target == tmp_path / "user" / s.model / "kv_scales" / "e2m3.json"
    finally:
        (mdir / "kv_scales").chmod(0o755)


def test_calibrate_runs_the_container_and_reports_the_scales(monkeypatch, tmp_path, capsys):
    out = tmp_path / "scales.json"
    seen = {}

    def fake_run_once(argv, name):
        seen.update(name=name)
        return 0

    monkeypatch.setattr(dockerctl, "run_once", fake_run_once)
    r = CliRunner().invoke(cli.app, ["calibrate", "--config", "NVFP4_MJOLNIR", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert seen["name"] == "mjolnir-calibrate"
    assert f"wrote {out}" in r.output


def test_calibrate_mounts_the_checkouts_kernel_package(monkeypatch, tmp_path):
    src = tmp_path / "checkout" / "src" / "mjolnir"
    (src / "mjolnir").mkdir(parents=True)
    (src / "mjolnir" / "calibrate.py").write_text("# newer than the image's\n")
    monkeypatch.setattr("mjolnir_cli.config.find_repo_root", lambda: tmp_path / "checkout")
    r = CliRunner().invoke(cli.app, ["calibrate", "--config", "NVFP4_MJOLNIR", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "/opt/mjolnir-src:ro" in r.output and "PYTHONPATH=/opt/mjolnir-src" in r.output


def test_calibrate_draft_vocab_mounts_the_draftvocab_module(monkeypatch, tmp_path):
    src = tmp_path / "checkout" / "src" / "mjolnir"
    (src / "mjolnir").mkdir(parents=True)
    (src / "mjolnir" / "draftvocab.py").write_text("")
    monkeypatch.setattr("mjolnir_cli.config.find_repo_root", lambda: tmp_path / "checkout")
    r = CliRunner().invoke(cli.app, ["calibrate", "--config", "NVFP4_MJOLNIR", "--draft-vocab", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "python3 -m mjolnir.draftvocab" in r.output


def test_calibrate_without_a_checkout_omits_the_source_mount(monkeypatch):
    def no_repo():
        raise FileNotFoundError("needs a mjolnir checkout")

    monkeypatch.setattr("mjolnir_cli.config.find_repo_root", no_repo)
    r = CliRunner().invoke(cli.app, ["calibrate", "--config", "NVFP4_MJOLNIR", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "/opt/mjolnir-src" not in r.output
