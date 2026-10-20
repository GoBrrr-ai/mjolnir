"""AC6 regression (feature.md): the availability tables change no thor serve
plan.

``mjolnir serve up --dry-run`` prints the ``docker run`` argv and returns
before touching docker (or the platform overlays). For every shipped config
the plan must stay exactly what it was before the cycle-3a refactor: the
thor default image, the default port mapping, the config's own
byte-identical ``vllm serve`` command line behind ``-lc``, and its
deterministic backend env (``MJOLNIR_CONFIG``, the fp6 scales pointer, every
``.env`` sidecar knob). Parametrized over ``scan_configs(configs_root())``
so a new config joins the pin automatically. ``stale_image`` is monkeypatched
(the one pre-dry-run docker probe) — the test never contacts docker.
"""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from mjolnir_cli import cli, dockerctl, overlays
from mjolnir_cli.config import (
    DEFAULT_IMAGE,
    DEFAULT_PORT,
    Settings,
    config_attention_backend,
    config_field,
    configs_root,
    scan_configs,
)
from mjolnir_cli.platforms import REGISTRY

ENTRIES = scan_configs(configs_root())

# The vllm serve line build_run_args appends behind `-lc` on `serve up`
# (no serve_args): byte-identical per config, and no host paths in it.
_SERVE_TAIL = (
    "--host 0.0.0.0 --cpu-offload-gb 0 --pipeline-parallel-size 1 "
    "--disable-fastapi-docs --enable-force-include-usage "
    "--enable-prompt-tokens-details --download-dir /data/models/huggingface"
)


@pytest.fixture(autouse=True)
def _plan_from_flags_only(monkeypatch, tmp_path):
    # the plan must come from the flags + baked defaults only: no remembered
    # state, no host MJOLNIR_* overrides (MJOLNIR_PLATFORM stays at the
    # suite's thor pin — the platform under regression)
    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "state.json"))
    for var in (
        "MJOLNIR_MODEL",
        "MJOLNIR_QUANT",
        "MJOLNIR_IMAGE",
        "MJOLNIR_PORT",
        "MJOLNIR_HF_CACHE",
        "MJOLNIR_CONFIGS_DIR",
        "MJOLNIR_ATTN_KV_SCALES",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: f"{e.model}:{e.quant}")
def test_serve_dry_run_argv_is_unchanged(tmp_path, monkeypatch, entry):
    monkeypatch.setattr(dockerctl, "stale_image", lambda image: None)
    r = CliRunner().invoke(cli.app, ["serve", "up", "--model", entry.model, "--config", entry.quant, "--dry-run"])
    assert r.exit_code == 0, r.output
    line = next(ln for ln in r.output.splitlines() if ln.startswith("docker run "))
    assert line.startswith("docker run -d --name mjolnir-vllm")  # detached, the one container
    serve = f"vllm serve --config /configs/{entry.model}/{entry.quant}/config.yaml {_SERVE_TAIL}"
    # byte-identical serving line behind -lc, on the thor default image + port
    assert f"-p {DEFAULT_PORT}:8000 {DEFAULT_IMAGE} -lc {serve}" in line
    # the config's own backend env rides along, exactly as before
    backend = config_attention_backend(entry.path) or ""
    fp6 = (config_field(entry.path, "kv-cache-dtype") or "").startswith("fp6")
    if backend in ("MJOLNIR", "CUSTOM"):
        assert f"-e MJOLNIR_CONFIG={entry.quant}" in line
        assert "-e VLLM_KV_CACHE_LAYOUT=HND" in line
        if fp6:
            assert "-e MJOLNIR_ATTN_KV_SCALES=/configs/" in line  # its calibration, container-side
        else:
            assert "-e MJOLNIR_ATTN_KV_SCALES=" not in line  # fp8 needs no scales
    else:
        assert "-e MJOLNIR_CONFIG=" not in line
        assert "-e MJOLNIR_ATTN_KV_SCALES=" not in line
    # every sidecar knob the config ships reaches the container env
    for kv_env in dockerctl.config_env(Settings(model=entry.model, quant=entry.quant)):
        assert f"-e {kv_env}" in line, kv_env


MODEL = "Qwen/Qwen3.8-27B"


def test_orin_awq_baseline_dry_run_and_merged_overlay(tmp_path, monkeypatch):
    """The Orin AWQ baseline end to end: the dry-run argv serves its config
    on orin's own (upstream) image with no Mjolnir env (FlashInfer reads the
    fp8 KV natively), and the orin overlay's 64GB tuning merges over the
    platform-neutral base — model identity, backends, KV dtype, MTP K=3 and
    the chat template come from config.yaml; max-num-seqs / 190K / 8K / 0.9 /
    prefix-match-unit only from platforms/orin.yaml. The serve-time gate
    accepts the merged config on orin."""
    monkeypatch.setattr(dockerctl, "stale_image", lambda image: None)
    env = {"MJOLNIR_PLATFORM": "orin", "MJOLNIR_STATE": str(tmp_path / "state.json")}
    r = CliRunner().invoke(
        cli.app, ["serve", "up", "--model", MODEL, "--config", "AWQ_FLASHINFER", "--dry-run"], env=env
    )
    assert r.exit_code == 0, r.output
    line = next(ln for ln in r.output.splitlines() if ln.startswith("docker run "))
    serve = f"vllm serve --config /configs/{MODEL}/AWQ_FLASHINFER/config.yaml {_SERVE_TAIL}"
    assert f"-p {DEFAULT_PORT}:8000 {REGISTRY['orin'].default_image} -lc {serve}" in line
    assert "-e MJOLNIR_CONFIG=" not in line  # FlashInfer serves fp8 natively: no Mjolnir reader env
    assert "-e MJOLNIR_ATTN_KV_SCALES=" not in line  # fp8 needs no scales

    d = configs_root() / MODEL / "AWQ_FLASHINFER"
    base = yaml.safe_load((d / "config.yaml").read_text())
    merged = overlays.merge_config(base, overlays.load_overlay(d, "orin") or {})
    assert merged["model"] == "twolven/Qwen3.8-27B-abliterated-AWQ-MTP"
    assert merged["served-model-name"] == MODEL
    assert merged["quantization"] == "compressed-tensors"
    assert merged["dtype"] == "bfloat16"
    assert merged["load-format"] == "safetensors"
    assert merged["attention-backend"] == "flashinfer"
    assert merged["kv-cache-dtype"] == "fp8"
    assert merged["speculative-config"] == {"method": "mtp", "num_speculative_tokens": 3}
    assert merged["chat-template"] == "/configs/Qwen/Qwen3.8_chat_template.jinja"
    for k, v in {
        "max-num-seqs": 5,
        "max-model-len": "190K",
        "max-num-batched-tokens": "8K",
        "gpu-memory-utilization": 0.9,
        "prefix-match-unit": 32,
    }.items():
        assert merged[k] == v, k
        assert k not in base, f"{k} is Orin-64GB tuning: the overlay's, never the base's"
    overlays.validate_serving(d / "config.yaml", "orin")  # does not raise
