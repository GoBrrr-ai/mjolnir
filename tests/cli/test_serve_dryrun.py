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
from typer.testing import CliRunner

from mjolnir_cli import cli, dockerctl
from mjolnir_cli.config import (
    DEFAULT_IMAGE,
    DEFAULT_PORT,
    Settings,
    config_attention_backend,
    config_field,
    configs_root,
    scan_configs,
)

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
