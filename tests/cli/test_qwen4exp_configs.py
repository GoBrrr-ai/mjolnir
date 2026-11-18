"""The Qwen3.8-Flash-Next configs: one served name, and the PLE slot-pool
cache (``MJOLNIR_PLE_DISK_CACHE_GB`` in the ``.env``) only together with vLLM's
disk table (``engram-config: {disk_offload: true}`` in the YAML), which is
its on/off switch: without it the cache variable would be silently ignored
and the table would be allocated in host memory."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

MODEL_DIR = Path(__file__).resolve().parents[2] / "configs" / "nvidia" / "Qwen3.8-Flash-Next"
CONFIGS = sorted(p.parent for p in MODEL_DIR.glob("*/config.yaml"))


def _env(config_dir: Path) -> dict[str, str]:
    out = {}
    for line in (config_dir / ".env").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def test_there_are_configs():
    assert CONFIGS


@pytest.mark.parametrize("config_dir", CONFIGS, ids=lambda p: p.name)
def test_ple_cache_comes_with_disk_offload(config_dir):
    cfg = yaml.safe_load((config_dir / "config.yaml").read_text())
    if float(_env(config_dir).get("MJOLNIR_PLE_DISK_CACHE_GB", "0")) > 0:
        assert cfg["engram-config"] == {"disk_offload": True}


def test_one_served_name_per_model_dir():
    names = {yaml.safe_load((d / "config.yaml").read_text())["served-model-name"] for d in CONFIGS}
    assert names == {"nvidia/Qwen3.8-Flash-Next"}
