"""User-local configs mount beside the read-only ``/configs``, not inside it.

Docker cannot create a mountpoint inside a read-only bind mount, so nesting a
local ``<vendor>/<model>/<quant>`` dir under ``/configs`` failed at container
start for any vendor/model the repo doesn't ship (``mkdirat … read-only file
system``). The local model dir is mounted at ``/configs-local/<model>`` instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mjolnir_cli import dockerctl
from mjolnir_cli.config import Settings

MODEL, QUANT = "NewVendor/New-Model-7B", "NVFP4_MJOLNIR_E2M3"


def _mounts(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]


def _envs(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]


@pytest.fixture
def local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "local"
    cdir = root / MODEL / QUANT
    cdir.mkdir(parents=True)
    (cdir / "config.yaml").write_text("model: org/new-model\nattention-backend: MJOLNIR\nkv-cache-dtype: fp6_e2m3\n")
    (cdir / ".env").write_text("MJOLNIR_DRAFT_VOCAB={configs}/draft_vocab.json\n")
    (root / MODEL / "kv_scales.json").write_text("{}")
    monkeypatch.setenv("MJOLNIR_CONFIGS_DIR", str(root))
    return root


def test_local_model_dir_mounted_beside_configs(local: Path, tmp_path: Path) -> None:
    s = Settings(model=MODEL, quant=QUANT, data_dir=tmp_path)
    argv = dockerctl.build_run_args(s)
    mounts = _mounts(argv)
    assert f"{local / MODEL}:/configs-local/{MODEL}:ro" in mounts
    # nothing is nested under the read-only /configs mount any more
    assert not [m for m in mounts if m.split(":")[1].startswith(f"/configs/{MODEL.split('/')[0]}")]
    assert f"--config /configs-local/{MODEL}/{QUANT}/config.yaml" in argv[-1]


def test_local_env_and_scales_point_into_the_local_mount(local: Path, tmp_path: Path) -> None:
    s = Settings(model=MODEL, quant=QUANT, data_dir=tmp_path)
    envs = _envs(dockerctl.build_run_args(s))
    assert f"MJOLNIR_DRAFT_VOCAB=/configs-local/{MODEL}/draft_vocab.json" in envs
    assert f"MJOLNIR_ATTN_KV_SCALES=/configs-local/{MODEL}/kv_scales.json" in envs


def test_one_shot_command_paths_follow_the_config(local: Path, tmp_path: Path) -> None:
    s = Settings(model=MODEL, quant=QUANT, data_dir=tmp_path)
    assert dockerctl.config_model_dir(s) == f"/configs-local/{MODEL}"


def test_repo_config_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MJOLNIR_CONFIGS_DIR", str(tmp_path / "empty"))
    s = Settings(model="Qwen/Qwen3.8-27B", quant="NVFP4_MJOLNIR_E2M3", data_dir=tmp_path)
    argv = dockerctl.build_run_args(s)
    assert dockerctl.config_model_dir(s) == "/configs/Qwen/Qwen3.8-27B"
    assert "--config /configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR_E2M3/config.yaml" in argv[-1]
    assert not [m for m in _mounts(argv) if "/configs-local" in m]
