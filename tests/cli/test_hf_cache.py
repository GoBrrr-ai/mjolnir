"""``MJOLNIR_HF_CACHE`` reuses a host HF hub cache in every container mjolnir starts."""

from __future__ import annotations

from pathlib import Path

import pytest

from mjolnir_cli import dockerctl
from mjolnir_cli.config import HFCacheError, Settings, host_hf_hub
from mjolnir_cli.verify import docker as verify_docker

HUB = dockerctl.HF_DOWNLOAD_DIR


def _mounts(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]


def _envs(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]


@pytest.fixture
def hub(tmp_path: Path) -> Path:
    h = tmp_path / "huggingface" / "hub"
    (h / "models--org--model").mkdir(parents=True)
    return h


# --- resolving the setting ------------------------------------------------------


def test_unset_means_none(monkeypatch):
    monkeypatch.delenv("MJOLNIR_HF_CACHE", raising=False)
    assert host_hf_hub() is None


def test_hub_folder_used_as_is(monkeypatch, hub):
    monkeypatch.setenv("MJOLNIR_HF_CACHE", str(hub))
    assert host_hf_hub() == hub.resolve()


def test_hf_home_root_resolves_to_its_hub(monkeypatch, hub):
    monkeypatch.setenv("MJOLNIR_HF_CACHE", str(hub.parent))
    assert host_hf_hub() == hub.resolve()


def test_tilde_is_expanded(monkeypatch, hub):
    monkeypatch.setenv("HOME", str(hub.parent.parent))
    monkeypatch.setenv("MJOLNIR_HF_CACHE", "~/huggingface/hub")
    assert host_hf_hub() == hub.resolve()


def test_relative_path_is_made_absolute(monkeypatch, hub):
    monkeypatch.chdir(hub.parent)
    monkeypatch.setenv("MJOLNIR_HF_CACHE", "hub")
    assert host_hf_hub() == hub.resolve()


def test_missing_path_is_an_error_not_a_redownload(monkeypatch, tmp_path):
    monkeypatch.setenv("MJOLNIR_HF_CACHE", str(tmp_path / "hugginface-typo" / "hub"))
    with pytest.raises(HFCacheError):
        host_hf_hub()


# --- serve / one-shot containers -------------------------------------------------


def test_no_mount_and_no_env_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("MJOLNIR_HF_CACHE", raising=False)
    argv = dockerctl.build_run_args(Settings(data_dir=tmp_path))
    assert not [m for m in _mounts(argv) if m.endswith(f":{HUB}")]
    assert not [e for e in _envs(argv) if e.startswith("HF_HUB_CACHE=")]


def test_hub_mounted_as_download_dir_and_hub_cache(monkeypatch, tmp_path, hub):
    (tmp_path / "models").mkdir()  # absent host dirs are skipped
    monkeypatch.setenv("MJOLNIR_HF_CACHE", str(hub))
    argv = dockerctl.build_run_args(Settings(data_dir=tmp_path))
    mounts = _mounts(argv)
    assert f"{hub.resolve()}:{HUB}" in mounts
    # nested inside the /data/models mount, so it must come after it
    assert mounts.index(f"{hub.resolve()}:{HUB}") > mounts.index(f"{tmp_path / 'models'}:/data/models")
    assert f"HF_HUB_CACHE={HUB}" in _envs(argv)
    assert f"--download-dir {HUB}" in argv[-1]


def test_one_shot_containers_get_it(monkeypatch, tmp_path, hub):
    monkeypatch.setenv("MJOLNIR_HF_CACHE", str(hub))
    argv = dockerctl.build_run_args(Settings(data_dir=tmp_path), command="true")
    assert f"{hub.resolve()}:{HUB}" in _mounts(argv)


def test_bad_path_fails_the_docker_command(monkeypatch, tmp_path):
    monkeypatch.setenv("MJOLNIR_HF_CACHE", str(tmp_path / "nope"))
    with pytest.raises(dockerctl.DockerError, match="MJOLNIR_HF_CACHE"):
        dockerctl.build_run_args(Settings(data_dir=tmp_path))


# --- verify kernel/vLLM test container -------------------------------------------


def test_verify_tests_container_nests_the_hub(monkeypatch, hub):
    monkeypatch.setenv("MJOLNIR_HF_CACHE", str(hub))
    assert verify_docker._hf_hub_mount() == ["-v", f"{hub.resolve()}:/root/.cache/huggingface/hub"]
    monkeypatch.delenv("MJOLNIR_HF_CACHE")
    assert verify_docker._hf_hub_mount() == []
