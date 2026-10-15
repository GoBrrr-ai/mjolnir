"""``MJOLNIR_HF_CACHE`` mounts a host HF hub cache as vLLM's download dir."""

from __future__ import annotations

from mjolnir_cli import dockerctl
from mjolnir_cli.config import Settings


def _mounts(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]


def test_no_hf_cache_mount_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("MJOLNIR_HF_CACHE", raising=False)
    argv = dockerctl.build_run_args(Settings(data_dir=tmp_path))
    assert not [m for m in _mounts(argv) if m.endswith(":/data/models/huggingface")]


def test_hf_cache_mounted_as_download_dir(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    (tmp_path / "models").mkdir()  # absent host dirs are skipped
    monkeypatch.setenv("MJOLNIR_HF_CACHE", str(hub))
    argv = dockerctl.build_run_args(Settings(data_dir=tmp_path))
    mounts = _mounts(argv)
    assert f"{hub}:/data/models/huggingface" in mounts
    # nested inside the /data/models mount, so it must come after it
    assert mounts.index(f"{hub}:/data/models/huggingface") > mounts.index(f"{tmp_path / 'models'}:/data/models")
    assert "--download-dir /data/models/huggingface" in argv[-1]


def test_hf_cache_also_reaches_one_shot_containers(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("MJOLNIR_HF_CACHE", str(hub))
    argv = dockerctl.build_run_args(Settings(data_dir=tmp_path), command="true")
    assert f"{hub}:/data/models/huggingface" in _mounts(argv)
