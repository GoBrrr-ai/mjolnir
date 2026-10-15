"""scripts/release_plan.py: next versions from the tag history, the image tag from the pinned vLLM."""

from __future__ import annotations

import datetime as dt
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("release_plan", ROOT / "scripts" / "release_plan.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

DAY = dt.date(2026, 10, 14)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "r"
    for rel, text in {
        "src/mjolnir/mjolnir/__init__.py": '__version__ = "0.1.0"\n',
        "src/mjolnir-cli/mjolnir_cli/__init__.py": '__version__ = "0.2.0"\n',
        "src/mjolnir-cli/mjolnir_cli/config.py": 'VLLM_VERSION = "0.31.0"\n',
        "docker/vllm-thor/Dockerfile": "ARG BASE_IMAGE=vllm/vllm-openai:v0.31.0-ubuntu2404\n",
        "configs/a.yaml": "x: 1\n",
    }.items():
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text(text)
    _git(r, "init", "-q")
    # hermetic: the scratch repo must not inherit the host's git config — a global
    # commit.gpgsign / tag.gpgsign would need the host's GPG agent or an editor
    # (git tag then fails with "Terminal is dumb, but EDITOR unset")
    _git(r, "config", "commit.gpgsign", "false")
    _git(r, "config", "tag.gpgsign", "false")
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return r


def _commit(r: Path, rel: str, text: str) -> None:
    (r / rel).write_text(text)
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", rel)


def test_first_release_takes_the_source_versions(repo):
    p = rp.plan(repo, "patch", "changed", DAY)
    assert p["vllm"] == "0.31.0" and p["umbrella"] == "release-2026.10.14"
    assert p["packages"]["kernel"]["tag"] == "mjolnir-kernel-v0.1.0"
    assert p["packages"]["cli"]["tag"] == "mjolnir-cli-v0.2.0"


def test_bumps_only_what_changed_since_its_tag(repo):
    _git(repo, "tag", "mjolnir-kernel-v0.1.0")
    _git(repo, "tag", "mjolnir-cli-v0.2.0")
    _git(repo, "tag", "release-2026.10.14")
    _commit(repo, "configs/a.yaml", "x: 2\n")  # a CLI-side change (configs ship in the wheel)
    p = rp.plan(repo, "minor", "changed", DAY)
    assert p["packages"]["kernel"]["release"] is False
    assert p["packages"]["cli"]["version"] == "0.3.0"
    assert p["umbrella"] == "release-2026.10.14.2"
    forced = rp.plan(repo, "patch", "both", DAY)
    assert forced["packages"]["kernel"]["version"] == "0.1.1"
    assert rp.plan(repo, "patch", "kernel", DAY)["packages"]["cli"]["release"] is False


def test_a_hand_bump_in_the_source_wins(repo):
    _git(repo, "tag", "mjolnir-kernel-v0.1.0")
    _commit(repo, "src/mjolnir/mjolnir/__init__.py", '__version__ = "1.0.0"\n')
    assert rp.plan(repo, "patch", "changed", DAY)["packages"]["kernel"]["version"] == "1.0.0"


def test_vllm_pin_mismatch_fails(repo):
    _commit(repo, "docker/vllm-thor/Dockerfile", "ARG BASE_IMAGE=vllm/vllm-openai:v0.32.0-ubuntu2404\n")
    with pytest.raises(SystemExit, match="bump both"):
        rp.plan(repo, "patch", "changed", DAY)
