"""scripts/release_plan.py: next versions from the tag history, the image tag from the pinned vLLM."""

from __future__ import annotations

import datetime as dt
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("release_plan", ROOT / "scripts" / "release_plan.py")
assert spec is not None and spec.loader is not None
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


def test_bundled_resources_release_the_cli_and_the_title_lists_only_what_ships(repo):
    _git(repo, "tag", "mjolnir-kernel-v0.1.0")
    _git(repo, "tag", "mjolnir-cli-v0.2.0")
    (repo / "docker" / "litellm").mkdir(parents=True)
    (repo / "docker" / "litellm" / "compose.yml").write_text("x\n")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "litellm")
    p = rp.plan(repo, "patch", "changed", DAY, image=False)
    assert p["packages"]["cli"]["release"] and not p["packages"]["kernel"]["release"]  # bundled into the CLI wheel
    assert p["title"] == "Mjolnir 2026.10.14 — mjolnir-cli 0.2.1"
    assert rp.plan(repo, "patch", "none", DAY, image=False)["title"].endswith("— no package or image changes")


def test_previous_umbrella_orders_same_day_releases(repo):
    for t in ("release-2026.10.14", "release-2026.10.14.2", "release-2026.10.9"):
        _git(repo, "tag", t)
    assert rp.previous_umbrella(repo) == "release-2026.10.14.2"


# ── scripts/release_notes.py ─────────────────────────────────────────────────

_nspec = importlib.util.spec_from_file_location("release_notes", ROOT / "scripts" / "release_notes.py")
assert _nspec is not None and _nspec.loader is not None
rn = importlib.util.module_from_spec(_nspec)
_nspec.loader.exec_module(rn)


def _merge_pr(r: Path, n: int, branch: str, title: str, rel: str, text: str, author: str = "t") -> None:
    """A PR merged the way GitHub does: a branch commit + a merge commit on main."""
    _git(r, "checkout", "-q", "-b", branch)
    (r / rel).parent.mkdir(parents=True, exist_ok=True)
    (r / rel).write_text(text)
    _git(r, "add", ".")
    _git(r, "-c", f"user.name={author}", "-c", "user.email=a@a", "commit", "-qm", f"wip {title}")
    _git(r, "checkout", "-q", "-")
    msg = f"Merge pull request #{n} from thor-arena/{branch}\n\n{title}"
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-ff", branch, "-m", msg)


def test_notes_list_prs_per_artifact_without_dependabot_or_wip(repo):
    _git(repo, "branch", "-M", "main")
    _git(repo, "tag", "mjolnir-kernel-v0.1.0")
    _git(repo, "tag", "mjolnir-cli-v0.2.0")
    _git(repo, "tag", "release-2026.10.14")
    _merge_pr(repo, 21, "perf/ring", "perf: 2-stage KV ring", "src/mjolnir/mjolnir/k.py", "x\n")
    _merge_pr(repo, 22, "dependabot/pip/torch", "Bump torch", "src/mjolnir/pyproject.toml", "y\n", "dependabot[bot]")
    _merge_pr(repo, 23, "cli/spec", "feat: verify spec", "src/mjolnir-cli/mjolnir_cli/spec.py", "z\n")
    _merge_pr(repo, 24, "img/patch", "fix: re-adapt a patch", "docker/vllm-thor/patches/a.patch", "p\n")
    _merge_pr(repo, 25, "img/docs", "docs: patch inventory", "docker/vllm-thor/PATCHES.md", "d\n")
    _commit(repo, "configs/a.yaml", "x: 9\n")  # pushed to main directly
    text = rn.body(
        repo,
        vllm="0.31.0",
        image=True,
        previous_umbrella="release-2026.10.14",
        kernel={"version": "0.1.1", "previous": "mjolnir-kernel-v0.1.0"},
        cli={"version": None, "previous": "mjolnir-cli-v0.2.0"},
    )
    kernel = text.split("## mjolnir-kernel 0.1.1")[1].split("##")[0]
    assert "- perf: 2-stage KV ring (#21)" in kernel
    assert "Bump torch" not in text and "wip" not in text  # Dependabot and branch commits left out
    assert "## mjolnir-cli" not in text  # not released: no section
    assert "| unchanged (0.2.0) |" in text and "uv tool install" not in text
    image = text.split("## image v0.31.0")[1]
    assert "- fix: re-adapt a patch (#24)" in image and "patch inventory" not in image  # docs are not in the image
    assert (
        "ships mjolnir-kernel 0.1.1 (see above)" in image and "perf: 2-stage" not in image
    )  # referenced, not repeated
    assert "- rebuilt on vLLM 0.31.0" in image


def test_notes_flag_a_base_bump_and_unreleased_kernel_changes(repo):
    _git(repo, "tag", "mjolnir-kernel-v0.1.0")
    _git(repo, "tag", "release-2026.10.14")
    _commit(repo, "src/mjolnir/mjolnir/__init__.py", '__version__ = "0.1.0"  # changed\n')
    _commit(repo, "docker/vllm-thor/Dockerfile", "ARG BASE_IMAGE=vllm/vllm-openai:v0.32.0-ubuntu2404\n")
    text = rn.body(
        repo,
        vllm="0.32.0",
        image=True,
        previous_umbrella="release-2026.10.14",
        kernel={"version": None, "previous": "mjolnir-kernel-v0.1.0"},
        cli={"version": None, "previous": None},
    )
    assert "base bumped to vLLM 0.32.0 (from 0.31.0)" in text
    assert "changes not released" in text  # the image would ship kernel code PyPI does not have
