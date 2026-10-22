"""scripts/release_plan.py: next versions from the tag history, the image tag from the pinned vLLM."""

from __future__ import annotations

import datetime as dt
import importlib.util
import subprocess
import sys
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
sys.modules["release_notes"] = rn  # the Entry dataclass resolves its module at class creation
_nspec.loader.exec_module(rn)


def _merge_pr(r: Path, n: int, branch: str, title: str, files: dict[str, str], author: str = "t") -> None:
    """A PR merged the way GitHub does: a branch commit + a merge commit on main."""
    _git(r, "checkout", "-q", "-b", branch)
    for rel, text in files.items():
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text(text)
    _git(r, "add", ".")
    _git(r, "-c", f"user.name={author}", "-c", "user.email=a@a", "commit", "-qm", f"wip {title}")
    _git(r, "checkout", "-q", "-")
    msg = f"Merge pull request #{n} from thor-arena/{branch}\n\n{title}"
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-ff", branch, "-m", msg)


def _section(text: str, head: str) -> str:
    """A group's body: everything after its header line up to the next one."""
    return text.split(head, 1)[1].split("\n### ", 1)[0].split("\n## ", 1)[0]


def test_notes_group_prs_by_artifact_set_without_duplication(repo):
    _git(repo, "branch", "-M", "main")
    _git(repo, "tag", "mjolnir-kernel-v0.1.0")
    _git(repo, "tag", "mjolnir-cli-v0.2.0")
    _git(repo, "tag", "release-2026.10.14")
    _merge_pr(
        repo,
        21,
        "perf/ring",
        "perf: 2-stage KV ring",
        {"src/mjolnir/mjolnir/k.py": "x\n", "src/mjolnir-cli/mjolnir_cli/spec.py": "z\n"},
    )  # touches two artifacts -> one bullet, in the pair group
    _merge_pr(repo, 22, "dependabot/pip/torch", "Bump torch", {"src/mjolnir/pyproject.toml": "y\n"}, "dependabot[bot]")
    _merge_pr(repo, 23, "cli/spec", "feat: verify spec", {"src/mjolnir-cli/mjolnir_cli/spec.py": "z2\n"})
    _merge_pr(repo, 24, "img/patch", "fix: re-adapt a patch", {"docker/vllm-thor/patches/a.patch": "p\n"})
    _merge_pr(repo, 25, "img/docs", "docs: patch inventory", {"docker/vllm-thor/PATCHES.md": "d\n"})
    _commit(repo, "configs/a.yaml", "x: 9\n")  # pushed to main directly
    text = rn.body(
        repo,
        vllm="0.31.0",
        image=True,
        previous_umbrella="release-2026.10.14",
        kernel={"version": "0.1.1", "previous": "mjolnir-kernel-v0.1.0"},
        cli={"version": "0.2.1", "previous": "mjolnir-cli-v0.2.0"},
    )
    assert text.count("(#21)") == 1  # a cross-artifact PR is listed exactly once
    assert "- perf: 2-stage KV ring (#21)" in _section(text, "### mjolnir-kernel 0.1.1 + mjolnir-cli 0.2.1")
    assert "### mjolnir-kernel 0.1.1\n" not in text  # nothing kernel-only: no kernel-only group
    cli = _section(text, "### mjolnir-cli 0.2.1")
    assert "- feat: verify spec (#23)" in cli and "- configs/a.yaml" in cli  # a direct push, by its subject
    assert "Bump torch" not in text and "wip" not in text  # Dependabot and branch commits left out
    assert "- docs: patch inventory (#25)" in _section(text, "### Repo, docs & tooling")  # no artifact: not dropped
    image = _section(text, "### image v0.31.0")
    assert "- fix: re-adapt a patch (#24)" in image and "patch inventory" not in image  # docs are not the image
    assert "- ships mjolnir-kernel 0.1.1" in image and "- rebuilt on vLLM 0.31.0" in image
    assert "| [mjolnir-cli](https://pypi.org/project/mjolnir-cli/) | **0.2.1** |" in text
    assert "uv tool install mjolnir-cli==0.2.1" in text and "pip install mjolnir-kernel==0.1.1" in text


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
    assert "### mjolnir-kernel" not in text  # the kernel is not released: its changes wait for its release


def test_notes_platform_matrix_maps_prs_from_changed_paths(repo):
    _git(repo, "tag", "mjolnir-kernel-v0.1.0")
    _git(repo, "tag", "mjolnir-cli-v0.2.0")
    _git(repo, "tag", "release-2026.10.14")
    _merge_pr(repo, 31, "kernel/l2", "kernel: L2 evict-first", {"src/mjolnir/mjolnir/k.py": "x\n"})
    _merge_pr(
        repo, 32, "orin/awq", "configs: Orin AWQ serving", {"configs/Qwen/AWQ_FLASHINFER/platforms/orin.yaml": "y\n"}
    )
    _merge_pr(repo, 33, "spark/docs", "docs: Spark baselines", {"docs/platforms/spark/README.md": "z\n"})
    text = rn.body(
        repo,
        vllm="0.31.0",
        image=False,
        previous_umbrella="release-2026.10.14",
        kernel={"version": "0.1.1", "previous": "mjolnir-kernel-v0.1.0"},
        cli={"version": "0.2.1", "previous": "mjolnir-cli-v0.2.0"},
    )
    matrix = _section(text, "## Platform notes")
    assert "| **thor** | #31 |" in matrix  # the kernel package is Thor-only (sm_110)
    assert "| **orin** | #32 |" in matrix  # the platform overlay + the AWQ serving config
    assert "| **spark** | #33 |" in matrix  # the platform docs
    assert "- kernel: L2 evict-first (#31)" in _section(text, "### mjolnir-kernel 0.1.1")
    assert "- configs: Orin AWQ serving (#32)" in _section(text, "### mjolnir-cli 0.2.1")  # configs ship in the CLI
    assert "- docs: Spark baselines (#33)" in _section(text, "### Repo, docs & tooling")
    assert "docker pull" not in text  # image=False: no install line, no image group


def test_notes_do_not_absorb_main_commits_into_a_merged_pr(repo):
    """A PR opened before a direct push to a kernel path must not inherit that
    push's files when it merges: the merge is diffed against its FIRST parent
    only (diffing all parents would attribute main's commits to the PR)."""
    _git(repo, "tag", "mjolnir-kernel-v0.1.0")
    _git(repo, "tag", "mjolnir-cli-v0.2.0")
    _git(repo, "tag", "release-2026.10.14")
    _git(repo, "checkout", "-q", "-b", "cli/late")  # the PR branches before the push
    late = repo / "src/mjolnir-cli/mjolnir_cli/late.py"
    late.parent.mkdir(parents=True, exist_ok=True)
    late.write_text("x\n")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=a@a", "commit", "-qm", "wip feat: late cli thing")
    _git(repo, "checkout", "-q", "-")  # the push lands on main while the PR is open
    direct = repo / "src/mjolnir/mjolnir/direct.py"
    direct.parent.mkdir(parents=True, exist_ok=True)
    direct.write_text("d\n")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "src/mjolnir/mjolnir/direct.py")
    msg = "Merge pull request #61 from thor-arena/cli/late\n\nfeat: late cli thing"
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-ff", "cli/late", "-m", msg)
    text = rn.body(
        repo,
        vllm="0.31.0",
        image=False,
        previous_umbrella="release-2026.10.14",
        kernel={"version": "0.1.1", "previous": "mjolnir-kernel-v0.1.0"},
        cli={"version": "0.2.1", "previous": "mjolnir-cli-v0.2.0"},
    )
    assert text.count("(#61)") == 1  # listed once — and only for the files the PR itself changed
    assert "- feat: late cli thing (#61)" in _section(text, "### mjolnir-cli 0.2.1")
    assert "- src/mjolnir/mjolnir/direct.py" in _section(text, "### mjolnir-kernel 0.1.1")  # the push, not the PR
    assert "### mjolnir-kernel 0.1.1 + mjolnir-cli 0.2.1" not in text  # no phantom pair group


def test_notes_annotate_a_re_release_with_no_changes(repo):
    _git(repo, "tag", "mjolnir-kernel-v0.1.0")
    _git(repo, "tag", "mjolnir-cli-v0.2.0")
    _git(repo, "tag", "release-2026.10.14")
    _merge_pr(repo, 41, "cli/thing", "feat: cli thing", {"src/mjolnir-cli/mjolnir_cli/x.py": "x\n"})
    text = rn.body(
        repo,
        vllm="0.31.0",
        image=True,
        previous_umbrella="release-2026.10.14",
        kernel={"version": "0.1.1", "previous": "mjolnir-kernel-v0.1.0"},  # forced: no kernel changes
        cli={"version": "0.2.1", "previous": "mjolnir-cli-v0.2.0"},
    )
    assert (
        "| [mjolnir-kernel](https://pypi.org/project/mjolnir-kernel/) | **0.1.1** (re-released, no changes) |" in text
    )
    assert "| [mjolnir-cli](https://pypi.org/project/mjolnir-cli/) | **0.2.1** |" in text
    assert "### mjolnir-kernel" not in text  # no empty group for the re-released kernel
    assert "- ships mjolnir-kernel 0.1.1" in _section(text, "### image v0.31.0")
