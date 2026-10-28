"""rawprune: the used-set mirrors the render's own selection (one doc per
stage/platform/model group — the newest default) and the prune git rms
exactly the dead ones."""

from __future__ import annotations

import subprocess
from pathlib import Path

from mjolnir_cli import rawprune
from mjolnir_cli.config import RepoLayout
from mjolnir_cli.verify import results


def _layout(tmp_path: Path) -> RepoLayout:
    return RepoLayout(
        repo_root=tmp_path,
        data_dir=tmp_path,
        state_file=tmp_path / "s.json",
        benchmarks_dir=tmp_path / "benchmarks",
        history_file=tmp_path / "benchmarks" / "history.jsonl",
    )


def _raw(
    layout: RepoLayout,
    stage: str,
    run_id: str,
    *,
    default: bool,
    platform: str | None = "orin",
    model: str | None = None,
) -> Path:
    doc = results.header(
        stage,
        run_id,
        image="narandill/mjolnir-vllm:v0.32.0",
        default=default,
        params={},
        kernel={"version": "0.0.0"},
        gate={},
        platform=platform,
    )
    if model is not None:
        doc["model"] = model
    return results.save(layout.verify_dir, doc)


def _git_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)


def test_prune_keeps_only_the_render_read(tmp_path):
    layout = _layout(tmp_path)
    old = _raw(layout, "sweep", "20261028-000001", default=True)  # superseded
    new = _raw(layout, "sweep", "20261028-000002", default=True)  # the headline
    _raw(layout, "sweep", "20261028-000003", default=False)  # an ad-hoc A/B
    timing = _raw(layout, "timing", "20261028-000004", default=True)  # uncharted stage
    other = _raw(layout, "sweep", "20261028-000005", default=True, model="other-model")  # its own pair
    broken = layout.verify_dir / "sweep" / "20261028-000006.json"
    broken.write_text("{not json")  # fails the schema check
    orphan = layout.verify_dir / "bogus" / "20261028-000007.json"
    orphan.parent.mkdir()
    orphan.write_text("{}")  # a stage outside STAGES
    _git_repo(tmp_path)

    rep = rawprune.audit(layout)
    assert rep.used == [str(new), str(other)]
    dead = {str(Path(p)) for p in rep.removed}
    assert {str(old), str(timing), str(broken), str(orphan)} <= dead
    assert not rep.missing  # everything render-read is committed

    rep = rawprune.prune(layout)
    assert not Path(old).exists() and not broken.exists() and not orphan.exists()
    assert Path(new).exists() and Path(other).exists()
    assert not rep.untracked and not rep.missing


def test_untracked_dead_and_uncommitted_used(tmp_path):
    layout = _layout(tmp_path)
    used_uncommitted = _raw(layout, "sweep", "20261028-000001", default=True)  # render-read, not in git
    dead_untracked = _raw(layout, "sweep", "20261028-000002", default=False)  # dead, not in git
    dead_tracked = _raw(layout, "timing", "20261028-000003", default=True)  # dead, in git
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", str(dead_tracked)], check=True)

    rep = rawprune.audit(layout)
    assert rep.used == [str(used_uncommitted)]
    assert rep.untracked == [str(dead_untracked)]
    assert rep.missing == [str(used_uncommitted)]

    rep = rawprune.prune(layout)
    assert rep.removed == [str(dead_tracked)]
    assert not dead_tracked.exists()
    assert dead_untracked.exists()  # the untracked dead is reported, never deleted
    assert used_uncommitted.exists()
