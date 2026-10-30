"""rawprune: the used-set mirrors the render's own selection (one doc per
stage/platform/model group — the newest default) and the prune git rms
exactly the dead ones. The writer (``results.save``) applies the same rule at
write time — placement + retirement — so dead default raws only exist in
legacy directories, which the prune still cleans (``_legacy_raw`` constructs
that state; ``_raw`` goes through the writer)."""

from __future__ import annotations

import json
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


def _doc(
    stage: str,
    run_id: str,
    *,
    default: bool,
    platform: str | None = "orin",
    model: str | None = None,
) -> dict:
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
    return doc


def _raw(layout: RepoLayout, stage: str, run_id: str, **kw) -> Path:
    """Save through the writer — the placement (tracked vs staging) included."""
    return results.save(layout.verify_dir, _doc(stage, run_id, **kw))


def _legacy_raw(layout: RepoLayout, stage: str, run_id: str, **kw) -> Path:
    """Write a raw directly — a pre-writer directory state (two defaults of one
    group present, an ad-hoc raw tracked), the only state the prune has work
    for now that the writer retires and relocates on save."""
    p = layout.verify_dir / stage / f"{run_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_doc(stage, run_id, **kw), indent=1) + "\n")
    return p


def _git_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)


def test_prune_keeps_only_the_render_read(tmp_path):
    layout = _layout(tmp_path)
    new = _raw(layout, "sweep", "20261028-000002", default=True)  # the headline
    old = _legacy_raw(layout, "sweep", "20261028-000001", default=True)  # superseded (legacy dir)
    adhoc = _raw(layout, "sweep", "20261028-000003", default=False)  # an ad-hoc A/B → staging
    timing = _raw(layout, "timing", "20261028-000004", default=True)  # uncharted stage → staging
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
    assert {str(old), str(broken), str(orphan)} <= dead
    assert not rep.missing  # everything render-read is committed

    rep = rawprune.prune(layout)
    assert not Path(old).exists() and not broken.exists() and not orphan.exists()
    assert Path(new).exists() and Path(other).exists()
    assert not rep.untracked and not rep.missing
    # the writer's placement: experiments and uncharted stages never enter the tracked dir
    assert adhoc == layout.staging_dir / "verify" / "sweep" / "20261028-000003.json"
    assert timing == layout.staging_dir / "verify" / "timing" / "20261028-000004.json"


def test_untracked_dead_and_uncommitted_used(tmp_path):
    layout = _layout(tmp_path)
    used_uncommitted = _raw(layout, "sweep", "20261028-000001", default=True)  # render-read, not in git
    dead_untracked = _legacy_raw(layout, "sweep", "20261028-000002", default=False)  # dead, not in git
    dead_tracked = _legacy_raw(layout, "timing", "20261028-000003", default=True)  # dead, in git
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


def test_save_replaces_the_previous_default_evidence(tmp_path):
    layout = _layout(tmp_path)
    old = _raw(layout, "sweep", "20261028-000001", default=True)
    new = _raw(layout, "sweep", "20261028-000002", default=True)
    assert not old.exists()  # retired at write time — git history keeps the old file
    assert new == layout.verify_dir / "sweep" / "20261028-000002.json"
    # a different model or platform is its own evidence group: untouched
    other_model = _raw(layout, "sweep", "20261028-000003", default=True, model="other-model")
    other_plat = _raw(layout, "sweep", "20261028-000004", default=True, platform="thor")
    assert other_model.exists() and other_plat.exists()
    # a model-less document belongs to the headline model's group (the render's
    # _matches rule): saving it explicitly retires the model-less doc and
    # vice versa — one evidence file per (stage, platform, model)
    explicit = _raw(layout, "sweep", "20261028-000005", default=True, model="Qwen/Qwen3.8-27B")
    assert not new.exists()
    model_less = _raw(layout, "sweep", "20261028-000006", default=True)
    assert not explicit.exists() and model_less.exists()
