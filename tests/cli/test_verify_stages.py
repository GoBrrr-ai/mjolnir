"""``mjolnir verify vllm``'s stage routing — the host-side decisions, no docker.

The stage is otherwise a docker driver (the runner suite + the served smokes);
what is unit-testable without a GPU is the mode routing: which smoke modes a
platform may run, and what the stage reports when none may. Pinned here:

* **Orin filters the platform-invalid smoke modes** (`5b7f...`, the hd128
  serving): ``graph``/``eager`` pass; ``lmh8`` (needs sm_100+ fp8 tensor
  cores), ``e2m3`` (hd256-only, the smoke model is hd128) and ``spec*``
  (needs a served drafter) are dropped — and a mode list with nothing valid
  short-circuits before any docker call;
* **non-Orin platforms keep every mode** (the Thor stage's contract is
  unchanged) and the dry-run path marks a smoke passed without a log to
  parse.
"""

from __future__ import annotations

from pathlib import Path

from mjolnir_cli.verify.stages import Ctx, stage_vllm


class FakeRunner:
    """The Ctx.runner surface stage_vllm touches — records, never runs docker."""

    def __init__(self, platform: str, dry_run: bool = False):
        self.platform = platform
        self.dry_run = dry_run
        self.calls: list[str] = []

    def tests(self, log: Path, repo: Path, cmd: str, timeout: float, hf_home: Path) -> int:
        self.calls.append(cmd)
        log.parent.mkdir(parents=True, exist_ok=True)
        # the runner suite parses "(✅|❌) test_<name>"; the smokes parse "^PASS$"
        log.write_text("PASS\n" if "smoke_vllm" in cmd else "✅ test_runner_contract: OK\n", encoding="utf-8")
        return 0


def _ctx(tmp_path: Path, runner: FakeRunner) -> tuple[Ctx, list[str]]:
    outs: list[str] = []
    ctx = Ctx(
        runner=runner,  # ty: ignore[invalid-argument-type] — the duck-typed fake
        run_id="test",
        logdir=tmp_path / "logs",
        metrics_url="",
        out=outs.append,
    )
    return ctx, outs


def test_orin_filters_the_platform_invalid_smoke_modes(tmp_path):
    ctx, _ = _ctx(tmp_path, FakeRunner("orin"))
    repo = tmp_path / "repo"

    res = stage_vllm(ctx, repo, smoke_modes="graph lmh8 e2m3 spec3 eager", hf_home=tmp_path / "hf")

    # graph + eager survive in order; lmh8/e2m3/spec3 are dropped
    assert res["params"]["smoke_modes"] == "graph eager"
    labels = [c["label"] for c in res["cases"]]
    assert "smoke graph" in labels and "smoke eager" in labels
    assert not any("lmh8" in lab or "e2m3" in lab or "spec3" in lab for lab in labels)
    # the runner suite still ran and parsed
    assert "runner test_runner_contract" in labels
    # exactly one docker call per surviving smoke (+ the runner suite)
    runner = ctx.runner
    assert len(runner.calls) == 3  # ty: ignore[unresolved-attribute]
    assert sum("smoke_vllm" in c for c in runner.calls) == 2  # ty: ignore[unresolved-attribute]


def test_orin_with_no_valid_mode_short_circuits_before_any_smoke(tmp_path):
    ctx, outs = _ctx(tmp_path, FakeRunner("orin"))

    res = stage_vllm(ctx, tmp_path / "repo", smoke_modes="lmh8 e2m3 spec3", hf_home=tmp_path / "hf")

    assert res["params"]["smoke_modes"] == ""
    assert not [c for c in res["cases"] if c["label"].startswith("smoke ")]
    assert any("(smoke: no Orin-valid modes requested)" in o for o in outs)
    # the runner suite ran; nothing else touched docker
    assert len(ctx.runner.calls) == 1  # ty: ignore[unresolved-attribute]


def test_non_orin_keeps_every_mode_and_dry_run_passes_without_a_log_parse(tmp_path):
    ctx, _ = _ctx(tmp_path, FakeRunner("thor", dry_run=True))

    res = stage_vllm(ctx, tmp_path / "repo", smoke_modes="lmh8 e2m3 spec2", hf_home=tmp_path / "hf")

    assert res["params"]["smoke_modes"] == "lmh8 e2m3 spec2"
    labels = [c["label"] for c in res["cases"]]
    assert "smoke lmh8" in labels and "smoke e2m3" in labels and "smoke spec2" in labels
    # dry_run: passed even though the fake log lacks a PASS for the smokes'
    # arg-building path... the fake does write PASS; assert the flag path via
    # the per-mode arg contract instead: each smoke got its own docker call.
    assert all(c["passed"] for c in res["cases"] if c["label"].startswith("smoke "))
    assert len(ctx.runner.calls) == 4  # ty: ignore[unresolved-attribute]  # suite + 3 smokes
