"""`mjolnir image` — the command bodies in ``commands/image.py``. Docker is
never contacted: the image list is faked (at the helper for the command
tests, at ``subprocess.run`` for the helper's own tests) and the
``build``/``gates`` subprocesses are intercepted. The picker paths run
non-interactively."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mjolnir_cli import cli, tasks
from mjolnir_cli.commands import image

runner = CliRunner()

IMAGES = ["narandill/mjolnir-vllm:v0.32.0", "vllm/vllm-openai:v0.32.0-ubuntu2404", "narandill/mjolnir-vllm:v0.31.0"]


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "state.json"))
    return tmp_path


@pytest.fixture
def _images(monkeypatch):
    monkeypatch.setattr(image, "_local_images", lambda: list(IMAGES))


@pytest.fixture
def _tty(monkeypatch):
    """The picker paths need a tty: CliRunner's stdin is a pipe."""

    class _FakeStdin:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", _FakeStdin())


def _state_path() -> Path:
    return Path(os.environ["MJOLNIR_STATE"])


def _docker_images_proc(lines: list[str], rc: int = 0):
    p = subprocess.CompletedProcess([], rc)
    p.stdout = "\n".join(lines) + "\n"
    return p


def test_local_images_dedupes_and_keeps_order(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: _docker_images_proc(["a:1", "b:2", "a:1", "<none>:<none>", "", "a:1"])
    )
    assert image._local_images() == ["a:1", "b:2"]


def test_local_images_is_empty_when_docker_is_missing(monkeypatch):
    def boom(*a, **kw):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", boom)
    assert image._local_images() == []


def test_image_use_explicit_image_persists_to_the_state_file(_images):
    r = runner.invoke(cli.app, ["image", "use", "vllm/vllm-openai:v0.32.0-ubuntu2404"])
    assert r.exit_code == 0, r.output
    assert "active image: vllm/vllm-openai:v0.32.0-ubuntu2404" in r.output
    assert json.loads(_state_path().read_text()) == {"image": "vllm/vllm-openai:v0.32.0-ubuntu2404"}


def test_image_use_unknown_image_fails_naming_the_count(_images):
    r = runner.invoke(cli.app, ["image", "use", "nope:latest"])
    assert r.exit_code == 4 and "not found locally (3 image(s)" in r.output


def test_image_bare_invocation_is_the_picker_which_refuses_a_non_tty(_images):
    r = runner.invoke(cli.app, ["image"])
    assert r.exit_code == 4 and "non-interactive session" in r.output
    assert "narandill/mjolnir-vllm:v0.32.0" in r.output  # the list was printed


def test_image_list_marks_the_active_one(_images):
    assert runner.invoke(cli.app, ["image", "use", "narandill/mjolnir-vllm:v0.31.0"]).exit_code == 0
    r = runner.invoke(cli.app, ["image", "list"])
    assert r.exit_code == 0, r.output
    assert "● narandill/mjolnir-vllm:v0.31.0  (active)" in r.output
    assert "  vllm/vllm-openai:v0.32.0-ubuntu2404" in r.output


def test_image_build_dry_run_prints_the_docker_command():
    r = runner.invoke(cli.app, ["image", "build", "--tag", "mine:v1", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "docker build -f" in r.output and "docker/vllm-thor/Dockerfile" in r.output
    assert "MJOLNIR_REVISION=" in r.output  # the checkout's revision is passed
    assert "-t mine:v1" in r.output


def test_image_build_outside_a_repo_names_the_pull(monkeypatch):
    class _NoRepo:
        def require_repo(self, what: str):
            raise FileNotFoundError(f"{what} needs a mjolnir checkout (run from inside one or set $MJOLNIR_REPO)")

    monkeypatch.setattr(image, "load_layout", lambda: _NoRepo())
    r = runner.invoke(cli.app, ["image", "build"])
    assert r.exit_code == 4 and "docker pull" in r.output


def test_image_gates_runs_the_arch_legs(monkeypatch):
    seen = []

    def fake_run_task(s, name, args, dry_run=False):
        seen.append(name)
        return 0

    monkeypatch.setattr(tasks, "run_task", fake_run_task)
    r = runner.invoke(cli.app, ["image", "gates", "--image", "mine:v9", "--arch", "orin", "--dry-run"])
    # fp6-expand is arch-agnostic (the patched-flashinfer expand canary) and
    # rides every board's legs.
    assert r.exit_code == 0 and seen == ["orin-gates", "fp6-expand", "functional"], seen
    seen.clear()
    r = runner.invoke(cli.app, ["image", "gates", "--image", "mine:v9", "--task", "fi-update", "--dry-run"])
    assert r.exit_code == 0 and seen == ["fi-update"], seen


def test_image_use_with_no_local_images_at_all(monkeypatch):
    monkeypatch.setattr(image, "_local_images", list)
    r = runner.invoke(cli.app, ["image", "use", "mine:v1"])
    assert r.exit_code == 4 and "no local docker images found" in r.output


def test_image_bare_picker_picks_and_persists(_images, _tty, monkeypatch):
    # called directly: CliRunner replaces sys.stdin, which would re-hide the tty
    from mjolnir_cli import picker

    monkeypatch.setattr(picker, "select", lambda title, labels, default=None: "vllm/vllm-openai:v0.32.0-ubuntu2404")
    image.image_use(None)  # no Exit raised: the pick won
    assert json.loads(_state_path().read_text()) == {"image": "vllm/vllm-openai:v0.32.0-ubuntu2404"}


def test_image_picker_cancel_keeps_the_active_image(_images, _tty, monkeypatch, capsys):
    import typer

    from mjolnir_cli import picker

    def cancel(*a, **kw):
        raise picker.Cancelled

    monkeypatch.setattr(picker, "select", cancel)
    with pytest.raises(typer.Exit) as ei:
        image.image_use(None)
    assert ei.value.exit_code == 1
    assert "cancelled — active image unchanged" in capsys.readouterr().err
    assert not _state_path().exists()  # nothing was written


def test_image_build_runs_docker_and_suggests_making_it_active(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    r = runner.invoke(cli.app, ["image", "build", "--tag", "mine:v1"])
    assert r.exit_code == 0, r.output
    assert calls and calls[0][0:3] == ["docker", "build", "-f"]  # the repo-root build context is appended
    assert calls[0][-1] == str(Path(calls[0][3]).parents[2])  # context = the Dockerfile's repo root
    assert "built mine:v1" in r.output and "image gates --image mine:v1" in r.output
    assert "make the new build active" in r.output  # the active image is the baked-in default, not mine:v1


def test_image_build_propagates_the_docker_exit_code(monkeypatch):
    monkeypatch.setattr(subprocess, "call", lambda cmd: 17)
    r = runner.invoke(cli.app, ["image", "build", "--tag", "mine:v1"])
    assert r.exit_code == 17
