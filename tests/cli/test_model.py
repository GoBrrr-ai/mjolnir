"""`mjolnir model` — the command bodies in ``commands/model.py`` (plus the
``_common`` state file they sit on). The config scan is faked; the picker
paths run non-interactively (no tty → the honest error, exit 4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mjolnir_cli import cli
from mjolnir_cli.commands import _common, model
from mjolnir_cli.config import ConfigEntry

runner = CliRunner()

ENTRIES = [
    ConfigEntry(model="Qwen/Qwen3.8-27B", quant="NVFP4", path=Path("/x/NVFP4/config.yaml")),
    ConfigEntry(model="Qwen/Qwen3.8-27B", quant="NVFP4_MJOLNIR_E2M3", path=Path("/x/E2M3/config.yaml")),
    ConfigEntry(model="nvidia/Qwen3.6-35B-A3B", quant="NVFP4_DFLASH", path=Path("/x/DFLASH/config.yaml")),
]


@pytest.fixture(autouse=True)
def _state_and_configs(tmp_path, monkeypatch):
    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "state.json"))
    monkeypatch.setattr(model, "load_layout", lambda: type("L", (), {})())
    monkeypatch.setattr(model, "scan_all_configs", lambda layout: ENTRIES)
    return tmp_path


@pytest.fixture
def _tty(monkeypatch):
    """The picker paths need a tty: CliRunner's stdin is a pipe."""

    class _FakeStdin:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", _FakeStdin())


def test_model_use_remembers_an_explicit_config():
    r = runner.invoke(cli.app, ["model", "use", "Qwen/Qwen3.8-27B", "NVFP4_MJOLNIR_E2M3"])
    assert r.exit_code == 0, r.output
    assert "active: Qwen/Qwen3.8-27B / NVFP4_MJOLNIR_E2M3" in r.output
    st = json.loads(_state_path().read_text())
    assert st == {"model": "Qwen/Qwen3.8-27B", "quant": "NVFP4_MJOLNIR_E2M3"}


def _state_path() -> Path:
    import os

    return Path(os.environ["MJOLNIR_STATE"])


def test_model_use_filters_by_one_argument_then_picks():
    # one arg narrows the pool to one candidate → chosen without a picker
    r = runner.invoke(cli.app, ["model", "use", "nvidia/Qwen3.6-35B-A3B"])
    assert r.exit_code == 0, r.output
    assert "NVFP4_DFLASH" in r.output


def test_model_use_unknown_config_fails_naming_the_list():
    r = runner.invoke(cli.app, ["model", "use", "Qwen/Qwen3.8-27B", "NOPE"])
    assert r.exit_code == 4 and "config not found" in r.output


def test_model_use_no_match_for_the_given_filters():
    r = runner.invoke(cli.app, ["model", "use", "vendor/Missing"])
    assert r.exit_code == 4 and "no config matches" in r.output


def test_model_bare_invocation_is_the_picker_which_refuses_a_non_tty():
    # bare `mjolnir model` runs the PickerGroup's bare_fn; without a tty the
    # honest error wins (the table is printed first, then exit 4)
    r = runner.invoke(cli.app, ["model"])
    assert r.exit_code == 4 and "non-interactive session" in r.output
    assert "NVFP4_MJOLNIR_E2M3" in r.output  # the config table was printed


def test_model_list_prints_the_table_and_the_active_marker():
    _common.save_state({"model": "Qwen/Qwen3.8-27B", "quant": "NVFP4"})
    r = runner.invoke(cli.app, ["model", "list"])
    assert r.exit_code == 0, r.output
    assert "NVFP4_MJOLNIR_E2M3" in r.output and "● active: Qwen/Qwen3.8-27B / NVFP4" in r.output
    assert "no active config yet" not in r.output


def test_model_list_json_emits_the_records(tmp_path):
    r = runner.invoke(cli.app, ["model", "list", "--json"])
    assert r.exit_code == 0, r.output
    doc = json.loads(r.output)
    assert doc["active"] == {"model": None, "quant": None}
    assert [c["quant"] for c in doc["configs"]] == ["NVFP4", "NVFP4_MJOLNIR_E2M3", "NVFP4_DFLASH"]
    assert Path(doc["state_file"]) == _state_path()


def test_model_list_says_when_there_is_nothing(monkeypatch):
    monkeypatch.setattr(model, "scan_all_configs", lambda layout: [])
    r = runner.invoke(cli.app, ["model", "list"])
    assert r.exit_code == 0 and "no configs found" in r.output
    r = runner.invoke(cli.app, ["model", "use"])
    assert r.exit_code == 4 and "no configs found" in r.output


# ── the _common state file ──────────────────────────────────────────────────


def test_save_state_merges_and_load_state_reads_it_back(tmp_path, monkeypatch):
    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "sub" / "state.json"))
    _common.save_state({"model": "Qwen/Qwen3.8-27B"})
    _common.save_state({"quant": "NVFP4"})  # merge, not replace
    assert _common.load_state() == {"model": "Qwen/Qwen3.8-27B", "quant": "NVFP4"}
    assert (tmp_path / "sub" / "state.json").exists()  # parents created


def test_load_state_survives_a_corrupt_file(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    p.write_text("{not json")
    monkeypatch.setenv("MJOLNIR_STATE", str(p))
    assert _common.load_state() == {}


def test_load_state_is_empty_when_the_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "absent.json"))
    assert _common.load_state() == {}


def test_settings_resolves_from_the_state_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "state.json"))
    _common.save_state({"model": "Qwen/Qwen3.8-27B", "quant": "NVFP4_MJOLNIR_E2M3"})
    s = _common.settings(None, None, None, None)
    assert (s.model, s.quant) == ("Qwen/Qwen3.8-27B", "NVFP4_MJOLNIR_E2M3")
    s = _common.settings("Qwen/Qwen3.8-27B", "NVFP4", None, None)  # explicit flags win
    assert s.quant == "NVFP4"


def test_model_picker_picks_and_persists(_tty, monkeypatch):
    # called directly: CliRunner replaces sys.stdin, which would re-hide the tty

    from mjolnir_cli import picker

    monkeypatch.setattr(picker, "select", lambda title, labels, default=None: "Qwen/Qwen3.8-27B / NVFP4")
    model.model_use(None, None)  # no Exit raised: the pick won
    st = json.loads(_state_path().read_text())
    assert st == {"model": "Qwen/Qwen3.8-27B", "quant": "NVFP4"}


def test_model_partial_args_with_several_candidates_go_through_the_picker(_tty, monkeypatch):
    from mjolnir_cli import picker

    monkeypatch.setattr(picker, "select", lambda title, labels, default=None: "Qwen/Qwen3.8-27B / NVFP4")
    model.model_use("Qwen/Qwen3.8-27B", None)  # two candidates for the model → pick
    st = json.loads(_state_path().read_text())
    assert st["quant"] == "NVFP4"


def test_model_picker_cancel_keeps_the_active_config(_tty, monkeypatch, capsys):
    import typer

    from mjolnir_cli import picker

    def cancel(*a, **kw):
        raise picker.Cancelled

    monkeypatch.setattr(picker, "select", cancel)
    with pytest.raises(typer.Exit) as ei:
        model.model_use(None, None)
    assert ei.value.exit_code == 1
    assert "cancelled — active config unchanged" in capsys.readouterr().err
    assert not _state_path().exists()


def test_model_list_flags_the_local_configs_dir(monkeypatch):
    entries = [
        ConfigEntry(model="vendor/M", quant="A", path=Path("/x/a"), source="repo"),
        ConfigEntry(model="vendor/M", quant="B", path=Path("/x/b"), source="local"),
    ]
    monkeypatch.setattr(model, "scan_all_configs", lambda layout: entries)
    r = runner.invoke(cli.app, ["model", "list"])
    assert r.exit_code == 0 and "local = " in r.output


def test_model_list_without_an_active_config_names_the_command():
    r = runner.invoke(cli.app, ["model", "list"])
    assert r.exit_code == 0 and "no active config yet — run 'mjolnir model'" in r.output
