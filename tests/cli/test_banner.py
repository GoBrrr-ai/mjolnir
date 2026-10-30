"""The Mjolnir boot banner: committed art, the info block to its right
(version / author / repo / config), ANSI colouring, the emit-once guard and
the standalone module. Pure Python — no CUDA, runs in CI."""

from __future__ import annotations

import subprocess
import sys

from mjolnir import banner as banner_mod
from mjolnir.banner import ART, HEAD_START, banner, emit


def _start(model: bool) -> int:
    k = 4 if model else 3
    return (len(ART) - k) // 2


def test_plain_banner_layout():
    lines = banner("1.2.3", color=False).split("\n")
    assert len(lines) == len(ART)
    i1 = _start(model=False)
    assert lines[i1].endswith("Mjolnir v1.2.3")
    assert lines[i1 + 1].endswith("by Mieszko Syty (ms1design)")
    assert lines[i1 + 2].endswith("https://github.com/GoBrrr-ai/mjolnir")
    for i, line in enumerate(lines):
        if i1 <= i < i1 + 3:
            assert line.startswith(ART[i])
        else:
            assert line == ART[i]
    assert "\033" not in banner("1.2.3", color=False)


def test_plain_banner_model_line():
    lines = banner("1.2.3", color=False, model="NVFP4_MJOLNIR_E2M3_LMH8_VT64K").split("\n")
    i1 = _start(model=True)
    assert lines[i1].endswith("Mjolnir v1.2.3")
    assert lines[i1 + 1].endswith("by Mieszko Syty (ms1design)")
    assert lines[i1 + 2].endswith("https://github.com/GoBrrr-ai/mjolnir")
    assert lines[i1 + 3].endswith("Model: NVFP4_MJOLNIR_E2M3_LMH8_VT64K")
    assert all(line == ART[i] for i, line in enumerate(lines) if not (i1 <= i < i1 + 4))


def test_color_banner():
    esc = "\033"
    head, white, reset = esc + "[1;93m", esc + "[1m", esc + "[0m"
    lines = banner("1.2.3", color=True).split("\n")
    i1 = _start(model=False)
    assert f"{head}Mjolnir{reset}{white} v1.2.3{reset}" in lines[i1]
    assert f"{white}by Mieszko Syty (ms1design){reset}" in lines[i1 + 1]
    assert f"{white}https://github.com/GoBrrr-ai/mjolnir{reset}" in lines[i1 + 2]
    for i, line in enumerate(lines):
        assert reset in line
        if i >= HEAD_START:
            assert line.startswith(head)
        elif i == i1:
            assert head in line  # the "Mjolnir" word only
        else:
            assert head not in line
    # the monochrome rendering is the colour one with the escapes stripped
    stripped = banner("1.2.3", color=True)
    for esc_seq in (head, white, reset):
        stripped = stripped.replace(esc_seq, "")
    assert stripped == banner("1.2.3", color=False)


def _emit_records(caplog) -> list:
    return [r for r in caplog.records if r.name == "vllm.mjolnir"]


def test_emit_once(monkeypatch, caplog):
    monkeypatch.setattr(banner_mod, "_shown", False)
    monkeypatch.delenv("MJOLNIR_CONFIG", raising=False)
    with caplog.at_level("INFO"):
        emit()
        emit()
    records = _emit_records(caplog)
    # three records: the art between two empty-string lines (the vLLM logo's pattern)
    assert len(records) == 3
    assert records[0].getMessage() == "" and records[2].getMessage() == ""
    art = records[1].getMessage()
    assert "Mjolnir v" in art
    assert "Model:" not in art
    # still shown once after the flag is restored
    assert banner_mod._shown


def test_emit_model(monkeypatch, caplog):
    monkeypatch.setattr(banner_mod, "_shown", False)
    monkeypatch.setenv("MJOLNIR_CONFIG", "NVFP4_MJOLNIR_E2M3")
    with caplog.at_level("INFO"):
        emit()
    records = _emit_records(caplog)
    assert len(records) == 3
    assert "Model: NVFP4_MJOLNIR_E2M3" in records[1].getMessage()


def test_standalone_module():
    r = subprocess.run([sys.executable, "-m", "mjolnir.banner"], capture_output=True, text=True, check=True)
    # one empty line on top and bottom, like emit()'s empty log records
    assert r.stdout.startswith("\n") and r.stdout.endswith("\n\n")
    assert "Mjolnir v" in r.stdout
    # piped: not a TTY, so no ANSI escapes
    assert "\033" not in r.stdout
