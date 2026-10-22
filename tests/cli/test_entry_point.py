"""The console-script entry point (``mjolnir_cli.cli:main``).

Review cycle 2, IMPROVEMENT 3: on a host detection cannot name (a GPU-less
CI machine, a laptop, an unknown OEM unit) every settings-resolving command
used to die with a raw Python traceback — only ``platform show``/``hw`` had
bespoke handlers. The entry point wraps ``app()``: an uncaught
:class:`~mjolnir_cli.platforms.UnknownPlatform` prints the probe instructions
to stderr and exits 2, for every command at once.

The tests run the REAL entry point (``sys.argv`` → ``cli.main()``) with the
detection fakes installed: a raw traceback would surface as
``UnknownPlatform`` propagating instead of the asserted ``SystemExit``.
"""

from __future__ import annotations

import sys

import pytest

from mjolnir_cli import cli
from platform_fixtures import install


def _invoke(monkeypatch, argv: list[str], tmp_path) -> None:
    """Run the console-script entry point exactly as an install would."""
    monkeypatch.setattr(sys, "argv", ["mjolnir", *argv])
    monkeypatch.setenv("MJOLNIR_STATE", str(tmp_path / "state.json"))
    cli.main()


@pytest.mark.parametrize(
    "argv",
    [
        ["serve", "up", "--dry-run"],
        ["verify", "sweep", "--dry-run"],
        ["bench", "compare"],
    ],
)
def test_an_unknown_host_fails_cleanly_with_the_probe_instructions(monkeypatch, tmp_path, capsys, argv):
    # every probe finds nothing (a GPU-less CI host); no bespoke handler
    # exists in these command bodies — the entry point is the one handler
    install(monkeypatch, smi=None, jetson=None, dmi=None, model_file=None)
    with pytest.raises(SystemExit) as ei:
        _invoke(monkeypatch, argv, tmp_path)
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "could not detect the platform" in err
    assert "MJOLNIR_PLATFORM" in err  # the escape hatch is named too
    assert "nvidia-smi" in err  # ...along with the probes to run


def test_a_detectable_host_still_runs_normally(monkeypatch, tmp_path, capsys):
    # the wrapper is transparent when detection resolves: click's normal
    # exit 0, nothing on stderr — the handler only fires on UnknownPlatform
    install(monkeypatch, smi="NVIDIA Thor, 11.0")
    with pytest.raises(SystemExit) as ei:
        _invoke(monkeypatch, ["platform", "show"], tmp_path)
    assert ei.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "platform: thor" in captured.out


def test_python_m_entry_point_is_the_same_handler(tmp_path):
    """``python -m mjolnir_cli`` routes through main() too — the module's
    ``if __name__`` block calls sys.exit(main()), not the raw app(). Both -m
    spellings (the package and the module) share that handler."""
    import os
    import subprocess
    import sys

    from mjolnir_cli import __version__

    env = {**os.environ, "MJOLNIR_STATE": str(tmp_path / "state.json")}
    for target in ("mjolnir_cli", "mjolnir_cli.cli"):
        r = subprocess.run([sys.executable, "-m", target, "version"], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == f"mjolnir {__version__}"
