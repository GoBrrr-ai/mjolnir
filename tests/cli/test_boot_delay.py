"""Boot delay: vLLM containers hold until the host has been up long enough to
stop them by hand after a power cycle (thor-arena/mjolnir#27)."""

import subprocess

import pytest

from mjolnir_cli import dockerctl
from mjolnir_cli.config import Settings


@pytest.fixture(autouse=True)
def _no_hf_cache(monkeypatch):
    monkeypatch.delenv("MJOLNIR_HF_CACHE", raising=False)


def test_default_is_two_minutes(monkeypatch):
    monkeypatch.delenv("MJOLNIR_BOOT_DELAY_S", raising=False)
    head = dockerctl.boot_delay_prefix()
    assert "/proc/uptime" in head and "-lt 120" in head and "sleep $((120 - up))" in head


def test_zero_disables(monkeypatch):
    monkeypatch.setenv("MJOLNIR_BOOT_DELAY_S", "0")
    assert dockerctl.boot_delay_prefix() == ""


def test_serve_and_one_shot_commands_start_with_it(monkeypatch, tmp_path):
    monkeypatch.setenv("MJOLNIR_BOOT_DELAY_S", "90")
    serve = dockerctl.build_run_args(Settings(data_dir=tmp_path))[-1]
    one_shot = dockerctl.build_run_args(Settings(data_dir=tmp_path), command="true")[-1]
    for cmd in (serve, one_shot):
        assert cmd.startswith(dockerctl.boot_delay_prefix())
    assert "vllm serve " in serve


@pytest.mark.parametrize("uptime, slept", [("30.5 1.0", True), ("500.2 1.0", False)])
def test_shell_waits_only_early_in_boot(tmp_path, monkeypatch, uptime, slept):
    """Run the generated shell with /proc/uptime and sleep swapped for fakes."""
    monkeypatch.setenv("MJOLNIR_BOOT_DELAY_S", "120")
    up = tmp_path / "uptime"
    up.write_text(uptime)
    head = dockerctl.boot_delay_prefix().replace("/proc/uptime", str(up))
    script = f"sleep() {{ echo SLEPT $1; }}; {head} echo STARTED"
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert out.rstrip().endswith("STARTED")
    assert ("SLEPT 90" in out) is slept
