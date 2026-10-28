"""A stopped mjolnir-vllm (left by a crash/reboot) must not block `serve up`."""

from mjolnir_cli import cachectl, config, dockerctl


def _fake_docker(monkeypatch, state: str | None):
    """state: None = no container, "running" / "exited" otherwise."""
    calls: list[list[str]] = []

    def run_quiet(args, ok_codes=(0, 1)):
        calls.append(args)
        if args[1] == "inspect":
            if state is None:
                return 1, ""
            fmt = args[3]
            return 0, ("true" if state == "running" else "false") if "Running" in fmt else state
        return 0, ""

    monkeypatch.setattr(dockerctl, "_run_quiet", run_quiet)
    monkeypatch.setattr(dockerctl, "_docker", lambda: "docker")
    monkeypatch.setattr(cachectl, "prepare_bench", lambda: None)
    return calls


def _rm_calls(calls):
    return [c for c in calls if c[1] == "rm"]


def test_container_exists(monkeypatch):
    _fake_docker(monkeypatch, None)
    assert not dockerctl.container_exists()
    _fake_docker(monkeypatch, "exited")
    assert dockerctl.container_exists()
    assert not dockerctl.container_running()


def test_serve_up_removes_stopped_container(monkeypatch, tmp_path):
    calls = _fake_docker(monkeypatch, "exited")
    ran: list[list[str]] = []
    monkeypatch.setattr(dockerctl, "_run", lambda args, **kw: ran.append(args))
    monkeypatch.setattr(dockerctl, "missing_kv_scales", lambda s: None)
    monkeypatch.setattr(dockerctl, "build_run_args", lambda s, serve_args=(), extra_env=(): ["docker", "run"])
    s = config.Settings(data_dir=tmp_path)
    dockerctl.serve_up(s, wait=False)
    assert _rm_calls(calls) == [["docker", "rm", "-f", dockerctl.CONTAINER_NAME]]
    assert ran == [["docker", "run"]]


def test_serve_up_fresh_host_removes_nothing(monkeypatch, tmp_path):
    calls = _fake_docker(monkeypatch, None)
    monkeypatch.setattr(dockerctl, "_run", lambda args, **kw: None)
    monkeypatch.setattr(dockerctl, "missing_kv_scales", lambda s: None)
    monkeypatch.setattr(dockerctl, "build_run_args", lambda s, serve_args=(), extra_env=(): ["docker", "run"])
    dockerctl.serve_up(config.Settings(data_dir=tmp_path), wait=False)
    assert _rm_calls(calls) == []


def test_serve_down_cleans_stopped_container(monkeypatch, capsys):
    calls = _fake_docker(monkeypatch, "exited")
    dockerctl.serve_down(config.Settings())
    assert _rm_calls(calls) == [["docker", "rm", "-f", dockerctl.CONTAINER_NAME]]
    assert "removed a stopped container" in capsys.readouterr().out
