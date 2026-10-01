"""Perses + Prometheus observability stack for the vLLM server.

Same contract as litellm: **independent** of the vLLM server — ``perses
down`` leaves the server running, and ``serve down`` leaves the stack up.
Prometheus scrapes the server's /metrics through
``host.docker.internal:<port>`` (the resolved port, same as
``mjolnir serve``), so the stack works whether the server is up or down.
The two vLLM dashboards (vendored from vllm-project/vllm,
``docker/perses/dashboards/``) are PROVISIONED into Perses from a read-only
folder — no manual import.

All user-persisted files live in one dir — ``$MJOLNIR_PERSER_DATA`` or
``$MJOLNIR_DATA/perses`` (default ``~/.local/share/mjolnir/perses``): the
env file, the rendered prometheus.yml, and the stack data (``db/`` = Perses
file DB, ``prom/`` = Prometheus TSDB).

The compose machinery (env file, data dir, up/down/status/logs, rollback,
health-wait) lives in :mod:`mjolnir.stack`; this module is the stack's glue:
the spec, the prometheus.yml rendering, the health URL, the banner.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mjolnir import stack
from mjolnir.config import Settings, find_repo_root

SPEC = stack.StackSpec(
    name="perses",
    compose_rel=Path("docker/perses/docker-compose.yaml"),
    env_file_var="MJOLNIR_PERSER_ENV",
    data_dir_var="MJOLNIR_PERSER_DATA",
    data_dir_env="PERSER_DATA_DIR",
    data_subdirs=("db", "prom"),
    port_env="PERSER_PORT",
    default_port=8080,
    wait_timeout_s=300.0,
)
DEFAULT_PORT = SPEC.default_port
PROMETHEUS_DEFAULT_PORT = 9090


class PersesError(stack.StackError):
    pass


# ── prometheus.yml rendering ───────────────────────────────────────────────


def vllm_metrics_target(s: Settings, override: str | None = None) -> str:
    """The Prometheus scrape target for the VLLM server as ``host:port``
    (prometheus target form — the path comes from ``metrics_path``): the
    resolved port on host.docker.internal (the host-published port the
    vLLM container listens on), or the ``PERSER_VLLM_TARGET`` env-file
    override."""
    return override or f"host.docker.internal:{s.port}"


def render_prometheus(s: Settings, vllm_target: str | None = None) -> Path:
    """Render ``prometheus.template.yaml`` → ``data_dir()/prometheus.yml``
    with the vLLM scrape target of the active config."""
    tpl = find_repo_root() / SPEC.compose_rel.parent / "prometheus.template.yaml"
    if not tpl.exists():
        raise PersesError(f"prometheus template not found: {tpl}")
    rendered, missing = stack.render(tpl.read_text(), {"VLLM_METRICS_TARGET": vllm_metrics_target(s, vllm_target)})
    if missing:
        raise PersesError(f"unresolved variable(s) in the prometheus template: {', '.join(missing)}")
    try:
        yaml.safe_load(rendered)
    except yaml.YAMLError as e:
        raise PersesError(f"rendered prometheus config invalid: {e}".replace("\n", " "))
    out = SPEC.data_dir() / "prometheus.yml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    return out


# ── stack glue (the compose driver is mjolnir.stack) ───────────────────────


def _port_of(env: dict[str]) -> int:
    try:
        return int(env.get("PERSER_PORT", DEFAULT_PORT))
    except ValueError:
        return DEFAULT_PORT


def _prom_port_of(env: dict[str]) -> int:
    try:
        return int(env.get("PROMETHEUS_PORT", PROMETHEUS_DEFAULT_PORT))
    except ValueError:
        return PROMETHEUS_DEFAULT_PORT


def up(
    s: Settings, port: int | None = None, wait: bool = True, wait_timeout_s: float = 300.0, dry_run: bool = False
) -> int:
    """Start the stack (prometheus + perses) with the vLLM dashboards
    provisioned. The vLLM scrape target = the active config's port (the
    server may be up or down)."""
    file_env = stack.load_env_file(SPEC.env_file_path())
    target = vllm_metrics_target(s, file_env.get("PERSER_VLLM_TARGET"))
    extra: dict[str, str] = {}
    if port:
        extra["PERSER_PORT"] = str(port)

    cfg_path: Path | None = None

    def prepare(s2: Settings) -> dict[str, str]:
        nonlocal cfg_path
        for sub in SPEC.data_subdirs:
            d = SPEC.data_dir() / sub
            d.mkdir(parents=True, exist_ok=True)
            # the containers run nonroot (perses uid 65532, prometheus
            # 65534) and need to write their state into these host dirs
            d.chmod(0o777)
        cfg_path = render_prometheus(s2, file_env.get("PERSER_VLLM_TARGET"))
        if dry_run:
            print(f"(rendered prometheus config: {cfg_path})")
        return extra

    def health(env: dict[str]) -> str:
        return f"http://127.0.0.1:{_port_of(env)}/api/v1/projects"

    def banner(env: dict[str]) -> list[str]:
        p, pp = _port_of(env), _prom_port_of(env)
        return [
            f"perses:     http://127.0.0.1:{p}  (dashboards: Performance Statistics, Query Statistics)",
            f"prometheus: http://127.0.0.1:{pp}  (scrapes {target}/metrics — targets: http://127.0.0.1:{pp}/targets)",
        ]

    try:
        stack.up(
            SPEC,
            s,
            prepare=prepare,
            wait=wait,
            wait_timeout_s=wait_timeout_s,
            dry_run=dry_run,
            health=health,
            banner=banner,
        )
    except stack.StackError as e:
        raise PersesError(str(e))
    return 0


def down(purge: bool = False, dry_run: bool = False) -> int:
    """Stop the stack (the vLLM server keeps running; data kept by default,
    ``purge`` wipes the perses db + prometheus TSDB)."""
    return stack.down(SPEC, purge=purge, dry_run=dry_run, keep_note="prometheus TSDB + perses db survive")


def _rendered_target() -> str | None:
    """The scrape target the rendered ``prometheus.yml`` actually points at
    (what the *running* stack is scraping) — the source of truth, which may
    differ from the settings-resolved port (e.g. after an ``up`` with a
    different port). None if not yet rendered / unreadable."""
    yml = SPEC.data_dir() / "prometheus.yml"
    if not yml.exists():
        return None
    try:
        data = yaml.safe_load(yml.read_text())
    except yaml.YAMLError:
        return None
    for job in (data or {}).get("scrape_configs", []):
        for sc in job.get("static_configs", []):
            targets = sc.get("targets") or []
            if targets:
                return targets[0]
    return None


def status(s: Settings, port: int | None = None) -> dict:
    env = stack.resolve_env(SPEC, {"PERSER_PORT": str(port)} if port else None)
    p, pp = _port_of(env), _prom_port_of(env)
    target = _rendered_target() or vllm_metrics_target(
        s, stack.load_env_file(SPEC.env_file_path()).get("PERSER_VLLM_TARGET")
    )
    services = stack.project_services(SPEC)
    live = any(v.get("state") == "running" for v in services.values())
    return stack.status(
        SPEC,
        extra={
            "perses": {
                "url": f"http://127.0.0.1:{p}",
                "api": stack.probe(f"http://127.0.0.1:{p}/api/v1/projects") if live else "down",
            },
            "prometheus": {
                "url": f"http://127.0.0.1:{pp}",
                "health": stack.probe(f"http://127.0.0.1:{pp}/-/healthy") if live else "down",
            },
            "vllm_target": f"http://{target}/metrics",
            "data_dir": str(SPEC.data_dir()),
        },
    )


def logs(follow: bool = False, tail: int = 100, service: str | None = None) -> int:
    """Tail a stack service's logs (default ``perses``; ``--service
    prometheus`` for the scraper)."""
    return stack.logs(SPEC, follow=follow, tail=tail, service=service)


def running_port() -> int | None:
    """The Perses UI port if the stack is up (the ``serve status`` hint),
    else None."""
    svc = stack.project_services(SPEC).get(SPEC.name) or {}
    if svc.get("state") != "running":
        return None
    return _port_of(stack.resolve_env(SPEC))
