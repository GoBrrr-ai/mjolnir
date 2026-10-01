"""LiteLLM proxy stack: ``litellm`` + ``postgres`` + ``redis``.

Independent of the vLLM server — ``mjolnir litellm down`` leaves the vLLM
running, and ``mjolnir serve down`` leaves the proxy up. The proxy reaches
the vLLM through ``host.docker.internal`` (the host-published vLLM port), so
the two never share a docker network and neither knows the other's
container name.

The config is rendered on every ``mjolnir litellm up`` from the user's
template (``$MJOLNIR_LITELLM_TEMPLATE`` or, by default, inside the litellm
data dir: ``~/.local/share/mjolnir/litellm/litellm_config.template.yaml``)
with the *currently selected* model/config (``mjolnir model``):
``${MAIN_LLM_MODEL}`` = the served name, ``${MAIN_LLM_MODEL_NAME}`` = the raw
``model:`` field, ``${RAW_MODEL_SUFFIXED}`` = ``<model>-<quant>``.
Everything else resolves from the user's env file (``$MJOLNIR_LITELLM_ENV``,
default ``~/.local/share/mjolnir/litellm/litellm.env``).

ALL user-persisted litellm files live in one dir — ``$MJOLNIR_DATA/litellm``
(default ``~/.local/share/mjolnir/litellm``): the template, the env file,
the rendered config, and the stack data (postgres + redis).

The compose machinery (env file, data dir, up/down/status/logs, rollback,
health-wait) lives in :mod:`mjolnir.stack`; this module is the stack's glue:
the spec, the template rendering, the model-variable injection.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from mjolnir import stack
from mjolnir.config import Settings, _config_field

SPEC = stack.StackSpec(
    name="litellm",
    compose_rel=Path("docker/litellm/docker-compose.yaml"),
    env_file_var="MJOLNIR_LITELLM_ENV",
    data_dir_var="MJOLNIR_LITELLM_DATA",
    data_dir_env="LITELLM_DATA_DIR",
    data_subdirs=("db", "redis"),
    port_env="LITELLM_PORT",
    default_port=8000,
    wait_timeout_s=600.0,
)
DEFAULT_PROXY_PORT = SPEC.default_port


class LitellmError(stack.StackError):
    pass


# ── paths ──────────────────────────────────────────────────────────────────


def template_path() -> Path:
    """The user's template: ``$MJOLNIR_LITELLM_TEMPLATE`` or, by default,
    in the litellm data dir:
    ``~/.local/share/mjolnir/litellm/litellm_config.template.yaml``."""
    env = os.environ.get("MJOLNIR_LITELLM_TEMPLATE")
    if env:
        return Path(env).expanduser()
    return data_dir() / "litellm_config.template.yaml"


def env_file_path() -> Path:
    """The user's env file (optional): ``$MJOLNIR_LITELLM_ENV`` or, by
    default, in the litellm data dir:
    ``~/.local/share/mjolnir/litellm/litellm.env``."""
    return SPEC.env_file_path()


def data_dir() -> Path:
    """Stack data root: ``$MJOLNIR_LITELLM_DATA`` or
    ``$MJOLNIR_DATA/litellm`` (default ``~/.local/share/mjolnir/litellm``);
    holds the template/env/rendered config + the postgres/redis state
    dirs."""
    return SPEC.data_dir()


def compose_file() -> Path:
    return SPEC.compose_file()


# ── env loading + rendering ────────────────────────────────────────────────


def _model_field(s: Settings) -> str:
    """The raw ``model:`` field of the config yaml (falls back to the model
    path) — the ``MAIN_LLM_MODEL_NAME`` the template's alias table wants."""
    return _config_field(s.config_path, "model") or s.model


def injected_env(s: Settings, override_base_url: str | None = None) -> dict[str, str]:
    """The variables mjolnir injects from the active model/config."""
    name = _model_field(s)
    return {
        "MAIN_LLM_MODEL": s.served_model,
        "MAIN_LLM_MODEL_NAME": name,
        "RAW_MODEL_SUFFIXED": f"{name}-{s.quant}",
        "LITELLM_MAIN_MODEL_BASE_URL": override_base_url or f"http://host.docker.internal:{s.port}/v1",
    }


def resolved_env(s: Settings, proxy_port: int | None = None) -> dict[str, str]:
    """Merged env for rendering + compose interpolation:
    user env file < injected model variables (the CLI selection wins) <
    stack defaults (data dir, proxy port)."""
    env = stack.load_env_file(env_file_path())
    extra = injected_env(s, override_base_url=env.get("LITELLM_MAIN_MODEL_BASE_URL"))
    # CLI flag > env file > default
    if proxy_port:
        extra["LITELLM_PORT"] = str(proxy_port)
    return stack.resolve_env(SPEC, extra)


def validate(rendered: str) -> list[str]:
    """Problems with a rendered config (empty list = ok)."""
    problems: list[str] = []
    try:
        doc = yaml.safe_load(rendered)
    except yaml.YAMLError as e:
        return [f"rendered config is not valid YAML: {e}".replace("\n", " ")]
    ml = doc.get("model_list") if isinstance(doc, dict) else None
    if not isinstance(ml, list):
        problems.append("no model_list in the rendered config")
    else:
        for i, e in enumerate(ml):
            name = (e or {}).get("model_name")
            if not name or not str(name).strip():
                problems.append(f"model_list[{i}] has an empty model_name — check the template variables")
    return problems


def render_config(s: Settings, proxy_port: int | None = None) -> Path:
    """Render the template → ``data_dir()/litellm_config.yaml``; raises
    ``LitellmError`` on a missing template or unresolved variables."""
    tpl = template_path()
    if not tpl.exists():
        example = compose_file().parent / "litellm_config.template.example.yaml"
        msg = (
            f"template not found: {tpl}\n"
            "  start from the example (or override the location with "
            "$MJOLNIR_LITELLM_TEMPLATE):\n"
            f"  cp {example} {tpl}"
        )
        legacy = Path.home() / "litellm_config.template.yaml"
        if legacy.exists() and legacy != tpl:
            msg += f"\n  found a legacy copy — move it:  mv {legacy} {tpl}"
        raise LitellmError(msg)
    env = resolved_env(s, proxy_port)
    rendered, missing = stack.render(tpl.read_text(), env)
    if missing:
        src = env_file_path()
        raise LitellmError(
            "unresolved template variable(s): "
            + ", ".join("${" + m + "}" for m in missing)
            + f"\n  define them in {src} (see the examples in "
            f"{compose_file().parent})"
        )
    problems = validate(rendered)
    if problems:
        raise LitellmError("rendered config invalid: " + "; ".join(problems))
    out = data_dir() / "litellm_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    return out


# ── stack glue (the compose driver is mjolnir.stack) ───────────────────────


def _port_of(env: dict[str]) -> int:
    try:
        return int(env.get("LITELLM_PORT", DEFAULT_PROXY_PORT))
    except ValueError:
        return DEFAULT_PROXY_PORT


def up(
    s: Settings, proxy_port: int | None = None, wait: bool = True, wait_timeout_s: float = 600.0, dry_run: bool = False
) -> int:
    """Render the config and start the stack (proxy + postgres + redis)."""
    cfg_path: Path | None = None

    def prepare(s2: Settings) -> dict[str, str]:
        nonlocal cfg_path
        cfg_path = render_config(s2, proxy_port)
        if dry_run:
            print(f"(rendered config: {cfg_path})")
        return {}

    def health(env: dict[str]) -> str:
        return f"http://127.0.0.1:{_port_of(env)}/health/liveliness"

    def banner(env: dict[str]) -> list[str]:
        p = _port_of(env)
        return [
            f"litellm proxy: http://127.0.0.1:{p} (master key: {env.get('LITELLM_MASTER_KEY', 'sk-1234')})",
            f"  UI:  http://127.0.0.1:{p}/ui"
            f"  ({env.get('LITELLM_UI_USERNAME', 'admin')}/"
            f"{env.get('LITELLM_UI_PASSWORD', 'admin')})",
            f"  vllm main model: {env['MAIN_LLM_MODEL']} @ {env['LITELLM_MAIN_MODEL_BASE_URL']}",
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
            compose_wait=True,
        )
    except stack.StackError as e:
        env = resolved_env(s, proxy_port)
        hint = (
            f"\n  is host port {_port_of(env)} already taken? try: mjolnir litellm up --port <other>"
            if "port is already allocated" in str(e)
            else ""
        )
        raise LitellmError(str(e) + hint)
    return 0


def down(purge: bool = False, dry_run: bool = False) -> int:
    """Stop the stack (data kept; ``purge`` wipes postgres + redis data)."""
    return stack.down(SPEC, purge=purge, dry_run=dry_run, keep_note="postgres/redis state survives")


def status(s: Settings, proxy_port: int | None = None) -> dict:
    env = resolved_env(s, proxy_port)
    port = _port_of(env)
    url = f"http://127.0.0.1:{port}"
    return stack.status(
        SPEC,
        extra={
            "proxy": {
                "url": url,
                "liveliness": stack.probe(url + "/health/liveliness"),
                "config": str(data_dir() / "litellm_config.yaml"),
                "template": str(template_path()),
            },
        },
    )


def logs(follow: bool = False, tail: int = 100) -> int:
    return stack.logs(SPEC, follow=follow, tail=tail)
