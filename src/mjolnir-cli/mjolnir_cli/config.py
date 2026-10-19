"""Mjolnir defaults + config resolution.

Everything the CLI needs has a baked-in default (Qwen3.8-27B NVFP4 served by
the Mjolnir image with ``--attention-backend MJOLNIR``); flags on each
command override them, then the remembered state (``mjolnir model`` /
``mjolnir image`` → ``$MJOLNIR_STATE``, default ``~/.mjolnir-state.json``),
then ``$MJOLNIR_*`` env vars. No ``.env`` file required.

The CLI runs from a repo checkout (``uv run mjolnir`` — configs, raws and
charts are the repo's own) or from a PyPI install (``uv tool install
mjolnir-cli`` — configs and the side-stack files come bundled in the wheel,
benchmark results land in ``~/.local/share/mjolnir/benchmarks``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# platform_of / series_key: re-exported beside model_of so existing imports
# stay one stop apart (tests pin the identity: config.platform_of is platforms.platform_of)
from mjolnir_cli.platforms import (
    DEFAULT_PLATFORM,
    REGISTRY,
    resolve_platform,
    platform_of as platform_of,
    series_key as series_key,
)

# ── Model / config ───────────────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"


def model_of(rec: dict) -> str | None:
    """The model a history / accuracy record belongs to; None when the record
    does not say. There is deliberately no default: a guessed model is how runs
    of two models get compared, so a model-less record matches no model."""
    return rec.get("model") or None


def by_model(recs) -> dict[str | None, list[dict]]:
    """Records grouped by :func:`model_of`, models and records in first-seen order."""
    groups: dict[str | None, list[dict]] = {}
    for r in recs:
        groups.setdefault(model_of(r), []).append(r)
    return groups


def model_short(rec: dict) -> str:
    """'Qwen3.8-27B' for 'Qwen/Qwen3.8-27B': the model as labels show it."""
    m = model_of(rec)
    return m.split("/")[-1] if m else "unknown model"


# context depths every end-to-end bench (perf / ab / e2e) measures by default:
# up to 128K (the ncu kernel stages keep their 256K top: verify ALL_LS)
E2E_DEPTHS: tuple[int, ...] = (0, 4096, 8192, 16384, 32768, 65536, 131072)
# the fastest Mjolnir config so far (fp8 lm_head + 64K draft vocab, opt-in via its .env):
# measured next to the default everywhere (verify spec / kv-cache-size / profile, bench ab / accuracy)
FAST_QUANT = "NVFP4_MJOLNIR_E2M3_LMH8_VT64K"
DEFAULT_QUANT = "NVFP4_MJOLNIR_E2M3_LMH8_VT64K"  # --attention-backend MJOLNIR (this repo's kernel)
BASELINE_QUANT = "NVFP4"  # FlashInfer — the fastest stock backend on Thor
FA2_QUANT = "NVFP4_FA2"  # FlashAttention 2 — vLLM 0.32.0's sm_110 default (bf16 KV)
SERVED_MODEL_NAME = "Qwen/Qwen3.8-27B"

# ── Image / serving ──────────────────────────────────────────────────────────
# The image tag follows the upstream vLLM version it overlays (see
# .github/workflows/docker-image.yml): narandill/mjolnir-vllm:v<vllm>.
# The default platform's record (mjolnir_cli.platforms) is the source of truth
# for these constants; they are re-exported here for every existing consumer.
# The literal VLLM_VERSION line below is grepped by docker-image.yml (`sed
# 's/^VLLM_VERSION = "..."'`) to sync with the Dockerfile's BASE_IMAGE — keep
# it equal to the record's vllm_version (tests pin both sides).
VLLM_VERSION = "0.32.0"
_thor = REGISTRY[DEFAULT_PLATFORM]
IMAGE_REPO = _thor.image_repo
DEFAULT_IMAGE = _thor.default_image
STOCK_IMAGE = _thor.stock_image
DEFAULT_PORT = 6001  # host port; vLLM metrics live on the same port
CONTAINER_NAME = "mjolnir-vllm"

# Friendly backend labels for the history log / charts, keyed by quant config.
BACKEND_LABELS = {
    "NVFP4_MJOLNIR": "Mjolnir",
    "NVFP4_MJOLNIR_E2M3": "Mjolnir fp6 KV",
    "NVFP4_MJOLNIR_E2M3_LMH8_VT64K": "Mjolnir fp6 KV + fp8 lm_head + 64K draft vocab",
    "NVFP4_MJOLNIR_E2M3_DSPARK": "Mjolnir fp6 KV + DSpark",
    "NVFP4_MJOLNIR_E2M3_DFLASH2": "Mjolnir fp6 KV + DFlash2",
    "NVFP4": "FlashInfer",
    "NVFP4_DFLASH": "FlashInfer + DFlash",
    "NVFP4_DSPARK": "FlashInfer + DSpark",
    "NVFP4_DFLASH2": "FlashInfer + DFlash2",
    "NVFP4_FA2": "FlashAttention 2",
    "NVFP4_FA4hd256": "FlashAttention 4 hd256",
    "AWQ_FLASHINFER": "AWQ int4 + FlashInfer",
}

# The repo marker: a checkout has the image build context.
_REPO_MARKER = Path("docker") / "vllm-thor" / "Dockerfile"


@dataclass(frozen=True)
class ConfigEntry:
    """One model config discovered under ``configs/<vendor>/<model>/<quant>/config.yaml``
    (the repo's / bundled configs, or the user-local dir — see ``source``)."""

    model: str  # "vendor/Model", e.g. "Qwen/Qwen3.8-27B"
    quant: str  # config name, e.g. "NVFP4_MJOLNIR"
    path: Path
    source: str = "repo"  # "repo" (configs/ or bundled) | "local" (~/.local/share/mjolnir/configs)

    @property
    def backend(self) -> str:
        return BACKEND_LABELS.get(self.quant, self.quant)


def scan_configs(configs_dir: Path) -> list[ConfigEntry]:
    """Scan ``configs/<vendor>/<model>/<quant>/config.yaml`` into pickable
    entries. Everything config-specific lives in the config dir: the
    ``.env`` sidecar and a drafter's ``kv_scales.json``; the model-level files
    (shared ``kv_scales.json``, ``draft_vocab.json``, chat template) stay
    next to it."""
    out: list[ConfigEntry] = []
    if not configs_dir.is_dir():
        return out
    return [
        ConfigEntry(model=str(p.parent.parent.relative_to(configs_dir)), quant=p.parent.name, path=p)
        for p in sorted(configs_dir.glob("*/*/*/config.yaml"))
    ]


class HFCacheError(FileNotFoundError):
    """``$MJOLNIR_HF_CACHE`` is set but does not name a usable HF hub cache."""


def host_hf_hub() -> Path | None:
    """The host HF *hub* cache to reuse (``$MJOLNIR_HF_CACHE``), or None when unset.

    Accepts the hub folder (``~/.cache/huggingface/hub``) or the HF_HOME root
    above it (``~/.cache/huggingface`` — its ``hub/`` is used). ``~`` and
    relative paths are resolved. A missing or non-directory path raises
    :class:`HFCacheError` instead of silently falling back to a fresh download.
    """
    raw = os.environ.get("MJOLNIR_HF_CACHE", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise HFCacheError(
            f"MJOLNIR_HF_CACHE={raw}: {path} is not a directory — point it at an HF hub cache "
            "(e.g. ~/.cache/huggingface/hub) or unset it"
        )
    if (path / "hub").is_dir() and not any(path.glob("models--*")):
        return path / "hub"  # the HF_HOME root was given
    return path


def user_configs_dir() -> Path:
    """The user-local model-config root: ``~/.local/share/mjolnir/configs``
    (under the project data dir ``MJOLNIR_DATA``); override with
    ``$MJOLNIR_CONFIGS_DIR``. Layout: ``<vendor>/<model>/<quant>/config.yaml``."""
    return Path(_env("MJOLNIR_CONFIGS_DIR", str(Path.home() / ".local" / "share" / "mjolnir" / "configs"))).expanduser()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def config_field(path: Path, key: str) -> str | None:
    """One top-level scalar from a config yaml (None if absent/unreadable)."""
    try:
        data = yaml.safe_load(Path(path).read_text())
    except Exception:  # noqa: BLE001 - a malformed config is simply not offered # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    v = data.get(key)
    return str(v) if v else None


def config_speculative(path: Path) -> dict | None:
    """The config's speculative decoding as ``{"method": "mtp", "tokens": 3}``
    (None when the config has none or is unreadable)."""
    try:
        data = yaml.safe_load(Path(path).read_text())
    except Exception:  # noqa: BLE001 - a malformed config is simply not offered
        return None
    spec = data.get("speculative-config") if isinstance(data, dict) else None
    if not isinstance(spec, dict) or not spec.get("num_speculative_tokens"):
        return None
    return {"method": str(spec.get("method", "")), "tokens": int(spec["num_speculative_tokens"])}


def config_drafter_on_mjolnir(path: Path) -> bool:
    """Does the config run an external drafter (``speculative-config.model``:
    DSpark, DFlash, EAGLE, ...) on the Mjolnir backend? Only when it says so in
    ``speculative-config.attention_backend`` (MJOLNIR / CUSTOM): vLLM gives a
    DSpark / DFlash drafter its own backend choice — auto unless set there,
    never the target's. Its layers then need fp6 scales of their own,
    calibrated with the config (``<config>.kv_scales.json``)."""
    try:
        data = yaml.safe_load(Path(path).read_text())
    except Exception:  # noqa: BLE001 - a malformed config is simply not offered
        return False
    spec = data.get("speculative-config") if isinstance(data, dict) else None
    if not isinstance(spec, dict) or not spec.get("model"):
        return False
    return str(spec.get("attention_backend") or "").upper() in ("MJOLNIR", "CUSTOM")


def config_attention_backend(path: Path) -> str | None:
    """The attention backend a config selects (``attention-backend`` or
    ``attention-config.backend``), upper-cased; None when unset."""
    try:
        data = yaml.safe_load(Path(path).read_text())
    except Exception:  # noqa: BLE001 - a malformed config is simply not offered
        return None
    if not isinstance(data, dict):
        return None
    v = data.get("attention-backend")
    if not v and isinstance(data.get("attention-config"), dict):
        v = data["attention-config"].get("backend")
    return str(v).upper() if v else None


# ── repo / resources ─────────────────────────────────────────────────────────


def bundled_root() -> Path:
    """Resources shipped inside the wheel (configs, side stacks, HW script)."""
    return Path(__file__).resolve().parent / "bundled"


def find_repo_root() -> Path:
    """Locate the mjolnir repo: $MJOLNIR_REPO wins, else walk up from CWD
    looking for ``docker/vllm-thor/Dockerfile``. Raises FileNotFoundError
    outside a checkout (PyPI installs then run from bundled resources)."""
    env = os.environ.get("MJOLNIR_REPO")
    if env:
        return Path(env)
    cur = Path.cwd().resolve()
    for cand in [cur, *cur.parents]:
        if (cand / _REPO_MARKER).exists():
            return cand
    raise FileNotFoundError(
        "mjolnir repo not found (run from inside a checkout or set $MJOLNIR_REPO); "
        "this command needs the repo — clone https://github.com/thor-arena/mjolnir"
    )


def repo_root_or_none() -> Path | None:
    try:
        return find_repo_root()
    except FileNotFoundError:
        return None


def resource_path(rel: str | Path) -> Path:
    """A repo resource: the checkout's copy when in a checkout (so edits are
    live), else the copy bundled in the wheel."""
    root = repo_root_or_none()
    if root is not None and (root / rel).exists():
        return root / rel
    return bundled_root() / rel


def configs_root() -> Path:
    """The canonical model-config tree (repo ``configs/`` or the bundled copy)."""
    return resource_path("configs")


def scan_all_configs(layout: RepoLayout | None = None) -> list[ConfigEntry]:
    """Every config the CLI can serve: the repo's (or bundled) ``configs/`` +
    the user-local dir. Local configs never shadow a canonical config with
    the same ``(model, quant)`` — the reviewed copy wins."""
    entries = scan_configs(configs_root())
    seen = {(e.model, e.quant) for e in entries}
    for e in scan_configs(user_configs_dir()):
        if (e.model, e.quant) not in seen:
            entries.append(ConfigEntry(model=e.model, quant=e.quant, path=e.path, source="local"))
            seen.add((e.model, e.quant))
    entries.sort(key=lambda e: (e.model, e.quant))
    return entries


@dataclass
class RepoLayout:
    """Where the CLI reads and writes: the checkout (if any), the user data
    dir, and the benchmark tree (raws + history + charts)."""

    repo_root: Path | None
    data_dir: Path
    state_file: Path
    benchmarks_dir: Path
    history_file: Path

    @property
    def is_repo(self) -> bool:
        return self.repo_root is not None

    @property
    def raw_dir(self) -> Path:
        return self.benchmarks_dir / "raw"

    @property
    def verify_dir(self) -> Path:
        """Structured verify results: ``raw/verify/<stage>/<run-id>.json``."""
        return self.raw_dir / "verify"

    @property
    def charts_dir(self) -> Path:
        base = self.repo_root if self.repo_root is not None else self.benchmarks_dir
        return base / "assets" / "benchmarks"

    def require_repo(self, what: str) -> Path:
        if self.repo_root is None:
            raise FileNotFoundError(f"{what} needs a mjolnir checkout (run from inside one or set $MJOLNIR_REPO)")
        return self.repo_root


def load_layout() -> RepoLayout:
    root = repo_root_or_none()
    data = Path(_env("MJOLNIR_DATA", "~/.local/share/mjolnir")).expanduser()
    bench_default = (root / "benchmarks") if root is not None else (data / "benchmarks")
    bench = Path(_env("MJOLNIR_BENCHMARKS", str(bench_default))).expanduser()
    return RepoLayout(
        repo_root=root,
        data_dir=data,
        state_file=Path(_env("MJOLNIR_STATE", str(Path.home() / ".mjolnir-state.json"))),
        benchmarks_dir=bench,
        history_file=Path(_env("MJOLNIR_HISTORY", str(bench / "history.jsonl"))).expanduser(),
    )


@dataclass
class Settings:
    """Resolved runtime settings for a command."""

    model: str = field(default_factory=lambda: _env("MJOLNIR_MODEL", DEFAULT_MODEL))
    quant: str = field(default_factory=lambda: _env("MJOLNIR_QUANT", DEFAULT_QUANT))
    platform: str | None = None  # unresolved until resolve() (flag > env > host detection)
    image: str = field(default_factory=lambda: _env("MJOLNIR_IMAGE", DEFAULT_IMAGE))
    port: int = field(default_factory=lambda: _env_int("MJOLNIR_PORT", DEFAULT_PORT))
    data_dir: Path = field(default_factory=lambda: Path("~/.local/share/mjolnir").expanduser())
    label: str | None = None  # explicit backend-label override (history log); else derived from quant

    @property
    def config_path(self) -> Path:
        """The resolved config file: the canonical
        ``configs/<model>/<quant>/config.yaml`` (repo or bundled) if present,
        else the user-local one, else the canonical path (for the not-found
        error)."""
        canon = configs_root() / self.model / self.quant / "config.yaml"
        if canon.exists():
            return canon
        local = user_configs_dir() / self.model / self.quant / "config.yaml"
        if local.exists():
            return local
        return canon

    @property
    def attention_backend(self) -> str | None:
        return config_attention_backend(self.config_path)

    @property
    def served_model(self) -> str:
        """The name the served model answers to: ``served-model-name`` from
        the config yaml, else its ``model`` field (vLLM's default served
        name), else the baked-in constant."""
        for key in ("served-model-name", "model"):
            v = config_field(self.config_path, key)
            if v:
                return v
        return SERVED_MODEL_NAME

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def metrics_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/metrics"

    @property
    def backend_label(self) -> str:
        """Explicit --label override if set, else the friendly name for the
        quant config (BACKEND_LABELS), else the quant string itself."""
        return self.label or BACKEND_LABELS.get(self.quant, self.quant)


def resolve(
    cli_model: str | None,
    cli_quant: str | None,
    cli_image: str | None,
    cli_port: int | None,
    cli_platform: str | None = None,
) -> Settings:
    """CLI flags > env vars > host detection (there is no remembered
    platform state). An unknown flag or env value raises ``ValueError``
    naming the shipped keys; an unidentifiable host raises
    ``UnknownPlatform`` naming the probes. The image default follows the
    platform: on a non-default platform with no explicit image, its record's
    default image is used instead of the default platform's."""
    platform = resolve_platform(cli_platform)
    s = Settings(platform=platform)
    if cli_model:
        s.model = cli_model
    if cli_quant:
        s.quant = cli_quant
    if cli_image:
        s.image = cli_image
    elif platform != DEFAULT_PLATFORM and s.image == DEFAULT_IMAGE:
        s.image = REGISTRY[platform].default_image  # no $MJOLNIR_IMAGE: the platform's own default
    if cli_port:
        s.port = cli_port
    return s
