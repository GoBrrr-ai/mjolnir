"""The CLI-side platform registry and detection — the data layer that makes
the CLI model- and platform-agnostic.

One frozen :class:`Platform` per shipped platform. The registry is the source
of truth for the image constants :mod:`mjolnir_cli.config` re-exports (the
thor record *is* today's constants); research platforms carry no image of
ours yet, so their default image is the upstream one.

Detection is source-class based (task/research/detection.md): four
:class:`PlatformSource` classes — ``nvidia-smi`` (the only one visible in
every container), ``jetson-files``, ``dmi`` and the jetson-containers model
copy ``/tmp/nv_jetson_model`` — each probe to a raw string, mapped through
:func:`platform_for_raw`. Two source classes agreeing on a platform is
``confirmed``; a single class is ``single-source``; no evidence (or
contradictory evidence) raises :class:`UnknownPlatform` naming the probes —
never a silent wrong-platform guess. The model copy is operator-controlled
(world-writable ``/tmp`` — security audit LOW): it probes last and its vote
needs another class's corroboration. ``MJOLNIR_PLATFORM`` always overrides.
Stdlib only: no torch, and the filesystem/subprocess probes run inside
``probe()`` (never at import).
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

DEFAULT_PLATFORM = "thor"


class UnknownPlatform(Exception):
    """The host could not be identified (no evidence, or contradictory
    evidence) — the message names the probes and the ``MJOLNIR_PLATFORM``
    escape hatch instead of guessing."""


@dataclass(frozen=True)
class Platform:
    """One shipped platform: its architecture, images and maturity."""

    key: str
    name: str
    sm: tuple[int, int]
    sm_family: str
    image_repo: str
    default_image: str
    stock_image: str | None
    vllm_version: str | None
    status: str  # derived from _IMPLEMENTED (the kernels' shipped table): "supported" | "research" — hand-set only for exotic states (e.g. "not_supported")
    # Tool capability: does the platform's rootfs/image ship Nsight Compute?
    # False platforms refuse the ncu-profiled stages with a clear error and
    # carry their headline data in the events-timed stages instead.
    ncu: bool = True
    # Renderer-facing data (cycle 3a): the host software the platform needs.
    # No renderer consumes it yet; no behavior hangs on it.
    host_requirements: str = ""
    # Renderer-facing data (cycle 3e): the short label for tables whose
    # columns are platforms (the config README's serving-profile table).
    short: str = ""
    # Renderer-facing data (cycle 3k): the platform's pinned headline model —
    # the pair its kernel index and `## End to end` section render for. Empty
    # until a platform has a measured model (thor: the repo's default model —
    # the same value as config.DEFAULT_MODEL, kept literal here because the
    # registry cannot import config).
    headline_model: str = ""
    # Renderer-facing data: the memory system the bandwidth charts draw as the
    # roofline (the effective-KV-bandwidth chart's peak line, its y-limit and
    # its cache-size subtitle). 0 / "" = unknown — chart_bandwidth falls back
    # to PEAK_GBS / "LPDDR5X" / "the L2" (the Thor values). thor: LPDDR5X
    # 273 GB/s, 32 MiB L2; orin: LPDDR5 204.8 GB/s, 4 MB L2; spark: LPDDR5X
    # 273 GB/s (GB10), GPU L2 unsourced.
    mem_bandwidth_gbs: float = 0.0
    mem_label: str = ""
    l2: str = ""
    # Renderer-facing data: the platform's emoji gif under assets/emoji/ (the
    # root README's platform strip). Empty = no dedicated art yet — the strip
    # falls back to the generic Mjolnir hammer (mjolnir_emoji.gif).
    emoji: str = ""


# The kernels' shipped table, mirrored from ``mjolnir.platforms._SHIPPED``
# (the packages cannot import each other — the sm rows AND the implemented
# flags are pinned in sync by
# tests/cli/test_kernel_platforms.py::test_kernel_and_cli_registries_agree).
# The status strings derive from it: a platform whose kernel is implemented
# serves, every other row is research. Flip the flag in the kernels' code
# and here, and every doc updates from the one source.
_IMPLEMENTED: dict[str, bool] = {
    "thor": True,
    "spark": False,
    "orin": True,
}


def _status(key: str) -> str:
    return "supported" if _IMPLEMENTED[key] else "research"


REGISTRY: dict[str, Platform] = {
    "thor": Platform(
        key="thor",
        name="Jetson AGX Thor",
        sm=(11, 0),
        sm_family="sm_110",
        image_repo="narandill/mjolnir-vllm",
        default_image="narandill/mjolnir-vllm:v0.32.0",
        stock_image="vllm/vllm-openai:v0.32.0-ubuntu2404",
        vllm_version="0.32.0",
        status=_status("thor"),
        host_requirements="JetPack 7.x",
        short="Thor",
        headline_model="Qwen/Qwen3.8-27B",
        emoji="mjolnir_thor_emoji.gif",
        mem_bandwidth_gbs=273.0,
        mem_label="LPDDR5X",
        l2="the 32 MiB L2",
    ),
    # Spark: no Mjolnir kernel/image exists yet (baselines only until one
    # lands), so the upstream image is the default.
    "spark": Platform(
        key="spark",
        name="DGX Spark",
        sm=(12, 1),
        sm_family="sm_121",
        image_repo="narandill/mjolnir-spark",
        default_image="vllm/vllm-openai:v0.32.0-ubuntu2404",
        stock_image="vllm/vllm-openai:v0.32.0-ubuntu2404",
        vllm_version=None,
        status=_status("spark"),
        host_requirements="DGX OS + CUDA 13",
        short="Spark",
        emoji="mjolni_spark_emoji.gif",
        mem_bandwidth_gbs=273.0,
        mem_label="LPDDR5X",
    ),
    "orin": Platform(
        key="orin",
        name="Jetson Orin",
        sm=(8, 7),
        sm_family="sm_87",
        # One image serves every platform; the tag follows the upstream vLLM
        # version. The stock image stays the baselines-only side.
        image_repo="narandill/mjolnir-vllm",
        default_image="narandill/mjolnir-vllm:v0.32.0",
        stock_image="vllm/vllm-openai:v0.32.0-ubuntu2404",
        vllm_version=None,
        status=_status("orin"),
        ncu=True,  # JP7 ships Nsight Compute (nsight-compute-2026.1.1 in the r39.2 repo)
        host_requirements="JetPack 7.2.1+ (L4T 39.2.1, CUDA 13.2)",
        short="Orin",
        # measured on the Orin (the sweep, kv-cache-size and the e2e pair are
        # all this model): the pair its kernel index and `## End to end`
        # section render for
        headline_model="Qwen/Qwen3.8-27B",
        emoji="mjolnir_orin_emoji.gif",
        mem_bandwidth_gbs=204.8,
        mem_label="LPDDR5",
        l2="the 4 MB L2",
    ),
}


# Renderer-facing data (cycle 3g): the root README's hardware table renders
# the status as icon + word. Keyed by status — a new status needs a new entry
# here, never a platform-name branch.
STATUS_ICON: dict[str, str] = {"supported": "✅", "research": "⏳", "not_supported": "❌"}


def get(key: str) -> Platform:
    """The record for ``key``; KeyError names the shipped platforms."""
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown platform {key!r} — shipped platforms: {', '.join(REGISTRY)}") from None


def platform_ncu(platform: str | None) -> bool:
    """Does the platform ship Nsight Compute? Unresolved platforms assume
    yes (the default-platform table)."""
    if platform is None:
        return True
    return REGISTRY[platform].ncu


def _validate(key: str) -> str:
    """A platform key from a flag or the env; ValueError names the shipped keys."""
    if key not in REGISTRY:
        raise ValueError(f"unknown platform {key!r} — shipped platforms: {', '.join(REGISTRY)}")
    return key


# ── the sources ─────────────────────────────────────────────────────────────


class PlatformSource(Protocol):
    """One independent evidence class: a ``name`` and a probe that returns the
    raw value (or None when it finds nothing) and never raises."""

    name: str

    def probe(self) -> str | None: ...


def _read_first(read: Callable[[Path], str | None], paths: Sequence[str]) -> str | None:
    """The content of the first readable path (None when there is none)."""
    for p in paths:
        try:
            content = read(Path(p))
        except OSError:  # unreadable is a normal absence here, never an error
            content = None
        if content:
            return content
    return None


class NvidiaSmiSource:
    """The GPU-driver source: ``nvidia-smi`` name and compute capability,
    queried *separately* — a combined query dies entirely on drivers that do
    not know the compute_cap field. One instance runs each query at most once
    (``platform show`` shares one spawn between detect() and gpu_name());
    a fresh instance re-queries."""

    name: str = "nvidia-smi"
    _NAME: ClassVar[list[str]] = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    _CC: ClassVar[list[str]] = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]

    def __init__(self, run: Callable[..., subprocess.CompletedProcess] | None = None) -> None:
        self._run = run  # injectable for tests; subprocess.run is looked up at probe time
        self._raw: str | None = None
        self._asked = False

    def probe(self) -> str | None:
        if not self._asked:
            self._asked = True
            name = self._first_field(self._NAME)
            if name is not None:  # no name row: the driver is unusable either way
                cc = self._cc(self._CC)
                self._raw = f"{name}, {cc}" if cc else name
        return self._raw

    def _first_field(self, argv: list[str]) -> str | None:
        row = self._row(argv)
        return row.split(",")[0].strip() if row else None

    def _cc(self, argv: list[str]) -> str | None:
        row = self._row(argv)
        if row is None:
            return None
        m = re.search(r"\d+\.\d+", row)  # the cc, wherever the driver puts it
        return m.group(0) if m else None

    def _row(self, argv: list[str]) -> str | None:
        run = self._run if self._run is not None else subprocess.run
        try:
            p = run(argv, capture_output=True, text=True, check=False)
        except OSError:  # nvidia-smi absent (or not executable): no evidence
            return None
        if p.returncode != 0:
            return None
        out = (p.stdout or "").strip()
        return out.splitlines()[0] if out else None


class JetsonFilesSource:
    """The Jetson board-file source: device-tree model/compatible and the L4T
    release — root-owned kernel/OS interfaces. Host-only on an unmounted
    Jetson — their absence in a container is not evidence of absence (the raw
    value simply names what it read). The jetson-containers model copy
    (``/tmp/nv_jetson_model``) is NOT here: it is world-writable, so it is a
    separate, corroboration-only source (see :class:`JetsonModelFileSource`)."""

    name: str = "jetson-files"
    PATHS: ClassVar[tuple[str, ...]] = (
        "/proc/device-tree/compatible",
        "/proc/device-tree/model",
        "/etc/nv_tegra_release",
    )

    def __init__(self, read: Callable[[Path], str | None] | None = None) -> None:
        self._read: Callable[[Path], str | None] = read or _default_reader

    def probe(self) -> str | None:
        parts = []
        for p in self.PATHS:
            content = self._read(Path(p))
            if content:
                parts.append(content.replace("\0", " ").strip())  # device-tree strings are NUL-terminated
        return "\n".join(parts) if parts else None


class DmiSource:
    """The DMI / DGX-OS source (Spark, also readable in-container): the DMI
    product name, else ``/etc/dgx-release``."""

    name: str = "dmi"
    PATHS: ClassVar[tuple[str, ...]] = ("/sys/class/dmi/id/product_name", "/etc/dgx-release")

    def __init__(self, read: Callable[[Path], str | None] | None = None) -> None:
        self._read: Callable[[Path], str | None] = read or _default_reader

    def probe(self) -> str | None:
        return _read_first(self._read, self.PATHS)


class JetsonModelFileSource:
    """The jetson-containers model copy (``/tmp/nv_jetson_model``) — an
    operator-controlled hint, NOT a full evidence class: the file lives in
    world-writable ``/tmp`` (the cycle-2 security audit LOW), so any local
    user could plant one. ``corroboration_required`` makes its vote count
    only when another source class names the same platform — alone it is
    provenance-only and can neither confirm, contradict nor steer detection."""

    name: str = "nv_jetson_model"
    corroboration_required: bool = True
    PATHS: ClassVar[tuple[str, ...]] = ("/tmp/nv_jetson_model",)

    def __init__(self, read: Callable[[Path], str | None] | None = None) -> None:
        self._read: Callable[[Path], str | None] = read or _default_reader

    def probe(self) -> str | None:
        content = _read_first(self._read, self.PATHS)
        return content.replace("\0", " ").strip() if content else None


def _default_reader(path: Path) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


# The per-instance singletons the CLI resolves through (the probe cache lives
# on the instances, so a process asks the host at most once per source). The
# /tmp model copy probes last: an operator-controlled hint, corroborates only.
DEFAULT_SOURCES: tuple[PlatformSource, ...] = (
    NvidiaSmiSource(),
    JetsonFilesSource(),
    DmiSource(),
    JetsonModelFileSource(),
)


# ── the mapping table ───────────────────────────────────────────────────────


def normalize(raw: str) -> str:
    """NVIDIA's own rule (TensorRT-Edge-LLM detect.py): ``[_-]+`` becomes a
    space — ``NVIDIA_DGX_Spark`` and an OEM ``DGX Spark`` normalize alike."""
    return re.sub(r"[_\-]+", " ", raw or "").strip().casefold()


# name substrings (normalized), compute capability, board-file markers and
# L4T release majors per platform. Names match with `contains`, never `==`
# (the Tegra driver appends " (nvgpu)"; Thor's device-tree model string
# differs between factory and BSP firmware — match markers, not the model).
_MARKERS: dict[str, dict] = {
    "thor": {
        "names": ("thor",),
        "cc": "11.0",
        "boards": ("tegra264", "p3834", "thor"),
        "l4t": (38, 39),
    },
    "orin": {
        "names": ("orin",),
        "cc": "8.7",
        "boards": ("tegra234", "p3701", "p3767", "orin"),
        "l4t": (35, 36),
    },
    "spark": {
        "names": ("gb10",),
        "cc": "12.1",
        "boards": ("dgx spark", "gb10"),
        "l4t": (),
    },
}

_SMI_CSV = re.compile(r",\s*(\d+\.\d+)\s*$")  # "<name>, <cc>" — the nvidia-smi CSV form
_L4T = re.compile(r"#\s*R(\d+)")


def _platform_for_name(name: str) -> str | None:
    norm = normalize(name)
    for plat, m in _MARKERS.items():
        if any(n in norm for n in m["names"]):
            return plat
    return None


def platform_for_raw(raw: str | None) -> str | None:
    """The platform a raw probe value names (None when it names none).

    An ``nvidia-smi`` CSV cross-checks the GPU name against the compute
    capability: contradiction is a hard error — the error names both raw
    values — never a silent guess."""
    if raw is None:
        return None
    m = _SMI_CSV.search(raw)
    if m is not None:
        cc = m.group(1)
        name_plat = _platform_for_name(raw[: m.start()])
        cc_plat = next((p for p, mk in _MARKERS.items() if mk["cc"] == cc), None)
        if name_plat is not None and cc_plat is not None and name_plat != cc_plat:
            raise UnknownPlatform(
                f"contradictory GPU identity {raw!r}: the name maps to {name_plat!r} "
                f"but compute capability {cc} maps to {cc_plat!r} — shipped platforms: {', '.join(REGISTRY)}"
            )
        return name_plat or cc_plat
    norm = normalize(raw)
    for plat, mk in _MARKERS.items():
        if any(b in norm for b in mk["boards"]):
            return plat
    l4t = _L4T.search(raw)
    if l4t:
        major = int(l4t.group(1))
        for plat, mk in _MARKERS.items():
            if major in mk["l4t"]:
                return plat
    return None


# ── detection ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Detection:
    """The resolved platform plus how sure the detection is: ``confirmed``
    (two source classes agree), ``single-source`` (one class) or
    ``env-override`` (``MJOLNIR_PLATFORM`` — no probes at all).
    ``provenance`` is one ``(source name, raw value)`` pair per source that
    probed non-None, in source order (a show-time concern only)."""

    platform: str
    confidence: str  # "confirmed" | "single-source" | "env-override"
    provenance: tuple[tuple[str, str], ...]


_PROBES = (
    "nvidia-smi --query-gpu=name --format=csv,noheader (and --query-gpu=compute_cap separately)",
    "jetson-files (/proc/device-tree/compatible, /proc/device-tree/model, /etc/nv_tegra_release)",
    "dmi (/sys/class/dmi/id/product_name, /etc/dgx-release)",
    "nv_jetson_model (/tmp/nv_jetson_model — operator-controlled: corroborates another source class, never one alone)",
)


def detect(sources: Sequence[PlatformSource] | None = None) -> Detection:
    """The platform this host is, from the given source classes (default
    ``DEFAULT_SOURCES``). Two source classes agreeing is ``confirmed``; one is
    ``single-source``; no (or contradictory) evidence raises
    :class:`UnknownPlatform` naming the probes. Sources that set
    ``corroboration_required`` (the world-writable ``/tmp`` model copy) vote
    only alongside another class: alone they are provenance-only and can
    neither confirm nor contradict. ``MJOLNIR_PLATFORM`` beats all of it — an
    explicit operator is never second-guessed, and never probed."""
    env = os.environ.get("MJOLNIR_PLATFORM", "").strip()
    if env:
        return Detection(_validate(env), "env-override", ())
    srcs = DEFAULT_SOURCES if sources is None else sources
    provenance: list[tuple[str, str]] = []
    votes: dict[str, list[tuple[str, str]]] = {}
    hints: dict[str, list[tuple[str, str]]] = {}  # corroboration-only sources
    for src in srcs:
        raw = src.probe()
        if raw is None:
            continue
        provenance.append((src.name, raw))
        plat = platform_for_raw(raw)
        if plat is None:
            continue
        if getattr(src, "corroboration_required", False):
            hints.setdefault(plat, []).append((src.name, raw))
        else:
            votes.setdefault(plat, []).append((src.name, raw))
    for plat, named in hints.items():
        if plat in votes:  # a hint joins a trusted vote; it never contradicts or steers one
            votes[plat].extend(named)
    if not votes:
        raise UnknownPlatform(
            "could not detect the platform on this host — probes that find nothing: "
            + "; ".join(_PROBES)
            + ". Set MJOLNIR_PLATFORM to one of: "
            + ", ".join(REGISTRY)
        )
    if len(votes) > 1:
        parts = [f"{plat} ← {'; '.join(raw for _, raw in named)}" for plat, named in votes.items()]
        raise UnknownPlatform(
            "contradictory platform evidence on this host — "
            + " | ".join(parts)
            + ". Set MJOLNIR_PLATFORM to one of: "
            + ", ".join(REGISTRY)
        )
    plat, named = next(iter(votes.items()))
    confidence = "confirmed" if len(named) >= 2 else "single-source"
    return Detection(plat, confidence, tuple(provenance))


def resolve_platform(cli_flag: str | None) -> str:
    """The active platform: the ``--platform`` flag > env ``MJOLNIR_PLATFORM``
    > host detection. A bad flag or env value raises ``ValueError`` naming the
    shipped keys; an unknown host raises :class:`UnknownPlatform` naming the
    probes. There is deliberately no silent fallback to the default platform."""
    if cli_flag:
        return _validate(cli_flag.strip())
    env = os.environ.get("MJOLNIR_PLATFORM", "").strip()
    if env:
        return _validate(env)
    return detect().platform


def gpu_name(sources: Sequence[PlatformSource] | None = None) -> str:
    """The GPU model string for the record's ``gpu`` field (``""`` when there
    is none — detection here degrades, never raises). With no ``sources``, a
    fresh nvidia-smi source probes (callers that want to share one spawn with
    :func:`detect` pass their sources — see ``platform show``)."""
    for src in sources if sources is not None else (NvidiaSmiSource(),):
        if src.name == NvidiaSmiSource.name:
            raw = src.probe()
            return raw.split(",")[0].strip() if raw else ""
    return ""


def platform_of(rec: Mapping) -> str:
    """The platform a record belongs to: its explicit ``platform`` field, else
    ``"thor"`` — the repo's entire committed history is Thor, so platform-less
    records resolve there. Orthogonal to ``model_of``: model-less is not
    platform-less. Unknown values pass through unvalidated."""
    return str(rec.get("platform") or "thor")


def series_key(rec: Mapping) -> tuple[str, str, str]:
    """``(platform, model, config)`` — the identity a chart series belongs to."""
    return (platform_of(rec), str(rec.get("model") or ""), str(rec.get("config") or ""))


def stock_image(platform: str, env_override: str | None) -> str | None:
    """The reference (official vLLM) image for a platform: ``env_override``
    (``$MJOLNIR_REF_IMAGE``) wins over the record's stock image."""
    return env_override or get(platform).stock_image
