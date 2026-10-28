"""Serve orchestration: drive the vLLM docker container from Python.

The vLLM server runs in a container because the whole point of this repo is
a *custom* image (the sm_110 patch stack + the Mjolnir decode kernel behind
``--attention-backend MJOLNIR``) — ``mjolnir serve`` replaces the
bash/docker-compose glue, not the container.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

from mjolnir_cli import cachectl, docker_version, overlays
from mjolnir_cli.config import (
    CONTAINER_NAME,
    HFCacheError,
    Settings,
    config_drafter_on_mjolnir,
    config_field,
    config_speculative,
    configs_root,
    host_hf_hub,
    user_configs_dir,
)

# Container-side env (the Thor-specific knobs from the old vllm.yaml service).
_CONTAINER_ENV: list[str] = [
    "DO_NOT_TRACK=1",
    "VLLM_NO_USAGE_STATS=1",
    "MAX_JOBS=4",
    "NVCC_THREADS=2",
    "FLASHINFER_NVCC_THREADS=2",
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600",
    "FLASHINFER_DISABLE_VERSION_CHECK=1",
    "VLLM_CACHE_ROOT=/data/cache",
    "VLLM_ASSETS_CACHE=/data/assets",
    "VLLM_TUNED_CONFIG_FOLDER=/data/kernels-cache",
    "FLASHINFER_JIT_DIR=/data/flashinfer-cache",
    # FlashInfer 0.7 keeps its JIT modules, CuTe-DSL objects and downloaded
    # cubins under $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer (default:
    # $HOME, i.e. the container's throwaway /root): point it at the mounted
    # cache so fp4_gemm_cutlass & co. build once, not on every server start
    "FLASHINFER_WORKSPACE_BASE=/data/flashinfer-cache",
    "TRITON_CACHE_DIR=/data/triton-cache",
    "TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor",
    "TORCH_COMPILE_CACHE_DIR=/data/cache/torch_compile_cache",
    "XDG_CACHE_HOME=/data/cache/xdg",
    "TIKTOKEN_ENCODINGS_BASE=/data/tiktoken",
    "TIKTOKEN_RS_CACHE_DIR=/data/tiktoken",
    "VLLM_LOGGING_LEVEL=INFO",
    "NVIDIA_VISIBLE_DEVICES=all",
    "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics",
    "CUDA_VISIBLE_DEVICES=0",
    "VLLM_TRITON_USE_TD=1",
    "PYTHONUNBUFFERED=1",
    "SAFETENSORS_FAST_GPU=1",
    "NVIDIA_DISABLE_REQUIRE=1",
    "NVIDIA_FORWARD_COMPAT=1",
    "NCCL_P2P_DISABLE=1",
    "ENABLE_TRIATTENTION=0",
    "CUDA_MANAGED_FORCE_DEVICE_ALLOC=1",
    "TORCH_CUDA_ARCH_LIST=11.0a",
    "FLASHINFER_CUDA_ARCH_LIST=11.0a",
    "CUDA_TOOLKIT_PATH=/usr/local/cuda-13",
    "CUTE_DSL_ARCH=sm_110a",
    "TORCH_MATMUL_PRECISION=high",
    "VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel",
    "VLLM_USAGE_SOURCE=production",
    "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas",
    "VLLM_USE_BREAKABLE_CUDAGRAPH=0",
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    # Thor-only host overrides from the old compose file:
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
    "VLLM_MARLIN_USE_ATOMIC_ADD=1",
]

_HOST_MOUNTS = [
    "/etc/nv_tegra_release",  # L4T id (Jetson)
    "/tmp/nv_jetson_model",
    "/etc/localtime",
    "/etc/machine-id",
]


HF_DOWNLOAD_DIR = "/data/models/huggingface"  # vLLM --download-dir in every container


class DockerError(RuntimeError):
    pass


def _docker() -> str:
    if shutil.which("docker") is None:
        raise DockerError("docker not found on PATH")
    docker_version.ensure_docker_version(offer=False)  # one notice per process; no prompt, nothing run
    return "docker"


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, **kw)


def _run_quiet(args: list[str], ok_codes: tuple[int, ...] = (0, 1)) -> tuple[int, str]:
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode not in ok_codes:
        raise DockerError(f"{' '.join(args[:3])}… failed ({p.returncode}): {(p.stderr or p.stdout).strip()[:500]}")
    return p.returncode, p.stdout.strip()


def container_running(name: str = CONTAINER_NAME) -> bool:
    _, out = _run_quiet([_docker(), "inspect", "-f", "{{.State.Running}}", name])
    return out == "true"


def container_exists(name: str = CONTAINER_NAME) -> bool:
    """Running or not: a stopped container (``--restart no``, left behind when
    the host crashed or rebooted mid-serve) still holds the name."""
    code, _ = _run_quiet([_docker(), "inspect", "-f", "{{.State.Status}}", name])
    return code == 0


# Mjolnir reads K/V fastest from head-major pages (each KV head's page is one
# contiguous block): +11-17% decode at long context in `mjolnir verify
# hndsweep`. vLLM allocates that layout (LBHNC) with VLLM_KV_CACHE_LAYOUT=HND;
# both fallbacks (FlashAttention 4, FlashInfer) read it through strides, so
# MJOLNIR configs get it by default.
MJOLNIR_ENV_DEFAULTS = {"VLLM_KV_CACHE_LAYOUT": "HND"}

# fp6 (e2m3) KV needs per-channel scales from `mjolnir calibrate`, stored
# beside the model's configs (mounted at /configs in the container): one file
# per KV format in kv_scales/ (KV_SCALES_DIR).
KV_SCALES_DIR = "kv_scales"


# What the image's installed mjolnir package is built from: `mjolnir image
# build` stamps the checkout's HEAD into this label.
REVISION_LABEL = "org.opencontainers.image.revision"
KERNEL_PATHS = ("src/mjolnir", "docker/vllm-thor")


def stale_image(image: str) -> str | None:
    """Why ``image`` serves an older Mjolnir than this checkout, or None.

    Serving uses the package installed in the image (only verify and
    calibrate overlay the checkout's source), so a checkout that changed
    ``src/mjolnir`` or the patch stack after the build serves old code. None
    when it cannot tell: no checkout, no label (e.g. the official image), or
    a revision git does not know."""
    try:
        from mjolnir_cli.config import find_repo_root

        root = find_repo_root()
    except FileNotFoundError:
        return None
    try:
        rc, rev = _run_quiet(
            [_docker(), "image", "inspect", "-f", '{{ index .Config.Labels "' + REVISION_LABEL + '" }}', image]
        )
    except (DockerError, OSError):  # no docker / no image: nothing to say
        return None
    rev = rev.strip()
    if rc != 0 or not rev or rev in ("unknown", "<no value>"):
        return None
    r = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", rev, "HEAD", "--", *KERNEL_PATHS],
        capture_output=True,
    )
    if r.returncode != 1:  # 0: same code; >1: revision unknown here
        return None
    log = (
        subprocess.run(
            ["git", "-C", str(root), "log", "--oneline", f"{rev}..HEAD", "--", *KERNEL_PATHS],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    head = f"{len(log)} commit(s)" if log else "changes"
    return (
        f"{image} was built from {rev}; {head} to {', '.join(KERNEL_PATHS)} since "
        f"({'; '.join(log[:3])}{' ...' if len(log) > 3 else ''}). It serves the old code — "
        "rebuild: mjolnir image build"
    )


def _kv_format(path: Path) -> str:
    """The calibration file's stem for a config's ``kv-cache-dtype``:
    ``fp6_e2m3`` -> ``e2m3``, ``fp8_e5m2`` -> ``e5m2``, ``fp8`` -> ``fp8``.
    Defaults to the fp6 (e2m3) cache ``mjolnir calibrate`` produces."""
    dtype = config_field(path, "kv-cache-dtype") or "fp6_e2m3"
    return dtype.split("_", 1)[-1] if "_" in dtype else dtype


def _drafter(path: Path) -> str:
    """The drafter file's name: the config's speculative ``method``
    (``dspark`` -> ``dspark.kv_scales.json``). The method, not the drafter's
    marketing name — one method serves several (``dflash``: DFlash,
    DFlash2)."""
    spec = config_speculative(path) or {}
    return str(spec.get("method") or "drafter")


def kv_scales_host_path(s: Settings) -> Path:
    """Where an fp6 config's scales are read from (repo or user-local
    configs dir).

    The calibration is the model's, one file per KV format:
    ``kv_scales/<format>.json`` beside the config dirs — ``e2m3`` for the
    fp6 (e2m3) cache, shared by every config calibrated for that format. A
    config whose external drafter (DSpark, DFlash, ...) runs on the Mjolnir
    backend has layers no other config has: its scales are
    ``kv_scales/<drafter>.kv_scales.json``, written by
    ``mjolnir calibrate --config <config>``."""
    if config_drafter_on_mjolnir(s.config_path):
        return s.config_path.parent.parent / KV_SCALES_DIR / f"{_drafter(s.config_path)}.kv_scales.json"
    return s.config_path.parent.parent / KV_SCALES_DIR / f"{_kv_format(s.config_path)}.json"


def calibrate_targets(s: Settings, out: Path | None, draft_vocab: bool) -> Path:
    """Where one ``mjolnir calibrate`` run writes: the per-format file, or
    the explicit ``--out`` (``--draft-vocab`` writes its one file too). An
    unwritable configs dir (the wheel-installed CLI's bundled copy) moves
    the whole model-relative layout under the user configs dir."""
    if draft_vocab:
        target = out or kv_scales_host_path(s).with_name("draft_vocab.json")
    elif out is not None:
        target = out
    elif config_drafter_on_mjolnir(s.config_path):
        target = s.config_path.parent.parent / KV_SCALES_DIR / f"{_drafter(s.config_path)}.kv_scales.json"
    else:
        target = s.config_path.parent.parent / KV_SCALES_DIR / f"{_kv_format(s.config_path)}.json"
    if os.access(target.parent if target.parent.exists() else Path.home(), os.W_OK):
        return target
    root = user_configs_dir() / s.model
    return root / _model_rel(s, target)


def _model_rel(s: Settings, p: Path) -> Path:
    """``p`` relative to the model's config dir (flat for paths outside it)."""
    try:
        return p.relative_to(s.config_path.parent.parent)
    except ValueError:
        return Path(p.name)


def missing_kv_scales(s: Settings) -> str | None:
    """Why an fp6 config cannot start (its calibration file is missing), or None."""
    if (s.attention_backend or "") not in ("MJOLNIR", "CUSTOM") or os.environ.get("MJOLNIR_ATTN_KV_SCALES"):
        return None
    if not (config_field(s.config_path, "kv-cache-dtype") or "").startswith("fp6"):
        return None
    path = kv_scales_host_path(s)
    if path.exists():
        return None
    return f"{s.quant} serves fp6 KV but {path} is missing — calibrate it first: mjolnir calibrate --config {s.quant}"


def backend_env(s: Settings) -> list[str]:
    """Container env for the config's attention backend: the MJOLNIR
    defaults (overridable from the host env) + every ``MJOLNIR_ATTN_*`` /
    ``THOR_T5_*`` kernel knob set on the host. An fp6_e2m3 config also gets
    MJOLNIR_ATTN_KV_SCALES pointed at its calibration file."""
    out: list[str] = []
    if (s.attention_backend or "") in ("MJOLNIR", "CUSTOM"):
        out.append(f"MJOLNIR_CONFIG={s.quant}")  # the boot banner's `Model:` line
        for k, v in MJOLNIR_ENV_DEFAULTS.items():
            out.append(f"{k}={os.environ.get(k, v)}")
        if (config_field(s.config_path, "kv-cache-dtype") or "").startswith("fp6") and not os.environ.get(
            "MJOLNIR_ATTN_KV_SCALES"
        ):
            out.append(f"MJOLNIR_ATTN_KV_SCALES={_container_path(s, kv_scales_host_path(s))}")
    out += config_env(s)
    for k, v in sorted(os.environ.items()):
        if k.startswith(("MJOLNIR_ATTN_", "THOR_T5_")):
            out.append(f"{k}={v}")
    return out


def config_env(s: Settings) -> list[str]:
    """``KEY=VALUE`` lines from the config's ``.env`` sidecar (inside its
    config dir) — how an opt-in experiment config sets env vars vLLM's YAML
    cannot carry (e.g. ``MJOLNIR_LM_HEAD=fp8``). ``{configs}`` expands to the
    model's config dir as the container sees it; ``#`` starts a comment."""
    side = s.config_path.parent / ".env"
    if not side.exists():
        return []
    cdir = config_model_dir(s)
    out = []
    for line in side.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out.append(f"{k.strip()}={v.strip().replace('{configs}', cdir)}")
    return out


LOCAL_CONFIGS_MOUNT = "/configs-local"


def is_local_config(s: Settings) -> bool:
    """The active config is user-local (not in the repo / bundled configs)."""
    return s.config_path != configs_root() / s.model / s.quant / "config.yaml" and s.config_path.exists()


def config_model_dir(s: Settings) -> str:
    """The model's config dir as the container sees it: ``/configs/<model>`` for a
    repo config; for a user-local one its whole model dir is mounted at
    ``/configs-local/<model>`` — outside the read-only ``/configs`` mount, where
    Docker cannot create mountpoints for a vendor/model the repo doesn't have."""
    return f"{LOCAL_CONFIGS_MOUNT}/{s.model}" if is_local_config(s) else f"/configs/{s.model}"


def _container_path(s: Settings, host: Path) -> str:
    """A file under the configs dir, as the container sees it."""
    try:
        return "/configs/" + str(Path(host).relative_to(configs_root()))
    except ValueError:  # user-local file: its position under the mounted model dir
        try:
            rel = str(Path(host).relative_to(user_configs_dir() / s.model))
        except ValueError:
            rel = Path(host).name
        return f"{LOCAL_CONFIGS_MOUNT}/{s.model}/{rel}"


def serving_config(s: Settings) -> tuple[Path, str] | None:
    """(host file, container path) of the config the server actually receives:
    the platform overlay deep-merged over ``config.yaml`` and written under
    the data dir when the config carries one for this platform, else ``None``
    (the plain config passes through). The overlay must reach vLLM, not only
    the validate gate — ``validate_serving`` checks the merge but the merge
    itself was silently dropped before this existed (the Orin tuning —
    utilization, max-num-seqs, max-model-len — never applied to a serve)."""
    if s.platform is None:
        return None
    overlay = overlays.load_overlay(Path(s.config_path).parent, s.platform)
    if not overlay:
        # no overlay, or an empty marker one (a platforms/ file that only
        # lists the platform): the plain config passes through unchanged
        return None
    base = yaml.safe_load(Path(s.config_path).read_text())
    merged = overlays.merge_config(base or {}, overlay)
    out = s.data_dir / "serve" / f"{s.model.replace('/', '--')}--{s.quant}--{s.platform}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(merged, sort_keys=False))
    return Path(out), "/data/serve-config.yaml"


def build_run_args(
    s: Settings,
    image: str | None = None,
    *,
    name: str = CONTAINER_NAME,
    detach: bool = True,
    command: str | None = None,
    extra_env: tuple[str, ...] = (),
    extra_mounts: tuple[tuple[str, str], ...] = (),
    serve_args: tuple[str, ...] = (),
    chown: tuple[str, ...] = (),
) -> list[str]:
    """Assemble the full ``docker run`` argv for the vLLM container (the
    server by default; ``command`` runs something else with the same mounts,
    env and GPU access — ``mjolnir calibrate``). ``serve_args`` are appended
    to ``vllm serve`` and override the config (e.g. ``--speculative-config``
    for ``mjolnir verify spec``). ``chown`` lists container paths a
    ``command`` writes into host mounts: they are handed to the invoking
    user when it ends (the container runs as root)."""
    image = image or s.image
    try:
        hf_hub = host_hf_hub()
    except HFCacheError as exc:
        raise DockerError(str(exc)) from exc
    cmd = [
        _docker(),
        "run",
        *(["-d"] if detach else ["--rm"]),
        "--name",
        name,
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--privileged",
        "--user",
        "0:0",
        # no --memory-swappiness: cgroup v2 (JetPack 7) has no per-container
        # swappiness, docker only warns and discards it
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--ulimit",
        "nofile=65536:65536",
        "--restart",
        "no",
        # base image ENTRYPOINT is ["vllm","serve"] — pin ours, so the command
        # below is the full argv (unpinned, docker would compose
        # "vllm serve bash -lc ..." and argparse would reject it at startup)
        "--entrypoint",
        "bash",
    ]

    for k in _CONTAINER_ENV:
        cmd += ["-e", k]
    # One arm64 image serves every shipped Tegra platform (Thor and Orin are
    # both aarch64; the kernel family resolves at runtime via mjolnir.platforms).
    # The arch-sensitive JIT envs follow the platform -- the Thor pins in
    # _CONTAINER_ENV are the build default, these override them (docker's
    # last -e wins). An explicit extra_env entry still wins over these.
    if getattr(s, "platform", None) == "orin":
        cmd += [
            "-e",
            "TORCH_CUDA_ARCH_LIST=8.7",
            "-e",
            "FLASHINFER_CUDA_ARCH_LIST=8.7",
            "-e",
            "CUTE_DSL_ARCH=sm_87",
            # torch 2.13's expandable segments query NVML fabric info when
            # creating a segment; the Orin's NVML has no fabric info and the
            # query hard-fails at the first allocation ("Expected
            # NVML_SUCCESS == ...nvmlDeviceGetGpuFabricInfoV_"). The Thor's
            # NVML implements it, so _CONTAINER_ENV's expandable_segments:True
            # stays there; the plain caching allocator loses nothing that
            # matters on the unified-memory board.
            "-e",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False",
        ]
    for kv in backend_env(s):
        cmd += ["-e", kv]
    for kv in extra_env:
        cmd += ["-e", kv]
    if os.environ.get("HF_TOKEN"):
        cmd += ["-e", f"HF_TOKEN={os.environ['HF_TOKEN']}"]
    if hf_hub is not None:
        # config / tokenizer lookups resolve through the hub cache, not --download-dir
        cmd += ["-e", f"HF_HUB_CACHE={HF_DOWNLOAD_DIR}"]

    d = s.data_dir
    serving = serving_config(s)
    mounts = [
        (str(d / "models"), "/data/models"),
        (str(d / "cache"), "/data/cache"),
        (str(d / "kernels-cache"), "/data/kernels-cache"),
        (str(d / "flashinfer-cache"), "/data/flashinfer-cache"),
        (str(d / "triton-cache"), "/data/triton-cache"),
        (str(d / "tiktoken"), "/data/tiktoken"),
        (str(configs_root()), "/configs:ro"),
        ("/dev/shm", "/dev/shm"),
        ("/etc/localtime", "/etc/localtime:ro"),
        ("/etc/machine-id", "/etc/machine-id:ro"),
    ]
    # $MJOLNIR_HF_CACHE: reuse a host HF hub cache as vLLM's --download-dir (and
    # HF_HUB_CACHE, above) instead of a second copy of every checkpoint under the
    # data dir; the more-specific mount wins over /data/models. The container
    # runs as root, so files it adds to the hub are root-owned (as with any
    # other root container sharing that cache).
    if hf_hub is not None:
        mounts.append((str(hf_hub), HF_DOWNLOAD_DIR))
    # A user-local model config (outside the repo's configs/) is mounted as
    # its whole model dir at /configs-local/<model> (config dirs + the model's
    # kv_scales/ calibration + draft_vocab.json), beside — not inside — the
    # read-only /configs mount; repo files (chat templates) stay reachable
    # under /configs.
    if is_local_config(s):
        mounts.append((str(user_configs_dir() / s.model), f"{config_model_dir(s)}:ro"))
    if serving is not None:
        mounts.append((str(serving[0]), serving[1]))
    mounts += list(extra_mounts)
    for host, container in mounts:
        if host.startswith("/data") is False and not Path(host).exists() and not host.startswith(("/dev/shm", "/etc/")):
            continue  # optional host files (e.g. /etc/nv_tegra_release)
        cmd += ["-v", f"{host}:{container}"]
    if command is not None:
        return [*cmd, image, "-lc", command + owner_suffix(chown)]
    cmd += ["-p", f"{s.port}:8000"]

    serve = (
        "vllm serve "
        f"--config {(serving[1] if serving else f'{config_model_dir(s)}/{s.quant}/config.yaml')} "
        "--host 0.0.0.0 "
        "--cpu-offload-gb 0 "
        "--pipeline-parallel-size 1 "
        "--disable-fastapi-docs "
        "--enable-force-include-usage "
        "--enable-prompt-tokens-details "
        "--download-dir /data/models/huggingface"
    )
    if serve_args:
        import shlex

        serve += " " + " ".join(shlex.quote(a) for a in serve_args)
    cmd += [image, "-lc", serve]
    return cmd


def host_owner() -> str | None:
    """``uid:gid`` of the user running the CLI (the sudo caller under sudo);
    ``None`` off POSIX."""
    if not hasattr(os, "getuid"):
        return None
    uid, gid = os.getuid(), os.getgid()
    if uid == 0 and os.environ.get("SUDO_UID"):
        uid, gid = int(os.environ["SUDO_UID"]), int(os.environ.get("SUDO_GID", gid))
    return f"{uid}:{gid}"


def owner_suffix(paths: tuple[str, ...] | list[str]) -> str:
    """Shell tail for a root container command: hand ``paths`` (container
    paths in host mounts) to the host user, keep the command's exit code.
    Without it a file the container writes into the checkout is root-owned
    and git hooks (end-of-file-fixer) cannot rewrite it."""
    owner = host_owner()
    if not paths or owner is None:
        return ""
    import shlex

    quoted = " ".join(shlex.quote(p) for p in paths)
    return f"; rc=$?; chown -R {owner} {quoted} 2>/dev/null; exit $rc"


def run_once(argv: list[str], name: str) -> int:
    """Run a one-shot container in the foreground (``docker run --rm``) and
    always clean up after it: remove the container even when the CLI is
    interrupted (stopping the docker client leaves the container running,
    holding the GPU), then clear the host page cache — on Thor the cache the
    container filled stays resident after it exits."""
    cachectl.prepare_bench()
    try:
        return subprocess.run(argv).returncode
    except KeyboardInterrupt:
        print(f"\ninterrupted — removing container {name}", file=sys.stderr)
        return 130
    finally:
        with contextlib.suppress(DockerError):
            _run_quiet([_docker(), "rm", "-f", name])
        cachectl.prepare_bench()


def serve_up(
    s: Settings,
    wait: bool = True,
    wait_timeout_s: float = 1800.0,
    dry_run: bool = False,
    serve_args: tuple[str, ...] = (),
    extra_env: tuple[str, ...] = (),
) -> None:
    cmd = build_run_args(s, serve_args=serve_args, extra_env=extra_env)
    if dry_run:
        print(" ".join(cmd))
        return
    try:
        overlays.validate_serving(s.config_path, s.platform)
    except ValueError as e:
        raise DockerError(str(e)) from None
    why = missing_kv_scales(s)
    if why:
        raise DockerError(why)
    # Always-before: clear the host page cache before starting vLLM (a clean,
    # reproducible model-load). No-op + hint if the grant is missing; never
    # prompts, never blocks.
    cachectl.prepare_bench()
    for sub in [
        s.data_dir / p for p in ("models", "cache", "kernels-cache", "flashinfer-cache", "triton-cache", "tiktoken")
    ]:
        sub.mkdir(parents=True, exist_ok=True)
    if container_running():
        print("container already running — stopping the old one first")
        serve_down(s)
    elif container_exists():
        print("removing a stopped container left by an earlier run")
        _run_quiet([_docker(), "rm", "-f", CONTAINER_NAME])
    _run(cmd)
    if wait:
        wait_healthy(s, wait_timeout_s)


def serve_down(s: Settings, dry_run: bool = False) -> None:
    if not container_running():
        if container_exists() and not dry_run:
            _run_quiet([_docker(), "rm", "-f", CONTAINER_NAME])
            print("removed a stopped container (not running)")
        else:
            print("nothing to stop (container not running)")
        return
    if dry_run:
        print(f"{_docker()} rm -f {CONTAINER_NAME}")
        return
    _run_quiet([_docker(), "rm", "-f", CONTAINER_NAME])
    print("stopped")
    # After stopping: clear the page cache (release the serving file activity
    # that populated it). No-op + hint if the grant is missing.
    cachectl.prepare_bench()


def _log_size() -> int | None:
    """The container's log file size on the host (a cheap boot-progress
    probe; None when the log path cannot be stat'ed)."""
    try:
        p = subprocess.run(
            [_docker(), "inspect", "-f", "{{.LogPath}}", CONTAINER_NAME], capture_output=True, text=True, timeout=10
        )
        path = p.stdout.strip()
        return os.path.getsize(path) if p.returncode == 0 and path and path != "<no value>" else None
    except Exception:  # noqa: BLE001 - progress probing degrades, never raises
        return None


def wait_healthy(s: Settings, timeout_s: float = 1800.0) -> None:
    """Poll the OpenAI /health endpoint until the server answers (model load
    + JIT on Thor takes minutes). A container that exits mid-wait fails
    immediately — a dead serve burns no timeout (the 1800s blind wait on a
    crashed boot cost a full point per leg in the 10/10c sweeps).

    A first boot also *compiles* — inductor + the CUDA graphs ran 34 minutes
    on a cold Orin (orin2's §4) — so the wait is progress-aware: while the
    container's log keeps advancing the deadline extends (up to 4× the
    timeout); a boot that stops logging for ``timeout_s`` is dead and fails.
    Pass ``--wait-timeout`` to serve up for a tighter or looser window."""
    url = f"http://127.0.0.1:{s.port}/health"
    t0 = time.time()
    print(f"waiting for the server at {url} (up to {timeout_s:.0f}s; extends while the boot makes progress)…")
    last_size = _log_size()
    last_progress = time.time()
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"healthy after {time.time() - t0:.0f}s")
                    return
        except Exception:  # noqa: BLE001
            pass
        if container_exists() and not container_running():
            logs = subprocess.run([_docker(), "logs", "--tail", "12", CONTAINER_NAME], capture_output=True, text=True)
            tail = (logs.stdout + logs.stderr).strip().splitlines()
            hint = "\n".join(f"  {ln}" for ln in tail[-6:])
            raise RuntimeError(
                f"the {CONTAINER_NAME} container exited before becoming healthy "
                f"(after {time.time() - t0:.0f}s) — last logs:\n{hint}"
            )
        size = _log_size()
        if size is not None and size != last_size:
            last_size, last_progress = size, time.time()
        elapsed = time.time() - t0
        idle = time.time() - last_progress
        if elapsed >= 4 * timeout_s or (elapsed >= timeout_s and idle >= timeout_s):
            raise TimeoutError(
                f"server not healthy after {elapsed:.0f}s"
                + (f" (no log progress for {idle:.0f}s)" if elapsed >= timeout_s else "")
                + " — check: mjolnir serve logs"
            )
        time.sleep(5)


def status(s: Settings) -> dict:
    running = container_running()
    state: dict = {"container": CONTAINER_NAME, "running": running}
    if running:
        _, out = _run_quiet(
            [_docker(), "inspect", "-f", "{{.State.Status}} started {{.State.StartedAt}}", CONTAINER_NAME]
        )
        state["state"] = out
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{s.port}/health", timeout=5) as r:
            state["health"] = "ok" if r.status == 200 else f"HTTP {r.status}"
    except Exception:  # noqa: BLE001 - the health probe reports, never raises
        state["health"] = "unreachable"
    return state


def logs(s: Settings, follow: bool = False, tail: int = 100) -> int:
    if not container_running():
        print("container not running", file=sys.stderr)
        return 1
    args = [_docker(), "logs", "--tail", str(tail)]
    if follow:
        args += ["-f"]
    args += [CONTAINER_NAME]
    try:
        return subprocess.call(args)
    except KeyboardInterrupt:
        return 0


# ANSI escape sequences: CSI (SGR colors, cursor moves) + OSC (e.g. hyperlinks).
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def dump_server_log(since_epoch: float | None, out_path: Path) -> bool:
    """Capture the vLLM container's logs to ``out_path``, with ANSI colors
    stripped. ``since_epoch`` (unix seconds) bounds the capture from that
    instant; ``None`` captures the whole container log since it started
    (startup kernel-dispatch lines + everything after — what ``bench perf``
    saves for kernel debugging). Best-effort: returns ``False`` (no raise)
    when the container is absent/not running or docker fails — a log
    capture failure never fails the bench."""
    if not container_running():
        return False
    args = [_docker(), "logs"]
    if since_epoch is not None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z", time.gmtime(since_epoch))
        args += ["--since", ts]
    args += [CONTAINER_NAME]
    try:
        with out_path.open("w") as f:
            subprocess.run(args, stdout=f, stderr=f, check=False)
    except (OSError, DockerError):
        return False
    try:
        out_path.write_text(_strip_ansi(out_path.read_text()))
    except OSError:
        return False
    return True
