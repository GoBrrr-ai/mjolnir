"""Container plumbing for ``mjolnir verify``: harness runs, ncu legs, full
profiles and test runs — each in a named, timed-out, reaped container.

The kernel under test is the Thor harness (``python3 -m
mjolnir.arch.thor.harness``). In a checkout the repo's ``src/mjolnir`` is mounted
over the image's installed copy (``PYTHONPATH``), so every run measures the
current source; a PyPI install runs the kernel baked into the image.
"""

from __future__ import annotations

import os
import random
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mjolnir_cli.config import host_hf_hub
from mjolnir_cli.platforms import DEFAULT_PLATFORM, platform_ncu
from mjolnir_cli.verify import ncu
from mjolnir_cli.verify.legs import FILTERS, Leg

DRIVERS_DIR = Path(__file__).resolve().parent.parent / "drivers"
NCU_HOME_DEFAULT = "/opt/nvidia/nsight-compute/2025.3.0"


def resolve_ncu_home() -> str:
    """The host's Nsight Compute install: ``$NCU_HOME`` wins; else the newest
    versioned dir under ``/opt/nvidia/nsight-compute`` that carries the ncu
    binary (JP names them by version — the Thor's 2025.3.0, the Orin's
    2026.1.1); else the Thor default (the pre-detection layout)."""
    env = os.environ.get("NCU_HOME")
    if env:
        return env
    try:
        cands = sorted(p for p in Path(NCU_HOME_DEFAULT).parent.glob("*") if (p / "ncu").is_file())
    except OSError:
        cands = []
    return str(cands[-1]) if cands else NCU_HOME_DEFAULT


_IN_SRC = "/opt/mjolnir-src"


def ensure_image(image: str) -> str | None:
    """Make sure ``image`` is available locally (pull it if not) BEFORE any
    stage starts, so a missing image is one clear error instead of one
    failed container per case. Returns None if ready, else a message."""
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0:
        return None
    p = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
    if p.returncode == 0:
        return None
    why = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["pull failed"]
    return (
        f"image {image} is not available locally and could not be pulled ({why[0]}).\n"
        f"   build it from this checkout:  mjolnir image build --tag {image}\n"
        f"   or use another image:         --image <tag>   (or: mjolnir image use <tag>)"
    )


_IN_DRIVERS = "/opt/mjolnir-drivers"
_IN_NCU = "/ncu"


def default_ref_image(platform: str = DEFAULT_PLATFORM) -> str:
    """Where the FlashInfer / FA2 reference legs run: the OFFICIAL upstream
    vLLM image, never ours — the Mjolnir image patches FlashInfer paths too,
    so a reference measured there would not be stock vLLM. ``$MJOLNIR_REF_IMAGE``
    wins over the platform record's stock image."""
    from mjolnir_cli.platforms import stock_image

    ref = stock_image(platform, os.environ.get("MJOLNIR_REF_IMAGE"))
    if ref is None:
        raise ValueError(f"platform {platform!r} defines no stock image to reference")
    return ref


def _hf_hub_mount() -> list[str]:
    """$MJOLNIR_HF_CACHE nested over the HF_HOME mount: the smoke models come
    from the shared hub too (HF_HOME's other state stays in ``hf_home``)."""
    hub = host_hf_hub()
    return ["-v", f"{hub}:/root/.cache/huggingface/hub"] if hub is not None else []


@dataclass
class Runner:
    image: str
    logdir: Path
    platform: str = DEFAULT_PLATFORM  # which kernel family the harness stages drive
    ref_image: str = field(default_factory=default_ref_image)  # FlashInfer / FA2 legs
    kernel_src: Path | None = None  # checkout's src/mjolnir (mounted over the image copy)
    ncu_home: str = field(default_factory=resolve_ncu_home)
    timeout: float = 180.0
    dry_run: bool = False
    hq: int = 24
    hk: int = 4
    d: int = 256
    page: int = 128
    batch: int = 1  # sequences per decode step (each of L tokens) — `verify sweep --batch`
    q_len: int = 1  # query tokens per sequence (MTP=3 verification is 4) — `verify sweep --q-len`

    _ready: set = field(default_factory=set, repr=False)

    # ── low level ──────────────────────────────────────────────────────────
    def image_for(self, leg: Leg) -> str:
        """Mjolnir legs and the fa4 leg run on the Mjolnir image (the FA4
        hd256 + fp8 patch group exists only there — stock vLLM downgrades
        FA4 to FA2 for hd256 + quantized KV); FlashInfer / FA2 references
        on the official vLLM image."""
        return self.image if leg.kind in ("t5", "fa4") else self.ref_image

    def _ensure(self, image: str, log: Path) -> bool:
        if self.dry_run or image in self._ready:
            return True
        err = ensure_image(image)
        if err:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(err + "\n")
            print(f"  ! {err}", file=sys.stderr)
            return False
        self._ready.add(image)
        return True

    def _name(self, prefix: str) -> str:
        return f"{prefix}-{os.getpid()}-{random.randint(0, 1 << 30)}"

    def _overlay(self) -> list[str]:
        if self.kernel_src is None:
            return []
        return ["-v", f"{self.kernel_src}:{_IN_SRC}:ro", "-e", f"PYTHONPATH={_IN_SRC}"]

    def _run(self, argv: list[str], log: Path, name: str, timeout: float | None = None) -> int:
        log.parent.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            print("  $ " + " ".join(shlex.quote(a) for a in argv))
            log.write_text("(dry run)\n")
            return 0
        with log.open("w") as fh:
            try:
                p = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout or self.timeout)
                return p.returncode
            except KeyboardInterrupt:
                # the docker client dies with the CLI; the container would keep the GPU
                subprocess.run(["docker", "rm", "-f", name], capture_output=True)
                raise
            except subprocess.TimeoutExpired:
                # stopping only the docker CLI leaves the container — and the GPU — stuck
                subprocess.run(["docker", "kill", name], capture_output=True)
                subprocess.run(["docker", "rm", "-f", name], capture_output=True)
                fh.write(f"\nTIMEOUT after {timeout or self.timeout:.0f}s (container killed: hung kernel?)\n")
                return 124

    # ── the kernel harness ─────────────────────────────────────────────────
    def harness_args(self, q: str, args: list[str] | str) -> list[str]:
        """The kernel harness command for this platform: the module and the
        dtype spellings differ per family (the Thor harness takes DSL dtype
        names, the Orin harness bf16/fp8), the shared stage flags stay the
        same (the Orin harness accepts the common aliases)."""
        if isinstance(args, str):
            args = shlex.split(args)
        if self.platform == "orin":
            # bf16 Q until the qpack wiring lands — the Orin runner raises on
            # fp8 Q today; when it arrives this maps the leg's q like the Thor.
            return ["-m", "mjolnir.arch.orin.harness", "--q_dtype", "bf16", "--kv_dtype", "fp8", *args]
        qd = "Float8E4M3FN" if q == "fp8" else "BFloat16"
        return ["-m", "mjolnir.arch.thor.harness", "--q_dtype", qd, "--kv_dtype", "Float8E4M3FN", *args]

    def harness(self, log: Path, q: str, args: list[str] | str, timeout: float | None = None) -> tuple[int, str]:
        name = self._name("mjolnir-verify")
        argv = ["docker", "run", "--rm", "--name", name, "--gpus", "all", "--entrypoint", "python3"]
        argv += [*self._overlay(), self.image, *self.harness_args(q, args)]
        rc = self._run(argv, log, name, timeout)
        return rc, log.read_text(errors="replace") if log.exists() else ""

    # ── ncu-profiled legs ──────────────────────────────────────────────────
    def leg_command(self, leg: Leg, L: int, iters: int) -> tuple[str, list[str]]:
        """(python command line inside the container, extra docker -e args)."""
        common_env = [
            f"BENCH_L={L}",
            f"BENCH_ITERS={iters}",
            f"BENCH_HQ={self.hq}",
            f"BENCH_HK={self.hk}",
            f"BENCH_D={self.d}",
            f"BENCH_PAGE={self.page}",
            f"BENCH_B={self.batch}",
            f"BENCH_QLEN={self.q_len}",
        ]
        if leg.kind == "t5":
            args = self.harness_args(
                leg.q,
                [
                    "--seqlen",
                    str(L),
                    "--heads_q",
                    str(self.hq),
                    "--heads_k",
                    str(self.hk),
                    "--headdim",
                    str(self.d),
                    "--page_size",
                    str(self.page),
                    *(["--batches", str(self.batch)] if self.batch != 1 else []),
                    *(["--q_len", str(self.q_len)] if self.q_len != 1 else []),
                    "--iterations",
                    str(iters),
                    "--warmup_iterations",
                    "0",
                    "--skip_ref_check",
                    "--no_l2_flush",
                    *shlex.split(leg.flags),
                ],
            )
            return "python3 " + " ".join(shlex.quote(a) for a in args), []
        script = {"fi": "fi_driver.py", "fa2": "fa2_driver.py", "fa4": "fa4_driver.py"}[leg.kind]
        env = common_env + [f"{k}={v}" for k, v in leg.env]
        return f"python3 {_IN_DRIVERS}/{script}", env

    def ncu_leg(
        self, log: Path, leg: Leg, L: int, iters: int, cache: str = "all", clock: str = "base"
    ) -> tuple[int, dict | None]:
        if not platform_ncu(self.platform):
            raise RuntimeError(
                f"ncu profiling is not available on the {self.platform!r} platform "
                "(no Nsight Compute in the L4T rootfs) — the events-timed stages "
                "(sweep, correctness, timing) carry this platform's headline data"
            )
        cmd, env = self.leg_command(leg, L, iters)
        name = self._name("mjolnir-ncu")
        argv = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--gpus",
            "all",
            "--network",
            "host",
            "--security-opt",
            "seccomp=unconfined",
            "--cap-add",
            "SYS_ADMIN",
            "--entrypoint",
            "bash",
            "-v",
            f"{self.ncu_home}:{_IN_NCU}:ro",
            "-v",
            f"{DRIVERS_DIR}:{_IN_DRIVERS}:ro",
            "-e",
            "TMPDIR=/tmp/ncu-mjolnir",
        ]
        if leg.kind == "t5":
            argv += self._overlay()
        for e in env:
            argv += ["-e", e]
        flt = FILTERS[leg.kind]
        inner = (
            f"mkdir -p /tmp/ncu-mjolnir && {_IN_NCU}/ncu --target-processes all "
            f"--clock-control {clock} --cache-control {cache} --kernel-name {shlex.quote(flt)} "
            f"--print-summary per-kernel --metrics {ncu.METRICS} {cmd}"
        )
        image = self.image_for(leg)
        if not self._ensure(image, log):
            return 2, None
        argv += [image, "-c", inner]
        rc = self._run(argv, log, name, timeout=max(self.timeout, 600.0))
        if rc != 0 or self.dry_run:
            return rc, None
        from mjolnir_cli.verify.legs import MAIN, MERGE

        kv_bytes = self.batch * L * self.hk * self.d * 2 * leg.kv_bits / 8
        return rc, ncu.parse_file(log, MAIN[leg.kind], MERGE[leg.kind], L, kv_bytes)

    def driver_wall(self, log: Path, script: str, L: int, iters: int, env_extra=()) -> tuple[int, float | None]:
        """Events-timed wall clock of a driver (the FlashInfer reference on
        platforms without ncu): parses the driver's ``mean X us/iter`` line.
        One warm run happens inside the driver's first launches (JIT), so the
        caller passes enough iters to amortize it."""
        name = self._name("mjolnir-driver")
        argv = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--gpus",
            "all",
            "--entrypoint",
            "python3",
            "-v",
            f"{DRIVERS_DIR}:{_IN_DRIVERS}:ro",
        ]
        for k, v in (
            ("BENCH_L", L),
            ("BENCH_ITERS", iters),
            ("BENCH_HQ", self.hq),
            ("BENCH_HK", self.hk),
            ("BENCH_D", self.d),
            ("BENCH_PAGE", self.page),
            ("BENCH_B", self.batch),
            ("BENCH_QLEN", self.q_len),
        ):
            argv += ["-e", f"{k}={v}"]
        for k, v in env_extra:
            argv += ["-e", f"{k}={v}"]
        argv += [self.ref_image, f"{_IN_DRIVERS}/{script}"]  # the reference legs run on the stock image
        rc = self._run(argv, log, name, timeout=max(self.timeout, 600.0))
        text = log.read_text(errors="replace") if log.exists() else ""
        m = re.search(r"mean ([0-9.]+) us/iter", text)
        return rc, (float(m.group(1)) if m else None)

    # ── full ncu profile (fp6prof) ─────────────────────────────────────────
    def ncu_profile(
        self, outdir: Path, name: str, leg: Leg, L: int, count: int = 3, section_set: str = "full"
    ) -> tuple[int, dict]:
        """Profile ``count`` launches of the leg's decode kernel with a full
        section set; returns (rc, {"raw": {...}, "units": {...}, "details": {...}})."""
        outdir.mkdir(parents=True, exist_ok=True)
        cmd, _env = self.leg_command(leg, L, count + 2)
        secs = (
            "--section SpeedOfLight --section MemoryWorkloadAnalysis --section WarpStateStats "
            "--section SchedulerStats --section Occupancy --section LaunchStats --section ComputeWorkloadAnalysis"
        )
        rc = 1
        for attempt in (f"--set {section_set}", secs):
            cname = self._name("mjolnir-prof")
            inner = (
                f"mkdir -p /tmp/ncu-mjolnir && {_IN_NCU}/ncu --target-processes all --clock-control base "
                f"--cache-control all --kernel-name {shlex.quote(FILTERS[leg.kind])} "
                f"--launch-skip 1 --launch-count {count} {attempt} -f -o /out/{name} {cmd} && "
                f"{_IN_NCU}/ncu --import /out/{name}.ncu-rep --page raw --csv > /out/{name}.raw.csv && "
                f"{_IN_NCU}/ncu --import /out/{name}.ncu-rep --page details --csv > /out/{name}.details.csv && "
                f"chmod a+rw /out/{name}.*"
            )
            from mjolnir_cli.dockerctl import owner_suffix

            inner += owner_suffix([f"/out/{name}.ncu-rep", f"/out/{name}.raw.csv", f"/out/{name}.details.csv"])
            argv = [
                "docker",
                "run",
                "--rm",
                "--name",
                cname,
                "--gpus",
                "all",
                "--security-opt",
                "seccomp=unconfined",
                "--cap-add",
                "SYS_ADMIN",
                "--entrypoint",
                "bash",
                "-v",
                f"{self.ncu_home}:{_IN_NCU}:ro",
                "-v",
                f"{DRIVERS_DIR}:{_IN_DRIVERS}:ro",
                "-v",
                f"{outdir}:/out",
                "-e",
                "TMPDIR=/tmp/ncu-mjolnir",
            ]
            image = self.image_for(leg)
            if not self._ensure(image, outdir / f"{name}.log"):
                return 2, {}
            argv += (self._overlay() if leg.kind == "t5" else []) + [image, "-c", inner]
            rc = self._run(argv, outdir / f"{name}.log", cname, timeout=max(self.timeout, 900.0))
            if rc == 0:
                break
            print(f"  ! {name}: ncu exit {rc} with [{attempt}] — retrying with fewer sections", file=sys.stderr)
        if rc != 0 or self.dry_run:
            return rc, {}
        try:
            raw, units, best_id = ncu.parse_raw_csv((outdir / f"{name}.raw.csv").read_text(errors="replace"))
            det_p = outdir / f"{name}.details.csv"
            details = ncu.parse_details_csv(det_p.read_text(errors="replace"), best_id) if det_p.exists() else {}
        except (OSError, ValueError, ZeroDivisionError) as e:  # a bad export loses one profile, not the run
            print(f"  ! {name}: could not parse the ncu export ({e}) — see {outdir}", file=sys.stderr)
            return 1, {}
        if not raw:
            print(f"  ! {name}: the ncu export has no metrics — see {outdir / (name + '.raw.csv')}", file=sys.stderr)
            return 1, {}
        return rc, {"raw": raw, "units": units, "details": details}

    # ── vLLM-side tests (runner tests, smoke) ──────────────────────────────
    def tests(self, log: Path, repo: Path, cmd: str, timeout: float, hf_home: Path) -> int:
        name = self._name("mjolnir-vllm")
        hf_home.mkdir(parents=True, exist_ok=True)
        argv = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--gpus",
            "all",
            "--ipc=host",
            "--entrypoint",
            "bash",
            "-v",
            f"{repo / 'src' / 'mjolnir'}:{_IN_SRC}:ro",
            "-v",
            f"{repo / 'tests' / 'kernel'}:/work/tests:ro",
            "-v",
            f"{hf_home}:/root/.cache/huggingface",
            *_hf_hub_mount(),
            "-e",
            f"PYTHONPATH={_IN_SRC}",
            "-e",
            f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
            "-e",
            "TMPDIR=/tmp",
            # The arch-sensitive JIT envs follow the platform (the same
            # override build_run_args makes): the kernel compiles for the
            # device this container runs on, and any Triton/Inductor paths
            # in the smoke target it too.
            *(
                ("-e", "TORCH_CUDA_ARCH_LIST=8.7", "-e", "FLASHINFER_CUDA_ARCH_LIST=8.7", "-e", "CUTE_DSL_ARCH=sm_87")
                if self.platform == "orin"
                else ()
            ),
            "-w",
            "/work",
            self.image,
            "-c",
            cmd,
        ]
        return self._run(argv, log, name, timeout)
