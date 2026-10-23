"""The ``mjolnir verify`` stages — a Python port of the old ``verify.sh``.

Every stage returns ONE structured result document (``results.py``) that is
saved under ``benchmarks/raw/verify/<stage>/<run_id>.json`` and printed as a
terminal table. Charts, README tables and the deep-dive docs are generated
from those documents only (``mjolnir plot --readme``).

Default L axis for every stage: ``ALL_LS`` (128 → 262144 tokens) — the
kernel has to be fast at every context length, not at a hand-picked one.
"""

from __future__ import annotations

import re
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

from mjolnir_cli.verify import legs as L_
from mjolnir_cli.legs import KERNEL, SMOKE_MODES
from mjolnir_cli.verify import results
from mjolnir_cli.verify.docker import Runner

ALL_LS: tuple[int, ...] = (128, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
DEFAULT_ITERS = 50
# Run-to-run noise of a SAME-RUN ratio (reference time ÷ variant time) on a
# dedicated Thor (no live server, ncu base clocks, cold L2). Measured from the
# 2026-10-11 bench: the same FlashInfer/Mjolnir pair repeated in 4-5 stage runs
# gives a per-L ratio cv of 0.6-2.6% (median 1.25%), so ±3% is ~2.3σ of a
# typical cell. Absolute µs drift more (~10% between runs at long L, both legs
# together), so only ever compare legs measured in the same run. L=16K is
# bimodal (FI/Mjolnir 0.98 or 1.14 depending on the run) — never judge a change
# on 16K alone. Was 7% while the GPU was shared with a live vLLM server.
NOISE_PCT = 3.0

_MEDIAN = re.compile(r"^median ([0-9.]+) us(?:, ([0-9.e+]+) GB/s)?", re.MULTILINE)
_FULL_MEDIAN = re.compile(r"^full median ([0-9.]+) us", re.MULTILINE)


@dataclass
class Ctx:
    """What every stage gets: the runner, where to write, how to gate."""

    runner: Runner
    run_id: str
    logdir: Path
    metrics_url: str
    gate: bool = True
    gate_confirm: int = 3
    gate_timeout: float = 3600.0
    ls: tuple[int, ...] = ALL_LS
    iters: int = DEFAULT_ITERS
    out: Callable[[str], None] = print
    gate_log: dict = field(default_factory=lambda: {"gated": False, "clean": None, "server": "unknown"})

    def wait_clean(self) -> None:
        """Clean-window gate before a measurement: waits only if the live
        vLLM server is reachable; an offline server = nothing to gate against."""
        if not self.gate or self.runner.dry_run:
            return
        from mjolnir_cli.gate import server_load, wait_for_idle

        load = server_load(self.metrics_url)
        if load is None:
            self.gate_log.update(server="offline", gated=False)
            return
        self.gate_log.update(server="up", gated=True)
        if load != (0.0, 0.0):
            self.out(f"   … waiting for a clean window (running={load[0]:g} waiting={load[1]:g})")
        ts = wait_for_idle(self.metrics_url, confirm=self.gate_confirm, timeout_s=self.gate_timeout)
        self.gate_log["clean"] = ts is not None and self.gate_log.get("clean") is not False


@dataclass(frozen=True)
class Stage:
    name: str
    title: str
    kind: str  # "correctness" | "ncu" | "harness" | "profile" | "tests" | "serving"
    summary: str  # one line (help text, docs index)
    fn: Callable[..., dict] | None = None


def _median(text: str) -> tuple[float | None, float | None]:
    m = _MEDIAN.search(text or "")
    if not m:
        return None, None
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else None)


# ── ncu sweeps (sweep / hndsweep / ring / prefetch / ab / kvfmt) ────────────


def run_series(ctx: Ctx, legs: list[L_.Leg], stage: str, cache: str = "all", clock: str = "base") -> list[dict]:
    """Profile every leg at every L; one series dict per leg."""
    series = []
    for leg in legs:
        pts = []
        ctx.out(f"   {leg.label}  [{leg.describe()}]")
        for L in ctx.ls:
            ctx.wait_clean()
            log = ctx.logdir / stage / f"{_slug(leg.id)}-L{L}.txt"
            rc, pt = ctx.runner.ncu_leg(log, leg, L, ctx.iters, cache=cache, clock=clock)
            if pt:
                pts.append(pt)
                ctx.out(f"      L={L:>7,}  {pt['total_us']:>10.1f} µs  {pt['gbps']:>6.1f} GB/s")
            elif not ctx.runner.dry_run:
                ctx.out(f"      L={L:>7,}  FAILED (exit {rc}) — {log}")
        series.append(
            {
                "id": leg.id,
                "label": leg.label,
                "kind": leg.kind,
                "role": leg.role,
                "flags": leg.flags,
                "env": dict(leg.env),
                "q": leg.q,
                "kv_format": leg.kv_format,
                "kv_layout": leg.kv_layout,
                "kv_bits": leg.kv_bits,
                "points": pts,
            }
        )
    return series


def _wall_rows(ctx: Ctx, legs: list[L_.Leg], stage: str, iters: int, ls: tuple[int, ...]) -> list[dict]:
    """Harness wall-clock medians (cold L2, b=1) — includes the launch gap a
    separate split-KV merge kernel costs, which ncu cannot see."""
    rows = []
    for L in ls:
        for leg in legs:
            if leg.kind != "t5":
                continue
            ctx.wait_clean()
            log = ctx.logdir / stage / "wall" / f"{_slug(leg.id)}-L{L}.log"
            args = ["--seqlen", str(L), "--skip_ref_check", "--warmup_iterations", "10", "--iterations", str(iters)]
            rc, text = ctx.runner.harness(log, leg.q, args + shlex.split(leg.flags))
            us, gbps = _median(text)
            rows.append({"L": L, "series": leg.id, "median_us": us, "gbps": gbps, "ok": rc == 0 and us is not None})
    return rows


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", s).strip("_")


SWEEP_LEGS = KERNEL["verify.sweep"]  # mjolnir_cli.legs


def stage_sweep(ctx: Ctx, legs: str = SWEEP_LEGS, page: int | None = None, batch: int = 1, q_len: int = 1) -> dict:
    """Every leg at every L. ``batch`` sequences of L tokens with ``q_len``
    query tokens each (default one decode token of one sequence; serving with
    MTP=3 verifies 4 tokens per sequence: ``batch=4, q_len=4``)."""
    sel = L_.legs_from_ids(legs)
    single_token = {"fa2", "fa4"}
    if (batch, q_len) != (1, 1) and any(x.kind in single_token for x in sel):
        # the FA2/FA4 drivers only time one decode token of one sequence
        if legs != SWEEP_LEGS:
            raise ValueError("the fa2/fa4 legs measure b=1, q_len=1 only — drop them from --legs for --batch/--q-len")
        sel = [x for x in sel if x.kind not in single_token]
        ctx.out(f"   (fa2/fa4 legs skipped: they measure b=1, q_len=1 only — legs {' '.join(x.id for x in sel)})")
    ctx.runner.batch, ctx.runner.q_len = batch, q_len
    params: dict = {"legs": [x.id for x in sel]}
    if (batch, q_len) != (1, 1):
        params.update(batch=batch, q_len=q_len)
    return {"series": run_series(ctx, sel, "sweep"), "params": params}


def stage_hndsweep(ctx: Ctx, legs: str = KERNEL["verify.hndsweep"]) -> dict:
    sel = L_.legs_from_ids(legs)
    return {"series": run_series(ctx, sel, "hndsweep"), "params": {"legs": [x.id for x in sel]}}


def _variant_stage(
    ctx: Ctx, stage: str, variants: list[L_.Leg], refs: str, wall_iters: int = 0, wall_ls: tuple[int, ...] = ()
) -> dict:
    ref_legs = L_.legs_from_ids(refs) if refs else []
    series = run_series(ctx, ref_legs + variants, stage)
    params: dict[str, Any] = {"variants": [(v.id, v.flags) for v in variants], "refs": refs}
    doc = {"series": series, "params": params}
    if wall_iters:
        ctx.out("   wall clock (harness median, cold L2, b=1) — includes launch gaps")
        doc["rows"] = _wall_rows(ctx, variants, stage, wall_iters, wall_ls or ctx.ls)
        params["wall_iters"] = wall_iters
    return doc


RING_DEFAULT = "2 3 4 5 6"


def stage_ring(ctx: Ctx, stages: str = RING_DEFAULT, base: str = "", refs: str = "fi") -> dict:
    """KV ring depth (``--kv_stages``) at the harness default."""
    variants = [L_.t5_variant(f"stages={s}", f"{base} --kv_stages {s}".strip()) for s in stages.split()]
    variants.insert(0, L_.t5_variant("default", base))
    return _variant_stage(ctx, "ring", variants, refs)


def stage_prefetch(ctx: Ctx, distances: str = "0 1 2 4", base: str = "--kv_format fp8", refs: str = "fi") -> dict:
    """L2 prefetch distance (``--l2_prefetch``, in 128-row tiles ahead).
    Runs on fp8 KV: the prefetch path is not wired for nvfp4 and the fp6
    ring is already tuned without it."""
    variants = [L_.t5_variant(f"prefetch={k}", f"{base} --l2_prefetch {k}".strip()) for k in distances.split()]
    return _variant_stage(ctx, "prefetch", variants, refs)


AB_DEFAULT = (
    "base:--kv_evict_first 0|evict:--kv_evict_first 1|pdl:--kv_evict_first 0 --pdl|both:--kv_evict_first 1 --pdl"
)


def stage_ab(ctx: Ctx, variants: str = AB_DEFAULT, refs: str = "fi", wall_iters: int = 200) -> dict:
    """Free-form A/B of harness flag sets (``name:flags|name:flags``)."""
    return _variant_stage(ctx, "ab", L_.parse_variants(variants), refs, wall_iters)


KVFMT_DEFAULT = (
    "fp8-nhd:--kv_format fp8 --kv_layout nhd|fp8-interleaved:--kv_format fp8 --kv_layout interleaved|"
    "fp8-hnd:--kv_format fp8 --kv_layout hnd|e2m3-hnd:--kv_format e2m3 --kv_layout hnd|"
    "e3m2-hnd:--kv_format e3m2 --kv_layout hnd|e2m1-hnd:--kv_format e2m1 --kv_layout hnd|"
    "nvfp4-hnd:--kv_format nvfp4 --kv_layout hnd"
)


def stage_kvfmt(
    ctx: Ctx, variants: str = KVFMT_DEFAULT, refs: str = KERNEL["verify.kvfmt.refs"], wall_iters: int = 200
) -> dict:
    """KV cache layouts × formats (fp8 / fp6 / fp4 / nvfp4; NHD / interleaved / HND)."""
    return _variant_stage(ctx, "kvfmt", L_.parse_variants(variants), refs, wall_iters)


# ── harness stages (timing / pdl / quant) ───────────────────────────────────


def stage_timing(ctx: Ctx, dtypes: str = "fp8 bf16", ctas: str = "1 2", iters: int = 50) -> dict:
    """Harness-timed medians at every L, 1 vs 2 CTAs/SM, per Q dtype.

    The Orin GEMV has no CTAs/SM axis and no fp8-Q path yet: one config per
    (dtype, L), bf16 Q only."""
    orin = ctx.runner.platform == "orin"
    if orin:
        dtypes = "bf16"
        ctas_vals = ["1"]
    else:
        ctas_vals = ctas.split()
    rows = []
    for dt in dtypes.split():
        for L in ctx.ls:
            for c in ctas_vals:
                ctx.wait_clean()
                log = ctx.logdir / "timing" / f"{dt}-L{L}-ctas{c}.log"
                args = f"--seqlen {L} --skip_ref_check --warmup_iterations 5 --iterations {iters}"
                if not orin:
                    args += f" --ctas_per_sm {c}"
                rc, text = ctx.runner.harness(log, dt, args)
                us, gbps = _median(text)
                rows.append({"q": dt, "L": L, "ctas_per_sm": int(c), "median_us": us, "gbps": gbps, "ok": rc == 0})
                if us is not None:
                    ctx.out(f"   {dt:<5} L={L:>7,} ctas/SM={c}  {us:>10.1f} µs")
    return {"rows": rows, "params": {"dtypes": dtypes, "ctas": ctas, "iters": iters}}


def stage_orinbench(ctx: Ctx, iters: int = 200) -> dict:
    """Orin same-run comparison at every L (and the MTP M=4 verify shape):
    the GEMV kernel with prebuilt launches (the kernel number) vs the full
    serving path (runner.run: arg packing + the LSE combine) vs the
    FlashInfer FA2-TC reference — all on this host, same conditions. The
    M=1 series use the sweep schema so the README's headline context can
    read this stage wherever the ncu sweep cannot run (no ncu on L4T)."""
    assert ctx.runner.platform == "orin", "stage_orinbench runs on the Orin platform"
    # Correctness gate first: the exact configs the timing runs, plus the
    # ragged short-sequence case that exercises the masked-split paths.
    for args in (
        "--seqlen 4096 --q_len 1",
        "--seqlen 4096 --q_len 8 --batches 8",
        "--seqlen 64 --q_len 8 --batches 8",
    ):
        clog = ctx.logdir / "orinbench" / f"check-{args.replace(' ', '_').replace('--', '')}.log"
        rc, _ = ctx.runner.harness(clog, "bf16", args)
        if rc != 0:
            return {"error": f"correctness gate failed ({args}), see {clog}"}
    pts: dict[str, dict[str, list]] = {"mjolnir": {}, "mjolnir-full": {}, "fi": {}}
    m4 = []
    for M in (1, 4):
        for L in ctx.ls:
            ctx.wait_clean()
            log = ctx.logdir / "orinbench" / f"mjolnir-M{M}-L{L}.log"
            args = f"--seqlen {L} --q_len {M} --skip_ref_check --warmup_iterations 20 --iterations {iters}"
            rc, text = ctx.runner.harness(log, "bf16", args)
            us, gbps = _median(text)
            full_m = _FULL_MEDIAN.search(text or "")
            full_us = float(full_m.group(1)) if full_m else None
            flog = ctx.logdir / "orinbench" / f"fi-M{M}-L{L}.log"
            frc, fi_us = ctx.runner.driver_wall(
                flog,
                "fi_driver.py",
                L,
                iters,
                env_extra=(
                    ("FI_LAYOUT", "NHD"),
                    ("BENCH_PAGE", "16"),
                    ("BENCH_QLEN", str(M)),
                    ("BENCH_WARMUP", "20"),
                ),
            )
            kv_gb = ctx.runner.batch * L * ctx.runner.hk * ctx.runner.d * 2 * 1.0 / 1e9
            ok = rc == 0 and us is not None and frc == 0 and fi_us is not None
            row = {
                "M": M,
                "L": L,
                "kernel_us": us,
                "gbps": gbps,
                "fi_us": fi_us,
                "ratio": (fi_us / us) if ok and us else None,
                "ok": ok,
            }
            if M == 1:
                if us is not None:
                    pts["mjolnir"].setdefault("pts", []).append({"L": L, "total_us": us, "gbps": gbps})
                    pts["mjolnir-full"].setdefault("pts", []).append({"L": L, "total_us": full_us})
                if fi_us is not None:
                    pts["fi"].setdefault("pts", []).append({"L": L, "total_us": fi_us})
            else:
                m4.append(row)
            if ok:
                ctx.out(f"   M={M} L={L:>7,}  kernel {us:>9.1f} µs  FI {fi_us:>9.1f} µs  ({fi_us / us:>5.2f}x)")
            else:
                ctx.out(f"   M={M} L={L:>7,}  incomplete (mjolnir rc={rc}, fi rc={frc}) — {ctx.logdir / 'orinbench'}")

    def series(sid, label, kind, role, kv_layout, points):
        return {
            "id": sid,
            "label": label,
            "kind": kind,
            "role": role,
            "kv_format": "fp8",
            "kv_layout": kv_layout,
            "kv_bits": 8.0,
            "points": points,
        }

    return {
        "series": [
            series("mjolnir", "Mjolnir GEMV (fp8 KV, HND)", "t5", "ours", "hnd", pts["mjolnir"].get("pts", [])),
            series("mjolnir-full", "Mjolnir serving path", "t5", "ours", "hnd", pts["mjolnir-full"].get("pts", [])),
            series("fi", "FlashInfer FA2-TC (fp8 KV, NHD)", "fi", "baseline", "nhd", pts["fi"].get("pts", [])),
        ],
        # The M=4 rows and the platform tag live in the doc BODY: params only
        # carries the stage's tunable options (the default-run detection
        # compares it to the defaults).
        "m4": m4,
        "params": {"iters": iters},
    }


def stage_pdl(ctx: Ctx, batches: str = "1 8", dtypes: str = "fp8", iters: int = 200) -> dict:
    """Programmatic dependent launch on/off (wall clock — ncu serializes
    kernels, so it cannot see PDL)."""
    rows = []
    for dt in dtypes.split():
        for b in batches.split():
            for L in ctx.ls:
                base = f"--batches {b} --seqlen {L} --skip_ref_check --warmup_iterations 10 --iterations {iters}"
                if int(b) > 1:
                    base += " --varlen"
                out = {}
                for mode, extra in (("off", ""), ("on", " --pdl")):
                    ctx.wait_clean()
                    log = ctx.logdir / "pdl" / f"{dt}-b{b}-L{L}-{mode}.log"
                    _, text = ctx.runner.harness(log, dt, base + extra)
                    out[mode] = _median(text)[0]
                d = (out["on"] - out["off"]) / out["off"] * 100 if out["on"] and out["off"] else None
                rows.append(
                    {"q": dt, "batches": int(b), "L": L, "off_us": out["off"], "on_us": out["on"], "delta_pct": d}
                )
                if d is not None:
                    ctx.out(f"   {dt} b={b} L={L:>7,}  off {out['off']:.1f} µs  on {out['on']:.1f} µs  ({d:+.1f}%)")
    return {"rows": rows, "params": {"batches": batches, "dtypes": dtypes, "iters": iters}}


QUANT_FORMATS = "fp8 e2m3 e3m2 e2m1 nvfp4"


def stage_quant(ctx: Ctx, formats: str = QUANT_FORMATS, batches: int = 2) -> dict:
    """Accuracy of each KV format: kernel output vs attention over the
    unquantized float KV (same queries), relative L2 error."""
    rows = []
    for f in formats.split():
        for L in ctx.ls:
            log = ctx.logdir / "quant" / f"{f}-L{L}.log"
            extra = "--kv_format fp8 --kv_from_float" if f == "fp8" else f"--kv_format {f}"
            rc, text = ctx.runner.harness(log, "fp8", f"--batches {batches} --seqlen {L} {extra}", timeout=900)
            m = re.search(r"rel_l2=([0-9.eE+-]+)", text)
            mx = re.search(r"unquantized\| max=([0-9.eE+-]+)", text)
            rel = float(m.group(1)) if m else None
            rows.append(
                {
                    "format": f,
                    "bits": L_.KV_BITS.get(f),
                    "L": L,
                    "rel_l2": rel,
                    "max_abs": float(mx.group(1)) if mx else None,
                    "emulator_pass": bool(re.search(r"^PASS$", text, re.MULTILINE)),
                    "ok": rc == 0 and rel is not None,
                }
            )
            if rel is not None:
                ctx.out(f"   {f:<6} L={L:>7,}  rel L2 {rel:.4f}")
    return {"rows": rows, "params": {"formats": formats, "batches": batches}}


# ── fp6prof (full ncu profiles) ─────────────────────────────────────────────

PROF_DEFAULT = (
    "fp8-hnd:--kv_format fp8 --kv_layout hnd|e2m3-hnd:--kv_format e2m3 --kv_layout hnd|"
    "e2m1-hnd:--kv_format e2m1 --kv_layout hnd"
)
# The metrics kept per profile (the full set is ~1000 metrics x launches —
# the .ncu-rep files keep everything; the JSON keeps what the charts read).
PROF_METRICS = (
    "gpu__time_duration.sum",
    "lts__t_sectors.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_requests_srcunit_tex_op_read.sum",
    "lts__t_sector_hit_rate.pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed.sum",
    "smsp__cycles_active.avg",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic",
    "launch__grid_size",
)


def stage_fp6prof(ctx: Ctx, variants: str = PROF_DEFAULT, count: int = 3, section_set: str = "full") -> dict:
    """Full ncu profiles of the decode kernel per KV format: bytes moved vs
    time, L2 request granularity, warp stalls — why fp6 saves fewer µs than
    bytes. ``.ncu-rep`` files land in the run's log dir (open in Nsight Compute)."""
    profiles = []
    for leg in L_.parse_variants(variants):
        for L in ctx.ls:
            ctx.wait_clean()
            name = f"{_slug(leg.id)}-L{L}"
            ctx.out(f"   {leg.label} L={L:,}")
            rc, prof = ctx.runner.ncu_profile(ctx.logdir / "fp6prof", name, leg, L, count, section_set)
            if rc != 0 or not prof:
                profiles.append({"series": leg.id, "label": leg.label, "L": L, "ok": False})
                continue
            raw = prof["raw"]
            keep = {k: raw[k] for k in PROF_METRICS if k in raw}
            stalls = {
                k.replace("smsp__average_warps_issue_stalled_", "").replace("_per_issue_active.ratio", ""): v
                for k, v in raw.items()
                if k.startswith("smsp__average_warps_issue_stalled_") and k.endswith("_per_issue_active.ratio")
            }
            pipes = {
                k: v
                for k, v in raw.items()
                if re.search(r"pipe_(tc|tensor|tma|uniform)|tmem|utc", k) and "pct_of_peak" in k
            }
            profiles.append(
                {
                    "series": leg.id,
                    "label": leg.label,
                    "flags": leg.flags,
                    "kv_format": leg.kv_format,
                    "kv_bits": leg.kv_bits,
                    "L": L,
                    "ok": True,
                    "metrics": keep,
                    "stalls": stalls,
                    "pipes": pipes,
                    "details": prof["details"],
                    "report": str(ctx.logdir / "fp6prof" / f"{name}.ncu-rep"),
                }
            )
    return {"profiles": profiles, "params": {"variants": variants, "count": count, "set": section_set}}


# ── correctness ─────────────────────────────────────────────────────────────

# (label, args, q dtypes). Cases pin the vLLM production config (fp8 KV, NHD)
# unless they pass their own --kv_format / --kv_layout (later flags win);
# "default:" cases run the bare harness default (e2m3 on HND for fp8 Q).
_PIN = "--kv_format fp8 --kv_layout nhd"
BOTH = ("fp8", "bf16")
FP8 = ("fp8",)
CASES: list[tuple[str, str, tuple[str, ...]]] = [
    ("L=8192", "--seqlen 8192", BOTH),
    ("L=128 (1 chunk, direct out)", "--seqlen 128", BOTH),
    ("L=1000 (partial tile mask)", "--seqlen 1000", BOTH),
    ("L=1 (minimal)", "--seqlen 1", BOTH),
    ("L=65536 (ring wraps)", "--seqlen 65536", BOTH),
    ("varlen 6x<=20000", "--batches 6 --seqlen 20000 --varlen", BOTH),
    ("varlen 32x<=2048 (>1 wave)", "--batches 32 --seqlen 2048 --varlen", BOTH),
    ("spec q_len=2 varlen 4x<=3000", "--batches 4 --seqlen 3000 --varlen --q_len 2", BOTH),
    ("spec q_len=4 2x777", "--batches 2 --seqlen 777 --q_len 4", BOTH),
    ("page 16 varlen", "--batches 4 --seqlen 1000 --varlen --page_size 16", BOTH),
    ("page 64 varlen", "--batches 4 --seqlen 1000 --varlen --page_size 64", BOTH),
    ("contiguous pages", "--seqlen 8192 --contiguous_pages", BOTH),
    ("2 CTAs/SM (40-CTA plan)", "--seqlen 8192 --ctas_per_sm 2", BOTH),
    ("forced 7 chunks", "--seqlen 8192 --kv_splits 7", BOTH),
    ("padded work list (40)", "--batches 3 --seqlen 5000 --varlen --pad_work_to 40", BOTH),
    ("max_splits=1 (no reduction)", "--seqlen 8192 --max_splits 1", BOTH),
    ("headdim 128", "--seqlen 8192 --headdim 128", BOTH),
    ("KV interleaved (B,H,N,2D)", "--batches 4 --seqlen 3000 --varlen --page_size 16 --kv_layout interleaved", BOTH),
    ("KV flashinfer (B,2,N,H,D)", "--batches 4 --seqlen 3000 --varlen --page_size 32 --kv_layout flashinfer", BOTH),
    ("strided Q/O (fused QKV view)", "--batches 4 --seqlen 3000 --varlen --strided_qo", BOTH),
    ("per-tensor descales", "--batches 4 --seqlen 3000 --varlen --descale per_tensor", BOTH),
    ("per-head descales", "--batches 4 --seqlen 3000 --varlen --descale per_head", BOTH),
    (
        "vLLM-like: interl+strided+tensor p16",
        "--batches 8 --seqlen 6000 --varlen --page_size 16 --kv_layout interleaved --strided_qo --descale per_tensor",
        BOTH,
    ),
    ("PDL (decode + reduction)", "--batches 4 --seqlen 20000 --varlen --pdl", BOTH),
    ("L2 prefetch 2, page 128", "--batches 2 --seqlen 30000 --varlen --l2_prefetch 2", BOTH),
    ("L2 prefetch 4, page 16", "--batches 4 --seqlen 9000 --varlen --page_size 16 --l2_prefetch 4", BOTH),
    ("PDL + L2 prefetch 1, q_len=2", "--batches 3 --seqlen 5000 --varlen --q_len 2 --pdl --l2_prefetch 1", BOTH),
    ("HND layout p128", "--seqlen 8192 --kv_layout hnd", BOTH),
    ("HND layout p16 varlen", "--batches 4 --seqlen 3000 --varlen --page_size 16 --kv_layout hnd", BOTH),
    ("fp8 from float (accuracy ref)", "--batches 2 --seqlen 5000 --varlen --kv_from_float", FP8),
    ("e3m2 KV p128", "--seqlen 8192 --kv_format e3m2", FP8),
    ("e2m3 KV p16 varlen", "--batches 4 --seqlen 3000 --varlen --page_size 16 --kv_format e2m3", FP8),
    (
        "e2m3 KV vLLM layout (interl p128 q_len=4)",
        "--batches 4 --seqlen 5000 --varlen --page_size 128 --q_len 4 --kv_format e2m3 --kv_layout interleaved "
        "--strided_qo",
        FP8,
    ),
    ("e2m1 KV p128", "--seqlen 8192 --kv_format e2m1", FP8),
    (
        "e2m1 KV HND p16 q_len=2",
        "--batches 3 --seqlen 2500 --varlen --page_size 16 --q_len 2 --kv_layout hnd --kv_format e2m1",
        FP8,
    ),
    ("nvfp4 KV p128", "--seqlen 8192 --kv_format nvfp4", FP8),
    ("nvfp4 KV p16 varlen", "--batches 4 --seqlen 3000 --varlen --page_size 16 --kv_format nvfp4", FP8),
    (
        "nvfp4 KV HND q_len=2 pad",
        "--batches 3 --seqlen 2500 --varlen --q_len 2 --kv_layout hnd --kv_format nvfp4 --pad_work_to 40",
        FP8,
    ),
    ("default: e2m3 HND L=8192", "@default --seqlen 8192", BOTH),
    ("default: L=65536 (ring wraps)", "@default --seqlen 65536", BOTH),
    ("default: varlen 6x<=20000", "@default --batches 6 --seqlen 20000 --varlen", BOTH),
    (
        "default: q_len=3 p16 pad",
        "@default --batches 4 --seqlen 2000 --varlen --q_len 3 --page_size 16 --pad_work_to 40",
        BOTH,
    ),
    ("default: headdim 128 + PDL", "@default --batches 2 --seqlen 9000 --varlen --headdim 128 --pdl", BOTH),
    # parallel drafters (DSpark / DFlash): a non-causal block of 1 + 7 queries,
    # the DSpark drafter's heads (40 query / 8 KV, d = 128)
    (
        "non-causal q_len=8 (drafter block)",
        "@default --batches 4 --seqlen 5000 --varlen --q_len 8 --heads_q 40 --heads_k 8 --headdim 128 --non_causal",
        BOTH,
    ),
    (
        "non-causal q_len=8 fp8 p16 pad",
        "--batches 3 --seqlen 2500 --varlen --q_len 8 --heads_q 40 --heads_k 8 --headdim 128 --page_size 16 "
        "--non_causal --pad_work_to 40",
        BOTH,
    ),
    ("ring depth 2, fp8 (ring wraps)", "--seqlen 65536 --kv_layout hnd --kv_stages 2", BOTH),
    ("ring depth 2, e2m3 varlen p16", "@default --batches 4 --seqlen 9000 --varlen --page_size 16 --kv_stages 2", FP8),
    ("kv evict_first", "--batches 4 --seqlen 20000 --varlen --kv_evict_first", BOTH),
    ("evict + PDL L=8192", "--seqlen 8192 --kv_evict_first --pdl", BOTH),
    (
        "evict + PDL q_len=3 + pad",
        "--batches 4 --seqlen 2000 --varlen --q_len 3 --kv_evict_first --pdl --pad_work_to 40",
        BOTH,
    ),
    (
        "evict + PDL, interl p16",
        "--batches 4 --seqlen 9000 --varlen --page_size 16 --kv_layout interleaved --kv_evict_first --pdl",
        BOTH,
    ),
    (
        "vLLM-like spec q_len=3 p16",
        "--batches 4 --seqlen 2000 --varlen --q_len 3 --page_size 16 --kv_layout interleaved --strided_qo "
        "--descale per_tensor --pad_work_to 40",
        BOTH,
    ),
    # E1 fused last-CTA merge (counter self-reset covered by the replay case)
    ("fused merge e2m3 HND varlen", "@default --batches 2 --seqlen 16384 --varlen --fused_merge", FP8),
    ("fused merge L=65536, replay x8", "@default --seqlen 65536 --fused_merge --iterations 8", FP8),
    ("fused merge q_len=2 fp8 HND", "--batches 3 --seqlen 9000 --varlen --q_len 2 --kv_layout hnd --fused_merge", FP8),
]


# One representative case per group: the fast pre-check autoresearch runs on
# every experiment (~10 containers) before the full matrix is worth paying for.
QUICK_CASES = (
    "L=8192",
    "L=1000 (partial tile mask)",
    "L=65536 (ring wraps)",
    "varlen 6x<=20000",
    "spec q_len=2 varlen 4x<=3000",
    "page 16 varlen",
    "KV interleaved (B,H,N,2D)",
    "HND layout p16 varlen",
    "e2m3 KV p16 varlen",
    "default: e2m3 HND L=8192",
    "default: varlen 6x<=20000",
    "evict + PDL q_len=3 + pad",
    "non-causal q_len=8 (drafter block)",
    "fused merge e2m3 HND varlen",
)


def case_args(args: str) -> str:
    if args.startswith("@default"):
        return args[len("@default") :].strip()
    return f"{_PIN} {args}"


LABEL_W = max(len(label) for label, _, _ in CASES)


def _short_check(check: str | None) -> str:
    """'|kern-emu| max=0.03051 mean=0.0007534' -> 'max 3.05e-02  mean 7.53e-04'
    (fixed width, so the live correctness table never wraps)."""
    if not check:
        return ""
    m = re.search(r"max=([0-9.eE+-]+)\s+mean=([0-9.eE+-]+)", check)
    if not m:
        return check[:26]
    try:
        return f"max {float(m.group(1)):.2e}  mean {float(m.group(2)):.2e}"
    except ValueError:
        return check[:26]


_ORIN_CHECK = re.compile(
    r"check b=(\d+) M=(\d+) L<=(\d+) kv=(\w+) split=(\w+): rel_err=([0-9.na]+) \(tol [0-9.]+\) -> (OK|FAIL)"
)


def _orin_correctness(ctx: Ctx, timeout: float) -> dict:
    """The Orin GEMV's correctness matrix: one container runs the harness
    sweep (page size x M x L against the fp32 reference); the check lines
    parse into the shared case schema."""
    log = ctx.logdir / "correctness" / "orin-sweep.log"
    t0 = time.time()
    rc, text = ctx.runner.harness(log, "bf16", "--sweep", timeout=timeout)
    cases = []
    for m in _ORIN_CHECK.finditer(text):
        b, M, L, kv, split, rel, verdict = m.groups()
        cases.append(
            {
                "label": f"b={b} M={M} L<={int(L):,} kv={kv} split={split}",
                "q": "bf16",
                "args": ["--batches", b, "--q_len", M, "--seqlen", L, "--kv_dtype", kv],
                "passed": verdict == "OK",
                "rc": rc,
                "seconds": round(time.time() - t0, 1),
                "check": f"rel_err={rel} (tol 0.05)",
                "log": str(log),
            }
        )
    for c in cases:
        mark = "✅" if c["passed"] else "❌"
        ctx.out(f" {mark} {c['label']:<40} {c['check']}")
    if not cases and not ctx.runner.dry_run:
        ctx.out(f" ❌ no check lines parsed — {log}")
    return {"cases": cases, "params": {"matrix": "ps x M x L (orin harness --sweep)"}}


def stage_correctness(
    ctx: Ctx,
    dtypes: str = "fp8 bf16",
    match: str = "",
    fail_fast: bool = False,
    timeout: float = 180.0,
    quick: bool = False,
) -> dict:
    """Kernel vs the bit-faithful emulator: output + LSE + poisoned-page NaN
    check, for every case x Q dtype."""
    if ctx.runner.platform == "orin":
        return _orin_correctness(ctx, timeout)
    want = set(dtypes.split())
    cases = []
    for label, args, dts in CASES:
        if match and match.lower() not in label.lower():
            continue
        if quick and label not in QUICK_CASES:
            continue
        for dt in dts:
            if dt not in want:
                continue
            log = ctx.logdir / "correctness" / f"{len(cases) + 1:02d}-{dt}-{_slug(label)}.log"
            t0 = time.time()
            rc, text = ctx.runner.harness(log, dt, case_args(args), timeout=timeout)
            ok = rc == 0 and bool(re.search(r"^PASS$", text, re.MULTILINE))
            check = re.search(r"^check: (.*?)(;|$)", text, re.MULTILINE)
            plan = re.search(r"plan: SplitPlan\((.*)\)", text)
            c = {
                "label": label,
                "q": dt,
                "args": case_args(args),
                "passed": ok or ctx.runner.dry_run,
                "rc": rc,
                "seconds": round(time.time() - t0, 1),
                "check": check.group(1) if check else None,
                "plan": plan.group(1) if plan else None,
                "log": str(log),
            }
            cases.append(c)
            mark = "✅" if c["passed"] else "❌"
            ctx.out(f" {mark} {dt:<5} {label:<{LABEL_W}} {_short_check(c['check']):<26} {c['seconds']:>4.0f}s")
            if not c["passed"]:
                for line in [
                    ln for ln in text.splitlines() if re.search(r"Error|Traceback|Assertion|NaN|Mismatch", ln)
                ][:3]:
                    ctx.out(f"      │ {line}")
                if fail_fast:
                    return {"cases": cases, "params": {"dtypes": dtypes, "match": match, "quick": quick}}
    return {"cases": cases, "params": {"dtypes": dtypes, "match": match, "quick": quick}}


# ── vllm (runner tests + end-to-end smoke) ──────────────────────────────────


def parse_smoke(text: str) -> dict:
    """Metrics from a ``smoke_vllm.py`` log (absent ones are None)."""
    m = re.search(r"mean match ([0-9.]+), worst prefix (\d+)", text)
    lp = re.search(
        r"logprob agreement: mean \|Δ\| (\S+), max \|Δ\| (\S+), near-ties (\d+)/(\d+), worst gap (\S+)", text
    )

    def num(v: str) -> float | None:
        return None if v == "n/a" else float(v.rstrip(","))

    return {
        "mean_match": float(m.group(1)) if m else None,
        "worst_prefix": int(m.group(2)) if m else None,
        "mean_dlp": num(lp.group(1).rstrip(",")) if lp else None,
        "max_dlp": num(lp.group(2).rstrip(",")) if lp else None,
        "near_ties": int(lp.group(3)) if lp else None,
        "divergences": int(lp.group(4)) if lp else None,
        "worst_gap": num(lp.group(5)) if lp else None,
    }


def smoke_summary(c: dict) -> str:
    """One compact line for a smoke case: logprob agreement, then the match."""
    parts = []
    if c.get("mean_dlp") is not None:
        parts.append(f"|Δlp| {c['mean_dlp']:.3f} mean · {c['max_dlp']:.3f} max")
    if c.get("divergences") is not None:
        parts.append(f"near-ties {c['near_ties']}/{c['divergences']}")
    if c.get("mean_match") is not None:
        parts.append(f"match {c['mean_match']:.2f}")
    return ("  " + " · ".join(parts)) if parts else ""


def stage_vllm(
    ctx: Ctx,
    repo: Path,
    smoke_modes: str = SMOKE_MODES,
    model: str = "Qwen/Qwen3-0.6B",
    tokens: int = 128,
    timeout: float = 1800.0,
    hf_home: Path | None = None,
) -> dict:
    """The vLLM plugin: runner-vs-emulator tests (every KV layout, strided Q,
    scales, q_len 2, CUDA-graph replay), then greedy MJOLNIR-vs-FLASHINFER
    smoke runs in a real ``vllm`` process, judged by logprob agreement and
    near-tie divergences (see ``tests/kernel/smoke_vllm.py``)."""
    hf = hf_home or Path.home() / ".cache" / "huggingface"
    cases = []
    log = ctx.logdir / "vllm" / "test_runner.log"
    rc = ctx.runner.tests(log, repo, "python3 tests/test_runner.py", timeout, hf)
    text = log.read_text(errors="replace") if log.exists() else ""
    for line in text.splitlines():
        m = re.search(r"(✅|❌) (test_\S+)", line)
        if m:
            cases.append({"label": f"runner {m.group(2).rstrip(':')}", "passed": m.group(1) == "✅", "log": str(log)})
            ctx.out("   " + line.strip())
    if rc != 0 and not any(not c["passed"] for c in cases):
        cases.append({"label": f"runner tests (exit {rc})", "passed": ctx.runner.dry_run, "log": str(log)})
    for mode in smoke_modes.split():
        args = f"--model {model} --max_tokens {tokens}"
        if mode == "eager":
            args += " --eager"
        elif mode == "e2m3":  # Mjolnir on a 6-bit e2m3 KV cache (calibrates its own scales)
            args += " --kv fp6_e2m3"
        elif mode == "lmh8":  # Mjolnir with the opt-in fp8 lm_head (MJOLNIR_LM_HEAD=fp8)
            args += " --lm-head fp8"
        elif mode.startswith("spec"):
            args += f" --spec {mode[4:] or 2}"
        elif mode != "graph":
            continue
        log = ctx.logdir / "vllm" / f"smoke-{mode}.log"
        rc = ctx.runner.tests(log, repo, f"python3 tests/smoke_vllm.py {args}", timeout, hf)
        text = log.read_text(errors="replace") if log.exists() else ""
        ok = (rc == 0 and bool(re.search(r"^PASS$", text, re.MULTILINE))) or ctx.runner.dry_run
        cases.append({"label": f"smoke {mode}", "passed": ok, **parse_smoke(text), "log": str(log)})
        ctx.out(f" {'✅' if ok else '❌'} smoke {mode:<6}" + smoke_summary(cases[-1]))
        if not ok:
            for line in re.findall(r"^  ❌ (.*)$", text, re.MULTILINE)[:3]:
                ctx.out(f"      │ {line}")
    return {"cases": cases, "params": {"smoke_modes": smoke_modes, "model": model, "tokens": tokens}}


def _profile_stage(ctx: Ctx, **kw) -> dict:
    from mjolnir_cli.verify.profile import stage_profile

    return stage_profile(ctx, **kw)


def _spec_stage(ctx: Ctx, **kw) -> dict:
    from mjolnir_cli.verify.spec import stage_spec

    return stage_spec(ctx, **kw)


def _kvcache_stage(ctx: Ctx, **kw) -> dict:
    from mjolnir_cli.verify.kvcache import stage_kvcache

    return stage_kvcache(ctx, **kw)


# ── registry ────────────────────────────────────────────────────────────────

STAGES: dict[str, Stage] = {
    s.name: s
    for s in [
        Stage(
            "correctness",
            "Correctness matrix",
            "correctness",
            "kernel vs the bit-faithful emulator: every layout, format, page size, q_len, Q dtype",
            stage_correctness,
        ),
        Stage(
            "sweep",
            "Headline L-sweep",
            "ncu",
            "Mjolnir vs FlashInfer vs FlashAttention 4 hd256 (fp8 Q+KV, on the Mjolnir image), ncu kernel time at every L",
            stage_sweep,
        ),
        Stage(
            "hndsweep",
            "Head-major pages",
            "ncu",
            "FlashInfer on NHD and HND vs Mjolnir on HND — is HND a kernel win or a layout win?",
            stage_hndsweep,
        ),
        Stage(
            "kvfmt",
            "KV layouts and formats",
            "ncu",
            "fp8 / fp6 / fp4 / nvfp4 KV on NHD / interleaved / HND pages (ncu + wall clock)",
            stage_kvfmt,
        ),
        Stage(
            "quant",
            "KV format accuracy",
            "harness",
            "relative L2 error of each KV format vs unquantized KV, at every L",
            stage_quant,
        ),
        Stage("ring", "KV ring depth", "ncu", "TMA→MMA ring depth (--kv_stages) sweep at every L", stage_ring),
        Stage("prefetch", "L2 prefetch", "ncu", "TMA L2 prefetch distance (--l2_prefetch) sweep", stage_prefetch),
        Stage(
            "pdl",
            "Programmatic dependent launch",
            "harness",
            "PDL on/off wall clock (decode overlaps the previous kernel; merge launches early)",
            stage_pdl,
        ),
        Stage(
            "ab", "Flag A/B", "ncu", "any harness flag sets, ncu + wall clock (default: evict-first x PDL)", stage_ab
        ),
        Stage(
            "fp6prof",
            "Full profiles per KV format",
            "profile",
            "full ncu profiles: L2 sectors vs requests vs time, warp stalls, per KV format",
            stage_fp6prof,
        ),
        Stage(
            "timing", "Harness timing", "harness", "harness-timed medians, 1 vs 2 CTAs/SM, fp8 and bf16 Q", stage_timing
        ),
        Stage(
            "orinbench",
            "Orin same-run comparison",
            "harness",
            "GEMV kernel vs full serving path vs FlashInfer FA2-TC, events-timed at every L (Orin only)",
            stage_orinbench,
        ),
        Stage(
            "kv-cache-size",
            "KV cache capacity",
            "serving",
            "tokens of KV each config fits (vLLM's startup sizing), and the concurrency that buys per context",
            _kvcache_stage,
        ),
        Stage(
            "spec",
            "Speculative decoding",
            "serving",
            "MTP acceptance per draft position and decode t/s for 1-N draft tokens, per config — the best draft count",
            _spec_stage,
        ),
        Stage(
            "profile",
            "Decode step profile",
            "serving",
            "torch-profiler trace of a decode workload per config: CUDA-graph replays, eager kernels, GPU idle",
            _profile_stage,
        ),
        Stage(
            "vllm",
            "vLLM integration",
            "tests",
            "runner-vs-emulator tests + greedy MJOLNIR vs FLASHINFER smoke runs",
            stage_vllm,
        ),
    ]
}

# `mjolnir verify all` / `mjolnir verify bench`
ALL_ORDER = ["correctness", "sweep", "hndsweep", "kvfmt", "quant", "ring", "prefetch", "pdl", "ab", "fp6prof"]
BENCH_ORDER = [s for s in ALL_ORDER if s != "correctness"]


def summarize(stage: str, doc: dict) -> dict:
    """Headline numbers stored with the run (the README recomputes from the
    series; this is for `mjolnir verify show` and the ledger)."""
    out: dict = {}
    by = results.series_by_id(doc)
    if stage in ("sweep", "hndsweep"):
        ours = by.get("mjolnir") or by.get("mjolnir-fp8")
        for base_id in ("fi", "fi-hnd", "fa2", "fa4"):
            if ours and base_id in by:
                sp = results.speedup(by[base_id], ours)
                if sp:
                    out[f"vs_{base_id}"] = {
                        "geomean": round(results.geomean(sp.values()) or 0, 4),
                        "min": round(min(sp.values()), 4),
                        "max": round(max(sp.values()), 4),
                        "per_L": {str(k): round(v, 4) for k, v in sp.items()},
                    }
    if stage == "kv-cache-size":
        out = {
            r["id"]: {"kv_tokens": r.get("kv_tokens"), "max_concurrency": r.get("max_concurrency")}
            for r in doc.get("rows", [])
        }
    if stage == "spec":
        from mjolnir_cli.verify.spec import best_k

        out = {
            f"{cfg} c={conc}": {"k": r["k"], "decode_tps": r["decode_tps"], "acceptance_length": r["acceptance_length"]}
            for (cfg, conc), r in sorted(best_k(doc.get("rows", [])).items())
        }
    if stage == "correctness" or stage == "vllm":
        cs = doc.get("cases", [])
        out = {"passed": sum(1 for c in cs if c.get("passed")), "failed": sum(1 for c in cs if not c.get("passed"))}
    return out


def eprint(*a):
    print(*a, file=sys.stderr)
