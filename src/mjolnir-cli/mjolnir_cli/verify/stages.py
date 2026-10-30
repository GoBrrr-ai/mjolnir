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
    # True = the stage cannot run without Nsight Compute on any platform; the
    # sweep is False (dual: the ncu legs on thor, the events-timed branch on
    # the no-ncu platforms). The kernel-tree chapters filter on it.
    requires_ncu: bool = False


def _median(text: str) -> tuple[float | None, float | None]:
    m = _MEDIAN.search(text or "")
    if not m:
        return None, None
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else None)


# ── ncu sweeps (sweep / hndsweep / ring / prefetch / ab / kvfmt) ────────────


def run_series(ctx: Ctx, legs: list[L_.Leg], stage: str, cache: str = "all", clock: str = "none") -> list[dict]:
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


SWEEP_LEGS = KERNEL["verify.sweep"]  # mjolnir_cli.legs


def stage_sweep(ctx: Ctx, legs: str = SWEEP_LEGS, page: int | None = None, batch: int = 1, q_len: int = 1) -> dict:
    """Every leg at every L. ``batch`` sequences of L tokens with ``q_len``
    query tokens each (default one decode token of one sequence; serving with
    MTP=3 verifies 4 tokens per sequence: ``batch=4, q_len=4``)."""
    if ctx.runner.platform == "orin":
        # No ncu on L4T: the events-timed same-run comparison IS this
        # platform's headline sweep (the legs are its series ids).
        return _orin_sweep(ctx, legs, q_len)
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


_slug = L_.slug  # the shared leg-id slug (the correctness module imports it too)


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
    if ctx.runner.platform == "orin":
        # The mma kernel has no tunable KV ring: no depth axis to sweep.
        ctx.out("   (N/A on Orin: the mma kernel has no KV ring — nothing to measure)")
        return {"error": "the Orin mma kernel has no KV ring — nothing to measure"}
    variants = [L_.t5_variant(f"stages={s}", f"{base} --kv_stages {s}".strip()) for s in stages.split()]
    variants.insert(0, L_.t5_variant("default", base))
    return _variant_stage(ctx, "ring", variants, refs)


def stage_prefetch(ctx: Ctx, distances: str = "0 1 2 4", base: str = "--kv_format fp8", refs: str = "fi") -> dict:
    """L2 prefetch distance (``--l2_prefetch``, in 128-row tiles ahead).
    Runs on fp8 KV: the prefetch path is not wired for nvfp4 and the fp6
    ring is already tuned without it."""
    if ctx.runner.platform == "orin":
        # The mma kernel issues no L2 prefetch: no distance axis.
        ctx.out("   (N/A on Orin: the mma kernel has no L2-prefetch knob — nothing to measure)")
        return {"error": "the Orin mma kernel has no L2-prefetch knob — nothing to measure"}
    variants = [L_.t5_variant(f"prefetch={k}", f"{base} --l2_prefetch {k}".strip()) for k in distances.split()]
    return _variant_stage(ctx, "prefetch", variants, refs)


def stage_ab(ctx: Ctx, variants: str = "", refs: str = "fi", wall_iters: int = 200) -> dict:
    """Free-form A/B of harness flag sets (``name:flags|name:flags``)."""
    variants = variants or L_.stage_variants(L_.AB_VARIANTS, ctx.runner.platform)
    return _variant_stage(ctx, "ab", L_.parse_variants(variants), refs, wall_iters)


def stage_kvfmt(ctx: Ctx, variants: str = "", refs: str = KERNEL["verify.kvfmt.refs"], wall_iters: int = 200) -> dict:
    """KV cache layouts × formats (fp8 / fp6 / fp4 / nvfp4; NHD / interleaved / HND)."""
    variants = variants or L_.stage_variants(L_.KVFMT_VARIANTS, ctx.runner.platform)
    return _variant_stage(ctx, "kvfmt", L_.parse_variants(variants), refs, wall_iters)


# ── harness stages (timing / pdl / quant) ───────────────────────────────────


def stage_timing(ctx: Ctx, dtypes: str = "fp8 bf16", ctas: str = "1 2", iters: int = 50) -> dict:
    """Harness-timed medians at every L, 1 vs 2 CTAs/SM, per Q dtype.

    The Orin mma kernel pins one CTA/SM configuration: one config per
    (dtype, L) — bf16 Q and fp8 Q (the runner's dynamic qpack) on fp8 KV."""
    orin = ctx.runner.platform == "orin"
    ctas_vals = ["1"] if orin else ctas.split()
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


def _orin_sweep(ctx: Ctx, legs: str, q_len: int) -> dict:
    """The sweep's Orin branch: the same-run comparison at every L with
    prebuilt launches (the kernel number), the full serving path
    (runner.run: arg packing + the LSE combine) and the FlashInfer FA2-TC
    reference -- all on this host, same conditions, on the stock image.
    No ncu exists on L4T, so this branch IS the platform's headline sweep;
    the series carry the sweep schema (id/label/kind/role/points) and save
    under the sweep stage with the platform tag, so every reader -- charts,
    README context, tables -- consumes them unchanged."""
    want = set(legs.split(",")) if legs else {"mjolnir", "mjolnir-full", "fi"}
    # Correctness gate first: the exact configs the timing runs, plus the
    # ragged short-sequence case that exercises the masked-split paths.
    for args in (
        f"--seqlen 4096 --q_len {q_len}",
        "--seqlen 4096 --q_len 8 --batches 8",
        "--seqlen 64 --q_len 8 --batches 8",
    ):
        clog = ctx.logdir / "sweep" / f"check-{args.replace(' ', '_').replace('--', '')}.log"
        rc, _ = ctx.runner.harness(clog, "bf16", args)
        if rc != 0:
            ctx.out(f"   ❌ correctness gate failed ({args}) — {clog}")
            return {"error": f"correctness gate failed ({args}), see {clog}"}
    pts: dict[str, list] = {"mjolnir": [], "mjolnir-full": [], "fi": []}
    for L in ctx.ls:
        ctx.wait_clean()
        log = ctx.logdir / "sweep" / f"mjolnir-L{L}.log"
        args = f"--seqlen {L} --q_len {q_len} --skip_ref_check --warmup_iterations 20 --iterations {ctx.iters}"
        rc, text = ctx.runner.harness(log, "bf16", args)
        us, gbps = _median(text)
        full_m = _FULL_MEDIAN.search(text or "")
        full_us = float(full_m.group(1)) if full_m else None
        flog = ctx.logdir / "sweep" / f"fi-L{L}.log"
        frc, fi_us = ctx.runner.driver_wall(
            flog,
            "fi_driver.py",
            L,
            ctx.iters,
            env_extra=(("FI_LAYOUT", "NHD"), ("BENCH_PAGE", "16"), ("BENCH_QLEN", str(q_len)), ("BENCH_WARMUP", "20")),
        )
        ok = rc == 0 and us is not None and frc == 0 and fi_us is not None
        if us is not None:
            pts["mjolnir"].append({"L": L, "total_us": us, "gbps": gbps})
            pts["mjolnir-full"].append({"L": L, "total_us": full_us})
        if fi_us is not None:
            pts["fi"].append({"L": L, "total_us": fi_us})
        if ok:
            ctx.out(
                f"   L={L:>7,}  kernel {us:>9.1f} µs  full {full_us or 0:>9.1f} µs  FI {fi_us:>9.1f} µs  ({fi_us / us:>5.2f}x)"
            )
        else:
            ctx.out(f"   L={L:>7,}  incomplete (mjolnir rc={rc}, fi rc={frc}) — {ctx.logdir / 'sweep'}")

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

    all_series = [
        series("mjolnir", "Mjolnir MMA (fp8 KV, HND)", "t5", "ours", "hnd", pts["mjolnir"]),
        series("mjolnir-full", "Mjolnir serving path", "t5", "ours", "hnd", pts["mjolnir-full"]),
        series("fi", "FlashInfer FA2-TC (fp8 KV, NHD)", "fi", "baseline", "nhd", pts["fi"]),
    ]
    return {"series": [s for s in all_series if s["id"] in want], "params": {}}


def stage_pdl(ctx: Ctx, batches: str = "1 8", dtypes: str = "fp8", iters: int = 200) -> dict:
    """Programmatic dependent launch on/off (wall clock — ncu serializes
    kernels, so it cannot see PDL)."""
    if ctx.runner.platform == "orin":
        # PDL needs sm_90+; the Orin mma family has no launch-overlap lever.
        ctx.out("   (N/A on Orin: PDL does not exist below sm_90 — nothing to measure)")
        return {"error": "PDL does not exist below sm_90 — nothing to measure on Orin"}
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


def stage_quant(ctx: Ctx, formats: str = "", batches: int = 2) -> dict:
    formats = formats or L_.stage_variants(L_.QUANT_FORMATS, ctx.runner.platform)
    """Accuracy of each KV format: kernel output vs attention over the
    unquantized float KV (same queries), relative L2 error."""
    rows = []
    for f in formats.split():
        for L in ctx.ls:
            log = ctx.logdir / "quant" / f"{f}-L{L}.log"
            extra = "--kv_format fp8 --kv_from_float" if f == "fp8" else f"--kv_format {f}"
            # The fp6 reference dequantizes the whole pool on the host on top
            # of the same per-page gather fp8's reference runs — the 900s
            # default ran out at L≥65K on the Orin (the e2m3 rows came back
            # "—" in orin2's §6.3) — scale the window with L.
            rc, text = ctx.runner.harness(
                log, "fp8", f"--batches {batches} --seqlen {L} {extra}", timeout=900 if L <= 32768 else 2400
            )
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


def stage_fp6prof(ctx: Ctx, variants: str = "", count: int = 3, section_set: str = "full") -> dict:
    variants = variants or L_.stage_variants(L_.PROF_VARIANTS, ctx.runner.platform)
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
    orin = ctx.runner.platform == "orin"
    log = ctx.logdir / "vllm" / "test_runner.log"
    # Each family's runner suite (no vLLM): tests/test_runner.py drives the
    # Thor runner against its bit-faithful emulator; tests/test_orin_runner.py
    # drives the Orin runner (the mma kernel) against the fp32 reference on the
    # vLLM 0.32 serving contract -- live device seq_lens, FULL-graph replay
    # without a metadata rebuild, the fused draft advance, padding rows,
    # the non-causal drafter block.
    suite = "tests/test_orin_runner.py" if orin else "tests/test_runner.py"
    rc = ctx.runner.tests(log, repo, f"python3 {suite}", timeout, hf)
    text = log.read_text(errors="replace") if log.exists() else ""
    for line in text.splitlines():
        m = re.search(r"(✅|❌) (test_\S+)", line)
        if m:
            cases.append({"label": f"runner {m.group(2).rstrip(':')}", "passed": m.group(1) == "✅", "log": str(log)})
            ctx.out("   " + line.strip())
    if rc != 0 and not any(not c["passed"] for c in cases):
        cases.append({"label": f"runner tests (exit {rc})", "passed": ctx.runner.dry_run, "log": str(log)})
    if orin:
        # The plugin smoke needs a head_size-256 model on this platform (both
        # Orin kernels serve hd256; the default smoke model is hd128 and would
        # run entirely on the fallback). The served end-to-end check is the
        # spec stage against the AWQ_MJOLNIR config.
        ctx.out("   (smoke: N/A on Orin -- the Orin kernels serve head_size 256, the smoke model is hd128)")
        return {"cases": cases, "params": {"smoke_modes": "", "model": model, "tokens": tokens}}
    # The smoke modes: e2m3 needs the vLLM fp6 cache-write patch (the
    # Mjolnir image), lmh8 needs fp8 tensor cores (sm_100+), spec needs a
    # served drafter.
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
from mjolnir_cli.verify.correctness import CASES as CASES  # noqa: E402 — the re-export (the tests/docs read stages.CASES); correctness.py owns the definition
from mjolnir_cli.verify.correctness import QUICK_CASES as QUICK_CASES  # noqa: E402 — the re-export, same surface
from mjolnir_cli.verify.correctness import case_args as case_args  # noqa: E402 — the re-export, same surface
from mjolnir_cli.verify.correctness import stage_correctness  # noqa: E402 (below the shared helpers it reuses)

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
            "Mjolnir vs FlashInfer (FA4 hd256 on Thor, FA2-TC on Orin), fp8 KV — ncu kernel time on Thor, "
            "events-timed where there is no ncu",
            stage_sweep,
        ),
        Stage(
            "hndsweep",
            "Head-major pages",
            "ncu",
            "FlashInfer on NHD and HND vs Mjolnir on HND — is HND a kernel win or a layout win?",
            stage_hndsweep,
            requires_ncu=True,
        ),
        Stage(
            "kvfmt",
            "KV layouts and formats",
            "ncu",
            "fp8 / fp6 / fp4 / nvfp4 KV on NHD / interleaved / HND pages (ncu + wall clock)",
            stage_kvfmt,
            requires_ncu=True,
        ),
        Stage(
            "quant",
            "KV format accuracy",
            "harness",
            "relative L2 error of each KV format vs unquantized KV, at every L",
            stage_quant,
        ),
        Stage(
            "ring",
            "KV ring depth",
            "ncu",
            "TMA→MMA ring depth (--kv_stages) sweep at every L",
            stage_ring,
            requires_ncu=True,
        ),
        Stage(
            "prefetch",
            "L2 prefetch",
            "ncu",
            "TMA L2 prefetch distance (--l2_prefetch) sweep",
            stage_prefetch,
            requires_ncu=True,
        ),
        Stage(
            "pdl",
            "Programmatic dependent launch",
            "harness",
            "PDL on/off wall clock (decode overlaps the previous kernel; merge launches early)",
            stage_pdl,
        ),
        Stage(
            "ab",
            "Flag A/B",
            "ncu",
            "any harness flag sets, ncu + wall clock (default: evict-first x PDL)",
            stage_ab,
            requires_ncu=True,
        ),
        Stage(
            "fp6prof",
            "Full profiles per KV format",
            "profile",
            "full ncu profiles: L2 sectors vs requests vs time, warp stalls, per KV format",
            stage_fp6prof,
            requires_ncu=True,
        ),
        Stage(
            "timing",
            "Harness timing",
            "harness",
            "harness-timed medians per Q dtype; 1 vs 2 CTAs/SM where the family has that axis",
            stage_timing,
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
