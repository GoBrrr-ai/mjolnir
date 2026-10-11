"""Every README / docs chart, rendered from raw data only.

Inputs: the latest *default* run of each ``mjolnir verify`` stage
(``benchmarks/raw/verify/<stage>/*.json``) and the e2e bench history
(``benchmarks/history.jsonl``). Outputs: ``assets/benchmarks/<name>.png``
(light) + ``<name>-dark.png`` (dark) per chart, plus ``manifest.json`` —
the list of charts that had data, which the README/docs templates read to
decide what to embed. A chart whose stage has no run yet is simply skipped.

``mjolnir plot`` (and ``mjolnir verify ... --plot``) re-renders everything;
the interactive versions live in ``interactive.py`` (GitHub Pages).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

from mjolnir_cli import theme
from mjolnir_cli.config import RepoLayout
from mjolnir_cli.history import load_records
from mjolnir_cli.theme import Mode
from mjolnir_cli.verify import results
from mjolnir_cli.verify.stages import NOISE_PCT

PEAK_GBS = 273.0  # Jetson AGX Thor LPDDR5X theoretical peak
NOISE = NOISE_PCT / 100  # same-run ratio noise band (see verify.stages.NOISE_PCT)
DPI = 160


# ── data ────────────────────────────────────────────────────────────────────


def load_verify(layout: RepoLayout) -> dict[str, dict]:
    """The latest default run of every stage that has one."""
    from mjolnir_cli.verify.stages import STAGES

    out = {}
    for name in STAGES:
        doc = results.latest(layout.verify_dir, name)
        if doc:
            out[name] = doc
    return out


def L_label(L: int) -> str:
    return f"{L // 1024}K" if L >= 1024 and L % 1024 == 0 else str(L)


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib missing — install the plot extra: uv sync (or pip install 'mjolnir-cli[plot]')"
        ) from e
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "savefig.dpi": DPI})
    return plt


def _log2_axis(ax, Ls: list[int], mode: Mode) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ls)
    ax.set_xticklabels([L_label(L) for L in Ls])
    ax.minorticks_off()
    ax.set_xlabel("context length L (tokens)")
    ax.set_xlim(Ls[0] / 1.5, Ls[-1] * 1.5)


def _place_labels(ax, mode: Mode, items: list[tuple[float, float, str]], min_gap_pt: float = 15.0) -> None:
    """Direct labels at line endpoints, nudged apart vertically so they never
    overlap (min ``min_gap_pt`` points between baselines)."""
    if not items:
        return
    fig = ax.figure
    fig.canvas.draw()
    to_pt = 72.0 / fig.dpi
    disp = [(ax.transData.transform((x, y))[1] * to_pt, x, y, t) for x, y, t in items]
    disp.sort(key=lambda d: d[0])
    placed: list[float] = []
    for yp, *_ in disp:
        placed.append(max(yp, placed[-1] + min_gap_pt) if placed else yp)
    # re-center the stack on the data so nudges spread both ways
    shift = (sum(d[0] for d in disp) - sum(placed)) / len(placed)
    placed = [p + shift for p in placed]
    for (yp, x, y, t), py in zip(disp, placed):
        ax.annotate(
            t,
            (x, y),
            xytext=(10, py - yp),
            textcoords="offset points",
            color=mode.text,
            fontsize=10,
            va="center",
            ha="left",
            fontweight="bold",
        )


def _noise_band(ax, mode: Mode, center: float = 1.0) -> None:
    ax.axhspan(center * (1 - NOISE), center * (1 + NOISE), color=mode.grid, alpha=0.9, zorder=1, lw=0)


def _parity_line(ax, mode: Mode, label: str) -> None:
    ax.axhline(1.0, color=mode.flashinfer, lw=1.6, zorder=3)
    ax.text(
        0.006,
        1.0,
        f"{label} ",
        transform=ax.get_yaxis_transform(),
        color=mode.text_dim,
        fontsize=9.5,
        va="bottom",
        ha="left",
    )


def _footer_run(fig, mode: Mode, doc: dict, extra: str = "") -> None:
    k = doc.get("kernel", {})
    ver = f"kernel {k.get('version', '?')}" + (f" ({k['git']})" if k.get("git") else "")
    p = doc.get("params", {})
    shape = f"h_q={p.get('hq', 24)} h_k={p.get('hk', 4)} d={p.get('d', 256)} page={p.get('page', 128)}"
    theme.footer(fig, mode, f"mjolnir verify {doc['stage']} · run {doc['run_id']} · {ver} · {shape}{extra}")


# ── kernel charts (verify) ──────────────────────────────────────────────────


def chart_speedup(data: dict, mode: Mode):
    """Headline: Mjolnir's speedup over FlashInfer (and FA2's, for scale)."""
    doc = data.get("sweep")
    if not doc:
        return None
    by = results.series_by_id(doc)
    if "fi" not in by:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    theme.apply(fig, ax, mode)
    _noise_band(ax, mode)
    _parity_line(ax, mode, "FlashInfer = 1.0×")
    Ls = sorted(results.point_map(by["fi"]))
    ymax = 1.2
    labels: list[tuple[float, float, str]] = []
    for sid in ("mjolnir", "mjolnir-fp8", "fa2"):
        if sid not in by:
            continue
        sp = results.speedup(by["fi"], by[sid])
        xs = [L for L in Ls if L in sp]
        if not xs:
            continue
        ys = [sp[L] for L in xs]
        ymax = max(ymax, max(ys))
        ax.plot(xs, ys, **theme.line_kw(mode, sid))
        g = results.geomean(ys)
        name = {"mjolnir": "Mjolnir · e2m3 KV", "mjolnir-fp8": "Mjolnir · fp8 KV", "fa2": "FlashAttention 2"}[sid]
        labels.append((xs[-1], ys[-1], f"{name}  ({g:.2f}× geomean)"))
    _log2_axis(ax, Ls, mode)
    ax.set_ylim(0, max(2.0, ymax * 1.12))
    ax.set_ylabel("speedup over FlashInfer (×, higher is faster)")
    ax.set_xlim(Ls[0] / 1.5, Ls[-1] * 6)  # room for the direct labels
    _place_labels(ax, mode, labels)
    theme.title(
        ax,
        mode,
        "Mjolnir decode vs FlashInfer — every context length",
        "ncu kernel time (decode + split-KV merge), cold L2, base clocks · one sequence, 24 query / 4 KV heads, "
        f"d=256 · gray band = ±{NOISE_PCT:.0f}% run-to-run noise",
    )
    _footer_run(fig, mode, doc)
    return fig


def chart_latency(data: dict, mode: Mode):
    """Decode time per step vs L (log-log), every leg of the sweep."""
    doc = data.get("sweep")
    if not doc:
        return None
    by = results.series_by_id(doc)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    theme.apply(fig, ax, mode)
    Ls: list[int] = []
    names = {
        "fa2": "FlashAttention 2 · bf16 KV",
        "fi": "FlashInfer · fp8 KV",
        "mjolnir-fp8": "Mjolnir · fp8 KV",
        "mjolnir": "Mjolnir · e2m3 KV",
    }
    for sid in ("fa2", "fi", "mjolnir-fp8", "mjolnir"):
        if sid not in by:
            continue
        pm = results.point_map(by[sid])
        xs = sorted(pm)
        Ls = sorted(set(Ls) | set(xs))
        ax.plot(xs, [pm[x] for x in xs], label=names[sid], **theme.line_kw(mode, sid))
    if not Ls:
        plt.close(fig)
        return None
    _log2_axis(ax, Ls, mode)
    ax.set_yscale("log")
    _log_y_ticks(ax)
    ax.set_ylabel("decode time per step (µs, log)")
    theme.legend(ax, mode, loc="upper left")
    theme.title(
        ax,
        mode,
        "Decode latency per step",
        "one decode step over the whole KV cache · lower is faster · FA2 reads 2× the bytes (no fp8 KV path)",
    )
    _footer_run(fig, mode, doc)
    return fig


def _log_y_ticks(ax) -> None:
    import matplotlib.ticker as mticker

    lo, hi = ax.get_ylim()
    ticks = []
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10**e
            if lo <= v <= hi:
                ticks.append(v)
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:g} ms" if v >= 1000 else f"{v:g} µs"))


def chart_bandwidth(data: dict, mode: Mode):
    """Effective KV bandwidth (real bytes / time) vs the LPDDR5X peak."""
    doc = data.get("sweep")
    if not doc:
        return None
    by = results.series_by_id(doc)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    theme.apply(fig, ax, mode)
    ax.axhline(PEAK_GBS, color=mode.text_dim, lw=1.2, zorder=2)
    ax.text(
        0.005,
        PEAK_GBS,
        f" LPDDR5X peak {PEAK_GBS:.0f} GB/s",
        transform=ax.get_yaxis_transform(),
        color=mode.text_dim,
        fontsize=9.5,
        va="bottom",
    )
    Ls: list[int] = []
    names = {
        "fa2": "FlashAttention 2",
        "fi": "FlashInfer",
        "mjolnir-fp8": "Mjolnir · fp8 KV",
        "mjolnir": "Mjolnir · e2m3 KV",
    }
    for sid in ("fa2", "fi", "mjolnir-fp8", "mjolnir"):
        if sid not in by:
            continue
        pm = results.point_map(by[sid], "gbps")
        xs = sorted(pm)
        Ls = sorted(set(Ls) | set(xs))
        ax.plot(xs, [pm[x] for x in xs], label=names[sid], **theme.line_kw(mode, sid))
    if not Ls:
        plt.close(fig)
        return None
    _log2_axis(ax, Ls, mode)
    ax.set_ylim(0, PEAK_GBS * 1.12)
    ax.set_ylabel("effective KV bandwidth (GB/s)")
    theme.legend(ax, mode, loc="lower right")
    theme.title(
        ax,
        mode,
        "How close each kernel gets to the memory roofline",
        "each leg's own KV bytes (e2m3 = 6 bit, fp8 = 8 bit, bf16 = 16 bit) ÷ decode time · decode is "
        "bandwidth-bound once the cache leaves the 32 MiB L2",
    )
    _footer_run(fig, mode, doc)
    return fig


def chart_hnd(data: dict, mode: Mode):
    """Head-major pages: FlashInfer NHD vs HND vs Mjolnir on HND (speedup vs FI-NHD)."""
    doc = data.get("hndsweep")
    if not doc:
        return None
    by = results.series_by_id(doc)
    if "fi" not in by:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    theme.apply(fig, ax, mode)
    _noise_band(ax, mode)
    _parity_line(ax, mode, "FlashInfer NHD = 1.0×")
    Ls = sorted(results.point_map(by["fi"]))
    labels = {"fi-hnd": "FlashInfer on HND pages", "mjolnir-fp8": "Mjolnir (fp8 KV) on HND pages"}
    top = 1.2
    lab_items: list[tuple[float, float, str]] = []
    for sid in ("fi-hnd", "mjolnir-fp8"):
        if sid not in by:
            continue
        sp = results.speedup(by["fi"], by[sid])
        xs = [L for L in Ls if L in sp]
        ys = [sp[L] for L in xs]
        if not xs:
            continue
        top = max(top, max(ys))
        ax.plot(xs, ys, **theme.line_kw(mode, sid))
        lab_items.append((xs[-1], ys[-1], labels[sid]))
    _log2_axis(ax, Ls, mode)
    ax.set_xlim(Ls[0] / 1.5, Ls[-1] * 6)
    ax.set_ylim(0.6, max(1.5, top * 1.1))
    _place_labels(ax, mode, lab_items)
    ax.set_ylabel("speedup over FlashInfer on NHD (×)")
    theme.title(
        ax,
        mode,
        "Head-major KV pages: a layout win for both, a bigger one for Mjolnir",
        f"same data, NHD (pages, 2, PS, h_k, d) vs HND (pages, 2, h_k, PS, d) · gray band = ±{NOISE_PCT:.0f}% noise",
    )
    _footer_run(fig, mode, doc)
    return fig


def _variant_matrix(doc: dict, ref_id: str = "fi") -> tuple[list[str], list[int], list[list[float | None]]] | None:
    by = results.series_by_id(doc)
    if ref_id not in by:
        return None
    ref = results.point_map(by[ref_id])
    rows, labels = [], []
    Ls = sorted(ref)
    for s in doc.get("series", []):
        if s.get("kind") != "t5":
            continue
        pm = results.point_map(s)
        rows.append([ref[L] / pm[L] if L in pm and pm[L] else None for L in Ls])
        labels.append(s["label"])
    return (labels, Ls, rows) if rows else None


def _heatmap(doc: dict | None, mode: Mode, title: str, subtitle: str, ref_id: str = "fi"):
    if not doc:
        return None
    m = _variant_matrix(doc, ref_id)
    if not m:
        return None
    labels, Ls, rows = m
    plt = _plt()
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    vals = np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)
    finite = vals[np.isfinite(vals)]
    # saturate at 2x either way (values are printed in every cell), so one
    # outlier cannot wash out the rest of the matrix
    span = min(1.0, max(0.25, float(np.nanmax(np.abs(np.log2(finite)))) if finite.size else 0.25))
    norm = TwoSlopeNorm(vmin=2**-span, vcenter=1.0, vmax=2**span)
    h = 1.6 + 0.48 * len(labels)
    fig, ax = plt.subplots(figsize=(12.5, max(3.2, h)))
    theme.apply(fig, ax, mode)
    ax.grid(False)
    ax.imshow(vals, cmap=theme.diverging_cmap(mode), norm=norm, aspect="auto", interpolation="nearest")
    # 2px surface gaps between cells
    for i in range(len(labels) + 1):
        ax.axhline(i - 0.5, color=mode.panel, lw=2)
    for j in range(len(Ls) + 1):
        ax.axvline(j - 0.5, color=mode.panel, lw=2)
    best_per_col = [np.nanargmax(vals[:, j]) if np.isfinite(vals[:, j]).any() else -1 for j in range(len(Ls))]
    for i in range(len(labels)):
        for j in range(len(Ls)):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            ax.text(
                j,
                i,
                f"{v:.2f}×",
                ha="center",
                va="center",
                fontsize=9.5,
                color=mode.text,
                fontweight="bold" if best_per_col[j] == i else "normal",
            )
    ax.set_xticks(range(len(Ls)))
    ax.set_xticklabels([L_label(L) for L in Ls])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=mode.text)
    ax.set_xlabel("context length L (tokens)")
    ref_label = results.series_by_id(doc)[ref_id]["label"]
    theme.title(
        ax, mode, title, f"{subtitle} · cell = {ref_label} time ÷ variant time (>1 = faster) · bold = fastest at that L"
    )
    _footer_run(fig, mode, doc)
    return fig


def chart_kvfmt(data: dict, mode: Mode):
    return _heatmap(
        data.get("kvfmt"),
        mode,
        "KV layouts × formats",
        "fp8 / fp6 (e2m3, e3m2) / fp4 (e2m1) / nvfp4 on NHD, interleaved and HND pages",
    )


def chart_ring(data: dict, mode: Mode):
    return _heatmap(data.get("ring"), mode, "KV ring depth (TMA → MMA stages)", "--kv_stages at the harness default")


def chart_prefetch(data: dict, mode: Mode):
    return _heatmap(data.get("prefetch"), mode, "TMA L2 prefetch distance", "--l2_prefetch tiles ahead, fp8 KV")


def chart_ab(data: dict, mode: Mode):
    return _heatmap(data.get("ab"), mode, "Flag A/B", "harness flag sets (default: L2 evict-first × PDL)")


def chart_tradeoff(data: dict, mode: Mode):
    """Accuracy vs speed per KV format: the case for e2m3."""
    kv, q = data.get("kvfmt"), data.get("quant")
    if not kv or not q:
        return None
    m = _variant_matrix(kv)
    if not m:
        return None
    labels, Ls, rows = m
    speed = {}
    for lab, r in zip(labels, rows):
        g = results.geomean([v for v in r if v])
        if g:
            speed[lab] = g
    err: dict[str, list[float]] = {}
    for row in q.get("rows", []):
        if row.get("rel_l2") is not None:
            err.setdefault(row["format"], []).append(row["rel_l2"])
    pts = []
    for lab, g in speed.items():
        fmt = lab.split("-")[0]
        if fmt in err and lab.endswith("hnd"):
            pts.append((sum(err[fmt]) / len(err[fmt]), g, lab))
    if not pts:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    theme.apply(fig, ax, mode)
    _noise_band(ax, mode)
    _parity_line(ax, mode, "FlashInfer = 1.0×")
    for e, g, lab in pts:
        ax.plot([e], [g], color=mode.mjolnir, marker="o", ms=10, mec=mode.panel, mew=2, zorder=5, ls="none")
        ax.annotate(
            lab.replace("-hnd", ""),
            (e, g),
            xytext=(10, 6),
            textcoords="offset points",
            color=mode.text,
            fontsize=10.5,
            fontweight="bold",
        )
    ax.set_xscale("log")
    import matplotlib.ticker as mticker

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[-1] in "25" else ""))
    ax.set_xlabel("relative L2 error vs unquantized KV (log, lower is more accurate)")
    ax.set_ylabel("geomean speedup over FlashInfer (×)")
    ax.set_ylim(0, max(2.4, max(g for _, g, _ in pts) * 1.15))
    theme.title(
        ax,
        mode,
        "Speed vs accuracy per KV format (HND pages)",
        "x: `mjolnir verify quant` (mean over L) · y: `mjolnir verify kvfmt` geomean over L · up-and-left is better",
    )
    _footer_run(fig, mode, kv)
    return fig


def chart_quant(data: dict, mode: Mode):
    q = data.get("quant")
    if not q:
        return None
    by: dict[str, list[float]] = {}
    bits = {}
    for r in q.get("rows", []):
        if r.get("rel_l2") is not None:
            by.setdefault(r["format"], []).append(r["rel_l2"])
            bits[r["format"]] = r.get("bits")
    if not by:
        return None
    order = sorted(by, key=lambda f: (-(bits.get(f) or 0), f))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    theme.apply(fig, ax, mode)
    means = [sum(by[f]) / len(by[f]) for f in order]
    lo = [m - min(by[f]) for f, m in zip(order, means)]
    hi = [max(by[f]) - m for f, m in zip(order, means)]
    x = range(len(order))
    ax.bar(x, means, width=0.55, color=mode.mjolnir, zorder=3)
    ax.errorbar(x, means, yerr=[lo, hi], fmt="none", ecolor=mode.text_dim, elinewidth=1.4, capsize=5, zorder=4)
    for i, (f, mval) in enumerate(zip(order, means)):
        ax.text(
            i, mval + hi[i], f"{mval:.3f}", ha="center", va="bottom", color=mode.text, fontsize=10, fontweight="bold"
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{f}\n{bits.get(f):g} bit" if bits.get(f) else f for f in order], color=mode.text)
    ax.set_ylabel("relative L2 error (lower is better)")
    theme.title(
        ax,
        mode,
        "KV format accuracy",
        "kernel output vs attention over the unquantized float KV, same queries · bar = mean over L, "
        "whisker = min–max over L",
    )
    _footer_run(fig, mode, q)
    return fig


def chart_pdl(data: dict, mode: Mode):
    doc = data.get("pdl")
    if not doc:
        return None
    rows = [r for r in doc.get("rows", []) if r.get("delta_pct") is not None]
    if not rows:
        return None
    batches = sorted({r["batches"] for r in rows})
    plt = _plt()
    fig, axes = plt.subplots(1, len(batches), figsize=(12.5, 4.6), sharey=True, squeeze=False)
    axes = list(axes[0])
    theme.apply(fig, axes, mode)
    lim = max(10.0, max(abs(r["delta_pct"]) for r in rows) * 1.25)
    for ax, b in zip(axes, batches):
        rs = sorted((r for r in rows if r["batches"] == b), key=lambda r: r["L"])
        x = range(len(rs))
        ax.axhspan(-NOISE * 100, NOISE * 100, color=mode.grid, zorder=1, lw=0)
        ax.axhline(0, color=mode.text_dim, lw=1)
        ax.bar(
            x,
            [r["delta_pct"] for r in rs],
            width=0.6,
            zorder=3,
            color=[mode.mjolnir if r["delta_pct"] < 0 else mode.flashinfer for r in rs],
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels([L_label(r["L"]) for r in rs])
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("L (tokens)")
        ax.text(0.01, 0.97, f"batch {b}", transform=ax.transAxes, color=mode.text, fontweight="bold", va="top")
    axes[0].set_ylabel("wall-clock change with PDL (%)")
    theme.title(
        axes[0],
        mode,
        "Programmatic dependent launch",
        f"harness median, cold L2 · orange = PDL faster, blue = slower · gray band = ±{NOISE_PCT:.0f}% noise",
    )
    _footer_run(fig, mode, doc)
    return fig


def chart_fp6prof(data: dict, mode: Mode):
    """Bytes drop to the ideal ratio; time does not — the TMA unpack cost."""
    doc = data.get("fp6prof")
    if not doc:
        return None
    profs = [p for p in doc.get("profiles", []) if p.get("ok")]
    if not profs:
        return None
    by_L: dict[int, dict[str, dict]] = {}
    order: list[str] = []
    for p in profs:
        by_L.setdefault(p["L"], {})[p["label"]] = p
        if p["label"] not in order:
            order.append(p["label"])
    base = order[0]
    others = order[1:]
    if not others:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, len(others), figsize=(12.5, 4.8), sharey=True, squeeze=False)
    axes = list(axes[0])
    theme.apply(fig, axes, mode)
    Ls = sorted(by_L)
    for ax, lab in zip(axes, others):
        t_r, s_r, ideal = [], [], None
        for L in Ls:
            b, p = by_L[L].get(base), by_L[L].get(lab)
            if not b or not p:
                t_r.append(float("nan"))
                s_r.append(float("nan"))
                continue
            bm, pm = b["metrics"], p["metrics"]
            bs = bm.get("lts__t_sectors_op_read.sum") or bm.get("lts__t_sectors.sum")
            ps_ = pm.get("lts__t_sectors_op_read.sum") or pm.get("lts__t_sectors.sum")
            t_r.append(pm["gpu__time_duration.sum"] / bm["gpu__time_duration.sum"])
            s_r.append(ps_ / bs if bs else float("nan"))
            ideal = p["kv_bits"] / b["kv_bits"]
        x = list(range(len(Ls)))
        w = 0.36
        ax.bar([i - w / 2 - 0.01 for i in x], s_r, width=w, color=mode.text_dim, zorder=3, label="L2 bytes read")
        ax.bar([i + w / 2 + 0.01 for i in x], t_r, width=w, color=mode.mjolnir, zorder=3, label="kernel time")
        if ideal:
            ax.axhline(ideal, color=mode.text, lw=1.4, zorder=4)
            ax.text(
                -0.55,
                ideal + 0.02,
                f"ideal {ideal:.2f} (bits ratio)",
                color=mode.text,
                fontsize=9.5,
                ha="left",
                va="bottom",
                bbox=dict(facecolor=mode.panel, edgecolor="none", pad=1.5),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([L_label(L) for L in Ls])
        ax.set_ylim(0, 1.15)
        ax.set_xlabel(f"L (tokens) · {lab} vs {base}")
    axes[0].set_ylabel(f"ratio vs {base}")
    theme.legend(axes[-1], mode, loc="upper right")
    theme.title(
        axes[0],
        mode,
        "Bytes vs time per KV format",
        "full ncu profiles · L2 bytes fall to the ideal ratio; kernel time shows how much of it becomes speed",
    )
    _footer_run(fig, mode, doc)
    return fig


# ── e2e charts (history.jsonl, `mjolnir bench ab`) ──────────────────────────

# (series id, label): native vLLM backends come ONLY from the official image;
# Mjolnir only from a Mjolnir image (see mjolnir_cli.e2e).
E2E_SERIES = (
    ("NVFP4", "FlashInfer · official vLLM"),
    ("NVFP4_FA2", "FlashAttention 2 · official vLLM"),
    ("NVFP4_MJOLNIR", "Mjolnir"),
)


def e2e_rows(layout: RepoLayout) -> dict[str, dict]:
    """Latest history record per series: stock configs on the official vLLM
    image, the Mjolnir config on a Mjolnir image. Stock backends measured on
    the Mjolnir image are ignored (that image patches FlashInfer too)."""
    from mjolnir_cli import e2e

    latest: dict[str, dict] = {}
    for r in load_records(layout.history_file):
        cfg = r.get("config")
        if not r.get("cells"):
            continue
        ok = (cfg == "NVFP4_MJOLNIR" and e2e.is_mjolnir(r)) or (cfg in ("NVFP4", "NVFP4_FA2") and e2e.is_official(r))
        if ok and (cfg not in latest or r.get("epoch", 0) >= latest[cfg].get("epoch", 0)):
            latest[cfg] = r
    return latest


def _e2e_chart(layout: RepoLayout, mode: Mode, metric: str, ylabel: str, ttl: str, sub: str):
    from mjolnir_cli import e2e

    rows = e2e_rows(layout)
    if "NVFP4_MJOLNIR" not in rows or len(rows) < 2:
        return None
    plt = _plt()
    concs = [c for c in (1, 4) if any(cc.get("concurrency") == c for r in rows.values() for cc in r["cells"])]
    if not concs:
        return None
    fig, axes = plt.subplots(1, len(concs), figsize=(12.5, 4.8), sharey=True, squeeze=False)
    axes = list(axes[0])
    theme.apply(fig, axes, mode)
    order = [(c, lbl) for c, lbl in E2E_SERIES if c in rows]
    for ax, conc in zip(axes, concs):
        ctxs = sorted(
            {c.get("context") or 0 for r in rows.values() for c in r["cells"] if c.get("concurrency") == conc}
        )
        w = 0.8 / len(order)
        for i, (cfg, lbl) in enumerate(order):
            vals = []
            for ctx in ctxs:
                cell = next(
                    (c for c in rows[cfg]["cells"] if c.get("concurrency") == conc and (c.get("context") or 0) == ctx),
                    None,
                )
                vals.append(((cell or {}).get(metric) or {}).get("mean") or 0)
            xs = [j + (i - (len(order) - 1) / 2) * w for j in range(len(ctxs))]
            ax.bar(
                xs,
                vals,
                width=w * 0.92,
                color=theme.color(mode, cfg),
                zorder=3,
                label=lbl if ax is axes[0] else None,
            )
        ax.set_xticks(range(len(ctxs)))
        ax.set_xticklabels([f"{L_label(c)} ctx" if c else "no ctx" for c in ctxs])
        ax.set_xlabel(f"{conc} concurrent request{'s' if conc > 1 else ''}")
    axes[0].set_ylabel(ylabel)
    theme.legend(axes[0], mode, loc="upper right")
    theme.title(axes[0], mode, ttl, sub)
    imgs = " vs ".join(dict.fromkeys(str(rows[c].get("image", "")) for c, _ in order))
    theme.footer(
        fig,
        mode,
        f"mjolnir bench e2e · {rows['NVFP4_MJOLNIR'].get('model', '')} · "
        f"{e2e.spec_label(*(rows[c] for c, _ in order))} · {imgs} · llama-benchy, gated clean windows",
    )
    return fig


def chart_e2e_decode(layout: RepoLayout, mode: Mode):
    return _e2e_chart(
        layout,
        mode,
        "tg_tps",
        "decode tokens/s (higher is better)",
        "End-to-end decode throughput",
        "the official vLLM image (native backends) vs the Mjolnir image",
    )


def chart_e2e_prefill(layout: RepoLayout, mode: Mode):
    return _e2e_chart(
        layout,
        mode,
        "pp_tps",
        "prompt tokens/s (higher is better)",
        "End-to-end prompt processing",
        "the official vLLM image (native backends) vs the Mjolnir image",
    )


# ── the README opening infographic (official vLLM image vs Mjolnir image) ──

INFOGRAPHIC = "opening-infographic.png"


def _display_family() -> str:
    """A clean display face for the hero numbers: the first static-weight
    sans available (variable fonts render thin in matplotlib)."""
    from matplotlib import font_manager

    for fam in ("Inter Display", "Inter", "Noto Sans", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"):
        try:
            path = font_manager.findfont(fam, fallback_to_default=False)
        except Exception:  # noqa: BLE001
            continue
        if path and (fam == "DejaVu Sans" or "dejavu" not in Path(path).name.lower()):
            return fam
    return "DejaVu Sans"


def render_infographic(layout: RepoLayout) -> Path | None:
    """``assets/opening-infographic.png`` — the README hero.

    ``assets/background.jpeg`` (the Thor-as-Mjolnir art on the left) stays
    untouched; everything sits on the dark right half: the claim, the
    scenario, and two stat columns — decode and prefill — each with the gain
    as the big number and the two measured throughputs as bars. Every number
    comes from the latest history pair (:func:`mjolnir_cli.e2e.latest_pair`):
    native FlashInfer on the official vLLM image vs the Mjolnir backend on the
    Mjolnir image, one request at the longest context both measured."""
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, Rectangle

    from mjolnir_cli import e2e

    if layout.repo_root is None:
        return None
    off, mj = e2e.latest_pair(layout.history_file)
    h = e2e.hero(off, mj) if off and mj else None
    bg_path = layout.repo_root / "assets" / "background.jpeg"
    if not h or not bg_path.exists():
        print("[mjolnir] no official-vLLM / Mjolnir e2e pair in history — infographic skipped (run: mjolnir bench e2e)")
        return None

    plt = _plt()
    bg = plt.imread(str(bg_path))
    H, W = bg.shape[:2]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.imshow(bg, extent=(0, W, 0, H), zorder=0)
    # scrim only behind the text: transparent over the art, navy on the right
    x = np.linspace(0, 1, 512)
    alpha = np.clip((x - 0.40) / 0.25, 0, 1) ** 1.5 * 0.72
    scrim = np.zeros((1, 512, 4))
    scrim[..., :3] = np.array([3, 7, 16]) / 255
    scrim[..., 3] = alpha
    ax.imshow(scrim, extent=(0, W, 0, H), aspect="auto", zorder=1)

    disp = _display_family()
    white, dim, faint = "#ffffff", "#b4bfcc", "#7f8a97"
    c_off, c_mj = theme.DARK.flashinfer, theme.LIGHT.mjolnir
    ver = str(off.get("image", "")).split(":")[-1].split("-")[0].lstrip("v")
    model = str(mj.get("model", "")).split("/")[-1]
    quant = str(mj.get("config", "")).split("_")[0]

    def T(xf, yf, text, size, color=white, weight="normal", ha="left", **kw):
        ax.text(xf * W, yf * H, text, fontsize=size, color=color, weight=weight, ha=ha, family=disp, zorder=6, **kw)

    L, R = 0.555, 0.955
    T(L, 0.855, "JETSON AGX THOR   ·   END-TO-END SERVING", 10.5, dim, "semibold")
    title = ax.text(L * W, 0.745 * H, "Mjolnir", fontsize=40, color=white, weight="bold", family=disp, zorder=6)
    x_after = title.get_window_extent(renderer=fig.canvas.get_renderer()).x1  # pixels == data units here
    ax.text(x_after + 16, 0.745 * H, f"vs vLLM {ver}", fontsize=40, color=dim, family=disp, zorder=6)
    T(
        L,
        0.675,
        f"official vLLM image, native FlashInfer  ·  1 request  ·  {e2e.ctx_label(h['ctx'])} context  ·  {model} {quant}",
        11,
        dim,
    )
    ax.add_patch(Rectangle((L * W, 0.625 * H), (R - L) * W, 1.0, color=(1, 1, 1, 0.18), lw=0, zorder=5))

    cols = (("tg_tps", "decode", "{:.1f}", "tokens / s"), ("pp_tps", "prefill", "{:,.0f}", "prompt tokens / s"))
    gap = 0.04
    cw = (R - L - gap) / 2
    for i, (key, name, fmt, unit) in enumerate(cols):
        x0 = L + i * (cw + gap)
        va, vb, gain = h[key]
        T(x0, 0.415, e2e.pct(gain), 52, white, "bold")
        word = "faster" if gain >= 0 else "slower"
        T(x0, 0.345, f"{word} {name}", 15, white)
        T(x0 + cw, 0.345, unit, 11, faint, ha="right")
        X, BW = x0 * W, cw * W
        vmax = max(va, vb)
        for j, (lbl, v, col) in enumerate((("vLLM", va, c_off), ("Mjolnir", vb, c_mj))):
            y = (0.215 - j * 0.105) * H
            ax.text(X, y + 12, lbl, fontsize=10.5, color=dim, family=disp, zorder=6)
            ax.text(
                X + BW,
                y + 12,
                fmt.format(v),
                fontsize=12,
                color=white,
                weight="bold",
                ha="right",
                family=disp,
                zorder=6,
            )
            bh = 0.016 * H
            ax.add_patch(
                FancyBboxPatch(
                    (X, y - bh),
                    BW,
                    bh,
                    boxstyle=f"round,pad=0,rounding_size={bh / 2:.1f}",
                    facecolor=(1, 1, 1, 0.10),
                    edgecolor="none",
                    zorder=5,
                )
            )
            ax.add_patch(
                FancyBboxPatch(
                    (X, y - bh),
                    max(BW * v / vmax, bh),
                    bh,
                    boxstyle=f"round,pad=0,rounding_size={bh / 2:.1f}",
                    facecolor=col,
                    edgecolor="none",
                    zorder=6,
                )
            )
    T(
        L,
        0.035,
        f"{off.get('image', '')}  vs  {mj.get('image', '')}  ·  {e2e.spec_label(off, mj)}  ·  llama-benchy, gated clean windows",
        9,
        faint,
    )

    out = layout.repo_root / "assets" / INFOGRAPHIC
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"[mjolnir] wrote {out}")
    return out


# ── registry + render ───────────────────────────────────────────────────────

KERNEL_CHARTS: dict[str, tuple[str, Callable]] = {
    # name -> (stage it documents, builder)
    "kernel-speedup": ("sweep", chart_speedup),
    "kernel-latency": ("sweep", chart_latency),
    "kernel-bandwidth": ("sweep", chart_bandwidth),
    "hnd-layout": ("hndsweep", chart_hnd),
    "kvfmt-heatmap": ("kvfmt", chart_kvfmt),
    "kvfmt-tradeoff": ("kvfmt", chart_tradeoff),
    "quant-accuracy": ("quant", chart_quant),
    "ring-heatmap": ("ring", chart_ring),
    "prefetch-heatmap": ("prefetch", chart_prefetch),
    "ab-heatmap": ("ab", chart_ab),
    "pdl-delta": ("pdl", chart_pdl),
    "fp6prof-bytes-vs-time": ("fp6prof", chart_fp6prof),
}
E2E_CHARTS: dict[str, Callable] = {"e2e-decode": chart_e2e_decode, "e2e-prefill": chart_e2e_prefill}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.2, dpi=DPI)
    return path


def render_all(layout: RepoLayout, data: dict | None = None) -> dict[str, dict]:
    """Render every chart that has data, in both modes; write the manifest.
    Returns ``{chart: {"stage": ..., "light": path, "dark": path}}``."""
    plt = _plt()
    data = data if data is not None else load_verify(layout)
    out_dir = layout.charts_dir
    manifest: dict[str, dict] = {}
    jobs: list[tuple[str, str, Callable]] = [
        (n, st, (lambda f: lambda m: f(data, m))(fn)) for n, (st, fn) in KERNEL_CHARTS.items()
    ]
    jobs += [(n, "e2e", (lambda f: lambda m: f(layout, m))(fn)) for n, fn in E2E_CHARTS.items()]
    for name, stage, build in jobs:
        entry = {"stage": stage}
        for mode in theme.MODES:
            fig = build(mode)
            if fig is None:
                break
            p = _save(fig, out_dir / f"{name}{mode.suffix}.png")
            plt.close(fig)
            entry[mode.name] = str(p.relative_to(layout.repo_root) if layout.repo_root else p)
        if "light" in entry:
            manifest[name] = entry
    info = render_infographic(layout)
    # stale renders (data no longer qualifies) must not linger next to the manifest
    keep = {Path(v).name for e in manifest.values() for k, v in e.items() if k in ("light", "dark")}
    for png in out_dir.glob("*.png"):
        if png.name not in keep:
            png.unlink()
    if info is None and layout.repo_root is not None:
        (layout.repo_root / "assets" / INFOGRAPHIC).unlink(missing_ok=True)
    if info is not None:
        manifest["opening-infographic"] = {
            "stage": "e2e",
            "light": str(info.relative_to(layout.repo_root)),
            "dark": str(info.relative_to(layout.repo_root)),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"[mjolnir] {len(manifest)} charts (light + dark) -> {out_dir}")
    return manifest
