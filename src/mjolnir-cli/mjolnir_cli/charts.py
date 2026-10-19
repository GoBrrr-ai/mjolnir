"""Kernel + experiment charts (verify raws); helpers in charts_base,
e2e/hero/infographic in charts_e2e, serving charts in charts_serving."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from collections.abc import Callable

from mjolnir_cli import charts_base, charts_e2e, charts_serving
from mjolnir_cli import theme
from mjolnir_cli.config import RepoLayout, model_short
from mjolnir_cli.e2e import latest_pair as e2e_latest_pair
from mjolnir_cli.theme import Mode
from mjolnir_cli.charts_e2e import E2E_CHARTS
from mjolnir_cli.charts_serving import ACCURACY_CHARTS
from mjolnir_cli.verify import results
from mjolnir_cli.verify.stages import NOISE_PCT

from mjolnir_cli.charts_base import (
    L_label,
    NOISE,
    PEAK_GBS,
    _footer_run,
    _heatmap,
    _log2_axis,
    _log_y_ticks,
    _noise_band,
    _parity_line,
    _place_labels,
    _plt,
    _variant_matrix,
)


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
        name = {"mjolnir": "Mjolnir · e2m3 KV", "mjolnir-fp8": "Mjolnir · fp8 KV", "fa2": "FlashAttention 2"}[sid]
        labels.append((xs[-1], ys[-1], name))  # the geomeans are in the README table, not here
    _log2_axis(ax, Ls, mode)
    ax.set_ylim(0, max(2.0, ymax * 1.12))
    ax.set_ylabel("speedup over FlashInfer (×, higher is faster)")
    ax.set_xlim(Ls[0] / 1.5, Ls[-1] * 3.6)  # room for the direct labels
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
    labels, _Ls, rows = m
    speed = {}
    for lab, r in zip(labels, rows, strict=True):
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
    lo = [m - min(by[f]) for f, m in zip(order, means, strict=True)]
    hi = [max(by[f]) - m for f, m in zip(order, means, strict=True)]
    x = range(len(order))
    ax.bar(x, means, width=0.55, color=mode.mjolnir, zorder=3)
    ax.errorbar(x, means, yerr=[lo, hi], fmt="none", ecolor=mode.text_dim, elinewidth=1.4, capsize=5, zorder=4)
    for i, (_f, mval) in enumerate(zip(order, means, strict=True)):
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
    for ax, b in zip(axes, batches, strict=True):
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
    for ax, lab in zip(axes, others, strict=True):
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
                bbox={"facecolor": mode.panel, "edgecolor": "none", "pad": 1.5},
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
    "kv-cache-capacity": ("kv-cache-size", charts_serving.chart_kvcache),
    "spec-drafts": ("spec", charts_serving.chart_spec),
}


# re-exports: consumers import everything from mjolnir_cli.charts
chart_kvcache = charts_serving.chart_kvcache
chart_spec = charts_serving.chart_spec
chart_autoresearch = charts_serving.chart_autoresearch
load_verify = charts_base.load_verify
_save = charts_base._save
_logotype_box = charts_base._logotype_box
multiple = charts_base.multiple
HERO_BACKGROUND = charts_base.HERO_BACKGROUND
INFOGRAPHIC = charts_base.INFOGRAPHIC
e2e_rows = charts_e2e.e2e_rows
render_hero = charts_e2e.render_hero
render_infographic = charts_e2e.render_infographic

# re-exports: consumers import everything from mjolnir_cli.charts
chart_kvcache = charts_serving.chart_kvcache
chart_spec = charts_serving.chart_spec
chart_autoresearch = charts_serving.chart_autoresearch
load_verify = charts_base.load_verify
_save = charts_base._save
_logotype_box = charts_base._logotype_box
multiple = charts_base.multiple
HERO_BACKGROUND = charts_base.HERO_BACKGROUND
INFOGRAPHIC = charts_base.INFOGRAPHIC
e2e_rows = charts_e2e.e2e_rows
render_hero = charts_e2e.render_hero
render_infographic = charts_e2e.render_infographic


def render_all(layout: RepoLayout) -> dict[str, dict]:
    """Render every chart that has data, dark-only; write the manifest.

    Manifest v2 (spec #8): a chart belongs to a (platform, model) pair — the
    verify-stage charts render once per measured pair into
    ``assets/benchmarks/<platform>/<model-short>/`` and their entries carry
    the ``platform``/``model`` identity; the e2e charts render per pair from
    that pair's history. The default pair renders last and owns the flat
    chart id (the root README and the thor case study quote it); a second
    pair's PNGs are written for the site but its entry yields the flat id.
    The accuracy charts span every measured model at once, so they stay flat
    with a null identity; ``hero``/``opening-infographic`` are root-page
    one-offs. Dark-only (user decision): GitHub renders the ``-dark.png``
    images (``chart()`` emits a plain ``<img>``), the Pages site is
    interactive Plotly — so only ``theme.DARK`` is rendered and entries carry
    a single ``dark`` path. Readers (``readme._env``'s ``chart()``) resolve an
    identity entry only for the matching pair and keep resolving legacy flat
    paths. Returns ``{chart: {"stage", "platform", "model", "dark"}}``."""
    plt = _plt()
    from mjolnir_cli import readme_targets

    out_dir = layout.charts_dir
    manifest: dict[str, dict] = {}

    def save(entry: dict, build: Callable, out: Path) -> bool:
        fig = build(theme.DARK)
        if fig is None:
            return False
        p = _save(fig, out / f"{entry['name']}{theme.DARK.suffix}.png")
        plt.close(fig)
        entry["dark"] = str(p.relative_to(layout.repo_root) if layout.repo_root else p)
        return True

    for platform, model in readme_targets.measured_pairs(layout):
        docs = readme_targets.pair_docs(layout, platform, model)
        if not docs:
            continue
        pair_dir = out_dir / platform / model_short({"model": model})
        for name, (stage, build) in KERNEL_CHARTS.items():
            if stage not in docs:
                continue
            entry: dict = {"name": name, "stage": stage, "platform": platform, "model": model}
            if save(entry, partial(build, docs), pair_dir):
                manifest[name] = {k: v for k, v in entry.items() if k != "name"}
        off, mj = e2e_latest_pair(layout.history_file, model=model, platform=platform)
        if off and mj:
            for name, build in E2E_CHARTS.items():
                entry = {"name": name, "stage": "e2e", "platform": platform, "model": model}
                if save(entry, partial(build, layout, platform=platform, model=model), pair_dir):
                    manifest[name] = {k: v for k, v in entry.items() if k != "name"}
    for name, build in ACCURACY_CHARTS.items():
        entry = {"name": name, "stage": "accuracy", "platform": None, "model": None}
        if save(entry, partial(build, layout), out_dir):
            manifest[name] = {k: v for k, v in entry.items() if k != "name"}
    info = render_infographic(layout)
    hero = render_hero(layout)
    # stale renders (data no longer qualifies) must not linger next to the manifest
    keep = {
        (layout.repo_root / e["dark"]).resolve() if layout.repo_root else Path(e["dark"])
        for e in manifest.values()
        if e.get("dark")
    }
    for png in out_dir.rglob("*.png"):
        if png.resolve() not in keep:
            png.unlink()
    if info is None and layout.repo_root is not None:
        (layout.repo_root / "assets" / INFOGRAPHIC).unlink(missing_ok=True)
    if hero is not None:
        manifest["hero"] = {
            "stage": "readme",
            "dark": str(hero.relative_to(layout.repo_root) if layout.repo_root else hero),
        }
    if info is not None:
        manifest["opening-infographic"] = {
            "stage": "e2e",
            "dark": str(info.relative_to(layout.repo_root) if layout.repo_root else info),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"[mjolnir] {len(manifest)} charts (dark) -> {out_dir}")
    return manifest
