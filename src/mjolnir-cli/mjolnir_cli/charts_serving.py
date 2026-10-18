"""Serving-side charts: KV-cache capacity, spec drafts, accuracy, autoresearch."""

from __future__ import annotations

import re
from collections.abc import Callable
from itertools import groupby

from mjolnir_cli import theme
from mjolnir_cli.config import RepoLayout
from mjolnir_cli.theme import Mode

from mjolnir_cli.charts_base import (
    L_label,
    _fmt_x,
    _int_locator,
    _plt,
)


def _kvcache_series(r: dict) -> str:
    if r.get("side") == "official":
        return "NVFP4"
    if r.get("config") in theme.SERIES_HATCH:
        return r["config"]
    return "mjolnir" if str(r.get("kv_dtype", "")).startswith("fp6") else "mjolnir-fp8"


def chart_kvcache(data: dict, mode: Mode):
    """KV capacity per config: tokens (left) and the concurrent e2e-shaped
    requests (context + 2048 prompt + 128 generated) they fit (right)."""
    from mjolnir_cli.readme_tables import kvcache_label

    doc = data.get("kv-cache-size")
    if not doc:
        return None
    rows = [r for r in doc.get("rows", []) if r.get("kv_tokens")]
    if not rows:
        return None
    pp, tg = doc.get("params", {}).get("e2e_pp", 2048), doc.get("params", {}).get("e2e_tg", 128)
    plt = _plt()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    theme.apply(fig, [ax0, ax1], mode)
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows, strict=True):
        sid = _kvcache_series(r)
        c = theme.color(mode, sid)
        hollow = not theme.filled(sid)
        ax0.barh(
            y,
            r["kv_tokens"] / 1e6,
            height=0.6,
            color=mode.panel if hollow else c,
            edgecolor=c if hollow or not theme.hatch(sid) else mode.panel,
            linewidth=2 if hollow else 0,
            hatch=theme.hatch(sid),
            zorder=3,
        )
        ax0.text(
            r["kv_tokens"] / 1e6,
            y,
            f"  {r['kv_tokens'] / 1e6:.2f} M",
            va="center",
            ha="left",
            color=mode.text,
            fontsize=10,
            fontweight="bold",
        )
    ax0.set_yticks(ys)
    ax0.set_yticklabels([theme.wrap(kvcache_label(r), 22) for r in rows], fontsize=9)
    ax0.set_xlabel("KV cache, million tokens")
    ax0.set_xlim(0, max(r["kv_tokens"] for r in rows) / 1e6 * 1.32)
    ctxs = [0, 4096, 8192, 16384, 32768, 65536, 131072]
    for r in rows:
        sid = _kvcache_series(r)
        fit = [r["kv_tokens"] / (c + pp + tg) for c in ctxs]
        ax1.plot(range(len(ctxs)), fit, label=theme.wrap(kvcache_label(r), 30), **theme.line_kw(mode, sid))
    ax1.axhspan(0, 4, color=mode.neutral, alpha=0.6, zorder=1, lw=0)
    ax1.text(len(ctxs) - 1, 4.4, "e2e bench: 1-4 concurrent requests", color=mode.text_dim, fontsize=9, ha="right")
    ax1.set_yscale("log", base=2)
    ax1.set_ylim(2, max(r["kv_tokens"] for r in rows) / (pp + tg) * 2.2)
    ax1.set_xticks(range(len(ctxs)))
    ax1.set_xticklabels([f"{L_label(c)}" if c else "0" for c in ctxs])
    ax1.set_xlabel(f"context per request (+ {pp:,} prompt + {tg} generated tokens)")
    ax1.set_ylabel("concurrent requests that fit")
    from matplotlib.ticker import FuncFormatter

    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    theme.legend(ax1, mode, loc="upper right")
    theme.title(
        ax0,
        mode,
        "KV cache capacity",
        "vLLM's own KV sizing at startup, same model and memory budget · request capacity ignores max-num-seqs",
    )
    models = ", ".join(dict.fromkeys(str(r.get("model", "")) for r in rows))
    theme.footer(fig, mode, f"mjolnir verify kv-cache-size · run {doc['run_id']} · {models}")
    return fig


def chart_spec(data: dict, mode: Mode):
    """Decode t/s per request vs draft count (one panel per concurrency) and
    per-position acceptance at the largest draft count."""
    from mjolnir_cli.readme_tables import spec_label

    doc = data.get("spec")
    if not doc:
        return None
    rows = [r for r in doc.get("rows", []) if r.get("decode_tps")]
    if not rows:
        return None
    concs = sorted({r["concurrency"] for r in rows})
    cfgs = list(dict.fromkeys(r["config"] for r in rows))
    plt = _plt()
    fig, axes = plt.subplots(
        1, len(concs) + 1, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1] * len(concs) + [1.1]}
    )
    axes = list(axes)
    theme.apply(fig, axes, mode)
    for ax, conc in zip(axes, concs, strict=False):  # axes has one extra panel (acceptance) past the concs
        for cfg in cfgs:
            pts = sorted((r["k"], r["decode_tps"]) for r in rows if r["config"] == cfg and r["concurrency"] == conc)
            if not pts:
                continue
            ax.plot(
                [k for k, _ in pts],
                [v for _, v in pts],
                label=spec_label(cfg).replace(" + ", "\n+ ", 1),
                **theme.line_kw(mode, cfg),
            )
            kb, vb = max(pts, key=lambda p: p[1])
            ax.annotate("★", (kb, vb), textcoords="offset points", xytext=(0, 9), ha="center", color=mode.text)
        ks = sorted({r["k"] for r in rows})
        ax.set_xticks(ks)
        ax.set_xlabel(f"draft tokens · {conc} request{'s' if conc > 1 else ''} in flight")
    axes[0].set_ylabel("decode tokens/s per request")

    # per-position acceptance at the largest k, first concurrency
    axp = axes[-1]
    kmax = max(r["k"] for r in rows)
    w = 0.8 / max(1, len(cfgs))
    for i, cfg in enumerate(cfgs):
        r = next((r for r in rows if r["config"] == cfg and r["k"] == kmax and r["concurrency"] == concs[0]), None)
        pp = (r or {}).get("per_pos") or []
        if not pp:
            continue
        xs = [j + (i - (len(cfgs) - 1) / 2) * w for j in range(len(pp))]
        hollow = not theme.filled(cfg)
        c = theme.color(mode, cfg)
        axp.bar(
            xs,
            pp,
            width=w * 0.92,
            color=mode.panel if hollow else c,
            edgecolor=c if hollow or not theme.hatch(cfg) else mode.panel,
            linewidth=1.6 if hollow else 0,
            hatch=theme.hatch(cfg),
            zorder=3,
        )
    axp.set_xticks(range(kmax))
    axp.set_xticklabels([f"#{j + 1}" for j in range(kmax)])
    axp.set_ylim(0, 1.5)  # headroom for the legend
    axp.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axp.set_xlabel(f"draft position · {kmax} drafts, {concs[0]} request")
    axp.set_ylabel("accepted (share of steps)")
    # the legend sits over the short late-position bars, clear of every line
    h, lbl = axes[0].get_legend_handles_labels()
    theme.legend(axp, mode, handles=h, labels=lbl, loc="upper right", fontsize=8.5)
    theme.title(
        axes[0],
        mode,
        "Speculative decoding: draft count",
        "same prompts at temperature 0 for every point · ★ = best draft count · decode = tokens after the first "
        "÷ time to the last",
    )
    models = ", ".join(dict.fromkeys(str(r.get("image", "")) for r in rows))
    theme.footer(fig, mode, f"mjolnir verify spec · run {doc['run_id']} · {models}")
    return fig


def _accuracy_series(r: dict) -> str:
    """Theme series for an accuracy row: official FlashInfer blue, FlashInfer
    on the Mjolnir image hollow blue, Mjolnir fp8 KV hollow orange, the fp6
    Mjolnir configs (and their experiments) filled orange."""
    from mjolnir_cli import e2e

    cfg = str(r.get("config", ""))
    if "MJOLNIR" not in cfg:
        return "NVFP4" if not e2e.is_mjolnir({"image": r.get("image", "")}) else "fi-hnd"
    if cfg in theme.SERIES_STYLE:
        return cfg
    return "NVFP4_MJOLNIR_E2M3"


def _accuracy_rows(layout: RepoLayout) -> tuple[list[dict], list[dict | None], list[str]]:
    from mjolnir_cli import accuracy

    rows = accuracy.latest_legs(accuracy.load_runs(layout))
    return rows, accuracy.baselines(rows), accuracy.suites_in(rows)


def _accuracy_footer(fig, mode: Mode, rows: list[dict]) -> None:
    from mjolnir_cli import accuracy

    runs = sorted({v["run_id"] for r in rows for v in r["runs"].values()})
    theme.footer(
        fig,
        mode,
        f"mjolnir bench accuracy · tool-eval-bench {accuracy.TOOL_COMMIT[:7]} · run{'s' if len(runs) > 1 else ''} "
        f"{', '.join(runs)}",
    )


def chart_accuracy_suites(layout: RepoLayout, mode: Mode):
    """One panel per suite: each config's score as a dot, the baseline as a
    dashed line, the difference printed next to the dot."""
    from mjolnir_cli import accuracy

    rows, bases, names = _accuracy_rows(layout)
    if not rows:
        return None
    plt = _plt()
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(max(7.5, 3.4 * n + 3.2), 1.9 + 0.55 * len(rows)), sharey=True)
    axes = list(axes) if n > 1 else [axes]
    theme.apply(fig, axes, mode)
    ys = list(range(len(rows)))[::-1]
    for ax, name in zip(axes, names, strict=True):
        # each row's baseline score: its own model's baseline row
        bs = [(base["suites"].get(name) or {}).get("score") if base else None for base in bases]
        vals = [(y, r, (r["suites"].get(name) or {}).get("score"), b) for y, r, b in zip(ys, rows, bs, strict=True)]
        got = [v for _, _, v, _ in vals if v is not None]
        if not got:
            ax.set_visible(False)
            continue
        refs = [b for b in bs if b is not None]
        lo, hi = min(got + refs), max(got + refs)
        pad = max(4.0, (hi - lo) * 0.35)
        ax.set_xlim(max(0, lo - pad), min(100, hi + pad) + pad * 0.6)
        # one dashed mark per model, spanning its rows (latest_legs keeps them together)
        for _, group in groupby(zip(ys, bs, bases, strict=True), key=lambda t: id(t[2])):
            span = list(group)
            if span[0][1] is not None:
                ax.vlines(
                    span[0][1],
                    span[-1][0] - 0.4,
                    span[0][0] + 0.4,
                    color=mode.text_dim,
                    lw=1.2,
                    ls=(0, (4, 3)),
                    zorder=2,
                )
        for (y, r, v, b), base in zip(vals, bases, strict=True):
            if v is None:
                continue
            sid = _accuracy_series(r)
            kw = theme.line_kw(mode, sid)
            if b is not None and r is not base:
                ax.plot([b, v], [y, y], color=kw["color"], lw=2, zorder=3, solid_capstyle="round")
            ax.plot(
                [v], [y], ls="none", **{k: kw[k] for k in ("color", "marker", "mfc", "mec", "mew")}, ms=10, zorder=5
            )
            txt = f"{v:.1f}{accuracy.fmt_delta(accuracy.delta(r, base, name))}"
            ax.annotate(
                txt,
                (v if b is None else max(v, b), y),
                textcoords="offset points",
                xytext=(9, 0),
                va="center",
                fontsize=9.5,
                color=mode.text,
            )
        ax.set_xlabel(f"{accuracy.suite_label(name)} score, %")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([theme.wrap(r["label"], 24) for r in rows], color=mode.text, fontsize=10)
    axes[0].set_ylim(-0.7, len(rows) - 0.3)
    theme.title(
        axes[0],
        mode,
        "Model quality per serving config",
        "tool-eval-bench over each config's OpenAI endpoint · dashed = "
        + (accuracy.baseline_caption(bases, ", ") or "baseline")
        + " · (±) = points vs it"
        + (f" · {accuracy.EXCLUDED_NOTE_SHORT}" if accuracy.any_excluded(rows, bases, names) else ""),
    )
    _accuracy_footer(fig, mode, rows)
    return fig


ACC_PAR_POINTS = 5.0


ACC_RED_POINTS = 30.0


def chart_accuracy_categories(layout: RepoLayout, mode: Mode):
    """Tool-call categories × configs: the share of points per cell, colored
    by how far the config falls behind the baseline — green on par (or
    better), then orange, then red the more points it loses."""
    from mjolnir_cli import accuracy

    rows, bases, names = _accuracy_rows(layout)
    pairs = [(r, b) for r, b in zip(rows, bases, strict=True) if r.get("categories")]
    if not pairs:
        return None
    rows, bases = (list(t) for t in zip(*pairs, strict=True))
    import numpy as np
    from matplotlib.colors import Normalize

    cats = sorted({c for r in rows for c in r["categories"]})
    pct = np.array([[r["categories"].get(c, {}).get("percent", np.nan) for c in cats] for r in rows], dtype=float)
    ref = np.array(  # per row: its own model's baseline
        [
            [b["categories"].get(c, {}).get("percent", np.nan) for c in cats]
            if b and b.get("categories")
            else [np.nan] * len(cats)
            for b in bases
        ],
        dtype=float,
    )
    delta = pct - ref
    loss = np.where(np.isfinite(delta), np.clip(-delta, 0, None), 0.0)  # points behind the baseline; ahead = 0
    span = max(ACC_RED_POINTS, float(loss.max()) if loss.size else ACC_RED_POINTS)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(9.0, 0.78 * len(cats) + 4.5), 1.8 + 0.75 * len(rows)))
    theme.apply(fig, ax, mode)
    ax.grid(False)
    ax.imshow(
        loss,
        cmap=theme.status_cmap(mode, ACC_PAR_POINTS, span),
        norm=Normalize(vmin=0.0, vmax=span),
        aspect="auto",
        interpolation="nearest",
    )
    for i in range(len(rows) + 1):
        ax.axhline(i - 0.5, color=mode.panel, lw=2)
    for j in range(len(cats) + 1):
        ax.axvline(j - 0.5, color=mode.panel, lw=2)
    for i in range(len(rows)):
        for j in range(len(cats)):
            if np.isfinite(pct[i, j]):
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=9.5, color=mode.text)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, color=mode.text)
    ax.set_yticks(range(len(rows)))
    marked = [accuracy.leg_excluded(r, b, names) for r, b in zip(rows, bases, strict=True)]
    ax.set_yticklabels(
        [theme.wrap(r["label"] + (" ‡" if m else ""), 24) for r, m in zip(rows, marked, strict=True)], color=mode.text
    )
    ax.set_xlabel("tool-call category (names: the table below)")
    theme.title(
        ax,
        mode,
        "Tool calls by category",
        "cell = share of points, % · green = on par with "
        + (accuracy.baseline_caption(bases, ", ") or "the first row")
        + f" (at most {ACC_PAR_POINTS:g} points behind, or ahead) · orange → red = further behind"
        + (f" · {accuracy.CATEGORY_EXCLUDED_NOTE}" if any(marked) else ""),
    )
    _accuracy_footer(fig, mode, rows)
    return fig


def chart_autoresearch(rows: list[dict], mode: Mode):
    """Top: the score of every experiment, the kept ones labelled, the running
    best as a staircase. Bottom: the per-L split — each experiment's
    FlashInfer÷Mjolnir ratio at every L over the baseline's — so a gain that
    lives at one L (or comes from a FlashInfer outlier) is visible at a glance."""
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    plt = _plt()
    if not rows:  # before the first exported run: say so instead of a broken image
        fig, ax = plt.subplots(figsize=(13.5, 0.4))
        theme.apply(fig, ax, mode)
        ax.set_axis_off()
        theme.title(
            ax,
            mode,
            "Autoresearch progress: no run exported yet",
            "mjolnir autoresearch export → research/runs/<tag>/, then mjolnir plot --readme",
        )
        return fig
    base = next((r for r in rows if r.get("status") == "baseline" and r.get("per_L")), None)
    Ls = sorted({int(L) for r in rows for L in (r.get("per_L") or {})})
    if Ls:
        fig, (ax, hx) = plt.subplots(
            2, 1, figsize=(13.5, 8.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.35], "hspace": 0.08}
        )
    else:  # rows without per-L numbers (an imported ledger): the score panel only
        fig, ax = plt.subplots(figsize=(13.5, 5.6))
        hx = None
    theme.apply(fig, [a for a in (ax, hx) if a is not None], mode)
    ids = [r["id"] for r in rows]
    scored = [r for r in rows if r.get("score")]
    kept = [r for r in scored if r.get("status") in ("keep", "baseline")]
    rest = [r for r in scored if r.get("status") not in ("keep", "baseline")]
    lo = min((r["score"] for r in scored), default=1.0)
    hi = max((r["score"] for r in scored), default=1.0)
    pad = max(0.02, (hi - lo) * 0.12)
    floor = lo - pad
    ax.scatter(
        [r["id"] for r in rest if r["status"] == "discard"],
        [r["score"] for r in rest if r["status"] == "discard"],
        s=22,
        color=mode.text_dim,
        alpha=0.45,
        zorder=3,
        label="discarded",
        linewidths=0,
    )
    noisy = [r for r in rest if r["status"] == "noisy"]
    if noisy:
        ax.scatter(
            [r["id"] for r in noisy],
            [r["score"] for r in noisy],
            s=30,
            facecolor=mode.panel,
            edgecolor=mode.text_dim,
            linewidths=1.2,
            zorder=3,
            label="noisy (re-evaluated)",
        )
    crashes = [r["id"] for r in rows if r.get("status") == "crash" or not r.get("score")]
    if crashes:
        ax.scatter(
            crashes,
            [floor + pad * 0.25] * len(crashes),
            marker="x",
            s=34,
            color=mode.text_dim,
            zorder=3,
            label="crashed",
        )
    if kept:
        xs, ys, best = [], [], None
        for r in rows:  # the running best, as a staircase across every experiment
            if r in kept and (best is None or r["score"] > best):
                best = r["score"]
            if best is not None:
                xs.append(r["id"])
                ys.append(best)
        ax.step(xs, ys, where="post", color=mode.mjolnir, lw=2.0, zorder=4, label="running best")
        ax.scatter(
            [r["id"] for r in kept],
            [r["score"] for r in kept],
            s=64,
            color=mode.mjolnir,
            edgecolor=mode.panel,
            linewidths=1.5,
            zorder=5,
            label="kept",
        )
        for r in kept:
            h = r.get("hypothesis", "")
            ax.annotate(
                h if len(h) <= 48 else h[:47] + "…",
                (r["id"], r["score"]),
                xytext=(6, 6),
                textcoords="offset points",
                rotation=22,
                fontsize=8.5,
                color=mode.text,
                zorder=6,
            )
    ax.set_ylim(floor, hi + pad * 2.2)
    ax.set_ylabel("FlashInfer ÷ Mjolnir\n(geomean over L, higher is better)")
    ax.yaxis.set_major_formatter(_fmt_x())
    theme.legend(ax, mode, loc="lower right", fontsize=9)
    n_keep = sum(r.get("status") == "keep" for r in rows)
    theme.title(
        ax,
        mode,
        f"Autoresearch progress: {len(rows)} experiments, {n_keep} kept",
        "score = FlashInfer time ÷ Mjolnir time (fp6 e2m3 KV), both legs in the same ncu run"
        + (" · below: each L against the baseline" if Ls else ""),
    )
    if hx is not None and not base:
        hx.set_visible(False)
    if hx is not None and base:
        vals = np.full((len(Ls), len(ids)), np.nan)
        for j, r in enumerate(rows):
            for i, L in enumerate(Ls):
                v, b = (r.get("per_L") or {}).get(str(L)), base["per_L"].get(str(L))
                if v and b:
                    vals[i, j] = v / b
        finite = vals[np.isfinite(vals)]
        span = min(0.5, max(0.1, float(np.nanmax(np.abs(np.log2(finite)))) if finite.size else 0.1))
        norm = TwoSlopeNorm(vmin=2**-span, vcenter=1.0, vmax=2**span)
        hx.grid(False)
        im = hx.imshow(
            vals,
            cmap=theme.diverging_cmap(mode),
            norm=norm,
            aspect="auto",
            interpolation="nearest",
            extent=(min(ids) - 0.5, max(ids) + 0.5, len(Ls) - 0.5, -0.5),
        )
        for j in ids:  # 2px surface gaps between experiments and between L
            hx.axvline(j - 0.5, color=mode.panel, lw=2)
        for i in range(len(Ls) + 1):
            hx.axhline(i - 0.5, color=mode.panel, lw=2)
        for r in rows:  # L set aside: the FlashInfer reference jumped there
            m = re.search(r"reference jumped: ([^)]*)\)", r.get("reason") or "")
            for lab in m.group(1).split(", ") if m else []:
                if lab in [L_label(L) for L in Ls]:
                    hx.text(
                        r["id"],
                        [L_label(L) for L in Ls].index(lab),
                        "ref?",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=mode.text,
                    )
        hx.set_yticks(range(len(Ls)))
        hx.set_yticklabels([L_label(L) for L in Ls], fontsize=8.5)
        hx.set_ylabel("L, vs baseline")
        cb = fig.colorbar(im, ax=[ax, hx], fraction=0.015, pad=0.01, aspect=12, anchor=(0.0, 0.0), shrink=0.3)
        cb.ax.tick_params(labelsize=8, colors=mode.text_dim)
        cb.outline.set_visible(False)
        cb.ax.yaxis.set_major_formatter(_fmt_x())
    bottom = hx if hx is not None and hx.get_visible() else ax
    bottom.set_xlabel("experiment #")
    bottom.xaxis.set_major_locator(_int_locator())
    ax.set_xlim(min(ids) - 0.8, max(ids) + 0.8)
    theme.footer(
        fig,
        mode,
        f"mjolnir autoresearch · {rows[-1].get('branch', '')}"
        + (" · ref? = FlashInfer reference jumped at that L: set aside by the judge" if hx is not None else ""),
        below_in=0.75 if hx is not None else 0.62,
    )
    return fig


ACCURACY_CHARTS: dict[str, Callable] = {
    "accuracy-suites": chart_accuracy_suites,
    "accuracy-categories": chart_accuracy_categories,
}
