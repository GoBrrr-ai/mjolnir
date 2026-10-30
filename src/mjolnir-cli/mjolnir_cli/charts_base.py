"""Shared chart helpers: data loading, axes, styling, fonts, save.

Split out of charts.py (file-length ratchet); the charts_* modules import
their helpers from here and charts.py re-exports the public names.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjolnir_cli import theme
from mjolnir_cli.config import DEFAULT_MODEL, RepoLayout, model_of
from mjolnir_cli.theme import Mode
from mjolnir_cli.verify import results
from mjolnir_cli.verify.stages import NOISE_PCT

PEAK_GBS = 273.0  # fallback memory peak when a platform record carries none


NOISE = NOISE_PCT / 100  # same-run ratio noise band (see verify.stages.NOISE_PCT)


DPI = 160


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
        import matplotlib as mpl

        mpl.use("Agg")
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
    for (yp, x, y, t), py in zip(disp, placed, strict=True):
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
    # the model the doc belongs to — model-less documents are the headline
    # model's (readme_targets._matches), so the footer names it as such
    model = (model_of(doc) or DEFAULT_MODEL).split("/")[-1]
    theme.footer(
        fig,
        mode,
        f"mjolnir verify {doc['stage']} · {model} · run {doc['run_id']} · {ver} · {shape}{extra}",
        right=theme.board_line(doc.get("platform")),
    )


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
        rows.append([ref[L] / pm[L] if pm.get(L) else None for L in Ls])
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


def _fmt_x():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _: f"{v:.2f}×")


def _int_locator():
    from matplotlib.ticker import MaxNLocator

    return MaxNLocator(integer=True)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    theme.fit_footer(fig)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12, dpi=DPI)
    return path


def multiple(gain_pct: float) -> str:
    """A gain of 100% or more as a multiple: +200% -> "3×", +150% -> "2.5×"."""
    m = f"{1 + gain_pct / 100:.1f}"
    return f"{m.removesuffix('.0')}×"


def _register_fonts() -> None:
    """Make the bundled Inter Display weights known to matplotlib, so the
    hero and infographic render the same on every host (Thor ships Noto Sans
    without its medium / semibold weights)."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    from matplotlib import font_manager

    for f in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
    _FONTS_REGISTERED = True


def _font_file(family: str, weight: str) -> str:
    bundled = FONT_DIR / f"InterDisplay-{_BUNDLED_WEIGHTS.get(weight, 'Regular')}.ttf"
    if family == "Inter Display" and bundled.exists():
        return str(bundled)
    from matplotlib import font_manager

    return font_manager.findfont(font_manager.FontProperties(family=family, weight=weight))


def _logotype_box(img, x_from: float = 0.62) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the white logotype on the dark right
    side of the background: near-white pixels right of ``x_from`` of the
    width (the lightning lives on the left)."""
    import numpy as np

    a = np.asarray(img.convert("RGB")).astype(int)
    x0 = int(a.shape[1] * x_from)
    white = a[:, x0:].min(axis=2) > 225
    ys, xs = np.nonzero(white)
    if len(xs) < 50:  # no logotype found: the right third's centre, upper half
        w, h = img.size
        return int(w * 0.70), int(h * 0.15), int(w * 0.86), int(h * 0.62)
    return int(xs.min() + x0), int(ys.min()), int(xs.max() + x0), int(ys.max())


def _display_family() -> str:
    """A clean display face for the hero numbers: the first static-weight
    sans available (variable fonts render thin in matplotlib)."""
    from matplotlib import font_manager

    _register_fonts()
    for fam in ("Inter Display", "Inter", "Noto Sans", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"):
        try:
            path = font_manager.findfont(fam, fallback_to_default=False)
        except Exception:  # noqa: BLE001
            continue
        if path and (fam == "DejaVu Sans" or "dejavu" not in Path(path).name.lower()):
            return fam
    return "DejaVu Sans"


HERO = "hero.jpg"


# the hero canvases: one per platform plus the root's (three hammers). The
# logotype is NOT baked into these — ``render_hero`` pastes the white
# mjolnir_logo.png itself, so the same canvas serves every render.
HERO_BG_ROOT = "mjolnir_hero_bg.jpeg"
HERO_BG_PLATFORM = "mjolnir_{platform}_bg.jpeg"
HERO_LOGO = "mjolnir_logo.png"


# the root hero: platform-neutral slogan, one pill per shipped platform.
# Slogan lines are authored explicitly — one tuple entry per rendered line
# (a `\n` inside an entry breaks too) — the renderer never re-wraps or
# re-sizes them.
ROOT_HERO_SLOGAN = (
    "Faster, bigger-context LLM serving at the edge",
    "with a dedicated decode kernel,",
    "and a tuned vLLM image, one CLI",
)


ROOT_HERO_PILLS = (
    ("Jetson Thor", "#8cc63f", "#3f8a12"),
    ("Jetson Orin", "#5aa6ff", "#2a5fd6"),
    ("DGX Spark", "#f2925a", "#d4501f"),
)


# per-platform heroes: an authored slogan (one entry per line) + two
# status-aware pills (the platform name and the kernel type, per the
# registry's maturity)
PLATFORM_HERO_SLOGAN = {
    "thor": ("Faster, bigger-context LLM serving", "on Jetson AGX Thor"),
    "orin": ("Faster, bigger-context LLM serving", "on Jetson Orin"),
    "spark": ("Faster, bigger-context LLM serving", "on DGX Spark"),
}

PLATFORM_HERO_PILLS = {
    "thor": (("Jetson AGX Thor", "#8cc63f", "#3f8a12"), ("Mjolnir · tcgen05 decode kernel", "#f2925a", "#d4501f")),
    "orin": (("Jetson Orin", "#8cc63f", "#3f8a12"), ("Mjolnir · mma decode kernel", "#f2925a", "#d4501f")),
    "spark": (("DGX Spark", "#8cc63f", "#3f8a12"), ("research — baselines only", "#f2925a", "#d4501f")),
}


def hero_platform_out(platform: str | None) -> str:
    """The rendered hero's filename: ``hero.jpg`` for the root, one per
    platform for the platform pages' header banners — JPEG: the banners are
    photographic composites, and the PNGs weighed >1 MB each."""
    return HERO if platform is None else f"hero-{platform}.jpg"


FONT_DIR = Path(__file__).parent / "fonts"  # Inter Display (SIL OFL 1.1, fonts/OFL.txt)


_BUNDLED_WEIGHTS = {"normal": "Regular", "medium": "Medium", "semibold": "SemiBold", "bold": "Bold"}


_FONTS_REGISTERED = False


INFOGRAPHIC = "opening-infographic.png"
