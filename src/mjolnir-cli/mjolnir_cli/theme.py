"""The Mjolnir chart design system — one visual language, light AND dark.

Every chart is rendered twice (``<name>.png`` light, ``<name>-dark.png``
dark) and the README switches with ``<picture>`` + ``prefers-color-scheme``,
so a chart always sits on its own surface on GitHub.

Color = identity, and identities are fixed (never assigned by rank):

* **Mjolnir** — orange (the hammer). A second Mjolnir series (fp8 KV next
  to the e2m3 headline) keeps the SAME orange with a hollow marker + a direct
  label: identity by hue, variant by marker fill + label (lighter/darker
  tints of one hue fail the normal-vision floor on the dark surface — the
  validator rejected every tint tried, so tints are not used).
* **FlashInfer** — blue (the strongest stock baseline).
* **FlashAttention 2** — aqua (stock vLLM 0.31.0's sm_110 default).

Those three slots are the validated reference palette's first three steps:
they pass the all-pairs colorblind (CVD ΔE ≥ 8) and normal-vision (ΔE ≥ 15)
checks in both modes (scripts/validate_palette.js; light-mode aqua sits
below 3:1 contrast → FA2 series are always direct-labeled and every chart
has a data table in the README). Speedup heatmaps use the diverging pair
blue (FlashInfer faster) ↔ neutral gray (parity) ↔ orange (Mjolnir faster).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    name: str  # "light" | "dark"
    bg: str  # figure background
    panel: str  # axes background (the chart surface the palette is validated on)
    text: str
    text_dim: str
    grid: str
    mjolnir: str
    flashinfer: str
    fa2: str
    neutral: str  # diverging midpoint / "other" marks
    suffix: str  # file-name suffix


LIGHT = Mode(
    name="light",
    bg="#ffffff",
    panel="#fcfcfb",
    text="#0b0b0b",
    text_dim="#52514e",
    grid="#e6e5e1",
    mjolnir="#eb6834",
    flashinfer="#2a78d6",
    fa2="#1baf7a",
    neutral="#f0efec",
    suffix="",
)

DARK = Mode(
    name="dark",
    bg="#0d1117",  # GitHub dark
    panel="#151b26",
    text="#e6edf3",
    text_dim="#8b949e",
    grid="#21262d",
    mjolnir="#d95926",
    flashinfer="#3987e5",
    fa2="#199e70",
    neutral="#383835",
    suffix="-dark",
)

MODES = (LIGHT, DARK)

# Series id -> (color attribute on Mode, marker, filled). Ids match verify
# legs and e2e configs, so a series has the same look in every chart.
SERIES_STYLE = {
    "mjolnir": ("mjolnir", "o", True),
    "mjolnir-fp8": ("mjolnir", "s", False),
    "fi": ("flashinfer", "D", True),
    "fi-hnd": ("flashinfer", "d", False),
    "fa2": ("fa2", "^", True),
    # e2e configs
    "NVFP4_MJOLNIR": ("mjolnir", "s", False),  # fp8 KV: the hollow Mjolnir variant
    "NVFP4_MJOLNIR_E2M3": ("mjolnir", "o", True),
    "NVFP4": ("flashinfer", "D", True),
    "NVFP4_FA2": ("fa2", "^", True),
    # the fast experiment config: Mjolnir orange, plus marker, dotted fill in bars
    "NVFP4_MJOLNIR_E2M3_LMH8_VT64K": ("mjolnir", "P", True),
}
# bar fill pattern per series (identity is never color-only: hollow = fp8 KV,
# dotted = the fp8 lm_head + draft-vocab experiment)
SERIES_HATCH = {"NVFP4_MJOLNIR_E2M3_LMH8_VT64K": ".."}


def color(mode: Mode, series_id: str) -> str:
    attr = SERIES_STYLE.get(series_id, ("text_dim", "o", True))[0]
    return getattr(mode, attr)


def marker(series_id: str) -> str:
    return SERIES_STYLE.get(series_id, ("text_dim", "o", True))[1]


def filled(series_id: str) -> bool:
    return SERIES_STYLE.get(series_id, ("text_dim", "o", True))[2]


def hatch(series_id: str) -> str | None:
    if not filled(series_id):
        return "//"
    return SERIES_HATCH.get(series_id)


def line_kw(mode: Mode, series_id: str) -> dict:
    """2px line + >=8px marker; hollow markers carry a variant (same hue)."""
    c = color(mode, series_id)
    return dict(
        color=c,
        lw=2.0,
        marker=marker(series_id),
        ms=8,
        mfc=c if filled(series_id) else mode.panel,
        mec=c,
        mew=2.0,
        zorder=5,
    )


def diverging_cmap(mode: Mode):
    """blue (FlashInfer faster) -> gray (parity) -> orange (Mjolnir faster)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(f"mjolnir-div-{mode.name}", [mode.flashinfer, mode.neutral, mode.mjolnir])


# status scale for "how much worse than the baseline" (accuracy heatmaps):
# soft tints in light mode, deep ones in dark mode, so the cell text (mode.text)
# stays readable on every step; identity colors are never reused for it
STATUS = {
    "light": ("#bbf7d0", "#fed7aa", "#fecaca"),  # on par / worse / much worse
    "dark": ("#14532d", "#7c2d12", "#7f1d1d"),
}


def status_cmap(mode: Mode, par: float, span: float):
    """0 → ``span`` points worse: green up to ``par``, then to orange at the
    middle of the rest, red at ``span``."""
    from matplotlib.colors import LinearSegmentedColormap

    good, warn, bad = STATUS[mode.name]
    p = max(0.0, min(0.9, par / span))
    return LinearSegmentedColormap.from_list(
        f"mjolnir-status-{mode.name}", [(0.0, good), (p, good), ((1 + p) / 2, warn), (1.0, bad)]
    )


def apply(fig, axes, mode: Mode) -> None:
    """Surfaces, recessive hairline grid, muted ticks — for one or many axes."""
    fig.patch.set_facecolor(mode.bg)
    for ax in axes if isinstance(axes, (list, tuple)) else [axes]:
        ax.set_facecolor(mode.panel)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=mode.text_dim, labelsize=10, which="both", length=0)
        ax.grid(True, color=mode.grid, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(mode.text_dim)
        ax.yaxis.label.set_color(mode.text_dim)


def title(ax, mode: Mode, text: str, subtitle: str | None = None, fontsize: int = 16) -> None:
    """Left-aligned bold title + one dim subtitle line above the axes."""
    nlines = text.count("\n") + 1
    pad = (26 if subtitle else 10) + (fontsize + 5) * (nlines - 1)
    t = ax.set_title(text, pad=pad)
    t.set_color(mode.text)
    t.set_fontsize(fontsize)
    t.set_fontweight("bold")
    _, y = t.get_position()
    t.set_position((0.0, y))
    t.set_horizontalalignment("left")
    if subtitle:
        ax.text(0, 1.018, subtitle, transform=ax.transAxes, color=mode.text_dim, fontsize=10.5, ha="left", va="bottom")


def footer(fig, mode: Mode, text: str) -> None:
    # below the axes (and their x labels): the bbox-tight save grows to fit
    y = min(ax.get_position().y0 for ax in fig.axes) - 0.13 if fig.axes else 0.0
    fig.text(0.01, y, text, ha="left", va="top", color=mode.text_dim, fontsize=8.5)
    fig.text(
        0.995,
        y,
        "MJOLNIR · Jetson AGX Thor · sm_110",
        ha="right",
        va="top",
        color=mode.text_dim,
        fontsize=8,
        family="monospace",
    )


def legend(ax, mode: Mode, **kw):
    leg = ax.legend(
        facecolor=mode.panel,
        edgecolor=mode.grid,
        labelcolor=mode.text,
        framealpha=1.0,
        **{"fontsize": 10, **kw},
    )
    return leg
