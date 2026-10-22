"""The Mjolnir chart design system — one visual language, light AND dark.

Static charts render the dark palette to ``assets/benchmarks/`` (``<name>.png``);
the interactive site renders both palettes client-side from ``MODES``.

Color = identity, and identities are fixed (never assigned by rank):

* **Mjolnir** — orange (the hammer). A second Mjolnir series (fp8 KV next
  to the e2m3 headline) keeps the SAME orange with a hollow marker + a direct
  label: identity by hue, variant by marker fill + label (lighter/darker
  tints of one hue fail the normal-vision floor on the dark surface — the
  validator rejected every tint tried, so tints are not used).
* **FlashInfer** — blue (the strongest stock baseline).
* **FlashInfer** — blue (the strongest stock baseline).
* **FlashAttention 2 / 4** — aqua (the FA-family slot). FA4 hd256 (the patched
  image's FA path: 1-CTA decode, no SplitKV, fp8 Q+KV) is the default FA leg —
  inverted triangle; FA2 (bf16 KV, available on demand) keeps the upright one.
  Identity by hue, variant by marker + label.

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
    "fa4": ("fa2", "v", True),  # the FA-family slot: FA4 hd256, inverted triangle
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
    return {
        "color": c,
        "lw": 2.0,
        "marker": marker(series_id),
        "ms": 8,
        "mfc": c if filled(series_id) else mode.panel,
        "mec": c,
        "mew": 2.0,
        "zorder": 5,
    }


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
    """Left-aligned bold title + a dim subtitle above the axes. The subtitle
    wraps at the axes' width (at `` · `` joints): one wider than the plot
    would stretch the saved image and leave an empty band on the right."""
    sub_lines = []
    if subtitle:
        fig = ax.figure
        pos = ax.get_position()  # the subtitle may run across every panel of the top row
        right = max(a.get_position().x1 for a in fig.axes if abs(a.get_position().y1 - pos.y1) < 0.02)
        width_in = (right - pos.x0) * fig.get_figwidth()
        per_line = max(30, int(width_in * 72 / (10.5 * 0.56)))  # DejaVu Sans ≈ 0.56 em per char
        cur = ""
        for part in subtitle.split(" · "):
            if cur and len(cur) + 3 + len(part) > per_line:
                sub_lines.append(cur)
                cur = part
            else:
                cur = f"{cur} · {part}" if cur else part
        sub_lines.append(cur)
    nlines = text.count("\n") + 1
    pad = (26 if subtitle else 10) + (fontsize + 5) * (nlines - 1) + 15 * max(0, len(sub_lines) - 1)
    t = ax.set_title(text, pad=pad)
    t.set_color(mode.text)
    t.set_fontsize(fontsize)
    t.set_fontweight("bold")
    _, y = t.get_position()
    t.set_position((0.0, y))
    t.set_horizontalalignment("left")
    if subtitle:
        ax.text(
            0,
            1.018,
            "\n".join(sub_lines),
            transform=ax.transAxes,
            color=mode.text_dim,
            fontsize=10.5,
            ha="left",
            va="bottom",
            linespacing=1.25,
        )


def footer(fig, mode: Mode, text: str, below_in: float | None = None) -> None:
    # below the axes (and their x labels): the bbox-tight save grows to fit.
    # Horizontally the two texts are pinned to the content's edges at save time
    # (``fit_footer``): placed at the figure's edges they would stretch the tight
    # bbox to the whole canvas and leave wide empty margins left and right.
    drop = below_in / fig.get_figheight() if below_in else 0.13
    y = min(ax.get_position().y0 for ax in fig.axes if ax.get_visible()) - drop if fig.axes else 0.0
    left = fig.text(0.01, y, text, ha="left", va="top", color=mode.text_dim, fontsize=8.5)
    right = fig.text(
        0.995,
        y,
        "MJOLNIR · Jetson AGX Thor · sm_110",
        ha="right",
        va="top",
        color=mode.text_dim,
        fontsize=8,
        family="monospace",
    )
    fig._mjolnir_footer = (left, right)


def fit_footer(fig) -> None:
    """Move the footer to the left / right edge of everything else on the
    figure (call right before a bbox-tight save)."""
    texts = getattr(fig, "_mjolnir_footer", None)
    if not texts:
        return
    for t in texts:
        t.set_in_layout(False)
    renderer = fig.canvas.get_renderer()
    bb = fig.get_tightbbox(renderer)  # inches, footer excluded
    w = fig.get_figwidth()
    left, right = texts
    left.set_x(bb.x0 / w)
    right.set_x(bb.x1 / w)
    # a footer wider than the content still widens the bbox, never clips
    for t in texts:
        t.set_in_layout(True)


def wrap(text: str, width: int = 24) -> str:
    """Break a long series label into lines at its natural joints: `` · ``
    (dropped at a break) and `` + `` (kept, leading the next line), so
    "Mjolnir · fp6 e2m3 KV + fp8 lm_head" reads "Mjolnir · fp6 e2m3 KV\n+ fp8 lm_head"
    instead of widening the chart."""
    import re

    parts = re.split(r"( · | \+ )", text)
    lines, cur = [], parts[0]
    for sep, word in zip(parts[1::2], parts[2::2], strict=False):
        if len(cur) + len(sep) + len(word) <= width:
            cur += sep + word
        else:
            lines.append(cur)
            cur = ("+ " if "+" in sep else "") + word
    lines.append(cur)
    return "\n".join(lines)


def legend(ax, mode: Mode, **kw):
    return ax.legend(
        facecolor=mode.panel,
        edgecolor=mode.grid,
        labelcolor=mode.text,
        framealpha=1.0,
        **{"fontsize": 10, **kw},
    )
