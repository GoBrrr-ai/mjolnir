"""The opening infographic as vector SVG over the platform hero art.

The canvas is the platform's ``mjolnir_<p>_bg.jpeg`` — the same art the
hero banners composite over — embedded once per infographic as a
display-resolution WebP data URI. An external ``<image href>`` was tried
first and rejected: an SVG rides in ``<img>`` in secure static mode,
where linked subresources proved unreliable (same-origin headless
Chrome painted them only for some element geometries; behind GitHub's
camo proxy it would be worse), while a data URI paints in every
svg-as-``<img>`` renderer. The embed is display-res on purpose — the
README never draws the panel wider than ~900 css px, and the hero keeps
the full-quality master — so an infographic costs ~60 KB, a third of
the raster JPEG it replaces, and its numbers diff as text.

Everything else is native vector: the bottom-left corner scrim is a
gradient (the radial equivalent of the raster original's mask), the
stat bars are rounded rects and every number is literal text. No fonts
are embedded (secure static mode loads none): the stack ends in generic
sans and every block is anchor-aligned, so metric drift cannot overlap
anything.
"""

from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path

from mjolnir_cli.charts_base import multiple
from mjolnir_cli.charts_e2e import big_value
from mjolnir_cli.config import DEFAULT_MODEL, DEFAULT_PLATFORM, RepoLayout, model_short
from mjolnir_cli.platforms import REGISTRY, platform_of
from mjolnir_cli.theme import DARK, LIGHT

#: the embedded background's resolution and quality: display-res WebP, not
#: the hero master's 1456 px jpeg (see module docstring) — WebP cuts the
#: same-viewing-distance bytes ~45% vs jpeg and decodes on every browser
#: since 2020; the data URI paints regardless
BG_WIDTH, BG_QUALITY = 1100, 70

# the scrims' navy and the halo stroke — the raster original's constants
SCRIM, HALO = "#03070f", "#02080f"
WHITE, DIM, FAINT = "#ffffff", "#b4bfcc", "#7f8a97"

# the photo's dark right half carries the stats; the same contrast-tuned
# pair the raster version drew its bars with
C_OFF, C_MJ = DARK.flashinfer, LIGHT.mjolnir

FONT = "Inter Display,Inter,'DejaVu Sans',sans-serif"

_WEIGHTS = {"normal": "400", "medium": "500", "semibold": "600", "bold": "700"}


def _px(pt: float) -> float:
    """The raster original's pt sizes on its 100 dpi canvas -> viewBox px."""
    return round(pt * 100 / 72, 1)


def _t(x, y, s, size, fill, weight="normal", anchor="start", extra="") -> str:
    """One text element; ``y`` is the baseline (the raster original's va)."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{_px(size)}" '
        f'fill="{fill}" font-weight="{_WEIGHTS[weight]}" text-anchor="{anchor}"{extra}>{escape(s)}</text>'
    )


def _halo(x, y, s, size, fill, weight="normal") -> str:
    """Text with the dark rim that keeps it readable over the hammer art."""
    return _t(
        x,
        y,
        s,
        size,
        fill,
        weight,
        extra=f' paint-order="stroke" stroke="{HALO}" stroke-opacity="0.55" '
        f'stroke-width="{_px(3):.1f}" stroke-linejoin="round"',
    )


def _bar(x: float, y_top: float, w: float, h: float, fill: float | str) -> str:
    fill_s = f"rgba(255,255,255,{fill})" if isinstance(fill, float) else fill
    return (
        f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{max(w, h):.1f}" height="{h:.1f}" rx="{h / 2:.1f}" fill="{fill_s}"/>'
    )


def _corner_scrim(w: float, h: float) -> str:
    """Bottom-left glow for the title block: the radial equivalent of the
    old product mask (full below y = 0.72H left of x = 0.5W, fading to
    x = 0.66W / up to y = 0.42H), centred just off the bottom edge."""
    cx, cy, rx, ry = 0.22 * w, 1.02 * h, 0.55 * w, 0.66 * h
    stops = "".join(
        f'<stop offset="{t:.2f}" stop-color="{SCRIM}" stop-opacity="{0.93 * (1 - t) ** 1.1:.3f}"/>'
        for t in (i / 10 for i in range(11))
    )
    return (
        f'<radialGradient id="scrimB" gradientUnits="userSpaceOnUse" cx="0" cy="0" r="1" '
        f'gradientTransform="translate({cx:.1f} {cy:.1f}) scale({rx:.1f} {ry:.1f})">{stops}</radialGradient>'
        f'<rect x="0" y="{0.28 * h:.1f}" width="{0.78 * w:.1f}" height="{0.72 * h:.1f}" fill="url(#scrimB)"/>'
    )


def _model_line(mj: dict) -> str:
    """``<model> on <board>`` — an unknown platform keeps its raw id."""
    plat = REGISTRY.get(platform_of(mj))
    return f"{mj.get('model', '')} on {plat.name if plat else platform_of(mj)}"


def _bg_data_uri(bg_path: Path) -> str:
    """The hero art as inline WebP: display-res (see module docstring),
    deterministic by construction — fixed LANCZOS resample, quality and
    effort, so unchanged art encodes to the same bytes on every host (the
    pinned pillow carries its own libwebp)."""
    from PIL import Image

    im = Image.open(bg_path).convert("RGB")
    if im.width > BG_WIDTH:
        im = im.resize((BG_WIDTH, round(im.height * BG_WIDTH / im.width)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=BG_QUALITY, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def render_infographic(layout: RepoLayout, platform: str = DEFAULT_PLATFORM, model: str = DEFAULT_MODEL) -> Path | None:
    """``assets/benchmarks/<platform>/<model-short>/opening-<platform>-infographic.svg``
    — one per (platform, model) pair with an e2e history, over that
    platform's hero art.

    Four stat blocks fill the dark right half — decode, prefill, time to
    first token, inter-token latency — each with the Mjolnir-side
    absolute value as the big number and the change + direction as the
    subtitle, over the two measured values as bars; the claim, the model +
    platform, the scenario and the provenance sit bottom-left, under the
    hammer. The data pipeline is the raster original's
    (:func:`mjolnir_cli.e2e.latest_pair` + :func:`mjolnir_cli.e2e.infographic_row`):
    one request at the smallest context both sides measured, native
    FlashInfer on the official vLLM image vs the Mjolnir backend on the
    Mjolnir image. The root README pins the default pair (thor); each
    platform page embeds its own pair's chart.
    """
    from PIL import Image

    from mjolnir_cli import e2e

    if layout.repo_root is None:
        return None
    off, mj = e2e.latest_pair(layout.history_file, model=model, platform=platform)
    if not off or not mj:
        return None
    h = e2e.infographic_row(off, mj)
    bg_path = layout.repo_root / "assets" / f"mjolnir_{platform}_bg.jpeg"
    if not h or not bg_path.exists():
        print(
            f"[mjolnir] no official-vLLM / Mjolnir e2e pair for ({platform}, {model}) in history —"
            " infographic skipped (run: mjolnir bench e2e)"
        )
        return None
    W, H = Image.open(bg_path).size

    def X(f: float) -> float:
        return f * W

    def Y(f: float) -> float:  # the raster original's y-up fraction -> svg baseline
        return (1 - f) * H

    ver = str(off.get("image", "")).split(":")[-1].split("-")[0].lstrip("v")
    quant = str(mj.get("config", "")).split("_")[0]
    kv = "fp6 KV" if "E2M3" in str(mj.get("config", "")) else "fp8 KV"

    def fmt_ttft(v: float) -> str:
        return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:,.0f} ms"

    # ── bottom-left: claim, model + platform, scenario, provenance ─────────
    BL, BY = 0.035, 0.05
    plat = REGISTRY[platform]
    title_y = Y(BY + 0.185)
    parts = [
        # the claim line, its halo shared by both spans of the title row
        _halo(X(BL), Y(BY + 0.275), f"{plat.name.upper()}   ·   END-TO-END SERVING", 10, DIM, "semibold"),
        _t(  # the one halo the plain <text> cannot carry: two styled spans
            X(BL),
            title_y,
            "",
            34,
            WHITE,
            extra=f' paint-order="stroke" stroke="{HALO}" stroke-opacity="0.55" '
            f'stroke-width="{_px(3):.1f}" stroke-linejoin="round"',
        ),
        _halo(X(BL), Y(BY + 0.120), _model_line(mj), 22, DIM),
        _halo(
            X(BL),
            Y(BY + 0.065),
            f"1 request  ·  {e2e.ctx_label(h['ctx'])} context  · {quant}  ·  {kv}  ·  {e2e.spec_short(off, mj)}"
            "  ·  vs the official vllm",
            10.5,
            DIM,
        ),
        _t(X(BL), Y(BY + 0.020), f"{off.get('image', '')}  vs  {mj.get('image', '')}  ·  llama-benchy", 8.5, FAINT),
    ]
    # the title row's spans, spliced into the halo carrier above: bold
    # white Mjolnir, dim vs vLLM <version> trailing 14px later
    parts[1] = parts[1].replace(
        "></text>",
        f'><tspan fill="{WHITE}" font-weight="700">Mjolnir</tspan>'
        f'<tspan dx="14" fill="{DIM}"> vs vLLM {escape(ver)}</tspan></text>',
    )

    # ── right: four stat blocks ────────────────────────────────────────────
    blocks = (
        ("tg_tps", "faster decode", "slower decode", "{:.1f}"),
        ("pp_tps", "faster prefill", "slower prefill", "{:,.0f}"),
        ("ttft_ms", "lower TTFT", "higher TTFT", None),
        ("itl_ms", "lower ITL", "higher ITL", "{:.1f} ms"),
    )
    L, R, gap = 0.56, 0.96, 0.045
    cw = (R - L - gap) / 2
    tops = (0.93, 0.48)
    for i, (key, better, worse, fmt) in enumerate(blocks):
        va, vb, gain = h.get(key) or (None, None, None)
        if va is None or vb is None or gain is None:
            continue
        x0, y0 = L + (i % 2) * (cw + gap), tops[i // 2]
        # the BIG number is the Mjolnir-side absolute value; the subtitle
        # carries the change + direction (throughput: the change in
        # tokens/s, a doubling or more as a multiple; latency: the change
        # in ms — a reduction, never the reciprocal speedup)
        change = e2e.pct(-gain) if key in ("ttft_ms", "itl_ms") else multiple(gain) if gain >= 100 else e2e.pct(gain)
        parts.append(_t(X(x0), Y(y0 - 0.115), big_value(key, vb), 50, WHITE, "bold"))
        parts.append(_t(X(x0), Y(y0 - 0.178), f"{change} {better if gain >= 0 else worse}", 16.5, WHITE))
        X0, BW = X(x0), cw * W
        vmax = max(va, vb)
        for j, (lbl, v, col) in enumerate((("vLLM", va, C_OFF), ("Mjolnir", vb, C_MJ))):
            y_top = Y(y0 - 0.255 - j * 0.085)
            bh = 0.014 * H
            shown = fmt_ttft(v) if fmt is None else fmt.format(v)
            parts.append(_t(X0, y_top - 9, lbl, 9.5, DIM))
            parts.append(_t(X0 + BW, y_top - 9, shown, 11, WHITE, "bold", anchor="end"))
            parts.append(_bar(X0, y_top, BW, bh, 0.10))  # the track, then the value
            parts.append(_bar(X0, y_top, BW * v / vmax, bh, col))

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
        f'<image x="0" y="0" width="{W}" height="{H}" preserveAspectRatio="none" href="{_bg_data_uri(bg_path)}"/>'
        + _corner_scrim(W, H)
        + "".join(parts)
        + "</svg>"
    )
    out = (
        layout.repo_root
        / "assets"
        / "benchmarks"
        / platform
        / model_short({"model": model})
        / f"opening-{platform}-infographic.svg"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    final = svg.encode() + b"\n"
    if not (out.exists() and out.read_bytes() == final):
        out.write_bytes(final)
    print(f"[mjolnir] wrote {out}")
    return out
