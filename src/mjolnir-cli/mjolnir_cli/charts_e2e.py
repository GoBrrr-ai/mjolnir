"""E2E serving charts, the hero image and the opening infographic."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

from mjolnir_cli import theme
from mjolnir_cli.config import DEFAULT_MODEL, DEFAULT_PLATFORM, RepoLayout, model_of
from mjolnir_cli.history import load_records
from mjolnir_cli.platforms import REGISTRY, platform_of
from mjolnir_cli.theme import Mode

from mjolnir_cli.charts_base import (
    HERO_BG_PLATFORM,
    HERO_BG_ROOT,
    HERO_LOGO,
    L_label,
    PLATFORM_HERO_PILLS,
    PLATFORM_HERO_SLOGAN,
    ROOT_HERO_PILLS,
    ROOT_HERO_SLOGAN,
    _display_family,
    _font_file,
    _plt,
    hero_platform_out,
    multiple,
)

E2E_SERIES = (
    ("NVFP4", "FlashInfer · official vLLM"),
    ("NVFP4_FA2", "FlashAttention 2 · official vLLM"),
    ("NVFP4_MJOLNIR", "Mjolnir · fp8 KV"),
    ("NVFP4_MJOLNIR_E2M3", "Mjolnir · fp6 KV"),
    ("NVFP4_MJOLNIR_E2M3_LMH8_VT64K", "Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab"),
)


MJOLNIR_E2E_CONFIGS = ("NVFP4_MJOLNIR", "NVFP4_MJOLNIR_E2M3", "NVFP4_MJOLNIR_E2M3_LMH8_VT64K")


def e2e_rows(layout: RepoLayout, model: str = DEFAULT_MODEL, platform: str | None = None) -> dict[str, dict]:
    """Latest history record per series: stock configs on the official vLLM
    image, the Mjolnir config on a Mjolnir image. Stock backends measured on
    the Mjolnir image are ignored (that image patches FlashInfer too). One
    model's runs: a series is one config of one model. ``platform`` narrows
    the series to one platform's records (None = any — the callers that
    predate per-pair payloads keep their behaviour)."""
    from mjolnir_cli import e2e
    from mjolnir_cli.platforms import platform_of

    latest: dict[str, dict] = {}
    for r in load_records(layout.history_file):
        cfg = r.get("config")
        if not r.get("cells") or model_of(r) != model:
            continue
        if platform is not None and platform_of(r) != platform:
            continue
        ok = (cfg in MJOLNIR_E2E_CONFIGS and e2e.is_mjolnir(r)) or (
            cfg in ("NVFP4", "NVFP4_FA2") and e2e.is_official(r)
        )
        if ok and (cfg not in latest or r.get("epoch", 0) >= latest[cfg].get("epoch", 0)):
            latest[cfg] = r
    return latest


def _e2e_series(rec: dict) -> tuple[str, str]:
    """(theme series id, legend label) for one side of the e2e pair."""
    from mjolnir_cli.config import BACKEND_LABELS

    cfg = str(rec.get("config", ""))
    label = dict(E2E_SERIES).get(cfg) or BACKEND_LABELS.get(cfg, cfg)
    if cfg in theme.SERIES_STYLE:
        return cfg, label
    return ("NVFP4_MJOLNIR_E2M3" if "MJOLNIR" in cfg else "NVFP4"), label


def _e2e_chart(
    layout: RepoLayout,
    mode: Mode,
    metric: str,
    ylabel: str,
    ttl: str,
    sub: str,
    platform: str = DEFAULT_PLATFORM,
    model: str = DEFAULT_MODEL,
):
    """The README table as bars: the same pair (official vLLM vs the Mjolnir
    config, :func:`mjolnir_cli.e2e.latest_pair` — each side's runs merged cell
    by cell), every (requests, context) cell either side measured, one panel
    per request count; a cell one side lacks is marked "not measured" instead
    of being dropped."""
    from mjolnir_cli import e2e

    off, mj = e2e.latest_pair(layout.history_file, model=model, platform=platform)
    if not off or not mj:
        return None
    cells = e2e.all_cells(off, mj)
    concs = sorted({c for c, _ in cells})
    if not concs:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, len(concs), figsize=(4.6 * len(concs) + 1.2, 4.8), sharey=True, squeeze=False)
    axes = list(axes[0])
    theme.apply(fig, axes, mode)
    sides = [(rec, *_e2e_series(rec)) for rec in (off, mj)]
    top = max((e2e.value(e2e.cell(rec, c, x), metric) or 0) for rec in (off, mj) for c, x in cells)
    for ax, conc in zip(axes, concs, strict=True):
        ctxs = sorted({x for c, x in cells if c == conc and (x or metric != "pp_tps")})  # no ctx: no prefill
        w = 0.8 / len(sides)
        for i, (rec, sid, lbl) in enumerate(sides):
            hollow = not theme.filled(sid)
            col = theme.color(mode, sid)
            labelled = ax is not axes[0]  # legend entry: the side's first drawn bar
            for j, ctx in enumerate(ctxs):
                x = j + (i - (len(sides) - 1) / 2) * w
                v = e2e.value(e2e.cell(rec, conc, ctx), metric)
                if v is None:
                    ax.text(
                        x,
                        top * 0.02,
                        "n/a",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        color=mode.text_dim,
                        zorder=4,
                    )
                    continue
                ax.bar(
                    x,
                    v,
                    width=w * 0.92,
                    color=mode.panel if hollow else col,
                    edgecolor=col if hollow or not theme.hatch(sid) else mode.panel,
                    linewidth=1.6 if hollow else 0,
                    hatch=theme.hatch(sid),
                    zorder=3,
                    label=None if labelled else lbl,
                )
                labelled = True
        ax.set_xticks(range(len(ctxs)))
        ax.set_xticklabels([f"{L_label(c)}" if c else "0" for c in ctxs])
        ax.set_xlabel(f"context · {conc} concurrent request{'s' if conc > 1 else ''}")
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(0, top * 1.08)
    # the legend sits in the title row, right-aligned: no panel is wide enough for it
    h, lbl = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        lbl,
        loc="lower right",
        bbox_to_anchor=(axes[-1].get_position().x1, axes[-1].get_position().y1 + 0.01),
        ncol=len(lbl),
        facecolor=mode.panel,
        edgecolor=mode.grid,
        labelcolor=mode.text,
        framealpha=1.0,
        fontsize=9,
    )
    theme.title(axes[0], mode, ttl, sub)
    theme.footer(
        fig,
        mode,
        f"mjolnir bench e2e · {mj.get('model', '')} · {e2e.spec_short(off, mj)} · "
        f"{off.get('image', '')} vs {mj.get('image', '')} · llama-benchy",
        right=theme.board_line(platform),
    )
    return fig


def chart_e2e_decode(layout: RepoLayout, mode: Mode, platform: str = DEFAULT_PLATFORM, model: str = DEFAULT_MODEL):
    return _e2e_chart(
        layout,
        mode,
        "tg_tps",
        "decode tokens/s (higher is better)",
        "End-to-end decode throughput",
        "official image (native FlashInfer) vs the Mjolnir image · the README table's cells",
        platform,
        model,
    )


def chart_e2e_prefill(layout: RepoLayout, mode: Mode, platform: str = DEFAULT_PLATFORM, model: str = DEFAULT_MODEL):
    return _e2e_chart(
        layout,
        mode,
        "pp_tps",
        "prompt tokens/s (higher is better)",
        "End-to-end prompt processing",
        "official image (native FlashInfer) vs the Mjolnir image · the README table's cells",
        platform,
        model,
    )


def _rgba(hex_color: str) -> tuple[int, int, int, int]:
    """``#rrggbb`` -> the opaque RGBA tuple PIL would produce for it."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255


def big_value(key: str, v: float) -> str:
    """The BIG absolute value of a stat block: int t/s with ``k`` scaling at
    ≥10k (decode, prefill); seconds at ≥1 s else ms (TTFT); ms (ITL)."""
    if key in ("tg_tps", "pp_tps"):
        return f"{v / 1000:.0f}k t/s" if v >= 10_000 else f"{v:.0f} t/s"
    if key == "ttft_ms":
        return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms"
    return f"{v:.1f} ms"


def _white_logo(layout: RepoLayout):
    """``assets/mjolnir_logo.png`` recolored white (the glyph is black on
    transparency; the alpha channel carries the shape, so the RGB channels
    simply go to the soft white the text uses)."""
    from PIL import Image

    if layout.repo_root is None:
        return None
    src = layout.repo_root / "assets" / HERO_LOGO
    if not src.exists():
        return None
    logo = Image.open(src).convert("RGBA")
    import numpy as np

    arr = np.array(logo)
    arr[..., :3] = np.where(arr[..., 3:] > 0, np.array([232, 238, 248]), arr[..., :3])
    return Image.fromarray(arr, "RGBA")


def render_hero(layout: RepoLayout, platform: str | None = None) -> Path | None:
    """The README header banner: the platform's ``mjolnir_<p>_bg.jpeg`` (the
    root: ``mjolnir_hero_bg.jpeg``, the three hammers), the white Mjölnir
    logotype pasted over the dark right half, a one-line slogan under it and
    a row of gradient pills — the root carries one pill per shipped platform,
    a platform banner its name + kernel-type pills. Rendered at 2x and
    downsampled for smooth edges."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    from mjolnir_cli.config import VLLM_VERSION

    if layout.repo_root is None:
        return None
    bg_name = HERO_BG_ROOT if platform is None else HERO_BG_PLATFORM.format(platform=platform)
    src = layout.repo_root / "assets" / bg_name
    if not src.exists():
        return None
    logo_img = _white_logo(layout)
    if logo_img is None:
        return None
    bg = Image.open(src).convert("RGBA")
    W, H = bg.size
    # the hammer art fills the left; the stack centres on the dark right half
    cx = W * 0.76
    S = 2  # draw at 2x, downsample: smooth pill edges and text
    fam = _display_family()

    over = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d, ds = ImageDraw.Draw(over), ImageDraw.Draw(shadow)

    # the white logotype, centred on the right half: big, near the top
    logo_h = round(H * 0.44)
    logo_w = round(logo_img.size[0] * logo_h / logo_img.size[1])
    if logo_w > 0.40 * W:  # a wide wordmark variant: cap by width instead
        logo_w = round(0.40 * W)
        logo_h = round(logo_w * logo_img.size[1] / logo_img.size[0])
    logo = logo_img.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
    ly = round(H * 0.04)
    over.alpha_composite(logo, (round(cx * S - logo_w * S / 2), ly * S))

    # slogan: centred line(s), shrunk to stay inside the dark right half
    # (the art fills the left ~55%), stacked upward to end just above the pills
    slogan = ROOT_HERO_SLOGAN if platform is None else (PLATFORM_HERO_SLOGAN[platform],)
    max_w = 0.42 * W * S
    size = round(H * 0.040)
    slogan_font = ImageFont.truetype(_font_file(fam, "medium"), size * S)
    while size > round(H * 0.024) and max(d.textlength(ln, font=slogan_font) for ln in slogan) > max_w:
        size -= 1
        slogan_font = ImageFont.truetype(_font_file(fam, "medium"), size * S)
    line_h = slogan_font.size * 1.38
    # the pill row anchors the bottom; the slogan block sits right above it
    pill_h = round(H * 0.060) * S
    pills_y = round((H - 0.055 * H) * S) - pill_h
    y = pills_y - round(H * 0.030 * S) - round(line_h * len(slogan))
    for line in slogan:
        w = d.textlength(line, font=slogan_font)
        x = cx * S - w / 2
        ds.text((x, y + 2 * S), line, font=slogan_font, fill=(0, 0, 0, 170))
        d.text((x, y), line, font=slogan_font, fill=(232, 238, 248, 255))
        y += line_h

    # pills: one centred row, shrunk if it would run past the dark half
    pills = ROOT_HERO_PILLS if platform is None else PLATFORM_HERO_PILLS[platform]
    labels = [(t.format(vllm=VLLM_VERSION), c0, c1) for t, c0, c1 in pills]
    pad_x, pill_h, gap = 18 * S, pill_h, 14 * S
    size = round(H * 0.030)
    pill_font = ImageFont.truetype(_font_file(fam, "semibold"), size * S)
    row_w = lambda fs: sum(d.textlength(t, font=fs) + 2 * pad_x for t, _, _ in labels) + gap * (len(labels) - 1)  # noqa: E731
    while size > round(H * 0.020) and row_w(pill_font) > max_w:
        size -= 1
        pill_font = ImageFont.truetype(_font_file(fam, "semibold"), size * S)
    widths = [d.textlength(t, font=pill_font) + 2 * pad_x for t, _, _ in labels]
    x = cx * S - (sum(widths) + gap * (len(widths) - 1)) / 2
    y = pills_y
    for (text, c0, c1), w in zip(labels, widths, strict=True):
        box = (int(x), y, int(x + w), y + pill_h)
        ds.rounded_rectangle((box[0], box[1] + 3 * S, box[2], box[3] + 3 * S), radius=pill_h // 2, fill=(0, 0, 0, 120))
        # horizontal gradient, clipped to the pill by a rounded mask
        a0, a1 = _rgba(c0), _rgba(c1)
        grad = Image.new("RGBA", (box[2] - box[0], pill_h))
        gw = max(1, grad.size[0] - 1)
        grad.putdata(
            [
                tuple(int(a0[k] + (a1[k] - a0[k]) * (i / gw)) for k in range(4))
                for _ in range(pill_h)
                for i in range(grad.size[0])
            ]
        )
        mask = Image.new("L", grad.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, grad.size[0] - 1, pill_h - 1), radius=pill_h // 2, fill=255)
        over.paste(grad, box[:2], mask)
        # a thin light rim on top
        d.rounded_rectangle(box, radius=pill_h // 2, outline=(255, 255, 255, 70), width=S)
        tw = d.textlength(text, font=pill_font)
        ty = y + (pill_h - pill_font.size) / 2 - pill_font.size * 0.08
        d.text((x + (w - tw) / 2, ty + S), text, font=pill_font, fill=(0, 0, 0, 90))
        d.text((x + (w - tw) / 2, ty), text, font=pill_font, fill=(255, 255, 255, 255))
        x += w + gap

    shadow = shadow.filter(ImageFilter.GaussianBlur(4 * S))
    layer = Image.alpha_composite(shadow, over).resize((W, H), Image.Resampling.LANCZOS)
    out = layout.repo_root / "assets" / hero_platform_out(platform)
    Image.alpha_composite(bg, layer).convert("RGB").save(out, optimize=True)
    return out


def render_infographic(layout: RepoLayout, platform: str = DEFAULT_PLATFORM, model: str = DEFAULT_MODEL) -> Path | None:
    """``assets/benchmarks/<platform>/<model-short>/opening-<platform>-infographic.png``
    — one per (platform, model) pair with an e2e history, over that
    platform's ``mjolnir_<p>_bg.jpeg`` canvas.

    Four stat blocks fill the dark right half — decode, prefill, time to
    first token, inter-token latency — each with the Mjolnir-side
    absolute value as the big number and the change + direction as the
    subtitle, over the two measured values as bars; the claim, the model +
    platform, the scenario and the provenance sit bottom-left, under the
    hammer (a bottom scrim keeps them readable over the handle). Every number
    comes from the pair's latest history row (:func:`mjolnir_cli.e2e.latest_pair`)
    — one request at the smallest context both sides measured
    (:func:`mjolnir_cli.e2e.infographic_row`) — native FlashInfer on the
    official vLLM image vs the Mjolnir backend on the Mjolnir image.

    The root README pins the default pair (thor); each platform page embeds
    its own pair's chart (a platform suffix on a per-platform chart is a
    data dimension, not the banned theme suffix).
    """
    import numpy as np
    from matplotlib.patches import FancyBboxPatch

    from mjolnir_cli import e2e
    from mjolnir_cli.config import model_short

    if layout.repo_root is None:
        return None
    off, mj = e2e.latest_pair(layout.history_file, model=model, platform=platform)
    if not off or not mj:
        return None
    h = e2e.infographic_row(off, mj)
    bg_path = layout.repo_root / "assets" / HERO_BG_PLATFORM.format(platform=platform)
    if not h or not bg_path.exists():
        print(
            f"[mjolnir] no official-vLLM / Mjolnir e2e pair for ({platform}, {model}) in history —"
            " infographic skipped (run: mjolnir bench e2e)"
        )
        return None
    out = (
        layout.repo_root
        / "assets"
        / "benchmarks"
        / platform
        / model_short({"model": model})
        / f"opening-{platform}-infographic.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    plt = _plt()
    bg = plt.imread(str(bg_path))
    H, W = bg.shape[:2]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.imshow(bg, extent=(0, W, 0, H), zorder=0)
    # scrims: navy on the right (the stats), and up from the bottom-left
    # corner (the title block over the hammer's handle); clear elsewhere
    gx, gy = np.meshgrid(np.linspace(0, 1, 512), np.linspace(1, 0, 256))  # row 0 = top
    right = np.clip((gx - 0.42) / 0.22, 0, 1) ** 1.5 * 0.76
    bottom = np.clip((0.58 - gy) / 0.30, 0, 1) ** 0.9 * 0.93 * np.clip((0.66 - gx) / 0.16, 0, 1)
    scrim = np.zeros((256, 512, 4))
    scrim[..., :3] = np.array([3, 7, 16]) / 255
    scrim[..., 3] = np.maximum(right, bottom)
    ax.imshow(scrim, extent=(0, W, 0, H), aspect="auto", zorder=1)

    disp = _display_family()
    white, dim, faint = "#ffffff", "#b4bfcc", "#7f8a97"
    c_off, c_mj = theme.DARK.flashinfer, theme.LIGHT.mjolnir
    ver = str(off.get("image", "")).split(":")[-1].split("-")[0].lstrip("v")
    quant = str(mj.get("config", "")).split("_")[0]
    kv = "fp6 KV" if "E2M3" in str(mj.get("config", "")) else "fp8 KV"

    from matplotlib import patheffects

    halo = [patheffects.withStroke(linewidth=3, foreground=(0.01, 0.03, 0.06, 0.55))]

    def T(xf, yf, text, size, color=white, weight="normal", ha="left", **kw):
        return ax.text(
            xf * W, yf * H, text, fontsize=size, color=color, weight=weight, ha=ha, family=disp, zorder=6, **kw
        )

    # ── bottom-left: claim, model + platform, scenario, provenance ─────────
    BL, BY = 0.035, 0.05  # left margin; lift of the whole block off the bottom edge
    plat_rec = REGISTRY[platform]
    T(BL, BY + 0.275, f"{plat_rec.name.upper()}   ·   END-TO-END SERVING", 10, dim, "semibold", path_effects=halo)
    title = T(BL, BY + 0.185, "Mjolnir", 34, white, "bold", path_effects=halo)
    x_after = title.get_window_extent(renderer=fig.canvas.get_renderer()).x1
    ax.text(
        x_after + 14,
        (BY + 0.185) * H,
        f"vs vLLM {ver}",
        fontsize=34,
        color=dim,
        family=disp,
        zorder=6,
        path_effects=halo,
    )
    plat = REGISTRY.get(platform_of(mj))
    T(BL, BY + 0.120, f"{mj.get('model', '')} on {plat.name if plat else platform_of(mj)}", 22, dim, path_effects=halo)
    T(
        BL,
        BY + 0.065,
        f"1 request  ·  {e2e.ctx_label(h['ctx'])} context  · {quant}  ·  {kv}  ·  {e2e.spec_short(off, mj)}"
        "  ·  vs the official vllm",
        10.5,
        dim,
        path_effects=halo,
    )
    T(
        BL,
        BY + 0.020,
        f"{off.get('image', '')}  vs  {mj.get('image', '')}  ·  llama-benchy",
        8.5,
        faint,
    )

    # ── right: four stat blocks ────────────────────────────────────────────
    blocks = (
        ("tg_tps", "faster decode", "slower decode", "", "{:.1f}"),
        ("pp_tps", "faster prefill", "slower prefill", "", "{:,.0f}"),
        ("ttft_ms", "lower TTFT", "higher TTFT", "", None),
        ("itl_ms", "lower ITL", "higher ITL", "", "{:.1f} ms"),
    )
    L, R, gap = 0.56, 0.96, 0.045
    cw = (R - L - gap) / 2
    tops = (0.93, 0.48)

    def fmt_ttft(v: float) -> str:
        return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:,.0f} ms"

    for i, (key, better, worse, unit, fmt) in enumerate(blocks):
        va, vb, gain = h.get(key) or (None, None, None)
        if va is None or vb is None or gain is None:
            continue
        x0 = L + (i % 2) * (cw + gap)
        y0 = tops[i // 2]
        # the BIG number is the Mjolnir-side absolute value; the subtitle
        # carries the change + direction. Throughput: the change in tokens/s,
        # a doubling or more as a multiple (+200% reads "3×"); latency: the
        # change in ms (−27% = 27% fewer ms — a reduction, never the
        # reciprocal speedup)
        change = e2e.pct(-gain) if key in ("ttft_ms", "itl_ms") else multiple(gain) if gain >= 100 else e2e.pct(gain)
        T(x0, y0 - 0.115, big_value(key, vb), 50, white, "bold")
        T(x0 + cw, y0 - 0.030, unit, 10.5, faint, ha="right")  # above the big value's caps
        T(x0, y0 - 0.178, f"{change} {better if gain >= 0 else worse}", 16.5, white)
        X, BW = x0 * W, cw * W
        vmax = max(va, vb)
        for j, (lbl, v, col) in enumerate((("vLLM", va, c_off), ("Mjolnir", vb, c_mj))):
            y = (y0 - 0.255 - j * 0.085) * H
            ax.text(X, y + 9, lbl, fontsize=9.5, color=dim, family=disp, zorder=6)
            ax.text(
                X + BW,
                y + 9,
                fmt_ttft(v) if fmt is None else fmt.format(v),
                fontsize=11,
                color=white,
                weight="bold",
                ha="right",
                family=disp,
                zorder=6,
            )
            bh = 0.014 * H
            for width, face, z in ((BW, (1, 1, 1, 0.10), 5), (max(BW * v / vmax, bh), col, 6)):
                ax.add_patch(
                    FancyBboxPatch(
                        (X, y - bh),
                        width,
                        bh,
                        boxstyle=f"round,pad=0,rounding_size={bh / 2:.1f}",
                        facecolor=face,
                        edgecolor="none",
                        zorder=z,
                    )
                )

    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"[mjolnir] wrote {out}")
    return out


E2E_CHARTS: dict[str, Callable] = {"e2e-decode": chart_e2e_decode, "e2e-prefill": chart_e2e_prefill}
