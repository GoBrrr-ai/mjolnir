"""README formatters: label, ratio, percent and markdown-table helpers."""

from __future__ import annotations

from mjolnir_cli.verify.stages import NOISE_PCT

_GREEN = "#16a34a"
_RED = "#dc2626"
_GREY = "grey"
_MINUS = "−"


def L_label(L: int) -> str:
    return f"{L // 1024}K" if L >= 1024 and L % 1024 == 0 else str(L)


def fmt_us(t: float | None) -> str:
    if t is None:
        return "—"
    return f"{t / 1000:.2f}&nbsp;ms" if t >= 1000 else f"{t:.1f}&nbsp;µs"


def ratio_md(r: float | None, bold_wins: bool = True) -> str:
    """``1.25×`` colored: green = faster beyond the ±NOISE_PCT noise band, grey =
    within noise, red = slower beyond noise (GitHub renders the LaTeX color)."""
    if r is None:
        return "—"
    noise = NOISE_PCT / 100
    color = _GREEN if r >= 1 + noise else (_GREY if r >= 1 - noise else _RED)
    text = f"{r:.2f}\\times"
    if bold_wins and r >= 1 + noise:
        text = f"\\mathbf{{{r:.2f}\\times}}"
    return f"$\\color{{{color}}}{{{text}}}$"


def color(text: str, color: str = "grey") -> str:
    """Render grey LaTeX text, escaping whitespace for math mode."""
    return rf"$\color{{{color}}}\scriptstyle\text{{{text}}}$"


def pct_md(p: float | None, lower_is_better: bool = True) -> str:
    """A colored % delta: green = better beyond the noise band, grey = within
    noise, red = worse beyond noise. Times: lower is better; throughput: higher."""
    if p is None:
        return "—"
    good = -p if lower_is_better else p
    color = _GREEN if good >= NOISE_PCT else (_RED if good <= -NOISE_PCT else _GREY)
    sign = "+" if p > 0 else (_MINUS if p < 0 else "")
    # We need to double-escape the `\\\%` as Github refuses to render the `%` symbol otherwise
    return f"$\\color{{{color}}}{{\\text{{{sign}{abs(p):.1f}\\\\%}}}}$"


def _table(header: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["---"] + ["--:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def ratio_pct(g: float | None, higher_better: bool = True) -> str:
    """The relative change of the measured value, colored by whether it is
    better: ``g`` is the signed gain (positive = Mjolnir better), shown as the
    change of the value itself — +x% throughput, −x% latency (a latency
    reduction, never the reciprocal speedup). Green = better beyond the
    ±E2E_NOISE_PCT band, grey = within it, red = worse beyond it."""
    if g is None:
        return "—"
    from mjolnir_cli import e2e

    noise = e2e.E2E_NOISE_PCT
    color = _GREEN if g >= noise else (_RED if g <= -noise else _GREY)
    shown = g if higher_better else -g
    sign = "+" if shown > 0 else (_MINUS if shown < 0 else "")
    g = shown
    # We need to double-escape the `\\\%` as Github refuses to render the `%` symbol otherwise
    return f"$\\color{{{color}}}{{\\text{{{sign}{abs(g):.1f}\\\\%}}}}$"


def kvcache_label(r: dict) -> str:
    """'official · FlashInfer fp8', 'Mjolnir · fp6 e2m3', ..."""
    kv = str(r.get("kv_dtype") or "auto").replace("fp6_e2m3", "fp6 e2m3")
    if r.get("side") != "official":
        extra = "".join(
            f" + {name}"
            for tag, name in (("LMH8", "fp8 lm_head"), ("VT64K", "64K draft vocab"), ("VT32K", "32K draft vocab"))
            if tag in str(r.get("config", "")).split("_")
        )
        return f"Mjolnir · {kv} KV{extra}"
    be = {"flashinfer": "FlashInfer", "flash_attn": "FlashAttention"}.get(str(r.get("backend") or "").lower())
    return f"official vLLM · {be or r.get('backend') or '?'} · {kv} KV"


def spec_label(cfg: str, side: str | None = None) -> str:
    return {
        "NVFP4": "official vLLM · FlashInfer fp8",
        "NVFP4_MJOLNIR": "Mjolnir · fp8 KV",
        "NVFP4_MJOLNIR_E2M3": "Mjolnir · fp6 KV",
        "NVFP4_MJOLNIR_E2M3_LMH8_VT64K": "Mjolnir · fp6 KV + fp8 lm_head + 64K draft vocab",
    }.get(cfg, cfg)
