"""README table builders + markdown formatting (split out of readme.py).

readme.py imports these; TABLES and the findings/rendering stay there.
"""

from __future__ import annotations

import re

from mjolnir_cli.verify import results
from mjolnir_cli.verify.stages import NOISE_PCT

_GREEN = "#16a34a"
_RED = "#dc2626"
_GREY = "grey"
_MINUS = "\u2212"


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


SHORT = {
    "fi": "FlashInfer",
    "fi-hnd": "FlashInfer HND",
    "fa2": "FA2",
    "fa4": "FA4 hd256",
    "mjolnir": "Mjolnir e2m3",
    "mjolnir-fp8": "Mjolnir fp8",
}


REF_SHORT = {"fi": "FI", "fi-hnd": "FI-HND", "fa2": "FA2", "fa4": "FA4"}


def _short(s: dict) -> str:
    return SHORT.get(s["id"], s["label"])


def _series_time_table(doc: dict, ref_ids: tuple[str, ...] = ("fi",)) -> str:
    """Rows = L; columns = each series' time, then Mjolnir's speedup vs each ref."""
    ser = doc.get("series", [])
    by = results.series_by_id(doc)
    refs = [by[r] for r in ref_ids if r in by]
    ours = [s for s in ser if s.get("kind") == "t5"]
    Ls = sorted({p["L"] for s in ser for p in s.get("points", [])})
    header = ["L"] + [_short(s) for s in ser]
    pairs = [(o, r) for r in refs for o in ours]
    header += [f"{_short(o).replace('Mjolnir ', '')} vs {REF_SHORT.get(r['id'], r['label'])}" for o, r in pairs]
    rows = []
    for L in Ls:
        times = {s["id"]: results.point_map(s).get(L) for s in ser}
        best = min((t for t in times.values() if t), default=None)
        r = [f"**{L_label(L)}**"]
        for s in ser:
            t = times[s["id"]]
            cell = fmt_us(t)
            r.append(f"**{cell}**" if t is not None and t == best else cell)
        for o, ref in pairs:
            rt, t = times.get(ref["id"]), times.get(o["id"])
            r.append(ratio_md(rt / t) if rt and t else "—")
        rows.append(r)
    geo = ["geomean"] + [""] * len(ser)
    for o, ref in pairs:
        g = results.geomean(results.speedup(ref, o).values())
        geo.append(ratio_md(g) if g else "—")
    rows.append(geo)
    return _table(header, rows)


def _variant_table(doc: dict, ref_id: str = "fi") -> str:
    """Rows = variants; columns = speedup vs the reference at each L + geomean."""
    by = results.series_by_id(doc)
    if ref_id not in by:
        return ""
    ref = by[ref_id]
    Ls = sorted(results.point_map(ref))
    header = ["variant", "flags"] + [L_label(L) for L in Ls] + ["geomean"]
    rows = []
    best_per_L: dict[int, float] = {}
    variants = [s for s in doc.get("series", []) if s.get("kind") == "t5"]
    for s in variants:
        for L, v in results.speedup(ref, s).items():
            best_per_L[L] = max(best_per_L.get(L, 0), v)
    for s in variants:
        sp = results.speedup(ref, s)
        cells = []
        for L in Ls:
            v = sp.get(L)
            c = ratio_md(v, bold_wins=False) if v else "—"
            cells.append(f"**{c}**" if v and v == best_per_L.get(L) else c)
        g = results.geomean(sp.values())
        rows.append(
            [f"`{s['label']}`", f"`{s['flags']}`" if s.get("flags") else "(default)", *cells, ratio_md(g) if g else "—"]
        )
    return _table(header, rows, ["---", "---"] + ["--:"] * (len(Ls) + 1))


def _wall_table(doc: dict) -> str | None:
    rows = [r for r in doc.get("rows", []) if r.get("median_us")]
    if not rows:
        return None
    labels = {s["id"]: s["label"] for s in doc.get("series", [])}
    sids = list(dict.fromkeys(r["series"] for r in rows))
    Ls = sorted({r["L"] for r in rows})
    base = sids[0]
    header = ["L"] + [labels.get(s, s) for s in sids]
    out = []
    for L in Ls:
        m = {r["series"]: r["median_us"] for r in rows if r["L"] == L}
        cells = [f"**{L_label(L)}**"]
        for s in sids:
            v = m.get(s)
            if v is None:
                cells.append("—")
            elif s == base or not m.get(base):
                cells.append(fmt_us(v))
            else:
                cells.append(f"{fmt_us(v)} {pct_md((v - m[base]) / m[base] * 100)}")
        out.append(cells)
    return _table(header, out)


def _quant_table(doc: dict) -> str:
    by: dict[str, dict[int, float]] = {}
    bits = {}
    for r in doc.get("rows", []):
        if r.get("rel_l2") is not None:
            by.setdefault(r["format"], {})[r["L"]] = r["rel_l2"]
            bits[r["format"]] = r.get("bits")
    Ls = sorted({L for v in by.values() for L in v})
    order = sorted(by, key=lambda f: (-(bits.get(f) or 0), f))
    header = ["format", "bits"] + [L_label(L) for L in Ls] + ["mean"]
    rows = []
    for f in order:
        vals = by[f]
        rows.append(
            [f"`{f}`", f"{bits.get(f):g}" if bits.get(f) else "—"]
            + [f"{vals[L]:.4f}" if L in vals else "—" for L in Ls]
            + [f"**{sum(vals.values()) / len(vals):.4f}**"]
        )
    return _table(header, rows, ["---", "--:"] + ["--:"] * (len(Ls) + 1))


def _pdl_table(doc: dict) -> str:
    rows = [r for r in doc.get("rows", []) if r.get("off_us")]
    Ls = sorted({r["L"] for r in rows})
    keys = sorted({(r["q"], r["batches"]) for r in rows})
    header = ["Q", "batch"] + [L_label(L) for L in Ls]
    out = []
    for q, b in keys:
        m = {r["L"]: r for r in rows if r["q"] == q and r["batches"] == b}
        out.append([q, str(b)] + [pct_md(m[L]["delta_pct"]) if L in m else "—" for L in Ls])
    return _table(header, out, ["---", "--:"] + ["--:"] * len(Ls))


def _fp6prof_table(doc: dict) -> str:
    profs = [p for p in doc.get("profiles", []) if p.get("ok")]
    by_L: dict[int, list[dict]] = {}
    for p in profs:
        by_L.setdefault(p["L"], []).append(p)
    header = [
        "L",
        "variant",
        "bits",
        "kernel time",
        "time ratio",
        "L2 bytes ratio",
        "ideal",
        "bytes / L2 request",
        "top stall (cycles/instr)",
    ]
    rows = []
    for L, ps in sorted(by_L.items()):
        b = ps[0]
        bt = b["metrics"].get("gpu__time_duration.sum")
        bs = b["metrics"].get("lts__t_sectors_op_read.sum") or b["metrics"].get("lts__t_sectors.sum")
        for p in ps:
            m = p["metrics"]
            t = m.get("gpu__time_duration.sum")
            sec = m.get("lts__t_sectors_op_read.sum") or m.get("lts__t_sectors.sum")
            req = m.get("lts__t_requests_srcunit_tex_op_read.sum")
            st = max(p.get("stalls", {}).items(), key=lambda kv: kv[1], default=None)
            rows.append(
                [
                    f"**{L_label(L)}**",
                    f"`{p['label']}`",
                    f"{p['kv_bits']:g}",
                    fmt_us(t / 1e3 if t else None),
                    f"{t / bt:.2f}" if t and bt else "—",
                    f"{sec / bs:.2f}" if sec and bs else "—",
                    f"{p['kv_bits'] / b['kv_bits']:.2f}",
                    f"{sec * 32 / req:.0f} B" if sec and req else "—",
                    f"{st[0]} {st[1]:.1f}" if st else "—",
                ]
            )
    return _table(header, rows, ["---", "---"] + ["--:"] * 6 + ["---"])


def _correctness_table(doc: dict) -> str:
    cs = doc.get("cases", [])
    groups: dict[str, list[dict]] = {}
    for c in cs:
        lab = c["label"]
        g = (
            "default config"
            if lab.startswith("default:")
            else "KV formats"
            if re.search(r"e2m3|e3m2|e2m1|nvfp4|from float", lab)
            else "KV layouts / strides"
            if re.search(r"interl|flashinfer|HND|strided|descale", lab, re.IGNORECASE)
            else "launch options"
            if re.search(r"PDL|prefetch|evict", lab)
            else "scheduling / paging"
            if re.search(r"page|CTAs|chunks|padded|splits|contiguous", lab)
            else "shapes / batches"
        )
        groups.setdefault(g, []).append(c)
    rows = []
    for g, items in groups.items():
        ok = sum(1 for c in items if c.get("passed"))
        mark = "✅" if ok == len(items) else "❌"
        qs = sorted({c.get("q", "") for c in items})
        rows.append([g, f"{mark} {ok}/{len(items)}", ", ".join(q for q in qs if q)])
    return _table(["case group", "passed", "Q dtypes"], rows, ["---", "--:", "---"])


def _vs_vllm_table(off: dict, mj: dict) -> str | None:
    """Official vLLM image vs the Mjolnir image, every shared cell: each
    metric as 'official → mjolnir (gain)'; gain is positive when Mjolnir is
    better (faster decode/prefill, lower TTFT/ITL)."""
    from mjolnir_cli import e2e

    rows = []
    for r in e2e.compare(off, mj, every=True):  # a cell one side lacks shows as "not measured", not dropped
        line = [f"{e2e.ctx_label(r['ctx'])}", str(r["conc"])]
        for m in e2e.METRICS:
            a, b, g = r[m.key]
            if a is None and b is None:
                line.append("—")
                continue
            if a is None or b is None:
                av = m.fmt.format(a) if a is not None else "*n/a*"
                bv = f"**{m.fmt.format(b)}**" if b is not None else "*n/a*"
                line.append(f"{av} → {bv}")
                continue
            line.append(f"{m.fmt.format(a)} → **{m.fmt.format(b)}** {ratio_pct(g, m.higher_better)}")
        rows.append(line)
    if not rows:
        return None
    head = ["context", "requests"] + [
        f"{m.label} ({m.unit}, {'↑' if m.higher_better else '↓'}) "
        + color("throughput change" if m.higher_better else "latency change")
        for m in e2e.METRICS
    ]
    return _table(head, rows, ["--:", "--:"] + ["--:"] * len(e2e.METRICS))


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


KVCACHE_CTXS = (4096, 8192, 16384, 32768, 65536, 131072)


def kvcache_label(r: dict) -> str:
    """'official · FlashInfer fp8', 'Mjolnir · fp6 e2m3', ..."""
    kv = str(r.get("kv_dtype") or "auto").replace("fp6_e2m3", "fp6 e2m3")
    if r.get("side") != "official":
        extra = "".join(
            f" + {name}"
            for tag, name in (("LMH8", "fp8 lm_head"), ("VT64K", "64K draft vocab"))
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


def _kvcache_table(doc: dict) -> str:
    rows = [r for r in doc.get("rows", []) if r.get("kv_tokens")]
    pp, tg = doc.get("params", {}).get("e2e_pp", 2048), doc.get("params", {}).get("e2e_tg", 128)
    base = rows[0]["kv_tokens"] if rows else None
    out = []
    for r in rows:
        fit = [str(r["kv_tokens"] // (c + pp + tg)) for c in KVCACHE_CTXS]
        out.append(
            [
                f"`{r['config']}`",
                kvcache_label(r),
                f"{r.get('kv_gib') or 0:.1f} GiB",
                f"{r['kv_tokens']:,}",
                f"{r['kv_tokens'] / base:.2f}×" if base else "—",
                f"{r.get('max_concurrency') or 0:.1f}× @ {L_label(r.get('max_model_len') or 0)}",
                *fit,
            ]
        )
    head = ["config", "serving", "KV memory", "KV tokens", "vs first", "max concurrency"]
    head += [f"fits @ {L_label(c)}" for c in KVCACHE_CTXS]
    mem = [k for k in ("load_gib", "weights_gib", "activation_gib", "non_torch_gib") if any(r.get(k) for r in rows)]
    if mem:  # where the rest of the budget went, when the logs carried it
        names = {"load_gib": "model load", "weights_gib": "weights", "activation_gib": "activation peak",
                 "non_torch_gib": "non-torch"}  # fmt: skip
        head += [f"{names[k]} {color('GiB')}" for k in mem]
        for r, line in zip(rows, out, strict=True):
            line += [f"{r[k]:.1f}" if r.get(k) else "—" for k in mem]
    return _table(head, out, ["---", "---"] + ["--:"] * (len(head) - 2))


def _spec_table(doc: dict) -> str:
    from mjolnir_cli.verify.spec import best_k

    rows = [r for r in doc.get("rows", []) if r.get("decode_tps")]
    best = {(cfg, c): r["k"] for (cfg, c), r in best_k(rows).items()}
    kmax = max((len(r.get("per_pos") or []) for r in rows), default=0)
    out = []
    for r in sorted(rows, key=lambda r: (r["config"], r["concurrency"], r["k"])):
        star = " **★**" if best.get((r["config"], r["concurrency"])) == r["k"] else ""
        pos = list(r.get("per_pos") or []) + [None] * (kmax - len(r.get("per_pos") or []))
        out.append(
            [
                spec_label(r["config"]),
                str(r["concurrency"]),
                f"{r['k']}{star}",
                f"{r.get('acceptance_length') or 0:.2f}",
                *(f"{p:.2f}" if p is not None else "" for p in pos),
                f"{r['decode_tps']:.1f}",
                f"{r.get('agg_tps') or 0:.1f}",
                f"{(r.get('ttft_ms') or 0) / 1000:.2f} s",
            ]
        )
    head = ["config", "c", "drafts", "tokens/step", *(f"accept @{i + 1}" for i in range(kmax))]
    head += ["decode t/s/req", "aggregate t/s", "TTFT"]
    return _table(head, out, ["---"] + ["--:"] * (len(head) - 1))
