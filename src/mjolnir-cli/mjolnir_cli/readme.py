"""README + docs generation — every number, table and chart reference is
computed from the committed raw data.

``mjolnir plot --readme`` renders:

* ``README.md``                  from ``README.md.j2``
* ``docs/kernel/**/*.md``        from the ``*.md.j2`` next to them

Inputs: the latest default run of each ``mjolnir verify`` stage
(``benchmarks/raw/verify/``), the e2e history (``benchmarks/history.jsonl``),
the chart manifest (``assets/benchmarks/manifest.json``, written by
``charts.render_all``) and the vLLM base tag in the Dockerfile. Re-rendering
is idempotent: the output only changes when the raw data does (CI checks it).

Template contract (Jinja2, StrictUndefined — a typo fails loud):

* ``ctx``            — a ``ReadmeContext``: versions, headline numbers, the
                       per-stage markdown tables (``ctx.tables[stage]``),
                       the per-stage findings (``ctx.findings[stage]``) and
                       ``ctx.pending`` (stages without a run yet)
* ``chart(name, alt)`` — a light/dark ``<picture>`` for a chart in the
                       manifest, ``""`` when the chart has no data yet
* ``has(stage)``     — the stage has a saved run
* ``root``           — relative path from the rendered file to the repo root
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import jinja2

from mjolnir_cli.config import DEFAULT_MODEL, IMAGE_REPO, RepoLayout
from mjolnir_cli.verify import results
from mjolnir_cli.verify.stages import ALL_LS, NOISE_PCT, STAGES

_GREEN = "#16a34a"
_RED = "#dc2626"
_GREY = "grey"
_MINUS = "−"


# ── formatting ──────────────────────────────────────────────────────────────


def L_label(L: int) -> str:
    return f"{L // 1024}K" if L >= 1024 and L % 1024 == 0 else str(L)


def fmt_us(t: float | None) -> str:
    if t is None:
        return "—"
    return f"{t / 1000:.2f} ms" if t >= 1000 else f"{t:.1f} µs"


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


def pct_md(p: float | None, lower_is_better: bool = True) -> str:
    """A colored % delta: green = better beyond the noise band, grey = within
    noise, red = worse beyond noise. Times: lower is better; throughput: higher."""
    if p is None:
        return "—"
    good = -p if lower_is_better else p
    color = _GREEN if good >= NOISE_PCT else (_RED if good <= -NOISE_PCT else _GREY)
    sign = "+" if p > 0 else (_MINUS if p < 0 else "")
    return f"$\\color{{{color}}}{{\\text{{{sign}{abs(p):.1f}\\%}}}}$"


def _table(header: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["---"] + ["--:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# ── per-stage tables + findings ─────────────────────────────────────────────


SHORT = {
    "fi": "FlashInfer",
    "fi-hnd": "FlashInfer HND",
    "fa2": "FA2",
    "mjolnir": "Mjolnir e2m3",
    "mjolnir-fp8": "Mjolnir fp8",
}
REF_SHORT = {"fi": "FI", "fi-hnd": "FI-HND", "fa2": "FA2"}


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
            [f"`{s['label']}`", f"`{s['flags']}`" if s.get("flags") else "(default)"]
            + cells
            + [ratio_md(g) if g else "—"]
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
            if re.search(r"interl|flashinfer|HND|strided|descale", lab, re.I)
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
    for r in e2e.compare(off, mj):
        line = [f"{e2e.ctx_label(r['ctx'])}", str(r["conc"])]
        for m in e2e.METRICS:
            a, b, g = r[m.key]
            if a is None or b is None:
                line.append("—")
                continue
            line.append(f"{m.fmt.format(a)} → **{m.fmt.format(b)}** ({ratio_pct(g)})")
        rows.append(line)
    if not rows:
        return None
    head = ["context", "requests"] + [f"{m.label} ({m.unit}, {'↑' if m.higher_better else '↓'})" for m in e2e.METRICS]
    return _table(head, rows, ["--:", "--:"] + ["--:"] * len(e2e.METRICS))


def ratio_pct(g: float | None) -> str:
    """A gain in %, colored like the rest of the README: green = better beyond
    the ±E2E_NOISE_PCT band, grey = within it, red = worse beyond it."""
    if g is None:
        return "—"
    from mjolnir_cli import e2e

    noise = e2e.E2E_NOISE_PCT
    color = _GREEN if g >= noise else (_RED if g <= -noise else _GREY)
    sign = "+" if g > 0 else (_MINUS if g < 0 else "")
    return f"$\\color{{{color}}}{{\\text{{{sign}{abs(g):.1f}\\%}}}}$"


def _findings(stage: str, doc: dict) -> list[str]:
    """Data-driven one-liners for the docs (no hand-written numbers)."""
    out: list[str] = []
    if stage == "spec":
        from mjolnir_cli.verify.spec import best_k

        rows = [r for r in doc.get("rows", []) if r.get("decode_tps")]
        for (cfg, conc), b in sorted(best_k(rows).items()):
            ref = next((r for r in rows if r["config"] == cfg and r["concurrency"] == conc and r["k"] == 3), None)
            vs = (
                f", {(b['decode_tps'] / ref['decode_tps'] - 1) * 100:+.1f}% vs 3 drafts" if ref and b is not ref else ""
            )
            out.append(
                f"{spec_label(cfg)}, {conc} request{'s' if conc > 1 else ''}: best with **{b['k']} draft"
                f"{'s' if b['k'] > 1 else ''}** — {b['decode_tps']:.1f} t/s per request, "
                f"{b['acceptance_length']:.2f} tokens per step{vs}."
            )
        by_cfg: dict = {}
        for r in rows:
            if r["k"] == 3 and r["concurrency"] == 1 and r.get("acceptance_length"):
                by_cfg[r["config"]] = r["acceptance_length"]
        if "NVFP4" in by_cfg:
            for cfg, a in by_cfg.items():
                if cfg != "NVFP4":
                    out.append(
                        f"Acceptance, 3 drafts, 1 request: {spec_label(cfg)} {a:.2f} vs official "
                        f"{by_cfg['NVFP4']:.2f} tokens/step ({(a / by_cfg['NVFP4'] - 1) * 100:+.1f}%) — "
                        "same prompts, temperature 0."
                    )
        return out
    if stage == "kv-cache-size":
        rows = {r["id"]: r for r in doc.get("rows", []) if r.get("kv_tokens")}
        e2m3 = next((r for r in rows.values() if str(r.get("kv_dtype", "")).startswith("fp6")), None)
        for other in rows.values():
            if e2m3 and other is not e2m3:
                gain = e2m3["kv_tokens"] / other["kv_tokens"] - 1
                out.append(
                    f"**{kvcache_label(e2m3)}** holds **{e2m3['kv_tokens']:,} tokens**, "
                    f"{gain * 100:+.0f}% vs {kvcache_label(other)} ({other['kv_tokens']:,})."
                )
        for r in rows.values():
            if r.get("max_concurrency"):
                out.append(
                    f"{kvcache_label(r)}: {r['max_concurrency']:.1f} requests at the full "
                    f"{L_label(r.get('max_model_len') or 0)} context (vLLM's own estimate)."
                )
        return out
    by = results.series_by_id(doc)
    noise = NOISE_PCT / 100
    if stage in ("sweep", "hndsweep"):
        base = by.get("fi")
        for sid in ("mjolnir", "mjolnir-fp8", "fi-hnd", "fa2"):
            if base and sid in by:
                sp = results.speedup(base, by[sid])
                if not sp:
                    continue
                g = results.geomean(sp.values())
                lo, hi = min(sp, key=sp.get), max(sp, key=sp.get)
                beats = sum(1 for v in sp.values() if v >= 1 + noise)
                out.append(
                    f"**{by[sid]['label']}** vs {base['label']}: geomean **{g:.2f}×**, "
                    f"from {sp[lo]:.2f}× (L={L_label(lo)}) to {sp[hi]:.2f}× (L={L_label(hi)}); "
                    f"faster beyond the ±{NOISE_PCT:.0f}% noise band at {beats} of {len(sp)} context lengths."
                )
    if stage in ("ring", "prefetch", "ab", "kvfmt") and "fi" in by:
        variants = [s for s in doc.get("series", []) if s.get("kind") == "t5"]
        if variants:
            Ls = sorted(results.point_map(by["fi"]))
            winners = {}
            for L in Ls:
                best = min(variants, key=lambda s: results.point_map(s).get(L, float("inf")))
                winners[L] = best["label"]
            counts: dict[str, int] = {}
            for w in winners.values():
                counts[w] = counts.get(w, 0) + 1
            top = max(counts, key=counts.get)
            out.append(f"Fastest variant at the most context lengths: **`{top}`** ({counts[top]} of {len(Ls)}).")
            geo = {s["label"]: results.geomean(results.speedup(by["fi"], s).values()) for s in variants}
            geo = {k: v for k, v in geo.items() if v}
            if geo:
                g_top = max(geo, key=geo.get)
                g_low = min(geo, key=geo.get)
                out.append(
                    f"Best geomean vs FlashInfer: **`{g_top}`** ({geo[g_top]:.2f}×); "
                    f"worst: `{g_low}` ({geo[g_low]:.2f}×)."
                )
                spread = geo[g_top] / geo[g_low] - 1
                if spread < noise:
                    out.append(f"All variants sit within {spread * 100:.0f}% of each other — inside the noise band.")
    if stage == "quant":
        by_f: dict[str, list[float]] = {}
        for r in doc.get("rows", []):
            if r.get("rel_l2") is not None:
                by_f.setdefault(r["format"], []).append(r["rel_l2"])
        if "fp8" in by_f:
            fp8 = sum(by_f["fp8"]) / len(by_f["fp8"])
            for f, v in sorted(by_f.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
                m = sum(v) / len(v)
                out.append(f"`{f}`: mean relative L2 error **{m:.3f}** ({m / fp8:.1f}× fp8's {fp8:.3f}).")
    if stage == "pdl":
        ds = [r["delta_pct"] for r in doc.get("rows", []) if r.get("delta_pct") is not None]
        if ds:
            inside = sum(1 for d in ds if abs(d) < NOISE_PCT)
            out.append(
                f"{inside} of {len(ds)} measurements move by less than the ±{NOISE_PCT:.0f}% noise band "
                f"(range {min(ds):+.1f}% … {max(ds):+.1f}%)."
            )
    if stage == "fp6prof":
        profs = [p for p in doc.get("profiles", []) if p.get("ok")]
        by_lab: dict[str, list[tuple[float, float, float]]] = {}
        by_L: dict[int, dict[str, dict]] = {}
        for p in profs:
            by_L.setdefault(p["L"], {})[p["label"]] = p
        for L, ps in by_L.items():
            labs = list(ps)
            b = ps[labs[0]]
            for lab in labs[1:]:
                p = ps[lab]
                t = p["metrics"]["gpu__time_duration.sum"] / b["metrics"]["gpu__time_duration.sum"]
                ideal = p["kv_bits"] / b["kv_bits"]
                by_lab.setdefault(lab, []).append((t, ideal, L))
        for lab, xs in by_lab.items():
            t = sum(x[0] for x in xs) / len(xs)
            ideal = xs[0][1]
            realized = (1 - t) / (1 - ideal) if ideal < 1 else 0
            out.append(
                f"`{lab}`: kernel time **{t:.2f}×** the baseline for {ideal:.2f}× the bytes — "
                f"{realized * 100:.0f}% of the byte saving becomes speed (mean over L)."
            )
    if stage == "correctness":
        cs = doc.get("cases", [])
        ok = sum(1 for c in cs if c.get("passed"))
        out.append(
            f"**{ok} / {len(cs)}** checks pass (kernel output and LSE vs the bit-faithful emulator, "
            "poisoned unused pages must stay NaN-free)."
        )
    return out


KVCACHE_CTXS = (4096, 8192, 16384, 32768, 65536, 131072)


def kvcache_label(r: dict) -> str:
    """'official · FlashInfer fp8', 'Mjolnir · fp6 e2m3', ..."""
    kv = str(r.get("kv_dtype") or "auto").replace("fp6_e2m3", "fp6 e2m3")
    if r.get("side") != "official":
        return f"Mjolnir · {kv} KV"
    be = {"flashinfer": "FlashInfer", "flash_attn": "FlashAttention"}.get(str(r.get("backend") or "").lower())
    return f"official vLLM · {be or r.get('backend') or '?'} · {kv} KV"


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
    return _table(head, out, ["---", "---"] + ["--:"] * (len(head) - 2))


def spec_label(cfg: str, side: str | None = None) -> str:
    return {
        "NVFP4": "official vLLM · FlashInfer fp8",
        "NVFP4_MJOLNIR": "Mjolnir · fp8 KV",
        "NVFP4_MJOLNIR_E2M3": "Mjolnir · fp6 KV",
    }.get(cfg, cfg)


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


TABLES = {
    "sweep": lambda d: _series_time_table(d, ("fi", "fa2")),
    "hndsweep": lambda d: _series_time_table(d, ("fi", "fi-hnd")),
    "kvfmt": lambda d: _variant_table(d, "fi"),
    "ring": lambda d: _variant_table(d, "fi"),
    "prefetch": lambda d: _variant_table(d, "fi"),
    "ab": lambda d: _variant_table(d, "fi"),
    "quant": _quant_table,
    "pdl": _pdl_table,
    "fp6prof": _fp6prof_table,
    "correctness": _correctness_table,
    "kv-cache-size": _kvcache_table,
    "spec": _spec_table,
}


# ── context ─────────────────────────────────────────────────────────────────


@dataclass
class ReadmeContext:
    vllm_version: str = ""
    model: str = DEFAULT_MODEL
    image: str = ""
    kernel_version: str = ""
    cli_version: str = ""
    ls: str = " ".join(L_label(L) for L in ALL_LS)
    noise_pct: str = f"{NOISE_PCT:.0f}"
    e2e_noise_pct: str = field(default_factory=lambda: f"{__import__('mjolnir_cli.e2e').e2e.E2E_NOISE_PCT:.0f}")
    # headline (sweep)
    has_sweep: bool = False
    geo_vs_fi: str = ""
    geo_vs_fi_fp8: str = ""
    geo_vs_fa2: str = ""
    min_vs_fi: str = ""
    max_vs_fi: str = ""
    min_vs_fi_L: str = ""
    max_vs_fi_L: str = ""
    long_vs_fi: str = ""  # at the largest L
    long_L: str = ""
    beats_fi: str = ""  # "8 of 8"
    us_64k_mjolnir: str = ""
    us_64k_fi: str = ""
    us_64k_fa2: str = ""
    correctness: str = ""  # "96 / 96"
    sweep_run: str = ""
    tables: dict = field(default_factory=dict)
    findings: dict = field(default_factory=dict)
    runs: dict = field(default_factory=dict)  # stage -> run_id
    pending: list = field(default_factory=list)
    # official vLLM image vs the Mjolnir image (mjolnir_cli.e2e)
    vs_vllm: str | None = None  # the comparison table
    vs_vllm_hero: dict = field(default_factory=dict)  # c=1 headline gains, formatted
    vs_vllm_official: str = ""
    vs_vllm_mjolnir: str = ""
    vs_vllm_spec: str = ""  # speculative decoding both sides ran with, e.g. "MTP, 3 speculative tokens"


def _vllm_base_version(layout: RepoLayout) -> str:
    if layout.repo_root is None:
        return ""
    dockerfile = layout.repo_root / "docker" / "vllm-thor" / "Dockerfile"
    if dockerfile.exists():
        m = re.search(r"vllm/vllm-openai:v(\d+\.\d+\.\d+)", dockerfile.read_text())
        if m:
            return m.group(1)
    return ""


def _version_from(path: Path) -> str:
    try:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', path.read_text())
        return m.group(1) if m else ""
    except OSError:
        return ""


def collect_context(layout: RepoLayout, data: dict | None = None) -> ReadmeContext:
    from mjolnir_cli.charts import load_verify

    data = data if data is not None else load_verify(layout)
    ctx = ReadmeContext(vllm_version=_vllm_base_version(layout))
    ctx.image = f"{IMAGE_REPO}:v{ctx.vllm_version}" if ctx.vllm_version else IMAGE_REPO
    if layout.repo_root is not None:
        ctx.kernel_version = _version_from(layout.repo_root / "src" / "mjolnir" / "mjolnir" / "__init__.py")
        ctx.cli_version = _version_from(layout.repo_root / "src" / "mjolnir-cli" / "mjolnir_cli" / "__init__.py")
    for stage, doc in data.items():
        ctx.runs[stage] = doc["run_id"]
        if stage in TABLES:
            try:
                ctx.tables[stage] = TABLES[stage](doc)
            except (KeyError, ZeroDivisionError, TypeError, ValueError) as e:
                print(f"[mjolnir] table {stage} skipped: {e!r}")
        ctx.findings[stage] = _findings(stage, doc)
        if stage in ("kvfmt", "ab"):
            w = _wall_table(doc)
            if w:
                ctx.tables[f"{stage}_wall"] = w
    ctx.pending = [s for s in STAGES if s not in data and s not in ("timing", "vllm")]

    sweep = data.get("sweep")
    if sweep:
        by = results.series_by_id(sweep)
        ours = by.get("mjolnir") or by.get("mjolnir-fp8")
        if ours and "fi" in by:
            ctx.has_sweep = True
            ctx.sweep_run = sweep["run_id"]
            sp = results.speedup(by["fi"], ours)
            ctx.geo_vs_fi = f"{results.geomean(sp.values()):.2f}"
            lo, hi = min(sp, key=sp.get), max(sp, key=sp.get)
            ctx.min_vs_fi, ctx.min_vs_fi_L = f"{sp[lo]:.2f}", L_label(lo)
            ctx.max_vs_fi, ctx.max_vs_fi_L = f"{sp[hi]:.2f}", L_label(hi)
            last = max(sp)
            ctx.long_vs_fi, ctx.long_L = f"{sp[last]:.2f}", L_label(last)
            ctx.beats_fi = f"{sum(1 for v in sp.values() if v >= 1 + NOISE_PCT / 100)} of {len(sp)}"
            if "mjolnir-fp8" in by:
                ctx.geo_vs_fi_fp8 = f"{results.geomean(results.speedup(by['fi'], by['mjolnir-fp8']).values()):.2f}"
            if "fa2" in by:
                ctx.geo_vs_fa2 = f"{results.geomean(results.speedup(by['fa2'], ours).values()):.2f}"
            pm = results.point_map(ours)
            if 65536 in pm:
                ctx.us_64k_mjolnir = fmt_us(pm[65536])
                ctx.us_64k_fi = fmt_us(results.point_map(by["fi"]).get(65536))
                if "fa2" in by:
                    ctx.us_64k_fa2 = fmt_us(results.point_map(by["fa2"]).get(65536))
    corr = data.get("correctness")
    if corr:
        cs = corr.get("cases", [])
        ctx.correctness = f"{sum(1 for c in cs if c.get('passed'))} / {len(cs)}"
    from mjolnir_cli import e2e

    off, mj = e2e.latest_pair(layout.history_file)
    if off and mj:
        ctx.vs_vllm = _vs_vllm_table(off, mj)
        ctx.vs_vllm_official, ctx.vs_vllm_mjolnir = e2e.side_label(off), e2e.side_label(mj)
        ctx.vs_vllm_spec = e2e.spec_label(off, mj)
        h = e2e.hero(off, mj)
        if h:
            ctx.vs_vllm_hero = {
                "ctx": e2e.ctx_label(h["ctx"]),
                "tg_tps": e2e.pct(h["tg_tps"][2]),
                "pp_tps": e2e.pct(h["pp_tps"][2]),
                "ttft_ms": f"{h['ttft_ms'][0] / 1000:.2f} s → {h['ttft_ms'][1] / 1000:.2f} s"
                if h["ttft_ms"][2]
                else "—",
                "itl_ms": f"{h['itl_ms'][0]:.1f} ms → {h['itl_ms'][1]:.1f} ms" if h["itl_ms"][2] else "—",
            }
    return ctx


# ── rendering ───────────────────────────────────────────────────────────────


def _manifest(layout: RepoLayout) -> dict:
    p = layout.charts_dir / "manifest.json"
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _env(layout: RepoLayout, ctx: ReadmeContext, out_path: Path) -> jinja2.Environment:
    import os

    manifest = _manifest(layout)
    root_rel = os.path.relpath(layout.repo_root or out_path.parent, out_path.parent).replace(os.sep, "/")
    prefix = "" if root_rel == "." else root_rel + "/"

    def chart(name: str, alt: str = "", width: str = "100%") -> str:
        m = manifest.get(name)
        if not m:
            return ""
        light, dark = prefix + m["light"], prefix + m.get("dark", m["light"])
        return (
            "<picture>\n"
            f'  <source media="(prefers-color-scheme: dark)" srcset="{dark}">\n'
            f'  <img src="{light}" alt="{alt or name}" width="{width}">\n'
            "</picture>"
        )

    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True, autoescape=False)
    env.globals.update(
        ctx=ctx,
        chart=chart,
        has=lambda stage: stage in ctx.runs,
        root=prefix,
        stages=STAGES,
    )
    return env


def render_file(layout: RepoLayout, tpl: Path, out: Path, ctx: ReadmeContext) -> Path:
    env = _env(layout, ctx, out)
    out.write_text(env.from_string(tpl.read_text()).render())
    return out


def render_readme(layout: RepoLayout, template: Path | None = None, out: Path | None = None) -> list[Path]:
    """Render README.md + every docs ``*.md.j2`` (checkout only)."""
    root = layout.require_repo("rendering the README")
    tpl = Path(template) if template else root / "README.md.j2"
    dest = Path(out) if out else root / "README.md"
    if not tpl.exists():
        raise FileNotFoundError(f"README template not found: {tpl}")
    ctx = collect_context(layout)
    written = [render_file(layout, tpl, dest, ctx)]
    for t in sorted((root / "docs").rglob("*.md.j2")):
        written.append(render_file(layout, t, t.with_suffix(""), ctx))
    for p in written:
        print(f"[mjolnir] wrote {p}")
    return written
