"""Terminal rendering of verify results (rich tables)."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.table import Table

from mjolnir_cli.verify import results
from mjolnir_cli.verify.stages import NOISE_PCT, STAGES

console = Console()


def fmt_us(t: float | None) -> str:
    if t is None:
        return "—"
    return f"{t / 1000:.2f} ms" if t >= 1000 else f"{t:.1f} µs"


def ratio_cell(r: float | None) -> str:
    if r is None:
        return ""
    noise = NOISE_PCT / 100
    color = "green" if r >= 1 + noise else ("yellow" if r >= 1 - noise else "red")
    return f"[{color}]{r:.2f}×[/{color}]"


def _bases(doc: dict) -> list[dict]:
    """Reference series for the ratio columns: the baselines present."""
    by = results.series_by_id(doc)
    return [by[i] for i in ("fi", "fi-hnd", "fa2") if i in by]


def series_table(doc: dict) -> Table:
    ser = doc.get("series", [])
    Ls = sorted({p["L"] for s in ser for p in s.get("points", [])})
    bases = _bases(doc)
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, title=f"{doc['stage']} · {doc['run_id']}", title_justify="left")
    t.add_column("series", style="bold")
    for L in Ls:
        t.add_column(f"L={L:,}", justify="right")
    best: dict[int, float] = {}
    for s in ser:
        for L, v in results.point_map(s).items():
            best[L] = min(best.get(L, v), v)
    for s in ser:
        pm = results.point_map(s)
        cells = []
        for L in Ls:
            v = pm.get(L)
            if v is None:
                cells.append("—")
                continue
            txt = fmt_us(v)
            if v == best.get(L):
                txt = f"[bold green]{txt}[/bold green]"
            ratios = []
            for b in bases:
                if b is s:
                    continue
                bv = results.point_map(b).get(L)
                if bv:
                    ratios.append(ratio_cell(bv / v))
            if ratios and s.get("kind") != "fi" and s.get("kind") != "fa2":
                txt += " " + "/".join(ratios)
            cells.append(txt)
        t.add_row(s["label"], *cells)
    if bases:
        t.caption = (
            "ratios = "
            + " / ".join(f"{b['label']} time ÷ this time" for b in bases)
            + (f"  (>1 faster · ±{NOISE_PCT:.0f}% is noise)")
        )
    return t


def rows_table(doc: dict) -> Table | None:
    rows = doc.get("rows") or []
    if not rows:
        return None

    # scalar columns (and short lists like per-position acceptance); nested
    # summaries (profile graphs, kernel tables) and paths stay in the JSON
    def shown(k, v) -> bool:
        if k in ("ok", "log", "trace", "image", "ttft_list_ms"):
            return False
        if isinstance(v, dict):
            return False
        return not (isinstance(v, list) and (len(v) > 8 or any(isinstance(x, (list, dict)) for x in v)))

    keys = [k for k in rows[0] if shown(k, rows[0][k])]
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, title=f"{doc['stage']} rows", title_justify="left")
    for k in keys:
        t.add_column(k, justify="right" if isinstance(rows[0][k], (int, float)) else "left")
    for r in rows:
        t.add_row(*[_fmt(r.get(k)) for k in keys])
    return t


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 10 else f"{v:,.1f}"
    return str(v)


def cases_summary(doc: dict) -> str:
    cs = doc.get("cases", [])
    bad = [c for c in cs if not c.get("passed")]
    if not bad:
        return f"[bold green]✅ all {len(cs)} checks passed[/bold green]"
    lines = [f"[bold red]❌ {len(bad)} failed[/bold red], [green]{len(cs) - len(bad)} passed[/green]"]
    for c in bad:
        lines.append(f"   [red]❌ {c.get('q', '')} {c['label']}[/red]  [dim]{c.get('log', '')}[/dim]")
    return "\n".join(lines)


def show(doc: dict) -> None:
    st = STAGES.get(doc["stage"])
    console.print(
        f"\n[bold cyan]━━ {st.title if st else doc['stage']}[/bold cyan]  "
        f"[dim]{doc['run_id']} · {doc.get('image', '')} · kernel {doc.get('kernel', {}).get('version', '?')}"
        f"{' (' + doc['kernel']['git'] + ')' if doc.get('kernel', {}).get('git') else ''}"
        f" · gate {doc.get('gate', {}).get('server', '?')}[/dim]"
    )
    if doc.get("series"):
        console.print(series_table(doc))
    t = rows_table(doc)
    if t is not None:
        console.print(t)
    if doc.get("profiles"):
        prof_table(doc)
    if doc.get("cases"):
        console.print(cases_summary(doc))
    if doc.get("summary"):
        for k, v in doc["summary"].items():
            if isinstance(v, dict) and "geomean" in v:
                console.print(
                    f"  [bold]{k}[/bold]: geomean {v['geomean']:.2f}×  (min {v['min']:.2f}×, max {v['max']:.2f}×)"
                )


def prof_table(doc: dict) -> None:
    """fp6prof: bytes vs time per KV format, per L (vs the first variant)."""
    profs = [p for p in doc.get("profiles", []) if p.get("ok")]
    if not profs:
        return
    by_L: dict[int, list[dict]] = {}
    for p in profs:
        by_L.setdefault(p["L"], []).append(p)
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, title="fp6prof: time vs bytes (ratios vs the first variant)")
    for c in ("L", "variant", "bits", "µs", "time ratio", "L2 sector ratio", "ideal", "L2 B/request", "top stall"):
        t.add_column(c, justify="right" if c not in ("variant", "top stall") else "left")
    for L, ps in sorted(by_L.items()):
        b = ps[0]
        bt = b["metrics"].get("gpu__time_duration.sum")
        bs = b["metrics"].get("lts__t_sectors_op_read.sum") or b["metrics"].get("lts__t_sectors.sum")
        for p in ps:
            m = p["metrics"]
            tm = m.get("gpu__time_duration.sum")
            sec = m.get("lts__t_sectors_op_read.sum") or m.get("lts__t_sectors.sum")
            req = m.get("lts__t_requests_srcunit_tex_op_read.sum")
            stall = max(p.get("stalls", {}).items(), key=lambda kv: kv[1], default=("", 0))
            t.add_row(
                f"{L:,}",
                p["label"],
                f"{p['kv_bits']:g}",
                fmt_us(tm / 1e3 if tm else None),
                f"{tm / bt:.2f}" if tm and bt else "—",
                f"{sec / bs:.2f}" if sec and bs else "—",
                f"{p['kv_bits'] / b['kv_bits']:.2f}",
                f"{sec * 32 / req:.0f}" if sec and req else "—",
                f"{stall[0]} {stall[1]:.1f}" if stall[0] else "",
            )
    console.print(t)
