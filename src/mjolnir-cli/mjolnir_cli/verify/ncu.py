"""Parse ``ncu --print-summary per-kernel`` logs into per-leg points."""

from __future__ import annotations

import re
from pathlib import Path

METRICS = "gpu__time_duration.sum,lts__t_sectors.sum,sm__cycles_active.sum,sm__ctas_launched.sum"

_UNIT_US = {
    "ns": 1e-3,
    "nsecond": 1e-3,
    "us": 1.0,
    "usecond": 1.0,
    "ms": 1e3,
    "msecond": 1e3,
    "s": 1e6,
    "second": 1e6,
}
_SUMMARY_METRICS = ("gpu__time_duration.sum", "lts__t_sectors.sum", "sm__ctas_launched.sum", "sm__cycles_active.sum")


def _num(x: str) -> float:
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return float("nan")


def parse_summary(text: str) -> list[dict]:
    """Kernel rows from an ncu per-kernel summary: name, invocations,
    avg/min µs, L2 sectors, CTAs."""
    rows: list[dict] = []
    cur: dict | None = None
    for line in text.splitlines():
        if "Invocations" in line and "Device" in line:
            m = re.search(r"Invocations\s+(\d+)", line)
            cur = {"header": line.strip(), "name": line.strip().split(" (")[0], "inv": int(m.group(1)) if m else 0}
            rows.append(cur)
            continue
        if cur is None:
            continue
        p = line.split()
        if len(p) >= 5 and p[0] in _SUMMARY_METRICS:
            unit, mn, avg = p[1], _num(p[2]), _num(p[-1])
            if p[0] == "gpu__time_duration.sum":
                k = _UNIT_US.get(unit, 1.0)
                cur["avg_us"], cur["min_us"] = avg * k, mn * k
            elif p[0] == "lts__t_sectors.sum":
                cur["sectors"] = avg
            elif p[0] == "sm__ctas_launched.sum":
                cur["ctas"] = avg
            elif p[0] == "sm__cycles_active.sum":
                cur["cycles"] = avg
    return [r for r in rows if "avg_us" in r]


def point(rows: list[dict], main: tuple[str, ...], merge: tuple[str, ...], L: int, kv_bytes: float) -> dict | None:
    """One L point of a leg: decode (main) + merge kernel time, L2 sectors,
    effective KV bandwidth (the leg's real KV bytes / total time)."""
    mains = [r for r in rows if any(p in r["header"] for p in main)]
    merges = [r for r in rows if any(p in r["header"] for p in merge) and r not in mains]
    if not mains:
        return None
    main_us = sum(r["avg_us"] for r in mains)
    merge_us = sum(r["avg_us"] for r in merges)
    total = main_us + merge_us
    return {
        "L": L,
        "total_us": round(total, 3),
        "main_us": round(main_us, 3),
        "merge_us": round(merge_us, 3),
        "min_us": round(sum(r.get("min_us", r["avg_us"]) for r in mains + merges), 3),
        "l2_sectors": sum(r.get("sectors", 0.0) for r in mains + merges),
        "ctas": int(mains[0].get("ctas", 0) or 0),
        "invocations": int(mains[0].get("inv", 0)),
        "gbps": round(kv_bytes / (total * 1e3), 2) if total > 0 else 0.0,
        "kernels": [
            {"name": r["name"][:120], "avg_us": round(r["avg_us"], 3), "inv": r["inv"]} for r in mains + merges
        ],
    }


def parse_file(path: Path, main, merge, L: int, kv_bytes: float) -> dict | None:
    try:
        return point(parse_summary(Path(path).read_text(errors="replace")), main, merge, L, kv_bytes)
    except OSError:
        return None


# ── full-profile (fp6prof) CSV pages ─────────────────────────────────────────


def parse_raw_csv(text: str) -> tuple[dict[str, float], dict[str, str], int | None]:
    """``ncu --import x --page raw --csv`` → (fastest launch's metrics,
    metric → unit, that launch's ID).

    Identical launches of the decode kernel flip up to ~25% in kernel time
    under replay (CTA-residency modes; fp6prof 20261017-210232: 299 vs
    372 µs on the same 40-CTA grid, at every L and format), so a mean over
    ``count`` launches mixes the modes and the ratios between legs land
    anywhere in between. The fastest launch is one self-consistent
    snapshot — time, stalls and bytes from the same replay — and matches
    what the wall-clock harness medians see."""
    import csv
    import io

    rows = list(csv.reader(io.StringIO(text)))
    hi = next((i for i, r in enumerate(rows) if "Kernel Name" in r or "ID" in r), None)
    if hi is None or len(rows) < hi + 3:
        return {}, {}, None
    hdr, unit, data = rows[hi], rows[hi + 1], rows[hi + 2 :]

    def fnum(x: str) -> float | None:
        try:
            return float(str(x).replace(",", ""))
        except ValueError:
            return None

    id_col = hdr.index("ID") if "ID" in hdr else None
    best: dict[str, float] = {}
    best_id: int | None = None
    best_t: float | None = None
    acc: dict[str, list[float]] = {}
    for r in data:
        vals = {k: fnum(v) for k, v in zip(hdr, r, strict=True)}
        for k, v in zip(hdr, r, strict=True):
            x = fnum(v)
            if x is not None:
                acc.setdefault(k, []).append(x)
        if id_col is not None:
            t = vals.get("gpu__time_duration.sum")
            id_v = vals.get("ID")
            if t is not None and id_v is not None and (best_t is None or t < best_t):
                best_t, best_id = t, int(id_v)
                best = {k: v for k, v in vals.items() if v is not None}
    # no ID/time columns -> degenerate export, fall back to the mean
    mean = best if best else {k: sum(v) / len(v) for k, v in acc.items() if v}
    # durations -> ns whatever unit this ncu prints
    units = dict(zip(hdr, unit, strict=True))
    tu = units.get("gpu__time_duration.sum", "nsecond").strip().lower()
    k_ns = {"nsecond": 1, "ns": 1, "usecond": 1e3, "us": 1e3, "msecond": 1e6, "ms": 1e6, "second": 1e9, "s": 1e9}
    if "gpu__time_duration.sum" in mean:
        mean["gpu__time_duration.sum"] *= k_ns.get(tu, 1)
        units["gpu__time_duration.sum"] = "ns"
    return mean, units, best_id


def parse_details_csv(text: str, best_id: int | None = None) -> dict[str, dict[str, float]]:
    """``--page details --csv`` → {section: {"metric (unit)": mean}}.

    With ``best_id``, only that launch's rows are kept (the fastest launch
    of parse_raw_csv) so details and raw describe the same replay."""
    import csv
    import io

    acc: dict[tuple[str, str], list[float]] = {}
    for r in csv.DictReader(io.StringIO(text)):
        if best_id is not None and r.get("ID", "") not in ("", str(best_id)):
            continue
        try:
            x = float(str(r.get("Metric Value", "")).replace(",", ""))
        except ValueError:
            continue
        unit = r.get("Metric Unit", "")
        name = r.get("Metric Name", "") + (f" ({unit})" if unit else "")
        acc.setdefault((r.get("Section Name", ""), name), []).append(x)
    out: dict[str, dict[str, float]] = {}
    for (sec, name), v in acc.items():
        out.setdefault(sec, {})[name] = sum(v) / len(v)
    return out
