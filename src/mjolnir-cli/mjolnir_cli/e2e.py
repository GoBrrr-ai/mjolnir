"""End to end: the official vLLM image vs the Mjolnir image, from history.

Everything here reads ``benchmarks/history.jsonl`` (written by ``mjolnir bench
perf|ab|e2e``) — the comparison table, the README section and the opening
infographic all come from the same pair of records:

* **official** — the latest run on the upstream image (``vllm/vllm-openai:…``)
  with a stock config: native vLLM, native FlashInfer;
* **mjolnir**  — the latest run of the Mjolnir backend config
  (``NVFP4_MJOLNIR``) on a Mjolnir image.

Stock backends (FlashInfer, FA2) measured *on the Mjolnir image* are never
used: that image patches FlashInfer paths too, so it is not native vLLM.

Metrics (per context × concurrency cell, mean over every measured run):

| key       | what                                   | better |
|-----------|----------------------------------------|--------|
| ``tg_tps``  | decode throughput, tokens/s (all requests) | higher |
| ``pp_tps``  | prefill throughput, prompt tokens/s        | higher |
| ``ttft_ms`` | time to first token, ms (e2e; falls back to llama-benchy's ``ttfr``) | lower |
| ``itl_ms``  | inter-token latency, ms = 1000 / per-request decode rate | lower |
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mjolnir_cli.config import DEFAULT_QUANT, IMAGE_REPO
from mjolnir_cli.history import load_records

# Older images of this project (kept so history rows still classify).
_MJOLNIR_IMAGE_PREFIXES = (IMAGE_REPO, "narandill/mjolnir-", "mjolnir/vllm-thor")


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    higher_better: bool
    fmt: str


METRICS: tuple[Metric, ...] = (
    Metric("tg_tps", "decode", "t/s", True, "{:.1f}"),
    Metric("pp_tps", "prefill", "t/s", True, "{:,.0f}"),
    Metric("ttft_ms", "TTFT", "ms", False, "{:,.0f}"),
    Metric("itl_ms", "ITL", "ms", False, "{:.1f}"),
)
BY_KEY = {m.key: m for m in METRICS}

# End-to-end numbers are wall-clock serving runs (two separate server launches,
# llama-benchy means over runs). The band that colors a gain: ±3%, the same as
# the kernel's same-run ncu band (verify.stages.NOISE_PCT). Caveat: with MTP
# speculative decoding the per-run decode t/s spreads ~10-15% (the acceptance
# rate varies by prompt), so a decode mean of 6 runs carries a ~2-6% standard
# error — a decode gain inside ~6% needs repeated sweeps (--repeat) to trust.
E2E_NOISE_PCT = 3.0


def is_official(rec: dict) -> bool:
    return str(rec.get("image", "")).startswith("vllm/vllm-openai")


def is_mjolnir(rec: dict) -> bool:
    return str(rec.get("image", "")).startswith(_MJOLNIR_IMAGE_PREFIXES)


def _merge(recs: list[dict]) -> dict | None:
    """One side's records → the newest record, with every (concurrency,
    context) cell taken from the newest run that measured it. A run at extra
    depths (``bench e2e --depths 65536,131072``) adds cells instead of
    replacing the table; runs with a different speculative setup than the
    newest one are left out. ``merged_runs`` lists the raw dirs used."""
    if not recs:
        return None
    recs = sorted(recs, key=lambda r: r.get("epoch", 0), reverse=True)
    head = recs[0]
    sp = spec_of(head)
    cells: dict = {}
    used: list[str] = []
    for r in recs:
        if spec_of(r) != sp:
            continue
        took = False
        for c in r.get("cells", []):
            k = (c.get("concurrency"), c.get("context") or 0, bool(c.get("prefill_phase")))
            if k not in cells:
                cells[k] = c
                took = True
        if took:
            used.append(str(r.get("raw", "")))
    out = dict(head)
    out["cells"] = sorted(cells.values(), key=lambda c: (c.get("concurrency") or 0, c.get("context") or 0))
    out["merged_runs"] = used
    return out


def latest_pair(history_file: Path, mjolnir_config: str | None = None) -> tuple[dict | None, dict | None]:
    """(official, mjolnir): each side's runs merged cell by cell, newest run
    first (:func:`_merge`) — a stock config on the official image, and the
    Mjolnir backend config (default ``DEFAULT_QUANT``) on a Mjolnir image.
    Nothing else qualifies; the official side follows its newest config."""
    want = mjolnir_config or DEFAULT_QUANT
    offs, mjs = [], []
    for r in load_records(history_file):
        if not r.get("cells"):
            continue
        if is_official(r) and r.get("config") != want:
            offs.append(r)
        elif is_mjolnir(r) and r.get("config") == want:
            mjs.append(r)
    if offs:  # one official config: the newest run's
        newest = max(offs, key=lambda r: r.get("epoch", 0)).get("config")
        offs = [r for r in offs if r.get("config") == newest]
    return _merge(offs), _merge(mjs)


def cell(rec: dict, conc: int, ctx: int) -> dict | None:
    return next(
        (
            c
            for c in rec.get("cells", [])
            if c.get("concurrency") == conc and (c.get("context") or 0) == ctx and not c.get("prefill_phase")
        ),
        None,
    )


def value(c: dict | None, key: str) -> float | None:
    """Mean of a metric in a cell, with fallbacks for records written before
    TTFT/ITL were stored (ttft ← ttfr; ITL ← 1000 / decode t/s at c=1)."""
    if not c:
        return None
    m = (c.get(key) or {}).get("mean")
    if m is not None:
        return float(m)
    if key == "ttft_ms":
        return value(c, "ttfr_ms")
    if key == "itl_ms":
        rate = (c.get("tg_req_tps") or {}).get("mean")
        if rate is None and (c.get("concurrency") or 1) == 1:
            rate = (c.get("tg_tps") or {}).get("mean")
        return 1000.0 / rate if rate else None
    return None


def gain_pct(key: str, official: float, mjolnir: float) -> float:
    """Signed improvement of Mjolnir over official, in % — positive = better,
    whichever direction the metric runs (faster decode, or lower TTFT)."""
    if BY_KEY[key].higher_better:
        return (mjolnir / official - 1.0) * 100.0
    return (official / mjolnir - 1.0) * 100.0


def shared_cells(off: dict, mj: dict) -> list[tuple[int, int]]:
    """(concurrency, context) cells both records measured, sorted."""
    a = {(c.get("concurrency"), c.get("context") or 0) for c in off.get("cells", []) if not c.get("prefill_phase")}
    b = {(c.get("concurrency"), c.get("context") or 0) for c in mj.get("cells", []) if not c.get("prefill_phase")}
    return sorted(a & b, key=lambda k: (k[0], k[1]))


def compare(off: dict, mj: dict) -> list[dict]:
    """One row per shared cell: {conc, ctx, <key>: (official, mjolnir, gain%)}."""
    rows = []
    for conc, ctx in shared_cells(off, mj):
        a, b = cell(off, conc, ctx), cell(mj, conc, ctx)
        row: dict = {"conc": conc, "ctx": ctx}
        for m in METRICS:
            va, vb = value(a, m.key), value(b, m.key)
            row[m.key] = (va, vb, gain_pct(m.key, va, vb) if va and vb else None)
        rows.append(row)
    return rows


def hero(off: dict, mj: dict) -> dict | None:
    """The infographic scenario: one request (c=1) at the longest context both
    sides measured. {ctx, tg_tps: (off, mj, gain), pp_tps: (...)}."""
    c1 = [ctx for conc, ctx in shared_cells(off, mj) if conc == 1]
    if not c1:
        return None
    ctx = max(c1)
    a, b = cell(off, 1, ctx), cell(mj, 1, ctx)
    out: dict = {"ctx": ctx}
    for key in ("tg_tps", "pp_tps", "ttft_ms", "itl_ms"):
        va, vb = value(a, key), value(b, key)
        out[key] = (va, vb, gain_pct(key, va, vb) if va and vb else None)
    if out["tg_tps"][2] is None or out["pp_tps"][2] is None:
        return None
    return out


def ctx_label(ctx: int) -> str:
    if not ctx:
        return "no ctx"
    return f"{ctx // 1024}K" if ctx % 1024 == 0 else f"{ctx:,}"


def pct(v: float | None) -> str:
    """Signed percent with a real minus sign, 1 decimal."""
    if v is None:
        return "—"
    return f"{'−' if v < 0 else '+'}{abs(v):.1f}%"


def spec_of(rec: dict) -> dict | None:
    """The speculative decoding a record ran with: stored in the record since
    the field exists, else read from the config it names."""
    if rec.get("spec") is not None:
        return rec["spec"] or None
    from mjolnir_cli.config import config_speculative, configs_root

    path = configs_root() / str(rec.get("model", "")) / f"{rec.get('config', '')}.yaml"
    return config_speculative(path) if path.exists() else None


def spec_label(*recs: dict) -> str:
    """``MTP, 3 speculative tokens`` (``on both sides`` / per side when they differ)."""

    def one(sp: dict | None) -> str:
        if not sp:
            return "no speculative decoding"
        n = sp["tokens"]
        return f"{sp['method'].upper() or 'speculative'}, {n} speculative token{'s' if n != 1 else ''}"

    labels = [one(spec_of(r)) for r in recs]
    if len(set(labels)) == 1:
        return labels[0]
    return " vs ".join(labels)


def spec_short(*recs: dict) -> str:
    """``MTP×3`` — the compact form for tight layouts (the infographic)."""
    labels = []
    for r in recs:
        sp = spec_of(r)
        labels.append(f"{(sp['method'] or 'spec').upper()}×{sp['tokens']}" if sp else "no spec. decoding")
    return labels[0] if len(set(labels)) == 1 else " vs ".join(labels)


def acceptance_line(off: dict, mj: dict) -> str:
    """'\nMTP tokens/step: 2.84 → 2.71 · preemptions 0 → 0' when both legs
    recorded spec stats (bench runs since the counters were added). Decode
    t/s = tokens per step / step time, so a lower acceptance length slows
    decode with the same kernel time."""
    a, b = off.get("spec_stats") or {}, mj.get("spec_stats") or {}
    if not a.get("acceptance_length") or not b.get("acceptance_length"):
        return ""
    return (
        f"\nspec. decoding tokens/step: {a['acceptance_length']:.2f} → {b['acceptance_length']:.2f}"
        f" · preemptions {a.get('preemptions', 0)} → {b.get('preemptions', 0)}"
    )


def side_label(rec: dict) -> str:
    return f"{rec.get('image', '')} · {rec.get('config', '')} ({rec.get('backend', '')})"
