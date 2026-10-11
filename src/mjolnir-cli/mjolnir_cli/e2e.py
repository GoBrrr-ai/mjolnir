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
# llama-benchy means over runs): their noise is larger than the kernel's
# same-run ncu ratios (verify.stages.NOISE_PCT). Per-run decode t/s spreads
# ~10-15% in history.jsonl; the means of 6+ runs per side differ by up to ~7%.
E2E_NOISE_PCT = 7.0


def is_official(rec: dict) -> bool:
    return str(rec.get("image", "")).startswith("vllm/vllm-openai")


def is_mjolnir(rec: dict) -> bool:
    return str(rec.get("image", "")).startswith(_MJOLNIR_IMAGE_PREFIXES)


def latest_pair(history_file: Path, mjolnir_config: str | None = None) -> tuple[dict | None, dict | None]:
    """(official, mjolnir) — the newest record of each side: a stock config on
    the official image, and the Mjolnir backend config (default
    ``NVFP4_MJOLNIR``) on a Mjolnir image. Nothing else qualifies."""
    want = mjolnir_config or DEFAULT_QUANT
    off = mj = None
    for r in load_records(history_file):
        if not r.get("cells"):
            continue
        if is_official(r) and r.get("config") != want:
            off = r if off is None or r.get("epoch", 0) >= off.get("epoch", 0) else off
        elif is_mjolnir(r) and r.get("config") == want:
            mj = r if mj is None or r.get("epoch", 0) >= mj.get("epoch", 0) else mj
    return off, mj


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


def side_label(rec: dict) -> str:
    return f"{rec.get('image', '')} · {rec.get('config', '')} ({rec.get('backend', '')})"
