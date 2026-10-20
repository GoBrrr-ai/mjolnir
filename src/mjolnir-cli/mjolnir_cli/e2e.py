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

from mjolnir_cli.config import BASELINE_QUANT, DEFAULT_MODEL, DEFAULT_QUANT, model_of
from mjolnir_cli.history import load_records
from mjolnir_cli.platforms import DEFAULT_PLATFORM, REGISTRY, platform_of

# Every platform's image repo, plus the legacy names (kept so history rows
# still classify): the shared narandill/mjolnir-* namespace and the
# pre-registry image.
_MJOLNIR_IMAGE_PREFIXES = (
    *(p.image_repo for p in REGISTRY.values()),
    "narandill/mjolnir-",
    "mjolnir/vllm-thor",
)


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
# the official-vs-Mjolnir comparison (README table, e2e charts, infographic)
# and `bench e2e`'s default depths stop here: the 64K / 128K cells take hours
# per leg; `bench perf|ab --depths` still measure beyond it
MAX_CTX = 32768


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


def latest_pair(
    history_file: Path,
    mjolnir_config: str | None = None,
    official_config: str | None = None,
    model: str | None = None,
    platform: str | None = None,
) -> tuple[dict | None, dict | None]:
    """(official, mjolnir): each side's runs merged cell by cell, newest run
    first (:func:`_merge`) — a stock config on the official image, and the
    Mjolnir backend config on a Mjolnir image: ``mjolnir_config``, else
    ``DEFAULT_QUANT`` when the model has runs of it, else the model's newest
    Mjolnir-image config (``DEFAULT_QUANT`` exists only for the default model).

    The official side is ``official_config`` (default ``BASELINE_QUANT``,
    native FlashInfer) when the history has it, else the newest official
    config — so an optional official leg (``bench e2e --optional``: DSpark,
    DFlash2 on stock vLLM) never replaces the baseline by being newer.

    Both sides are runs of one (``model``, ``platform``) pair (defaults: the
    default model, ``thor``): another model's — or another platform's — run
    under the same config name is never paired with them. Platform-less
    (legacy) records resolve to ``thor``."""
    model = model or DEFAULT_MODEL
    platform = platform or DEFAULT_PLATFORM
    base = official_config or BASELINE_QUANT
    recs = [
        r for r in load_records(history_file) if r.get("cells") and model_of(r) == model and platform_of(r) == platform
    ]
    want = mjolnir_config
    if want is None:
        mine = [r for r in recs if is_mjolnir(r)]
        want = (
            DEFAULT_QUANT
            if any(r.get("config") == DEFAULT_QUANT for r in mine) or not mine
            else max(mine, key=lambda r: r.get("epoch", 0)).get("config")
        )
    offs, mjs = [], []
    for r in recs:
        if is_official(r) and r.get("config") != want:
            offs.append(r)
        elif is_mjolnir(r) and r.get("config") == want:
            mjs.append(r)
    if any(r.get("config") == base for r in offs):
        offs = [r for r in offs if r.get("config") == base]
    elif offs:  # no baseline run yet: one official config, the newest run's
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
    TTFT/ITL were stored (ttft ← ttfr; ITL ← 1000 / decode t/s at c=1).
    Prefill is not reported for a cell without context."""
    if not c:
        return None
    if key == "pp_tps" and not (c.get("context") or 0):
        return None  # no context, no prefill to compare: the "no ctx" row is decode only
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


def change_pct(official: float, mjolnir: float) -> float:
    """The relative change of the measured value, in %: (Mjolnir − official) ÷
    official × 100. Throughput: +57% = 1.57× the tokens/s. Latency: −27% =
    27% fewer milliseconds (a latency reduction, NOT the reciprocal speedup
    official ÷ Mjolnir − 1, which would read +37%)."""
    return (mjolnir / official - 1.0) * 100.0


def gain_pct(key: str, official: float, mjolnir: float) -> float:
    """Signed improvement of Mjolnir over official, in % — positive = better
    whichever way the metric runs: the throughput increase for decode /
    prefill, the latency reduction (official − Mjolnir) ÷ official for TTFT /
    ITL. Same magnitude as :func:`change_pct`, sign flipped for latency."""
    ch = change_pct(official, mjolnir)
    return ch if BY_KEY[key].higher_better else -ch


def shared_cells(off: dict, mj: dict) -> list[tuple[int, int]]:
    """(concurrency, context) cells both records measured, sorted."""
    a = {(c.get("concurrency"), c.get("context") or 0) for c in off.get("cells", []) if not c.get("prefill_phase")}
    b = {(c.get("concurrency"), c.get("context") or 0) for c in mj.get("cells", []) if not c.get("prefill_phase")}
    return sorted(a & b, key=lambda k: (k[0], k[1]))


def all_cells(off: dict, mj: dict, max_ctx: int | None = MAX_CTX) -> list[tuple[int, int]]:
    """(concurrency, context) cells either record measured, up to ``max_ctx``, sorted."""
    a = {(c.get("concurrency"), c.get("context") or 0) for c in off.get("cells", []) if not c.get("prefill_phase")}
    b = {(c.get("concurrency"), c.get("context") or 0) for c in mj.get("cells", []) if not c.get("prefill_phase")}
    return sorted((k for k in a | b if max_ctx is None or k[1] <= max_ctx), key=lambda k: (k[0], k[1]))


def missing_contexts(off: dict, mj: dict, max_ctx: int | None = MAX_CTX) -> dict[str, list[int]]:
    """Contexts (up to ``max_ctx``) only one side measured: {"official": [...], "mjolnir": [...]}."""
    ok = lambda c: max_ctx is None or (c.get("context") or 0) <= max_ctx  # noqa: E731
    a = {c.get("context") or 0 for c in off.get("cells", []) if not c.get("prefill_phase") and ok(c)}
    b = {c.get("context") or 0 for c in mj.get("cells", []) if not c.get("prefill_phase") and ok(c)}
    return {"official": sorted(b - a), "mjolnir": sorted(a - b)}


def compare(off: dict, mj: dict, every: bool = False, max_ctx: int | None = MAX_CTX) -> list[dict]:
    """One row per shared cell (``every``: per cell either side measured, the
    other side None): {conc, ctx, <key>: (official, mjolnir, gain%)}."""
    rows = []
    for conc, ctx in all_cells(off, mj, max_ctx) if every else shared_cells(off, mj):
        a, b = cell(off, conc, ctx), cell(mj, conc, ctx)
        row: dict = {"conc": conc, "ctx": ctx}
        for m in METRICS:
            va, vb = value(a, m.key), value(b, m.key)
            row[m.key] = (va, vb, gain_pct(m.key, va, vb) if va and vb else None)
        rows.append(row)
    return rows


def hero(off: dict, mj: dict, max_ctx: int | None = MAX_CTX) -> dict | None:
    """The infographic scenario: one request (c=1), among the contexts up to
    ``max_ctx`` where both sides measured decode AND prefill, the one with
    the highest decode gain. {ctx, tg_tps: (off, mj, gain), pp_tps: (...), ...}."""
    best = None
    for conc, ctx in shared_cells(off, mj):
        if conc != 1 or (max_ctx is not None and ctx > max_ctx):
            continue
        a, b = cell(off, 1, ctx), cell(mj, 1, ctx)
        out: dict = {"ctx": ctx}
        for key in ("tg_tps", "pp_tps", "ttft_ms", "itl_ms"):
            va, vb = value(a, key), value(b, key)
            out[key] = (va, vb, gain_pct(key, va, vb) if va and vb else None)
        if out["tg_tps"][2] is None or out["pp_tps"][2] is None:
            continue
        if best is None or out["tg_tps"][2] > best["tg_tps"][2]:
            best = out
    return best


def infographic_row(off: dict, mj: dict) -> dict | None:
    """The opening infographic's ONE scenario row: C=1 at the smallest context
    both sides measured (4K in the latest pair) — every panel reads that
    single row, decode / TTFT / ITL included (no ctx=0 special-casing).
    Same shape as :func:`hero`: ``{ctx, <key>: (official, mjolnir, gain)}``;
    ``None`` when the pair shares no C=1 context cell."""
    for conc, ctx in shared_cells(off, mj):
        if conc != 1 or not ctx:
            continue
        a, b = cell(off, 1, ctx), cell(mj, 1, ctx)
        out: dict = {"ctx": ctx}
        for key in ("tg_tps", "pp_tps", "ttft_ms", "itl_ms"):
            va, vb = value(a, key), value(b, key)
            out[key] = (va, vb, gain_pct(key, va, vb) if va and vb else None)
        return out
    return None


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

    path = configs_root() / str(rec.get("model", "")) / str(rec.get("config", "")) / "config.yaml"
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
