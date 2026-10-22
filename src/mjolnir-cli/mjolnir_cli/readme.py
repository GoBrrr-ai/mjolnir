"""README + docs generation — every number, table and chart reference is
computed from the committed raw data.

``mjolnir plot --readme`` renders the render-target set (cycle 3b; discovery
+ contexts in :mod:`mjolnir_cli.readme_targets`):

* ``README.md``                          the root: hardware + models tables,
                                         the headline pinned to
                                         (thor, the default model)
* ``docs/platforms/README.md``           the platform index
* ``docs/platforms/<p>/README.md``       one per shipped platform (+ its
                                         kernel case study under ``kernel/``
                                         when it has raws)
* ``configs/<vendor>/<model>/README.md``                  one per model dir
* ``configs/…/<quant>/README.md``                         one per config dir
                                                          (its Benchmarks
                                                          section included)
* ``docs/cli.md``                        the CLI reference (from the commands;
                                         written by the ``plot`` command, which
                                         holds the root app — not by the render
                                         loop, which takes no command tree)
* ``docker/vllm-thor/README.md``         the Docker Hub description: plain
                                         markdown only (Docker Hub does not
                                         render inline HTML), absolute image /
                                         link URLs (no repo file hosting to
                                         resolve relatives against), no charts
                                         or data tables, well under Docker
                                         Hub's 25,000-byte description limit.
                                         ``docker-image.yml`` pushes it via
                                         ``peter-evans/dockerhub-description``.

Inputs: the latest default run of each ``mjolnir verify`` stage for the
page's (platform, model) pair (``benchmarks/raw/verify/``), the e2e history
(``benchmarks/history.jsonl``), the chart manifest
(``assets/benchmarks/manifest.json``, written by ``charts.render_all``) and
the vLLM base tag in the Dockerfile. Re-rendering is idempotent: the output
only changes when the raw data does (CI checks it).

Template contract (Jinja2, StrictUndefined — a typo fails loud):

* ``ctx``            — the target's context: for the kernel chapters a
                       ``ReadmeContext`` (versions, headline numbers, the
                       per-stage markdown tables ``ctx.tables[stage]``, the
                       per-stage findings ``ctx.findings[stage]`` and
                       ``ctx.pending``); the pages read their own dataclasses
                       from ``readme_targets``
* ``chart(name, alt)`` — an ``<img>`` for a chart in the
                       manifest (``""`` when the chart has no data for this
                       page's (platform, model) pair yet)
* ``has(stage)``     — the stage has a saved run
* ``root``           — relative path from the rendered file to the repo root
* ``platform``       — the platform row of the page being rendered (kernel
                       chapters); ``model_short`` — its headline model
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import jinja2

from mjolnir_cli import availability, readme_targets
from mjolnir_cli.config import (
    DEFAULT_MODEL,
    DEFAULT_PLATFORM,
    DEFAULT_QUANT,
    IMAGE_REPO,
    RepoLayout,
    by_model,
    config_speculative,
    configs_root,
    model_of,
)
from mjolnir_cli.platforms import REGISTRY, platform_of
from mjolnir_cli.verify import results
from mjolnir_cli.readme_tables import (
    L_label,
    _correctness_table,
    _fp6prof_table,
    _kvcache_table,
    _pdl_table,
    _quant_table,
    _series_time_table,
    _spec_table,
    _table,
    _variant_table,
    _vs_vllm_table,
    _wall_table,
    color,
    fmt_us,
    kvcache_label,
    spec_label,
)
from mjolnir_cli.verify.stages import ALL_LS, NOISE_PCT, STAGES
from mjolnir_cli.readme_tables import ratio_pct as ratio_pct  # re-export (tests)

_GREEN = "#16a34a"
_RED = "#dc2626"
_GREY = "grey"
_MINUS = "−"


# ── formatting ──────────────────────────────────────────────────────────────


# ── per-stage tables + findings ─────────────────────────────────────────────


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
        out.extend(
            f"{kvcache_label(r)}: {r['max_concurrency']:.1f} requests at the full "
            f"{L_label(r.get('max_model_len') or 0)} context (vLLM's own estimate)."
            for r in rows.values()
            if r.get("max_concurrency")
        )
        return out
    by = results.series_by_id(doc)
    noise = NOISE_PCT / 100
    if stage in ("sweep", "hndsweep"):
        base = by.get("fi")
        for sid in ("mjolnir", "mjolnir-fp8", "fi-hnd", "fa2", "fa4"):
            if base and sid in by:
                sp = results.speedup(base, by[sid])
                if not sp:
                    continue
                g = results.geomean(sp.values())
                lo, hi = min(sp, key=lambda k: sp[k]), max(sp, key=lambda k: sp[k])
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
            top = max(counts, key=lambda w: counts[w])
            out.append(f"Fastest variant at the most context lengths: **`{top}`** ({counts[top]} of {len(Ls)}).")
            geo = {s["label"]: results.geomean(results.speedup(by["fi"], s).values()) for s in variants}
            geo = {k: v for k, v in geo.items() if v}
            if geo:
                g_top = max(geo, key=lambda k: geo[k])
                g_low = min(geo, key=lambda k: geo[k])
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


TABLES = {
    "sweep": lambda d: _series_time_table(d, ("fi", "fa4", "fa2")),
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
    # the sweep headline finding
    has_sweep: bool = False
    geo_vs_fi: str = ""
    geo_vs_fi_fp8: str = ""
    geo_vs_fa2: str = ""
    geo_vs_fa4: str = ""
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
    us_64k_fa4: str = ""
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
    vs_vllm_missing: str = ""
    e2e_example: dict = field(default_factory=dict)
    e2e_max_ctx: str = field(default_factory=lambda: L_label(__import__("mjolnir_cli.e2e").e2e.MAX_CTX))
    # the README's Key Features grid: measured numbers, each None when its run is missing
    feat: dict = field(default_factory=dict)
    default_config: str = field(
        default_factory=lambda: __import__("mjolnir_cli.config").config.DEFAULT_QUANT
    )  # contexts only one side measured (the command that completes the table)
    # model quality per serving config (mjolnir bench accuracy)
    accuracy_table: str = ""
    accuracy_categories: str = ""
    accuracy_findings: list = field(default_factory=list)
    accuracy_runs: list = field(default_factory=list)  # run ids the table draws on
    accuracy_args: list = field(default_factory=list)  # (suite label, tool-eval-bench args)
    accuracy_baseline: str = ""
    accuracy_excluded_note: str = ""
    # the image's patch inventory (docker/vllm-thor/PATCHES.md) — verbatim table + row counts
    patches_table: str = ""
    patches_total: int = 0
    patches_pr: int = 0
    # the render-target identity this context belongs to: the (platform,
    # model) pair the tables/findings/e2e/accuracy were read for, its
    # registry row, and the root page's platform matrix + measured-pair index
    chart_pair: tuple[str, str] | None = None
    platform: Any = None
    platforms: list = field(default_factory=list)
    # the root README's Supported Models table (cycle 3g): one row per
    # scanned model, one availability (+ t/s) cell per registry platform
    models: list = field(default_factory=list)


def _vllm_base_version(layout: RepoLayout) -> str:
    if layout.repo_root is None:
        return ""
    dockerfile = layout.repo_root / "docker" / "vllm-thor" / "Dockerfile"
    if dockerfile.exists():
        m = re.search(r"vllm/vllm-openai:v(\d+\.\d+\.\d+)", dockerfile.read_text())
        if m:
            return m.group(1)
    return ""


def _patches(layout: RepoLayout) -> tuple[str, int, int]:
    """The patch inventory table from docker/vllm-thor/PATCHES.md, re-emitted
    verbatim — it is the single source of truth the build consumes. Returns
    (table, total, pr_backed); pr_backed = rows patching a vLLM PR."""
    p = (layout.repo_root / "docker" / "vllm-thor" / "PATCHES.md") if layout.repo_root else None
    if p is None or not p.exists():
        return "", 0, 0
    lines = p.read_text().splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("| File |")), None)
    if start is None or start + 1 >= len(lines) or not lines[start + 1].startswith("|-"):
        raise ValueError(f"{p}: the '| File | PR | What it fixes |' inventory table is missing")
    i = start + 2
    rows = []
    while i < len(lines) and lines[i].startswith("|"):
        rows.append(lines[i])
        i += 1
    if not rows:
        raise ValueError(f"{p}: the inventory table has no rows")
    return (
        "\n".join([lines[start], lines[start + 1], *rows]),
        len(rows),
        sum(1 for r in rows if "vllm-project/vllm/pull/" in r),
    )


def _version_from(path: Path) -> str:
    try:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', path.read_text())
        return m.group(1) if m else ""
    except OSError:
        return ""


def collect_context(
    layout: RepoLayout,
    data: dict | None = None,
    platform: str = DEFAULT_PLATFORM,
    model: str = DEFAULT_MODEL,
) -> ReadmeContext:
    """The context for one (platform, model) pair — the root README's and the
    kernel chapters' context. Defaults are the pinned headline pair (decision
    3); ``data`` overrides the pair's verify docs (as ``load_verify`` does
    unfiltered)."""
    data = data if data is not None else readme_targets.pair_docs(layout, platform, model)
    ctx = ReadmeContext(vllm_version=_vllm_base_version(layout))
    ctx.chart_pair = (platform, model)
    ctx.platform = next((p for p in readme_targets.platform_rows() if p.key == platform), None)
    ctx.platforms = readme_targets.platform_rows()
    ctx.models = _models_table(layout)
    ctx.image = f"{IMAGE_REPO}:v{ctx.vllm_version}" if ctx.vllm_version else IMAGE_REPO
    if layout.repo_root is not None:
        ctx.kernel_version = _version_from(layout.repo_root / "src" / "mjolnir" / "mjolnir" / "__init__.py")
        ctx.cli_version = _version_from(layout.repo_root / "src" / "mjolnir-cli" / "mjolnir_cli" / "__init__.py")
        ctx.patches_table, ctx.patches_total, ctx.patches_pr = _patches(layout)
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
    ctx.pending = [s for s in STAGES if s not in data and s not in ("timing", "vllm", "profile")]

    sweep = data.get("sweep")
    if sweep:
        by = results.series_by_id(sweep)
        ours = by.get("mjolnir") or by.get("mjolnir-fp8")
        if ours and "fi" in by:
            ctx.has_sweep = True
            ctx.sweep_run = sweep["run_id"]
            sp = results.speedup(by["fi"], ours)
            ctx.geo_vs_fi = f"{results.geomean(sp.values()):.2f}"
            lo, hi = min(sp, key=lambda k: sp[k]), max(sp, key=lambda k: sp[k])
            ctx.min_vs_fi, ctx.min_vs_fi_L = f"{sp[lo]:.2f}", L_label(lo)
            ctx.max_vs_fi, ctx.max_vs_fi_L = f"{sp[hi]:.2f}", L_label(hi)
            last = max(sp)
            ctx.long_vs_fi, ctx.long_L = f"{sp[last]:.2f}", L_label(last)
            ctx.beats_fi = f"{sum(1 for v in sp.values() if v >= 1 + NOISE_PCT / 100)} of {len(sp)}"
            if "mjolnir-fp8" in by:
                ctx.geo_vs_fi_fp8 = f"{results.geomean(results.speedup(by['fi'], by['mjolnir-fp8']).values()):.2f}"
            if "fa2" in by:
                ctx.geo_vs_fa2 = f"{results.geomean(results.speedup(by['fa2'], ours).values()):.2f}"
            if "fa4" in by:
                ctx.geo_vs_fa4 = f"{results.geomean(results.speedup(by['fa4'], ours).values()):.2f}"
            pm = results.point_map(ours)
            if 65536 in pm:
                ctx.us_64k_mjolnir = fmt_us(pm[65536])
                ctx.us_64k_fi = fmt_us(results.point_map(by["fi"]).get(65536))
                if "fa2" in by:
                    ctx.us_64k_fa2 = fmt_us(results.point_map(by["fa2"]).get(65536))
                if "fa4" in by:
                    ctx.us_64k_fa4 = fmt_us(results.point_map(by["fa4"]).get(65536))
    corr = data.get("correctness")
    if corr:
        cs = corr.get("cases", [])
        ctx.correctness = f"{sum(1 for c in cs if c.get('passed'))} / {len(cs)}"
    from mjolnir_cli import e2e

    off, mj = e2e.latest_pair(layout.history_file, model=model, platform=platform)
    if off and mj:
        ctx.vs_vllm = _vs_vllm_table(off, mj)
        ctx.vs_vllm_official, ctx.vs_vllm_mjolnir = e2e.side_label(off), e2e.side_label(mj)
        ctx.vs_vllm_spec = e2e.spec_label(off, mj)
        gaps = e2e.missing_contexts(off, mj)
        notes = [
            f"the {'Mjolnir' if side == 'mjolnir' else 'official'} side has no run at "
            f"{', '.join(e2e.ctx_label(c) for c in ctxs)} context yet (`mjolnir bench e2e --only {side} --depths "
            f"{','.join(str(c) for c in ctxs)}`)"
            for side, ctxs in gaps.items()
            if ctxs
        ]
        ctx.vs_vllm_missing = "; ".join(notes)
        h = e2e.hero(off, mj)
        if h and h["ttft_ms"][2] is not None:
            a, b = h["ttft_ms"][0], h["ttft_ms"][1]
            ctx.e2e_example = {
                "ctx": e2e.ctx_label(h["ctx"]),
                "a": a,
                "b": b,
                "change": e2e.change_pct(a, b),
                "ratio": a / b,
            }
        if h:
            ctx.vs_vllm_hero = {
                "ctx": e2e.ctx_label(h["ctx"]),
                "tg_tps": e2e.pct(h["tg_tps"][2]),
                "pp_tps": e2e.pct(h["pp_tps"][2]),
                "ttft_ms": f"{h['ttft_ms'][0] / 1000:.2f} s → {h['ttft_ms'][1] / 1000:.2f} s "
                f"({e2e.pct(-h['ttft_ms'][2])})"
                if h["ttft_ms"][2] is not None
                else "—",
                "itl_ms": f"{h['itl_ms'][0]:.1f} ms → {h['itl_ms'][1]:.1f} ms ({e2e.pct(-h['itl_ms'][2])})"
                if h["itl_ms"][2] is not None
                else "—",
            }
    from mjolnir_cli import accuracy

    acc_rows = accuracy.latest_legs(accuracy.load_runs(layout))  # read once, for the page and the hero
    _accuracy_context(ctx, acc_rows)
    _features(ctx, data, layout, acc_rows, platform=platform, model=model)
    return ctx


@dataclass(frozen=True)
class ModelCell:
    """One (model, platform) cell of the Supported Models table: the
    availability symbol plus — when history has it — the decode t/s at
    concurrency 1 / context 0 of the fastest config's latest run (its name
    goes into the cell's title)."""

    symbol: str
    tps: str  # "" when no config has a (1, 0) cell
    config: str  # the fastest config's name; "" without data


@dataclass(frozen=True)
class ModelsRow:
    """One row of the Supported Models table: a config-tree model and one
    cell per registry platform (insertion order follows the registry)."""

    model: str
    short: str
    cells: dict[str, ModelCell]


def _models_table(layout: RepoLayout) -> list[ModelsRow]:
    """The root README's Supported Models table (cycle 3g): one row per
    scanned model, one cell per registry platform. ✅ when ≥1 of the model's
    configs targets the platform and is natively available (the availability
    table), ❌ when every targeting config is explicitly unavailable, — when
    none targets it. "Targets" is the overlay rule
    (:func:`readme_targets.exists_on`) — the same rule the model pages'
    per-platform sections use. The t/s reuses the e2e cell readers: decode
    tokens/s at (1, 0) from the fastest targeting config's latest merged
    history run; an exact tie keeps the scan-first config."""
    from mjolnir_cli import e2e
    from mjolnir_cli.history import load_records

    scan = readme_targets.scan_configs(layout)
    history: dict[tuple[str, str | None, str], list[dict]] = {}
    if layout.history_file.exists():
        for r in load_records(layout.history_file):
            if r.get("cells"):
                history.setdefault((platform_of(r), model_of(r), str(r.get("config") or "")), []).append(r)
    rows = []
    for model in sorted({c.model for c in scan}):
        cells: dict[str, ModelCell] = {}
        for p in REGISTRY.values():
            configs = [c for c in scan if c.model == model and readme_targets.exists_on(c.path, p.key)]
            if not configs:
                cells[p.key] = ModelCell("—", "", "")
                continue
            states = [readme_targets.config_cell(c.path, c.quant, p.key).state for c in configs]
            if availability.NATIVE in states:
                symbol = readme_targets.SYMBOLS[availability.NATIVE]
            elif set(states) == {availability.UNAVAILABLE}:
                symbol = readme_targets.SYMBOLS[availability.UNAVAILABLE]
            else:
                symbol = readme_targets.SYMBOLS[availability.FALLBACK]
            best: tuple[float, str] | None = None
            for c in configs:
                merged = e2e._merge(history.get((p.key, model, c.quant), []))
                v = e2e.value(e2e.cell(merged, 1, 0), "tg_tps") if merged else None
                if v is not None and (best is None or v > best[0]):
                    best = (v, c.quant)
            cells[p.key] = ModelCell(symbol, f"{best[0]:.1f}" if best else "", best[1] if best else "")
        rows.append(ModelsRow(model=model, short=model.split("/")[-1], cells=cells))
    return rows


def _features(
    ctx: ReadmeContext,
    data: dict,
    layout: RepoLayout,
    acc_rows: list[dict],
    platform: str = DEFAULT_PLATFORM,
    model: str = DEFAULT_MODEL,
) -> None:
    """The numbers the Key Features grid quotes — from the same raw results as
    the tables (the grid falls back to plain text where a run is missing)."""
    from mjolnir_cli import accuracy, e2e

    f: dict = {"geo": ctx.geo_vs_fi or None}
    spec = config_speculative(configs_root() / DEFAULT_MODEL / DEFAULT_QUANT / "config.yaml")
    if spec and spec.get("method") == "mtp":
        f["mtp"] = spec["tokens"]
    off, mj = e2e.latest_pair(layout.history_file, model=model, platform=platform)
    h = e2e.hero(off, mj) if off and mj else None
    if h:
        f.update(
            ctx=e2e.ctx_label(h["ctx"]),
            decode=f"+{h['tg_tps'][2]:.0f}%",
            prefill=f"+{h['pp_tps'][2]:.0f}%",
            ttft=f"−{h['ttft_ms'][2]:.0f}%" if h["ttft_ms"][2] is not None else None,
        )
    kv = data.get("kv-cache-size")
    if kv:
        rows = [r for r in kv.get("rows", []) if r.get("kv_tokens")]
        base = next((r for r in rows if r.get("side") == "official"), None)
        fp6 = max(
            (r for r in rows if str(r.get("kv_dtype", "")).startswith("fp6")),
            key=lambda r: r["kv_tokens"],
            default=None,
        )
        if base and fp6:
            f["kv"] = f"+{(fp6['kv_tokens'] / base['kv_tokens'] - 1) * 100:.0f}%"
    rows = by_model(acc_rows).get(model, [])  # the hero names one model's baseline
    b = accuracy.baseline(rows)
    if b:
        mine = [r for r in rows if r is not b and str(r.get("config", "")).startswith("NVFP4_MJOLNIR")]
        deltas = [abs(d[0]) for r in mine for n in ("tools", "hardmode") if (d := accuracy.delta(r, b, n))]
        if deltas:
            f["acc"] = f"{max(deltas):.1f}"
    ctx.feat = f


def _accuracy_context(ctx: ReadmeContext, rows: list[dict]) -> None:
    from mjolnir_cli import accuracy

    if not rows:
        return
    bases = accuracy.baselines(rows)
    names = accuracy.suites_in(rows)
    ctx.accuracy_baseline = accuracy.baseline_caption(bases)
    if accuracy.any_excluded(rows, bases, names):
        ctx.accuracy_excluded_note = accuracy.EXCLUDED_NOTE
    ctx.accuracy_runs = sorted({v["run_id"] for r in rows for v in r["runs"].values()})
    args = {}
    for r in rows:
        for n, v in r["runs"].items():
            args.setdefault(n, " ".join(v.get("args") or []))
    ctx.accuracy_args = [(accuracy.suite_label(n), args[n]) for n in names if n in args]

    def cell(r: dict, base: dict | None, n: str) -> str:
        v = (r["suites"].get(n) or {}).get("score")
        if v is None:
            return "—"
        out = f"{v:.1f}"
        if (r.get("suites", {}).get(n) or {}).get("n", 1) > 1:
            out += f" <sub>n={r['suites'][n]['n']}</sub>"
        return out + accuracy.fmt_delta(accuracy.delta(r, base, n))

    head = ["config", "KV", *[f"{accuracy.suite_label(n)} %" for n in names], "completion"]
    body = []
    for r, base in zip(rows, bases, strict=True):
        comp = [x.get("completion_rate") for x in r["suites"].values() if x.get("completion_rate") is not None]
        label = f"**{r['label']}**" if r is base else r["label"]
        kv = str(r.get("kv_cache_dtype") or "auto").replace("fp6_e2m3", "fp6 e2m3").replace("auto", "bf16")
        body.append([label, kv, *[cell(r, base, n) for n in names], f"{min(comp):.0f}%" if comp else "—"])
    ctx.accuracy_table = _table(head, body, ["---", "---"] + ["--:"] * (len(head) - 2))

    cats = sorted({c for r in rows for c in r.get("categories", {})})
    if cats:
        labels = {c: next(r["categories"][c]["label"] for r in rows if c in r.get("categories", {})) for c in cats}
        marked = [accuracy.leg_excluded(r, b, names) for r, b in zip(rows, bases, strict=True)]
        chead = ["category", *[r["label"] + (" ‡" if m else "") for r, m in zip(rows, marked, strict=True)]]
        cbody = [
            [
                f"{c} · {labels[c]}",
                *[f"{r['categories'][c]['percent']:.0f}" if c in r.get("categories", {}) else "—" for r in rows],
            ]
            for c in cats
        ]
        ctx.accuracy_categories = _table(chead, cbody)
        if any(marked):
            ctx.accuracy_categories += f"\n\n<sub>{accuracy.CATEGORY_EXCLUDED_NOTE}.</sub>"

    for r, base in zip(rows, bases, strict=True):
        if base is None or r is base:
            continue
        deltas = [
            f"{accuracy.suite_label(n)} {d[0]:+.1f}{'‡' if d[1] else ''}"
            for n in names
            if (d := accuracy.delta(r, base, n))
        ]
        if deltas:
            changed = []
            if r.get("scenarios") and base.get("scenarios"):
                # a scenario one leg excluded counts as zero there, as in the ‡ delta
                rs, bs = r["scenarios"], base["scenarios"]
                every = rs.keys() | bs.keys()
                worse = sum(1 for k in every if rs.get(k, 0.0) < bs.get(k, 0.0))
                better = sum(1 for k in every if rs.get(k, 0.0) > bs.get(k, 0.0))
                gone = sorted(every - (rs.keys() & bs.keys()))
                changed = [
                    f"{worse} scenarios scored lower, {better} higher of {len(every)}"
                    + (f"; excluded {', '.join(gone)} counted as zero" if gone else "")
                ]
            ctx.accuracy_findings.append(
                f"**{r['label']}** vs {base['label']}: {', '.join(deltas)} points"
                + (f" ({changed[0]})" if changed else "")
                + "."
            )


# ── "how is this calculated" panels (one under every table / figure with % or ×) ──

_HOW: dict[str, str] = {
    "kernel": """\
- **Time per decode step** = Nsight Compute kernel time (`gpu__time_duration`) of the decode kernel **plus** its split-KV merge kernel, at base clocks with a cold L2, median over the iterations. The reference (FlashInfer, FlashAttention 4 hd256) is measured the same way, at the same L, in the same run.
- **Speedup ×** = reference time ÷ Mjolnir time. `1.24×` means Mjolnir needs 1 ÷ 1.24 = 81% of the reference's time per step (19% less time); below 1 the reference is faster.
- **Geomean over L** = exp(mean of ln(speedup at each L)): every context length weighs the same, and one large ratio cannot dominate the way it would in an arithmetic mean.
- **Colors / "beyond the noise band"**: green = speedup ≥ 1.03, grey = within ±3%, red ≤ 0.97. ±3% (`NOISE_PCT`) is how far a same-run ratio moves between repeated runs on a dedicated Thor.""",
    "variant": """\
- Each cell = **reference time ÷ variant time** (Nsight Compute kernel time, decode + merge, base clocks, cold L2, same run, same L). `1.10×` = the variant needs 1 ÷ 1.10 = 91% of the reference's time; below 1 it is slower.
- **geomean** = exp(mean of ln(ratio)) over the context lengths; **bold** = the fastest variant at that L.
- Green ≥ 1.03, grey within ±3% (`NOISE_PCT`, the run-to-run spread of a same-run ratio), red ≤ 0.97.""",
    "wall": """\
- Wall-clock median per launch over the stage's iterations (CUDA events), every variant in the same run.
- **%** = (variant − first column) ÷ first column × 100 — the change in **time**: negative = faster. `−6.0%` = 6% less time than the first column.
- Green = at least 3% less time, red = at least 3% more, grey = within ±3% (`NOISE_PCT`).""",
    "pdl": """\
- **%** = (time with PDL − time without PDL) ÷ time without PDL × 100, harness median per L, both measured back to back in the same run. Negative = PDL saves time.
- Green ≤ −3%, red ≥ +3%, grey within ±3% (`NOISE_PCT`).""",
    "fp6prof": """\
- **time ratio** = this row's Nsight Compute kernel time ÷ the first row's at the same L; **L2 bytes ratio** = L2 read sectors ÷ the first row's; **ideal** = this row's KV bits ÷ the first row's — the ratio a purely bandwidth-bound kernel would show.
- **bytes / L2 request** = 32 B × L2 read sectors ÷ L2 read requests.""",
    "kvcache": """\
- **KV tokens** and KV memory come from vLLM's own startup log ("GPU KV cache size", "Available KV cache memory") for the same model and memory budget.
- **vs first** = this row's KV tokens ÷ the first row's (`1.33×` = 33% more tokens).
- **fits @ L** = ⌊KV tokens ÷ (L + 2,048 prompt + 128 generated)⌋ — whole requests of the e2e bench's shape that fit at once (`max-num-seqs` ignored).""",
    "spec": """\
- **tokens/step** = 1 + accepted draft tokens ÷ verify steps, from vLLM's speculative-decoding counters read before and after the block; **accept @i** = the share of verify steps whose i-th draft was accepted.
- **decode t/s per request** = (generated tokens − 1) ÷ (time of the last token − time of the first token); **aggregate t/s** = all generated tokens ÷ the block's wall time.
- ★ = the draft count with the highest decode t/s for that config and concurrency.""",
    "accuracy": """\
- **Score** = points ÷ maximum points × 100, each scenario 2 (pass), 1 (partial) or 0 (fail); GSM8K / MMLU = accuracy in %; IFEval = prompt-level accuracy in %.
- **(±)** = this row's score − the baseline row's score, in percentage **points**, not a relative change: 86.5 vs 88.0 is −1.5 points (a relative change would be −1.7%).
- **n** = runs averaged (`--trials`); **completion** = the share of scenarios the endpoint answered — compare rows only at equal completion.""",
}


def _signed(v: float) -> str:
    return f"{'−' if v < 0 else '+'}{abs(v):.1f}%"


def _how_e2e(ctx: Any) -> str:
    ex = getattr(ctx, "e2e_example", None)
    noise = getattr(ctx, "e2e_noise_pct", "3")
    lines = [
        "- **Throughput** (decode, prefill — tokens/s, higher is better): change = (Mjolnir − official) ÷ official × 100. "
        "`+57%` = 1.57× the tokens per second. On the infographic a change of +100% or more is written as the multiple "
        "Mjolnir ÷ official (`+200%` = `3×`).",
        "- **Latency** (TTFT, ITL — milliseconds, lower is better): change = (Mjolnir − official) ÷ official × 100, so a "
        "negative value is a **latency reduction**: `−27%` = 27% fewer milliseconds. It is *not* the reciprocal speedup "
        "official ÷ Mjolnir − 1, which reads larger for the same pair.",
    ]
    if ex:
        lines.append(
            f"- **Worked example** ({ex['ctx']} context, 1 request): TTFT {ex['a']:,.0f} → {ex['b']:,.0f} ms: "
            f"({ex['b']:,.0f} − {ex['a']:,.0f}) ÷ {ex['a']:,.0f} × 100 = **{_signed(ex['change'])}** (a latency reduction "
            f"of {-ex['change']:.1f}%). The reciprocal speedup would be {ex['a']:,.0f} ÷ {ex['b']:,.0f} = {ex['ratio']:.2f}× "
            f"({_signed((ex['ratio'] - 1) * 100)}) — the same measurement, a different convention."
        )
    lines += [
        "- **Values**: the mean over the measured runs per cell (6 measured after 2 warm-ups, llama-benchy): decode = "
        "generated tokens ÷ generation time per request (summed over requests at higher concurrency), prefill = prompt "
        "tokens ÷ prompt-processing time, TTFT = time to the first content token, ITL = 1000 ÷ the per-request decode rate.",
        f"- **Colors**: green = better by at least {noise}%, grey = within ±{noise}% "
        "(`E2E_NOISE_PCT`), red = worse by at least that. With speculative decoding a single decode cell moves more "
        "between runs (acceptance depends on the prompt), so repeat (`--repeat`) before reading a few-% difference.",
    ]
    return "\n".join(lines)


def how(ctx: Any, kind: str) -> str:
    """A collapsed "How this is calculated" panel for the tables / figures of
    ``kind`` (GitHub renders the markdown inside ``<details>``). The e2e
    panel reads the context's noise band (any context carrying
    ``e2e_noise_pct`` works — the page contexts do too)."""
    body = _how_e2e(ctx) if kind == "e2e" else _HOW[kind]
    return f"<details>\n<summary>How the % and × here are calculated</summary>\n\n{body}\n\n</details><br />"


# ── rendering ───────────────────────────────────────────────────────────────


def _manifest(layout: RepoLayout) -> dict:
    p = layout.charts_dir / "manifest.json"
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _env(layout: RepoLayout, ctx: Any, out_path: Path) -> jinja2.Environment:
    import os

    manifest = _manifest(layout)
    root_rel = os.path.relpath(layout.repo_root or out_path.parent, out_path.parent).replace(os.sep, "/")
    prefix = "" if root_rel == "." else root_rel + "/"

    def chart(name: str, alt: str = "", width: str = "100%", pair: tuple[str, str] | None = None) -> str:
        """A plain ``<img>`` for a manifest chart. A pair-scoped entry
        (manifest v2) resolves only for the page's own (platform, model)
        pair — ``pair`` overrides it (the config pages' kernel sections
        quote the headline pair's charts); entries without an identity
        (legacy flat paths, hero, accuracy) resolve everywhere."""
        m = manifest.get(name)
        if not m:
            return ""
        want = pair or getattr(ctx, "chart_pair", None)
        if want is not None and m.get("platform") is not None and (m["platform"], m.get("model")) != want:
            return ""
        return f'<img src="{prefix + m["dark"]}" alt="{alt or name}" width="{width}">'

    env = jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
        # repo-root-relative, so templates can {% include %} shared partials
        # (the kernel-index partial the platform pages embed)
        loader=jinja2.FileSystemLoader(str(layout.repo_root or out_path.parent)),
    )
    # jinja2 leaves Environment.globals unannotated, so a checker infers the
    # DEFAULT_NAMESPACE value union for it: cast to the mapping it actually is
    cast(dict[str, Any], env.globals).update(
        how=lambda kind: how(ctx, kind),
        ctx=ctx,
        color=color,
        chart=chart,
        has=lambda stage: stage in getattr(ctx, "runs", {}),
        root=prefix,
        stages=STAGES,
        platform=getattr(ctx, "platform", None),
        model_short=str(getattr(ctx, "model", DEFAULT_MODEL)).split("/")[-1],
    )
    return env


def render_file(layout: RepoLayout, tpl: Path, out: Path, ctx: Any) -> Path:
    env = _env(layout, ctx, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(env.from_string(tpl.read_text()).render())
    return out


def render_all(layout: RepoLayout, root_override: tuple[Path, Path] | None = None) -> list[Path]:
    """Render every target (plan §1): the root README, the platform tree
    (``docs/platforms/`` — index, per-platform pages and the kernel case
    studies), every model README and every config's README (the merged
    Benchmarks section included) — plus the docker README. The target set and
    the contexts come from :mod:`mjolnir_cli.readme_targets`.
    ``root_override`` (template, out) redirects the root README render (the
    ``plot --readme-template --readme-out`` flags). The CLI reference is not
    written here — it needs the root app, which only the ``plot`` command
    holds (``clidoc.write_reference``)."""
    root = layout.require_repo("rendering the README")
    written: list[Path] = []
    docker_tpl = root / "docker" / "vllm-thor" / "README.md.j2"
    if docker_tpl.exists():
        written.append(render_file(layout, docker_tpl, docker_tpl.with_suffix(""), collect_context(layout)))
    for target in readme_targets.discover_targets(layout):
        for tpl, out, ctx in target.outputs(root, layout):
            if root_override and out == root / "README.md":
                tpl, out = root_override
            if not tpl.is_file():
                raise FileNotFoundError(f"template not found: {tpl}")
            written.append(render_file(layout, tpl, out, ctx))
    for p in written:
        print(f"[mjolnir] wrote {p}")
    return written


def render_readme(layout: RepoLayout, template: Path | None = None, out: Path | None = None) -> list[Path]:
    """Render the full target set (:func:`render_all`). ``template``/``out``
    override the root README's template and output (the ``mjolnir plot
    --readme-template/--readme-out`` flags); the other targets always render
    to their spec paths."""
    root = layout.require_repo("rendering the README")
    if template is None and out is None:
        return render_all(layout)
    tpl = Path(template) if template else root / "README.md.j2"
    dest = Path(out) if out else root / "README.md"
    if not tpl.exists():
        raise FileNotFoundError(f"README template not found: {tpl}")
    return render_all(layout, root_override=(tpl, dest))
