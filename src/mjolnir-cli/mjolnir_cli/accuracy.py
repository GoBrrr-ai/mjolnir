"""``mjolnir bench accuracy`` — does a serving config keep the model's quality?

Each leg (a config, optionally image-pinned, like ``bench ab``) is served in
turn and benchmarked with `tool-eval-bench
<https://github.com/SeraphimSerapis/tool-eval-bench>`_ over the OpenAI
endpoint. The default run is the standard 69 tool-calling scenarios plus the
28 Hard Mode ones (``tool-eval-bench run --hardmode``); any other
tool-eval-bench invocation is forwarded verbatim after ``--`` — GSM8K, MMLU
and IFEval go through the same adapter (``-- --gsm8k --mmlu --ifeval
--skip-tool-eval``).

tool-eval-bench keeps its results in ``./data/benchmarks.sqlite`` and
``./runs/`` of the directory it runs in; every leg runs in its own directory
under ``benchmarks/raw/accuracy/<run>/<leg>/``, so the database holds exactly
that leg's runs. :func:`summarize_db` reads it back into the committed
``accuracy.json`` (``mjolnir.accuracy/1``): per suite the score, per
tool-call category the share of points, per scenario the points — what the
docs table and charts are rendered from.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

FORMAT = "mjolnir.accuracy/1"
TOOL_REPO = "https://github.com/SeraphimSerapis/tool-eval-bench"
# pinned: a score is only comparable to one produced by the same scenario set
TOOL_COMMIT = "b60e2f00f6c44352a3680356e7c7948f777c5280"
TOOL_SPEC = f"tool-eval-bench[hf] @ git+{TOOL_REPO}.git@{TOOL_COMMIT}"
DEFAULT_ARGS = ("run", "--hardmode", "--seed", "42", "--timeout", "600")
HARD_CATEGORY = "P"  # tool-eval-bench's Hard Mode category

# suite -> (label, unit note); order = table / chart order
SUITES = {
    "tools": "Tool calls",
    "hardmode": "Hard Mode",
    "gsm8k": "GSM8K",
    "mmlu": "MMLU",
    "ifeval": "IFEval",
    "needle": "Needle",
}


class AccuracyError(RuntimeError):
    pass


def find_tool() -> list[str]:
    """The tool-eval-bench invocation: ``$MJOLNIR_TOOL_EVAL_BENCH`` if set,
    else the pinned commit through ``uvx`` (no install needed), else one on
    PATH."""
    env = os.environ.get("MJOLNIR_TOOL_EVAL_BENCH", "").strip()
    if env:
        return shlex.split(env)
    if shutil.which("uvx"):
        return ["uvx", "--from", TOOL_SPEC, "tool-eval-bench"]
    if shutil.which("tool-eval-bench"):
        return ["tool-eval-bench"]
    raise AccuracyError(f"tool-eval-bench not found — install uv (uvx runs it) or: uv tool install '{TOOL_SPEC}'")


def tool_args(passthrough: list[str] | tuple[str, ...], base_url: str) -> list[str]:
    """The forwarded tool-eval-bench arguments (the default run when none),
    pointed at the leg's server unless they name one."""
    args = list(passthrough) or list(DEFAULT_ARGS)
    if not any(a == "--base-url" or a.startswith("--base-url=") for a in args):
        args += ["--base-url", base_url]
    return args


# ── reading tool-eval-bench's database ──────────────────────────────────────


def _loads(v) -> dict:
    try:
        return json.loads(v) if v else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _pct(earned: float, mx: float) -> float | None:
    return round(earned / mx * 100, 2) if mx else None


def summarize_db(db: Path) -> dict:
    """tool-eval-bench's SQLite → ``{"suites": {suite: {...}}, "categories":
    {...}, "scenarios": {...}}``. Several completed runs of a kind (``--trials``)
    are averaged; each suite keeps its per-run scores."""
    if not db.exists():
        return {"suites": {}, "categories": {}, "scenarios": {}, "runs": []}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT run_id, created_at, status, run_type, scores_json, config_json FROM scenario_runs "
            "ORDER BY created_at"
        ).fetchall()
    finally:
        con.close()
    per_suite: dict[str, list[dict]] = {}
    cats: dict[str, dict] = {}
    scen: dict[str, list[float]] = {}
    runs = []
    for run_id, created, status, run_type, scores_json, config_json in rows:
        runs.append({"run_id": run_id, "created_at": created, "status": status, "run_type": run_type})
        if status != "completed":
            continue
        sc = _loads(scores_json)
        if run_type == "tool_eval":
            cs = sc.get("category_scores") or []
            std = [c for c in cs if c.get("category") != HARD_CATEGORY]
            hard = [c for c in cs if c.get("category") == HARD_CATEGORY]
            for name, group in (("tools", std), ("hardmode", hard)):
                e, m = sum(c.get("earned", 0) for c in group), sum(c.get("max", 0) for c in group)
                if m:
                    per_suite.setdefault(name, []).append(
                        {
                            "score": _pct(e, m),
                            "earned": e,
                            "max": m,
                            "pass": sum(c.get("pass_count", 0) for c in group),
                            "partial": sum(c.get("partial_count", 0) for c in group),
                            "fail": sum(c.get("fail_count", 0) for c in group),
                            "final_score": sc.get("final_score"),
                            "completion_rate": sc.get("completion_rate", 100.0),
                            "safety_warnings": len(sc.get("safety_warnings") or []),
                            "run_id": run_id,
                        }
                    )
            for c in cs:
                d = cats.setdefault(c["category"], {"label": c.get("label", c["category"]), "percent": []})
                if c.get("max"):
                    d["percent"].append(_pct(c.get("earned", 0), c["max"]))
            for r in sc.get("scenario_results") or []:
                sid = r.get("scenario_id")
                if sid is not None and r.get("points") is not None:
                    scen.setdefault(sid, []).append(float(r["points"]))
        else:
            acc = sc.get("accuracy", sc.get("final_score"))
            if acc is None:
                continue
            extra = {
                k: sc[k]
                for k in ("instruction_accuracy", "correct", "total", "errors", "completion_rate", "truncated")
                if k in sc
            }
            per_suite.setdefault(run_type, []).append({"score": round(float(acc), 2), **extra, "run_id": run_id})
    suites = {}
    for name, items in per_suite.items():
        scores = [i["score"] for i in items if i.get("score") is not None]
        if not scores:
            continue
        last = dict(items[-1])
        last.pop("score", None)
        suites[name] = {"score": round(mean(scores), 2), "scores": scores, "n": len(scores), **last}
    return {
        "suites": suites,
        "categories": {
            k: {"label": v["label"], "percent": round(mean(v["percent"]), 2)}
            for k, v in sorted(cats.items())
            if v["percent"]
        },
        "scenarios": {k: round(mean(v), 3) for k, v in sorted(scen.items())},
        "runs": runs,
    }


# ── the run ─────────────────────────────────────────────────────────────────


def leg_id(model: str, image: str, config: str) -> str:
    side = "official" if not _is_mjolnir(image) else "mjolnir"
    return f"{side}-{config}"


def _is_mjolnir(image: str) -> bool:
    from mjolnir_cli import e2e

    return e2e.is_mjolnir({"image": image})


def kv_label(rec: dict) -> str:
    kv = str(rec.get("kv_cache_dtype") or "auto")
    return {"fp6_e2m3": "fp6 e2m3 KV", "fp8": "fp8 KV", "fp8_e4m3": "fp8 KV", "auto": "bf16 KV"}.get(kv, f"{kv} KV")


def leg_label(rec: dict) -> str:
    """'official vLLM · FlashInfer · fp8 KV', 'Mjolnir · fp6 e2m3 KV + fp8 lm_head + 64K draft vocab', ...
    — every label names the KV-cache format."""
    from mjolnir_cli.config import BACKEND_LABELS

    cfg = str(rec.get("config", ""))
    kv = kv_label(rec)
    if not _is_mjolnir(str(rec.get("image", ""))):
        return f"official vLLM · {BACKEND_LABELS.get(cfg, cfg)} · {kv}"
    if "MJOLNIR" not in cfg:
        return f"Mjolnir image · {BACKEND_LABELS.get(cfg, cfg)} · {kv}"
    extra = "".join(
        f" + {name}"
        for tag, name in (("LMH8", "fp8 lm_head"), ("VT32K", "32K draft vocab"), ("VT64K", "64K draft vocab"))
        if tag in cfg.split("_")
    )
    return f"Mjolnir · {kv}{extra}"


def run_accuracy(
    s,
    layout,
    legs: list[tuple[str, str, str]],
    passthrough: list[str],
    restart: bool = True,
) -> Path:
    """Serve each ``(model, image, config)`` leg (restarted per leg unless
    ``restart`` is off: then the running server is the one leg), run
    tool-eval-bench against it, and write ``accuracy.json``. Returns its path."""
    from mjolnir_cli import dockerctl
    from mjolnir_cli.config import config_field

    tool = find_tool()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = layout.raw_dir / "accuracy" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "format": FORMAT,
        "run_id": run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": {"repo": TOOL_REPO, "commit": TOOL_COMMIT, "command": tool},
        "args": list(passthrough) or list(DEFAULT_ARGS),
        "legs": [],
    }
    out = out_dir / "accuracy.json"
    for model, image, config in legs:
        s.model, s.image, s.quant = model, image, config
        lid = leg_id(model, image, config)
        leg_dir = out_dir / lid
        leg_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 62}\n  accuracy leg: {model}/{config}  (image {image})\n{'=' * 62}", flush=True)
        rec = {
            "id": lid,
            "model": model,
            "image": image,
            "config": config,
            "kv_cache_dtype": config_field(s.config_path, "kv-cache-dtype") or "auto",
            "env": dockerctl.config_env(s),
        }
        rec["label"] = leg_label(rec)
        try:
            if restart:
                dockerctl.serve_down(s)
                dockerctl.serve_up(s, wait=True)
            cmd = tool + tool_args(passthrough, s.base_url)
            print(f"[mjolnir] {' '.join(shlex.quote(c) for c in cmd)}\n[mjolnir]   in {leg_dir}", flush=True)
            t0 = time.time()
            rc = subprocess.call(cmd, cwd=leg_dir)
            rec["exit_code"] = rc
            rec["wall_s"] = round(time.time() - t0, 1)
            if rc not in (0, 2):  # 2: e.g. --fail-on-safety tripped — the scores still stand
                print(f"[mjolnir] tool-eval-bench exited {rc} — keeping what it saved", file=sys.stderr)
            dockerctl.dump_server_log(None, leg_dir / "vllm-server.log")
        finally:
            if restart:
                print(f"  leg done: stopping the vLLM server ({image})", flush=True)
                dockerctl.serve_down(s)
        rec.update(summarize_db(leg_dir / "data" / "benchmarks.sqlite"))
        doc["legs"].append(rec)
        out.write_text(json.dumps(doc, indent=1) + "\n")
    return out


# ── reading the committed results ───────────────────────────────────────────


def load_runs(layout) -> list[dict]:
    root = layout.raw_dir / "accuracy"
    docs = []
    for p in sorted(root.glob("*/accuracy.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("format") == FORMAT:
            docs.append(d)
    return sorted(docs, key=lambda d: d.get("run_id", ""))


def latest_legs(docs: list[dict]) -> list[dict]:
    """One row per (model, image, config): every suite from the newest run
    that measured it (an A/B run and a later GSM8K run merge into one row).
    Order: first appearance."""
    rows: dict[tuple, dict] = {}
    for d in docs:
        for leg in d.get("legs", []):
            key = (leg.get("model"), leg.get("image"), leg.get("config"))
            row = rows.setdefault(
                key,
                {k: leg.get(k) for k in ("id", "model", "image", "config", "label", "kv_cache_dtype", "env")}
                | {"suites": {}, "categories": {}, "scenarios": {}, "runs": {}},
            )
            row["kv_cache_dtype"] = leg.get("kv_cache_dtype") or row.get("kv_cache_dtype")
            row["label"] = leg_label(row)  # recomputed: labels improve after a run is stored
            for name, suite in (leg.get("suites") or {}).items():
                row["suites"][name] = suite
                row["runs"][name] = {"run_id": d.get("run_id"), "args": d.get("args")}
            if (leg.get("suites") or {}).keys() & {"tools", "hardmode"}:
                row["categories"] = leg.get("categories") or row["categories"]
                row["scenarios"] = leg.get("scenarios") or row["scenarios"]
    out = [r for r in rows.values() if r["suites"]]
    if len({r.get("model") for r in out}) > 1:  # rows of several models: say which
        for r in out:
            r["label"] = f"{str(r.get('model', '')).split('/')[-1]} · {r['label']}"
    return out


def baseline(rows: list[dict], model: str | None = None) -> dict | None:
    """The reference row for deltas, among ``model``'s rows (all rows when
    None): native FlashInfer on the official image when measured, else the
    first official-image row, else the first row. Rows of another model are
    never a baseline — their scores are not comparable."""
    rows = [r for r in rows if model is None or r.get("model") == model]
    official = [r for r in rows if not _is_mjolnir(str(r.get("image", "")))]
    for r in official:
        if r.get("config") == "NVFP4":
            return r
    return (official or rows or [None])[0]


def baseline_for(rows: list[dict], row: dict) -> dict | None:
    """The baseline ``row`` is compared with: its own model's."""
    return baseline(rows, row.get("model"))


def suites_in(rows: list[dict]) -> list[str]:
    seen = {n for r in rows for n in r["suites"]}
    return [n for n in SUITES if n in seen] + sorted(seen - set(SUITES))


def suite_label(name: str) -> str:
    return SUITES.get(name, name.upper())
