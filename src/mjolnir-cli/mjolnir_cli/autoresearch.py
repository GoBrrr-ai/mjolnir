"""``mjolnir autoresearch`` — an autonomous kernel-research loop with a ledger.

After karpathy/autoresearch: an agent edits ONE file, runs a FIXED,
time-boxed evaluation, keeps the change if the metric improved and resets it
otherwise — forever, while every experiment lands on a ledger. Here:

* the editable file  — ``src/mjolnir/mjolnir/t5_decode.py`` (the kernel);
* the fixed harness  — ``mjolnir autoresearch evaluate``: the quick
  correctness subset (``mjolnir verify correctness --quick``) + an ncu sweep
  of FlashInfer vs Mjolnir (harness default) at every L;
* the metric         — the geomean over L of FlashInfer time / Mjolnir time,
  both measured in the SAME run (drift on the shared Thor cancels out);
* the decision       — made by this module, not the agent: ``keep`` if
  correctness passes and the score beats the best kept score by more than
  ``--min-gain`` (default 2%), ``discard`` otherwise, ``crash`` if a stage
  failed to produce data. The agent then advances or resets the branch.
* the ledger         — ``research/ledger.jsonl`` (+ a ``ledger.tsv`` view):
  one row per experiment with the hypothesis, the agents involved, the
  sources (URLs / arXiv ids), the commit, the per-L numbers and the verify
  run ids. Untracked while the loop runs (a discard is a ``git reset
  --hard``, which must not rewind the notebook); ``mjolnir autoresearch
  export`` snapshots it into ``research/runs/<tag>/`` for committing.

The loop itself is an OpenCode session (``mjolnir autoresearch run``)
driving ``.opencode/agent/autoresearch.md`` (the orchestrator; the helpers it delegates to are user-defined OpenCode agents)
→ ``web-researcher`` / ``arxiv-researcher`` (ideas + sources) →
``python-engineer`` (the change) → ``debugger`` (crashes, correctness
failures). The program they follow is ``research/program.md``.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from mjolnir_cli.config import load_layout
from mjolnir_cli.legs import KERNEL
from mjolnir_cli.verify import results

console = Console()
app = typer.Typer(
    help="Autonomous kernel research: OpenCode agents propose, implement and test kernel changes; "
    "every experiment is scored by a fixed harness and recorded on the ledger.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

KERNEL_FILE = Path("src/mjolnir/mjolnir/t5_decode.py")
LEDGER_DIR = Path("research")
TSV_COLUMNS = ["id", "ts", "commit", "status", "score", "best_before", "correctness", "agent", "hypothesis"]


# ── ledger ──────────────────────────────────────────────────────────────────


def ledger_path(root: Path) -> Path:
    return root / LEDGER_DIR / "ledger.jsonl"


def load_ledger(root: Path) -> list[dict]:
    p = ledger_path(root)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def best_kept(rows: list[dict]) -> dict | None:
    kept = [r for r in rows if r.get("status") in ("keep", "baseline") and r.get("score")]
    return max(kept, key=lambda r: r["score"]) if kept else None


def append_ledger(root: Path, row: dict) -> None:
    d = root / LEDGER_DIR
    d.mkdir(parents=True, exist_ok=True)
    with (d / "ledger.jsonl").open("a") as f:
        f.write(json.dumps(row, sort_keys=False) + "\n")
    tsv = d / "ledger.tsv"
    new = not tsv.exists()
    with tsv.open("a") as f:
        if new:
            f.write("\t".join(TSV_COLUMNS) + "\n")
        vals = []
        for c in TSV_COLUMNS:
            v = row.get(c, "")
            if isinstance(v, float):
                v = f"{v:.4f}"
            vals.append(str(v).replace("\t", " ").replace("\n", " "))
        f.write("\t".join(vals) + "\n")


def decide(score: float | None, correct: bool, best: float | None, min_gain: float) -> str:
    """The fixed keep/discard rule (the agent never grades its own work)."""
    if score is None:
        return "crash"
    if not correct:
        return "discard"
    if best is None:
        return "baseline"
    return "keep" if score > best * (1 + min_gain) else "discard"


# ── git helpers ─────────────────────────────────────────────────────────────


def _git(root: Path, *args: str, check: bool = True) -> str:
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout.strip()


def _root() -> Path:
    return load_layout().require_repo("autoresearch")


# ── commands ────────────────────────────────────────────────────────────────


def _init(root: Path, tag: str, base: str, evaluate_baseline: bool, image: Optional[str]) -> int:
    branch = f"autoresearch/{tag}"
    if _git(root, "branch", "--list", branch):
        console.print(f"[red]{branch} already exists — pick a fresh tag (a run never reuses a branch)[/red]")
        return 4
    if _git(root, "status", "--porcelain", "--", str(KERNEL_FILE)):
        console.print(f"[red]{KERNEL_FILE} has uncommitted changes — commit or stash them first[/red]")
        return 4
    if load_ledger(root):
        console.print(
            "[red]research/ledger.jsonl holds a previous run — `mjolnir autoresearch export` it, then delete it[/red]"
        )
        return 4
    _git(root, "checkout", "-b", branch, base)
    console.print(f"[green]on {branch}[/green] (from {base})")
    if evaluate_baseline:
        return _evaluate(root, hypothesis="baseline: the starting kernel", agent="human", sources="", image=image)
    return 0


@app.command()
def init(
    tag: str = typer.Argument(..., help="run tag, e.g. oct11 — the branch is autoresearch/<tag>"),
    base: str = typer.Option("main", "--base", help="branch to start from"),
    evaluate_baseline: bool = typer.Option(True, "--baseline/--no-baseline", help="score the starting kernel now"),
    image: Optional[str] = typer.Option(None, "--image"),
):
    """Create the research branch and score the starting kernel (the baseline row)."""
    raise typer.Exit(_init(_root(), tag, base, evaluate_baseline, image))


def _evaluate(
    root: Path,
    hypothesis: str,
    agent: str,
    sources: str,
    image: Optional[str],
    min_gain: float = 0.02,
    ls: Optional[str] = None,
    iters: int = 50,
    gate: bool = True,
    skip_correctness: bool = False,
) -> int:
    from mjolnir_cli.verify.cli import PREFLIGHT_RC, execute
    from mjolnir_cli.verify.stages import ALL_LS

    rows = load_ledger(root)
    best = best_kept(rows)
    commit = _git(root, "rev-parse", "--short", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain", "--", str(KERNEL_FILE)))
    if dirty:
        console.print(
            "[yellow]the kernel has uncommitted changes — commit the experiment first "
            "(the ledger records commits)[/yellow]"
        )
        return 4
    correct, corr_run = True, None
    if not skip_correctness:
        rc, doc = execute(
            "correctness",
            {"dtypes": "fp8 bf16", "match": "", "fail_fast": True, "timeout": 300.0, "quick": True},
            {},
            image=image,
            ls=None,
            iters=iters,
            gate=gate,
            dry_run=False,
            timeout=300.0,
            save=True,
            plot=False,
        )
        if rc == PREFLIGHT_RC:
            return rc  # no docker / no image: an environment problem, not an experiment — no ledger row
        correct = rc == 0
        corr_run = doc["run_id"] if doc else None
    score, per_L, sweep_run = None, {}, None
    if correct:
        rc, doc = execute(
            "sweep",
            {"legs": KERNEL["autoresearch.evaluate"]},
            {},  # never a headline run
            image=image,
            ls=ls,
            iters=iters,
            gate=gate,
            dry_run=False,
            timeout=300.0,
            save=True,
            plot=False,
        )
        if doc:
            by = results.series_by_id(doc)
            if "fi" in by and "mjolnir" in by:
                sp = results.speedup(by["fi"], by["mjolnir"])
                want = set(int(x) for x in (ls or " ".join(map(str, ALL_LS))).replace(",", " ").split())
                if sp and set(sp) >= want:
                    score = results.geomean(sp.values())
                    per_L = {str(k): round(v, 4) for k, v in sp.items()}
            sweep_run = doc["run_id"]
    status = decide(score, correct, best["score"] if best else None, min_gain)
    row = {
        "id": len(rows) + 1,
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": commit,
        "status": status,
        "score": round(score, 4) if score else None,
        "best_before": best["score"] if best else None,
        "min_gain": min_gain,
        "correctness": "pass" if correct else "fail",
        "per_L": per_L,
        "agent": agent,
        "hypothesis": hypothesis,
        "sources": [s.strip() for s in sources.split(",") if s.strip()],
        "verify_runs": {"correctness": corr_run, "sweep": sweep_run},
    }
    append_ledger(root, row)
    color = {"keep": "green", "baseline": "cyan", "discard": "yellow", "crash": "red"}[status]
    console.print(
        f"\n[bold {color}]{status.upper()}[/bold {color}]  score {row['score']}  "
        f"(best kept {row['best_before']}, needs > {row['best_before'] and round(row['best_before'] * (1 + min_gain), 4)})"
        f"  correctness {row['correctness']}  commit {commit}"
    )
    if status in ("discard", "crash"):
        console.print(f"[dim]  revert: git reset --hard {best['commit'] if best else 'HEAD~1'}[/dim]")
    return 0


@app.command()
def evaluate(
    hypothesis: str = typer.Option(..., "--hypothesis", "-m", help="what this experiment tries, one line"),
    agent: str = typer.Option("autoresearch", "--agent", help="who proposed/implemented it"),
    sources: str = typer.Option("", "--sources", help="comma-separated URLs / arXiv ids behind the idea"),
    min_gain: float = typer.Option(0.02, "--min-gain", help="required geomean gain over the best kept score"),
    ls: Optional[str] = typer.Option(None, "--ls", help="L values (default: all 8; a subset is never a keep)"),
    iters: int = typer.Option(50, "--iters"),
    gate: bool = typer.Option(True, "--gate/--no-gate"),
    image: Optional[str] = typer.Option(None, "--image"),
):
    """Score the committed kernel: quick correctness + FlashInfer vs Mjolnir sweep;
    append the result and the fixed keep/discard decision to the ledger."""
    raise typer.Exit(_evaluate(_root(), hypothesis, agent, sources, image, min_gain, ls, iters, gate))


@app.command()
def ledger(
    tail: int = typer.Option(20, "--tail"),
    as_json: bool = typer.Option(False, "--json", "-j"),
):
    """Show the ledger (newest last)."""
    rows = load_ledger(_root())
    if as_json:
        typer.echo(json.dumps(rows[-tail:], indent=2))
        return
    if not rows:
        console.print("[yellow]empty ledger — start with: mjolnir autoresearch init <tag>[/yellow]")
        return
    best = best_kept(rows)
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    for c in ("#", "when", "commit", "status", "score", "agent", "hypothesis"):
        t.add_column(c, justify="right" if c in ("#", "score") else "left")
    colors = {"keep": "green", "baseline": "cyan", "discard": "yellow", "crash": "red"}
    for r in rows[-tail:]:
        st = r.get("status", "")
        mark = " ★" if best and r is best else ""
        t.add_row(
            str(r["id"]),
            r["ts"][5:16].replace("T", " "),
            r["commit"],
            f"[{colors.get(st, 'white')}]{st}[/]",
            f"{r['score']:.4f}{mark}" if r.get("score") else "—",
            r.get("agent", ""),
            r.get("hypothesis", "")[:70],
        )
    console.print(t)


@app.command()
def status():
    """Best kept score, experiment counts, the current branch."""
    root = _root()
    rows = load_ledger(root)
    best = best_kept(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    console.print(
        f"branch {_git(root, 'rev-parse', '--abbrev-ref', 'HEAD')} · {len(rows)} experiments · "
        + " · ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    )
    if best:
        console.print(
            f"best: [bold]{best['score']:.4f}×[/bold] vs FlashInfer (geomean over L) at {best['commit']} "
            f"— {best['hypothesis']}"
        )


@app.command()
def run(
    tag: Optional[str] = typer.Option(None, "--tag", help="start a new run first (init <tag>)"),
    model: Optional[str] = typer.Option(None, "--model", help="OpenCode model (provider/model) for the orchestrator"),
    prompt: str = typer.Option(
        "Read research/program.md and run the loop: setup is done when the ledger has a baseline row. "
        "Never stop to ask; record every experiment with `mjolnir autoresearch evaluate`.",
        "--prompt",
    ),
):
    """Launch the loop: an OpenCode session with the `autoresearch` agent."""
    root = _root()
    if shutil.which("opencode") is None:
        console.print("[red]opencode not found — install it: https://opencode.ai (npm i -g opencode-ai)[/red]")
        raise typer.Exit(2)
    if tag:
        rc = _init(root, tag, "main", True, None)
        if rc != 0:
            raise typer.Exit(rc)
    elif not load_ledger(root):
        console.print("[yellow]no ledger yet — run: mjolnir autoresearch init <tag>  (or pass --tag)[/yellow]")
        raise typer.Exit(4)
    cmd = ["opencode", "run", "--agent", "autoresearch"]
    if model:
        cmd += ["--model", model]
    cmd += [prompt]
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    raise typer.Exit(subprocess.call(cmd, cwd=root))


@app.command()
def export(
    tag: Optional[str] = typer.Option(None, "--tag", help="run tag (default: from the autoresearch/<tag> branch)"),
):
    """Snapshot the ledger into research/runs/<tag>/ (ledger.jsonl, ledger.tsv,
    SUMMARY.md) — commit that to keep the run's notebook with the repo."""
    root = _root()
    rows = load_ledger(root)
    if not rows:
        console.print("[yellow]empty ledger[/yellow]")
        raise typer.Exit(4)
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    tag = tag or (branch.split("/", 1)[1] if branch.startswith("autoresearch/") else branch)
    out = root / LEDGER_DIR / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    for name in ("ledger.jsonl", "ledger.tsv"):
        src = root / LEDGER_DIR / name
        if src.exists():
            shutil.copy(src, out / name)
    best = best_kept(rows)
    base = next((r for r in rows if r["status"] == "baseline"), None)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines = [
        f"# autoresearch run `{tag}`",
        "",
        f"- experiments: {len(rows)} (" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ")",
        f"- baseline: {base['score']}× vs FlashInfer at `{base['commit']}`" if base and base.get("score") else "",
        f"- best: **{best['score']}×** at `{best['commit']}` — {best['hypothesis']}" if best else "",
        "",
        "| # | commit | status | score | agent | hypothesis | sources |",
        "|--:|---|---|--:|---|---|---|",
    ]
    for r in rows:
        src = " ".join(f"[{i + 1}]({u})" if u.startswith("http") else u for i, u in enumerate(r.get("sources", [])))
        lines.append(
            f"| {r['id']} | `{r['commit']}` | {r['status']} | {r.get('score') or '—'} | {r.get('agent', '')} "
            f"| {r.get('hypothesis', '').replace('|', '/')} | {src} |"
        )
    (out / "SUMMARY.md").write_text("\n".join(x for x in lines if x is not None) + "\n")
    console.print(f"[green]exported {len(rows)} rows -> {out}[/green]")
