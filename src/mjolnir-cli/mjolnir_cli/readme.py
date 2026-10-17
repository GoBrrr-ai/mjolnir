"""README + docs generation — every number, table and chart reference is
computed from the committed raw data.

``mjolnir plot --readme`` renders:

* ``README.md``                  from ``README.md.j2``
* ``docs/kernel/**/*.md``        from the ``*.md.j2`` next to them
* ``docker/vllm-thor/README.md`` from ``docker/vllm-thor/README.md.j2`` — the
  Docker Hub description: plain markdown only (Docker Hub does not render inline
  HTML), absolute image / link URLs (no repo file hosting to resolve relatives
  against), no charts or data tables, well under Docker Hub's 25,000-byte
  description limit. ``docker-image.yml`` pushes it via
  ``peter-evans/dockerhub-description``.

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
from pathlib import Path

import jinja2

from mjolnir_cli.config import RepoLayout
from mjolnir_cli.readme_context import ReadmeContext, collect_context, how
from mjolnir_cli.readme_fmt import (
    L_label,
    color,
    fmt_us,
    kvcache_label,
    pct_md,
    ratio_md,
    ratio_pct,
    spec_label,
)
from mjolnir_cli.readme_tables import TABLES, _findings
from mjolnir_cli.verify.stages import STAGES

__all__ = [
    "L_label",
    "TABLES",
    "_findings",
    "fmt_us",
    "kvcache_label",
    "pct_md",
    "ratio_md",
    "ratio_pct",
    "spec_label",
    "ReadmeContext",
    "collect_context",
    "render_file",
    "render_readme",
]


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
        how=lambda kind: how(ctx, kind),
        ctx=ctx,
        color=color,
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
    from mjolnir_cli import clidoc
    from mjolnir_cli.cli import app as cli_app

    cli_md = root / "docs" / "cli.md"
    cli_md.parent.mkdir(parents=True, exist_ok=True)
    cli_md.write_text(clidoc.reference(cli_app))
    written = [render_file(layout, tpl, dest, ctx), cli_md]
    docker_tpl = root / "docker" / "vllm-thor" / "README.md.j2"
    if docker_tpl.exists():
        written.append(render_file(layout, docker_tpl, docker_tpl.with_suffix(""), ctx))
    for t in sorted((root / "docs").rglob("*.md.j2")):
        written.append(render_file(layout, t, t.with_suffix(""), ctx))
    for p in written:
        print(f"[mjolnir] wrote {p}")
    return written
