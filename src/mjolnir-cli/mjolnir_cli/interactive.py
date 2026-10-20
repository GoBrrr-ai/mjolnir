"""The interactive benchmark site (GitHub Pages): ``mjolnir plot --site <dir>``
writes the v2 tree — a static ``index.html`` shell, a small ``manifest.json``,
the real ``app.js``/``app.css`` assets and one lazy payload per measured
(platform, model) pair. Plotly (pinned, jsdelivr) draws the charts; colors
follow ``theme.py``'s validated palettes via app.css. The implementation lives
in :mod:`mjolnir_cli.site` (``data.py`` builders + ``page.py`` emission);
``build`` stays the CLI-facing entry point.
"""

from __future__ import annotations

from pathlib import Path

from mjolnir_cli.config import RepoLayout


def build(layout: RepoLayout, out_dir: Path, data: dict | None = None) -> Path:
    """Emit the site into ``out_dir``; return the ``index.html`` path.

    ``data`` (a preloaded ``load_verify`` result, the v1 single-page payload)
    is accepted for call compatibility and unused: v2 builds one payload per
    measured pair from the raw documents."""
    from mjolnir_cli.site.page import emit_site

    return emit_site(layout, out_dir, data)
