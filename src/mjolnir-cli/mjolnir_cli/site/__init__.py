"""The interactive benchmark site (GitHub Pages) — a static shell, a small
manifest and one lazy payload per measured (platform, model) pair.

``mjolnir plot --site <dir>`` writes the v2 tree (design §9): the shell never
embeds data — ``app.js`` fetches ``manifest.json`` first, then the selected
pair's ``data/<platform>/<model-short>.json``. Plotly is a pinned CDN
reference, never a bundled dependency. ``mjolnir_cli.interactive.build`` is
the thin entry point the CLI calls.
"""
