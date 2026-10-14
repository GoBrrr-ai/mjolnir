"""clidoc: docs/cli.md and the README's command table come from the typer commands."""

from __future__ import annotations

from mjolnir_cli import clidoc
from mjolnir_cli.cli import app


def test_reference_covers_every_command():
    cmds = clidoc.walk(app)
    ref = clidoc.reference(app)
    assert len(cmds) > 5
    for c in cmds[1:]:
        assert f"## `{c.path}`" in ref
    assert "<run>" not in ref.replace("`<run>`", "")  # placeholders escaped outside code


def test_summary_lists_leaf_commands_only():
    cmds = clidoc.walk(app)
    table = clidoc.summary_table(app)
    for c in cmds[1:]:
        assert (f"[`{c.path}`]" in table) == (not c.group)


def test_rst_conversion():
    assert clidoc._rst("``a <b>`` and <c>") == "`a <b>` and &lt;c&gt;"
    assert clidoc.first_sentence("One. Two.") == "One."
