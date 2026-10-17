"""Pin the public surface of ``mjolnir_cli.cli``.

These tests describe behavior that must survive splitting cli.py into
modules: the names other code imports from it, the console script target,
and the --help of every sub-app (and the order its commands are registered
in). The import direction (nothing but ``__main__`` imports
``mjolnir_cli.cli``) is a row of ``ONLY_IMPORTED_BY`` in
test_module_boundaries.py.
"""

from __future__ import annotations

import re
from importlib.metadata import entry_points

import pytest
from typer.testing import CliRunner

runner = CliRunner()

# Registration order, read by hand from the @<sub>_app.command decorators in cli.py.
SUB_APP_COMMANDS = {
    "serve": ["up", "down", "status", "logs"],
    "model": ["use", "list"],
    "image": ["use", "list", "build", "gates"],
    "litellm": ["up", "down", "status", "logs", "config"],
    "perses": ["up", "down", "status", "logs"],
    "hw": ["setup", "status"],
    "bench": ["perf", "ab", "accuracy", "e2e", "compare"],
}

# Root commands (not sub-apps): their --help shows these options.
ROOT_COMMAND_OPTIONS = {
    "history": ["--limit", "--json"],
    "plot": ["--readme", "--readme-template", "--readme-out", "--site"],
}


def _listed_commands(help_text: str) -> list[str]:
    """Command names from the rich 'Commands' panel, in the order shown."""
    in_panel = False
    names = []
    for line in help_text.splitlines():
        if line.startswith("╭─ Commands"):
            in_panel = True
        elif in_panel and line.startswith("╰"):
            break
        elif in_panel:
            m = re.match(r"│ ([a-z][\w-]*)\s{2,}", line)
            if m:
                names.append(m.group(1))
    return names


def _invoke_help(*args: str) -> str:
    from mjolnir_cli.cli import app

    result = runner.invoke(app, [*args, "--help"])
    assert result.exit_code == 0, result.output
    return result.output


def test_cli_module_exports_the_names_other_code_imports():
    from mjolnir_cli.cli import _print_accuracy, _print_compare, _run_legs, _settings, app

    assert callable(app)
    for fn in (_settings, _run_legs, _print_compare, _print_accuracy):
        assert callable(fn)


def test_mjolnir_console_script_resolves_to_cli_app():
    scripts = [ep for ep in entry_points(group="console_scripts") if ep.name == "mjolnir"]
    assert [ep.value for ep in scripts] == ["mjolnir_cli.cli:app"]
    from mjolnir_cli.cli import app

    assert scripts[0].load() is app


@pytest.mark.parametrize("sub", sorted(SUB_APP_COMMANDS))
def test_sub_app_help_exits_zero_and_lists_commands_in_registration_order(sub):
    assert _listed_commands(_invoke_help(sub)) == SUB_APP_COMMANDS[sub]


@pytest.mark.parametrize("sub", sorted(SUB_APP_COMMANDS))
def test_every_sub_app_command_help_exits_zero(sub):
    for cmd in SUB_APP_COMMANDS[sub]:
        _invoke_help(sub, cmd)


@pytest.mark.parametrize("name", sorted(ROOT_COMMAND_OPTIONS))
def test_root_command_help_exits_zero_and_shows_its_options(name):
    out = _invoke_help(name)
    assert re.search(rf"Usage: \S+ {name} \[OPTIONS\]", out)
    for opt in ROOT_COMMAND_OPTIONS[name]:
        assert opt in out
