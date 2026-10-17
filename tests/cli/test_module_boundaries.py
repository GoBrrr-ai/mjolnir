"""Pins mjolnir_cli's import graph, so a later change that re-tangles it fails CI.

Read straight from the source with ``ast`` (every import counts, function-local
ones included; only ``if TYPE_CHECKING:`` blocks are skipped, since a type-only
import never runs):

* the ``readme`` package is layered fmt <- tables / findings <- context / how
  <- ``__init__``, and nothing in it reaches up to ``charts`` or ``cli``;
* ``charts`` never imports ``readme`` (the labels both need live in
  ``mjolnir_cli.labels``, the verify data loader in ``verify.latest``);
* a name with a leading underscore stays inside its module: governed modules
  import none from elsewhere, and nobody imports one from a governed module.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "mjolnir-cli"
PKG = "mjolnir_cli"
README = "mjolnir_cli.readme"

# Modules whose private names must not cross a module boundary, in either
# direction. Later PRs of the file-length split (#33) extend this set as they
# give charts and commands the same treatment.
GOVERNED = {"mjolnir_cli.readme", "mjolnir_cli.labels", "mjolnir_cli.verify.latest"}


@dataclass(frozen=True)
class Use:
    """One reference from a module to another mjolnir_cli module."""

    target: str  # the module imported (or imported from)
    name: str | None  # the name taken from it (`from x import name`, `x.name`), None for the module itself
    line: int


def _module_name(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _modules() -> dict[str, Path]:
    return {_module_name(p): p for p in sorted((SRC / PKG).rglob("*.py"))}


def _is_type_checking(node: ast.If) -> bool:
    t = node.test
    return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
        isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
    )


def _private(name: str | None) -> bool:
    return bool(name) and name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def uses(source: str, module: str, known: set[str]) -> list[Use]:
    """Every mjolnir_cli module ``source`` (the module ``module``) refers to."""
    is_pkg = module in known and any(k.startswith(module + ".") for k in known)
    package = module if is_pkg else module.rpartition(".")[0]
    out: list[Use] = []
    aliases: dict[str, str] = {}  # local name -> the mjolnir_cli module it is bound to

    def resolve(node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        base = package.split(".")
        base = base[: len(base) - node.level + 1]
        return ".".join(base + ([node.module] if node.module else []))

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and _is_type_checking(node):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == PKG or a.name.startswith(PKG + "."):
                    out.append(Use(a.name, None, node.lineno))
                    aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom):
            src = resolve(node)
            if src == PKG or src.startswith(PKG + "."):
                for a in node.names:
                    sub = f"{src}.{a.name}"
                    if sub in known:  # `from mjolnir_cli import readme` imports a module
                        out.append(Use(sub, None, node.lineno))
                        aliases[a.asname or a.name] = sub
                    else:
                        out.append(Use(src, a.name, node.lineno))
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            out.append(Use(aliases[node.value.id], node.attr, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(source))
    return out


@pytest.fixture(scope="module")
def graph() -> dict[str, list[Use]]:
    mods = _modules()
    known = set(mods)
    return {m: uses(p.read_text(), m, known) for m, p in mods.items()}


def _in(module: str, root: str) -> bool:
    return module == root or module.startswith(root + ".")


def _readme_deps(graph: dict[str, list[Use]], module: str) -> set[str]:
    assert module in graph, f"{module} is missing — the readme package layout changed"
    return {u.target for u in graph[module] if _in(u.target, README) and u.target != module}


def test_walker_sees_function_local_and_attribute_uses_but_not_type_checking_blocks():
    src = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from mjolnir_cli import cli\n"
        "def f():\n    from mjolnir_cli import charts\n    return charts._x\n"
        "from .fmt import _y\n"
    )
    known = {"mjolnir_cli", "mjolnir_cli.charts", "mjolnir_cli.cli", "mjolnir_cli.readme", "mjolnir_cli.readme.fmt"}
    found = {(u.target, u.name) for u in uses(src, "mjolnir_cli.readme.tables", known)}
    assert found == {("mjolnir_cli.charts", None), ("mjolnir_cli.charts", "_x"), ("mjolnir_cli.readme.fmt", "_y")}


def test_readme_fmt_imports_no_other_readme_module(graph):
    assert _readme_deps(graph, "mjolnir_cli.readme.fmt") == set()


@pytest.mark.parametrize("module", ["mjolnir_cli.readme.tables", "mjolnir_cli.readme.findings"])
def test_readme_tables_and_findings_import_only_fmt_from_readme(graph, module):
    assert _readme_deps(graph, module) <= {"mjolnir_cli.readme.fmt"}


@pytest.mark.parametrize("module", ["mjolnir_cli.readme.context", "mjolnir_cli.readme.how"])
def test_readme_context_and_how_import_only_fmt_tables_and_findings(graph, module):
    allowed = {"mjolnir_cli.readme.fmt", "mjolnir_cli.readme.tables", "mjolnir_cli.readme.findings"}
    assert _readme_deps(graph, module) <= allowed


def test_nothing_in_readme_imports_charts_or_cli(graph):
    readme_mods = [m for m in graph if _in(m, README)]
    assert len(readme_mods) >= 6, readme_mods
    bad = [
        (m, u.target, u.line)
        for m in readme_mods
        for u in graph[m]
        if _in(u.target, "mjolnir_cli.charts") or _in(u.target, "mjolnir_cli.cli")
    ]
    assert bad == []


def test_render_readme_module_does_not_import_the_cli(graph):
    assert "mjolnir_cli.readme" in graph
    assert [u for u in graph["mjolnir_cli.readme"] if _in(u.target, "mjolnir_cli.cli")] == []


def test_charts_never_imports_readme(graph):
    charts_mods = [m for m in graph if _in(m, "mjolnir_cli.charts")]
    assert charts_mods
    bad = [(m, u.target, u.line) for m in charts_mods for u in graph[m] if _in(u.target, README)]
    assert bad == []


def _governed(module: str) -> bool:
    return any(_in(module, g) for g in GOVERNED)


def test_governed_modules_import_no_underscore_name_from_another_module(graph):
    assert all(any(_in(m, g) for m in graph) for g in GOVERNED), "a governed module is missing"
    bad = [
        (m, u.target, u.name, u.line)
        for m in graph
        if _governed(m)
        for u in graph[m]
        if u.target != m and _private(u.name)
    ]
    assert bad == []


def test_no_module_imports_an_underscore_name_from_a_governed_module(graph):
    bad = [
        (m, u.target, u.name, u.line)
        for m in graph
        for u in graph[m]
        if u.target != m and _governed(u.target) and _private(u.name)
    ]
    assert bad == []


def test_charts_load_verify_is_the_verify_layer_loader():
    from mjolnir_cli import charts
    from mjolnir_cli.verify import latest

    assert charts.load_verify is latest.load_verify
