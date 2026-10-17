"""scripts/check_file_length.py: the file-length ratchet, checked against scratch git repos."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_file_length", ROOT / "scripts" / "check_file_length.py")
cfl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfl)

LIMIT = 5


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def lines(n: int, *, newline: bool = True) -> str:
    return "\n".join(f"x{i}" for i in range(n)) + ("\n" if newline and n else "")


def table(grand: dict[str, int] | None = None, exempt: list[str] | None = None) -> str:
    out = "[tool.mjolnir.file-length]\nlimit = %d\n" % LIMIT
    out += 'include = ["*.py", "*.sh", "*.md"]\n'
    out += "exempt = [%s]\n" % ", ".join(f'"{e}"' for e in (exempt or []))
    out += "[tool.mjolnir.file-length.grandfathered]\n"
    for path, ceiling in (grand or {}).items():
        out += f'"{path}" = {ceiling}\n'
    return out


def make_repo(tmp_path, files, *, toml=None, untracked=()):
    r = tmp_path / "r"
    r.mkdir()
    contents = dict(files)
    contents["pyproject.toml"] = toml if toml is not None else table()
    for rel, text in {**contents, **dict.fromkeys(untracked, lines(LIMIT + 3))}.items():
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text(text)
    _git(r, "init", "-q")
    _git(r, "add", *[p for p in contents])
    return r


NOTE = "history check skipped"


def run(repo, capsys):
    """Current-state run (no --base); the 'history check skipped' note is stripped from stderr."""
    code = cfl.main([], root=repo)
    cap = capsys.readouterr()
    err = "".join(ln + "\n" for ln in cap.err.splitlines() if NOTE not in ln)
    return code, cap.out, err


def run_base(repo, capsys, base="HEAD"):
    code = cfl.main(["--base", base], root=repo)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def commit_base(repo, toml):
    """Commit `toml` as the pyproject.toml of a base commit, then return to the caller to edit it."""
    (repo / "pyproject.toml").write_text(toml)
    _git(repo, "add", "pyproject.toml")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")


def test_under_limit_passes_and_is_silent(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(LIMIT)})
    assert run(r, capsys) == (0, "", "")


def test_limit_plus_one_fails_with_one_line(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(LIMIT + 1)})
    code, out, err = run(r, capsys)
    assert code == 1 and out == ""
    assert len(err.splitlines()) == 1
    assert "a.py" in err and str(LIMIT + 1) in err and str(LIMIT) in err
    assert "grandfathered" in err or "split" in err


def test_grandfathered_at_ceiling_passes(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(8)}, toml=table({"a.py": 8}))
    assert run(r, capsys) == (0, "", "")


def test_grandfathered_above_ceiling_fails(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(9)}, toml=table({"a.py": 8}))
    code, out, err = run(r, capsys)
    assert code == 1 and out == ""
    assert len(err.splitlines()) == 1
    assert "a.py" in err and "9" in err and "8" in err


def test_grandfathered_stale_high_fails_with_lower_the_ceiling(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(7)}, toml=table({"a.py": 8}))
    code, out, err = run(r, capsys)
    assert code == 1 and out == ""
    assert len(err.splitlines()) == 1
    assert "a.py" in err and "7" in err and "8" in err
    assert "lower the ceiling to 7" in err


def test_grandfathered_at_or_under_limit_fails_remove_entry(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(LIMIT)}, toml=table({"a.py": 8}))
    code, out, err = run(r, capsys)
    assert code == 1 and out == ""
    assert len(err.splitlines()) == 1
    assert "a.py" in err and "remove the entry" in err and str(LIMIT) in err


def test_entry_for_missing_path_fails(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(1)}, toml=table({"gone.py": 8}))
    code, out, err = run(r, capsys)
    assert code == 1 and out == ""
    assert len(err.splitlines()) == 1
    assert "gone.py" in err and "remove the entry" in err


def test_entry_for_untracked_path_fails(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(1)}, toml=table({"loose.py": 8}), untracked=["loose.py"])
    code, out, err = run(r, capsys)
    assert code == 1
    assert "loose.py" in err and "remove the entry" in err


def test_exempt_file_over_limit_passes(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.md": lines(LIMIT + 4)}, toml=table(exempt=["a.md"]))
    assert run(r, capsys) == (0, "", "")


def test_untracked_file_over_limit_passes(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(1)}, untracked=["scratch.py"])
    assert run(r, capsys) == (0, "", "")


def test_uncovered_extension_passes(tmp_path, capsys):
    r = make_repo(tmp_path, {"data.txt": lines(LIMIT + 4)})
    assert run(r, capsys) == (0, "", "")


def test_no_trailing_newline_counts_last_line():
    assert cfl.count_lines(b"a\nb\nc") == 3
    assert cfl.count_lines(b"a\nb\nc\n") == 3
    assert cfl.count_lines(b"") == 0


def test_no_trailing_newline_at_limit_plus_one_fails(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(LIMIT + 1, newline=False)})
    code, _, err = run(r, capsys)
    assert code == 1 and "a.py" in err


def test_blank_and_comment_only_lines_count(tmp_path, capsys):
    body = "# c\n\n# c\n\n# c\n\n"  # 6 lines, none code
    r = make_repo(tmp_path, {"a.py": body})
    code, _, err = run(r, capsys)
    assert code == 1 and "6" in err


def test_include_matches_basename_at_any_depth(tmp_path, capsys):
    r = make_repo(tmp_path, {"deep/er/a.py": lines(LIMIT + 1)})
    code, _, err = run(r, capsys)
    assert code == 1 and "deep/er/a.py" in err


@pytest.mark.parametrize(
    "pattern,path,expected",
    [
        ("README.md", "README.md", True),
        ("README.md", "docs/x/README.md", False),
        ("docs/kernel/*.md", "docs/kernel/a.md", True),
        ("docs/kernel/*.md", "docs/kernel/sub/a.md", False),
        ("assets/benchmarks/**", "assets/benchmarks/a.png", True),
        ("assets/benchmarks/**", "assets/benchmarks/x/y/a.md", True),
        ("assets/benchmarks/**", "other/assets/benchmarks/a.md", False),
        ("**/*.json", "a/b.json", True),
        ("**/*.json", "x.json", True),
        ("**/*.json", "a/b/c/x.json", True),
        ("**/*.json", "x.jsonl", False),
        ("uv.lock", "uv.lock", True),
    ],
)
def test_exempt_glob_semantics(pattern, path, expected):
    assert cfl.glob_match(pattern, path) is expected


def test_exempt_glob_applied_end_to_end(tmp_path, capsys):
    files = {"README.md": lines(9), "docs/x/README.md": lines(9)}
    r = make_repo(tmp_path, files, toml=table(exempt=["README.md"]))
    code, _, err = run(r, capsys)
    assert code == 1
    assert "docs/x/README.md" in err and len(err.splitlines()) == 1


def test_multiple_failures_one_line_each(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(9), "b.py": lines(9)})
    code, out, err = run(r, capsys)
    assert code == 1 and out == "" and len(err.splitlines()) == 2


def test_missing_table_is_environment_error(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(1)}, toml="[project]\nname = 'x'\n")
    code, out, err = run(r, capsys)
    assert code == 2 and out == "" and "file-length" in err


def test_not_a_git_repo_is_environment_error(tmp_path, capsys):
    (tmp_path / "pyproject.toml").write_text(table())
    code, out, err = run(tmp_path, capsys)
    assert code == 2 and out == "" and err.strip()


def test_no_base_notes_skipped_history_check(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(1)})
    code = cfl.main([], root=r)
    cap = capsys.readouterr()
    assert code == 0 and cap.out == ""
    assert NOTE in cap.err and len(cap.err.splitlines()) == 1


def history_repo(tmp_path, base_ceiling, pr_ceiling, size, path="a.py", pr_path=None):
    r = make_repo(tmp_path, {path: lines(size)}, toml=table({path: base_ceiling}))
    commit_base(r, table({path: base_ceiling}))
    (r / "pyproject.toml").write_text(table({(pr_path or path): pr_ceiling}))
    return r


def test_base_ceiling_raised_fails(tmp_path, capsys):
    r = history_repo(tmp_path, 1287, 1288, 1288)
    code, out, err = run_base(r, capsys)
    assert code == 1 and out == ""
    assert "ceilings only go down" in err and "a.py" in err and "1287" in err and "1288" in err


def test_base_ceiling_lowered_with_file_shrunk_passes(tmp_path, capsys):
    r = history_repo(tmp_path, 1287, 1250, 1250)
    assert run_base(r, capsys) == (0, "", "")


def test_base_ceiling_unchanged_passes(tmp_path, capsys):
    r = history_repo(tmp_path, 8, 8, 8)
    assert run_base(r, capsys) == (0, "", "")


def test_new_entry_not_in_base_fails(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(8), "b.py": lines(8)}, toml=table({"a.py": 8}))
    commit_base(r, table({"a.py": 8}))
    (r / "pyproject.toml").write_text(table({"a.py": 8, "b.py": 8}))
    code, _, err = run_base(r, capsys)
    assert code == 1 and "no new entries" in err and "b.py" in err and "a.py" not in err


def test_rename_of_grandfathered_path_fails(tmp_path, capsys):
    r = history_repo(tmp_path, 8, 8, 8, pr_path="b.py")
    _git(r, "mv", "a.py", "b.py")
    code, _, err = run_base(r, capsys)
    assert code == 1 and "no new entries" in err and "b.py" in err


def test_removing_entry_passes(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(3)}, toml=table({"a.py": 8}))
    commit_base(r, table({"a.py": 8}))
    (r / "pyproject.toml").write_text(table())
    assert run_base(r, capsys) == (0, "", "")


def test_base_without_table_bootstraps_with_note(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(8)}, toml=table({"a.py": 8}))
    commit_base(r, "[project]\nname = 'x'\n")
    (r / "pyproject.toml").write_text(table({"a.py": 8}))
    code, out, err = run_base(r, capsys)
    assert code == 0 and out == ""
    assert "file-length" in err and "skip" in err and len(err.splitlines()) == 1


def test_base_without_pyproject_bootstraps(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(1)})
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x", "--", "a.py")
    code, _, err = run_base(r, capsys)
    assert code == 0 and "skip" in err


def test_bootstrap_still_reports_current_state_problems(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(9)})
    commit_base(r, "[project]\nname = 'x'\n")
    (r / "pyproject.toml").write_text(table())
    code, _, err = run_base(r, capsys)
    assert code == 1 and "a.py" in err


def test_unresolvable_base_is_environment_error(tmp_path, capsys):
    r = make_repo(tmp_path, {"a.py": lines(1)})
    commit_base(r, table())
    code, out, err = run_base(r, capsys, base="no-such-ref")
    assert code == 2 and out == "" and "no-such-ref" in err


def test_script_is_single_short_stdlib_file():
    src = (ROOT / "scripts" / "check_file_length.py").read_text()
    assert len(src.splitlines()) < 250
    for hardcoded in ("1000", "uv.lock", "*.py", "docs/kernel"):
        assert hardcoded not in src
