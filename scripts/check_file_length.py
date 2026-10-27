"""File-length ratchet: fail when a tracked file outgrows the table in pyproject.toml.

All policy lives in ``[tool.mjolnir.file-length]``; ``exclude`` (kernel family, generated docs) is
outside the ratchet entirely — no ceiling, no floor — and, unlike ``exempt``, may change freely.

``--base REF`` also enforces the history rules against REF: the table may only tighten
(no higher limit, no wider exempt, no narrower include, no raised ceiling, no new
entry); without it only current-state checks run and a stderr note says so. ``--fix``
retires grandfathered entries parked at the limit (the pre-commit hook passes it).

Exit codes: 0 ok, 1 violation (one stderr line each), 2 environment error. A clean run prints nothing.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

TABLE = ("tool", "mjolnir", "file-length")

# git exports GIT_DIR (hooks) / GIT_INDEX_FILE (partial commits): scrub, else -C treats <dir> as the work tree.
_GIT_LOC_VARS = frozenset(("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES", "GIT_INDEX_FILE"))


def _git_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _GIT_LOC_VARS}


class EnvError(Exception):
    """Something outside the policy is wrong (no git, no table, unreadable file)."""


def count_lines(data: bytes) -> int:
    """Newlines, plus one for a non-empty file whose last line has no newline."""
    return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)


_GLOB = {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}  # "**/" also matches zero dirs


def _glob_regex(pattern: str) -> re.Pattern[str]:
    parts = re.split(r"(\*\*/|\*\*|\*|\?)", pattern)
    return re.compile("".join(_GLOB.get(p, re.escape(p)) for p in parts) + r"\Z")


def glob_match(pattern: str, path: str) -> bool:
    """Match a repo-relative POSIX path, anchored at the root; ``**`` spans directories."""
    return _glob_regex(pattern).match(path) is not None


def parse_table(text: str, where: str) -> dict | None:
    """The validated file-length table from pyproject.toml text, or None when it has none."""
    try:
        node = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise EnvError(f"pyproject.toml at {where} is unreadable: {exc}") from exc
    for key in TABLE:
        node = node.get(key) if isinstance(node, dict) else None
    if not isinstance(node, dict):
        return None
    limit, grand = node.get("limit"), node.get("grandfathered", {})
    include, exempt = node.get("include"), node.get("exempt")
    exclude = node.get("exclude", [])
    bad = [k for k, v in (("include", include), ("exempt", exempt), ("exclude", exclude)) if not _str_list(v)]
    if not isinstance(limit, int) or isinstance(limit, bool):
        bad.append("limit")
    if not isinstance(grand, dict) or not all(isinstance(v, int) for v in grand.values()):
        bad.append("grandfathered")
    if bad:
        raise EnvError(f"[tool.mjolnir.file-length] at {where} is malformed: {', '.join(bad)}")
    return {"limit": limit, "include": include, "exempt": exempt, "exclude": exclude, "grandfathered": dict(grand)}


def _str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def load_table(root: Path) -> dict:
    try:
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError as exc:
        raise EnvError(f"cannot read {root / 'pyproject.toml'}: {exc}") from exc
    table = parse_table(text, "the work tree")
    if table is None:
        raise EnvError("pyproject.toml has no [tool.mjolnir.file-length] table")
    return table


def tracked_files(root: Path) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True, env=_git_env()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        raise EnvError(f"git ls-files failed in {root}: {exc} {detail.decode(errors='replace').strip()}") from exc
    # surrogateescape: a non-UTF-8 name round-trips to the same bytes on open()
    return [p for p in proc.stdout.decode("utf-8", "surrogateescape").split("\0") if p]


def is_covered(path: str, table: dict) -> bool:
    name = path.rsplit("/", 1)[-1]
    if not any(fnmatch.fnmatchcase(name, pat) for pat in table["include"]):
        return False
    return not any(glob_match(pat, path) for pat in table["exempt"] + table["exclude"])


def measure(root: Path, path: str) -> int | None:
    try:
        return count_lines((root / path).read_bytes())
    except OSError:
        return None  # tracked but absent from the work tree


def check(root: Path) -> list[str]:
    table = load_table(root)
    limit, grand = table["limit"], table["grandfathered"]
    tracked = set(tracked_files(root))
    covered = sorted(p for p in tracked if is_covered(p, table))
    problems: list[str] = []
    bad = problems.append
    for path in sorted(grand):
        if path not in covered:
            bad(f"{path}: grandfathered entry names no covered tracked file; remove the entry")
    for path in covered:
        n = measure(root, path)
        if n is None:  # tracked but deleted from the work tree without `git rm`
            if path in grand:
                bad(f"{path}: grandfathered file is missing from the work tree; remove the entry")
            continue
        ceiling = grand.get(path)
        if ceiling is not None and n <= limit:  # parked at the limit is fine; --fix retires it
            if ceiling != limit:
                bad(f"{path}: {n} lines is within the limit {limit}; set its ceiling to the limit ({limit})")
        elif ceiling is not None and n > ceiling:
            bad(f"{path}: {n} lines exceeds its ceiling {ceiling} (limit {limit}); split the file, keep the ceiling")
        elif ceiling is not None and n < ceiling:
            bad(f"{path}: {n} lines is below its ceiling {ceiling} (limit {limit}); lower the ceiling to {n}")
        elif ceiling is None and n > limit:
            bad(f"{path}: {n} lines exceeds the limit {limit}; split the file (generated or data: exempt it)")
    return problems


def retire(root: Path) -> list[str]:
    """--fix: drop grandfathered entries parked at the limit whose file is within it. Edits only
    those lines of the grandfathered section, and writes only if the result re-parses to exactly
    the old table minus those entries."""
    table, head = load_table(root), "[tool.mjolnir.file-length.grandfathered]"
    limit, grand = table["limit"], table["grandfathered"]
    done = [p for p, c in sorted(grand.items()) if c == limit and (measure(root, p) or limit + 1) <= limit]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    if not done or text.count(head) != 1:
        return []
    start = text.index(head) + len(head)
    end = text.find("\n[", start) + 1 or len(text)
    body = text[start:end]
    for p in done:
        body = re.sub(rf'^"{re.escape(p)}"[ \t]*=[ \t]*{limit}[ \t]*(#.*)?\n', "", body, count=1, flags=re.MULTILINE)
    new = text[:start] + body + text[end:]
    expected = {**table, "grandfathered": {p: c for p, c in grand.items() if p not in done}}
    if parse_table(new, "the fixed work tree") != expected:
        return []  # an unexpected layout: change nothing, the check below still reports
    (root / "pyproject.toml").write_text(new, encoding="utf-8")
    return [f"{p}: retired the grandfathered entry (within the limit {limit}); stage pyproject.toml" for p in done]


def _git_out(root: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, env=_git_env())
    except OSError as exc:
        raise EnvError(f"cannot run git: {exc}") from exc


def base_table(root: Path, ref: str) -> dict | None:
    """The file-length table at `ref`, or None when that ref has none (bootstrap)."""
    if _git_out(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode != 0:
        raise EnvError(f"--base {ref!r} does not resolve to a commit")
    proc = _git_out(root, "show", f"{ref}:pyproject.toml")
    if proc.returncode != 0:
        return None  # no pyproject.toml at the base
    return parse_table(proc.stdout.decode("utf-8", "replace"), repr(ref))


def check_history(root: Path, ref: str) -> tuple[list[str], str | None]:
    """The table may only tighten against `ref`: no higher limit, no wider exempt, no narrower
    include, no raised ceiling, no new entry; ``exclude`` is not ratcheted (deliberate carve-out)."""
    base = base_table(root, ref)
    if base is None:
        return (
            [],
            f"check_file_length: {ref} has no [tool.mjolnir.file-length] table; history check skipped (bootstrap)",
        )
    now, tighten = load_table(root), "the ratchet only tightens"
    problems: list[str] = []
    if now["limit"] > base["limit"]:
        problems.append(f"pyproject.toml: limit raised from {base['limit']} in {ref} to {now['limit']}; {tighten}")
    for pat in sorted(set(now["exempt"]) - set(base["exempt"])):
        problems.append(f"pyproject.toml: exempt pattern {pat!r} added (not in {ref}); {tighten}")  # noqa: PERF401
    for pat in sorted(set(base["include"]) - set(now["include"])):
        problems.append(f"pyproject.toml: include pattern {pat!r} removed (it is in {ref}); {tighten}")  # noqa: PERF401
    was = base["grandfathered"]
    for path, ceiling in sorted(now["grandfathered"].items()):
        if path not in was:
            problems.append(f"{path}: new grandfathered entry (ceiling {ceiling}) is not in {ref}; no new entries")
        elif ceiling > was[path]:
            problems.append(f"{path}: ceiling raised from {was[path]} in {ref} to {ceiling}; ceilings only go down")
    return problems, None


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description="File-length ratchet.")
    parser.add_argument("--base", metavar="REF", help="git ref whose grandfathered ceilings the table may not exceed")
    parser.add_argument("--fix", action="store_true", help="retire grandfathered entries parked at the limit")
    args = parser.parse_args(argv)
    try:
        root = root if root is not None else _repo_root()
        for line in retire(root) if args.fix else []:
            print(line, file=sys.stderr)
        problems = check(root)
        if args.base is None:
            note = "check_file_length: no --base given; history check skipped"
        else:
            history, note = check_history(root, args.base)
            problems += history
    except EnvError as exc:
        print(f"check_file_length: {exc}", file=sys.stderr)
        return 2
    for line in problems:  # a non-UTF-8 path prints with U+FFFD rather than raising
        print(line.encode("utf-8", "surrogateescape").decode("utf-8", "replace"), file=sys.stderr)
    if note:
        print(note, file=sys.stderr)
    return 1 if problems else 0


def _repo_root() -> Path:
    out = _git_out(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel")
    if out.returncode != 0:
        raise EnvError(f"not inside a git checkout: {out.stderr.decode(errors='replace').strip()}")
    return Path(out.stdout.decode().strip())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
