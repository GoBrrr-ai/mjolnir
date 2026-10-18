"""Host page-cache management + the one-time, targeted NOPASSWD sudo grant
that lets the kernel benches clear it without a password prompt.

The kernel compile / verify / bench path clears the host page cache
(``sysctl -w vm.drop_caches=3``) right before the measurement so a warm file
cache can't skew the numbers. That call needs root, so
``mjolnir grant-permissions`` installs a *targeted* NOPASSWD grant —
drop-caches only, **never full sudo** — as a single sudoers drop-in
(``/etc/sudoers.d/mjolnir-drop-caches``). After that one install (one sudo
password prompt) the cache clear runs passwordless on every run.

``prepare_bench()`` is the "always-before" hook the compile/test/bench entry
points call: it clears when the grant is present and silently skips (with a
hint) when it isn't — it never prompts and never fails a run.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import re
import subprocess
import sys
import tempfile

# The command the grant covers. sudoers requires the absolute path; the
# property is the kernel page/dentry/inode cache clear.
SYSCTL = "/usr/sbin/sysctl"
DROP_CACHES = "vm.drop_caches=3"
DROP_CACHES_CMD = f"{SYSCTL} -w {DROP_CACHES}"

# The sudoers drop-in the grant installs (one file, one rule).
SUDOERS_DROPIN = "/etc/sudoers.d/mjolnir-drop-caches"

# The rule is recognized in a `sudo -l` listing by the sysctl command + the
# property name. `=` is escaped to `\=` in the listing, but `vm.drop_caches`
# (a literal dot) survives verbatim — and the literal dot keeps it distinct
# from an unrelated `tee /proc/sys/vm/drop_caches` grant (a slash, not a dot).
_GRANTED_RE = re.compile(r"sysctl.*vm\.drop_caches")


def _user() -> str:
    """The invoking user (honors SUDO_USER when already running under sudo)."""
    return os.environ.get("SUDO_USER") or getpass.getuser()


def _rule() -> str:
    """The single sudoers rule: this user runs the cache clear passwordless —
    nothing else (explicitly *not* full sudo)."""
    return f"{_user()} ALL=(ALL) NOPASSWD: {DROP_CACHES_CMD}\n"


def drop_caches_granted() -> bool:
    """True if the current user can run the cache clear passwordless.

    Checked via the non-interactive ``sudo -n -l`` listing (read-only, no
    password prompt). A missing/locked sudo → False (callers skip the clear
    rather than fail)."""
    try:
        p = subprocess.run(["sudo", "-n", "-l"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return p.returncode == 0 and _GRANTED_RE.search(p.stdout) is not None


def drop_caches() -> int:
    """Clear the page cache passwordless (``sudo -n sysctl -w
    vm.drop_caches=3``). Returns the sysctl exit code (0 = cleared).

    The command is passed as separate argv elements (NOT one joined string)
    so ``sudo`` matches it against the ``/usr/sbin/sysctl -w vm.drop_caches=3``
    NOPASSWD rule — a single token would not match and would fall through to
    the password-requiring rule. ``-n`` makes a missing grant fail fast
    instead of prompting — callers that may run without the grant check
    :func:`drop_caches_granted` first or use :func:`prepare_bench`."""
    return subprocess.call(["sudo", "-n", SYSCTL, "-w", DROP_CACHES])


def grant_drop_caches(assume_yes: bool = False) -> int:
    """Install the targeted drop-caches NOPASSWD grant (idempotent; exit code).

    Already granted → no-op (0). Otherwise the single rule is written to a
    temp file, validated with ``visudo -cf`` (a bad rule can never reach
    sudoers), then installed as a 0440 root-owned drop-in via one ``sudo
    install`` — the single password prompt. Adds only the cache-clear
    command, never full sudo."""
    if drop_caches_granted():
        print("[mjolnir] drop-caches grant already present — nothing to do", file=sys.stderr)
        return 0
    if not assume_yes:
        try:
            ans = (
                input(
                    f"Install a NOPASSWD sudo grant for '{DROP_CACHES_CMD}' (cache clear only — not full sudo)? [y/N] "
                )
                .strip()
                .lower()
            )
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("[mjolnir] aborted — no change", file=sys.stderr)
            return 1

    rule = _rule()
    fd, tmp = tempfile.mkstemp(prefix="mjolnir-sudoers-")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(rule)
        chk = subprocess.run(["visudo", "-cf", tmp], capture_output=True, text=True)
        if chk.returncode != 0:
            print(f"[mjolnir] visudo rejected the rule:\n{chk.stdout}{chk.stderr}", file=sys.stderr)
            return 1
        # One privileged step: install as a 0440 root-owned drop-in.
        rc = subprocess.call(["sudo", "install", "-m", "0440", "-o", "root", "-g", "root", tmp, SUDOERS_DROPIN])
        if rc != 0:
            print(f"[mjolnir] failed to install the drop-in (exit {rc})", file=sys.stderr)
            return rc
        print(f"[mjolnir] installed {SUDOERS_DROPIN}", file=sys.stderr)
        return 0
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def prepare_bench() -> None:
    """The "always-before" clean-cache hook. Called right before a kernel
    compile/test/bench (``mjolnir verify``, ``image gates``) and around the server
    (before ``serve up``, after ``serve down``) so a warm file cache can't skew
    a measurement and freed cache is released after serving. Never raises and
    never prompts — if the grant is missing it prints a hint and proceeds
    (the work still happens, just on a warm cache).
    """
    if not drop_caches_granted():
        print(
            "[mjolnir] page-cache clear skipped (grant missing) — run 'mjolnir grant-permissions' to enable it",
            file=sys.stderr,
        )
        return
    rc = drop_caches()
    if rc == 0:
        print("[mjolnir] page cache cleared (vm.drop_caches=3)", file=sys.stderr)
    else:
        print(f"[mjolnir] page-cache clear failed (exit {rc}) — proceeding", file=sys.stderr)
