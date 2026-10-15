"""The Mjolnir boot banner.

Owned by the ``mjolnir`` package (not a vLLM patch): when the Mjolnir
backend serves a model, ``emit()`` logs the hammer exactly once, with the
info block (``Mjolnir v<version>``, author, repo, and the serving config
name when the CLI set ``MJOLNIR_CONFIG``) placed to the art's right,
vertically centred. It goes through vLLM's own logger, so it follows the
server's formatter the same way vLLM's own boot logo does (ANSI colour
only under a color formatter), and it can never break a server's
start-up. ``python -m mjolnir.banner`` previews it without vLLM.
"""

from __future__ import annotations

import os
import sys

__all__ = ["ART", "HEAD_START", "banner", "emit"]

# The hammer. Committed art — do not re-flow.
ART = [
    "         ▓██",
    "          █▒██",
    "          █  ░█▒",
    "          █ ▓█",
    "          ██ █",
    "          █  █░",
    "         ██  ▒█",
    "     █████▓  ██████",
    "     █    ▓▒      ██",
    "     ▒████    ████▓",
    "          ░███",
]
# Rows at/below this index make up the head (brand orange); above it, the handle.
HEAD_START = 7

_HEAD = "\033[1;93m"  # bold brand orange (the hammer head)
_WHITE = "\033[1m"  # bold default foreground (handle, author)
_RESET = "\033[0m"
_GAP = "   "
_AUTHOR = "by Mieszko Syty (ms1design)"
_URL = "https://github.com/thor-arena/mjolnir"


def banner(version: str, color: bool = False, model: str | None = None) -> str:
    """Render the banner: the art, with the info block (version, author,
    repo, and ``Model: <config>`` when given) to its right, vertically
    centred against the art. The message is one block of lines (no leading
    or trailing empty line — ``emit()`` renders those as their own empty
    log records, the way the vLLM boot logo's template does). ``color``
    switches on the ANSI codes (head in the brand orange, the rest bold
    white)."""
    info = [f"Mjolnir v{version}", _AUTHOR, _URL]
    if model:
        info.append(f"Model: {model}")
    n, k = len(ART), len(info)
    start = (n - k) // 2
    width = max(len(line) for line in ART)
    rows = []
    for i, line in enumerate(ART):
        idx = i - start
        if not color:
            if 0 <= idx < k:
                rows.append(line.ljust(width) + _GAP + info[idx])
            else:
                rows.append(line)
            continue
        art = line.ljust(width) + _GAP if 0 <= idx < k else line
        segs = [(_HEAD if i >= HEAD_START else _WHITE, art)]
        if 0 <= idx < k:
            if idx == 0:
                segs += [(_HEAD, "Mjolnir"), (_WHITE, f" v{version}")]
            else:
                segs += [(_WHITE, info[idx])]
        rows.append("".join(f"{code}{text}{_RESET}" for code, text in segs))
    return "\n".join(rows)


_shown = False


def emit() -> None:
    """Log the banner once, through vLLM's logger when importable (its
    formatter decides the ANSI colour, exactly like vLLM's own boot logo)
    and a stdlib logger otherwise. The art is one multi-line record; the
    empty lines on top and bottom are rendered as empty-string records
    (vLLM's formatter turns them into bare prefixed lines, like the boot
    logo's tail). The serving config name rides in ``MJOLNIR_CONFIG`` (set
    by the ``mjolnir`` CLI's serve path). Never raises: a banner must not
    be able to break a server's start."""
    global _shown
    if _shown:
        return
    _shown = True
    try:
        from mjolnir import __version__

        model = os.environ.get("MJOLNIR_CONFIG", "").strip() or None
        try:
            from vllm.logger import current_formatter_type, init_logger

            lgr = init_logger("vllm.mjolnir")
            color = current_formatter_type(lgr) == "color"
        except ImportError:
            import logging

            lgr = logging.getLogger("vllm.mjolnir")
            color = sys.stderr.isatty()
        lgr.info("")
        lgr.info("%s", banner(__version__, color=color, model=model))
        lgr.info("")
    except Exception:  # noqa: BLE001 - the banner is decoration, not a dependency
        pass


if __name__ == "__main__":  # `python -m mjolnir.banner [--plain]`: preview without vLLM
    import mjolnir

    plain = "--plain" in sys.argv[1:]
    msg = banner(
        mjolnir.__version__,
        color=sys.stdout.isatty() and not plain,
        model=os.environ.get("MJOLNIR_CONFIG", "").strip() or None,
    )
    print("")
    print(msg)
    print("")
