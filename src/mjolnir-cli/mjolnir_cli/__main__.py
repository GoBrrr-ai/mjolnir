"""Run the ``mjolnir`` CLI via ``python -m mjolnir_cli`` — the same entry
point as the console script (``mjolnir_cli.cli:main``), UnknownPlatform
handler included."""

import sys

from mjolnir_cli.cli import main

if __name__ == "__main__":
    sys.exit(main())
