"""The ``mjolnir`` CLI — one command surface for the whole repo.

mjolnir hw setup|status             pre-configure the Thor host (HW
                                    provisioning, idempotent)
mjolnir serve up|down|status|logs    drive the vLLM container
mjolnir model [use|list]            active model/config (bare = arrow-key
                                    picker over configs/)
mjolnir image [use|list|build|gates] active vLLM image (bare = arrow-key
                                    picker over local docker images)
mjolnir verify <stage>              kernel correctness + ncu benchmarks (Mjolnir vs
                                    FlashInfer vs FA2) -> benchmarks/raw/verify
mjolnir autoresearch ...            the autonomous kernel-research loop (OpenCode
                                    agents, every experiment on the ledger)
mjolnir gate                        the clean-window gate
mjolnir bench perf|ab               end-to-end perf / A/B (vLLM server)
mjolnir plot|history                charts (+ README.md with --readme) + the raw-data log
mjolnir litellm up|down|status|logs  the LiteLLM proxy stack (independent
                                    of the vLLM server)
mjolnir perses up|down|status|logs   the Perses + Prometheus observability
                                    stack (independent of the vLLM server)
"""

from __future__ import annotations

import sys

import typer
from typer.core import TyperGroup as _TyperGroup

from mjolnir_cli.autoresearch import app as _autoresearch_app
from mjolnir_cli.commands import (
    _common,
    bench,
    calibrate,
    hw,
    image,
    litellm,
    model,
    perses,
    platform,
    results,
    serve,
    version,
    window,
)
from mjolnir_cli.platforms import UnknownPlatform as _UnknownPlatform
from mjolnir_cli.verify.cli import app as _verify_app

__all__ = ["app"]

_NEXT_STEPS = """
Next steps:
  mjolnir serve up         serve the model (baked-in defaults: Qwen3.8-27B NVFP4,
                           --attention-backend MJOLNIR, port 6001)
  mjolnir serve status     container + health + request queue
  mjolnir verify sweep     Mjolnir vs FlashInfer vs FA2 at every L (ncu) -> raw data
  mjolnir verify all       correctness + every benchmark stage -> charts + README
  mjolnir bench ab         end-to-end A/B: Mjolnir vs FlashInfer vs FA2 configs
  mjolnir model / image    pick another model config / serving image
"""


def _root_group() -> type[_TyperGroup]:
    """The root group: bare ``mjolnir`` prints the help plus a next-steps
    guide instead of just the help."""

    class RootGroup(_TyperGroup):
        def invoke(self, ctx):
            if not ctx._protected_args:
                ctx.command.get_help(ctx)
                print(_NEXT_STEPS, flush=True)
                return None
            return super().invoke(ctx)

    return RootGroup


app = typer.Typer(
    context_settings=_common.HELP_CTX,
    cls=_root_group(),
    help="Mjolnir — a decode-attention kernel for Jetson AGX Thor (sm_110) and the "
    "vLLM image that serves it: serve / verify / bench / plot.",
)

# The sub-apps, in their help order. Each lives in commands/<name>.py; verify
# and autoresearch keep their own packages (they are not command topics).
app.add_typer(serve.app, name="serve")
app.add_typer(bench.app, name="bench")
app.add_typer(model.app, name="model")
app.add_typer(image.app, name="image")
app.add_typer(platform.app, name="platform")
app.add_typer(_verify_app, name="verify")
app.add_typer(_autoresearch_app, name="autoresearch")
app.add_typer(litellm.app, name="litellm")
app.add_typer(perses.app, name="perses")
app.add_typer(hw.app, name="hw")

# The root commands, in their help order — each owned by its commands module
# (plot is a factory: it holds the root app for the docs/cli.md render).
app.command()(window.gate)
app.command()(calibrate.calibrate)
app.command("grant-permissions")(window.grant_permissions)
app.command("drop-caches")(window.drop_caches)
app.command()(results.make_plot(app))
app.command()(results.history)
app.command()(version.version)


def main() -> None:
    """The console-script entry point: a host that detection cannot name must
    fail every resolving command with the probe instructions (exit 2), never
    a raw traceback (review cycle 2, IMPROVEMENT 3). ``platform show`` and
    ``hw`` keep their own handlers and exits; a bad flag/env value keeps its
    ``ValueError``."""
    try:
        app()
    except _UnknownPlatform as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    sys.exit(main())
