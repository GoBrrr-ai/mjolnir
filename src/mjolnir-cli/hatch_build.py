"""Bundle the repo resources the CLI needs into the mjolnir-cli wheel/sdist.

Builds from a checkout copy them from the repo root to ``mjolnir_cli/bundled/``
(model configs, the LiteLLM / Perses compose stacks, the Thor HW setup
script); builds from an sdist already carry them there, so the hook is a
no-op. At runtime ``mjolnir_cli.config.resource_path`` prefers a checkout's
copy and falls back to the bundled one.
"""

from __future__ import annotations

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# repo-relative path -> path inside the package
BUNDLED = {
    "configs": "mjolnir_cli/bundled/configs",
    "docker/litellm": "mjolnir_cli/bundled/docker/litellm",
    "docker/perses": "mjolnir_cli/bundled/docker/perses",
    "scripts/hw": "mjolnir_cli/bundled/scripts/hw",
}


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        repo = (Path(self.root) / ".." / "..").resolve()
        for rel, dest in BUNDLED.items():
            src = repo / rel
            if src.is_dir():
                build_data["force_include"][str(src)] = dest
