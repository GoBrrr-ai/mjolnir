# mjolnir-cli

**Serve, benchmark and document LLMs on the NVIDIA Jetson AGX Thor** with
[Mjolnir](https://github.com/thor-arena/mjolnir) — a decode-attention kernel
built for Thor (`sm_110`), a 6-bit KV-cache format for vLLM and a patched vLLM
image that runs them. This package is the `mjolnir` command that drives all of
it; the kernel itself ships separately as
[`mjolnir-kernel`](https://pypi.org/project/mjolnir-kernel/).

```bash
uv tool install mjolnir-cli      # or: pipx install mjolnir-cli
mjolnir hw setup                 # one-time Thor host setup (clocks, power mode, fans, swap)
mjolnir serve up                 # serve the default model on the Mjolnir image, port 6001
```

## What it does

- **Models as config dirs** — every served model config is a `config.yaml` of
  `vllm serve` flags (plus an optional `.env` of switches inside its dir; the
  model's calibration files sit next to the config dirs). The default
  configs ship inside this package; add your own under
  `~/.local/share/mjolnir/configs/`. `mjolnir model` picks the active one.
- **Serve** — `mjolnir serve up|down|status|logs` runs the
  [`narandill/mjolnir-vllm`](https://hub.docker.com/r/narandill/mjolnir-vllm)
  container with the active config, clears the page cache around it, and never
  touches a server it did not start.
- **Calibrate** — `mjolnir calibrate` measures per-channel KV-cache scales (for
  the 6-bit `e2m3` cache) and, with `--draft-vocab`, the drafter's vocabulary.
- **Benchmark** — `mjolnir verify <stage>` (kernel correctness and Nsight
  Compute benchmarks), `mjolnir bench e2e|ab|perf` (end-to-end serving with
  llama-benchy), `mjolnir bench accuracy` (tool-calling, GSM8K, MMLU, IFEval via
  tool-eval-bench). Every run waits for a clean GPU window and saves raw JSON.
- **Document** — `mjolnir plot --readme` renders every chart, table and the
  README from those raw results; `--site` builds the interactive page.
- **Research** — `mjolnir autoresearch` runs an agent loop that proposes,
  tests and logs kernel experiments.

Every command, with options and defaults:
**[the CLI reference](https://github.com/thor-arena/mjolnir/blob/main/docs/cli.md)**
(generated from the commands; `mjolnir <command> --help` prints the same).

## Library use

Two modules are declared a public, stable API — reuse them instead of
re-implementing the clean-window gate or the bench-history substrate:

- **`mjolnir_cli.gate`** — `wait_for_idle`, `server_load`, `server_counters`,
  `spec_delta`, the `CleanWindow` context manager, `PreflightError` and the
  `DEFAULT_*` tuning constants.
- **`mjolnir_cli.history`** — `append_record`, `load_records` and
  `SCHEMA_VERSION`; the history-row schema is documented in the module
  docstring. Rows carry `schema_version` — absence means the legacy (pre-0.3)
  format — and `load_records` returns rows raw, never normalized.

Anything else can change without notice; the module docstrings are the API
reference: [`gate.py`](https://github.com/thor-arena/mjolnir/blob/main/src/mjolnir-cli/mjolnir_cli/gate.py),
[`history.py`](https://github.com/thor-arena/mjolnir/blob/main/src/mjolnir-cli/mjolnir_cli/history.py).

`uv tool install mjolnir-cli` is an isolated CLI install — *not* importable
from another project. Library use means adding `mjolnir-cli` as a dependency
of your project (`uv add mjolnir-cli` / `pip install mjolnir-cli`). The
gate/history import chain is stdlib-only: the CLI dependency stack (typer,
rich, pyyaml, jinja2) is paid at install time, not import time. Python ≥ 3.12
is required (pip refuses older interpreters — relevant on JetPack 6-era
images), and type annotations ship (`py.typed`).

## Requirements

A Jetson AGX Thor (JetPack 7.x) with Docker and the NVIDIA Container Toolkit
for anything that touches the GPU; Python ≥ 3.12. Charts need the `plot` extra
(`uv tool install 'mjolnir-cli[plot]'`).

## Versions

`mjolnir-cli` and `mjolnir-kernel` version independently; the Docker image is
tagged after the upstream vLLM it is built on (`v0.31.0`). Release notes:
[GitHub Releases](https://github.com/thor-arena/mjolnir/releases).
