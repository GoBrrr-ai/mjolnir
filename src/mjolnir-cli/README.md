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

## Requirements

A Jetson AGX Thor (JetPack 7.x) with Docker and the NVIDIA Container Toolkit
for anything that touches the GPU; Python ≥ 3.10. Charts need the `plot` extra
(`uv tool install 'mjolnir-cli[plot]'`).

## Versions

`mjolnir-cli` and `mjolnir-kernel` version independently; the Docker image is
tagged after the upstream vLLM it is built on (`v0.31.0`). Release notes:
[GitHub Releases](https://github.com/thor-arena/mjolnir/releases).
