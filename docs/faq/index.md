# FAQ — User Stories

Task-oriented guides for the `mjolnir` CLI: each file is one story
("I want to do X") with the exact commands and the methodology behind them.
Read [`../README.md`](../README.md) first for what the repo is; read
[`../methodology/benchmarking.md`](../methodology/benchmarking.md) before you
trust or publish any number.

## Host setup

| Story | File |
|---|---|
| Pre-configure a fresh Jetson Thor for mjolnir (`mjolnir hw setup`) | [setup-thor.md](setup-thor.md) |

## Serving & defaults

| Story | File |
|---|---|
| Start / stop / inspect the vLLM server | [serve-server.md](serve-server.md) |
| Change the default vLLM image (and build a new one) | [default-image.md](default-image.md) |
| Change the default model / quant config | [default-model-quant.md](default-model-quant.md) |
| Add a new model config (repo or user-local, persisted at `~/`) | [new-model-config.md](new-model-config.md) |
| What wins: flag, state, env var, or baked-in default? | [settings-precedence.md](settings-precedence.md) |
| Run the LiteLLM proxy stack (independent of the vLLM server) | [litellm-proxy.md](litellm-proxy.md) |
| Serve on a 6-bit (e2m3) KV cache: calibrate, serve, check | [fp6-kv.md](fp6-kv.md) |

## Benchmarking

| Story | File |
|---|---|
| The clean-window gate — why every bench waits, and never kills the server | [clean-window-gate.md](clean-window-gate.md) |
| Official vLLM image vs Mjolnir: decode, prefill, TTFT, ITL (`mjolnir bench e2e`) | [e2e-vs-vllm.md](e2e-vs-vllm.md) |
| Try other speculative drafters — DSpark, DFlash2 (`bench e2e --optional`) | [spec-drafters.md](spec-drafters.md) |
| A/B a custom image against the default image | [ab-image.md](ab-image.md) |
| A/B different models against each other | [ab-models.md](ab-models.md) |
| Verify and benchmark the kernel (`mjolnir verify …`) | [verify-kernel.md](verify-kernel.md) |
| Where results land, and how charts / README / the site are rendered | [results-history-charts.md](results-history-charts.md) |

## Image & kernel work

| Story | File |
|---|---|
| Upgrade the Dockerfile base image and re-adapt the patch stack | [base-image-bump.md](base-image-bump.md) |
| Improve the Mjolnir kernel by hand: measure, A/B, publish | [verify-kernel.md](verify-kernel.md#improving-the-kernel) |
| Run the autonomous research loop overnight | [autoresearch.md](autoresearch.md) |
| Cut a release: PyPI packages, Docker image, site (one manual workflow) | [releasing.md](releasing.md) |
| Serve with `--attention-backend MJOLNIR` (image or stock vLLM) | [serve-server.md](serve-server.md#the-mjolnir-backend) |
