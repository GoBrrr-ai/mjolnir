---
name: "Bug report"
about: Something is failing: a CLI command, image build, patch, kernel verify, or bench
title: ""
labels: ["bug"]
assignees: []
---

## What happened

A clear, concise description of the failure. What command did you run and what error did you get (paste the traceback / stderr verbatim).

## What you expected to happen

## How to reproduce

Exact commands, in order, on a Thor host:

```
$ # e.g.
$ mjolnir serve up --image narandill/mjolnir-vllm:v0.30.0
$ mjolnir verify sweep
...
```

## Environment

| Field | Value |
|---|---|
| Image tag | `mjolnir image` (e.g. `narandill/mjolnir-vllm:v0.30.0`) |
| Base vLLM | (e.g. `vllm/vllm-openai:v0.30.0-ubuntu2404`) |
| Model + quant config | (e.g. `configs/Qwen/Qwen3.8-27B/NVFP4_MJOLNIR.yaml`) |
| `mjolnir` version | `mjolnir version` (+ git commit if from a checkout) |
| `mjolnir-kernel` version | `python -c 'import mjolnir; print(mjolnir.__version__)'` in the container |
| Thor state | output of `mjolnir hw status` (or "not run yet" if pre-setup) |

## Server / GPU state at the time

Was the live vLLM server up, and did you run inside a clean window?

```
$ mjolnir gate --once
...
```

If this is a **number** that looks wrong (not a crash): state whether the run was gated, and include the raw JSON (`benchmarks/raw/verify/<stage>/<run_id>.json` for kernel numbers) — wall-clock claims only count as same-window ratios, absolute kernel claims need NCU achieved bandwidth.

## Raw artifacts

Attach the relevant raws / logs:

- [ ] raw JSON (`benchmarks/raw/verify/…` or `benchmarks/raw/perf-…`)
- [ ] verify logs (`benchmarks/logs/verify-<run_id>/`)
- [ ] server log / `docker logs` excerpt
- [ ] `mjolnir image build` output (for patch/build failures)
- [ ] NCU profile (for kernel perf claims)

## Anything else

Workarounds found, relevant prior issues/PRs, upstream tracker links.
