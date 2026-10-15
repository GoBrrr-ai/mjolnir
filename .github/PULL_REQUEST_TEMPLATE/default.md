<!-- One or two sentences: what changes and why. Title: short, imperative,
     prefixed with the artifact (kernel: / configs: / verify: / image: …). -->

## Artifacts touched

- [ ] `mjolnir-kernel` (`src/mjolnir/`)
- [ ] `mjolnir-cli` (`src/mjolnir-cli/`, `configs/`, `docker/litellm/`, `docker/perses/`, `scripts/hw/`)
- [ ] image (`docker/vllm-thor/`, `LICENSE`, `NOTICE`)
- [ ] docs / research only

## Checked

- [ ] `uvx ruff@0.14.14 check src/ tests/` + `uv run pytest -q` pass locally
- [ ] if `benchmarks/` changed: re-rendered with `uv run mjolnir plot --readme` (CI fails if the committed README / docs don't reproduce)
- [ ] any GPU measurement was gated (`mjolnir gate`), the live vLLM server was never stopped
- [ ] new third-party code: `NOTICE` entry, license kept with the code

## Notes

<!-- commands run, links to raw artifacts or issues; negative results are fine. -->
