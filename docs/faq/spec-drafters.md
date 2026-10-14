# Other speculative drafters: DSpark and DFlash2

**Story:** "Mjolnir ships with Qwen3.8's built-in MTP head as the drafter. I
want to know whether a trained external drafter — DSpark or DFlash2 — decodes
faster, on stock vLLM and on the Mjolnir image."

Both drafters are native in vLLM 0.31.0, so they run on the images we already
use — no custom build. Four opt-in configs cover them:

| config | image | attention / KV | drafter |
|---|---|---|---|
| `NVFP4_DSPARK` | official `vllm/vllm-openai` | FlashInfer, fp8 KV | DSpark, K=7 |
| `NVFP4_DFLASH2` | official | FlashInfer, fp8 KV | DFlash2, K=7 |
| `NVFP4_MJOLNIR_E2M3_DSPARK` | Mjolnir | Mjolnir decode, fp6 KV | DSpark, K=7 |
| `NVFP4_MJOLNIR_E2M3_DFLASH2` | Mjolnir | Mjolnir decode, fp6 KV | DFlash2, K=7 |

Each is its base config (`NVFP4` / `NVFP4_MJOLNIR_E2M3`) with the drafter in
place of MTP. The target model is the same
(`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`), so a leg differs from its base in the
drafter only.

## Measure them

```bash
mjolnir bench e2e --optional                       # the headline pair, then all four (hours per leg)
mjolnir bench e2e --only official --optional       # just the official side's legs
mjolnir bench e2e --also mjolnir:NVFP4_MJOLNIR_E2M3_DSPARK   # one leg after the pair
```

Optional legs go into the history; the README table, charts and infographic
stay on the default legs. Compare afterwards with:

```bash
mjolnir bench compare --config NVFP4_MJOLNIR_E2M3_DSPARK                                # vs stock FlashInfer + MTP
mjolnir bench compare --official-config NVFP4_DSPARK --config NVFP4_MJOLNIR_E2M3_DSPARK # same drafter, both images
```

The legs are listed in `src/mjolnir-cli/mjolnir_cli/legs.py` (`OPTIONAL`).

## The drafters

**DSpark** (`method: dspark`): RadixArk's drafter — 5 dense Qwen3 layers plus
a Markov head over the target's hidden states (layers 4, 16, 28, 40, 52).
`Doopeworld/Qwen3.8-27B-DSpark-vLLM` has byte-identical weights to
`RadixArk/Qwen3.8-27B-DSpark`; only its `architectures` field is changed, to
`Qwen3DSparkModel`, which vLLM registers. The configs use:

- `num_speculative_tokens: 7`, the drafter's trained block size (not 8);
- `draft_sample_method: probabilistic` — greedy drafting measured about 23%
  slower in RadixArk's eval;
- adaptive verification off, which is the default. vLLM rejects it for
  Qwen3.8: its GDN layers cannot trim a verification on the device.

**DFlash2** (`method: dflash`, architecture `DFlash2DraftModel`,
[vllm#52816](https://github.com/vllm-project/vllm/pull/52816)): a
block-diffusion drafter — 5 Qwen3 layers with a 2048-token sliding window,
non-causal attention over the draft block, a grouped local convolution and a
top-16 candidate selector. The checkpoint is `incoai/Qwen3.8-27B-DFlash2`
(`z-lab/…` mirrors it). Its block size is 8, so K=7. It shares the target's
`lm_head`, so don't add the fp8 lm_head or draft-vocab knobs (`LMH8`, `VT*`)
to it. It needs vLLM's V2 model runner, the 0.31 default. If the log says the
V2 runner is not supported, the checkpoint drafts as DFlash1 without an error;
the acceptance rate shows it.

## How the drafter's layers are served

The drafter's 5 layers come after the target's 64, as layer indices 64–68.
All four configs keep them on **bf16 KV**, through
`kv-cache-dtype-skip-layers: ["sliding_window", "64", …, "68"]`:

- its block attention is non-causal, and bf16 runs on any backend;
- on the Mjolnir legs, `kv_scales.json` calibrates only the target. Calibrated
  fp6 scales exist only for the target's layers, so the drafter can't use fp6
  or run on `MJOLNIR`. It gets its own backend instead:
  `speculative-config.attention_backend: FLASHINFER`.

The target's verify steps (1 + 7 tokens per request) still run on the Mjolnir
decode kernel. Its plan carries the 7-token margin.

## Before trusting a number

None of the four configs has run on a Thor yet. On the first start:

- `mjolnir serve up --config <cfg>`, then check the log for:
  - the drafter's attention backend;
  - the KV dtype of layers 64–68;
  - on DFlash2, the model runner.
- If the Mjolnir legs refuse fp6 for the drafter's layers, the skip list did
  not reach them. Report the line and use an fp8 copy of the config
  (`kv-cache-dtype: fp8`, base `NVFP4_MJOLNIR`).
- `mjolnir verify spec --legs mjolnir:NVFP4_MJOLNIR_E2M3_DSPARK --drafts "5 6 7"`
  reads the acceptance per position. A much shorter accepted length than
  DSpark's (about 3–4.4 tokens) or DFlash2's (about 4.1–5.5 tokens) means
  something is off. Both are model-card numbers on an H200 with bf16 targets.
- Accuracy: `mjolnir bench accuracy --backends NVFP4_MJOLNIR_E2M3_DSPARK`.
  Probabilistic drafting is lossless in theory; check it.

Sources: [DSpark drafter (vLLM fork)](https://huggingface.co/Doopeworld/Qwen3.8-27B-DSpark-vLLM),
[DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
[vLLM v0.31.0 model registry](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/models/registry.py),
[Speculative decoding on Jetson](https://www.jetson-ai-lab.com/tutorials/speculative-decoding/).
