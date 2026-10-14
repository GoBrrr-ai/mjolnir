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
| `NVFP4_MJOLNIR_E2M3_DSPARK` | Mjolnir | Mjolnir decode, fp6 KV — the drafter too (own calibration) | DSpark, K=7 |
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

**DSpark on the Mjolnir image runs on Mjolnir and fp6 KV, like the target.**
That matters because its layers keep a K/V for every context token: 20 KB a
token on bf16 next to the target's 24 KB on fp6. On fp6 they take 7.5 KB, so
the cache holds about 1.4× the tokens of a bf16 drafter. Three pieces make it
work:

- **The backend, set explicitly.** vLLM chooses a DSpark / DFlash drafter's
  attention backend separately from the target's: `speculative-config.attention_backend`,
  or automatic selection when that is unset. It never inherits the target's
  `attention-backend`. The automatic choice knows nothing of fp6 and fails at
  startup with `No valid attention backend found … kv_cache_dtype=fp6_e2m3 …
  use_non_causal=True`. So the config sets `attention_backend: MJOLNIR` for
  the drafter.

- **Its own calibration.** `mjolnir calibrate --config NVFP4_MJOLNIR_E2M3_DSPARK`
  serves the config on fp8 KV with the drafter on Mjolnir and records the
  per-channel amax of the target's and the drafter's K/V. It writes them to
  `NVFP4_MJOLNIR_E2M3_DSPARK.kv_scales.json` next to the config. That file is
  separate from the model's `kv_scales.json`, so the other configs are
  untouched. `mjolnir serve` points the config at it, and refuses to start the
  config until the file exists.
- **The drafter's block on the decode kernel.** A DSpark draft step is one
  bonus query and 7 mask queries. They attend to the whole context and to each
  other: non-causal attention. The decode kernel now has a non-causal variant
  (`--non_causal` in the harness, `MJOLNIR_ATTN_NONCAUSAL` in vLLM), so these
  steps run on it directly from fp6. They are also CUDA-graph safe. Before
  this, they could only go to the FlashAttention 4 fallback, whose fp6 page
  expansion cannot be captured in a CUDA graph. `mjolnir verify correctness`
  checks the variant against the emulator, and the autoresearch quick set
  includes it: 1 + 7 queries, 40 query / 8 KV heads, d = 128.

**DFlash2 stays on bf16 KV and FlashInfer**, on both images:
`kv-cache-dtype-skip-layers: ["sliding_window", "64", …, "68"]` and
`speculative-config.attention_backend: FLASHINFER`.

- Its layers use a 2048-token sliding window, which neither the decode kernel
  nor the FA4 fallback implements.
- The window caps its KV at about 42 MB a request on bf16. fp6 would save
  about 26 MB a request, which is not worth a new kernel feature.

The official-image legs keep both drafters on bf16 KV. Stock vLLM has no fp6,
and bf16 runs the non-causal block on any backend.

The target's verify steps (1 + 7 tokens per request) run on the Mjolnir decode
kernel in every Mjolnir leg.

## Before trusting a number

None of the four configs has run on a Thor yet. On the first start:

- DSpark on Mjolnir: `mjolnir calibrate --config NVFP4_MJOLNIR_E2M3_DSPARK`
  first. Check its output lists the drafter's layers (64–68) next to the
  target's.
- `mjolnir serve up --config <cfg>`, then check the log for:
  - on DSpark (Mjolnir): `fp6_e2m3 KV scales ... for 5 layer(s)` for the
    drafter's group, and `Mjolnir decode enabled: h_q=40 h_k=8 d=128`;
  - on DFlash2: the drafter's attention backend, the KV dtype of layers 64–68,
    and the model runner.
- If DFlash2 on Mjolnir refuses fp6 for the drafter's layers, the skip list did
  not reach them. Report the line and use an fp8 copy of the config
  (`kv-cache-dtype: fp8`, base `NVFP4_MJOLNIR`).
- `MJOLNIR_ATTN_NONCAUSAL=0` sends the drafter's block steps back to the
  fallback, for an A/B. It works on fp8 KV only: on fp6, those fallback steps
  would have to be captured in CUDA graphs, which the backend refuses.
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
