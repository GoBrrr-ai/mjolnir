# NVFP4_MJOLNIR_E2M3_LMH8_VT64K

`nvidia/Qwen3.6-35B-A3B` · `configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR_E2M3_LMH8_VT64K/` — one serving
config: the `vllm serve` flags in YAML, the platform deltas in
`platforms/<platform>.yaml` overlays, the env-only switches in the `.env`
sidecar. [BENCHMARKS.md](BENCHMARKS.md) carries the numbers; [the model's
README](../README.md) the sibling configs.

## config.yaml

| key | value |
|---|---|
| `model` | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| `served-model-name` | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| `quantization` | `modelopt` |
| `trust-remote-code` | `True` |
| `max-model-len` | `262144` |
| `max-num-seqs` | `1` |
| `gpu-memory-utilization` | `0.45` |
| `enable-auto-tool-choice` | `True` |
| `tool-call-parser` | `qwen3_coder` |
| `reasoning-parser` | `qwen3` |
| `attention-backend` | `MJOLNIR` |
| `kv-cache-dtype` | `fp6_e2m3` |
| `speculative-config` | `{method: mtp, num_speculative_tokens: 5}` |



## .env sidecar

| key | value |
|---|---|
| `MJOLNIR_LM_HEAD` | `fp8` |
| `MJOLNIR_DRAFT_VOCAB` | `/configs/nvidia/Qwen3.6-35B-A3B/draft_vocab.json` |
| `MJOLNIR_DRAFT_VOCAB_SIZE` | `65536` |



## Per platform

One serve line per platform the overlays put this config on; the deltas name
the file each value came from (the base `config.yaml` or the overlay).


### Jetson AGX Thor

```bash
mjolnir serve up --platform thor --config NVFP4_MJOLNIR_E2M3_LMH8_VT64K
```



No overlay deltas on Jetson AGX Thor — the base `config.yaml` serves as-is.


Availability: ✅ The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.


Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=NVFP4_MJOLNIR_E2M3_LMH8_VT64K).
