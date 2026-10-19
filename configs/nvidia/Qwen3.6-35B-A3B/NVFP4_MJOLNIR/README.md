# NVFP4_MJOLNIR

`nvidia/Qwen3.6-35B-A3B` · `configs/nvidia/Qwen3.6-35B-A3B/NVFP4_MJOLNIR/` — one serving
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
| `dtype` | `bfloat16` |
| `trust-remote-code` | `True` |
| `enforce-eager` | `False` |
| `max-num-seqs` | `1` |
| `max-model-len` | `262144` |
| `max-num-batched-tokens` | `16K` |
| `gpu-memory-utilization` | `0.45` |
| `enable-auto-tool-choice` | `True` |
| `tool-call-parser` | `qwen3_coder` |
| `reasoning-parser` | `qwen3` |
| `mm-encoder-attn-backend` | `TORCH_SDPA` |
| `linear-backend` | `auto` |
| `mamba-cache-mode` | `align` |
| `mamba-cache-dtype` | `bfloat16` |
| `mamba-ssm-cache-dtype` | `bfloat16` |
| `enable-prefix-caching` | `True` |
| `enable-chunked-prefill` | `True` |
| `async-scheduling` | `True` |
| `attention-backend` | `MJOLNIR` |
| `enable-flashinfer-autotune` | `True` |
| `kv-cache-dtype` | `fp8` |
| `kv-cache-dtype-skip-layers` | `sliding_window` |
| `speculative-config` | `{method: mtp, num_speculative_tokens: 3}` |



## Per platform

One serve line per platform the overlays put this config on; the deltas name
the file each value came from (the base `config.yaml` or the overlay).


### Jetson AGX Thor

```bash
mjolnir serve up --platform thor --config NVFP4_MJOLNIR
```



No overlay deltas on Jetson AGX Thor — the base `config.yaml` serves as-is.


Availability: ✅ The Mjolnir kernel reads e4m3 KV natively on thor as its default dtype.


Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=NVFP4_MJOLNIR).
