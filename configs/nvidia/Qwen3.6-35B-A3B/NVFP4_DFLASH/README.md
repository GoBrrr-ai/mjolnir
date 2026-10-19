# NVFP4_DFLASH

`nvidia/Qwen3.6-35B-A3B` · `configs/nvidia/Qwen3.6-35B-A3B/NVFP4_DFLASH/` — one serving
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
| `kv-cache-dtype` | `fp8` |
| `attention-backend` | `flashinfer` |
| `enable-auto-tool-choice` | `True` |
| `tool-call-parser` | `qwen3_coder` |
| `reasoning-parser` | `qwen3` |
| `speculative-config` | `{method: dflash, model: z-lab/Qwen3.6-35B-A3B-DFlash, num_speculative_tokens: 12}` |



## .env sidecar

| key | value |
|---|---|
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `0` |
| `VLLM_GDN_DECODE_KERNEL` | `triton` |



## Per platform

One serve line per platform the overlays put this config on; the deltas name
the file each value came from (the base `config.yaml` or the overlay).


### Jetson AGX Thor

```bash
mjolnir serve up --platform thor --config NVFP4_DFLASH
```



No overlay deltas on Jetson AGX Thor — the base `config.yaml` serves as-is.


Availability: ✅ The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window.


Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=NVFP4_DFLASH).
