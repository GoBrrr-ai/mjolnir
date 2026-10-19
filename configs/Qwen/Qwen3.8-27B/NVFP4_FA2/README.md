# NVFP4_FA2

`Qwen/Qwen3.8-27B` · `configs/Qwen/Qwen3.8-27B/NVFP4_FA2/` — one serving
config: the `vllm serve` flags in YAML, the platform deltas in
`platforms/<platform>.yaml` overlays, the env-only switches in the `.env`
sidecar. [BENCHMARKS.md](BENCHMARKS.md) carries the numbers; [the model's
README](../README.md) the sibling configs.

## config.yaml

| key | value |
|---|---|
| `model` | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| `served-model-name` | `Qwen/Qwen3.8-27B` |
| `dtype` | `bfloat16` |
| `performance-mode` | `throughput` |
| `optimization-level` | `3` |
| `compilation-config` | `{cudagraph_num_of_warmups: 5, custom_ops: [+rotary_embedding], pass_config: {eliminate_noops: true,
    fuse_allreduce_rms: false}}` |
| `disable-custom-all-reduce` | `True` |
| `trust-remote-code` | `True` |
| `enforce-eager` | `False` |
| `max-num-seqs` | `10` |
| `max-model-len` | `160K` |
| `max-num-batched-tokens` | `16K` |
| `gpu-memory-utilization` | `0.88` |
| `enable-auto-tool-choice` | `True` |
| `tool-call-parser` | `qwen3_coder` |
| `reasoning-parser` | `qwen3` |
| `attention-config` | `{backend: FLASH_ATTN, flash_attn_version: 2}` |
| `enable-flashinfer-autotune` | `True` |
| `mm-encoder-attn-backend` | `TORCH_SDPA` |
| `linear-backend` | `auto` |
| `kv-cache-dtype` | `auto` |
| `kv-cache-dtype-skip-layers` | `sliding_window` |
| `mamba-cache-mode` | `align` |
| `mamba-cache-dtype` | `bfloat16` |
| `mamba-ssm-cache-dtype` | `bfloat16` |
| `language-model-only` | `False` |
| `mm-encoder-tp-mode` | `data` |
| `mm-processor-cache-type` | `shm` |
| `enable-prefix-caching` | `True` |
| `enable-chunked-prefill` | `True` |
| `async-scheduling` | `True` |
| `speculative-config` | `{method: mtp, num_speculative_tokens: 3}` |
| `override-generation-config` | `{min_p: 0.0, presence_penalty: 0.0, repetition_penalty: 1.0, temperature: 1.0, top_k: 20,
  top_p: 0.95}` |
| `default-chat-template-kwargs` | `{enable_thinking: true, preserve_thinking: true, reasoning_effort: medium}` |
| `chat-template` | `/configs/Qwen/Qwen3.8_chat_template.jinja` |



## Per platform

One serve line per platform the overlays put this config on; the deltas name
the file each value came from (the base `config.yaml` or the overlay).


### Jetson AGX Thor

```bash
mjolnir serve up --platform thor --config NVFP4_FA2
```



No overlay deltas on Jetson AGX Thor — the base `config.yaml` serves as-is.


Availability: ✅ BF16 (and fp16) KV is the unquantized baseline: every backend class serves it wherever that backend itself runs.


Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=NVFP4_FA2).
