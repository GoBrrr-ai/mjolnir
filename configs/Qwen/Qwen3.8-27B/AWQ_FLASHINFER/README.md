# AWQ_FLASHINFER

`Qwen/Qwen3.8-27B` · `configs/Qwen/Qwen3.8-27B/AWQ_FLASHINFER/` — one serving
config: the `vllm serve` flags in YAML, the platform deltas in
`platforms/<platform>.yaml` overlays, the env-only switches in the `.env`
sidecar. [BENCHMARKS.md](BENCHMARKS.md) carries the numbers; [the model's
README](../README.md) the sibling configs.

## config.yaml

| key | value |
|---|---|
| `model` | `twolven/Qwen3.8-27B-abliterated-AWQ-MTP` |
| `served-model-name` | `Qwen/Qwen3.8-27B` |
| `quantization` | `compressed-tensors` |
| `dtype` | `bfloat16` |
| `load-format` | `safetensors` |
| `performance-mode` | `throughput` |
| `optimization-level` | `3` |
| `compilation-config` | `{cudagraph_num_of_warmups: 5, custom_ops: [+rotary_embedding], pass_config: {eliminate_noops: true,
    fuse_allreduce_rms: false}}` |
| `disable-custom-all-reduce` | `True` |
| `trust-remote-code` | `True` |
| `enforce-eager` | `False` |
| `enable-auto-tool-choice` | `True` |
| `tool-call-parser` | `qwen3_coder` |
| `reasoning-parser` | `qwen3` |
| `attention-backend` | `flashinfer` |
| `enable-flashinfer-autotune` | `True` |
| `linear-backend` | `auto` |
| `kv-cache-dtype` | `fp8` |
| `kv-cache-dtype-skip-layers` | `sliding_window` |
| `mamba-cache-mode` | `align` |
| `mamba-cache-dtype` | `bfloat16` |
| `mamba-ssm-cache-dtype` | `bfloat16` |
| `language-model-only` | `True` |
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


### Jetson Orin

```bash
mjolnir serve up --platform orin --config AWQ_FLASHINFER
```


| key | value | from |
|---|---|---|
| `max-num-seqs` | `5` | platforms/orin.yaml |
| `max-model-len` | `190K` | platforms/orin.yaml |
| `max-num-batched-tokens` | `8K` | platforms/orin.yaml |
| `gpu-memory-utilization` | `0.9` | platforms/orin.yaml |
| `prefix-match-unit` | `32` | platforms/orin.yaml |



Availability: ✅ Marlin serves W4A16 (GPTQ/AWQ) from compute capability 7.5 up — the bandwidth-optimized classic edge format.


Explore the numbers in [the interactive charts](https://thor-arena.github.io/mjolnir/#config=AWQ_FLASHINFER).
