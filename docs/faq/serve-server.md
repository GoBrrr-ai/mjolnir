# Starting, stopping, and inspecting the server

**Story:** "I want to serve the active model with the active image, check
that it's healthy, and look at what's in the queue — without hand-rolling
`docker run`."

## The commands

```bash
mjolnir serve up          # start (active model/config/image; baked-in defaults if no state)
mjolnir serve status       # container state + /health + queue load (running/waiting)
mjolnir serve logs -f      # tail the vLLM container logs
mjolnir serve down         # stop the container
```

`serve up` waits for `/health` before returning (first launch takes minutes:
model download + FlashInfer/Triton JIT on Thor). It never touches a running
server's state — see the
[clean-window gate](clean-window-gate.md) for why killing the server to
"clean the GPU" is forbidden.

## One-off overrides (no state change)

```bash
mjolnir serve up --model Qwen/Qwen3.8-27B --config NVFP4 \
                 --image narandill/mjolnir-vllm:v0.31.0 \
                 --port 6001
mjolnir serve up --dry-run     # print the exact docker run, do nothing
```

- `--config` picks the attention backend too: `NVFP4_MJOLNIR` (default,
  Mjolnir), `NVFP4` (FlashInfer), `NVFP4_FA2` (FlashAttention 2, bf16 KV).
- `--wait/--no-wait`, `--wait-timeout` — health-wait behavior.

## What a first launch does

- Downloads the HF checkpoint into `~/.local/share/mjolnir/models/`
  (`--download-dir /data/models/huggingface` in the container). Already have
  the model in a host HF cache? `export MJOLNIR_HF_CACHE=~/.cache/huggingface/hub`
  (the HF_HOME root `~/.cache/huggingface` works too) mounts that hub as the
  download dir and `HF_HUB_CACHE` instead — no second copy, config and
  tokenizer included. Serve, calibrate and the verify containers all use it.
  A path that doesn't exist is an error, never a silent re-download. The
  containers run as root, so anything they add to that hub is root-owned.
- Keeps the FlashInfer JIT cache, Triton cache, and kernel cache on the
  persistent `~/.local/share/mjolnir/` volumes, so **restarts are fast** and
  the one-time JIT cost is paid once.
- Mounts `configs/` read-only at `/configs` inside the container; the serve
  command is `vllm serve --config /configs/<model>/<quant>/config.yaml …`
  (add your local config: [new-model-config.md](new-model-config.md)).
- Holds vLLM until the **host** has been up 2 minutes (`MJOLNIR_BOOT_DELAY_S`,
  default `120`, `0` turns it off). Mid-session there is no wait. After a hang
  and a power cycle, it leaves time to `ssh` in and stop a container that a
  restart policy would bring straight back
  ([#27](https://github.com/thor-arena/mjolnir/issues/27)).

## Sanity check after up

```bash
mjolnir serve status     # expect: running + health ok + queue 0/0
mjolnir gate --once      # expect: CLEAN
```

## The Mjolnir backend

The default config (`configs/<model>/NVFP4_MJOLNIR/config.yaml`) sets
`attention-backend: MJOLNIR`; for it, `serve up` adds `VLLM_KV_CACHE_LAYOUT=HND`
(the kernel reads head-major pages) and passes through any `MJOLNIR_ATTN_*`
variable from your shell (the knobs are listed in
[`src/mjolnir/README.md`](../../src/mjolnir/README.md#configuration)):

```bash
MJOLNIR_ATTN_Q_DTYPE=bf16 mjolnir serve up      # e.g. keep Q in bf16
```

Without the CLI, on the image:

```bash
docker run --gpus all --ipc host -e VLLM_KV_CACHE_LAYOUT=HND -p 8000:8000 \
  narandill/mjolnir-vllm:v0.31.0 \
  vllm serve Qwen/Qwen3.8-27B --attention-backend MJOLNIR
```

On a stock vLLM 0.31.0 (no image patch), `pip install mjolnir-kernel[dsl]` and
use `--attention-backend CUSTOM`: the plugin registers itself in the `CUSTOM`
slot through the `vllm.general_plugins` entry point. Prefill and mixed batches
run on the fallback backend (`MJOLNIR_ATTN_FALLBACK`): FlashAttention 4 on the
image, FlashInfer on a stock vLLM (whose FA on sm_110 cannot read fp8 KV);
Mjolnir takes the decode steps.
