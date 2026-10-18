# A/B: custom image vs the default image

**Story:** "I built `my/mjolnir-vllm:dev` and want to know whether it's
faster than the current default image — a fair, gated, end-to-end number,
not a vibes estimate."

## The one command

`mjolnir bench ab` restarts the server **per leg** (it restarts on purpose
— each leg is a different image or config, so the container must be
recreated), runs a full gated perf bench per leg, writes one history row-set
per leg, and re-renders the charts at the end.

Legs are `<config>` or `<image>:<config>` — the image carries the backend
(the kernel stack is baked into the image), the config carries the serving
parameters. Compare two images on the *same* config:

```bash
mjolnir bench ab --backends \
  'narandill/mjolnir-vllm:v0.32.0:NVFP4_MJOLNIR,my/mjolnir-vllm:dev:NVFP4_MJOLNIR'
```

Compare backends on one image (the default invocation — Mjolnir vs
FlashInfer vs FlashAttention 2):

```bash
mjolnir bench ab                     # default legs: NVFP4_MJOLNIR,NVFP4,NVFP4_FA2 on the active image
```

Pin a bare config leg to an image without writing it in every leg:
`mjolnir bench ab --image my/mjolnir-vllm:dev --backends
NVFP4_MJOLNIR,NVFP4`.

Protocol per leg (defaults): one gated sweep (6 measured runs per cell +
2 warmups, `--runs`), contexts 0/4K/8K × concurrency 1/2/4, `--exact-tg`
pinned (output length fixed — kills EOS-early-stop variance); `--repeat`
adds independent windows. Tune with `--runs/--repeat/--gate/--model/--port`.

## Schedule it deliberately

- It stops the live server between legs (`serve down` → `serve up` with
  health wait). The GPU is shared — pick a quiet window
  ([clean-window-gate.md](clean-window-gate.md)).
- The bench never *reimplements* the gate: every sweep waits for its own
  clean window; a dirty sweep is discarded, not averaged in.

## Read the comparison

```bash
mjolnir history                       # the tg t/s tables, newest first
mjolnir history --json | jq '.[0].backend'   # scriptable: one record per sweep
ls benchmarks/raw/                    # perf-<ts>/benchy-r*.json — full raw data
```

Each leg is a row-set in `benchmarks/history.jsonl` labeled with image +
model + config + backend (e.g. "Mjolnir" vs "FlashInfer"), so
`mjolnir plot --readme` shows them in `assets/benchmarks/e2e-decode.png` /
`e2e-prefill.png` and the README's end-to-end tables.
Judge with `tg_tps` (decode throughput, tokens/s) per cell — the number the
kernel work moves — and `pp_tps` for prefill.
