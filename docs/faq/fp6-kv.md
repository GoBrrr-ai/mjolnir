# Serving on a 6-bit (e2m3) KV cache

**Story:** "I want the Mjolnir kernel's fp6 KV format in vLLM, not just in
the harness — fewer KV bytes per token, more tokens in memory."

## What it is

`kv-cache-dtype: fp6_e2m3` stores every K and V element as an e2m3 code
(sign, 2-bit exponent, 3-bit mantissa: 0 … ±7.5), four codes in three bytes —
**25% fewer KV bytes than fp8, 33% more tokens in the same memory**. The
Mjolnir decode kernel reads it directly (`tcgen05 kind::f8f6f4`); a value is
`code × scale`, with one static scale per layer, KV head and **channel**.

| step | who reads the cache | how |
|---|---|---|
| decode (incl. MTP verify / draft) | Mjolnir (T5) | packed fp6 straight from the pages |
| prefill, mixed, chunked | FlashAttention 4 (the fallback) | the step's pages are expanded to e4m3 in a scratch first: `code × s_c / S`, with one descale `S` per head |
| every step | Mjolnir writes the cache | one Triton kernel: scale, round, pack, scatter |

The image adds the dtype to vLLM (`thor-fp6-e2m3-kv-cache.patch`); only the
`MJOLNIR` backend with the `FLASH_ATTN` fallback accepts it.

## The scales: calibrate once per model

e2m3's range is tiny (steps of 0.125 below 1, max 7.5 — about 60× of dynamic
range), so the scales are fitted to the data **per channel**. Per KV head is
not enough: K has outlier channels (Qwen3-0.6B layer 0: K amax 486 in one
channel, a few units in the rest), and a head-wide scale of 486 / 7.5 rounds
every small channel to zero — the first smoke run with per-head scales got the
first token wrong.

Per-channel scales cost nothing at decode time — they never reach the kernel
as descales:

- **K**: `q · k = Σ_c (q_c · s_c) · code_c` — the channel scales multiply Q
  inside the fused fp8 Q pack (same one launch).
- **V**: `O_c = s_c · Σ_t p_t · code_tc` — one multiply of the output.
- **FA4 fallback**: the scratch holds `e4m3(code × s_c / S)` with
  `S = max_c(s_c) × 7.5 / 448` per head as FA4's descale (one e4m3 rounding).

vLLM 0.31 has no runtime KV-scale calculation, so Mjolnir calibrates offline:

```bash
mjolnir serve down                  # calibration loads the model itself — in a window you own
mjolnir calibrate                   # the active model; fp8 KV, varied prompts, ~minutes
# -> configs/<model>/kv_scales.json  (per layer, KV head and channel amax of K and V)
mjolnir serve up --config NVFP4_MJOLNIR_E2M3
```

`mjolnir serve` points `MJOLNIR_ATTN_KV_SCALES` at the `kv_scales.json` next
to the config; set it yourself to use another file. Commit the file with the
config — it is part of the serving setup. Without it, the server refuses to
start with fp6 (no silent bad scales). Files from before per-channel scales
(`mjolnir.kv_scales/1`, one amax per head) still load, but give per-head
accuracy — re-run `mjolnir calibrate`.

Knobs: `MJOLNIR_ATTN_KV_SCALE_HEADROOM` (default 1.0) multiplies every scale —
above 1 leaves room for values beyond the calibrated amax (they saturate at
±7.5 × scale otherwise) at the cost of resolution near zero.

## Check it before switching

```bash
mjolnir verify vllm                                 # includes "smoke e2m3": calibrates
                                                    # Qwen3-0.6B, serves it on fp6, compares
                                                    # logprobs with FlashInfer on fp8
mjolnir bench e2e --only mjolnir --config NVFP4_MJOLNIR_E2M3
mjolnir bench compare --config NVFP4_MJOLNIR_E2M3
```

What to expect: the fp6 kernel reads 25% fewer bytes; in ncu it is 1.03–1.09×
the fp8 kernel at 4K–32K and **1.15–1.18× at 64K–256K**. End to end, attention
is a fraction of a decode step on Qwen3.8-27B (16 of 64 layers; the NVFP4
weights dominate), so the gain shows at long contexts and several concurrent
requests — and as KV capacity. Accuracy on synthetic data (`docs/kernel/quant.md`):
relative L2 error 0.052 vs fp8's 0.045; on a real model, judge it with the
smoke test and your own evals before making fp6 the default.
