# AWQ_MJOLNIR — Qwen3.8-27B on Jetson Orin (sm_87), the Mjolnir decode kernels

twolven's AWQ (compressed-tensors int4 W4A16, the built-in MTP head) with the
fp8 KV on the Mjolnir Orin kernels: the FMA GEMV and the `mma.m16n8k16`
decode kernel (both hd 256, fp8 KV, q_len ≤ 8 — the MTP K=3 verify shape is
q_len 4). The weights ride Marlin W4A16 (the native int4 path on sm_87); the
prefill and the mixed steps ride the fallback — resolved automatically:
the FA4 default fails its fp8-KV probe on sm_87 and the backend downgrades
to FlashInfer (`MJOLNIR_ATTN_FALLBACK=FLASHINFER` forces it explicitly if
ever needed).

The AGX Orin 64GB memory/batch tuning lives in `platforms/orin.yaml` (shared
with AWQ_FLASHINFER). fp8 KV ships no scales (the checkpoint has none; the
descales default to 1.0 — the same posture as AWQ_FLASHINFER).

## Status

- [ ] serve smoke on the AGX Orin (the first vLLM-path test for this family —
      `verify vllm` is N/A until the hd128 widening or a hd256 smoke model)
- [ ] `bench e2e` vs AWQ_FLASHINFER (restarts the server per leg — needs an
      owned window)
- [ ] `bench accuracy`
