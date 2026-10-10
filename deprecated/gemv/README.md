# Archived: the FA4 hd256 GEMV decode kernel

Before Mjolnir, this repo's decode kernel was a **pure-FMA GEMV** kernel
(`sm100_hd256_decode_gemv`) patched into vLLM's vendored FlashAttention 4 for
head_dim 256, plus a tensor-core `mma.sync` alternative (`tc_spark`) and the
first `tcgen05` prototype (`sm100_hd256_decode_t5`) that became Mjolnir. The GEMV
kernel reached about 0.57× of FlashInfer's decode on Thor at long context: it was
latency-bound on a bandwidth-bound device, which is the lesson that set
Mjolnir's design (tensor cores + TMA ring + a one-wave split-KV plan; see
[`docs/kernel/`](../../docs/kernel/README.md)).

Nothing here is built, linted, tested or shipped in the image. It is kept so
the history — and the measurements behind the design decisions — stay
traceable.

| path | what |
|---|---|
| `kernel/` | the kernels, verify suites, bench scripts and their README (was `docker/vllm-thor/fa4-gemv-kernel/`) |
| `patches/` | the vLLM build patch that dispatched M=1 hd256 decode to the GEMV kernel |
| `docs/` | design, FlashInfer gap analysis, ring fix, long-context diagnosis, TC parity plans, the FAQ guides and the literature review |
| `benchmarks/` | the GEMV raw results and its `history.jsonl` row |
| `assets/` | the README charts of the GEMV era |
| `cli/vfa.py`, `scripts/`, `opencode/` | the kernel-dev CLI tooling, the 4-way ncu sweep, the old kernel-iteration skill |

The FA4 1-CTA hd256 patch (`thor-fa4-hd256-1cta-decode-sm110`) is **not**
archived — it still ships in the image for the `NVFP4_FA4hd256` config.
