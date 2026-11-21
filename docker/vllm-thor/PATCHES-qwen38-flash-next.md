# Qwen3.8-Flash-Next (qwen4_exp) serving stack (2026-11-17)

The deep-dive part of [PATCHES.md](PATCHES.md) for these patches (their rows are in
the PATCHES.md table); split out to keep PATCHES.md under the file-length limit.

The patches and `mjolnir.qwen4exp` modules that serve `nvidia/Qwen3.8-Flash-Next-NVFP4`
and `nvidia/Qwen3.8-Flash-Next-FP4-Local` on Thor (configs under
`configs/nvidia/Qwen3.8-Flash-Next/`), on vLLM 0.34.0. Each Thor-authored
piece is inert unless a config opts in through its `.env`; its vLLM hunks
import `mjolnir.qwen4exp` only inside the switched branch. The loader backport
is always on and changes nothing for checkpoints without nested shards. The
measured effects quoted below come from the contributor's Thor runs on the
official **v0.33.0** image with the same code as file overlays
([NemoClaw-Thor](https://github.com/pastoriomarco/NemoClaw-Thor) `69df863`,
`serving/docs/QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md`, `FLASH-NEXT-SPEED-PLAN.md`);
they are not Mjolnir evidence until `mjolnir bench` reproduces them, and they
are re-measured on 0.34. vLLM 0.34 carries the CSF-compressed NVFP4 scales
FP4-Local needs (#64312 + #64869), so no backport is needed for them.

**Legend.** The run labels below are the contributor's 0.33 measurement steps
(named as in the NemoClaw-Thor docs above); each adds to the previous one.

| label | what it is | config it led to |
|---|---|---|
| R2 | FP4-Local, PLE table on NVMe with prefetched lookups, MTP 4 | |
| R3 | R2 + reduced draft vocabulary + online FP8 MTP dense layers | |
| R4b (-K3) | all routed experts on Marlin W4A16 (K3: MTP 3) | |
| R5 | Marlin MoE + FlashInfer MXFP8 side layers, FP8 draft head, draft FULL graphs | |
| R6 | R5 + screened target head + FlashInfer MoE | |
| R7b | R6 at memory share 0.70 + PLE prewarm | |
| R8a | R7b + the PLE slot-pool cache (0.5 GiB) at memory share 0.75 | |
| R8b | R8a + CuTe DSL W4A4 routed experts | `FP4LOCAL` (+ C2 + C3) |
| N1 | the NVFP4 checkpoint on the R8b stack, side layers BF16 | |
| N2 | N1 + online FP8 side layers (incl. the QSA `qkv_proj` with its fused indexer) | |
| N3 | N2 with the QSA `qkv_proj` kept BF16 | `NVFP4_SFP8` (+ C1 + C2 + C3) |
| C1 | only the QSA indexer kept BF16, the rest of `qkv_proj` FP8 | `NVFP4_SFP8` |
| C2 | FP8 hyperconnection GEMMs (`QWEN4EXP_HC_FP8`) | both |
| C3 | MXFP8 CUTLASS tactic pin (`MJOLNIR_MXFP8_TACTIC`) | both |

**Index subdirectory (`64739`).** Upstream backport (open PR, split out of
#64312), test files left out. Without it the FP4-Local checkpoint needs a
hand-made directory with the shards linked at the top level (how it was
served before this port). **Drop when** the base carries #64739.

**PLE slot-pool cache in front of the disk table (`thor-qwen4-ple-disk`, `mjolnir/qwen4exp/ple_disk*.py`).**
Both checkpoints carry one PLE n-gram table (layer 1, ~320 M rows; 26.8 GiB
NVFP4 in FP4-Local, 47.7 GiB FP8 in NVFP4). Kept in pinned host memory or on
the GPU, it allocated ~112 GB on Thor's unified pool before loading and the
model could not start. vLLM 0.34 (#59958) keeps it in the checkpoint files
with `--engram-config '{"disk_offload": true}'` (set in the YAML), the on/off
switch for the disk table: each step syncs the stream once, then reads that
step's distinct rows with one serial `pread` each on the main thread, with no
cache and no prefetch. With `disk_offload` on and `MJOLNIR_PLE_DISK_CACHE_GB` > 0
(`.env`), the patch builds Mjolnir's layer instead (not a subclass of
upstream's, so neither the per-step sync nor upstream's staging applies): it
finds the shard files and offsets from the loader's views the way upstream
does (`/proc/self/maps`), and `bind_file_shards` opens a GPU CLOCK slot pool
of that size (design after TensorRT Edge-LLM 0.12 `--ple-residency cached`)
inside `load_model`, so the pool counts as model memory before KV sizing. A
step reads only its missed rows (page-aligned merged `pread`s, parallel
readers for the ones that wait for the disk; prefill-sized miss sets via the
mmap gather), uploads them with an int32 slot map and decodes through the
slot map with upstream's lookup kernels; prefetch
(`MJOLNIR_PLE_DISK_PREFETCH`, default on) stages a step on a background thread
while the preceding decoder layer runs. With the cache at 0 or unset,
upstream's reader runs unchanged. Needs PIECEWISE graphs (the lookup is an
eager break) and ETP = TP = 1. Measured effect (contributor's Thor runs,
FP4-Local, recipe R7b → R8a in the contributor's doc; **measured on 0.33 with
Mjolnir's own disk reader underneath, against that reader without the pool;
to be re-measured on 0.34 against upstream's `disk_offload`**): decode GPU
time per step 51.7 → 49.3–50.0 ms, warm 7.6K prefill 3.2 → 3.04 s, KV 311,928
tokens at share 0.70 → 475,281 at 0.75 (the pool removes the page-cache
reservation that held the share at 0.70). CPU tests:
`tests/cli/test_qwen4exp_ple_disk_{table,cache,layer}.py`. **Drop when**
upstream's disk table gains a row pool and asynchronous prefetch (e.g. vLLM
#66941, #66464); the switch then stays `disk_offload` alone.

**Draft vocabulary and 8-bit drafter (`thor-qwen4-draft-head`).** The head
(248,320 × 2,560 BF16, 1.27 GB) is read by the target and by every MTP draft
pass. The draft passes score only the configured vocabulary
(`configs/nvidia/Qwen3.8-Flash-Next/draft_vocab.json`: the contributor's
47,149-id coding vocabulary — all byte and special tokens — padded to 47,232
rows) through their own head, built from the checkpoint's `lm_head` rows while
the drafter loads, so it is in the profiled model memory and, with
`QWEN4EXP_DRAFT_HEAD_FP8=1`, quantized online (block FP8, 0.125 GB per pass
instead of 1.27 GB) without a resident BF16 copy. `QWEN4EXP_MTP_DENSE_FP8=1`
does the same for the drafter's dense projections (fc_embedding/fc_hidden,
QSA qkv/o, shared expert: 0.139 → 0.072 GB per pass; estimated ~0.3 ms, not
measured alone; the QSA indexer projection is a target too, so vLLM 0.34,
which fuses it into `qkv_proj` only when both get the same method, keeps it
FP8 inside the fused GEMM as on 0.33). The file format, `MJOLNIR_DRAFT_VOCAB(_SIZE)` and
`mjolnir calibrate --draft-vocab` are `mjolnir.lmhead`'s. **For every
qwen4_exp model** `MJOLNIR_DRAFT_VOCAB` therefore builds this drafter-owned
head and `mjolnir.lmhead` skips its own VocabTrim for that drafter;
`MJOLNIR_LM_HEAD` does not cover the reduced head (`QWEN4EXP_DRAFT_HEAD_FP8`
does), and a checkpoint with a quantized `lm_head` is not supported by this
path (the rows are sliced from the BF16 checkpoint tensor). lmhead's VocabTrim
cannot serve qwen4_exp as is: its BF16 → fp8 kept head needs
`MJOLNIR_LM_HEAD=fp8`, which also re-quantizes the target head (changed target
logits, and no BF16 head left for the screened head below), and its per-row
Triton GEMV was never measured on this model. Measured (FP4-Local, one
stream): draft vocabulary + MTP dense FP8 (R2 → R3) bench prose / code / count
29.4 / 45.9 / 54.9 → 37.5 / 56.7 / 51.1 tok/s (counting drops: the coding
vocabulary covers number words less well); the FP8 draft head ~1.9 ms per step
(R5); a 47K vs 65K vocabulary A/B showed no difference. Drafter-only: the
target output is unchanged. **Drop when** vLLM carries a reduced draft
vocabulary for MTP (#58578 is a feature request).

**Screened target head (`thor-qwen4-screened-head`).** PR #60566's bound and
three Triton kernels, unchanged; switched by env instead of
`--screened-lm-head`, built at the end of `Qwen4ExpForCausalLM.load_weights`
(so the 0.64 GB FP8 copy is profiled model memory) and decided per target
`sample()` call from host-side state only (no sync; persistent buffers, every
row count pre-compiled). Exact except BF16 rounding ties (the PR measured 12
greedy flips in 1.36M live rows, all ties); the rejection sampler reads the
same top-k target probabilities, so MTP stays exact. Refuses next to
`MJOLNIR_LM_HEAD=fp8` (full-head calls would then use the fp8 head while the
exact pass keeps the BF16 rows). Measured (FP4-Local, R5 → R6): sampling incl. head GPU
5.3 → 2.6 ms per step. Opt-in only, as the watch list below asks for #60566.
`QWEN4EXP_LM_HEAD_SCREEN_CHECK=N` compares with the full head in-server,
`QWEN4EXP_LM_HEAD_SCREEN_STATS=<s>` counts screened / full-head calls.
The `Sampler.add_request` replacement mirrors vLLM 0.34's, including
`may_mask_logits` (the rejection sampler's masked-vocabulary-block skip), and
adds only #60566's logits-processor split; re-check it at every bump.
**Drop when** the base carries #60566 (then `--screened-lm-head`).

**Side FP8 and C1 (`thor-qwen4-side-fp8`).** The NVFP4 checkpoint leaves 192
dense target modules BF16 (5.86 GB read per step). `QWEN4EXP_SIDE_FP8=1`
quantizes them online at load (128×128 block FP8 by default, CUTLASS
block-scaled GEMM, no resident BF16 copy) through the stock composition that
`--quantization-config '{"targets": …}'` uses, plus the guard the flag lacks:
it refuses to re-quantize layers the checkpoint already quantizes. N1 → N2
(all kinds): step GPU 57.4–57.6 → 45.6–46.2 ms, but Tool Eval 91 → 86. The
QSA indexer shares the QSA `qkv_proj` GEMM; keeping that GEMM BF16 (N3,
`QWEN4EXP_SIDE_FP8_SKIP=self_attn.qkv_proj`, which keeps the indexer BF16 too)
restored 93; C1 keeps only the 12 indexer projections BF16 (2.950 instead of
3.359 GB per step): −1.3…−1.5 ms per step, Tool Eval 94, 94 (N3 + C1 + C2).
**On vLLM 0.34** `Qwen4ExpQSAAttention` fuses the indexer into `qkv_proj` only
when both resolve to the same quant method type (`nvidia/qsa.py`), so C1 is
just `QWEN4EXP_SIDE_FP8_SKIP=self_attn.indexer.index_qk_proj`: the indexer
stays BF16 and its own linear. Without a skip the indexer is a target too and
stays FP8 inside the fused GEMM, as on 0.33. These numbers were measured on
0.33, where C1 needed an unfuse switch (`QWEN4EXP_QSA_UNFUSE_INDEXER`, an init
subclass of the QSA layer) that 0.34 no longer needs; they are re-measured on
0.34. Changes target outputs (W8A8): opt-in per config. **Drop when** vLLM's
own `--quantization-config` targets refuse quantized layers (then
config-only).

**C2: FP8 hyperconnections (`thor-qwen4-hc-fp8`).** The HC GEMMs read
1.279 GB BF16 per target step; online 8-bit cuts that to 0.661 GB (and the
resident weights by ~0.6 GB). Measured on Thor (interleaved runs in one
session): GPU time per step −1.3 ms (N3 + C1) / −1.4 ms (R8b + C3), step
47.3 → 46.2 ms on R8b + C3, unchanged on N3 + C1 (host-bound there), warm
7.6K prefill +0.13 s, KV +21–25K tokens; Tool Eval in noise (R8b 92, 91;
N3 + C1 94, 94). Changes target outputs (W8A8 in every layer's residual
mixing): opt-in per config. **Drop when** upstream quantizes the qwen4_exp
HC projections. On 0.34 a converted up projection also skips the new fused
`hc_up_gate_mix` kernel (SM100 family only, so inert on Thor), which checks the
weight dtype at build time and would otherwise read the quantized weight.

**Draft FULL graphs (`thor-draft-full-cg-piecewise`).** Qwen3.8-Flash-Next's
target must stay PIECEWISE (the PLE lookup and QSA are host-side eager
breaks), which in 0.33 leaves draft passes 1..K-1 eager: ~100+ launches each
including the head. The draft decode of an MTP drafter is a uniform 1-token
batch whose QSA layer replays from persistent buffers only, so it can take a
FULL graph regardless of the target's mode. vLLM #60989 (0.34) moved the
decode-graph decision into `AutoRegressiveSpeculator.init_cudagraph_manager`
(`spec_decode/autoregressive/speculator.py`), so the gate sits there and the
switch now applies to every autoregressive drafter (MTP, EAGLE/EAGLE3,
standalone draft models, Gemma4); `multi_module_mtp`, `dflash` and `ngram`
keep their own gates. On 0.34 the QFN drafter does not take the fused
multi-step decode: on Thor vLLM logs `Attention backend(s)
QWEN4_EXP_QSA_INDEXER cannot update draft decode metadata in place; falling
back to rebuilding attention metadata eagerly`, then `[spec-step] draft
decode: FULL CUDA graphs` — each draft pass rebuilds its metadata on the host
and replays a FULL graph (Mjolnir's `thor-fused-draft-t5-plan-refresh`, which
targets the fused path, is not involved). Measured on
0.33 (FP4-Local, one stream): eager draft passes 2–4 took
1.8 ms GPU / 4.5–6 ms host each plus ~0.9 ms host metadata (R3); with the
graphs (R5) 1.2–1.3 ms GPU / ~0.1 ms host. Step GPU time did not change
(R4b-K3 54.8 → 54.9 ms): the gain is host time, which counts once the host
no longer runs ahead. Drafter-only: target output unchanged. Mjolnir's sm_110 draft-CG gate found a *small* graphed
drafter slower than eager at N=1, hence off by default. The 0.34 draft path
is re-measured on Thor.

**C3: MXFP8 tactic pin (`thor-mxfp8-tactic-sm110`).** FP4-Local ships its 192
side matrices MXFP8 (`FlashInferCutlassMxfp8LinearKernel`). GPU micro-bench
over those 192 matrices per forward, streamed from DRAM: tactic 0 vs 8 is
14,014 → 12,209 µs at M=4 (one stream, MTP 3), 13,931 → 12,311 µs at M=16,
15,318 → 14,524 µs at M=256; tactic 8 (tile 128×64×128, cluster 2×2×1) is
within 0.5 % of the best tactic per shape up to M=256, larger tiles win above,
so `256:8` keeps prefill on the stock call. Bit-identical to the stock call
(eager and `torch.compile`, M = 1…8192). Serving (R8b → R8b + C3, without
C2): cycle 48.8 → 47.0 ms per decode step. With C2 the 97 HC up projections are MXFP8
too (`QWEN4EXP_HC_FP8_FORMAT` default `fp8_per_block,mxfp8`), on both
checkpoints, so the pin covers them as well: on Thor (2026-11-17, the HC
chain alone) tactic 8 cuts it 5.97 → 4.86 ms at M=4 (−1.11 ms; −1.08 / −1.07 ms
at M=16 / 24), bit-identical (`torch.equal` on state and output, per site and
over the chain, every M). Both recipes therefore set `MJOLNIR_MXFP8_TACTIC=256:8`;
on NVFP4 it covers only the HC up projections (its side layers are block FP8). A plugin hook was not used: the pin sits inside
`apply_weights` (a wrapper would copy its body or replace the module-global
`mm_mxfp8` for every caller). Keep `enable-flashinfer-autotune: false`. Canary:
`kernel_test_sm110_gates.py` T13 (bit-identity vs the stock call). FlashInfer
0.7.3 (vLLM 0.34) carries byte-identical CUTLASS MXFP8 sources and the same
private runner API as the 0.7.2.post1 the numbers come from; T13 re-checks
bit-identity on the 0.34 image. **Drop
when** FlashInfer's default tactic choice for small M on sm_110 matches it.

