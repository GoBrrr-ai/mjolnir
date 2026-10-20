# Kernel experiment backlog (hand-curated)

> **Program of record:** [`research/program.md`](program.md) — the autoresearch
> loop's ranked ideas and the keep/discard judge rule live **there**. This file
> is the hand-curated backlog extracted from `AGENTS.md` (open work),
> `docs/platforms/thor/kernel/*` (each chapter's "Ideas to try" / open
> questions) and git history (the deleted `docs/research/decode-experiments.md`
> and retired implementations). Statuses: **untried** · **tried-retired** ·
> **landed** (shipped as a default — do not re-run as an experiment) ·
> **resolved** (diagnosed, nothing to fix).
> The `ab` stage is the generic experiment slot: `mjolnir verify ab --variants
> "name:--flag"` — see [ab.md](../docs/platforms/thor/kernel/ab.md).

## Active ideas

1. **e2e at the shapes that matter** — decode gains only turn into throughput
   at long contexts (64K+) with concurrency; measure the MTP verify shape and
   capacity-as-throughput cells.
   *Why:* AGENTS.md "Open work"; [sweep.md](../docs/platforms/thor/kernel/sweep.md);
   [kv-cache-size.md](../docs/platforms/thor/kernel/kv-cache-size.md).
   *How:* `mjolnir verify sweep --batch 4 --q-len 4` (MTP=3 verify shape, ncu);
   `mjolnir bench perf --concurrency 8,16` (depths default 0–128K);
   `mjolnir bench e2e --depths 65536,131072 --concurrency 1 --runs 3`
   (per-request cells merge into the table).
   *Status:* untried.
2. **Native e2e headline** — FlashInfer on the official `vllm/vllm-openai`
   image vs Mjolnir, end to end (never on the Mjolnir image — it patches
   FlashInfer paths).
   *Why:* AGENTS.md "Open work".
   *How:* `mjolnir bench e2e` — restarts the server per leg; schedule
   deliberately in a window you own.
   *Status:* untried.
3. **Headline kernel data refresh** — every stage once on the current kernel,
   raws committed, charts re-rendered (README shows "pending" for stages
   without a default run).
   *Why:* AGENTS.md "Open work".
   *How:* `mjolnir verify bench` (includes correctness), commit
   `benchmarks/raw/verify/**`, then `uv run mjolnir plot --readme`.
   *Status:* untried.
4. **fp6 byte-vs-time floor** — e2m3 reads 0.75× the bytes of fp8 but runs at a
   flat ~0.84–0.86× (a uniform ~10–17% latency floor at every L; L2 at 17–25%
   of peak, `long_scoreboard` the top stall). Not the 64-byte L2 requests
   (SW128 retired), not the block table (readahead retired), not the 16K–32K
   hole (ncu artifact). Open lever named in the chapters: ring depth chosen
   per L at runtime.
   *Why:* AGENTS.md "Open work" (fp6 byte-vs-time gap);
   [fp6prof.md](../docs/platforms/thor/kernel/fp6prof.md) "Ideas to try" → Open.
   *How:* `mjolnir verify fp6prof`; `mjolnir verify sweep --batch 4 --q-len 4`.
   The kernel-side ranked list with commands is
   [`research/program.md`](program.md) § "Ideas, ranked by expected value".
   *Status:* untried (the SW128 / readahead / artifact legs are retired, see
   below).
5. **fp4 KV with finer scales** — per-channel K scales folded into Q and
   per-token V scales folded into P (KIVI-style), to bring e2m1/nvfp4 accuracy
   toward fp6; +33% KV capacity over fp6 if accuracy holds.
   *Why:* AGENTS.md "Open work" (fp4 with finer scales);
   [kvfmt.md](../docs/platforms/thor/kernel/kvfmt.md);
   [quant.md](../docs/platforms/thor/kernel/quant.md).
   *How:* `mjolnir verify quant` + `mjolnir verify kvfmt` (accuracy vs speed
   per format); on a real model `mjolnir bench accuracy`; later a
   `NVFP4_MJOLNIR_FP4KV` config through `mjolnir verify vllm` (smoke) and
   `mjolnir verify spec`.
   *Status:* untried.
6. **NVFP4 KV with a Hadamard rotation** (BitDecoding / KVLinC): rotate K's
   head dim and fold the rotation into Q (like the per-channel scales).
   Expected ≤1% at ≤32K, ~+5% at 128K, +33% capacity over fp6; high effort +
   accuracy risk.
   *Why:* deleted `docs/research/decode-experiments.md` #8 (git `c8cd814`).
   *How:* `mjolnir verify quant`, `mjolnir verify kvfmt`; then the config →
   `verify vllm` smoke → `verify spec` path as in #5.
   *Status:* untried.
7. **Decode beyond attention — the six topics** (AGENTS.md "Open work";
   ranked list with A/B commands in the deleted
   `docs/research/decode-experiments.md`, git `c8cd814`):
   - **Draft-step overhead** — *resolved:* no robust per-draft penalty (the
     earlier "+2.3 ms/draft" was run noise; Mjolnir runs FULL graphs + fused
     draft decode, 0.6% GPU idle). Do not re-hunt without new evidence.
   - **fp8 lm_head** — *landed:* `MJOLNIR_LM_HEAD` (`…_LMH8`), measured +23%
     at k=3; part of `DEFAULT_QUANT`. Config:
     `NVFP4_MJOLNIR_E2M3_LMH8_VT64K`.
   - **VocabTrim draft head** — *landed:* 64K draft vocab (`…_VT64K`), +40%
     decode vs the fp6 head at 1 request; part of `DEFAULT_QUANT`. The 32K
     variant is retired (−9% acceptance on benchy text vs −3% for 64K).
     Re-calibrate after model changes: `mjolnir calibrate --config …
     --draft-vocab`.
   - **fp8 MTP layer** — *untried:* the drafter's transformer layer is still
     bf16 in the checkpoint; ~+3–4% expected (−1 GB/step), low effort on top
     of the fp8 head. [spec.md](../docs/platforms/thor/kernel/spec.md)
     "Ideas to try". Opt-in config `…_MTP8` → `mjolnir verify vllm` (smoke) →
     `mjolnir verify spec --legs mjolnir:NVFP4_MJOLNIR_E2M3,mjolnir:…_MTP8
     --drafts "3 4 5"` (acceptance must hold) → `mjolnir bench ab --backends …
     --repeat 2`.
   - **Draft length** — *partially landed:* k=4 won the sweep and
     `DEFAULT_QUANT` ships MTP=5; the open part is **adaptive per-request
     draft length** (stop drafting when the draft's top-1 confidence drops;
     SpecDec++ / PACER style), behind a flag.
     *How:* `mjolnir verify spec --drafts "2 3 4 5"`; then the adaptive flag,
     same command.
   - **Long-context decode** — merged into #1 above.
8. **Short-context route** (PersistentKV-style) — below ~1–2K tokens skip
   split-KV + merge entirely (Mjolnir is ~0.5× FlashInfer at L=128:
   +16 µs × 16 layers); expected <0.5% e2e, low effort.
   *Why:* deleted `docs/research/decode-experiments.md` #9.
   *How:* `mjolnir verify sweep --ls "128 1024 2048 4096" --batch 4 --q-len 4`.
   *Status:* untried.
9. **FA4 first-use compile stalls / short-prompt TTFT** — one-off ~2.7–3 s
   stalls on the Mjolnir image (first 4 requests at c=4, one at c=1) — looks
   like FA4's CuTe-DSL compile per prefill shape (`cannot be converted to a
   JitArgument` per compile); short-prompt TTFT is ~0.2 s slower than official
   overall. Fix candidates: persist FA4's compiled kernels (the CuTe DSL
   export/load the Mjolnir kernel cache uses) or warm the prefill shapes at
   startup.
   *Why:* deleted `docs/research/decode-experiments.md` (diagnosis section).
   *How:* `grep -c "cannot be converted to a JitArgument"
   benchmarks/logs/verify-<run>/spec/*.log`; A/B per-request TTFT
   (`ttft_p50_ms` in `verify spec` raws) before/after the fix.
   *Status:* untried.
10. **Opt-in drafter legs on a Thor** — DSpark / DFlash2, K=7, instead of MTP
    (`NVFP4_DSPARK` / `NVFP4_DFLASH2` official image;
    `NVFP4_MJOLNIR_E2M3_DSPARK` / `…_DFLASH2` Mjolnir) — not yet run on a
    Thor.
    *Why:* AGENTS.md "Configs (Qwen3.8-27B)"; [spec-drafters
    FAQ](../docs/faq/spec-drafters.md).
    *How:* `mjolnir bench e2e --optional` (restarts per leg).
    *Status:* untried.
11. **DSpark on Mjolnir + fp6 drafter** — the DSpark drafter runs on Mjolnir
    with fp6 too: its own calibration (`kv_scales/dspark.kv_scales.json`,
    `mjolnir calibrate --config …`) and the kernel's non-causal variant for
    its block steps (`MJOLNIR_ATTN_NONCAUSAL`). Not yet run.
    *Why:* AGENTS.md "Configs (Qwen3.8-27B)".
    *How:* calibrate, then `mjolnir verify vllm` (smoke) → `bench e2e
    --optional`.
    *Status:* untried.
12. **Spark bring-up probes** — research platform (`sm_121`, DGX OS + CUDA 13,
    official-image baselines only). Verify per image at bring-up: (a) FA2
    availability is build-dependent — grep the "Using … attention backend"
    log line; (b) FlashInfer GDN-on-SM121 was mid-fix upstream (PR #3731
    merged, #3960 open) — re-check at the FlashInfer version the image pins
    before serving the Qwen3.6/3.8 hybrids; (c) the MoE backend changed
    across vLLM 0.24→0.26 (FLASHINFER_CUTLASS removed, −13% decode) — pin the
    image and grep the MoE-backend log line, don't assume kernel continuity;
    (d) perf-mode/optimization-level cross-platform defaults are untested.
    *Why:* AGENTS.md "Platform layer (detection-first)" +
    "Upstream tracking"; detailed provenance in the Spark stack audit
    (untracked `task/research/spark-stack.md`).
    *How:* serve each candidate image, grep the backend/MoE log lines, record
    under the platform; `mjolnir platform list` to confirm maturity.
    *Status:* untried.
13. **Orin bring-up probe** — evidence is thinner than Thor/Spark (one field
    report, JP 7.2 + vLLM 0.26.0; no NVIDIA Orin vLLM playbook found). Host
    needs JP 7.2.1+ (L4T 39.2.1, CUDA 13.2.2); default config
    `AWQ_FLASHINFER` (int4 W4A16 + fp8 KV on FlashInfer).
    *Why:* AGENTS.md "Platform layer" + "Configs (Qwen3.8-27B)".
    *How:* serve `AWQ_FLASHINFER` from the official image on JP 7.2.1+, grep
    the attention-backend log line, record one e2e row.
    *Status:* untried.
14. **FA4 + hd256 + FP8 KV (prefill) port** — blocked upstream: FA4's hd256
    kernel is bf16-only (`descale_tensors is None` assert) and vLLM downgrades
    FA4→FA2 for hd256 + quantized KV, so FlashInfer is the only FP8-KV path
    for the hd256 layers. The 6-step probe-gated plan (upstream descale branch
    → port ~200–400 LOC → widen gates → sm_110 bring-up → config `block_size=
    128` → e2e A/B) is in AGENTS.md § "FA4 + hd256 + FP8 KV (prefill)".
    *Status:* blocked — wait for / request the upstream FP8-descale branch
    (RFC #2456); re-check before investing.

## Per-chapter leads

Genuine "worth trying" items from each chapter's *Ideas to try*; pointers
(`→ #n`) dedupe into Active ideas.

### [sweep.md](../docs/platforms/thor/kernel/sweep.md)
- batch > 1 and the MTP verify shape → #1.
- varlen batches: `mjolnir verify sweep --batches 8 --varlen`. *untried.*
- page size 16 (vLLM's FlashInfer backend on Thor): `mjolnir verify sweep
  --page-size 16`. *untried.*

### [hndsweep.md](../docs/platforms/thor/kernel/hndsweep.md)
- FlashInfer at page size 16 and 64, where its native vLLM path runs:
  `mjolnir verify hndsweep --page-size 16` (and `--page-size 64`). *untried.*

### [ring.md](../docs/platforms/thor/kernel/ring.md)
- 1 CTA per SM with a deeper ring (double the smem per CTA fits deeper
  rings): `mjolnir verify ring --base "--ctas_per_sm 1"`. *untried.* (The
  other direction, `ctas_per_sm=3`, dies on TMEM — see program.md.)

### [prefetch.md](../docs/platforms/thor/kernel/prefetch.md)
- prefetch only the first tiles of each split (warm-up), not the steady state
  — the steady-state distances lost (see Retired #6), the warm-up variant is
  untried. Kernel flag first, then `mjolnir verify prefetch`.

### [pdl.md](../docs/platforms/thor/kernel/pdl.md)
- measure PDL inside a full vLLM CUDA graph, where the neighbors are real
  model kernels: `mjolnir verify profile` with and without
  `MJOLNIR_ATTN_PDL=1`. *untried* (isolated wall clock shows no consistent
  win — see Retired #7).

### [fp6prof.md](../docs/platforms/thor/kernel/fp6prof.md)
- the Open item (uniform latency floor vs byte ratio) → #4.

### [kv-cache-size.md](../docs/platforms/thor/kernel/kv-cache-size.md)
- e2e sweep at 8–16 concurrent requests and 64K+ context → #1.
- fp4 KV once its accuracy holds on a real model → #5/#6.

### [kvfmt.md](../docs/platforms/thor/kernel/kvfmt.md) / [quant.md](../docs/platforms/thor/kernel/quant.md)
- fp4 with finer-grained scales + e2e evals on a real model → #5.

### [spec.md](../docs/platforms/thor/kernel/spec.md)
- fp8 MTP layer → #7d; adaptive draft length → #7e.

### [correctness.md](../docs/platforms/thor/kernel/correctness.md)
- add a correctness case for any shape your model uses (`stages.CASES` in
  `src/mjolnir-cli/mjolnir_cli/verify/stages.py`); GQA ratios other than 6 are
  already covered by the 40 query / 8 KV drafter cases. *untried* (tooling).

### [ab.md](../docs/platforms/thor/kernel/ab.md)
- the generic experiment slot (anything, via `--variants`) — that is what the
  autoresearch loop and program.md use; see the top note.

## Retired / no-win — do not retry without new evidence

1. **Fused last-CTA split-KV merge** — correct (97/97) but 0.56× at 4K and
   loses at every L; retired in favour of a cheaper (non-fused) merge
   (program.md idea 6). Autoresearch run 1; commit `566468f`
   (`--fused_merge`, default off). *tried-retired.*
2. **hd256 SplitKV on Thor** — inert: the merged upstream heuristic
   (`arch // 10 in [10, 11]`, FA4 #2916) auto-disables on the 20-SM Thor for
   our decode shapes (48 clusters > 20 SMs → 0 splits), corroborating the
   in-tree no-win. AGENTS.md "Upstream tracking". Retry only if the upstream
   scheduler changes. *tried-retired (upstream).*
3. **Block-table readahead (bt_lookahead, E2a)** — correct but no win: wall
   ratios move ±2–4% around zero at 16K–128K and the `long_scoreboard` share
   does not drop — the stall is not the block table. Commit `39fa326`
   (flag-first); [fp6prof.md](../docs/platforms/thor/kernel/fp6prof.md).
   *tried-retired.*
4. **SW128 (keep fp6 L2 requests at 128 B) for every L** — ~5% faster at
   64K–256K but slower at short L; gated at ≥64K it is +1.7% overall — under
   the 2% bar alone. Commit `e2d09c2`; fp6prof.md. Retry only stacked on a
   short-L win (program.md idea 4). *tried-retired (standalone).*
5. **fp6 unpacked in registers instead of the TMA** — not worth it: the
   nvfp4 convert warpgroup shows what a conversion pass costs (0.48–0.69×
   FlashInfer). fp6prof.md. *tried-retired (analysis).*
6. **Steady-state L2 prefetch** — distances 1/2/4 are 7–27% slower at ≥8K;
   `prefetch=0` wins the geomean (1.14×). With an occupancy-pinned grid the
   prefetch duplicates traffic. [prefetch.md](../docs/platforms/thor/kernel/prefetch.md).
   The warm-up-only variant stays active (Per-chapter leads). *tried-retired.*
7. **PDL as a default** — 12 of 16 wall-clock cells within ±3% noise (range
   −3.4%…+5.7%), +15% at L=128 in the worst cell; stays opt-in
   (`MJOLNIR_ATTN_PDL=1`). [pdl.md](../docs/platforms/thor/kernel/pdl.md).
   The in-graph measurement stays active (Per-chapter leads). *tried-retired
   (as default).*
8. **Interleaved / NHD pages for Mjolnir** — fp8-interleaved geomean 1.00×
   and fp8-nhd 1.04× vs 1.25× on HND (red at most L ≥ 8K); HND is the layout.
   [kvfmt.md](../docs/platforms/thor/kernel/kvfmt.md) +
   [hndsweep.md](../docs/platforms/thor/kernel/hndsweep.md). *tried-retired.*
9. **Per-head KV scales for fp6** — failed smoke e2m3 (Qwen3-0.6B: K outlier
   channels, amax 486 in layer 0, the other channels rounded to 0 — wrong
   first token). Scales are per (layer, KV head, channel), `kv_scales/2`; a
   `/1` file still loads but is per head. AGENTS.md "Open work" (fp6 KV).
   Re-calibrate after model changes. *tried-retired.*
10. **VT32K draft vocab** — superseded by the 64K vocab: on llama-benchy text
    VT32K loses ~9% acceptance vs ~3% for VT64K at nearly the same speed.
    Deleted decode-experiments.md (git `c8cd814`). *tried-retired.*
11. **The draft-step overhead hunt** — resolved: no robust per-draft penalty
    exists (per-draft slopes move ±5 ms between runs, decode CV ~10% with
    MTP); Mjolnir's drafter runs FULL graphs + fused draft decode at 0.6% GPU
    idle. Deleted decode-experiments.md. *resolved — nothing to fix.*
12. **The 16K–32K fp6 "hole"** — resolved as an ncu aggregation artifact:
    identical launches flip ±25% under replay and fp6prof averaged 2–3 of
    them; the fastest-launch snapshot fix (2026-10-17, `39fa326`) shows wall
    clock flat at 0.84–0.86× fp8 at every L. AGENTS.md "Open work" +
    [fp6prof.md](../docs/platforms/thor/kernel/fp6prof.md). *resolved.*
13. **`MJOLNIR_ATTN_DRAFT_T5=0` on fp6** — not a variant, a refused
    combination: the drafter's fp6 fallback steps (page expansion from a
    per-step page map) were captured into FULL graphs and replayed stale
    (acceptance 0, HTTP 500) — the backend now refuses it at startup, as any
    fp6 fallback inside a capture. Deleted decode-experiments.md.
    *resolved (guard shipped).*
