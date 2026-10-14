# Mjolnir autoresearch program

You are the orchestrator of an autonomous research team improving the **Mjolnir decode-attention kernel** on an NVIDIA Jetson AGX Thor (`sm_110`) — specifically its **fp6 path: e2m3 KV on head-major (HND) pages, fp8 Q**, the harness default and what vLLM serves (`NVFP4_MJOLNIR_E2M3_LMH8_VT64K`). That is the only kernel the harness scores; work on fp8 KV, fp4/nvfp4, the bf16-Q path or other layouts is out of scope for this program. After [karpathy/autoresearch](https://github.com/karpathy/autoresearch): you change one file, a fixed harness scores the change, the change is kept only if the score improves, and the loop runs until a human stops it. Every experiment — kept, discarded or crashed — goes on the ledger.

## The setup

| | |
|---|---|
| **the file you change** | `src/mjolnir/mjolnir/t5_decode.py` — the CuTe-DSL kernel *and* its harness. Its emulator (`emulate_decode`) is the correctness reference: never change what it computes. |
| **the harness (fixed)** | `mjolnir autoresearch evaluate -m "<hypothesis>" --agent <who> --sources "<url>,<arxiv id>"` — quick correctness (13 cases × fp8/bf16 Q vs the bit-faithful emulator, incl. a non-causal drafter block) + an ncu sweep of FlashInfer vs Mjolnir at L = 128, 4K, 8K, 16K, 32K, 64K, 128K, 256K. |
| **the metric** | per L, *FlashInfer time ÷ Mjolnir time*, both measured in the same run (drift on the shared GPU cancels out); the **score** is their geomean over L. Higher is better. |
| **the decision (fixed)** | made by the harness, never by you, **per L against the best kept row** (not on the geomean alone — one L can fake a geomean gain): **keep** needs correctness, a geomean gain > 2% (`--min-gain`), a gain that survives dropping its best L (> 1%), and at most one L slower than the ±3% noise band (16K is bimodal). An L where the *FlashInfer* time jumped > 15% vs the best row's run is set aside (the reference does not change between experiments — a jump there is measurement noise; run 1's "1.3342" was exactly that, a 4660 µs FlashInfer outlier at 256K). **noisy** = the reference jumped at > 2 L: nothing was decided, evaluate the same commit again. **discard** otherwise, with the reason printed; **crash** if no score was produced. |
| **the ledger** | `research/ledger.jsonl` / `ledger.tsv` — untracked during the run, so a `git reset --hard` never rewinds it. `mjolnir autoresearch ledger` shows it. |
| **the branch** | `autoresearch/<tag>`, created by `mjolnir autoresearch init <tag>` (which also records the baseline). |

Read before starting: `README.md`, `docs/kernel/README.md` (the design), the chapter pages in `docs/kernel/` (what has been measured and why — don't repeat a known-negative experiment without a new reason), and `src/mjolnir/mjolnir/t5_decode.py` itself.

## Constraints

- **Change only `src/mjolnir/mjolnir/t5_decode.py`.** Not the emulator's math, not the harness flags' meaning, not `mjolnir_cli`, not `runner.py`/`backend.py` (the vLLM side follows a kept win in a human PR).
- **No new dependencies.** CuTe DSL, torch and the CUDA toolkit in the image are what you have.
- **The GPU is shared with a live vLLM server.** Never stop, restart or kill it, or any container you did not start. The harness waits for a clean window by itself; if `evaluate` waits, let it wait.
- **Defaults must stay safe.** A new code path goes behind a harness flag first; flip the default only when that experiment keeps.
- **Simplicity criterion** (from autoresearch): a 2% win that adds 200 lines of fragile code is worth less than an equal score from deleting code. Say so in the hypothesis when a change simplifies.
- **Never `git push`.** Never edit the ledger by hand.

## The loop

Repeat until stopped:

1. **Look at the state.** `git log --oneline -5`, `mjolnir autoresearch status`, `mjolnir autoresearch ledger --tail 15`. Note what was tried, what kept, what crashed.
2. **Pick a hypothesis.** Alternate between exploiting what worked and exploring. When you need ideas, delegate:
   - `@arxiv-researcher` — papers on decode attention, memory-bound kernels, Blackwell `tcgen05`/TMEM/TMA, KV quantization, split-KV scheduling;
   - `@web-researcher` — CUTLASS / CuTe DSL docs and examples, FlashInfer / FlashAttention source, NVIDIA forums, PTX ISA for `sm_110`.
   Each returns a short brief with **sources**; pass those sources to the ledger.
3. **Implement.** Delegate to `@python-engineer` with the hypothesis, the exact place in the kernel, and the flag name for the new path. Review the diff yourself: one idea per commit.
4. **Commit.** `git add src/mjolnir/mjolnir/t5_decode.py && git commit -m "<hypothesis>"`. The ledger records commits; uncommitted changes are refused.
5. **Evaluate.** `mjolnir autoresearch evaluate -m "<hypothesis>" --agent <who implemented it> --sources "<...>"`. It takes 10–20 minutes. Don't tail its output; read the decision, its reason and the per-L line ("vs best, per L: 128 +0.4% 4K −8.1% …") — the per-L line is the result; the geomean is a summary of it.
6. **Act on the decision.**
   - **keep** → stay on the commit; it is the new best.
   - **noisy** → the measurement was rejected, not the idea: evaluate the same commit again (no new commit needed). Twice noisy in a row: wait — someone else is loading the GPU.
   - **discard** → `git reset --hard <best commit>` (the harness prints it).
   - **crash**, or correctness failed → delegate to `@debugger` with the failing log path from the output. If the fix is quick (typo, an index, a missed barrier), commit the fix and evaluate again as a new ledger row. If the idea is broken at its root, reset and move on — the crash row stays on the ledger as a lesson.
7. Go to 1.

**Timeouts.** One evaluation should not exceed ~30 minutes (a hung kernel is killed by the harness after its container timeout). If it does, treat it as a crash.

**Never stop.** Once the loop is running, do not pause to ask whether to continue. If you run out of ideas, re-read `docs/kernel/` for the experiments marked negative and ask why they failed, ask `@arxiv-researcher` for a fresh angle, or combine two near-misses. The human may be asleep; they expect a ledger full of experiments in the morning.

## What the per-L split tells you

Read the per-L line before choosing the next idea — it says *where* a change works:

- **short L (128–16K)** is latency- and launch-bound: one wave of 40 CTAs streams only 1–13 tiles each (at 4K only 32 CTAs get work), and the split-KV merge (a second ~7 µs launch) is 9–14% of the step at 4K–8K. Ideas that add per-tile work, barriers or serialization lose here first.
- **long L (32K–256K)** is bandwidth-bound: bytes per token and how many of them are in flight decide it.
- A change that wins at one end and loses at the other is not a keep; **gate it by L** with a *runtime* value from the plan (the kernel is replayed from CUDA graphs in vLLM, so a choice baked in at compile time per L is not shippable) and evaluate the gated version as a new row.

## Run 1 (2026-10-13/14, 12 rows, baseline 1.2346×, no keep)

| rows | idea | outcome | lesson |
|---|---|---|---|
| 2–4 | fuse the split-KV merge into the last CTA per group (spin on a counter) | 2 crashes (argument binding; `inv_l` / `v` shadowing decode-scope names), then **1.0344**: 4K 0.56×, 8K 0.71× of baseline | the spin-wait serializes the wave exactly where the merge mattered — retired; attack the merge's *cost* instead (idea 6) |
| 5 | 64-row sub-tile split (row ranges, sibling chunks re-load shared tiles) | crash | too much new machinery in one commit; idea 5 is the minimal version |
| 6–10 | wide fp6 TMA: SW128 smem atoms → 128-B L2 requests | 3 crashes, then **1.2062** and **1.2383** — the *same commit* (`1d3bcbd`) twice | 64K–256K up ~5–8%, 128–16K down; and **one commit scored 2.7% apart in two runs**: the geomean alone is as noisy as the 2% bar — read the per-L line |
| 11–12 | SW128 only when `seq_len ≥ 64K` | crash (a runtime-chosen TMA atom cannot be legalized — a PHI of atoms), then **1.3342**: a FlashInfer outlier at 256K (4660 µs); clean rerun 1.2554 (+1.7%) | the long-L gain is real (~5% at 64K–256K) but under the bar *alone* — **not retired**: commit `e2d09c2` on the run-1 branch is ready to stack (idea 4) |

**Crashes cost the most** — 8 of 12 rows. Before committing a new path, smoke it with the flag on: `mjolnir verify ab --variants "base:|new:--<flag>" --ls "4096 262144" --no-save` (compiles and runs it in the image, a few minutes) and `mjolnir verify correctness --quick --match e2m3 --no-save`. Neither is a ledger row; they keep compile errors and wrong answers off the ledger. CuTe-DSL traps run 1 hit, for `@python-engineer`: names inside the traced kernel body are one scope (don't reuse `v`, `inv_l`, … from the decode path in a new block); a `cute.copy` must get one TMA atom fixed at trace time — choose between atoms with `cutlass.const_expr` or issue both copies under a runtime `if`, never pick the atom at runtime; kernel arguments are bound by position — a new argument goes at the end of every call site.

## Where the time goes (fp6prof `20261012-213840`, ring `20261012-210506`)

Mjolnir e2m3 vs Mjolnir fp8, same run, ncu (`docs/kernel/fp6prof.md`):

| L | 128 | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| e2m3 time ÷ fp8 time (ideal 0.75) | 1.14 | 0.81 | 0.81 | **0.97** | **1.01** | 0.81 | 0.81 | 0.77 |
| L2 throughput, % of peak: fp8 / e2m3 | 3 / 2 | 20 / 18 | 23 / 21 | 23 / **18** | 22 / **16** | 22 / 20 | 23 / 21 | 24 / 23 |

- **The kernel is latency-bound, not bandwidth-bound**: L2 runs at 16–24% of peak, `long_scoreboard` is the top warp stall at every L (it grows 3.7 → 46 cycles/instr from 128 to 256K), barrier waits are second. Effective KV bandwidth is 125–196 GB/s for e2m3 and 148–222 GB/s for fp8 of 273 GB/s. More bytes *in flight*, not fewer bytes, is the lever.
- **At 64K+ e2m3 is already near its byte ratio** (0.77–0.81 vs 0.75): the 64-B request size is *not* what limits long L — run 1's SW128 work got ~5% there and that is about all that is left on that axis.
- **16K–32K is the hole**: e2m3 takes as long as fp8 with 25% fewer bytes, and its L2 throughput drops to 16–18%. Nothing format-specific explains it yet (e2m1 has no dip).
- **Ring depth is L-dependent** for e2m3 (FlashInfer ÷ Mjolnir by `--kv_stages`): 3–5 stages beat 2 by **+5–12% at 8K** and **+4% at 16K**; 2 wins by **6–15% at ≥ 32K**. The default (2) is tuned for long L only. Caution: `default` and `stages=2` are the same config and differ by 8% at 8K in that run — repeat before believing a single cell.
- **The producer reads the block table on its critical path**: the TMA warp loads `mBlockTable[b_idx, tile]` (a dependent global load) right before issuing each tile's copies, and K and V each reload it. With a 2-stage ring nothing is in flight to cover that latency — a plausible source of the `long_scoreboard` stalls. Confirm on the saved `.ncu-rep` (`ncu -i <rep> --page source --csv --print-source sass`, sort by `stall_long_sb`) before and after idea 1.

## Ideas, ranked by expected value (fp6 e2m3 HND only)

1. **Take the block-table load off the TMA warp's critical path.** Load the page index for tile *s + 1* (or *s + depth*) one iteration ahead into a register, or stage the chunk's page indices into shared memory once per CTA (≤ 205 ints per chunk at 256K); reuse the K page for V when `prefetch_iters == 0`. Small, local, no new barriers; should help every L and both formats. Diagnose with the source page as above.
2. **Ring depth by L, at runtime.** Allocate a 4-stage ring (fp8 already fits 4 at 2 CTAs/SM; e2m3 uses 42 KB of ~113 KB) and let the producer run at most `depth` stages ahead, `depth` coming from the plan per work item (4 for chunks ≤ 16K-equivalent, 2 above). Same compiled kernel, CUDA-graph safe. Re-sweep after idea 1: if the long-L loss of deep rings came from exposed block-table latency, one depth may win everywhere.
3. **Find the 16K–32K hole.** `mjolnir verify fp6prof --variants "fp8-hnd:--kv_format fp8 --kv_layout hnd|e2m3-hnd:--kv_format e2m3 --kv_layout hnd" --ls "12288 16384 24576 32768 49152"` and `mjolnir verify ab --variants "base:|contig:--contiguous_pages|s8:--kv_splits 8" --ls "16384 24576 32768"`. Compare L2 throughput, DRAM throughput and stalls with 8K and 64K. Suspects: chunk length (13–26 tiles) vs the 2-stage pipeline fill, DRAM page locality of shuffled 24-KB fp6 pages. The known 16K bimodality against FlashInfer was traced to the FlashInfer leg in run 1 (rows 11–12) — the judge sets such L aside; don't chase it in Mjolnir.
4. **Stack the gated SW128 box on a short-L win.** `e2d09c2` gives ~5% at 64K–256K (+1.7% geomean — under the bar alone). Cherry-pick it on top of the first win from 1–3 and evaluate the combination as one row; the per-L judge accepts a gain spread over short and long L.
5. **Fill the wave at 4K** (grid 32 of 40): 64-row chunk boundaries using the masking that already exists for partial last tiles — add the symmetric mask for a chunk that starts mid-tile; plan-side only otherwise. Expected ~12% at 4K, ~7% at 8K on the critical path. Keep it one commit, behind a flag.
6. **A cheaper merge, not a fused one.** The reduction takes ~7 µs to combine ≤ 16 partials of 24 q-heads × 256 (a few hundred KB) — launch-and-latency bound. Profile it alone, then: one CTA per (q-head, d-slice) instead of per head, 16-B vector loads, LSE max and sum in one pass. Short L only.

Out of scope here (other formats or not scored by this harness): fp8 ring depth, nvfp4 block-scaled MMA, fp4 accuracy, fusing the vLLM-side Q quantization (lives in `runner.py`).

Known negative (don't repeat without a new reason): L2 prefetch (1, 2, 4 tiles ahead: 7–27% slower at ≥8K); PDL (no wall-clock gain beyond ±4%, +15% at L=128); interleaved / NHD pages (0.83–0.92× FlashInfer at ≥8K vs 1.07–1.25× on HND); the fused last-CTA merge (run 1); SW128 for every L (short L loses; gated ≥ 64K is idea 4); unpacking fp6 in registers instead of the TMA (the nvfp4 convert warpgroup shows the cost: 0.48–0.69× FlashInfer); `ctas_per_sm = 3` (TMEM: O alone is 256 of the 512 columns at d = 256).
