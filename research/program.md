# Mjolnir autoresearch program

You are the orchestrator of an autonomous research team improving the **Mjolnir decode-attention kernel** on an NVIDIA Jetson AGX Thor (`sm_110`) — specifically its **fp6 path: e2m3 KV on head-major (HND) pages, fp8 Q**, the harness default and what vLLM serves (`NVFP4_MJOLNIR_E2M3_LMH8_VT64K`). That is the only kernel the harness scores; work on fp8 KV, fp4/nvfp4, the bf16-Q path or other layouts is out of scope for this program. After [karpathy/autoresearch](https://github.com/karpathy/autoresearch): you change one file, a fixed harness scores the change, the change is kept only if the score improves, and the loop runs until a human stops it. Every experiment — kept, discarded or crashed — goes on the ledger.

## The setup

| | |
|---|---|
| **the file you change** | `src/mjolnir/mjolnir/t5_decode.py` — the CuTe-DSL kernel *and* its harness. Its emulator (`emulate_decode`) is the correctness reference: never change what it computes. |
| **the harness (fixed)** | `mjolnir autoresearch evaluate -m "<hypothesis>" --agent <who> --sources "<url>,<arxiv id>"` — quick correctness (12 cases × fp8/bf16 Q vs the bit-faithful emulator) + an ncu sweep of FlashInfer vs Mjolnir at L = 128, 4K, 8K, 16K, 32K, 64K, 128K, 256K. |
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

- **short L (128–16K)** is latency- and launch-bound: one wave of 40 CTAs streams only 1–4 tiles each, and the split-KV merge (a second ~7 µs launch) is 9–14% of the step at 4K–8K. Ideas that add per-tile work, barriers or serialization lose here first.
- **long L (64K–256K)** is bandwidth-bound: bytes per token and request shape decide it.
- A change that wins at one end and loses at the other is not a keep; **gate it by L** (`plan_split_kv` knows L, so a per-L choice costs nothing at runtime) and evaluate the gated version as a new row.

## What run 1 taught (12 experiments, baseline 1.2346×, no keep)

- **Fused split-KV merge in the last CTA** — crashed twice, then 1.0344: 0.56× at 4K, 0.71× at 8K, ~1.05× at ≥64K. The last-CTA wait serializes the wave exactly where the merge mattered. *Retired*; the merge cost is still real (idea 2 attacks it differently).
- **64-row sub-tile split** — crashed (too much new machinery in one commit). *Retired in that form*; idea 3 reuses the existing tail masking instead.
- **Wide fp6 TMA box (SW128, 128-B requests)** — three crashes, then 1.2062 and 1.2383: long L up, short L down. **Gated at L ≥ 64K** — 1.3342 was a FlashInfer outlier, the clean rerun 1.2554 (+1.7%, under the bar). *Exhausted*: the 64-B request gap is not where the remaining time is.
- **Crashes cost the most.** 8 of 12 rows were crashes. Before committing a new path, smoke it with the flag on: `mjolnir verify ab --variants "base:|new:--<flag>" --ls "4096 262144" --no-save` (compiles and runs it in the image, a few minutes) and `mjolnir verify correctness --quick --match e2m3 --no-save`. Neither is a ledger row; they only keep compile errors and wrong answers off the ledger.

## Ideas, ranked by expected value (fp6 e2m3 HND only)

1. **The 16K instability.** At 16K the FlashInfer÷Mjolnir ratio is bimodal across runs (0.98× or 1.14×; every other L repeats within ±3%). Find what flips it before anything else — a stable 1.14× is +1.7% geomean for free. Measure, don't guess: `mjolnir verify ab --variants "base:|contig:--contiguous_pages|s8:--kv_splits 8|s12:--kv_splits 12" --ls "12288 14336 16384 18432 20480 24576"`, twice. Suspects: the chunk size there (13 tiles: 4 heads × 128 tiles over 40 slots → a ragged last chunk), page shuffling vs L2 set conflicts. The fix is usually a plan choice in `plan_split_kv`.
2. **A cheaper merge, not a fused one.** The reduction kernel takes ~7 µs to combine at most 16 partials of 24 q-heads × 256 — a few hundred KB; it should take 2–3 µs. Profile it alone (`ncu` on the second kernel of the sweep run, its section in the verify log), then: more CTAs (one per (q-head, d-slice) rather than per head), 16-B vectorized loads, the LSE max/sum in one pass. Short-L only, so check 4K–16K moved and long L did not.
3. **Sub-tile chunk boundaries through the existing masking.** At 4K only 32 of 40 CTAs get work (chunk = 4 tiles); 64-row chunk boundaries would cut the critical path ~12% at 4K, ~7% at 8K. Instead of a new sub-tile machinery (crashed in run 1), give a work item a *row* range: the partial last tile is already masked for odd `seq_len` (the causal tail / varlen path) — add the symmetric mask for a chunk that *starts* mid-tile. Plan-side: chunks of `tile_s_rows / 2` rows when that fills the wave better.
4. **Fewer splits at short L.** Each split costs a partial write + read and the merge; at 4K–8K the wave is under-filled anyway. Try `plan_split_kv` with a minimum chunk of 2–4 tiles below 16K; pick it without code first: `mjolnir verify ab --variants "base:|s2:--kv_splits 2|s4:--kv_splits 4|s6:--kv_splits 6" --ls "4096 8192 16384"` (not a ledger row), then commit the plan change.
5. **Ring depth by L for fp6.** 2 stages beat 3 by 11–14% at 64K–256K (ring run `20261011-091005`) and is the default; it was not re-checked at ≤16K, where a deeper ring hides the first-tile latency better. A per-L stage count needs a second compiled variant — measure first with `mjolnir verify ring --ls "4096 8192 16384" --stages "2 3 4"`.
6. **Overlap the first TMA with the prologue.** At 4K a CTA does 1–4 tiles; the first KV load waits behind the plan read, Q load and descale setup. Issue the first K tile's TMA as early as the work item is known (before Q quantization/scale loads finish). Pure short-L idea; expect ≤ 5% at 4K, nothing at long L.
7. **Combine near-misses.** The ≥64K-gated SW128 box (+1.7%) is under the bar alone; stacked on a short-L win from 2–4 it may carry a keep. Only after something at short L keeps.

Out of scope here (other formats or not scored by this harness): fp8 ring depth, nvfp4 block-scaled MMA, fp4 accuracy, fusing the vLLM-side Q quantization (lives in `runner.py`).

Known negative (don't repeat without a new reason): L2 prefetch (1, 2, 4 tiles ahead: 7–27% slower at ≥8K); PDL (no wall-clock gain beyond ±4%, +15% at L=128); interleaved / NHD pages (0.83–0.92× FlashInfer at ≥8K vs 1.07–1.25× on HND); the fused last-CTA merge; the wide SW128 fp6 box, ungated or gated at ≥64K; `ctas_per_sm = 3` (TMEM: O alone is 256 of the 512 columns at d = 256).
