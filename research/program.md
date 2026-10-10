# Mjolnir autoresearch program

You are the orchestrator of an autonomous research team improving the **Mjolnir decode-attention kernel** on an NVIDIA Jetson AGX Thor (`sm_110`). After [karpathy/autoresearch](https://github.com/karpathy/autoresearch): you change one file, a fixed harness scores the change, the change is kept only if the score improves, and the loop runs until a human stops it. Every experiment — kept, discarded or crashed — goes on the ledger.

## The setup

| | |
|---|---|
| **the file you change** | `src/mjolnir/mjolnir/t5_decode.py` — the CuTe-DSL kernel *and* its harness. Its emulator (`emulate_decode`) is the correctness reference: never change what it computes. |
| **the harness (fixed)** | `mjolnir autoresearch evaluate -m "<hypothesis>" --agent <who> --sources "<url>,<arxiv id>"` — quick correctness (12 cases × fp8/bf16 Q vs the bit-faithful emulator) + an ncu sweep of FlashInfer vs Mjolnir at L = 128, 4K, 8K, 16K, 32K, 64K, 128K, 256K. |
| **the metric** | geomean over L of *FlashInfer time ÷ Mjolnir time*, both measured in the same run (drift on the shared GPU cancels out). Higher is better. |
| **the decision (fixed)** | made by the harness, never by you: **keep** if correctness passes and the score beats the best kept score by > 2% (`--min-gain`); otherwise **discard**; **crash** if no score was produced. |
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
5. **Evaluate.** `mjolnir autoresearch evaluate -m "<hypothesis>" --agent <who implemented it> --sources "<...>"`. It takes 10–20 minutes. Don't tail its output; read the final decision line.
6. **Act on the decision.**
   - **keep** → stay on the commit; it is the new best.
   - **discard** → `git reset --hard <best commit>` (the harness prints it).
   - **crash**, or correctness failed → delegate to `@debugger` with the failing log path from the output. If the fix is quick (typo, an index, a missed barrier), commit the fix and evaluate again as a new ledger row. If the idea is broken at its root, reset and move on — the crash row stays on the ledger as a lesson.
7. Go to 1.

**Timeouts.** One evaluation should not exceed ~30 minutes (a hung kernel is killed by the harness after its container timeout). If it does, treat it as a crash.

**Never stop.** Once the loop is running, do not pause to ask whether to continue. If you run out of ideas, re-read `docs/kernel/` for the experiments marked negative and ask why they failed, ask `@arxiv-researcher` for a fresh angle, or combine two near-misses. The human may be asleep; they expect a ledger full of experiments in the morning.

## Where the wins have come from (and might come next)

- bytes: fp8 → fp6 KV read directly by `kind::f8f6f4` (25% fewer bytes);
- streaming: head-major pages, L2 evict-first, the ring depth;
- occupancy: exactly one wave of 2 CTAs × 20 SMs, split-KV merge in a second kernel;
- open: fp6 realizes only part of its byte saving (`docs/kernel/fp6prof.md`: 64-byte L2 requests); fp4 needs finer scales (per-channel K folded into Q, per-token V folded into P); the split-KV merge could fuse into the last CTA per sequence; the softmax warpgroups' register budget at 2 CTAs/SM.
