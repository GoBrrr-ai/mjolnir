# Mjolnir autoresearch program — Orin (`sm_87`)

You are the orchestrator of an autonomous research team improving the **Mjolnir
decode-attention kernel on the Jetson AGX Orin** — the `arch/orin` mma kernel
(`mma.sync` Ampere tensor cores, hd256, fp8 and fp6-e2m3 KV, head-major
logical views). It is the family's **only** kernel (the GEMV retired to
`gemv-archive`; the runner rejects the old `MJOLNIR_ATTN_ORIN_IMPL` knob), it
serves the Orin configs (`AWQ_MJOLNIR`, `AWQ_MJOLNIR_E2M3` — the default — and
`AWQ_MJOLNIR_E2M3_VT64K`), and it currently **loses to FlashInfer's FA2-TC at
L ≥ 8K**: the second-Orin validation (branch `collab/orin2`, 2026-10-27) measured
the same-run sweep at **1.89× @128, 1.20× @4K, ≈0.71× @16K+ — geomean 0.855×**.
Closing that long-L gap (without losing the short-L win) is the program. After
[karpathy/autoresearch](https://github.com/karpathy/autoresearch): you change
one file, a fixed harness scores the change, the change is kept only if the
score improves, and the loop runs until a human stops it. Every experiment —
kept, discarded or crashed — goes on the ledger.

## The setup

| | |
|---|---|
| **the board** | this host: AGX Orin 64 GB, sm_87 (16 SMs × 128 cores), 204.8 GB/s LPDDR5. A live vLLM server shares the GPU — never stop, restart or kill it, or any container you did not start. The harness waits for a clean window by itself (`--gate`); if `evaluate` waits, let it wait. |
| **the files you change** | the kernel family `src/mjolnir/mjolnir/arch/orin/` — `mma.py` (the kernel), `runner.py`, `plan.py`, `combine.py`, `harness.py`. The fp32 references (`ref_decode`, the harness `check()` path) are the correctness spec: never change what they compute. |
| **the harness (fixed)** | `mjolnir autoresearch evaluate -m "<hypothesis>" --agent <who> --sources "<url>,<arxiv id>"` — the Orin correctness matrix (fp8 **and** e2m3 KV sweeps vs the fp32 reference, ~5 min) + the events-timed L-sweep of FlashInfer FA2-TC vs Mjolnir at L = 128 … 262144. **No ncu exists on L4T** — the events medians are the instrument; only same-run ratios mean anything. Pass `--image vllm/vllm-openai:v0.32.0-ubuntu2404` so the FlashInfer reference stays native (the checkout mounts over it for our legs). |
| **the metric** | per L, *FlashInfer time ÷ Mjolnir time*, both measured in the same run (drift cancels); the **score** is their geomean over L. Higher is better. |
| **the decision (fixed)** | made by the harness, never by you — `judge()`: per L against the best kept row, keep needs a > 2% geomean gain that survives dropping its best L with at most one L slower than the ±3% band; a FlashInfer leg that jumped > 15% sets that L aside (more than two and the run is `noisy`); a structured sub-bar gain is `parked` (tag `ar/parked/<sha>`, stackable, never the reference). |
| **the ledger** | `research/ledger.jsonl` / `ledger.tsv` — untracked during the run. `mjolnir autoresearch ledger` shows it; `export` snapshots to `research/runs/<tag>/`. |

Read before starting: `research/runs/orin1/notes/` (the design notebook — why
the GEMV died, how the mma got built, the stride/root-cause hunts and the
fp6 bring-up; don't repeat a known-negative without a new reason),
`docs/platforms/orin/` (what is measured), and the kernel family itself.

## The baseline

A run re-baselines itself: `mjolnir autoresearch init <tag>` (or
`run --tag <tag>`) measures the starting kernel fresh — that row is the
reference. Never compare to 0.855 by hand (different board: the second Orin
is a 32 GB module). The baseline correctness + sweep need a clean window;
the gate takes minutes when the server is busy — that is expected.

## Constraints

- **Change only the kernel family `arch/orin/`.** Not the references, not the
  harness flags' meaning, not `mjolnir_cli`, not `backend.py`/`builder.py`
  (the vLLM side follows a kept win in a human PR). The sweep's correctness
  gate runs before every timing — a correctness failure is a total outage
  (there is no fallback kernel to serve instead).
- **fp6 e2m3 is the serving format.** Every kept change keeps the e2m3
  matrix green (the evaluate covers it) — an fp8-only win is not a win.
- **The harness pins are design decisions, not typos**: bf16 Q only (the
  fp8-Q pack retired with the GEMV), 2 CTAs/SM (255 regs + the smem bind).
  Changing an assert is a design change — say so in the hypothesis and expect
  the judge to hold the change to the same bar.
- **No new dependencies.** torch, cutlass and the CUDA toolkit in the image
  are what you have.
- **Defaults must stay safe.** A new code path goes behind a harness flag
  first; flip the default only when that experiment keeps.
- **Simplicity criterion**: a 2% win that adds 200 lines of fragile code is
  worth less than an equal score from deleting code. Say so in the
  hypothesis when a change simplifies.

## Where the time goes (start here)

From the committed profiles and the design notes: the mma keeps the tensor
pipes busy (the GEMV's issue-bound death is solved), but the **long-L band
runs ≈0.71× FI** while the short-L win (1.89× @128) rests on launch-shape.
The levers the notes leave open, roughly in order of expected value:

1. **The long-L KV read path** — page-16 pool rows, the split-KV plan's
   split count vs the combine's launch floor, and what the L2 does at
   16 SMs. The harness sweeps page sizes 16–128 in correctness but times
   the serving page only — a kept win may need the plan, not the kernel.
2. **The combine** — `combine.py`'s partials merge runs per step; its cost
   is flat vs L while its benefit grows — measure where the crossover sits.
3. **The fp6 path parity** — fp8 and e2m3 time the same in the matrix, but
   serving runs e2m3 with per-channel scales folded (`q_mul`/`o_mul`); a
   lever found on fp8 must be re-validated on e2m3 (the matrix does that
   for correctness, not for time — add a timing leg behind a flag first).

Anything that wins only at 128/4K and loses 16K+ is a gating candidate, not
a brick — the judge will discard it; spend the effort on the long band.
