# Python Engineer: ring-pacing 4-stage KV — SMOKE (orchestrator probe)

## Plan Reference
Orchestrator task: smoke the uncommitted gated-ring experiment
(4-stage fp6 ring + runtime producer pacing, PACED-scoped) on branch
autoresearch/oct21. Probes only — no commit, no fixes.

## Result: HANG — STOPPED before the A/B triad

`uv run mjolnir verify correctness --quick --match e2m3 --no-save`
→ 2 of 4 cases **hung** (stage harness killed the container at its own
180 s timeout; log = the timeout line only, 53 bytes). Per the task's
stop rule, step 2 (verify ab) was NOT run.

## Case-by-case vs baseline 20261021-081808

| case | baseline | this run (20261021-132850) |
|---|---|---|
| fp8 e2m3 KV p16 varlen | 4.38e-04 / 3.82e-05 ✅ | 4.38e-04 / 3.82e-05 — **identical** |
| fp8 default: e2m3 HND L=8192 | 2.05e-04 / 2.07e-05 ✅ | **HUNG** — TIMEOUT 180s, no numbers |
| bf16 default: e2m3 HND L=8192 | 3.91e-03 / 3.02e-04 ✅ | 3.91e-03 / 3.02e-04 — **identical** |
| fp8 fused merge e2m3 HND varlen | 9.73e-04 / 6.02e-05 ✅ | **HUNG** — TIMEOUT 180s, no numbers |

## Facts pinned down

- `PACED = kv_stages > 2 and kv_format in ("e2m3","e3m2")` (t5_decode.py).
  - Passing bf16 case: `kv_dtype: Float8E4M3FN` → PACED **off**, plain
    4-stage — behaves exactly like baseline.
  - Both hung cases: `kv_dtype: Float6E2M3FN` → PACED **on**.
  - Passing fp8 p16 case: e2m3 KV too (PACED on), nhd layout, num_seqs=4,
    chunk=10, max chunks/seq=3 — so PACED does not hang unconditionally.
- Hung configs (from baseline logs, same shapes): 02 = num_seqs=1,
  chunk=7, max chunks/seq=10, hnd; 04 = num_seqs=2, L=16384 varlen,
  chunk=15, hnd, fused_merge. Common: hnd layout + e2m3 KV.
- Hang is in-kernel (harness label: "hung kernel?"); logs contain no
  driver output at all, so compile-hang vs launch-hang is not
  distinguishable from the logs.
- No leftover containers; GPU untouched (0% util before and after).
  Baseline run 20261021-081808 passed all cases on the pre-change tree.

## Files

- benchmarks/logs/verify-20261021-132850/correctness/02-fp8-default_e2m3_HND_L=8192.log (hang)
- benchmarks/logs/verify-20261021-132850/correctness/04-fp8-fused_merge_e2m3_HND_varlen.log (hang)

## Status

COMPLETE (smoke halted at step 1 per stop rule)
