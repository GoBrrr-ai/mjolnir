# Python Engineer: smoke — softmax_reduce_every (K=4 default) (digest)
2026-10-21; python-engineer; smoke of the uncommitted softmax_reduce_every experiment — correctness attempt run 20261021-184225 + K=1 probe container smoke-refk1-8k.
**Verdict:** STOP — all 4 correctness cases FAIL at trace (compile) time with the identical error; no GPU numbers produced, ref checks + K-sweep not burned.
**Key facts:**
- Error everywhere: `error: NameError in '__call__': name 'c_f32x2' is not defined` at t5_decode.py:1338 (`correction[i] = c_f32x2[0]`).
- Root cause: `c_f32x2` first-defined INSIDE the staged `if do_reduce:` / `else:` branches and read after — the CuTe DSL requires variables used around staged control flow to be defined BEFORE the region (same rule the pre-existing `tSrM_lane = cutlass.Float32(0)` honours).
- `do_reduce` is dynamic (`s` is a cutlass.range index, iters_s runtime) → staged for every K; the K=1 diagnostic probe failed identically at the identical line → no K value compiles as written.
- Fix direction (orchestrator's call, not applied): define `c_f32x2 = (cutlass.Float32(1.0), cutlass.Float32(1.0))` before the `if do_reduce:` (~line 1332); `tSrM_lane` is fine.
- Correctness logs: benchmarks/logs/verify-20261021-184225/correctness/.
**Full report:** `git show 56caea5:research/runs/oct21/notes/smoke-softmax-reduce-k4.md`
