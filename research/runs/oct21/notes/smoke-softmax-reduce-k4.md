# Python Engineer: smoke — softmax_reduce_every (K=4 default)

## Plan Reference

Orchestrator smoke request (branch autoresearch/oct21, HEAD fd2a0c8 + uncommitted
softmax_reduce_every experiment in the working tree). Read-only smoke: NO commits,
NO source edits.

## Checks

1. `verify correctness --quick --match e2m3 --no-save` — 4/4 PASS expected; compare
   max/mean per case vs baseline 20261021-081808; STOP if any FAIL or >3x baseline.
2. Ref checks vs emulator (one-shot containers, --entrypoint python3,
   PYTHONPATH=/mnt/mjolnir-src, timeout 600, kill by name):
   K=4 @ 8192, K=4 @ 65536, K=8 @ 65536.
3. Timing K-sweep: `verify ab --variants "k1:…|k2:…|k4:|k8:…" --ls "8192 65536" --no-save`.

## GPU state at start

No vLLM server (gate UNREACHABLE, 0% util) — clean window; verify commands gate
themselves.

## Result

**STOP — all 4 correctness cases FAIL at trace (compile) time.** No numbers
produced anywhere; baseline comparison moot.

- Run: `verify correctness --quick --match e2m3 --no-save` → 4 failed, 0 passed,
  every case the identical error, 4-5s each (never reached the GPU):
  `error: NameError in '__call__': name 'c_f32x2' is not defined`
  → `t5_decode.py:1338` (`correction[i] = c_f32x2[0]`).
- Root cause: the correction block first-defines `c_f32x2` INSIDE the branches of
  the staged `if do_reduce:` / `else:` and reads it after. The CuTe DSL requires
  variables used around staged control flow to be defined BEFORE the region (its
  own suggestion says exactly this — same rule the pre-existing
  `tSrM_lane = cutlass.Float32(0)  # Avoid dynamic register indexing` honours).
  `do_reduce` is dynamic (`s` is a `cutlass.range` index, `iters_s` runtime), so
  the region is staged for every K — the K=1 diagnostic probe (ad-hoc container,
  existing `--softmax_reduce_every 1` CLI flag, no edits) failed with the
  IDENTICAL error at the identical line, so no K value compiles as written.
- Fix direction (orchestrator's call, not applied): define
  `c_f32x2 = (cutlass.Float32(1.0), cutlass.Float32(1.0))` before the
  `if do_reduce:` (line ~1332), keep the branch rebinds. `tSrM_lane` is fine
  (defined at region top before its reads). The rest of the diff (tSrM hoist,
  gate shape, harness plumbing) is unvalidated until this compiles.

## Not run (blocked)

- Ref checks K=4 @ 8192/65536, K=8 @ 65536: same trace failure guaranteed
  (K-independent compile error); not burned.
- `verify ab` K-sweep: every leg (k1 included) dies at trace; not burned.

## Run ids

- `20261021-184225` — correctness attempt (no JSON, `--no-save`; logs under
  `benchmarks/logs/verify-20261021-184225/correctness/`).
- K=1 probe: ad-hoc one-shot container `smoke-refk1-8k` (no run id; log
  `/tmp/opencode/smoke-refk1-8k.log`).

## Status

COMPLETE — stopped per the fail-fast rule (any FAIL → STOP and report).
