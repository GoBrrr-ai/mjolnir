# Python Engineer: ring-pacing Phase 1 (4-stage fp6 ring + runtime producer pacing)

## Plan Reference

autoresearch step — Phase 1 spec (ring pacing), inherits Phase 0 probe
(`/tmp/opencode/ring_probe.py`, GPU-validated) after wiping a previous agent's
scratch implementation (spin-bound `>= 1000000` marker) from the worktree.

## Files In Scope

- src/mjolnir/mjolnir/t5_call.py (derivation lives here: `_default_kv_stages`)
- src/mjolnir/mjolnir/t5_decode.py (counter, pacing, consumer adds, DEEP_ITERS)
- src/mjolnir/mjolnir/t5_harness.py (--ring_uncapped flag)

## Approach

Flip the fp6/2-CTA/d=256 default ring to 4 stages; add a smem `ring_consumed`
counter: elected atomic_add next to each `k_handle`/`v_handle.release()` in the
MMA KQ/VP warps, producer spin before each KV acquire in the TMA warp, capped
at 2 stages in flight for chunks > DEEP_ITERS (13 tiles ≈ the 16K bench point's
chunk), full ring for short chunks or `--ring_uncapped`.

NOTE on the spec formula: `depth = kv_stages if (uncapped or iters_s > DEEP_ITERS) else 4`
is degenerate for the only config that traces (kv_stages == 4 == else-branch →
pacing never engages). The data (deep wins 8K-16K, shallow >= 32K), the probe's
validated `spin_compound` idiom and DEEP_ITERS = "the 16K-equivalent chunk" all
give the inverse: cap = kv_stages when (uncapped or iters_s <= DEEP_ITERS), else 2.
Implemented as the probe-validated compound predicate (runtime if/else depth
selection can't bind-escape staged regions — t5_call.py:242).

## SOLID Analysis

- S: pacing is one mechanism in the kernel's producer warp; derivation stays in
  `_default_kv_stages` (t5_call.py); harness flag threading unchanged shape.
- D: `ring_uncapped` is a constructor knob; runner.py/backend.py untouched
  (default False = paced default).

## Status

COMPLETE

## Files Modified

- src/mjolnir/mjolnir/t5_call.py — `_default_kv_stages` fp6/2-CTA/d=256 → 4 stages
- src/mjolnir/mjolnir/t5_decode.py — DEEP_ITERS=13, ring_uncapped param, smem
  counter + warp-0 zero, producer pacing (PS==blk_tile_s + paged paths),
  elected consumer adds at both release sites
- src/mjolnir/mjolnir/t5_harness.py — --ring_uncapped flag threaded through run()

## Verification

- py_compile: OK; ruff 0.14.14 (config): PASS; check_file_length --base origin/main: EXIT 0
- Image import check (no GPU): DEEP_ITERS=13, ring_uncapped last param default False
- SIM102 noqa on the zeroing if is load-bearing (verified via /tmp copy)

## Notes for Reviewer

Spec's inline depth formula was degenerate (always kv_stages for the only
tracing config); implemented the probe-validated inversion — cap = kv_stages
when (uncapped or chunk <= 13 tiles), else 2 — see note in the progress file.
