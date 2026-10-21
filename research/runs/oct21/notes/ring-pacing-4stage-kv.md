# Python Engineer: ring-pacing 4-stage KV (fp6 harness config)

## Plan Reference
Autoresearch experiment — 4-stage KV ring + runtime producer pacing
(research/program.md idea 2); orchestrator task spec.

## Files In Scope
- src/mjolnir/mjolnir/t5_decode.py (kernel: counter, pacing, increments, param)
- src/mjolnir/mjolnir/t5_call.py (derivation 2 -> 4 for fp8-Q e2m3/e3m2 2-CTA d=256)
- src/mjolnir/mjolnir/t5_harness.py (--ring_uncapped flag)

## Phase 0 (all verified against the IMAGE's DSL 4.8.0 in narandill/mjolnir-vllm:v0.32.0)
1. smem atomic: cute.arch.atomic_add(ptr, val, *, sem, scope) — works on the
   smem _Pointer from SmemAllocator.allocate_array; proven on GPU (exact counts).
   FORBIDDEN shape: binding the result (`old = atomic_add(...)`) inside
   elect_one inside cutlass.range fails IR verify ("does not dominate").
2. while tracing: dynamic `while` with compound traced conditions + loop-carried
   Int32 re-read compiles and runs correctly (probe spin_simple/spin_compound).
3. release election: consumer release (TCGen05Mma op) = arrive_tcgen05mma in
   `with cute.arch.elect_one():` — same guard used for the increments.
4. derivation: t5_call.py _default_kv_stages returns 2 for e2m3/e3m2+2CTA+spt4;
   4-stage smem = 77888 B < 115736 B budget.

## GPU probe (ring_probe.py, /tmp/opencode)
full (producer 2/tile + 2 elected adds/tile, spin cap): PASS at iters 2/5/40,
capped + uncapped, exact produced==counter==2*iters; cap engages (spins>0);
2-stage config traces no counter code. Deadlocks in early probe runs were
probe bugs (unguarded producer warp; result binding).

## SPEC DEVIATION (flagged)
Task formula `depth = kv_stages if (ring_uncapped or iters_s > 13) else 4`
is a constant-4 no-op at kv_stages=4 (default leg == --ring_uncapped leg).
Implemented the coherent reading consistent with the task summary line,
research/program.md idea 2 and the 20261012-210506 sweep:
deep (kv_stages) for iters_s <= 13, 2-stage behavior (2) above,
ring_uncapped -> always kv_stages — expressed as the predicate
`produced - consumed >= 2 and (not deep or produced - consumed >= kv_stages)`.

## Status
IN PROGRESS
