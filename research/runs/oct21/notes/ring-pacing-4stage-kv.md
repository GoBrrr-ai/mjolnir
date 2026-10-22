# Python Engineer: ring-pacing 4-stage KV (fp6 harness config) (digest)
2026-10-21; python-engineer; Phase-0 GPU probes (/tmp/opencode/ring_probe.py, DSL 4.8.0 in narandill/mjolnir-vllm:v0.32.0) + the spec for the 4-stage fp6 ring + runtime producer pacing (became the parked brick `ar/parked/2f20c4f`).
**Verdict:** Phase-0 mechanisms GPU-validated and the spec's depth formula shown degenerate (constant-4 no-op) — implemented instead as the probe-validated inverted predicate that became the pacing brick.
**Key facts:**
- smem `atomic_add(ptr, val, *, sem, scope)` works on SmemAllocator `_Pointer`s (exact GPU counts); FORBIDDEN shape: binding the result (`old = atomic_add(...)`) inside elect_one inside cutlass.range fails IR verify ("does not dominate").
- Dynamic `while` with compound traced conditions + loop-carried Int32 re-read compiles and runs (probe spin_simple/spin_compound).
- Consumer release election (TCGen05Mma op) = `arrive_tcgen05mma` in `with cute.arch.elect_one():`.
- Derivation: t5_call.py `_default_kv_stages` → 4 for e2m3/e3m2 + 2-CTA + spt4; 4-stage smem 77,888 B < 115,736 B budget.
- Probe: full variant (producer 2/tile + 2 elected adds/tile) PASS at iters 2/5/40, capped + uncapped, exact produced==counter==2*iters; 2-stage config traces no counter code.
- Spec deviation (flagged): task formula `depth = kv_stages if (ring_uncapped or iters_s > 13) else 4` is degenerate at kv_stages=4; implemented `produced - consumed >= 2 and (not deep or produced - consumed >= kv_stages)` — deep ≤ 13-tile chunks, 2-stage behavior above, `--ring_uncapped` → always full.
**Full report:** `git show 56caea5:research/runs/oct21/notes/ring-pacing-4stage-kv.md`
