# Debug Report: ring-pacing-deadlock (digest)
2026-10-21; debugger (subagent spawn, HEAD fd2a0c8, uncommitted pacing experiment in t5_call/t5_decode/t5_harness); bisect containers repro1/repro2, b3a–t1, d7/d8, fix1–fix5 (no ledger run ids).
**Verdict:** root cause found — the producer's plain smem load inside the traced dynamic-while spin is served from a stale register (0 forever) while the same address reads back 4 from the KQ warp; the atomic-RMW re-read fixes every hung shape, and the fix became AGENTS.md's "CuTe-DSL spin trap" rule.
**Key facts:**
- Hang: first launch on every active shape (fp8-Q + e2m3/e3m2 KV, kv_stages > 2) — PS==128 and paged, HND and NHD, cap 2 and 4, merge modes (16-row bisect matrix; kv_stages=2 PASS — machinery spliced out).
- Instrumented dump (run d7): all three warps materialize the same smem address (1236); KQ warp's adds executed (k_adds=4) and its own store→load read-back returned 4; the producer's spin re-read saw 0 on every iteration.
- THE FIX (verbatim, all 4 producer spin sites in t5_decode.py — grep `consumed = ring_consumed[0]`):
  `consumed = cute.arch.atomic_add(ring_consumed, cutlass.Int32(0), sem="relaxed", scope="cta")`
- Fix validation: d8 PASS, fix1–fix3 PASS ×3 (190–210 µs, elected atomic adds restored), fix4 fused-merge varlen PASS 268 µs, fix5 HND p16 varlen PASS 177 µs — counter mechanics were fine; only the read path was broken.
- The GPU probe missed it: `full` variant (plain-load spin + elected adds) passes 8/8 → shape-sensitive mislowering; `spin_compound` SPUN OUT 8/8 (proves cap engagement only, no adder).
- Cap arithmetic exonerated (b3b `--ring_uncapped` still HANGS); elect_one atomic adds correct as-is; debug copy /tmp/opencode/mjolnir-dbg, patch script /tmp/opencode/patch_dbg2.py, logs /tmp/opencode/dbg-*.log.
**Full report:** `git show 56caea5:research/runs/oct21/notes/ring-pacing-deadlock.md`
