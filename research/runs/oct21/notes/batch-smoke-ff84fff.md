# Python Engineer: batch smoke (ff84fff) (digest)
2026-10-21; python-engineer; smoke of batch commit ff84fff (brick + wide-TMA port) — correctness run 20261021-170320, wide-path ref checks (64K + 8K control), wide-vs-narrow triad 20261021-170638 + ncu engagement probe.
**Verdict:** correctness ✅ and ref checks ✅ but the SW128 effect is ❌ absent — the wide ring is inert in this build (byte-identical request streams, triad deltas = same-run drift); the 4-stage ring + pacing shows no regression, and the inertness finding spawned wide-sw128-ring-inert.md.
**Key facts:**
- `verify correctness --quick --match e2m3`: 4/4 PASS, errors bit-identical to baseline 20261021-081808 (run 20261021-170320).
- Ref checks (one-shot containers, PYTHONPATH overlay): 64K PASS |kern-emu| 6.102e-05 / 7.215e-06; 8K PASS 2.052e-04 / 2.069e-05 (== case 2).
- Triad 20261021-170638: noWide == pre-batch sweeps (605.0/2057.1 vs 593.7–603.8 / 2057.6–2087.8); wide NOT faster — 64K +2–3%, 256K −2.4% ncu / −3.5% wall.
- Engagement probe (lts__t_{sectors,requests}_srcunit_tex_op_read.sum): wide vs noWide byte-identical request streams at 256K (2.01 sectors/req = 64-B); forced-wide (`--fp6-wide-tma-min-l 0`) at 8192 == narrow (197,860 vs 197,876 requests); kernel names identical between legs (zero added smem — second view).
- fp6prof 20261021-094358 confirms the 8192 narrow request count (197,824).
**Full report:** `git show 56caea5:research/runs/oct21/notes/batch-smoke-ff84fff.md`
