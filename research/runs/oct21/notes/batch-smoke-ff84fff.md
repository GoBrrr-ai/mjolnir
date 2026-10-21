# Python Engineer: batch smoke (ff84fff)

## Plan Reference

Orchestrator smoke request (branch autoresearch/oct21, HEAD ff84fff):
correctness quick vs baseline 20261021-081808, wide-path ref check (harness, no --skip_ref_check, 64K + 8K control), wide-vs-narrow triad (verify ab).

## Files In Scope

- none (read-only smoke; no source edits, no commits)

## Results

- correctness --quick --match e2m3: 4/4 PASS, errors bit-identical to baseline (run 20261021-170320).
- Ref checks (one-shot containers, PYTHONPATH overlay so the checkout is imported):
  64K PASS (|kern-emu| 6.102e-05/7.215e-06), 8K PASS (2.052e-04/2.069e-05, == case 2).
- Triad (20261021-170638): noWide == pre-batch sweeps (605.0/2057.1 vs 593.7-603.8 / 2057.6-2087.8);
  wide NOT faster: 64K +2-3%, 256K −2.4% ncu / −3.5% wall.
- Engagement probe (ncu, matched metrics lts__t_{sectors,requests}_srcunit_tex_op_read.sum):
  wide vs noWide byte-identical request streams at 256K (2.01 sectors/req = 64-B);
  forced-wide (`--fp6-wide-tma-min-l 0`) at 8192 == narrow (197,860 vs 197,876 requests).
  → the wide ring is inert in this build: no request-width change, no timing change.
  Kernel names identical between legs (zero added smem — second view, consistent).
  fp6prof 20261021-094358 confirms the 8192 narrow request count (197,824).

## Verdict

Correctness ✅ · ref checks ✅ · SW128 effect ❌ absent (wide inert; triad deltas = same-run drift).
4-stage ring + pacing: no regression (noWide ≈ pre-batch at every L).

## Status

COMPLETE
