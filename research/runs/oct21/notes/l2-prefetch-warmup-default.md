# Python Engineer: warm-up-only L2 prefetch default — smoke review

## Plan Reference

Orchestrator request (autoresearch/oct21 worktree): review the working-tree
experiment that makes warm-up-only L2 prefetch the kernel default, run sanity
gates + GPU smokes A (correctness) and B (p0 vs p2 A/B).

## Files In Scope

- src/mjolnir/mjolnir/t5_decode.py
- src/mjolnir/mjolnir/t5_harness.py

## Approach

Verbatim `git diff` review, then three gates (py_compile, ruff 0.14.14,
check_file_length --base origin/main), then two self-gated GPU smokes
(`verify correctness --quick --match e2m3 --no-save` and
`verify ab --variants "p0:--l2_prefetch 0|p2:--l2_prefetch 2" --ls "4096 262144" --no-save`).
No fixes, no commits.

## Diff summary (full verbatim in session transcript)

- `t5_decode.py`:
  - `l2_prefetch` default 0 → 2 (warm-up only; steady-state variant retired,
    lost 7-27% at >=8K).
  - nvfp4 path: assert replaced by forcing `self.l2_prefetch = 0` (packed path
    has no prefetch block) so the warm-up default can't break nvfp4.
  - PS==128 path: steady-state prefetch arms removed; only `s == 0` (chunk's
    first iteration) prefetches the next L2P tiles.
  - PS!=128 path: `do_p = s >= 0` special arm removed; warm-up only.
- `t5_harness.py`: `l2_prefetch` default 0 → 2 in `run()` and the CLI
  `--l2_prefetch` argparse default, help text updated.

## Status

COMPLETE

## Gate / smoke results

- py_compile: COMPILE-OK
- ruff 0.14.14: All checks passed!
- check_file_length --base origin/main: exit 0
- Smoke A (correctness, e2m3): ✅ all 4 checks passed
  (fp8 varlen 4.38e-04 max / bf16 default 3.91e-03 max / fused merge 9.73e-04 max)
- Smoke B (p0 vs p2, L=4096/262144): both legs compiled and timed.
  - wall: p0 51.6 µs / 2077.4 µs; p2 51.4 µs / 2087.6 µs — p2 at parity with
    p0 (within the ±3% noise band), 1.18× / 1.33-1.34× vs FlashInfer.
  - ncu ab rows: p0 77.7 / 1932.1 µs; p2 77.3 / 1939.2 µs (~0.4% apart).
- Log dirs: benchmarks/logs/verify-20261021-083931 (A),
  benchmarks/logs/verify-20261021-084014 (B). Nothing committed.
