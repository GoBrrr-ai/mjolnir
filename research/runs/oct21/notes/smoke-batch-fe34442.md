# Python Engineer: smoke — batch fe34442 (brick + split coarsening)

## Plan Reference

Orchestrator smoke request (branch autoresearch/oct21, HEAD fe34442 — the batch:
brick + split coarsening, default cap 8 for T ≥ 128 tiles). Read-only smoke:
NO commits, NO source edits.

## Checks

1. `verify correctness --quick --match e2m3 --no-save` — 4/4 PASS; the two
   varlen cases (p16 varlen T≈157, fused merge varlen) may shift within class
   vs baseline 20261021-081808; the two L=8192 cases (T=64 < 128) must be
   bit-identical. STOP if any FAIL or >3x growth.
2. Ref checks vs emulator (one-shot containers, --entrypoint python3,
   PYTHONPATH=/mnt/mjolnir-src, timeout 600, kill by name, no --skip_ref_check):
   --seqlen 16384 and 262144, --q_dtype Float8E4M3FN --kv_format e2m3
   --kv_layout hnd, default cap 8; grep each log for the plan: line
   (chunk=16@16K, 256@256K).
3. Timing triad: `verify ab --variants "cap16:--long_split_cap 16|cap8:" --ls
   "8192 16384 32768 65536 262144" --no-save`.

## GPU state at start

Gate UNREACHABLE (no vLLM metrics endpoint), nvidia-smi 0% util, no compute
apps — clean window; verify commands gate themselves.

## Results (so far)

### 1. Correctness — 20261021-195438 (`--quick --match e2m3 --no-save`)

4/4 PASS (check strings are `:.4g`):

| case | run 20261021-195438 | baseline 20261021-081808 | delta | plan |
|---|---|---|---|---|
| e2m3 KV p16 varlen (fp8) | max 4.38e-04 / mean 3.821e-05 | 4.38e-04 / 3.821e-05 | identical | chunk=10, max chunks/seq=3 — **unchanged** (cap not binding: T≈157 total tiles over 4 varlen seqs, unconstrained plan already ≤ 8 chunks/seq) |
| default e2m3 HND L=8192 (fp8) | max 2.052e-04 / mean 2.069e-05 | 2.052e-04 / 2.069e-05 | bit-identical | chunk=7 (T=64 < 128, no coarsening) |
| default e2m3 HND L=8192 (bf16) | max 3.906e-03 / mean 3.019e-04 | 3.906e-03 / 3.019e-04 | bit-identical | chunk=7 |
| fused merge e2m3 HND varlen (fp8) | max 9.735e-04 / mean 6.020e-05 | 9.735e-04 / 6.016e-05 | max identical; mean +0.07% (within class) | chunk 15→**16** tiles, max chunks/seq 9→**8** — coarsening active |

No FAIL, nothing >3x.

### 2. Ref checks vs emulator (one-shot containers, checkout at /mnt/mjolnir-src)

- smoke-ref-16k (rc=0, PASS): chunk=16 tiles, work=8, slots/wave=10, max
  chunks/seq=8 ✓ · |kern-emu| max=0.0001219 mean=1.465e-05
- smoke-ref-256k (rc=0, PASS): chunk=256 tiles, work=8, slots/wave=10, max
  chunks/seq=8 ✓ · |kern-emu| max=3.041e-05 mean=3.524e-06

### 3. Timing triad — 20261021-200107 (`verify ab`, --no-save)

Plans (per-leg logs, benchmarks/logs/verify-20261021-200107/ab/wall/):

| L | cap16 | cap8 |
|---|---|---|
| 8192 | chunk=7, work=10 (T=64<128, no coarsening) | chunk=7, work=10 (identical) |
| 16384 | chunk=13, work=10 | chunk=**16**, work=**8** |
| 32768 | chunk=26, work=10 | chunk=**32**, work=**8** |
| 65536 | chunk=52, work=10 | chunk=**64**, work=**8** |
| 262144 | chunk=205, work=10 | chunk=**256**, work=**8** |

Same-run cap8/cap16 time ratios (ncu 81.7/171.3/330.4/585.9/2036.9 vs
84.0/183.6/322.0/597.3/2077.1; wall 114.3/171.9/294.7/519.7/1983.5 vs
121.7/167.8/288.6/536.4/1989.4):

| L | ncu cap8/cap16 | wall cap8/cap16 |
|---|---|---|
| 8192 | 1.028 (identical plans → pure drift) | 1.065 (identical plans → pure drift) |
| 16384 | 1.072 | 0.976 |
| 32768 | 0.975 | 0.979 |
| 65536 | 1.019 | 1.032 |
| 262144 | 1.020 | 1.003 |

No big cap8 loss: 32K wins both instruments; 16K ncu +7.2% but wall −2.4%
(16K-32K ncu replay flips ±25% per fp6prof history); 64K/256K ≈ noise. The
8192 identical-plan legs calibrate this run's leg drift at +2.8% ncu / +6.5%
wall.

## Status

COMPLETE — correctness ✅ (4/4, within class) · ref checks ✅ (chunk 16/256,
8 chunks/seq) · triad ✅ (no kill-criterion loss; cap8 ≈ cap16, wall wins at
16K/32K)

## Run ids

- 20261021-195438 — correctness (--no-save; logs under benchmarks/logs/verify-20261021-195438/)
- smoke-ref-16k / smoke-ref-256k — ref checks (ad-hoc containers, no run id; /tmp/opencode/ref-*.log)
- 20261021-200107 — verify ab triad (--no-save; logs under benchmarks/logs/verify-20261021-200107/)
