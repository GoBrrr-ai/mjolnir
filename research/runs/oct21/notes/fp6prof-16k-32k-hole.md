# Python Engineer: fp6prof — 16K-32K hole diagnostic

## Plan Reference

Subagent diagnostic task (read-only): profile fp8 vs e2m3 KV on HND pages at
L = 12288/16384/24576/32768/49152. No ledger row, no commits, no source edits.

## Files In Scope

- none (diagnostic run only; output in `benchmarks/logs/verify-20261021-094358/`
  and `benchmarks/raw/verify/fp6prof/20261021-094358.json`)

## Approach

Ran the exact orchestrator command (dry-run validated first: 10 legs, variant
flags parsed correctly, `--set full` ncu, gate on). EXIT_CODE=0, all 10
profiles ok, gate offline (clean window). Extracted metrics from the raw JSON
(console table truncates stall names and omits L2/DRAM %s).

## Key numbers (e2m3/fp8)

| L | t ratio | fp8 L2% | e2m3 L2% | ΔL2% |
|---|---------|---------|----------|------|
| 12,288 | 0.93 | 23.5 | 19.1 | −4.4 |
| 16,384 | 0.86 | 25.0 | 21.8 | −3.2 |
| 24,576 | 0.93 | 23.5 | 18.9 | −4.6 |
| 32,768 | **1.01** | 26.0 | 19.3 | −6.7 |
| 49,152 | 0.87 | 25.3 | 21.9 | −3.4 |

- Geomean t ratio 0.918. The hole is sharpest at **32,768 (1.01 — no win)**;
  24,576 (0.93) is also below the 0.86–0.87 neighbors.
- e2m3 moves 0.75× the sectors but its L2% drops 3.2–6.7 pp while time drops
  ≤13% — the byte saving is not converting to time in 24K–32K.
- long_scoreboard dominates everywhere; at 32K e2m3 stalls *more* than fp8
  (24.4 vs 19.6) despite 25% fewer bytes.
- L2 B/request: fp8 ≈ 129–131 (full 128B lines), e2m3 ≈ 65–66 (64B requests).
- Effective read bandwidth: fp8 194–211 GB/s vs e2m3 156–180 GB/s — e2m3 never
  reaches fp8's absolute bandwidth even at the Ls where it wins.
- Hit rate ≈ 0.4–1.8% (pure streaming); `gpu__compute_memory_throughput` ==
  `lts__throughput` (L2 is the bottleneck level).

## Status

COMPLETE — diagnostic only, nothing committed.
