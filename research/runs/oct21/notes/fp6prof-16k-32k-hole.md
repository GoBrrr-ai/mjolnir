# Python Engineer: fp6prof — 16K-32K hole diagnostic (digest)
2026-10-21; python-engineer; fp6prof diagnostic run 20261021-094358 (fp8 vs e2m3 KV on HND pages, L=12288/16384/24576/32768/49152, 10 legs, `--set full` ncu, gated clean window).
**Verdict:** the byte saving does not convert to time in 24K–32K — e2m3 moves 0.75× sectors with L2% 3.2–6.7 pp lower yet time only ≤13% better, and 32K shows no win at all (t ratio 1.01) with 64-B L2 requests confirmed (65–66 B/request vs fp8's 129–131).
**Key facts:**
- Per-L t ratios (e2m3/fp8): 12288 → 0.93, 16384 → 0.86, 24576 → 0.93, 32768 → **1.01** (no win), 49152 → 0.87; geomean 0.918.
- e2m3 L2% drops 3.2–6.7 pp vs fp8 at every L — bytes saved ≠ time saved in the hole band.
- L2 bytes/request: fp8 ≈ 129–131 (full 128-B lines), e2m3 ≈ 65–66 (64-B requests) — the request-geometry evidence.
- Effective read bandwidth: fp8 194–211 GB/s vs e2m3 156–180 GB/s — e2m3 never reaches fp8's absolute BW even where it wins.
- long_scoreboard dominates everywhere; at 32K e2m3 stalls MORE than fp8 (24.4 vs 19.6) despite 25% fewer bytes.
- Hit rate ≈ 0.4–1.8% (pure streaming); `gpu__compute_memory_throughput` == `lts__throughput` (L2 is the bottleneck level).
- Raw: benchmarks/raw/verify/fp6prof/20261021-094358.json; logs benchmarks/logs/verify-20261021-094358/.
**Full report:** `git show 56caea5:research/runs/oct21/notes/fp6prof-16k-32k-hole.md`
