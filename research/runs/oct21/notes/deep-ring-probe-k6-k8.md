# Deep-ring probe: 6- and 8-stage KV rings at short chunks (digest)
2026-10-22; python-engineer; probe run 20261021-215735 (`verify ab --no-save`; ref-check containers k6-8k / k8-8k / k8-4k; no ledger row — probe without a commit).
**Verdict:** FAIL vs the pre-registered rule (k6/k8 ≥ 3% over k4 at 8192 with both instruments agreeing; no ±3% regression at 4096/16384) — deeper rings win ncu-only at 8K but the wall inverts; the depth lever is fully swept: 4 stages is wall-optimal, the brick's shipped depth stands.
**Key facts:**
- Hypothesis: the parked brick (`ar/parked/2f20c4f`) proved 4 stages (2 tiles in flight) beat 2 by ~13% ncu at 8K; 6/8 stages (3–4 tiles in flight) fit smem at ght=16 (~11 KB/stage + ~20 KB fixed ≤ 115.7 KB) and were never measured; zero-commit probe via `--kv_stages` (bypasses the derive, `round_to_tile=False`).
- Correctness: ref checks PASS at k6@8192, k8@8192, k8@4096 — bit-identical to the default derive (max 2.052e-04 / mean 2.069e-05 at 8K; 2.356e-04 / 2.887e-05 at 4K); |lse−emu| 9.537e-07 everywhere; ring depth buffers loads only, math order untouched.
- FlashInfer fp8 NHD reference: 60.6 / 96.7 / 188.8 µs at 4K/8K/16K.
- ncu (4K/8K/16K): k2 50.5/89.4/175.4 · k4 50.6/87.0/172.2 · k6 50.5/84.1/190.2 · k8 50.1/83.2/169.1.
- Wall (median µs): k2 85.9/114.5/180.0 · k4 83.7/112.5/187.2 · k6 83.1/114.5/187.5 · k8 84.3/122.7/208.8.
- 8K: ncu agrees (k6 −3.3%, k8 −4.4%) but wall inverts (k6 +1.8%, k8 +9.1% — k8's 8K wall slower than even k2). 16K: k6 +10.5% ncu, k8 +11.5% wall (208.8 µs; +16% vs k2). 4K parity (chunk = 4 fits any ring).
- Mechanism: deeper rings add prologue/launch overhead (barrier init, first-fill) that ncu replay amortizes but a cold wall launch pays in full — the same wall-vs-ncu trap as the SW128 episode.
- Aside: k6@16K ncu 190.2 = 0.99× FlashInfer, the only sub-parity cell — the artifact-prone bimodal cell, treat as noise.
**Full report:** `git show 9bcd25b:research/runs/oct21/notes/deep-ring-probe-k6-k8.md`
