# Deep-ring probe: 6- and 8-stage KV rings at short chunks

Probe write-up, 2026-10-22. Run id `20261021-215735` (`verify ab --no-save`);
ref-check containers `k6-8k` / `k8-8k` / `k8-4k`. No ledger row (probe without
a commit).

## Hypothesis

The KV ring depth at short chunks was under-explored. The parked brick
(`ar/parked/2f20c4f`) proved 4 stages (2 tiles in flight) beat 2 by ~13% ncu
at 8K; 6 and 8 stages (3–4 tiles in flight) fit smem at ght=16
(~11 KB/stage + ~20 KB fixed ≤ 115.7 KB) and were never measured. Zero-commit
probe via `--kv_stages` (bypasses the derive, `round_to_tile=False`).

## Correctness

Ref checks PASS at k6@8192, k8@8192, k8@4096 — bit-identical to the default
derive (max 2.052e-04 / mean 2.069e-05 at 8K; 2.356e-04 / 2.887e-05 at 4K);
|lse−emu| 9.537e-07 everywhere. Ring depth buffers loads only, math order
untouched.

## Timing

`verify ab` run `20261021-215735`, `--no-save`; FlashInfer fp8 NHD reference
60.6 / 96.7 / 188.8 µs at 4K/8K/16K.

ncu (µs, 4K/8K/16K):

| config | 4K | 8K | 16K |
|--------|-----:|-----:|-----:|
| k2 | 50.5 | 89.4 | 175.4 |
| k4 | 50.6 | 87.0 | 172.2 |
| k6 | 50.5 | 84.1 | 190.2 |
| k8 | 50.1 | 83.2 | 169.1 |

Wall (median µs, 4K/8K/16K):

| config | 4K | 8K | 16K |
|--------|-----:|-----:|-----:|
| k2 | 85.9 | 114.5 | 180.0 |
| k4 | 83.7 | 112.5 | 187.2 |
| k6 | 83.1 | 114.5 | 187.5 |
| k8 | 84.3 | 122.7 | 208.8 |

## Verdict

Pre-registered rule: k6/k8 ≥ 3% over k4 at 8192 with both instruments
agreeing; no ±3% regression at 4096/16384. **FAIL.**

- 8K: ncu agrees (k6 −3.3%, k8 −4.4%) but wall inverts (k6 +1.8%, k8 +9.1% —
  k8's 8K wall is slower than even k2).
- 16K: k6 +10.5% ncu, k8 +11.5% wall (208.8 µs; +16% vs k2).
- 4K parity (chunk = 4 fits any ring).

## Mechanism

Deeper rings add prologue/launch overhead (barrier init, first-fill) that ncu
replay amortizes but a cold wall launch pays in full — the same wall-vs-ncu
trap as the SW128 episode. The depth lever is now fully swept: 4 stages is
wall-optimal; the brick's shipped depth stands.

(Aside: k6@16K ncu 190.2 = 0.99× FlashInfer, the only sub-parity cell — the
artifact-prone bimodal cell, treat as noise.)
