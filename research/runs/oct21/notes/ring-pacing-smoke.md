# Python Engineer: ring-pacing 4-stage KV — SMOKE (orchestrator probe) (digest)
2026-10-21; python-engineer; smoke of the uncommitted gated-ring experiment — correctness run 20261021-132850.
**Verdict:** HANG — 2 of 4 correctness cases hung at the harness's own 180 s timeout (the fp8-Q + e2m3-KV HND cases); stopped before the A/B triad per the stop rule.
**Key facts:**
- Hung: 02 fp8 default e2m3 HND L=8192 (chunk=7, max chunks/seq=10) and 04 fp8 fused-merge e2m3 HND varlen (L=16384, chunk=15) — TIMEOUT 180s, no numbers.
- Passing: bf16 default (kv_dtype Float8E4M3FN → PACED off) and fp8 p16 varlen nhd (PACED on, chunk=10, max chunks/seq=3) — both bit-identical to baseline 20261021-081808 → PACED does not hang unconditionally.
- `PACED = kv_stages > 2 and kv_format in ("e2m3","e3m2")` (t5_decode.py); hang is in-kernel ("hung kernel?"), compile- vs launch-hang indistinguishable from the logs.
- Root cause found next session: the producer's plain smem-load spin (see ring-pacing-deadlock.md).
- Logs: benchmarks/logs/verify-20261021-132850/correctness/02-*.log and 04-*.log; GPU 0% util throughout, no leftover containers.
**Full report:** `git show 56caea5:research/runs/oct21/notes/ring-pacing-smoke.md`
