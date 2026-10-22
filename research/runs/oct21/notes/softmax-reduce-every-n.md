# Python Engineer: softmax-reduce-every-n (round 2 — revert of the correction branch) (digest)
2026-10-21; python-engineer; round-2 softmax_reduce_every experiment — correctness run 20261021-185149, ref checks (staged copy /tmp/opencode/smoke-reduce/src), K-sweep run 20261021-185837.
**Verdict:** experiment DEAD — timing flat (K-sweep k1/k2/k4/k8 within noise at 8192 and 65536, ncu AND wall: the max reduction is not on the critical path) AND e4m3-P has zero headroom above a frozen reference (t5_const.py:34-37: `FP8_P_SCALE = 448.0`, invariant "p in [0,1]", p=1 maps exactly to the e4m3 max → stale-reference P saturates/clips), producing a real accuracy regression.
**Key facts:**
- Round-2 change: reverted the correction block to the fd2a0c8 form (byte-identical, pattern-extract diff) — on non-reduce tiles tSrM == tSrM_prev so the original correction formula self-produces exp2(0)=1 exactly; the branch bought nothing (round 1 died at trace: smoke-softmax-reduce-k4.md).
- Correctness 20261021-185149: 2 failed / 2 passed — e2m3 p16 varlen max 5.68e-02 (2/24576 elements), fused merge e2m3 varlen max 3.48e-01 (163/12288 elements); ALL cases ≥4.6× baseline max.
- Ref checks PASS: K4@8k 0.0181, K4@64k 0.0056, K8@64k 0.0066, K1@64k 6.1e-05 (pre-change class); |lse−emu| 9.5e-07 everywhere — (m, l) exact as designed.
- K-sweep 20261021-185837: k1/k2/k4/k8 FLAT within noise at 8192 and 65536 (ncu AND wall) — no win to bank even before the accuracy problem.
- Saturation mechanism: any key in the frozen window scoring above it gives p > 1 → p·448 > 448 → e4m3 clamps; e2m3 per-channel scales produce occasional large score jumps → varlen/fused-merge (many chunks) hit it most; the bf16 path (no clamp) still degraded 19.5× — stale-reference P quantization hurts beyond saturation.
- Fix directions left to the orchestrator: (a) clamp the P argument (loses moved mass where it matters), (b) wider p_scale budget only when stale, (c) reduce every tile but skip the smem round trip, (d) park the idea — timing shows no win to chase. Nothing committed.
**Full report:** `git show 56caea5:research/runs/oct21/notes/softmax-reduce-every-n.md`
