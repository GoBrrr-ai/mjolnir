# Python Engineer: softmax-reduce-every-n (round 2 — revert of the correction branch)

## Plan Reference

Orchestrator round 2: round 1's branched correction block failed at TRACE time
(`c_f32x2` first-defined inside staged `if do_reduce:`/`else:` branches — see
task/python-engineer/smoke-softmax-reduce-k4.md). Deeper insight: on non-reduce
tiles tSrM == tSrM_prev, so the ORIGINAL correction formula self-produces
exp2(0)=1 exactly — the branch bought nothing.

## Change

- REVERTED the correction block to the fd2a0c8 form (verified byte-identical
  via pattern-extract diff against `git show fd2a0c8:`).
- Added to the pacing-gate comment: "On non-reduce tiles tSrM == tSrM_prev so
  the correction formula self-produces exp2(0)=1 — no branch needed."
- Kept: param + flag, do_reduce gate, gated warp_fmax/smem_fmax/barrier/
  broadcast, hoisted tSrM allocation, m_use fixup + tSrM_prev.store every tile.

## Smoke (all runs 2026-10-21, no commits, --no-save)

1. `verify correctness --quick --match e2m3 --no-save` → **2 failed, 2 passed**
   (run 20261021-185149). FAILs: e2m3 p16 varlen (max 5.68e-02, 2/24576 el),
   fused merge e2m3 varlen (max 3.48e-01, 163/12288 el). ALL cases ≥4.6×
   baseline max. NOT a trace bug — a real accuracy regression.
2. Ref checks (one-shot containers, 600s kill-by-name, staged copy
   /tmp/opencode/smoke-reduce/src): K4@8k PASS 0.0181; K4@64k PASS 0.0056;
   K8@64k PASS 0.0066; K1@64k PASS 6.1e-05 (pre-change class). |lse-emu|
   9.5e-07 everywhere — (m, l) exact as designed.
3. `verify ab` K-sweep (run 20261021-185837): k1/k2/k4/k8 FLAT within noise at
   8192 and 65536 (ncu AND wall) — the max reduction is not on the critical
   path at these shapes; no win to bank even before the accuracy problem.

## Root cause of the accuracy regression (confirmed in code)

`t5_const.py:34-37`: fp8 P is stored as `e4m3(p * FP8_P_SCALE)`, `FP8_P_SCALE
= 448.0`, documented invariant **"p in [0, 1]"** — p=1 maps exactly to the
e4m3 max, so there is ZERO headroom above the reference. With a frozen
reference, any key in the frozen window scoring above it gives p > 1 →
p·448 > 448 → e4m3 SATURATES (clamps) → clipped top elements. e2m3 per-channel
scales produce occasional large score jumps → saturation; varlen/fused-merge
(many chunks) hit it most. The "8-binade budget" from the proposal is downward
only (underflow, same as fresh-max); the bf16 path (no clamp) still degraded
19.5×, so stale-reference P quantization hurts beyond saturation too.

## Status

COMPLETE — reported, nothing committed. Experiment dead in current form; fix
directions for the orchestrator: (a) keep the P≤1 invariant by clamping the
P argument (P = exp2(min(s − m_frozen, 0))·c — loses the moved-mass exactly
where it matters), (b) rescale P-domain via a wider p_scale budget only when
the reference is stale, (c) reduce every tile but skip only the smem round
trip, or (d) park the idea — timing shows no win to chase.
