# Python Engineer: SW128 brick (e2d09c2) port assessment (digest)
2026-10-21; python-engineer; analysis-only port assessment of run-1 brick e2d09c2 (on 50ad3ce) onto the split family at fd2a0c8.
**Verdict:** COMPLETE (analysis only) — protocol reset verified and a 3-hunk placement map produced; the port itself later proved moot (the wide derivation is inert — wide-sw128-ring-inert.md).
**Key facts:**
- Protocol reset verified: `git reset --hard fd2a0c8`; `git diff --stat 43c659b fd2a0c8 -- src/mjolnir/` → empty; `git tag -l 'ar/parked/*'` → ar/parked/2f20c4f intact.
- The brick is the legalization FIX for 50ad3ce ("gated wide fp6 KV TMA … default 64K"): rebinding `tma_atom_k = tma_atom_k_wide` inside a runtime `if wide_ring:` creates a TMA-atom PHI the CuTe-DSL tracer cannot legalize — fix: atoms stay SSA-const, each copy site selects its (atom, views) pair under `cutlass.const_expr(WIDE)` + runtime `wide_ring`.
- Placement: hunk 1 → wide-view setup after t5_decode.py:705 (no rebind); hunks 2–3 → K copy site 734-742, V copy site 747-755 (parents textually identical).
- Prerequisites live only in 50ad3ce (WIDE const + gate, wide TMA atoms, gmem/views); t5_plan.py needs no placement.
- L2P prefetch sites (717/719/725/727) keep the NARROW atom (deliberate); current fp6 default ring 2 stages @ 2 CTAs/SM composes; t5_*.py excluded from the file-length ratchet.
**Full report:** `git show 56caea5:research/runs/oct21/notes/sw128-brick-port-assessment.md`
