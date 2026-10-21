# Python Engineer: SW128 brick (e2d09c2) port assessment

## Plan Reference

Autoresearch run follow-up — dump run-1 SW128 brick for porting onto the split family.

## Protocol Reset Verification

- `git reset --hard fd2a0c8` → HEAD `fd2a0c8` ("autoresearch: re-export run 1 ledger"),
  parent `43c659b` (the harness merge).
- Kernel-identical: `git diff --stat 43c659b fd2a0c8 -- src/mjolnir/` → empty (only
  `research/runs/run1/*` differ).
- `git status --porcelain -- src/mjolnir/` → empty.
- `git tag -l 'ar/parked/*'` → `ar/parked/2f20c4f` (ring-pacing parked row, intact).

## Brick

- `e2d09c2` — "fix gated wide TMA legalization: no TMA-atom PHI (use clean atom at each
  copy site)". 1 file, +50/−16, `src/mjolnir/mjolnir/t5_decode.py` only (run-1 monolith).
- Sits on `50ad3ce` ("gated wide fp6 KV TMA: stream SW128 ring iff
  `seq_len >= fp6_wide_tma_min_l` (default 64K)") — the brick is the legalization FIX for
  that experiment; the full SW128 footprint spans both commits.
- Root cause: rebinding `tma_atom_k = tma_atom_k_wide` inside a runtime `if wide_ring:`
  branch creates a TMA-atom PHI the CuTe-DSL tracer cannot legalize. Fix: atoms stay
  SSA-const; each copy site selects its (atom, views) pair under
  `cutlass.const_expr(WIDE)` + runtime `wide_ring`.

## Placement Map (brick hunk → current family @ fd2a0c8)

| Hunk | Current location |
|---|---|
| 1. Remove `tma_atom_k/v = tma_atom_k/v_wide` rebind (keep wide views) | `t5_decode.py` PS==128 KV-producer branch: insert wide-view setup after line 705 (end of the `tGSsV, tGSgV` tma_partition), before the s-loop at 707 — **without** the rebind |
| 2. K copy site (const_expr WIDE/wide_ring atom selection) | `t5_decode.py:734-742` — `cute.copy(` at 734, `tma_atom_k` at 735, inside `if s < iters_s:` / `for dk in range_constexpr(tiles_dk)` |
| 3. V copy site | `t5_decode.py:747-755` — `cute.copy(` at 747, `tma_atom_v` at 748, inside `if s >= prefetch_iters:` / `for dm …` |

Hunks 2–3 apply cleanly: the parent copy loops are textually identical to current
734-743 / 747-756.

### Prerequisites (referenced by the hunks; live in sibling 50ad3ce, absent at fd2a0c8)

- `WIDE` constexpr + `wide_ring` runtime gate (`seq_len >= fp6_wide_tma_min_l`, default
  64K) → `t5_harness.py` (flag + gate) → `t5_call.py __init__/__call__` → `decode()`
  signature (`t5_decode.py:259-263`).
- `tma_atom_k_wide` / `tma_atom_v_wide` → `t5_call.py:440-455` (next to the PS==128
  `make_tiled_tma_atom_A` block), from a wide SW128 smem layout derived next to
  `smem_layout_k/v` (`t5_call.py:403-408`); wide ring depth near `_default_kv_stages`
  (`t5_call.py:52-62`) / smem budget (~240-290).
- `mK_wide`/`mV_wide` gmem + `tAsK_wide`/`tAsV_wide` smem views → host side
  `t5_call.py __call__`, device side `t5_decode.py decode()` body.
- `t5_plan.py`: **no placement** — `plan_split_kv`/`SplitPlan`/`kernel_strides` are
  orthogonal to ring swizzle/width.

### Porting notes

- L2P prefetch sites (`t5_decode.py:717/719/725/727`) keep the NARROW atom — the brick
  deliberately left prefetch narrow (gmem-side hint; smem views irrelevant).
- Current fp6 default ring is 2 stages @ 2 CTAs/SM (`t5_call.py:61`, post run-1 ring
  sweep) — SW128 predates that default; a re-port must compose with it.
- `t5_*.py` excluded from the file-length ratchet — the +~30 lines are fine.

## Status

COMPLETE (analysis only — no code changed; reset protocol verified, brick dumped, map done)
