# Debug Report: wide-SW128-fp6-ring-inert

## Issue Reference

Orchestrator delegation (batch commit `ff84fff`, branch `autoresearch/oct21`). The
gated SW128 wide fp6 KV ring (port of run-1 commits `50ad3ce` + `e2d09c2`) never
issues a 128-B L2 request: triad 20261021-170638 shows wide vs noWide
byte-identical read streams at 256K and at forced-wide 8192, while correctness
passes everywhere.

## Bug Summary

The "wide" derivation is a **semantic no-op**: `make_smem_layout_a` under the
fp8-MMA substitution produces a layout **byte-identical** to the narrow fp6-MMA
derivation (both already carry the `K_SW128` atom `S<3,4,3>`), so every wide
object — TMA atoms, smem views, MMA fragments — is identical to its narrow
counterpart and the runtime gate/re-bind cannot change behavior. The fp6 ring's
64-B requests are a **TMA-descriptor property of the packed-6-bit gmem box**
(128 elements = 96 B inner extent), which no smem-side change can move.

## Reproduction

### Command (bisect probe — hardcoded-wide build, /tmp copy, never the worktree)

```bash
# /tmp/opencode/wtma/src-hardwide = worktree src/mjolnir with ONE edit:
#   t5_decode.py:388  wide_ring = seq_len >= self.fp6_wide_tma_min_l
# → wide_ring = True  # PROBE: hardcoded-wide (compile-time; no runtime gate)
timeout 900 docker run --rm --name wtma-hardwide --gpus all \
  --security-opt seccomp=unconfined --cap-add SYS_ADMIN \
  -v /opt/nvidia/nsight-compute/2025.3.0:/ncu:ro \
  -v /tmp/opencode/wtma/src-hardwide:/opt/mjolnir-src:ro -e PYTHONPATH=/opt/mjolnir-src \
  --entrypoint bash narandill/mjolnir-vllm:v0.32.0 -c \
  "/ncu/ncu --target-processes all --clock-control base --cache-control all \
   --kernel-name 'regex:MixedInputFusedMultiHeadAttentionDecode' \
   --metrics lts__t_sectors_op_read.sum,lts__t_requests_srcunit_tex_op_read.sum \
   --print-summary per-kernel python3 -m mjolnir.t5_decode --q_dtype Float8E4M3FN \
   --kv_dtype Float8E4M3FN --kv_format e2m3 --kv_layout hnd --seqlen 8192 \
   --skip_ref_check --iterations 1"
```

### Output (bisect matrix, verbatim)

| # | Build (all one-shot containers, decode kernel, L=8192, HND) | requests | sectors | sectors/req | verdict |
|---|---|---|---|---|---|
| 1 | **hardcoded-wide** (compile-time, no runtime gate) — `src-hardwide` | 197,953 | 405,960 | **2.051** | still 64-B → derivation produces no wide path (suspects 1/2 moot) |
| 2 | **fp8 control** (`--kv_format fp8`, same probe) — `src-hardwide` | 132,493 | 537,178 | **4.053** | 128-B ✓ metric methodology sound |
| 3 | **runtime-gated wide** (unpatched worktree src, `--fp6-wide-tma-min-l 0`) | 198,010 | 406,499 | **2.053** | ≡ hardcoded-wide (0.03% jitter) → gate/re-bind change nothing |
| 4 | triad 20261021-170638 (256K wide vs noWide, orchestrator's evidence) | 6,293,744.67 / 6,293,829.67 | — | 2.01 both | byte-identical ✓ consistent |
| 5 | fp6prof 20261021-094358 (L=12288): fp8-hnd vs e2m3-hnd | 197,824 / 296,351 | 807,274 / 613,463 | 4.08 / 2.07 | fp8=128-B, fp6=64-B with **identical smem layouts** |

Ref check with the wide path compile-time forced (`--seqlen 65536`, emulator
comparison, `src-hardwide`): `|kern-emu| max=6.102e-05` → **PASS** (trivially —
the "wide" objects are the narrow ones).

### Consistent: YES (deterministic; reproduced 4 independent ways)

## Investigation Log

[CHECKED] Static config: harness `page_size=128` default, `blk_tile_s =
tile_s_rows = 128` (t5_plan.py:13) → `PS == blk_tile_s` holds; harness `run()`
defaults `fp6_wide_tma=True`, `fp6_wide_tma_min_l=65536` (t5_harness.py:163-164)
passed to the kernel (t5_harness.py:266-267) → `WIDE` is compile-time TRUE for
the triad's wide leg. Not a flag-plumbing bug.

[PROBE — derivation, host-side, no GPU] Built both layouts exactly as
t5_call.py does (container, `@cute.jit` context, `make_trivial_tiled_mma(Float6E2M3FN|Float8E4M3FN, Float8E4M3FN, K, K, f32, CtaGroup.ONE, (128,32), OperandSource.SMEM)`, `make_smem_layout_a(mma, (128,32,128), Int8, 4)`):

```
narrow partition_shape_A: ((128, 32), 1, 4)     wide partition_shape_A: ((128, 32), 1, 4)
narrow K: S<3,4,3> o 0 o ((128,32),1,4,4):((128,1),0,32,16384)   inner: S<3,4,3>
wide   K: S<3,4,3> o 0 o ((128,32),1,4,4):((128,1),0,32,16384)   inner: S<3,4,3>
K layouts equal? True          V layouts equal? True
fp8 KV ring (Float8E4M3FN, stages 2): S<3,4,3> o ((128,32),1,4,2):((128,1),0,32,16384)
make_smem_layout_atom(K_SW128, Int8) = S<3,4,3> o (8,128):(128,1)   ← S<3,4,3> IS K_SW128
make_smem_layout_atom(K_SW64,  Int8) = S<2,4,3> o (8,64):(64,1)
```

**The wide layout ≡ the narrow layout ≡ the fp8 ring's layout.** The port
analysis's premise ("make_smem_layout_a keys the swizzle atom off the A-operand
major_mode_size: 128 for fp8 → SW128, 64 for fp6 → SW64") is false: the chooser
(`get_smem_layout_atom_ab`) sees (a) the MMA's A-instruction tile via
`partition_shape_A` — the kind::f8f6f4 MMA's A tile is (128, 32), **same as the
fp8 MMA's** → `a_smem_shape_mn_k = (128, 128)` for both — and (b) the dtype
width — `kv_smem_dtype` is `Int8` (1-B containers) in **both** derivations
(t5_call.py:48). Same inputs → same `K_SW128` atom.

[COROLLARY] `tma_atom_{k,v}_wide` is built by the *same* call with the *same*
arguments as `tma_atom_{k,v}` (t5_call.py:473-487 vs 521-536; only
`smem_layout_*_wide` differs, and it doesn't) → identical descriptors.
`tGSs{K,V}_wide ≡ tGSs{K,V}` and `tAt{K,V}_wide ≡ tAt{K,V}` for the same
reason. The runtime `if wide_ring:` re-binds (t5_decode.py:799-803, 1260-1262,
1324-1326) and the per-site atom selection shuffle **identical objects**.

[PROVEN — GPU bisect] Matrix rows 1–3 above: hardcoded-wide and runtime-gated
wide both 2.05 sectors/req; fp8 control 4.05. The inertness is in the
derivation, and no gate/staging fix could change it.

[ROOT OF THE 64-B REQUESTS] fp6 and fp8 rings share the same SW128-atom smem
layout, yet issue 2.05 vs 4.05 sectors/req (matrix row 5) → the request size is
set by the **TMA descriptor's gmem box**, not the smem swizzle: the packed
6-bit gmem row gives a 128-element (mma_tile_k) box an inner extent of
**96 B** (vs fp8's 128 B). Geometry check for reaching 128-B requests with
d-major packed-fp6 KV at d=256: a row is 192 B; box inner extents are
96 B (128 elems, current), 192 B (256 elems — not a 128-B multiple), 384 B
(512 elems — exceeds the TMA boxDim[0] ≤ 256 limit). **128-B-swizzle requests
are geometrically unreachable for packed-fp6 d-major KV at d=256**; the only
levers are gmem row padding (192→256 B: erases the fp6 storage saving) or
mma_tile_k=256 (32-KiB stages — deep kernel surgery).

[CHECKED — run-1 ledger] research/runs/run1/ledger.jsonl rows 9–12:
- Rows 9/10 (1d3bcbd, ungated "wide"): per-L vs FI moved vs the row-1 baseline —
  but 1d3bcbd ≡ baseline functionally: c99442c's fp8-MMA layout derivation ≡
  the fp6 derivation (probe above), and 1d3bcbd only deleted the crashing
  hand-built WIDE TMA branch, falling through to the stock `PS==blk_tile_s`
  atom (eefa370 already had it, line 662). **The wide path never engaged in
  run 1 either — the rows 9/10/12 short-L drops and long-L "wins" are
  FlashInfer-side drift** (row-12's own verdict: "FlashInfer-side false
  positive"; row 1 swept Oct 13 22:02, rows 9–12 Oct 14 morning). The row-11
  (50ad3ce) and rows 2–8 crashes came from the hand-built wide TMA-atom
  branch, not from the layout.
- Consequence: AGENTS.md's "not the 64-byte L2 requests (SW128 tried: ~5% at
  ≥64K only)" rests on an inert build + FI noise — the SW128 hypothesis was
  never actually tested. It is re-opened, and matrix row 5 says the smem side
  is the wrong lever regardless.

[RELATED] `runner.py` `VariantKey` lacks fp6_wide_tma/fp6_wide_tma_min_l →
wide/noWide would collide in `_COMPILED` + the on-disk kernel cache. Moot today
(backend.py never sets the flag; the harness compiles via plain `cute.compile`,
t5_harness.py:475, no cache) — but must be fixed if the flag is ever wired to
the backend.

## Root Cause

ROOT CAUSE: the ported wide path is inert **by construction**. Its "SW128
derivation" (`make_smem_layout_a` under an fp8-MMA substitution, dtype
`kv_smem_dtype=Int8`) yields a ComposedLayout **identical** to the narrow fp6
derivation, because the f8f6f4 MMA's A-instruction tile equals the fp8 MMA's
((128,32) → `a_smem_shape_mn_k` (128,128) → `K_SW128` for both) and the smem
dtype is 1 byte in both. Every wide object is therefore the narrow object; the
`wide_ring` gate and the re-binds are no-ops. The fp6 ring already carries the
SW128 smem atom — its 64-B L2 requests come from the TMA descriptor's gmem box
for the packed 6-bit data (96-B inner extent per 128-element mma_k chunk), a
lever the smem layout does not control.

EVIDENCE: bisect matrix (rows 1–5 above) + host-side layout probe + t5_call.py
argument identity + run-1 ledger re-read.

## Recommended Fix

The feature cannot achieve its goal as designed — do not "fix the gate".
Two options:

**A (minimal, recommended): remove the dead wide-TMA machinery.** Exact edit
spec against the worktree (all hunks are self-contained `if
cutlass.const_expr(WIDE):` blocks or flag plumbing; after removal
`wide_ring`/`WIDE` no longer exist and every copy site reverts to the single
narrow `cute.copy`):

1. `src/mjolnir/mjolnir/t5_call.py`
   - 407–415: delete the wide-KV comment block + `WIDE = cutlass.const_expr(...)`.
   - 421–447: delete the `if cutlass.const_expr(WIDE):` block (fp8
     `make_trivial_tiled_mma` ×2 + `make_smem_layout_a` ×2).
   - 519–543: delete the wide `make_tiled_tma_atom_A` block **and** the
     `else:` alias branch (with WIDE gone the aliases are unused).
   - 591–596: delete the six wide launch args (`smem_layout_k_wide`,
     `tma_atom_k_wide`, `tma_tensor_k_wide`, `smem_layout_v_wide`,
     `tma_atom_v_wide`, `tma_tensor_v_wide`).
2. `src/mjolnir/mjolnir/t5_decode.py`
   - 307–314: delete the six wide `decode()` params (+ their comment).
   - 353–357: delete the WIDE comment + const.
   - 384–388: delete the `if cutlass.const_expr(WIDE): wide_ring = ...` block.
   - 404–407: delete the wide descriptor prefetch.
   - 612–626: delete the wide ring views/fragments block.
   - 775–803: delete the wide tilings/partitions + `if wide_ring:` re-bind.
   - 857–875 and 895–913: collapse each `if cutlass.const_expr(WIDE): if
     wide_ring: … else: …` back to the single narrow `cute.copy(tma_atom_k,
     tGSgK[None, 0, dk, l_idx], tGSsK[None, k_handle.index], …)` (resp. V).
   - 1258–1262 and 1322–1326: delete both consumer re-point blocks
     (`tAtK_cvt = tAtK_wide` / `tAtV_cvt = tAtV_wide`).
3. `src/mjolnir/mjolnir/t5_harness.py`
   - 163–164: delete the `fp6_wide_tma` / `fp6_wide_tma_min_l` run() params.
   - 266–267: delete the pass-through.
   - 706–718: delete both CLI flags.
4. Re-run `uv run scripts/check_file_length.py --base origin/main` (the file
   shrinks — ceilings only ratchet down, safe) and `uvx ruff@0.14.14 check
   src/ tests/`.

Stopgap alternative if removal must wait: flip the harness default
`t5_harness.py:163` `fp6_wide_tma: bool = True` → `False` so legs stop paying
compile time for dead code, and correct the comment block (t5_call.py:407-414)
which asserts the false SW64/SW128 premise.

**B (only if the 128-B hypothesis is to be pursued): the sole in-layout route
is `mma_tile_k = 256`** (t5_call.py:189) — a 256-element packed box is 192 B
(inner extent > 128 B; predicted to lift the descriptor's swizzle span to
128-B per the "largest power-of-2 ≤ box bytes" rule that fits all five matrix
rows) → stage = 128×256×1 B = 32 KiB, `kv_stages` drops 4→2 (same 64 KiB KV
smem). Blast radius: tiles_dk, the MMA K-loops, TMEM cvt staging, the stage
barrier tx count (`_kv_tx_dtype`), the pacing `DEEP_ITERS` heuristics — a
multi-hour change whose payoff has never been measured (run 1's "5% at ≥64K"
was an inert build + FI noise). Not recommended before a cheap
hypothesis test exists.

## Related Issues

- `runner.py` `VariantKey` (src/mjolnir/mjolnir/runner.py:216) omits
  `fp6_wide_tma`/`fp6_wide_tma_min_l` — a compile-cache collision waiting for
  anyone who wires the flag into the backend (moot after option A).
- AGENTS.md "fp6 byte-vs-time gap" bullet cites "SW128 tried: ~5% at ≥64K only"
  — based on run-1 rows 9/10, which measured a functionally unchanged kernel
  vs a drifted FlashInfer reference. The bullet's conclusion may still hold,
  but its evidence does not; worth correcting when the ledger is next read.
- The same "re-derive the layout under a substitute MMA" pattern cannot work
  for ANY sub-8-bit format: the chooser's inputs (instruction A-tile + 1-B smem
  dtype) are invariant under the A-dtype substitution for the whole f8f6f4
  family (e2m1/e2m3/e3m2 all take (128,32) A tiles).

## Status

INVESTIGATION COMPLETE — ready for orchestrator. Root cause: the wide
derivation is a proven no-op (wide ≡ narrow layouts/atoms/views; 64-B requests
are a packed-6-bit gmem-box property); recommended fix: remove the dead
wide-TMA machinery (exact edit spec above). Do not commit from this session.
