# TC decode — the "not fully Blackwell" question: fp8-PV `mma.sync` blockers + tcgen05

**Status: INVESTIGATION COMPLETE (read-only — no code changes, no GPU runs).**
Closes the open item in `tc-parity-plan.md` §2.2a ("Investigation: delegated (see the
follow-up research task), not yet decided"). Companion to `tc-parity-plan.md`,
`decode-1cta-clean.md`, `decode-kernel-feasibility.md`, `gemv-fi-gap-final.md`.

Primary object of study:
`vllm_flash_attn/cute/sm100_hd256_decode_tc.py` (the TC kernel; the shipped copy at
`docker/vllm-thor/fa4-gemv-kernel/sm100_hd256_decode_tc.py` is **byte-identical** —
same md5, so every line ref below applies to both).

---

## 0. TL;DR / verdict

The TC kernel is an **`mma.sync` (Sm80-class) kernel**, not a Blackwell (`tcgen05`)
kernel. Its QK leg is already fp8 (the native `m16n8k32` fp8 tensor-core mma), but its
**PV leg is bf16 by construction** because of **two genuine sm_110a `mma.sync`
blockers**, both confirmed with exact line refs (§1):

- **(a) the e4m3 P A-fragment cannot be produced in registers** — the in-register
  C→A reshape (`reshape_acc_to_frgA`) is 16-bit-geometry only, and even a correctly
  sized fp8 view is impossible because the C-fragment and fp8-A-fragment
  **thread→element mappings disagree** (75–100% of threads hold none or half of the
  elements they need → cross-thread data movement → smem round-trip is mandatory).
- **(b) the e4m3 V B-operand ldmatrix faults on sm_110a** (`cudaErrorMisalignedAddress`)
  — the QK leg sidesteps it by loading K **element-wise** from the mma's own B-partition
  (proven); the PV leg sidesteps it by **staging V as bf16** (`sVb`) and ldmatrix'ing
  that (proven).

**Q2 — fp8-PV `mma.sync` feasibility: FEASIBLE, ~0 gain at M=1.** All three pieces have
in-tree precedent (§2): V B-operand element-wise (the K-leg idiom), P smem round-trip +
element-wise A load (the FA3 `sP` idiom in `flash_fwd_sm90.py`), and `cvt.rn.satfinite`
e4m3 (PTX ISA 8.1, sm_89+; P ∈ [0,1] under the running-max softmax, so saturation never
fires in practice — safety net only). Cost ≈ one smem round-trip per n-block; removes the
32 KiB `sVb` + its convert + barrier. **It buys nothing at M=1** (memory-bound; the QK
mma is already fp8; the PV bf16 mma is not the limiter) — its only value is in the
**M≥8 (MTP verify) / prefill** regime, which is the varlen-M workstream.

**Q3 — tcgen05: NOT locked to 2CTA, and NOT worth adopting in the TC kernel.**
`tcgen05.mma.cta_group::1` is legal PTX, and **a 1CTA tcgen05 hd256 decode kernel already
exists in-tree** (`sm100_hd256_2cta_fmha_forward.py` with `use_2cta=False`, shipped as the
v11 patch, T10/DK-1 GO) — measured **207.1 µs** at L=8192/M=1 (1.56× its 2CTA form, but
**2.25× slower than the production FI path in the same window**). Adding tcgen05 to the
TC kernel = building a redundant second tcgen05 hd256 decode kernel that is structurally
the wrong primitive for M=1 (64-row M-mode floor + per-block TMEM/TMA/warp-specialization
machinery). **Rejected.**

**Recommendation (§4):** keep the TC kernel as the `mma.sync` memory-bound streamer it is
(fp8 QK + bf16 PV + `sVb` staging). Do **not** do fp8-PV or tcgen05 now. Revisit as a
**single M≥8 dispatch spike** ("M≥8 decode: FA4-1CTA tcgen05 vs TC-mma+fp8-PV vs
GEMV-varlen-M"), where PV mma throughput and the M-tile floor actually bind — and at that
point the fp8-PV plan in §2.2 is the implementation path if the TC-mma option wins.

---

## 1. Q1 — the two sm_110a `mma.sync` blockers, confirmed

### 1.1 Blocker (a): the e4m3 P A-fragment does not pack in registers

**Where the PV mma takes P today.** Two call sites, identical shape — the first
n-block and the steady-state loop:

```
:802  rP = cute.make_fragment_like(acc_S, PT)
:803  rP.store(acc_S.load().to(PT))
:804  tOrP = layout_utils.reshape_acc_to_frgA(rP)
...
:815  sm80_utils.gemm_rs(thr_mma_pv, acc_O, tOrP, tOrVt, tOsVt[None, None, None, 0],
:816                     smem_thr_copy_V)
```

and the same at `:876-878` / `:884-886`. `PT = DTYPE` (`:291`) — P is **always cast to
bf16/fp16**, and the PV mma op is **always** `MmaF16BF16Op(DTYPE, f32, (16,8,16))` even on
the fp8 leg (`:420-421`, selected at `:420-428`), with the in-kernel comment at
`:412-419` documenting the blocker: *"a fp8 PV mma would need a P smem round-trip
(FA2 pattern); instead P is cast to bf16 in-regs … and V is cast per-element from fp8
smem."*

**Mechanism 1 — the `frgA` view is 16-bit-geometry only.**
`reshape_acc_to_frgA` (quack `layout_utils.py:195-235`) has no fp8 branch — the comment
at `:199` is `# TODO: Sm90 FP8`. The Sm80 branch (`:224-234`) reinterprets the C
fragment `(4, MMA_M, MMA_N)` as `((4,2), MMA_M, MMA_N/2)`: a k-block of **8 elements**
(4 cols × 2 n-tile pairs). The 16-bit PV mma `m16n8k16` A-fragment is exactly that
geometry — 8 elements per 16-col k-tile per thread — so the view is *bit-exact* for bf16
(and that is why the whole bf16 leg, and the fp8 leg's bf16-PV, are green).

For the fp8 mma `m16n8k32`, the A-fragment needs **16 e4m3 per 32-col k-tile per thread**
(4 × b32 registers). Feeding it the 8-element view makes the mma's k-loop index past the
view into **undefined register bits → NaN** — the "k-loop over-ranges" in the `:412-419`
comment.

**Mechanism 2 — even a correctly-sized fp8 view is impossible: the thread mappings
disagree.** This is the deeper problem, confirmed by fragment-layout analysis (PTX ISA
`mma.sync` fragment layouts):

- The QK **C-fragment** (`m16n8`, fp32 acc): thread T holds S-rows {T/4, T/4+8} at cols
  **{2·(T%4), 2·(T%4)+1}** of each 8-col n-tile → over a 32-col window, thread T holds
  cols {2t, 2t+1, 8+2t, 9+2t, 16+2t, 17+2t, 24+2t, 25+2t} where t = T%4.
- The fp8 **A-fragment** (`m16n8k32`, e4m3): thread T holds rows {T/4, T/4+8} at cols
  **{4t..4t+3}** and **{16+4t..16+4t+3}** of the 32-col k-tile.

  | T%4 | C holds (cols) | fp8-A needs (cols) | overlap |
  |-----|----------------|--------------------|---------|
  | 0   | 0,1, 8,9, 16,17, 24,25 | 0..3, 16..19 | **half** (0,1,16,17) |
  | 1   | 2,3, 10,11, 18,19, 26,27 | 4..7, 20..23 | **∅** |
  | 2   | 4,5, 12,13, 20,21, 28,29 | 8..11, 24..27 | **∅** |
  | 3   | 6,7, 14,15, 22,23, 30,31 | 12..15, 28..31 | **half** (14,15,28,29) |

  Threads T%4 ∈ {1,2} hold **none** of the elements the fp8 A-fragment needs; T%4 ∈
  {0,3} hold exactly half. The in-register `reshape_acc_to_frgA` is a *view* — it
  reinterprets the same 4 registers per thread, it cannot move data between threads —
  so an in-register fp8 P A-fragment is impossible **regardless of view size**. (For
  16-bit, the A-fragment's thread mapping *is* {2t, 2t+1} per 8-col window — identical
  to the C mapping — which is exactly why the bf16 path works and the fp8 path doesn't.)

  Cross-thread movement is mandatory; in CuTe-DSL the vehicle is an smem round-trip
  (warp-shuffles are possible but strictly more complex for no cost win).

**Verdict: blocker (a) is real, and the fix is a P smem round-trip — not a view fix.**

### 1.2 Blocker (b): the e4m3 V B-operand ldmatrix faults on sm_110a

**The fault.** The 8-bit ldmatrix op (`LdMatrix16x8x8bOp`) faults with a **genuine
`cudaErrorMisalignedAddress`** on k-major smem on sm_110a — kernel comment `:560-564`.
Root-cause research: `task/research/cute-dsl-ldmatrix-fp8-b-operand/summary.md` — the op
is not PTX-faithful (it lowers to `m16n16.b8` + a permutation and requires 8-byte-aligned
row bases); the element mapping it does deliver is wrong for our B-operand orientation.

**The retile is also broken.** The b16-typed-fp8 **k-major** tiled-copy retile
(`tSrK_cp = retile(tSrK)` / `tSsK = partition_S`) **delivers only ~33 of the 256 d-values**
on sm_110a (probe `kdump`; kernel comment `:108-111`).

**How the QK leg sidesteps both (proven, green in the 4-combo gate).** K is loaded
**element-wise** from the mma's own B-partition — `sm100_hd256_decode_tc.py`:

```
:549  sK_b = thr_mma_qk.partition_B(sK[None, None, 0])   # fragment-slot → (n_kv, d) map
...
:115      tSrK[_e] = sK_b[_e]                            # per-slot element copy
:116      cute.gemm(tiled_mma_qk, acc_S, tSrQ[None, None, k], tSrK[None, None, k], acc_S)
```

Q's A-operand is loaded element-wise the same way (`:538-539` `tCsQ =
thr_mma_qk.partition_A(sQ)`; `:676` `tSrQ[_q] = tCsQ[_q]`). No ldmatrix anywhere on the
fp8 QK leg. The fp8 QK mma itself is the native `MmaFP8Op(f8, f32, (16,8,32))`
(`:402`) with `mma_k = 32` (`:403`).

**How the PV leg sidesteps it (proven): V staging.** The fp8 leg stages V as bf16:
`_convert_v_to_bf16` (`:120-136`) converts `sV` (fp8, just cp.async'd) → `sVb` (bf16,
`:447-465` allocation, `:479-484` selection) element-wise over the shared `(n_kv, d)`
logical grid (different swizzles, same logical shape — CuTe applies each tensor's own
swizzle), with a fp8-leg-only `sync_threads()` after the all-threads write and before the
all-threads ldmatrix read (`:809-814` first block; `:880-883` steady state — the sVb
cross-warp race fixed per `tc-parity-plan.md` §2.2). The PV ldmatrix then reads `sVb`
**exactly like the proven bf16 leg** (atom `:593-599`; the fp8-leg ldmatrix atom at
`:569-570` is **dead code**).

**Verdict: blocker (b) is real; the element-wise workaround exists and is proven on the
K leg — it is simply not ported to the PV leg (the `sVb` staging is used instead).**

---

## 2. Q2 — fp8-PV `mma.sync`: feasibility, plan, cost

**Feasible. All three pieces have in-tree precedent.**

### 2.1 V B-operand — element-wise load from `sV` (port the K-leg idiom)

`sV`'s smem layout `(n_kv, d)` **is exactly** the PV B-operand's logical (K, N) grid —
no re-layout. Build the B partition from the PV mma itself:

```python
sV_b = thr_mma_pv.partition_B(sV[None, None, 0])     # fragment slot → (n_kv, d)
# in the n-block loop:
for _e in cutlass.range_constexpr(cute.size(tOrV)):
    tOrV[_e] = sV_b[_e]                               # e4m3, per-slot copy
cute.gemm(tiled_mma_pv, acc_O, tOrP, tOrV, acc_O)     # MmaFP8Op(16,8,32), k=32
```

Same pattern as the K leg (`:549`/`:115`); cost = 128 e4m3 element loads per thread per
n-block (identical to what the K leg already pays). **Bonus: removes `sVb` (32 KiB smem),
`_convert_v_to_bf16`, and its barrier.**

### 2.2 P A-operand — smem round-trip + element-wise A load

In-register is dead (§1.1). The round-trip:

1. **Store** e4m3 P into an `sP` smem buffer via a tiled copy built from the **QK mma's
   C partition** (the C→smem mapping is the proven one), after `rP.store(acc_S.load().to
   (e4m3))` — i.e. the QK-`partition_C` → `sP` direction.
2. `fence_view_async_shared` + barrier.
3. **Load** the A fragment **element-wise** from `sP` via the **PV mma's A partition**
   (`sP_a = thr_mma_pv.partition_A(sP[...])`; `tOrP[_e] = sP_a[_e]`) — the Q-leg idiom
   (`:676`).

In-tree precedent — the FA3 Hopper kernel does exactly this in its
`mma_pv_is_rs=False` mode (`flash_fwd_sm90.py`):

```
:1304  acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
:1305      wg_mma_pv, (tile_m, tile_hdimv, tile_n), sP, sVt)     # A comes from sP smem
:1312  smem_copy_atom_P = utils.get_smem_store_atom(...)
:1314  smem_thr_copy_P = make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk)  # QK's C partition
:1316  tPsP = smem_thr_copy_P.partition_D(sP)
...
:1765  tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
:1768  tOrP_cur.store(tOrP_acc.load().to(self.dtype))
:1773  if const_expr(not self.mma_pv_is_rs):
:1774      tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
:1775      cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
:1776      cute.arch.fence_view_async_shared(); cute.arch.sync_warp()
```

(plus the `sP` buffer at `:95-103`, `:144-145`, `:180/189`, `:335-337`, `:670-672`; the
`mma_pv_is_rs=True` RS-mode default and the fp8-KV-forced RS mode are in
`interface.py:1425-1433`.)

### 2.3 The e4m3 cast — `cvt.satfinite`

- PTX: `cvt.rn.satfinite.e4m3x2.f32 d, a, b` — "convert a, b to .e4m3 and pack as
  .e4m3x2 output" (PTX ISA §9.7.10.24, examples); `cvt.satfinite.{e4m3x2,e5m2x2}.{f32,
  f16x2}` for **sm_89** introduced in **PTX ISA 8.1** → available on sm_100/sm_110.
  In CuTe-DSL this is just `.to(cutlass.Float8E4M3)` on the P fragment.
- `.satfinite` is **inherent to the e4m3 instruction form** (there is no non-saturating
  f32→e4m3 variant): finite overflow saturates to ±448 (the e4m3fn max); e4m3fn has no
  Inf/NaN encodings.
- **It never fires in practice here**: P is the output of the running-max online softmax
  in the log2 domain — `P = exp2(s·scale − max·scale)` with the running max over all
  columns (`softmax.py:150-178`) → **P ∈ [0, 1]** (masked OOB columns: s = −inf → P = 0;
  the `row_max_cur == -inf` guard at `:165` makes even an all-OOB tile yield P = 0, not
  NaN). So satfinite is a safety net, not a functional requirement.
- In-tree e4m3-P-cast precedent with the scaling trick: the FA4 hd256 kernel scales P by
  2⁸ before the e4m3 cast to use the upper code points, bounded by
  `assert 8 < _LOG2_DTYPE_MAX[e4m3]` (`sm100_hd256_2cta_fmha_forward.py:1883-1894`, cast at
  `:1935`); the generic kernel carries the same low-precision-scaling note
  (`flash_fwd_sm100.py:2190, :2198-2201, :2632-2634` per `fa4-hd256-route-trace.md`).

### 2.4 Cost / gain

| | today (bf16 PV) | fp8 PV |
|---|---|---|
| PV mma per 64-row n-block | 4 × `m16n8k16` (16-bit A from regs) | 2 × `m16n8k32` (fp8 A from smem, B from smem) |
| P path | in-regs (free) | smem store + barrier + element-wise A load (~32 e4m3 in/out per thread) |
| V path | `sVb` convert (64 bf16 writes) + barrier + ldmatrix | element-wise e4m3 B load (128 loads) — K-leg idiom |
| smem | `sVb` 32 KiB + barriers ×2 | `sP` (32 e4m3 per n-block, reused) + barrier ×1 |

Net smem traffic ≈ wash; one 32 KiB buffer and one convert+barrier go away. **Gain at
M=1: ~0** — the kernel is memory-bound (the KV load is the limiter; `tc-parity-plan.md`
§2.2a, `gemv-long-context-diagnosis.md`), the QK mma is already fp8 (the dominant mma and
the dominant DRAM bytes), and the PV bf16 mma is not on the critical path. **Gain at
M≥8/prefill: real** — 2× PV mma throughput, fewer PV issues per n-block, but that regime
needs the varlen-M generalization first (the GEMV varlen-M workstream is the same
regime).

---

## 3. Q3 — tcgen05: the "fully Blackwell" question

**Is tcgen05 locked to 2CTA / FA3 / FA4? No.**

- `blackwell_helpers.py:201` (and siblings): the raw PTX emitted is
  `tcgen05.mma.cta_group::1.kind::…` / `::2` — **`cta_group::1` is a first-class, legal
  form**; the kind string (`f16`/`tf32`/`f8f6f4`/…) is mapped by the op
  (`_tcgen05_mma_kind`). The generic FA4 kernel selects
  `cta_group = TWO if use_2cta_instrs else ONE` (`flash_fwd_sm100.py:539`).
- tcgen05's **M-mode floor is 64 (1CTA) / 128 (2CTA); there is no 32- or 16-row MMA**
  (`decode-kernel-feasibility.md` §1). At M=1 that is 63 wasted rows per CTA — and the
  measured M=1 cost is dominated by the per-block TMEM/TMA/warp-specialization
  machinery, not the M-tile (DK-1b: *"the cost is the 2CTA cluster, not the M-tile"* —
  and after 1CTA the remainder is still ~2.25× vs FI).

**A 1CTA tcgen05 hd256 decode kernel already exists in-tree and runs on sm_110a.**
`sm100_hd256_2cta_fmha_forward.py` with `use_2cta=False` — shipped by
`thor-fa4-hd256-1cta-decode-sm110.patch` (v11, decode M≤8 carve-out,
`thor-fa4-hd256-1cta-decode-sm110.patch` at `docker/vllm-thor/patches/`; T10 canary
PASS; DK-1 GO in `decode-1cta-clean.md`):

- 12-warp specialization (`:132-136`): softmax 0-3, correction 4-7, mma warp 8, load warp 9.
- P lives in **TMEM**: `OperandSource.TMEM` (`:469`); the softmax warp loads S from TMEM
  (`Ld32x32bOp`), does exp2, casts to e4m3 (`:1935`, with the `max_offset=8` low-precision
  scaling, `:1883-1894`), stores P to TMEM (`St32x32bOp`, `:1936-1957`); the PV mma takes
  **A = P from TMEM** (recast to q_dtype, `:1236-1243`) and **B = V from smem**
  (`:1248-1255`); a single leader thread issues the mma (`:1181`); TMEM offsets
  s=0/o=256/p (`:154-156`).
- fp8-KV on this kernel is the Thor FP8 patch (`thor-fa4-hd256-fp8-sm110.patch`, T11
  canary PASS — the kernels *run* on sm_110a).

**Measured** (`decode-1cta-clean.md` §2, gated clean window, L=8192/M=1 medians):
FA4 1CTA tcgen05 **207.1 µs** vs FA4 2CTA **323.0 µs** (1.56×) vs FlashInfer production
decode path **92.2 µs** (1CTA/FI = **2.25×**). In the GEMV workstream's window, GEMV
(ns=20) **222.8 µs** vs FI **202.05 µs** (1.10×) — so across windows the consistent
ordering is **FI (fa2-tc) < 1CTA-tcgen05 ≲ GEMV**, i.e. the tcgen05 decode kernel is not
the best M=1 kernel we have, and GEMV is within ~10% of FI.

**Therefore "adopting tcgen05 in the TC kernel" means building a second, competing in-tree
tcgen05 hd256 decode kernel** — redundant with the existing one, and the existing one is
already measured to lose at M=1. The tcgen05 M=1 loss is structural (64-row M-tile floor +
TMEM/TMA/warp-specialization overhead), not a tuning gap.

**The M≥8/prefill "Blackwell gap" is already closed by existing in-tree assets**: the FA4
2CTA/1CTA tcgen05 kernels + the Thor FP8 patch cover M≥8 decode and prefill (the 1CTA
carve-out admits M≤8; prefill runs 2CTA). The *remaining* hd256+FP8-KV route issue is
vLLM's **FA4→FA2 policy downgrade** for quantized KV (`fa4-hd256-route-trace.md`), which
is an upstream fp8-descale-branch matter (the plan in `AGENTS.md` "Open work"), not a
TC-kernel matter.

---

## 4. Recommendation

1. **Keep the TC kernel as it is** — the `mma.sync` memory-bound streamer for M=1
   long-context decode (its design point; it is stronger than GEMV at long L,
   `tc-parity-plan.md` "The goal"), with the fp8 QK mma + bf16 PV + `sVb` staging.
2. **Do not adopt tcgen05 into the TC kernel** (rejected: redundant with the in-tree FA4
   1CTA tcgen05 decode kernel; structurally the wrong primitive at M=1; measured to lose
   at M=1).
3. **Do not implement fp8-PV `mma.sync` now** — ~0 gain at M=1 (the PV bf16 mma is not
   the limiter; §2.4). **Defer to the M≥8/varlen-M workstream**, where PV mma throughput
   and the tcgen05 M-tile floor actually bind. If/when the M≥8 spike wants an `mma.sync`
   option, the implementation path is §2.1–§2.3 (V element-wise B; P smem round-trip +
   element-wise A, FA3-`sP` precedent; `cvt.rn.satfinite` — PTX ISA 8.1 / sm_89+, P ∈ [0,1]
   so no saturation in practice).
4. **Frame the M≥8 question as one dispatch spike**, not a kernel rewrite:
   "M≥8 (MTP verify) decode: **FA4-1CTA tcgen05** (in-tree, T10/DK-1 GO) vs
   **TC-mma + fp8-PV** (this plan) vs **GEMV-varlen-M** (open workstream)" — measure in a
   gated window, then set the dispatch.

---

## 5. References / prior art (exact)

| Artifact | Location | Establishes |
|---|---|---|
| TC kernel (object of study) | `vfa-tree/vllm_flash_attn/cute/sm100_hd256_decode_tc.py` (= `docker/vllm-thor/fa4-gemv-kernel/sm100_hd256_decode_tc.py`, byte-identical, 975 lines) | all TC line refs above |
| fp8 QK mma + element-wise K/Q | TC kernel `:402-403` (op), `:549`/`:115-116` (K), `:538-539`/`:676` (Q), `:108-111` (broken-retile comment) | blocker (b) workaround, proven |
| PV always bf16 + blocker comment | TC kernel `:291` (PT), `:412-428` (op + comment), `:802-817`/`:876-886` (sites) | blocker (a) state |
| `sVb` staging | TC kernel `:120-136` (convert), `:447-465` (buffer), `:479-484` (selection), `:809-814`/`:880-883` (barrier) | blocker (b) PV workaround |
| 8-bit ldmatrix fault | TC kernel `:560-564`; `task/research/cute-dsl-ldmatrix-fp8-b-operand/summary.md` | root cause (not PTX-faithful) |
| `reshape_acc_to_frgA` (16-bit only) | quack `layout_utils.py:195-235` (Sm80 branch `:224-234`; `# TODO: Sm90 FP8` at `:199`) | blocker (a) mechanism 1 |
| FA3 P smem round-trip precedent | `flash_fwd_sm90.py:95-103, :144-145, :180/189, :335-337, :670-672, :1304-1305, :1312-1317, :1765-1776`; `interface.py:1425-1433` (RS-mode forcing) | §2.2 pattern, in-tree |
| tcgen05 raw PTX (cta_group::1/::2) | `blackwell_helpers.py:201` (+kind map); `flash_fwd_sm100.py:539` | tcgen05 not 2CTA-locked |
| FA4 hd256 1CTA tcgen05 decode | `sm100_hd256_2cta_fmha_forward.py:132-136, :154-156, :469, :1181, :1236-1255, :1883-1957`; patch `thor-fa4-hd256-1cta-decode-sm110.patch`; canary T10 | the in-tree tcgen05 decode kernel |
| 1CTA tcgen05 measurements | `docs/fa4-hd256-fp8/decode-1cta-clean.md` (207.1 µs; 1.56× vs 2CTA; 2.25× vs FI in-window) | tcgen05 loses at M=1 |
| tcgen05 M-mode floor 64/128 | `docs/fa4-hd256-fp8/decode-kernel-feasibility.md` §1 | structural M=1 cost |
| GEMV vs FI window | `docker/vllm-thor/fa4-gemv-kernel/README.md` (222.8 vs 202.05 µs); `docs/fa4-hd256-fp8/gemv-fi-gap-final.md` | M=1 ordering |
| cvt e4m3 PTX | PTX ISA §9.7.10.24: `cvt.rn.satfinite.e4m3x2.f32 d, a, b`; `cvt.satfinite.{e4m3x2,e5m2x2}.{f32,f16x2}` for sm_89, PTX ISA 8.1 | §2.3 cast |
| P ∈ [0,1] under online softmax | `vllm_flash_attn/cute/softmax.py:150-178` (log2 running max; −inf guard `:165`) | satfinite = safety net |
| design context | `docs/fa4-hd256-fp8/tc-parity-plan.md` §2.2a; `docs/thor-stack/fa4-hd256-route-trace.md`; `AGENTS.md` (Open work — the upstream fp8-descale plan) | the open item closed by this report |
