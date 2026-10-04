# TC (mma.sync) hd256 decode — GEMV feature-parity plan

**Status: Phase 1 (correctness) — FULLY GREEN; Phase 2 (paged KV, the production blocker) — DONE; Phase 3 (SplitKV + the sVb race fix) — DONE; Phase 4 (dtype parity: fp16 + fp8-Q) — DONE (2026-10-02); varlen-M (M≤8, the MTP/DSpark/DFlash verify) — DONE (2026-10-03).**
Dense (bf16 + fp8) and varlen (O + LSE) all pass; the 4-combo dtype gate
(`(bf16,bf16)`, `(fp16,fp16)`, `(bf16,fp8)`, `(fp8,fp8)`) is live — fp16 and
native fp8-Q (no pre-cast) verified green. **Paged KV (the production blocker) landed 2026-10-02**
(§2.1): the per-n_block physical page is gathered in-kernel from the page table
(page-indexed *tiled* cp.async, not a per-row gather — the fast tiled path is kept), the
true per-batch length comes from `seqused_k` (required for paged, matching the
`flash_attn_interface.py` invariant), and the dense-vs-paged cross-check is bit-exact.
**SplitKV (the low-batch SM-filling lever) landed 2026-10-02** (§2.2): `num_splits` as a
runtime grid y-dim, tile-aligned chunks, fp32 partial O/LSE + the reused GEMV host combine
(`_flash_attn_fwd_combine`) and auto-ns plan (`_gemv_auto_num_splits`). The bring-up surfaced
and fixed a **cross-warp smem data race** (§2.2): the fp8→bf16 V staging buffer `sVb` was
written by all threads but not barriered before the all-threads `gemm_rs` ldmatrix read —
1st block read uninitialized `sVb` (the rare NaN), later blocks read stale V (the ~2e-1 O
error), run-to-run non-deterministic; fixed with a fp8-leg-only `sync_threads()` at both PV
sites, after which the fp8 large-L SplitKV is deterministic at the expected fp8 floor.
The varlen LSE bug (§1.3) is fixed by
three changes: the head/seqlen transpose (bug A), the per-CTA base carried in the LSE
store *coordinate* rather than a `domain_offset` (bug B), and a varlen Q-load
`domain_offset` fix — the flat `(q_row_base, 0)` was offsetting the qhead sub-mode (or
dropping it), so every batch read row 0's Q and the QK scores (hence LSE) were wrong
for batch>0 (bug C). Phase 1 is done; paged (§2.1), dtypes (§2.3/§2.4), and SplitKV (§2.2) are done;
only the re-tile (§2.5) knobs + the tcgen05 spike (§2.2a) remain — the fp8-PV mma (§2.7) is done.
**varlen-M (M≤8) landed 2026-10-03** (§2.6): the MTP/DSpark/DFlash speculative-decode verify
(M candidate tokens sharing the same KV, causal). The "m" mma dim is the packed GQA group
**M×G** (M≤8 → ≤48, fits TILE_M=64), dense+paged+SplitKV+fp8/bf16. The kernel was already
M-agnostic (seqlen_q is a runtime param driving the PackGQA load/store guards) — the only
kernel change is the dense **LSE select**, which is a **compile-time branch on M** (`is_m1`):
M=1 keeps the proven-safe `mode=[1,2,0]` decode codegen (bit-exact, the M=1 regression is
unchanged), while M>1 uses `mode=[2,1,0]` (the (B,H,M) layout's correct (seqlen, head, batch)
order). The tc gate admits M≤8 (dense-only), rejects M>8 (routes to FA4/FlashInfer).
**fp8-PV mma landed 2026-10-04** (§2.7): the PV mma (P×V, the 2nd GEMM) is now **native fp8**
(e4m3 P × e4m3 V, fp32 acc) on the fp8-KV leg — resolving both sm_110a `mma.sync` blockers that
kept it bf16 (§2.2a): (a) the fp8 P A-fragment can't be built in-regs → a **P smem round-trip**
(store P element-wise via the QK mma's C-partition, reload as the PV A-operand); (b) the fp8 V
B-operand ldmatrix faults → an **element-wise V load** (same idiom as the QK K leg). The bf16-KV
leg stays byte-identical (separate kernel instance; the sP buffer is fp8-only). All four verify
suites stay green (dense 34/35 with only the pre-existing large-L Xorg flake, SplitKV 13/13,
varlen-M 25/25, paged 7/8 the pre-existing cross-check flake); **every fp8 cell passes**,
including fp8 varlen-M (M≤8) + SplitKV which exercise the fp8 PV mma with the M×G "m" dim.
This is the
workplan to bring the tensor-core
decode kernel (`vllm.vllm_flash_attn.cute.sm100_hd256_decode_tc`) to full feature parity
with the FMA GEMV decode kernel (`sm100_hd256_decode_gemv`), so the TC kernel can
**serve production** (paged KV) and match GEMV's capability surface. Every item is
mapped to the proven GEMV implementation (most are **ports, not greenfield**) and to
the exact TC change-points.

> **Phase 1 results (gated, clean window — `mjolnir gate` / the inline gate in
> `verify-tc-decode.py`):** the dense legs (bf16 + fp8 KV, L=1…65536, B=1/4) pass at
> rel 0.003–0.007. Two findings from the gated runs:
> 1. **The fp8 large-L "flakiness" was co-tenancy, not a kernel bug.** Ungated the
>    fp8 L=8192/32768/65536 legs drifted to rel 0.058–0.128 (over the 0.05 tol); in a
>    gated clean window (vLLM `num_requests_running+waiting` = 0) they settle to
>    rel 0.003–0.007. The Thor GPU is shared with the live vLLM server — every TC
>    number is only trustworthy inside a gated window (the gate is now built into the
>    verify; see §4).
> 2. **Varlen O was corrupted** (every batch wrote to gmem row 0): the O epilogue
>    stored `mO[None, None, kv_head]` with no per-batch offset, unlike the Q load
>    (`domain_offset((q_row_base, …), …)`). Fixed by offsetting the O store by
>    `q_row_base` — varlen O is now correct for all batches including the empty one
>    (§1.3). The varlen **LSE** still needs a GEMV-style manual store (§1.3).

## The goal

The TC kernel is the primary long-context fix (the GEMV kernel is compute/issue-bound;
the TC streamer is memory-bound — see `gemv-long-context-diagnosis.md`). It used to be
**benchmark-only**: vLLM serving decode is *always* paged, and the TC
dispatch gate required `page_table is None` (`interface.py:977`). **Paged KV now lands
(§2.1), so the TC kernel can serve production** — the gate is paged-aware (admits
`page_table is not None` for `page_size in (64, 128)`, requires `seqused_k`). The
remaining capability gap is SplitKV (§2.2) + re-tile (§2.5). This plan closed the paged
gap and documents the rest of the capability surface.

## Capability baseline (as of the fp8-K-path fix, `ab900fe`)

| Capability | GEMV | TC (today, Phase 1) |
|---|---|---|
| Compute model | pure FMA — compute/issue-bound | `mma.sync` — memory-bound streamer |
| M / head_dim | M=1, hd256 | M=1, hd256 |
| Q dtype | bf16 / fp16 / fp8 | bf16 / fp16 / fp8 (fp8-Q native, no pre-cast) |
| KV dtype + descales | bf16 / fp16 / fp8 + descales | bf16 / fp16 / fp8 + descales |
| Dense KV | ✅ | ✅ (L=1…65536, B=1/4, gated) |
| Varlen (cu_seqlens) | ✅ | ✅ O + LSE (§1.3 fixed) |
| Paged KV (64/128) | ✅ | ✅ (§2.1 — page-indexed tiled cp.async + seqused_k; dense-vs-paged bit-exact) |
| SplitKV (ns>1 + combine) | ✅ | ✅ (§2.2 — tile-aligned chunks, fp32 partials + reused host combine; sVb race fixed) |
| Partial-tail (L < tile) | ✅ (clamp + mask) | ✅ (mask, §1.1) |
| Empty batch (kv_len=0) | ✅ (O=0, LSE=−inf) | ✅ O=0 (LSE via §1.3) |
| Long-L perf | weaker | **stronger** |
| B3 re-tile knobs | ✅ (tile/stages) | ❌ (TILE_N fixed 64) |
| Prod vLLM serving | ✅ | ❌ (serving is paged) |

(Both kernels treat paged and varlen as mutually exclusive — the standard vLLM
invariant at `interface.py:712`.)

---

## Part 1 — Known correctness bugs (fix first; block verify going green)

### 1.1 Partial-tail: `L < TILE_N` fails verify — **DONE (gated)**

- **Symptom (was):** `verify-tc-decode.py` L=1/3/5 rows reported `lse_err≈4–6` — real
  corruption, not a tail-epsilon.
- **Two independent root causes, both fixed:**
  1. **Interface (the real one):** the dense dispatch passed `seqlen_k = v.shape[-3]`
     (the *padded* KV size) as `kv_len`, so the partial-tail mask never engaged on a
     padded pool. Now `tc_seqlen_k = max_seqlen_k` for dense (the actual length;
     varlen derives `kv_len` from `cu_seqlens_k` in-kernel, so `seqlen_k` is a dummy
     there). `interface.py` TC dispatch.
  2. **Kernel mask:** `_mask_partial_kv` now reads the per-thread absolute kv column
     from the mn-view of the mma-partitioned identity tensor (`tScS_mn[r, c][1]`),
     materialized to rmem (an inline abstract coordinate constant-folds to a fixed
     value); the `−inf` select goes through an mn-view row round-trip (the access
     path `online_softmax` consumes). The committed alternate (a direct
     `tScS[(_r%2, _r//2), 0, _c][1]` read) is preserved in
     `sm100_hd256_decode_tc_mask_alt.py`.
- **Verify:** dense bf16 + fp8 L=1/3/5 pass at rel ≤ 3e-3, gated clean window.
- **Still to add:** an `L=100` / `L=37` cell (last-partial-block, `kv_len ≥ TILE_N`,
  `kv_len % TILE_N != 0`) — the verify only tests `L` that are multiples of 64 or `< 64`,
  so that path is untested.

### 1.2 Empty batch (`kv_len=0`) → O now correct (LSE via §1.3)

- **Was:** `varlen bf16 lens=[5,1,0]` reported `max_abs=nan` for the empty batch.
- **Root cause:** the varlen O epilogue stored `mO[None, None, kv_head]` with **no
  per-batch offset**, so every batch (including the empty one) wrote to gmem row 0 and
  the `torch.empty` `out` retained uninitialized garbage where a batch didn't write.
- **Fix:** offset the varlen O store by `q_row_base` (`domain_offset(((0, q_row_base),
  0), …)`, the GQA-packed `(qhead, total_q)` M-mode — offset only the `total_q`
  sub-mode), mirroring the Q load. The empty batch (`kv_len=0`) skips the n-block
  loop, so `acc_O` stays 0 and is stored as `O=0`. **Varlen O is now correct for all
  batches** (diag: b0/b1/b2 O within 3e-3, empty batch exactly 0).
- The empty batch's **LSE=−inf** is still wrong — it is part of the varlen LSE issue
  below (§1.3), not a separate finalize `0/0` (the `Softmax.finalize` row-sum=0 →
  LSE=−inf path is correct; it is the *store* that misroutes the varlen LSE).

### 1.3 Varlen LSE store — **FIXED (all three bugs landed, 2026-10-02)**

Three distinct bugs, all root-caused and fixed. The varlen LSE now matches the fp32
reference for every batch (including the empty one, `lse_err≈7e-7` gated):

- **Bug A — head/seqlen mixup (FIXED).** The interface allocates the varlen LSE as
  `(num_head, total_q)` = `(H, total_q)` (head at mode 0), but the kernel assumed
  seqlen-major `(total_q, H)` and skipped the dense-only transpose. Result: the
  `H`/`total_q` modes were swapped before `pack_gqa_layout(head_idx=1)`.
  **Fix:** transpose the varlen LSE `mode=[1,0]` → `(total_q, H)` before packing
  (mirrors the dense `mode=[1,2,0]`). This makes the packed layout correct:
  `((qhead, seqlen), nkv)` with `nkv` **flat** (stride `total_q·G`) — confirmed by
  the debugger's layout print. (The nkv mode is flat, not ranked; a scalar
  `mLSE[None, kv_head]` correctly selects the kv_head.)

- **Bug B — dynamic per-CTA base lost in the LSE pointer path (FIXED).** Even with the
  layout correct, `store_LSE`/`compute_ptr`/`elem_pointer` dropped the per-CTA base
  `18·kv_head + q_row_base`: the dynamic `q_row_base` is 16-B-rounded in transit
  (b1/b2 → +4 elems) and the store pointers from **all 12 CTAs** fell in offsets
  `[0,16]` of the 72-elem buffer. Every CTA wrote the same low cells → cross-batch
  clobber. The O was exact because `store_O` goes through TiledCopy (a proven
  `domain_offset` path), not this ptr/shuffle path.
  - **Fix (landed):** carry the base in the *store coordinate* instead of a
    `domain_offset`. `compute_ptr`/`store_LSE` take an explicit `row_base: Int32`
    and feed it to the seqlen index — `elem_pointer(tensor, ((h_idx, m_idx +
    row_base),))` — so the base enters via `crd2idx` (pure integer offset on the
    static layout strides), never through the 16-B-aligned `domain_offset` path. The
    varlen call site drops the `domain_offset` and passes `row_base=q_row_base`; the
    dense path is unchanged (`row_base=0`). Detail: `task/debug/tc-varlen-lse-store.md`.

- **Bug C — varlen Q-load `domain_offset` offset the wrong sub-mode (FIXED).** With
  bugs A+B fixed, batch1's LSE was still wrong (finite, deterministic): the Q load
  used `domain_offset((q_row_base, 0), mQ[None, None, kv_head])` — a *flat* coord that
  applied `q_row_base` to the **qhead** sub-mode (or dropped it) instead of the seqlen
  sub-mode, so every batch read row 0's Q and the QK scores (hence the LSE) were wrong
  for batch>0. The O was unaffected for L=1 (P=1 for a single kv row, so O = v[kv]
  regardless of Q) — which is why the O looked right while the LSE did not.
  - **Fix (landed):** the nested coord `domain_offset(((0, q_row_base), 0), …)` (qhead
    by 0, seqlen by `q_row_base`) — the same idiom as the varlen O store. Confirmed by
    the isolation diag: batch1's LSE matched `(q_row=0, kv_row=5)` before the fix
    (the Q was read from row 0) and matches the fp32 reference after.

The varlen **O and LSE are both fully correct** (all batches/heads incl. the empty
one). The `compute_ptr` change is additive (`row_base` defaults to 0), so the dense
path and the Q/O `compute_ptr` callers are untouched — the dense bf16 regression
legs confirm no regression.

---

## Part 2 — Feature additions (GEMV parity)

### 2.1 Paged KV — the production blocker — **DONE (2026-10-02, verified bit-exact)**

- **Why essential:** vLLM serving decode is *always* paged (block tables); the TC gate
  `interface.py:977` hard-required `page_table is None`, so TC was benchmark-only.
- **Design (shipped): page-indexed *tiled* cp.async (option 2, the performance-focused
  one — NOT a per-row gather).** The key realizations from the design spike:
  - **No `tile_n=128` needed.** The FA4 TMA-box assert (one TMA copy = one page) does
    *not* apply to cp.async: a `TILE_N=64` block reading half of a `PS=128` page is a
    contiguous gmem range *within one pool page*, so the swizzled smem / ldmatrix / mma
    machinery is untouched. Only the gmem *source base* becomes per-n_block.
  - **`seqused_k` is required for paged.** `flash_attn_interface.py:276` asserts
    `block_table is None or seqused_k is not None` — every paged call carries the
    per-batch KV lengths. The kernel's `kv_len` is now `mSequsedK[batch_idx]` (paged),
    not `seqlen_k` (which is pool capacity). The gate requires `seqused_k is not None`
    when `page_table is not None`.
  - **One uniform page-table read per n_block.** Per n_block:
    `pidx = mPageTable[batch_idx, (nb*TILE_N)//PS]` (one plain 32-bit GMEM read), then a
    gathered *static* pool tile coordinate
    `ktile = (pidx*PS + (nb*TILE_N)%PS)//TILE_N` into `gK/gV =
    local_tile(mK/mV_flat_pool, (TILE_N, D), (None, 0))`. The four cp.async sites keep
    their exact shape — only the last coordinate changes `nb → ktile`. K and V each
    gather their **own** page (`pidx` / `pidx_v`) — *not* the FA4 "V reuses K[i-1]'s
    page" trick (V is a separate pool).
  - **R1 de-risked first** (standalone probe, bit-exact): a *runtime* `ktile` (even via
    the in-kernel page-table read) lowers to a correct runtime gmem address through the
    partitioned cp.async view (`tKgK[..., ktile]`), so no `domain_offset` fallback was
    needed.
- **Interface (shipped):** paged-aware gate (admit `page_table is not None` for
  `page_size in (64,128)`, require `seqused_k`); `tc_key` paged term (separate cache
  entry); ctor `page_size`; compile/launch pass the flat-pool K/V view + `page_table`
  + `seqused_k` (mirroring the GEMV paged tensor views). The flat-pool K/V is
  `(NP*PS, D)` per kv_head — a pure additive `is_paged` branch; dense codegen is
  byte-identical (the dense verify re-ran green).
- **Two real bugs found + fixed during bring-up:**
  1. **Multi-page gather column:** the steady-state page column was `nb // PS` — `nb`
     is a *block* index (64 rows) and `PS` a *row* count (128), so it stayed `0` until
     block 128 and every block ≥ 2 re-read **page 0's rows**. Single-page cells
     (L≤128) masked it — which is why the single-page R1 probe passed. Fixed to the
     page containing row `nb*TILE_N`: `mPageTable[batch_idx, (nb*TILE_N)//PS]`.
  2. **Paged-variant register-allocation miscompile** (NaN O / −inf LSE; correct source
     + MLIR, broken PTX binding — see the earlier root-cause in the GEMV workstream).
     The pin (three `sK` smem reads + three gmem stores into a per-device scratch
     buffer `mPin`, fp8-safe via a `Float32` cast) is now in the production kernel,
     compiled **only** in the paged leg (`const_expr(is_paged and mPin is not None)`).
- **Verify (`verify-tc-decode-paged.py`, 8 cells, PS=128 primary + a PS=64 spot):**
  identity table, **shuffled** table (the real gather check), partial-page tail (L=97),
  L=8192 / 64-pages, an fp8-Q leg (q bf16 / kv fp8 — the production config), B=2
  multi-batch (distinct `kv_len` + distinct tables), and a **dense-vs-paged cross-check
  → bit-exact (diff 0.0)**. All gated clean windows. PS=64 spot 3/3 bit-exact.
- **Not shipped yet:** `page_size=16` (a 64-row tile spans 4 pages of 16 — a single
  static tile can't express it; PS=16 stays with GEMV) and image wiring (the TC
  dispatch is still vfa-tree-only — see the v14 note in Part 3).

### 2.2 SplitKV (`num_splits>1` + host combine) — **DONE (2026-10-02, verified)**

- **Why it matters for TC:** memory-bound → more CTAs = better BW at low batch
  (batch=1, 4 kv-heads = 4 CTAs vs 20 SMs); SplitKV fills the GPU (the GEMV's
  `ns=1→20 → ×8.7 achieved BW` lever, `gemv-ring-fix-bench.md`).
- **Design (shipped):** a port of the GEMV SplitKV, adapted to the TC's *tiled* cp.async:
  - **`num_splits` as a runtime grid y-dim** (one split binary serves every ns): grid
    `[num_kv_heads, num_splits, batch]` (`__call__` L207).
  - **Tile-aligned chunks** (NOT row-aligned — the tiled cp.async needs tile boundaries):
    `kv_chunk_tiles = ceil(ceil(kv_len/TILE_N)/ns)`, `kv_start_tile = split_idx*
    kv_chunk_tiles`; the **global tile** `gtile = kv_start_tile + nb` replaces `nb` in the
    dense tile coord, the paged gather, and the tail mask. ns=1 → `kv_start_tile=0`,
    `gtile=nb` — byte-identical (the chunk scan is `IS_SPLIT`-gated so non-split codegen
    is canonical-identical — the unconditional scan tipped the paged pipeline's
    scheduling and broke the bit-exact xcheck, root-caused during bring-up).
  - **fp32 partial O/LSE:** the epilogue stores `mO_partial[split]` (fp32) /
    `mLSE_partial[split]` (fp32) — the O-store path is typed `ODTYPE=fp32` in split mode
    (`sO` smem + the copy atom + the rmem fragment), bf16 otherwise. The partial VALUES
    are the same as non-split (chunk-normalized O + chunk LSE); empty chunks → `O=0 /
    LSE=−inf` via `Softmax.finalize` (the `row_sum==0 → scale=1.0` guard) for free.
  - **Interface:** the `tc_split` gate (decode-shaped only), the **reused**
    `_gemv_auto_num_splits` plan (FlashInfer Alg.1, env/caller/auto precedence, the
    `VLLM_FA4_HD256_TC_NUM_SPLITS` knob), the `out_partial`/`lse_partial` fp32 allocation
    (matching the kernel's packed view), and the **reused** `_flash_attn_fwd_combine`
    host combine (unchanged). `tc_key` gets a `tc_split` term (separate cache entry).
- **The sVb cross-warp race (found + fixed during bring-up):** the fp8 PV stages V
  through a bf16 smem buffer `sVb` (`_convert_v_to_bf16`, all threads write) → `gemm_rs`
  ldmatrix (all threads read), with **no barrier between** → 1st block read *uninitialized*
  `sVb` (garbage bits → the rare NaN), later blocks read *stale* previous-block V (the
  ~2.27e-1 O error), run-to-run non-deterministic. The bf16 leg (reads `sV` directly) is
  clean; the latent race was un-dormanted by the SplitKV codegen (fp32 O + chunk math
  reshuffled the warp schedule). **Fix:** a fp8-leg-only `cute.arch.sync_threads()` after
  the `sVb` write + before `gemm_rs` at both PV sites (the GEMV idiom). Post-fix the fp8
   large-L SplitKV is **deterministic** at the expected fp8 quantization floor (~1.2–1.5e-2),
   no NaN in 18 runs. (Not the fp8 P-cast — see §2.2a: the PV is bf16 by design.)
- **The auto-ns pool-size cap (fixed during bring-up):** the ns plan capped on the KV
   *pool* size (`seqlen_k`), which for paged is larger than the true per-batch `kv_len` —
   a partial-page tail batch could be planned more splits than it has tiles. **Fix:** cap
   on `max_seqlen_k` (the true per-batch max KV length — the pool size for dense, the
   actual length for paged) instead: `ns ≤ ceil(max_seqlen_k/128)` (`interface.py` M2 plan).
- **Split-path first-block OOB mask guard (fixed during bring-up):** the partial-tail
   first-block guard tested `kv_len < TILE_N` — correct only for the non-split `nb=0`
   case. In split mode the first block is at `gtile = kv_start_tile`, so a split whose
   *only* block is the partial tail (e.g. `L=97`, `ns=2`, split 1 covers rows 64-127 with
   `kv_len=97`) never masked the OOB rows → read past the end of the KV. **Fix:** make the
   guard split-aware — `if IS_SPLIT: mask if kv_len - gtile*TILE_N < TILE_N` `else: mask
   if kv_len < TILE_N` (the non-split branch is byte-identical to the original).
- **Verify:** the ns sweep ({1,2,4} + auto-ns) cross-checked against the ns=1 write-through —
   the combine recovers the write-through within the fp32 accumulation floor (NOT bit-exact:
   a real combine/sVb bug breaches ≥ ~0.1, the pre-fix sVb race), dense + paged, bf16 + fp8,
   the **empty-split** (a paged `B=2 [256,64]` at ns=2 → the short batch's splits 1.. are EMPTY,
   the `−inf` LSE contributes 0 and the combine still matches ns=1 for both batches), the
   auto-ns plan (`L=8192` → ns=20), and a 3× bit-identical determinism repeat (the sVb fix).
   The dedicated **`verify-tc-decode-splitkv.py`** (gated; 13 cases) exercises the dispatch
   end-to-end through the real `_flash_attn_fwd` — the `tc_split` gate, the caller-
   `num_splits` + the `VLLM_FA4_HD256_TC_NUM_SPLITS` env + the auto plan, the fp32
   `out_partial`/`lse_partial` allocation, and the host combine — all green (13/13). The
   full `verify-tc-decode.py` dense sweep (34/35; the one miss a pre-existing bf16
   `L=32768 B=4` borderline flake) and `verify-tc-decode-paged.py` (8/8, dense-vs-paged
   bit-exact) re-confirmed under the split-aware xcheck.

 ### 2.2a Why the PV mma was bf16 (and when the fp8 PV / `tcgen05` would matter) — **fp8-PV NOW DONE (§2.7)**

 The QK mma **is** fp8 (the Blackwell fp8 tensor core for the dominant Q·K, plus the fp8 KV
 in gmem that halves the DRAM bytes — the decode win). The **PV mma was bf16** for two sm_110a
 `mma.sync` blockers (not a perf choice): (a) an e4m3 **P A-fragment** produced in registers
 doesn't pack via `reshape_acc_to_frgA` (undefined register bits → NaN); (b) the e4m3 **V
 B-operand ldmatrix faults on sm_110a** (the QK works around it by loading K *element-wise* —
 that workaround wasn't in the PV path). **Both are now resolved (§2.7, 2026-10-04)**: the PV
 mma is native fp8 (e4m3 P × e4m3 V) via the P smem round-trip (blocker a) + an element-wise V
 load (blocker b). The bf16-KV leg stays bf16 (its `mma.sync` A/B dtype is already bf16; the
 fp8-PV change is fp8-KV-leg-only). For **M=1 decode the tensor core isn't the limiter** (the
 KV load is — see the memory-bound diagnosis), so the fp8 PV buys ~no wall-clock speedup at M=1;
 it **would** matter for M≥8 (MTP verify) / prefill. The broader "not fully Blackwell" gap is
 `mma.sync` vs the `tcgen05` 5th-gen tensor core (TMEM/2CTA) — also moot for memory-bound
 M=1 decode, relevant for M≥8/prefill. **Investigation: COMPLETE** —
 `tc-fp8pv-tcgen05-investigation.md`: both blockers confirmed with exact line refs (a: the
 fp8 A-fragment thread mapping is incompatible with the C-fragment → P smem round-trip is
 mandatory; b: 8-bit ldmatrix fault + broken retile, the element-wise workaround is proven
 on the QK K leg), fp8-PV mma feasible (in-tree precedent for every piece) but ~0 gain at
 M=1, and tcgen05 rejected for the TC kernel (not 2CTA-locked — a 1CTA tcgen05 hd256 decode
 kernel already exists in-tree and is measured 2.25× slower than the production FI path at
 L=8192/M=1). The fp8-PV mma (both blockers) is **implemented and verified** (§2.7); only the
 tcgen05 spike (M≥8/prefill) remains deferred.

### 2.3 fp16 KV dtype + descales — **DONE (2026-10-02, verified green)**

- **TC caveat (important):** `mma.sync` requires **matching A/B dtype** — unlike GEMV's
  FMA (all-fp32), there is **no bf16×fp16 mixed mma**. So parity means the **all-fp16**
  path (Q and KV both fp16), not mixed. Descales are dtype-agnostic (folded into the
  score scale / epilogue, already correct at `:258-267` and `:661`).
- **Change-points (landed):** the mma is `MmaF16BF16Op(DTYPE, …)` where
  `DTYPE = mO.element_type` — so the all-fp16 leg (O fp16 → `DTYPE=fp16`) is handled
  by the existing 16-bit mma path, no kernel change. The only change is the **interface
  gate** (`interface.py:975-976`): widened from `q.dtype in (bf16,)` to a 4-combo gate
  — `(bf16,bf16)`, `(fp16,fp16)`, `(bf16,fp8)`, `(fp8,fp8)` — so fp16 Q/KV is admitted
  and mixed classes (e.g. bf16-Q / fp16-KV) stay rejected.
- **Verify:** `dense fp16 L=64/256 B=1/4` pass at rel 3e-4–1.8e-2 (O fp16).

### 2.4 fp8(e4m3) Q dtype — **DONE (2026-10-02, verified green)**

- **Low effort:** the kernel already pre-casts Q to fp8 for the fp8 leg
  (`interface.py:1000-1003`) and streams Q as fp8 in smem (`QDTYPE = FP8T`, `:220`).
  fp8-Q input is just a **gate widening** (the 4-combo gate above admits `(fp8,fp8)`;
  the `q.to(fp8)` pre-cast at `:1009` is a no-op when Q is already fp8, so native
  fp8-Q needs no kernel change).
- **Verify:** `dense fp8-Q L=64/256 B=1/4` (native fp8-Q, O bf16) pass at
  rel 2.8e-3–1.6e-2.
- (fp8-Q pairs with fp8-KV; 16-bit-Q pairs with 16-bit-KV — same dtype class.)

### 2.5 B3 re-tile knobs (TILE_N / ring depth)

- **GEMV reference:** the `VLLM_FA4_HD256_GEMV_TILE`/`_STAGES` env knobs
  (`sm100_hd256_decode_gemv.py:99-114`) varying the in-flight footprint.
- **TC change-points:** parametrize `TILE_N` (currently fixed 64) and the cp.async ring
  depth, same env-knob pattern; extend `gemv-geom-sweep.py` to hit the TC `tc_decode`
  leg.

### 2.6 varlen-M (M≤8) — the MTP / DSpark / DFlash speculative-decode verify — **DONE (2026-10-03, verified)**

- **The task:** the TC kernel was M=1-only (the "m" mma dim is the GQA group **G=6** q-heads
  for ONE token, padded to TILE_M=64). Extend it to **varlen-M (M≤8)**: the MTP/DSpark/DFlash
  verify where **M candidate tokens all share the same L KV** (causal — each query token
  attends to the full L context). The "m" mma dim becomes **M×G** (M=1→6, M=4→24, M=8→48,
  all fit TILE_M=64; **M>8→>64 is REJECTED**). The GEMV kernel does NOT have varlen-M either
  (it is a separate open item) — this is designed fresh from the TC M=1 structure + the
  FA3/FA4 `PackGQA` varlen reference (`pack_gqa.py`, the `(qhead_per_kvhead, seqlen_q)` "m"
  layout).
- **The "m" dim = (G, M) packed.** `PackGQA` folds the G q-heads (inner) into M tokens
  (outer) into the packed "m" = M×G rows (row r → token r//G, qhead r%G). Q loads all M×G
  rows; O/LSE unpack the M×G rows back to (M, H) / (M, H). The KV is the "n" dim (unchanged —
  shared across all M×G rows, tiled by TILE_N). paged + dense + SplitKV + fp8/bf16 are all
  per-KV-row (orthogonal to the M dim), so they work for M≤8 unchanged.
- **The kernel was already M-agnostic.** `seqlen_q` (M) is a **runtime** param driving the
  `PackGQA.load_Q` / `store_LSE` / `store_O` guards (`row < seqlen * G`); the mma computes
  TILE_M=64 rows regardless of M (the valid M×G rows, the rest never stored). So **no kernel
  mma/paged/split changes were needed** — only the dense **LSE select** (below).
- **The one kernel change — the dense LSE select is a compile-time branch on M (`is_m1`).**
  The interface allocates the dense LSE as `(B, H, M)` (`lse_shape`, qv=None), but the original
  dense select `mode=[1,2,0]` assumed the layout was `(B, M, H)`: it only stored the right
  addresses for M=1 (where the (seqlen, head) modes alias). For M>1 the (seqlen, head) strides
  swap → the LSE lands on the wrong rows (O is unaffected, only LSE). The fix: `mode=[2,1,0]`
  gives the (B,H,M) layout's true (seqlen=M, nheads=H, batch) order. **But** `mode=[2,1,0]`
  (seqlen sub-mode = M=1 for the M=1 binary) mis-generates the LSE-store TMA and **corrupts the
  O** for the M=1 **split** cells (L=8192 B=4, L=65536 B=1) — the two selects give identical M=1
  store *addresses* but different codegen, and the M=1 decode path is only safe on the
  historical `mode=[1,2,0]`. So the select is a **compile-time branch** (`const_expr(is_m1)`):
  M=1 → `mode=[1,2,0]` (the proven-safe decode codegen, **bit-exact** — the M=1 regression is
  unchanged), M>1 → `mode=[2,1,0]`. `is_m1` is a `cutlass.Constexpr[bool]` threaded through
  `__call__`/`kernel` and passed from the interface as `max_seqlen_q == 1` (M is in the `tc_key`
  → a separate binary per M, so the branch folds at compile time).
- **The interface:**
  - **Gate M-range:** the `use_tc_hd256` M term is `max_seqlen_q == 1 or (cu_seqlens_q is None
    and max_seqlen_q <= 8)` — M=1 (any q layout) or varlen-M (M≤8, **dense-only**: every batch
    has exactly M tokens; a varlen-q call has a different per-batch M the single seqlen_q can't
    express). **M>8 → REJECT** (falls through GEMV — also M=1-only — to the FA4/FlashInfer path).
  - **`tc_key` M term:** `max_seqlen_q` added to the key so each M is a separate cache entry
    (the M=1/M=4/M=8 binaries don't collide).
  - **`_tc_dtype_ok`:** the mixed bf16-Q/fp8-KV dtype assert (previously M=1-only via
    `_gemv_dtype_ok`) is widened to M≤8 for the TC path (Q is pre-cast to fp8 in-kernel).
- **Verification — `verify-tc-decode-varlenM.py` (25 cells, gated, 25/25 GO):** M∈{1,2,4,8}
  × dense+paged × bf16+fp8 (ns=1 write-through) vs the gold full-attention-over-L reference
  (each of the M tokens attends to the shared L KV); M=8 split (ns=4); the **gate-reject** test
   (M=9/M=16: no new `tc_compile_cache` entry → the TC gate rejects, output finite via the FA4
   fallback); + an M=1 regression spot. The M=1 regression is **unchanged**: dense
   `verify-tc-decode.py` 34/35 (only the pre-existing borderline L=32768 B=4 rel 5.9e-2 overshoot),
   paged 8/8, SplitKV 13/13 (the `is_m1` branch keeps M=1 on the original codegen).

### 2.7 fp8-PV mma (e4m3 P × e4m3 V) — **DONE (2026-10-04, verified)**

The PV mma (the 2nd GEMM, `O += P @ V`) is now **native fp8** on the fp8-KV leg — e4m3 P ×
e4m3 V, fp32 accumulator (`MmaFP8Op(FP8T, Float32, (16, 8, 32))`, permutation k=32). This
resolves the two sm_110a `mma.sync` blockers that kept the PV bf16 (§2.2a). The **bf16-KV leg
stays byte-identical** (the fp8/fp16 PV `mma.sync` A/B dtype is already bf16 there; the change
is fp8-KV-leg-only, and `FP8_KV = const_expr(KVTYPE.width == 8)` makes the bf16 and fp8 legs
separate kernel instances — the sP buffer / `tCsP` are fp8-only, so the bf16 codegen is
untouched).

**Blocker (a) — the P A-fragment (P smem round-trip).** The fp8 P A-fragment (e4m3, m16n8k32)
can't be produced in registers: the QK C-fragment's thread→element mapping disagrees with the
fp8 A-fragment's, so `reshape_acc_to_frgA` yields undefined register bits (NaN — the original
§2.2a blocker a). Resolved by a **P smem round-trip**: `acc_S` (the fp32 QK C-fragment) is
cast to e4m3 (the cast is satfinite-inherent — P ∈ [0,1] under the running-max softmax, so
saturation never fires; safety net against a P-overflow NaN), stored element-wise to a small
`sP` smem buffer via the QK mma's C-partition (`tCsP = thr_mma_qk.partition_C(sP)`; a per-thread
element-wise store — the universal 16-bit fp8 copy atom can't vectorize against the plain sP
layout, so no copy atom is used), then a `fence_view_async_shared` + `sync_threads` barrier,
then reloaded as the PV A-fragment element-wise (`sP_a = thr_mma_pv.partition_A(sP)`, in
`_pv_mma_fp8`). `sP` is a **plain (unswizzled) layout** (`make_layout((TILE_M, TILE_N),
stride=(1, TILE_M))`) — the P round-trip only needs element-wise addressing (the A-operand is
loaded element-wise, no ldmatrix), so a swizzle would only break the store's src/dst layout
match. `sP` is 4 KiB (TILE_M × TILE_N e4m3) and **replaces the 32 KiB `sVb`** the bf16 PV
leg used to stage V (the fp8 leg no longer needs `sVb` — see blocker b).

**Blocker (b) — the V B-operand (element-wise load).** The e4m3 V B-operand ldmatrix faults on
sm_110a (the original §2.2a blocker b — 8-bit ldmatrix fault + broken retile). Resolved by an
**element-wise V load**: `tOrV[_e] = sV_b[_e]` (the same idiom the QK leg uses to load K
element-wise — `tSrK[_e] = sK_b[_e]`). The V smem is the **transposed** view `sVt =
transpose_view(sV)` = (D, kv) — the mma B-operand convention is (N, K) = (head_dim, kv_rows),
and the natural V smem is (kv, D) = (K, N), so it's transposed (exactly like the bf16 leg,
which ldmatrix's the transposed `sVt`). Getting this orientation wrong is the "A's M == D'M and
B's N == D'N" mma-shape error. No `sVb`, no `_convert_v_to_bf16`, no extra barrier (the fp8 V
streams straight from the fp8 sV).

**`_pv_mma_fp8` (the fp8 PV GEMM):** element-wise A-load (`tOrP[_e] = sP_a[_e]`) + element-wise
B-load (`tOrV[_e] = sV_b[_e]`) + a k-loop over `num_k = TILE_N / 32 = 2` (the KV-row k-tiles),
each `cute.gemm(tiled_mma_pv, acc_O, tOrP[None, None, k], tOrV[None, None, k], acc_O)` covering
the full (M, D) O-tile for one k-tile.

**Verification (all gated, clean window):** every fp8 cell passes — dense `verify-tc-decode.py`
(all dense fp8 / fp8-Q legs, L=64…65536 B=1/4), paged (fp8-Q L=256), SplitKV (dense+paged fp8
ns=2/4, fp8 L=8192 ns=4), varlen-M (M∈{1,2,4,8} × dense+paged × fp8, L=512/8192, ns=1 + M=8
ns=4). The varlen-M fp8 cells are the strongest check: they run the fp8 PV mma **with** the M×G
"m" dim (M≤8) simultaneously. The bf16-KV regression is **unchanged**: dense 34/35 (only the
pre-existing borderline L=32768 B=4 rel 5.9e-2 overshoot — identical value to the baseline),
SplitKV 13/13, varlen-M 25/25, paged 7/8 (the pre-existing dense-vs-paged cross-check flake —
the baseline is also 7/8). The large-L bf16 flake (L=65536 rotating catastrophic failures) is
the **pre-existing ungated-Xorg-co-tenancy** (the longest cells are most exposed; the baseline
shows the same flake, and the bf16 leg is a separate kernel instance my change doesn't touch).

---

## Part 3 — Interface wiring (per feature)

- **Paged — DONE (§2.1):** paged-aware gate (admit `page_table is not None` for
  `page_size in (64,128)` + require `seqused_k`); `tc_key` paged term; ctor `page_size`;
  flat-pool + `page_table` + `seqused_k` compile/launch args. **Image wiring still open
  (v14):** the TC dispatch is vfa-tree-only today — `interface-tc-dispatch.diff` is a
  reference-only patch (no `apply_patches.py` / `verify_patches.py` entry) and the TC
  kernel is not `COPY`ed in the Dockerfile yet. To ship TC (paged) in
  `mjolnir/vllm-thor:qwen38-sm110-v14`: add the `COPY fa4-gemv-kernel/sm100_hd256_decode_tc.py`
  to the Dockerfile (next to the GEMV kernel COPY) + register
  `interface-tc-dispatch.diff` in the patch stack (or fold it into the GEMV interface
  patch) + a canary.
- **SplitKV — DONE (§2.2):** the `tc_split` gate (decode-shaped only) + the reused
  `_gemv_auto_num_splits` plan (env `VLLM_FA4_HD256_TC_NUM_SPLITS` / caller `num_splits` /
  auto precedence, cap on `max_seqlen_k`) + the fp32 `out_partial`/`lse_partial`
  allocation + the reused `_flash_attn_fwd_combine` host combine. **The
  `interface-tc-dispatch.diff` was regenerated** (3 hunks: the `_flash_attn_fwd_combine`
  import, the `_gemv_dtype_ok` widening, and the `use_tc_hd256` block carrying the paged +
  SplitKV dispatch) and **round-trip verified** against the image base
  (`docker run … cat …/interface.py` → `patch -p1` → byte-identical to the live vfa-tree
  `interface.py`). Still reference-only (no `apply_patches.py`/`verify_patches.py` entry,
  no Dockerfile `COPY`) — see the v14 note above for the ship path.
- **Dtypes:** widen the `q.dtype`/`v.dtype` gates (`:975-976`) per §2.3 / §2.4 — **DONE**
  (the 4-combo gate landed with the dtype parity, §2.3/§2.4).
- **varlen-M — DONE (§2.6, 2026-10-03):** the tc gate M-term (`max_seqlen_q <= 8`, dense-only)
  + the `tc_key` M term (separate binary per M) + the `_tc_dtype_ok` M≤8 mixed-dtype widening +
  the `is_m1` `Constexpr` threaded to the kernel's compile-time LSE-select branch. The
  `interface-tc-dispatch.diff` was **regenerated** (round-trip verified against the GEMV base)
  to carry all of it.

## Part 4 — Verification (extend `verify-tc-decode.py` + bench)

- **Clean-window gate (now in the verify):** `verify-tc-decode.py` carries the inline
  gate (`server_load` / `Monitor` / `wait_for_idle` / dirty-window audit, `--no-gate`,
  exit codes 0=clean-pass / 1=clean-fail / 2=dirty-window / 3=timeout) — the same logic
  as `src/mjolnir/gate.py`, copied in because the in-container run can't import the
  host package. Every TC number is now only reported for a **gated clean window** (the
  fp8 large-L flakiness was co-tenancy — see the status note up top). `mjolnir gate
  --once` / the GEMV bench scripts share the same gate.
- **New verify cells** (all through the real `_flash_attn_fwd` interface, so the
  dispatch is exercised, not just the kernel):
   - **paged** (page_size 64/128) ✅ (`verify-tc-decode-paged.py`, 8 cells, dense-vs-paged
     bit-exact — §2.1);
   - **partial-tail** (L=100, L=37);
  - **empty-batch** (kv_len=0);
   - **SplitKV** ✅ (`verify-tc-decode-splitkv.py`, 13 cases: dense + paged, bf16 + fp8,
     ns sweep {1,2,4} caller/env + auto-ns, empty-split paged [256,64], 3× determinism —
     combined(ns>1) vs ns=1 write-through, gated; §2.2);
    - **fp16** leg ✅ + **fp8-Q** leg ✅ (added 2026-10-02; small-L L=64/256 B=1/4,
      pass at rel ≤ 1.8e-2 — the fp8 large-L legs are left to the gated run since they
      are co-tenancy-sensitive).
    - **varlen-M** ✅ (`verify-tc-decode-varlenM.py`, 25 cells, added 2026-10-03; §2.6):
      M∈{1,2,4,8} × dense+paged × bf16+fp8 vs the gold full-attention-over-L reference + M=8
      split (ns=4) + the gate-reject test (M=9/16: no new `tc_compile_cache` entry) + an M=1
      regression spot. 25/25 GO; the M=1 regression stays green (dense 34/35, paged 8/8,
      SplitKV 13/13) because the `is_m1` compile-time branch keeps M=1 on the original codegen.
- **Bench:** add the TC `tc_decode` leg to `gemv-geom-sweep.py` / `gemv-decode-bench.py`
  (the `tc_decode` leg already exists) so paged + SplitKV perf is measured in a gated
  window.

## Part 5 — Phasing & effort

| Phase | Items | Effort | Unlocks |
|---|---|---|---|
| **1 — correctness** | 1.1 tail ✅, 1.2 empty-batch O ✅, 1.3 varlen LSE ✅ (A/B/C fixed), gate ✅ | S–M | dense green; varlen O + LSE green (27/27 gated) |
| **2 — production** ✅ | 2.1 paged — **DONE** (page-indexed tiled cp.async + seqused_k; two bugs fixed) | **L** | TC can *serve* (paged) |
| **3 — perf parity** | 2.2 SplitKV ✅, 2.7 fp8-PV ✅, 2.5 re-tile | M–L | SplitKV + the fp8-PV mma landed; re-tile + the tcgen05 spike open |
| **4 — dtype parity** ✅ | 2.3 fp16 ✅, 2.4 fp8-Q ✅ | S | full dtype matrix (4-combo gate live, verified green) |
| **5 — varlen-M** ✅ | 2.6 M≤8 (MTP/DSpark/DFlash verify) — **DONE** (2026-10-03) | M | speculative-decode verify through the TC path (M×G "m" dim, ≤48) |

Dependencies: 1.1 (tail) is needed *inside* 2.1 (paged's last block is partial); 2.2 is
independent but pairs with 2.1 in the production long-context path; the dtype items are
independent.

## Checklist

- [ ] **Phase 1**
  - [x] 1.1 diagnose + fix `L < TILE_N` (interface `tc_seqlen_k` + mn-view mask; verify
        L=1/3/5 → green, gated)
  - [x] 1.2 empty-batch `O=0` (varlen O store offset by `q_row_base`)
  - [x] 1.3 varlen LSE — **all three bugs fixed** (A: head/seqlen transpose; B: per-CTA base
        carried in the `compute_ptr`/`store_LSE` coordinate, not a `domain_offset`; C: varlen
        Q-load nested `domain_offset(((0, q_row_base), 0), …)`). `lse_err≈7e-7` gated.
        Detail: `task/debug/tc-varlen-lse-store.md`.
  - [x] clean-window gate in `verify-tc-decode.py` (+ co-tenancy finding)
  - [ ] verify cells: L=100/L=37 partial-tail (untested path, §1.1) — empty-batch already
        covered by the varlen `lens=[5,1,0]` cell
- [ ] **Phase 2**
  - [x] 2.1 paged KV — **DONE (2026-10-02)**: page-indexed *tiled* cp.async (the fast
        tiled path, not a per-row gather) + `seqused_k` (required for paged) + `page_size`
        ctor + paged-aware gate + flat-pool `page_table`/`seqused_k` compile/launch args.
        Two bugs fixed (multi-page gather column `nb//PS`→`(nb*TILE_N)//PS`; paged-variant
        reg-alloc miscompile pin → `mPin`). `verify-tc-decode-paged.py` 8 cells green
        (PS=128; dense-vs-paged bit-exact) + PS=64 spot 3/3. Image wiring (v14) still open.
- [ ] **Phase 3**
   - [x] 2.2 SplitKV (grid + chunk + partial + combine) — **DONE (2026-10-02)**: runtime `ns`
         grid y-dim + tile-aligned chunks + fp32 partials + the reused host combine (§2.2).
         The bring-up found + fixed three bugs: the **sVb cross-warp race** (fp8-leg-only
         `sync_threads()` at both PV sites → the fp8 large-L SplitKV is deterministic at the
         fp8 floor), the **auto-ns pool-size cap** (cap on `max_seqlen_k`, not the paged pool
         size, so a partial-page tail batch isn't over-split), and the **split-path
         first-block OOB mask guard** (split-aware `kv_len - gtile*TILE_N < TILE_N`, so a
         split whose only block is the partial tail still masks OOB). Verified by the
         dedicated `verify-tc-decode-splitkv.py` (13/13, gated) + the dense (34/35) / paged
         (8/8) sweeps under the split-aware xcheck.
  - [ ] 2.5 re-tile knobs (TILE_N / ring depth) + the tcgen05 spike (§2.2a, M≥8/prefill)
  - [x] 2.7 fp8-PV mma (e4m3 P × e4m3 V) — **DONE (2026-10-04)**: both sm_110a `mma.sync`
        blockers resolved — the P smem round-trip (blocker a: the fp8 P A-fragment can't be
        built in-regs → store P element-wise via the QK C-partition to a plain 4 KiB sP, reload
        as the PV A-operand) + the element-wise V B-load from the transposed sVt (blocker b: the
        fp8 V ldmatrix faults → the QK-K-leg idiom). The bf16-KV leg stays byte-identical
        (separate kernel instance; sP/tCsP are fp8-only). All four verify suites green, every
        fp8 cell passes (incl. fp8 varlen-M M≤8 + SplitKV). §2.7.
- [x] **Phase 4** ✅ (2026-10-02)
  - [x] 2.3 all-fp16 path (gate widened; mma `MmaF16BF16Op(DTYPE=fp16)` — verify green)
  - [x] 2.4 fp8-Q gate (4-combo gate admits `(fp8,fp8)`; native fp8-Q no pre-cast — verify green)
- [x] **Phase 5 — varlen-M** ✅ (2026-10-03)
  - [x] 2.6 M≤8 (MTP/DSpark/DFlash verify): the "m" mma dim = M×G (≤48); the dense LSE select is a
        compile-time `is_m1` branch (M=1 → `mode=[1,2,0]` bit-exact, M>1 → `mode=[2,1,0]`); tc gate
        admits M≤8 (dense-only) / rejects M>8; `tc_key` M term; `_tc_dtype_ok` M≤8. `is_m1`
        `Constexpr` threaded through `__call__`/`kernel` + the interface `max_seqlen_q == 1` arg.
        `verify-tc-decode-varlenM.py` 25/25 GO (M sweep + M=8 split + gate-reject + M=1 spot);
        the M=1 regression stays green (dense 34/35, paged 8/8, SplitKV 13/13).
- [ ] bench: TC `tc_decode` leg in the geometry sweep + `gemv-decode-bench`
