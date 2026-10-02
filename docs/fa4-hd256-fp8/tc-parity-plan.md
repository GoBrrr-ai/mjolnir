# TC (mma.sync) hd256 decode — GEMV feature-parity plan

**Status: Phase 1 (correctness) — dense fully green, varlen O green, varlen LSE
diagnosed-but-open (§1.3, two bugs: head/seqlen mixup fixed, dynamic-base pointer-path
loss open). The varlen LSE fix is a small `pack_gqa` change with a dense-path
regression surface, deferred to a focused follow-up.** This is the workplan to bring the tensor-core
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
the TC streamer is memory-bound — see `gemv-long-context-diagnosis.md`). But as-is the
TC kernel is **benchmark-only**: vLLM serving decode is *always* paged, and the TC
dispatch gate requires `page_table is None` (`interface.py:977`). So it can never be
selected in production until **paged KV** lands. This plan closes that gap and the
rest of the capability surface.

## Capability baseline (as of the fp8-K-path fix, `ab900fe`)

| Capability | GEMV | TC (today, Phase 1) |
|---|---|---|
| Compute model | pure FMA — compute/issue-bound | `mma.sync` — memory-bound streamer |
| M / head_dim | M=1, hd256 | M=1, hd256 |
| Q dtype | bf16 / fp16 / fp8 | bf16 / fp8 (fp8-Q pre-cast in the dispatch) |
| KV dtype + descales | bf16 / fp16 / fp8 + descales | bf16 / fp8 + descales |
| Dense KV | ✅ | ✅ (L=1…65536, B=1/4, gated) |
| Varlen (cu_seqlens) | ✅ | ✅ O (LSE pending, §1.3) |
| Paged KV (16/128) | ✅ | ❌ |
| SplitKV (ns>1 + combine) | ✅ | ❌ (ns=1) |
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

### 1.3 Varlen LSE store — **OPEN (last Phase 1 item; deep bug, diagnosed)**

Two distinct bugs, both now root-caused (web-researcher + debugger, 2026-10-02):

- **Bug A — head/seqlen mixup (FIXED).** The interface allocates the varlen LSE as
  `(num_head, total_q)` = `(H, total_q)` (head at mode 0), but the kernel assumed
  seqlen-major `(total_q, H)` and skipped the dense-only transpose. Result: the
  `H`/`total_q` modes were swapped before `pack_gqa_layout(head_idx=1)`.
  **Fix:** transpose the varlen LSE `mode=[1,0]` → `(total_q, H)` before packing
  (mirrors the dense `mode=[1,2,0]`). This makes the packed layout correct:
  `((qhead, seqlen), nkv)` with `nkv` **flat** (stride `total_q·G`) — confirmed by
  the debugger's layout print. (The nkv mode is flat, not ranked; a scalar
  `mLSE[None, kv_head]` correctly selects the kv_head.)

- **Bug B — dynamic per-CTA base lost in the LSE pointer path (OPEN).** Even with the
  layout correct, `store_LSE`/`compute_ptr`/`elem_pointer` drop the per-CTA base
  `18·kv_head + q_row_base`: the dynamic `q_row_base` is 16-B-rounded in transit
  (b1/b2 → +4 elems) and the store pointers from **all 12 CTAs** fall in offsets
  `[0,16]` of the 72-elem buffer. Every CTA writes the same low cells → cross-batch
  clobber (batch1's row never written; batch2's `-inf` overwrites batch0's). The O is
  exact because `store_O` goes through TiledCopy (a proven `domain_offset` path), not
  this ptr/shuffle path. A one-line `domain_offset(((0, q_row_base),), …)` fix (v1)
  does **not** survive to the store pointer (the offset is corrupted in `elem_pointer`).
  - **Fix (planned, option 1):** bypass the broken path — pass the per-CTA base
    explicitly as a runtime `Int32` (`lse_base_off = kv_head·G·seqlen_q + q_row_base`)
    to `store_LSE`/`compute_ptr` and build the store pointer with raw
    `buffer_base + 4·(lse_base_off + total_q·h_idx + m_idx)` arithmetic (no tensor
    base / `domain_offset` / `elem_pointer`). Requires threading `total_q` (the dynamic
    `seqlen` stride) through. Detail: `task/debug/tc-varlen-lse-store.md` (v2).
  - **Possible bug C:** in one codegen batch1's in-register `row_sum` read ~`ln 1.96`
    off vs ref while O was exact (O rescales with the same `row_scale`) — likely a
    printf artifact under the broken codegen; re-verify after the pointer fix.

The varlen **O is fully correct** (all batches/heads incl. the empty one) — that is the
primary output. The LSE is secondary; the fix is a real (if small) `pack_gqa` change with
a regression surface on the dense path, so it is deferred to a focused follow-up rather
than landed mid-Phase-1.

---

## Part 2 — Feature additions (GEMV parity)

### 2.1 Paged KV — the production blocker (highest effort)

- **Why essential:** vLLM serving decode is *always* paged (block tables); the TC gate
  `interface.py:977` hard-requires `page_table is None`, so TC is benchmark-only today.
- **GEMV reference:** the paged gather at `sm100_hd256_decode_gemv.py:405-416, 442-449,
  532-539, 590-597`:
  ```
  grow = kv_start + crow
  page = grow // PS
  phys = mPageTable[batch, page]
  base = (phys*PS + (grow − page·PS)) * num_head_kv + kv_head
  ```
  Reuse the in-tree `paged_kv.py` / `flash_fwd_sm100.py` paged-view pattern.
- **TC change-points:** the TC KV cp.async is a **tiled** copy (`gmem_thr_copy_K/V`,
  `tKgK/tVgV`), *not* GEMV's per-element copy — so the page-table indirection cannot be
  inlined the same way. Two design options (needs a spike):
  1. **Per-row cp.async** for the paged leg (drop the tiled atom, gather per row like
     GEMV) — simplest, proven, slightly lower issue-rate;
  2. **Gathered gmem view** (a paged iterator over `mK/mV` so the tiled atom reads
     `phys` rows) — keeps the fast tiled path, more DSL work.
- **Also:** `page_size` becomes a constructor constant (0/16/128) as in GEMV `:94`;
  pass `mPageTable` through (already a param, currently an unused pass-through).

### 2.2 SplitKV (`num_splits>1` + host combine)

- **Why it matters for TC:** memory-bound → more CTAs = better BW at low batch
  (batch=1, 4 kv-heads = 4 CTAs vs 20 SMs); SplitKV fills the GPU.
- **GEMV reference:**
  - chunk scan `:224-230` — `kv_chunk = ceil(kv_len/ns); kv_start = split*kv_chunk;
    kv_len_c = min(chunk, max(0, kv_len − start))`;
  - grid `y=num_splits` `:162`;
  - split epilogue `:624-630` — chunk-normalized `O_partial = o·v_descale / l`,
    `LSE_partial = (m + log2 l)·ln2`;
  - host combine `interface.py:1243-1252` (`_flash_attn_fwd_combine`, the A1b ⊕-merge).
- **TC change-points:**
  - grid `sm100_hd256_decode_tc.py:209` `[num_kv_heads, 1, batch]` →
    `[num_kv_heads, num_splits, batch]`;
  - add the chunk (`kv_start`/`kv_len_c`) and drive the n-block loop `:572-649` over the
    *chunk's* blocks;
  - split the epilogue to `mO_partial`/`mLSE_partial` (fp32) when `mO_partial is not
    None` (the `IS_SPLIT` compile flag, GEMV `:194`);
  - **reuse** `_flash_attn_fwd_combine` unchanged.

### 2.3 fp16 KV dtype + descales

- **TC caveat (important):** `mma.sync` requires **matching A/B dtype** — unlike GEMV's
  FMA (all-fp32), there is **no bf16×fp16 mixed mma**. So parity means the **all-fp16**
  path (Q and KV both fp16), not mixed. Descales are dtype-agnostic (folded into the
  score scale / epilogue, already correct at `:258-267` and `:661`).
- **Change-points:** `get_smem_layout_atom(KVTYPE, D)` (`:305-306`) already keys on
  dtype; the 16-bit mma path (`sm80_utils.gemm/gemm_rs`) handles fp16 identically to
  bf16. Widen the interface gate to allow `(q, v)` both-fp16.

### 2.4 fp8(e4m3) Q dtype

- **Low effort:** the kernel already pre-casts Q to fp8 for the fp8 leg
  (`interface.py:1000-1003`) and streams Q as fp8 in smem (`QDTYPE = FP8T`, `:220`).
  fp8-Q input is just a **gate widening** (`interface.py:975` `q.dtype in
  (torch.bfloat16,)` → `{bf16, fp8}`) + skip the cast when Q is already fp8.
  (fp8-Q pairs with fp8-KV; 16-bit-Q pairs with 16-bit-KV — same dtype class.)

### 2.5 B3 re-tile knobs (TILE_N / ring depth)

- **GEMV reference:** the `VLLM_FA4_HD256_GEMV_TILE`/`_STAGES` env knobs
  (`sm100_hd256_decode_gemv.py:99-114`) varying the in-flight footprint.
- **TC change-points:** parametrize `TILE_N` (currently fixed 64) and the cp.async ring
  depth, same env-knob pattern; extend `gemv-geom-sweep.py` to hit the TC `tc_decode`
  leg.

---

## Part 3 — Interface wiring (per feature)

- **Paged:** drop the `page_table is None` gate (`interface.py:977`); add `page_size`
  to `tc_key` (`:1005`); pass `mPageTable` to the kernel call (`:177-178`).
- **SplitKV:** mirror the GEMV plan (`:1111-1139`) — reuse `_gemv_auto_num_splits` + the
  env/caller/cap precedence; allocate `out_partial`/`lse_partial`; add the `:1243`
  combine.
- **Dtypes:** widen the `q.dtype`/`v.dtype` gates (`:975-976`) per §2.3 / §2.4.

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
  - **paged** (page_size 16/128) for dense + varlen;
  - **partial-tail** (L=100, L=37);
  - **empty-batch** (kv_len=0);
  - **SplitKV** (ns>1, cross-checked against the ns=1 write-through);
  - **fp16** leg; **fp8-Q** leg.
- **Bench:** add the TC `tc_decode` leg to `gemv-geom-sweep.py` / `gemv-decode-bench.py`
  (the `tc_decode` leg already exists) so paged + SplitKV perf is measured in a gated
  window.

## Part 5 — Phasing & effort

| Phase | Items | Effort | Unlocks |
|---|---|---|---|
| **1 — correctness** | 1.1 tail ✅, 1.2 empty-batch O ✅, 1.3 varlen LSE (A fixed / B open), gate ✅ | S–M | dense green; varlen O green; varlen LSE → focused follow-up |
| **2 — production** | 2.1 paged (reuse 1.1 tail-clamp) | **L** | TC can *serve* (paged) |
| **3 — perf parity** | 2.2 SplitKV, 2.5 re-tile | M–L | fills SMs at low batch |
| **4 — dtype parity** | 2.3 fp16, 2.4 fp8-Q | S | full dtype matrix |

Dependencies: 1.1 (tail) is needed *inside* 2.1 (paged's last block is partial); 2.2 is
independent but pairs with 2.1 in the production long-context path; the dtype items are
independent.

## Checklist

- [ ] **Phase 1**
  - [x] 1.1 diagnose + fix `L < TILE_N` (interface `tc_seqlen_k` + mn-view mask; verify
        L=1/3/5 → green, gated)
  - [x] 1.2 empty-batch `O=0` (varlen O store offset by `q_row_base`)
  - [ ] 1.3 varlen LSE — **bug A (head/seqlen transpose) fixed**; **bug B (dynamic per-CTA
        base lost in `store_LSE`/`compute_ptr`/`elem_pointer` ptr path) open** → explicit-base
        pointer-arithmetic fix (bypass the ptr path), see `task/debug/tc-varlen-lse-store.md`
  - [x] clean-window gate in `verify-tc-decode.py` (+ co-tenancy finding)
  - [ ] verify cells: L=100/L=37 partial-tail + empty-batch
- [ ] **Phase 2**
  - [ ] 2.1 paged KV (design spike: per-row vs gathered-view) + `page_size` ctor
- [ ] **Phase 3**
  - [ ] 2.2 SplitKV (grid + chunk + partial + combine)
  - [ ] 2.5 re-tile knobs (TILE_N / ring depth)
- [ ] **Phase 4**
  - [ ] 2.3 all-fp16 path
  - [ ] 2.4 fp8-Q gate
- [ ] bench: TC `tc_decode` leg in the geometry sweep + `gemv-decode-bench`
