# The MMA kernel design log, part 3 (the device cu + the mma wiring)

Continues `mma-kernel-design-2.md` (part 2 ends after Session 8).

## Session 9 (2026-10-25): the device cu, and the mma actually wired

A source read of vLLM v0.32.0's V2 speculator
(`v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py`,
`cudagraph_utils.py`) settles the C bug without a probe:

- **The FULL graphs replay without a metadata rebuild.** `propose()` ->
  `prefill_cudagraph_manager.run_fullgraph()` for the draft prefill and
  `decode_cudagraph_manager.run_fullgraph()` for the fused draft steps
  is a bare `graph.replay()`. The metadata the graph saw was built ONCE,
  at capture, from `InputBatch.make_dummy` (seq_lens = num_tokens per
  request: 1 for a draft step, 1+K for the draft prefill). The drafter
  runner's host-planned `self._cu` therefore held the capture's dummy
  lengths for every replay: each draft decode step attended ONE key.
  That is the whole matrix: B (the drafter's ql=1 steps on Mjolnir, its
  draft prefill on the fallback via MAX_QLEN=1) keeps the prefill's good
  first draft and loses the rest -> 0.69/round; C also runs the draft
  prefill (ql=4) on the drafter's stale runner -> 0.0.
- **Eager is no cure either.** With an in-place updater (the fused path,
  our `update_draft_decode_metadata` is a no-op for T5 steps) the steps
  after the first reuse step 1's cu -- one key short per step.
- **The NaN in `call_mtp_ql1`** (out = nan, opart_sum = 0): kv_len = 0
  in every split -> lse = -inf everywhere -> the combine computed
  exp2(-inf - -inf) = NaN. The first-3-calls capture sees the startup
  dummy/profile steps, where seq_len 0 is legitimate (and a CUDA-graph
  padding request is seq_len 0 in every batch). The combine floors L at
  -FLT_MAX now: an all-empty row is 0, as documented.

The fix (runner): `plan()` stages nothing; `run()` writes
`cu[1:b+1] = cumsum(seq_lens[:b])` on the device right before the
launch, so the cumsum is captured with the kernel and every replay,
fused step and rewind reads the live lengths. `margin` and the UB are
harmless again, so the backend's `margin=0` / `.cpu()` sync were
reverted to main's: Thor's fused-draft plan needs the margin, and the
per-build sync cost both families the async scheduling.

The wiring audit also found:

- **The backend never selected the mma.** `GEMVDecodeRunner(impl=)`
  defaults to `gemv` and nothing passed `impl` — every serve so far ran
  the GEMV (Session 6e's "the mma decode kernels serving" was the GEMV).
  `arch/orin/__init__.py` now picks `mma` for fp8 KV / hd256 / q_len <= 8
  (`MJOLNIR_ATTN_ORIN_IMPL=gemv` opts out) and forces its bf16 Q (the
  mma asserts bf16; the backend's default is fp8 Q).
- **ns_cap=16 came from the backend's `MJOLNIR_ATTN_MAX_SPLITS` default**,
  which the runner took as the exact capacity. It is a cap now (the
  README's "max KV chunks per sequence"): the mma's 8 on AGX Orin.
- **The mma ignored `causal=False`** (the DSpark/DFlash block): the
  mask now folds `self.causal` at trace time (`_key_limit`).
- The mma's env probes (MJOLNIR_MMA16/REGCAP/REGPATH) are in the compile
  key, so the disk cache cannot serve one probe's binary to another.

Tests: `tests/kernel/test_orin_runner.py` (run by `verify vllm` on
Orin): live seq_lens vs the plan, graph capture on dummy lengths then
replay with the real ones + an in-place advance, padding rows 0 not NaN,
the q_len-4 verify, the non-causal block, the Runner's selection. Not
yet run on the board -- the A/B/C matrix is the gate (Session 8's plan).
