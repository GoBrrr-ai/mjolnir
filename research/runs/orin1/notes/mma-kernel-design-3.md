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

## Session 9b (2026-10-25, the board): the kernels ESCAPED every graph capture

The first on-board `verify vllm`: 5/6 -- the graph-replay test failed
with `mma: replay: rel_err 20.69`. The probe bisect
(/tmp/opencode/{replay,eager,bake,rows,varlen,descale,ladder,ps64}_probe.py):

1. **The captured CUMSUM re-ran at replay (cu=[0,2500,3016] -- live) but
   the KERNEL DID NOT** (`out == warm-up output: True`; after
   `out.zero_()` + a manual-cu replay the out stayed 0). The launch was
   never in the graph.
2. **Root cause: the TVM-FFI env-stream marker.** `_compile_args` ended
   with `make_fake_stream(use_tvm_ffi_env_stream=True)` -- per the DSL
   docs that SKIPS the stream param from the signature and reads the
   stream "through the environment stream obtained from caller context".
   Nobody set that env, so every launch landed on the DEFAULT stream:
   fine eagerly (vLLM's forward rides the default stream -- config A's
   short-context probes never noticed), fatal under capture (the launch
   escapes; the replay produces the capture-time result).
3. **The fix (committed): the Thor convention.** `_compile_args` traces a
   REAL `cuda.CUstream` (`_stream()` = torch's current stream) so the
   signature keeps the stream as an ordinary 16th launch arg, and
   `run()` appends `_stream()` per call -- the capture stream during
   `torch.cuda.graph`, the live stream eagerly. The combine gets the
   same treatment (a new trailing `stream` param + the arg). Probe
   verdict: the replay now re-runs the kernel (`out != warm`), both
   impls.
4. **The variant gains `+s1`**: the pre-s1 disk binaries carry the old
   15-arg signature -- the marker orphans them. `run()` also asserts
   the pool's page size against the runner's (the compile bakes every
   tensor dim; a mismatched pool reads garbage strides -- the probe
   hour's lesson: `_runner()` hardwired the test module's PS=16 while
   the probes fed ps=64 pools, and EVERY "ps=64 failure" was that
   stale-binary artifact, not a kernel bug).
5. ⚠️ **`ref_decode` full-batch looks wrong for b>=2** (the same kernel
   output matches the per-row ref and mismatches the full-batch one at
   ps=16/b=2). The tests only use it via per-row slices -- park, but
   check before any full-batch ref use.

The graph-replay test's own capture also writes the cu via a captured
`torch.cumsum` -- that part always worked (torch ops ride the capture
stream); only the FFI launches needed the fix.

## Session 10 (2026-10-26): THE ROOT CAUSE -- the kernels hardcoded a contiguous KV pool

The A/B/C subtraction + the C-gemv cross-check pinned the failure to the
ql=4 T5 path, but the decisive instrument was the **stride probe** (a
BHN2D-shaped pool sliced the way the serving's `_kv_views` does):

- contiguous pool (every offline probe + the whole test suite): rel 0.0019 OK
- the serving's exact views (`pool[..., :D].transpose(1, 2)`, strides
  (131072, 512, 32768, 1)): **rel 1.41 FAIL at every ql** -- freshly
  compiled for those strides, both kernels.

Mechanism: both kernels address KV rows with FLAT pointer arithmetic --
`(pidx*PS + off) * num_head_kv + kv_head` (`* D` in the mma) -- valid only
for a fully contiguous `(np, ps, h_k, d)` K tensor. vLLM 0.32 interleaves
K|V in the last dim (the logical [L,B,H,N,C], C=2D) under EVERY layout
name, so the flat math never holds in real serving; the docstring's "the
kernel takes strides as runtime arguments" was aspirational. Every offline
probe passed because the harness feeds the same contiguous tensor as K and
V. The FFI *does* carry + check strides (a mismatch fails loudly:
"Mismatched mK.strides[0]") -- consistent compile/launch strides in
serving, so nothing errored; the kernel just read the wrong addresses.

**The fix (39117cf, patch 0007)**: the KV gathers go through
`cute.crd2idx((page, row_in_page, kv_head, 0), mK.layout)` -- the tensor's
own layout -- and `_cute` marks the KV tensors `mark_layout_dynamic()` so
the strides stay runtime values (the dummy-compiled capture path stays
valid for any pool). One site in the mma's shared `g2s_stage`, four in the
gemv; `num_head_kv` (gemv) went dead and is gone. Variant `+s1` -> `+s2`
orphans the flat-math binaries. Post-fix stride probe: 12/12 OK (both
impls x contig/strided x ql 1/2/4); tests/kernel/test_orin_runner 6/6.

Consequences for the campaign bookkeeping: config A (the pin) never
exercised ANY Mjolnir kernel (the `_is_draft` group-mismark force-fallbacked
the 16 target layers with the mtp one), B exercised the ql=1 T5 path --
which was CORRECT because... the draft steps' strided reads were garbage
too -- B's 2.10 needs re-explanation (see session 10b). The ns=16
over-split theory died: the offline output is identical at ns=0/8/16.

Also confirmed benign-but-real: the kernel disk cache has never delivered
a binary on this board ("Function prefix ..._args_spec not found" on every
load -> a ~16s recompile per process; `_export`/`_disk_load` symbol
desync). Filed for later; the serving is correct without it.

## Session 10b (2026-10-26): the matrix flips clean; the pin drops

Post-fix graphs-on matrix (the same 17*23 probe, temp 0, max_tokens 60):

| leg | config | text | accepted |
| --- | --- | --- | --- |
| A | the pin (the group on FI) | correct | 42.0 |
| A-gemv | the pin + the gemv kernel | correct | 42.0 |
| B | MAX_QLEN=1 | correct | 42.0 |
| C | everything Mjolnir | correct | 42.0 |

All four byte-identical: the Mjolnir path reproduces the FlashInfer token
stream exactly under greedy decoding. 0006 (the pin removal) applied on
top. The pre-fix B contradiction (the ql=1 draft steps reading the same
strided pool the stride probe convicts) stays parked -- the pre-fix
binaries are orphaned and the post-fix correctness makes it academic; the
leading theory is that B's fused-draft metadata capture rode the fallback
gate, which the dump cannot show (the FB branch never logged).

Campaign state: kernel exonerated -> root cause (the contiguous-KV
hardcode) -> fixed -> matrix clean -> pin dropped. Next: the spec sweep
(official-AWQ_FLASHINFER vs mjolnir-AWQ_MJOLNIR, k=1..7), then the e2e
benches (bench e2e / bench accuracy -- now unlocked per the directive).

## Session 10c (2026-10-26): the kernel disk cache never worked -- and now it does

The "cache load failed -> compiling, cache off" on EVERY process start was
a store/load FORMAT MISMATCH:

- `cute.compile(..., --enable-tvm-ffi)` returns a TVMFFIJitCompiledFunction;
  its `export_to_c(path, function_name=...)` (tvm_ffi_provider.py) writes
  an object carrying `__tvm_ffi_<symbol>` + `__cute_internal_<symbol>_*`
  and NO `_args_spec`.
- `_disk_load` called `cute.runtime.load_module(path)` WITHOUT
  `enable_tvm_ffi=True`, so `ExternalBinaryModule.__getattr__` took the
  executor-style path, which does `engine.lookup("<prefix>_args_spec")` --
  a symbol the TVM-FFI export never writes. Every load failed
  ("Failed to lookup function '..._args_spec'"), every process recompiled
  (~16s) and re-stored an object nothing could ever load.

Confirmed on the board's stored object directly (no GPU): the .o has
`__tvm_ffi_mjolnir_orin_6976b2c...` and zero `_args_spec` strings;
`load_module(..., enable_tvm_ffi=True)` + `getattr(module, prefix)`
returns the `tvm_ffi.core Function` cleanly. Fix: 6414d69 (patch 0008) --
one flag. The first post-fix serve should log loads from /data/cache and
skip the ~16s compile.
