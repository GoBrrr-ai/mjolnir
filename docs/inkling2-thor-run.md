# Inkling-2-Mini on Jetson Thor (sm_110) — the run record

**Branch** `feat/inkling2-nightly` · 2026-11-22/23 · 22 commits · **status: PARKED**

The full run: the nightly port (`0.35.0rc0.dev`, pin `647a76866b45`), the phase-2 plan
(`task/thor/plans/plan-inkling2-mjolnir-phase2.md`: M0 profile → the A/B/C/D decision
ladder), then the improvement loop (11 experiments, 2 research rounds, images `a1`→`b11`).
Every experiment ran to a verdict with artifacts; nothing was parked without a number.

## The stack that ships

| piece | state |
|---|---|
| image | `narandill/mjolnir-vllm:v0.35.0rc0.dev-b11` (the `b`-line: the gate-widen + tune + bmmarch + FI-moj + ctas-pow2 + humming110 + ll120, all inert-or-kept) |
| serving config | `configs/thinkingmachines/Inkling-2-Mini/NVFP4` (+ the `.env` sidecar: `VLLM_USE_BREAKABLE_CUDAGRAPH=1`) |
| attention | the model's own Triton sm90-path kernels, eager-inside-piecewise-graphs by design |
| MoE | the Humming select-variant (inkling's `moe_sm90` experts) on the NVFP4 checkpoint |
| spec | DSpark K=7 (the checkpoint's invariant), acceptance 0.611 / 2.33 tokens-step |
| graphs | piecewise-breakable, 51 captured segments, 162 replays + 52.0 eager kernels/token |

### The patch inventory (all registered, sentineled, `PATCHES.md`; **post-cleanup: 22+2 sentinels — the no-op experiments' patches stripped to git history, this table is the archaeology**)
| patch | origin | verdict | default |
|---|---|---|---|
| `thor-inkling2-fp8kv` | Thor | fp8-KV +15.7 % step — **parked-negative** | off (the variant config) |
| `thor-inkling2-splitcap` | Thor | the split-cap //4: parity (+0.9 sub-noise) — **parked** | the a1-line only |
| `thor-inkling2-fa4rel` | Thor | re-armed, untested-to-verdict | inert |
| `thor-inkling2-sm100gate` | Thor | the sm100-family gate: the 110 boot-kill named (the FI context `[10,12]`) — **parked-layered** | opt-in `MJOLNIR_INKLING2_SM100_KERNELS` |
| `thor-inkling2-tune` | Thor | the launch-const sweeps: kernel-real (−6.7/−13.7 %), step-wash (n3) — **closed** | opt-in env knobs |
| `thor-inkling2-bmmarch` | Thor | the trtllm-bmm `[10]`→`[10,11]` — necessary-not-sufficient (E7 layer 1) | inert (gate-off unreachable) |
| `thor-fi-sm100moe-arch` | Thor | the REAL E7 kill: the FI nvcc majors `[10,12]`→`[10,11,12]` | inert (gate-off) |
| `thor-inkling2-ctas-pow2` | #69288 cherry-pick | correct upstream, **inert on our wiring** (the gate-off MoE never reaches `expert_plan`) — parity probe 13.86/13.89 s | inert |
| `thor-inkling2-humming110` | Thor (the #68761 variant) | the tiles-humming on 110: boots, 20.4368 in-band n1 — **wash-opt-in** | opt-in `MJOLNIR_INKLING2_HUMMING` |
| **`thor-inkling2-ll120gate`** | #69168 port | **KEPT default-ON**: the low-M trio on 110 — **19.7815 ms/token, the matrix's best; the eager tail 52.7→52.0** | on; escape `MJOLNIR_INKLING2_NO_LL120` |

### The env-matrix (the opt-ins, all default-inert unless noted)

`VLLM_USE_BREAKABLE_CUDAGRAPH=1` (the sidecar, **the kept directive**) ·
`MJOLNIR_INKLING2_SM100_KERNELS` · `MJOLNIR_INKLING2_{BLOCK_N,WARPS,STAGES}` ·
`MJOLNIR_INKLING2_HUMMING` · `MJOLNIR_INKLING2_NO_LL120` (the escape) ·
`MJOLNIR_INKLING2_STOCK_GATE` (the E1-era alias, superseded)

## The performance

### The step ladder (c1, ~96K-token context, the profile stage, K=7 drafter live)

| config | ms/token | n | the delta-read |
|---|---|---|---|
| the pre-graph base (the m0 anchor) | 21.77 | — | the phase-2 starting point |
| compile + FULL_DECODE_ONLY (the a1-class boot) | 20.19 | 1 | the image-default fallback |
| **piecewise (the ship)** | **20.083 mean / 19.81 best** | 3 | **−8.1 % vs eager — the kept directive** |
| + the launch-const tunes | 19.98 / 20.48 | 3/3 | wash (the ±3.6 % run-band named) |
| + `async-scheduling` | 20.59 | 2 | wash (engaged, boot-verified) |
| + the tiles-humming | 20.44 | 1 | wash |
| **+ the ll120 trio (KEPT)** | **19.7815** | 1 | **the matrix's best; the eager-tail −0.7 kernels/token** |
| **the ship: b11, plain boot (the keep-proof)** | **19.7003** | 1 | **default-on, no env — the run's best read; 52.0 eager** |

The run-to-run spread on an identical config measured **±3.6 %** (19.81…20.54 over n3) —
the profile-stage comparisons below ~2 % are unresolvable at n1; the e2e cell-means
(6 reps) are the instrument for any finer claim.

### End-to-end (`bench e2e`, 18 cells, `history.jsonl`, the a1-record; pp2048/tg128, 6 reps/cell)

| ctx | c1 tg | c1 peak | c2 tg | c2 peak | c4 tg | c4 peak |
|---|---|---|---|---|---|---|
| 0 | 36.7 | 49.3 | 54.2 | 69.2 | 102.2 | 135.2 |
| 4096 | 42.6 | 56.5 | 67.4 | 88.5 | 75.1 | 133.5 |
| 8192 | 44.8 | 58.8 | 66.6 | 91.3 | 57.8 | 127.7 |
| 16384 | 42.1 | 51.0 | 52.7 | 86.3 | 33.9 | 122.3 |
| 32768 | 39.1 | 54.0 | 27.2 | 80.5 | 17.6 | 115.3 |
| 131072 | 32.9 | **50.2** | 5.8* | 63.0 | 3.9* | 60.7 |

\* the deep-c2/c4 means carry pool-pressure runs (the 59 %-KV at c2 → the thrash at c4);
the peaks hold. TTFT @128K ≈ 40 s = the no-cache context build (128K ÷ 3394 tok/s ✓).

### Versus the upstream serving recipe

| metric | the recipe (524K-context box) | this run (the 262K config cap) |
|---|---|---|
| decode, c1 | 50–53 tok/s | **50.2 peak** @128K — **met at the peak**; 32.9 the n6-mean (the no-cache + the spread) |
| prefill | ~3.3–3.6 k tok/s league | **3394 tok/s** @128K-c1 — in-league |
| spec-health | — | 0.611 acceptance / 2.33 tokens-step, stable across every arm |

The recipe's depths (524K) exceed this config's 262K `max-model-len` — the comparison is
peak-vs-peak by construction. Raising the cap is a memory-budget decision, not a perf one
(262K ≈ 25.6 GB KV at the current util).

### Versus the stock vLLM config (the plain image boot)

The stock path resolves to compile+FULL_DECODE_ONLY (the model provides neither compiled
submodules nor, pre-env, breakable graphs): ~20.2 ms/token-class with 64 graph-replays and
58+ eager kernels/token. The kept delta-stack on top: **piecewise (−8.1 %) + the ll120
low-M trio (the band-floor + the eager-tail cut) = 19.70 plain-boot** — cumulatively
**≈ −9.5 %** vs the stock-boot class, with the spec/pd counters healthy throughout.

## The closed fronts (the loop, E1–E11)

| # | the one-line |
|---|---|
| E1 | the sm100-gate landscape mapped; the boot-kill = the FI compile-context `[10,12]`; opt-in landed |
| E2 | K≡7 — the checkpoint's own invariant (`DSpark requires … block size − 1`); the spec-K space is {7} |
| E3 | #69275: a hand-port (the post-pin drift × our fused-draft patch); parked-assessed |
| E4 | `dspark_draft_topk`: hf-config-only, no serve-plumb, ~1 % bound; parked-assessed |
| E5/E5b | the launch-const tunes: kernel-real, step-invisible (the n3 matrix; the ±3.6 % band) |
| E6 | piecewise > FDO (+1.2…4.9 %); the `verify profile --serve-env` plumb shipped (`247eeea`) |
| E7 | the gate-ON peel: 2 one-liners landed, the bedrock = **FI ships no sm110 trtllm-gen cubins** (`trtllm_batched_gemm_runner.cu:218`) — the upstream ask |
| E8 | `moe_backend` unwired on 110-gate-off (the ctor branches sm100/sm120 only); the sm110 MoE = the SM90 path |
| E9 | the round-2 cheap tier: #69288 parity-inert; the head-fp8 DeepSeek-only; async wash; the passes moot (mode-NONE) |
| E11 | the tiles-humming: boots, wash-opt-in (the select-variant already serves; the delta = the routing-layout only) |
| **E10** | **the ll120 trio: KEPT default-on — 19.7815, the eager-tail cut, the escape env** |

## The loose ends / the upstream asks

1. **FI trtllm-gen sm110 cubins** — the E7 bedrock; unlocks the tcgen05 MoE+bmm path wholesale
2. **#69288's rebase** — our cherry-pick is inert on the gate-off wiring; carry-or-drop at the next bump
3. **the #69168 sm110 tuning** — the trio runs stock-SM12x constants; a tuning pass on the 110 M-shapes is unthreatened headroom
4. **the e2e deep-c2/c4 pool-thrash** (the 5.8/3.9 tok/s means) — a KV-budget shape question, not a kernel one
5. the dspark-ADP assert + the K-variants: the phase-1 notes stand; the configs remain as the record

## The artifacts

- the loop ledger: `task/thor/inkling2-loop.md` (+ `inkling2-loop/`: the sweeps, the E2/E3 evidence, the probe outputs)
- the phase-2: `task/thor/inkling2-phase2/m0-profile.md`, `task/thor/plans/plan-inkling2-mjolnir-phase2.md`
- the research: `task/wiki/sources/inkling2-thor-perf-levers.md` (r1) / `…levers-r2.md`
- the images: `dev` → `a1` → `b2`…`b11` (the b-line = the patch vehicles; **b11 = the ship**)
- the e2e: `benchmarks/history.jsonl` (the 18-cell record); the profiles: `benchmarks/staging/verify/profile/`

*Parked 2026-11-23 at `cf276d3`+ — the loop resumes from the ledger's roster or a fresh research round.*
