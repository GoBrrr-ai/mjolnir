#!/usr/bin/env bash
# verify.sh — T5 paged fp8 decode kernel: correctness matrix, quick timing,
# and ncu sweeps against FlashInfer on Jetson Thor (sm_110).
#
# Run from the repo root (~/mjolnir):
#   ./verify.sh                      # correctness + timing + sweep
#   ./verify.sh correctness          # just the correctness matrix
#   ./verify.sh sweep diag           # any subset, in order
#   ./verify.sh all                  # correctness timing sweep diag
#   ./verify.sh ring                 # KV-ring depth sweep (best depth per L)
#   ./verify.sh vllm                 # vllm_thor_t5 runner tests + vLLM smoke test
#   ./verify.sh prefetch             # L2-prefetch distance sweep (ncu) vs FI
#   ./verify.sh pdl                  # PDL on/off A/B (harness wall time; ncu can't see PDL)
#   ./verify.sh ab                   # A/B of kernel flag sets: ncu sweep + wall time
#   ./verify.sh kvfmt                # KV layout / format A/B (fp8, HND, fp6, fp4, nvfp4)
#   ./verify.sh quant                # accuracy of each KV format vs unquantized KV
#   ./verify.sh hndsweep             # full L sweep: FI on NHD and HND vs T5 fp8 KV on HND
#   ./verify.sh fp6prof              # full ncu profiles: hnd vs e2m3-hnd vs e2m1-hnd
#   ./verify.sh report <sweep-dir>   # re-print a sweep report (no profiling)
#
# Stages:
#   correctness  every case below for each dtype in DTYPES; PASS = exit 0 and
#                the harness printed PASS (kernel vs bit-faithful emulation,
#                LSE, no NaN from poisoned pages)
#   timing       harness-timed (cold L2, no ncu) at 8K / 64K / 256K, 1 and 2
#                CTAs/SM
#   sweep        ncu L-sweep vs FI: FlashInfer (native driver) vs T5 fp8-Q at the
#                harness default (e2m3 KV, HND, 3-stage ring, 2 CTAs/SM, evict_first on)
#                SWEEP_CTAS_AB=1 adds the old 2-vs-1 CTAs/SM comparison first
#                (1/SM measured 0.80x FI at 128K-256K -> 2/SM is the default)
#   diag         unlocked clocks + shallower KV ring at long L
#   ring         KV-ring depth sweep (RING_STAGES at RING_CTAS CTAs/SM) vs FI,
#                with a depth x L table and the best depth per L
#   vllm         the vLLM plugin package (PKG): runner-vs-emulator tests (every
#                KV layout, strided Q, scales, q_len 2, CUDA-graph replay), then
#                greedy CUSTOM-vs-FLASHINFER smoke runs (SMOKE_MODES)
#
# Env (all optional):
#   DTYPES="fp8 bf16"   which Q dtypes to test
#   IMG                 docker image (default narandill/mjolnir-thor:v15)
#   TIMEOUT=180         per-case timeout in seconds (a hung case's container is killed)
#   FAIL_FAST=1         stop at the first failing correctness case
#   LOGDIR              where logs/reports go (default benchmarks/raw/verify-<ts>)
#   SWEEP_LS            L values for the sweeps (default: the sweep script's)
#   T5_CTAS_FULL        T5 --ctas_per_sm for the full sweep (default 2; with
#                       SWEEP_CTAS_AB=1: the faster of 2 and 1 at L >= 64K)
#   SWEEP_CTAS_AB=0     1: run the 2-vs-1 CTAs/SM sweeps before the full one
#   SWEEP_LEGS          legs of the full sweep (default "fi t5fp8": FlashInfer as vLLM
#                       runs it vs T5 at the harness default, e2m3 on HND; add fihnd /
#                       finhd / t5bf16 / gemv for FI-HND / same-data FI-NHD / bf16-Q T5 /
#                       the retired pure-FMA kernel)
#
# Harness defaults (what every stage runs unless it passes flags): fp8 Q ->
# e2m3 KV on head-major (HND) pages, bf16 Q -> fp8 KV on HND, default KV ring.
# The correctness matrix pins the vLLM config (fp8 KV, NHD) per case instead and
# has its own "default:" cases. vLLM itself still runs fp8 KV (its cache format).
#   ITERS=50            ncu iterations per point for every sweep stage (single
#                       cells move ~±7% between runs on this box at 20)
#   RING_STAGES="2 3 4 5 6"   depths for the ring stage
#   RING_CTAS=2               CTAs/SM for the ring stage
#   ALL_LS="128 4096 8192 16384 32768 65536 131072 262144"   default L values of the
#                       ab / kvfmt / ring / prefetch stages
#   RING_LS=$ALL_LS           L values for the ring stage
#   RING_LEG=t5fp8            t5fp8 | t5bf16 (which T5 path the ring stage sweeps)
#   RING_ITERS=50             profiled iterations per point (default: ITERS)
#   PKG                 the vllm_thor_t5 package dir (default $ROOT/src/vllm_thor_t5)
#   TESTS               its tests: test_runner.py, smoke_vllm.py
#                       (default $ROOT/tests/thor_t5)
#   SMOKE_MODES="graph eager"   smoke runs: graph | eager | spec<N> (ngram, N tokens);
#                       "" skips the smoke test
#   SMOKE_MODEL=Qwen/Qwen3-0.6B   SMOKE_TOKENS=128   TIMEOUT_VLLM=1800
#   HF_HOME             host HF cache mounted into the container
#                       (default ~/.cache/huggingface)
#   PF_DIST="0 1 2 4"   prefetch stage: --l2_prefetch distances (tiles ahead)
#   PF_LS / PF_LEG / PF_ITERS   prefetch stage L values / t5fp8|t5bf16 / iters
#                       (defaults: RING_LS, else $ALL_LS / t5fp8 / ITERS)
#   PDL_LS="8192 65536 262144"  PDL_BATCHES="1 8"  PDL_ITERS=200   pdl stage
#   AB_VARIANTS="base:--kv_evict_first 0|evict:--kv_evict_first 1|pdl:--kv_evict_first 0 --pdl|both:--kv_evict_first 1 --pdl"
#                       (flags are explicit: the harness default is the vLLM
#                       default, i.e. evict_first on for fp8, off for bf16)
#   AB_LS / AB_LEG / AB_ITERS    ab stage ncu L values / t5fp8|t5bf16 / iters
#                       (defaults: $ALL_LS / t5fp8 / ITERS)
#   AB_WALL_LS="4096 8192 16384"  AB_WALL_ITERS=200   ab stage wall-clock points (b=1)
#   AB_FI_LEGS="fi"     FlashInfer reference legs of the ab stage (fi | finhd | fihnd);
#                       with fihnd, table cells show "vs FI /vs FI-HND"
#   KVFMT_VARIANTS / KVFMT_LS / KVFMT_WALL_LS / KVFMT_FI_LEGS   kvfmt stage (an `ab`
#                       preset; defaults: fp8 nhd, fp8 interleaved, fp8 hnd, e2m3/e3m2/e2m1 on hnd;
#                       L $ALL_LS; wall 64K; FI legs "fi finhd fihnd")
#   HNDSWEEP_LEGS="fi finhd fihnd t5fp8"  HNDSWEEP_EXTRA (more T5 flags, e.g.
#                       "--kv_format e2m3")   hndsweep stage (L range: SWEEP_LS)
#   FI_PAGE=128         page size of the finhd/fihnd FlashInfer driver
#   PROF_VARIANTS="hnd:--kv_format fp8 --kv_layout hnd|e2m3-hnd:...|e2m1-hnd:..."  PROF_L=65536
#   PROF_COUNT=3  PROF_SET=full  PROF_TIMEOUT=900  NCU_HOME   fp6prof stage
#   QUANT_FORMATS="fp8 e3m2 e2m3 e2m1 nvfp4"  QUANT_LS="4096 32768"   quant stage
#   NO_COLOR / FORCE_COLOR
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# The kernel + harness: the source the vLLM plugin loads.
KERNEL="${KERNEL:-$ROOT/src/vllm_thor_t5/t5_decode.py}"
SWEEP="${SWEEP:-$ROOT/ncu-t5-fp8-sweep.sh}"
IMG="${IMG:-narandill/mjolnir-thor:v15}"
TIMEOUT="${TIMEOUT:-180}"
DTYPES="${DTYPES:-fp8 bf16}"
FAIL_FAST="${FAIL_FAST:-0}"
TS="$(date +%Y%m%d-%H%M%S)"
LOGDIR="${LOGDIR:-$ROOT/benchmarks/raw/verify-$TS}"
SWEEP_LS="${SWEEP_LS:-}"
# full L range: default for the ab / kvfmt / ring / prefetch ncu stages
ALL_LS="${ALL_LS:-128 4096 8192 16384 32768 65536 131072 262144}"
PKG="${PKG:-$ROOT/src/vllm_thor_t5}"
TESTS="${TESTS:-$ROOT/tests/thor_t5}"
SMOKE_MODES="${SMOKE_MODES-graph eager}"
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen3-0.6B}"
SMOKE_TOKENS="${SMOKE_TOKENS:-128}"
TIMEOUT_VLLM="${TIMEOUT_VLLM:-1800}"
HF_HOME_HOST="${HF_HOME:-$HOME/.cache/huggingface}"
SWEEP_ITERS="${ITERS:-50}"   # ncu iterations for all sweep stages
export ROOT          # the sweep script uses it to find task/ and vfa-tree/
export T5_KERNEL="$KERNEL"   # ... and profiles this kernel file

# ------------------------------------------------------------------ colours --
if { [[ -t 1 ]] || [[ -n "${FORCE_COLOR:-}" ]]; } && [[ -z "${NO_COLOR:-}" ]]; then
  B=$'\e[1m'; D=$'\e[2m'; G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; C=$'\e[36m'; N=$'\e[0m'
else
  B=; D=; G=; R=; Y=; C=; N=
fi
hr()   { printf '%s\n' "${C}${B}━━ $* ${N}"; }
info() { printf '%s\n' "${D}$*${N}"; }

# ---------------------------------------------------------------- preflight --
preflight() {
  local ok=1
  [[ -f "$KERNEL" ]] || { echo "${R}missing kernel:${N} $KERNEL"; ok=0; }
  [[ -f "$SWEEP" ]]  || { echo "${R}missing sweep script:${N} $SWEEP"; ok=0; }
  command -v docker >/dev/null || { echo "${R}docker not found${N}"; ok=0; }
  if ! docker image inspect "$IMG" >/dev/null 2>&1; then
    echo "${Y}image $IMG not present locally — docker will try to pull it${N}"
  fi
  if command -v mjolnir >/dev/null 2>&1; then
    mjolnir gate --once >/dev/null 2>&1 \
      || echo "${Y}⚠  mjolnir gate says the box is busy — timings will be noisy${N}"
  fi
  (( ok )) || exit 2
  mkdir -p "$LOGDIR"
  info "repo:   $ROOT"
  info "kernel: $KERNEL"
  info "sweep:  $SWEEP"
  info "logs:   $LOGDIR"
}

# ------------------------------------------------------------- T5 runner ---
# t5_run <logfile> <fp8|bf16> [harness args...]  -> exit status of the harness
# Containers are named so a timed-out (hung) kernel can be killed: stopping
# only the docker CLI leaves the container — and the GPU — stuck.
reap() {  # reap <rc> <container>: on timeout, kill the container
  local rc="$1" name="$2"
  if (( rc == 124 || rc == 137 )); then
    docker kill "$name" >/dev/null 2>&1
    docker rm -f "$name" >/dev/null 2>&1
  fi
}

t5_run() {
  local log="$1" dt="$2"; shift 2
  local q=BFloat16; [[ "$dt" == fp8 ]] && q=Float8E4M3FN
  local name="t5verify-$$-$RANDOM"
  timeout "$TIMEOUT" docker run --rm --name "$name" --gpus all --entrypoint python3 \
    -v "$KERNEL":/tmp/sm100_hd256_decode_t5.py:ro "$IMG" \
    /tmp/sm100_hd256_decode_t5.py --q_dtype "$q" --kv_dtype Float8E4M3FN "$@" \
    > "$log" 2>&1
  local rc=$?
  reap "$rc" "$name"
  (( rc == 124 )) && echo "TIMEOUT after ${TIMEOUT}s (container killed: hung kernel?)" >> "$log"
  return $rc
}

slug() { echo "$*" | sed -e 's/--//g' -e 's/[^A-Za-z0-9_.=-]/_/g' -e 's/__*/_/g' -e 's/^_//;s/_$//'; }

PASSED=(); FAILED=(); TIMED=()

# case_ <label> <harness args...>   (runs once per dtype in DTYPES)
# Cases run on the vLLM production config (fp8 KV, NHD) unless they pass their
# own --kv_format / --kv_layout (later flags win); CASE_BASE="" runs the bare
# harness defaults (e2m3 on HND for fp8 Q).
case_() {
  local label="$1"; shift
  local dt base
  read -ra base <<< "${CASE_BASE---kv_format fp8 --kv_layout nhd}"
  set -- "${base[@]}" "$@"
  for dt in $DTYPES; do
    local name="$dt $label"
    local log="$LOGDIR/correctness/$(printf '%02d' $(( ${#PASSED[@]} + ${#FAILED[@]} + 1 )))-$dt-$(slug "$*").log"
    mkdir -p "$LOGDIR/correctness"
    local t0=$SECONDS
    t5_run "$log" "$dt" "$@"
    local rc=$? dt_s=$(( SECONDS - t0 ))
    local check plan
    check="$(grep -m1 '^check:' "$log" | sed -e 's/^check: //' -e 's/; .*//')"
    plan="$(grep -m1 'plan:' "$log" | sed -e 's/.*plan: SplitPlan(//' -e 's/)$//')"
    if [[ $rc -eq 0 ]] && grep -qx 'PASS' "$log"; then
      PASSED+=("$name")
      printf ' %s✅%s %-46s %s%-38s%s %s%3ss%s\n' "$G" "$N" "$name" "$D" "${check:-}" "$N" "$D" "$dt_s" "$N"
      [[ -n "$plan" ]] && printf '    %s└ %s%s\n' "$D" "$plan" "$N"
    else
      FAILED+=("$name|$log")
      local why="exit $rc"; [[ $rc -eq 124 ]] && why="TIMEOUT ${TIMEOUT}s"
      printf ' %s❌ %-46s %s%s\n' "$R" "$name" "$why" "$N"
      grep -E -m3 'Error|error:|Traceback|Assertion|NaN|Mismatch|not close' "$log" \
        | sed "s/^/    ${D}│ /;s/\$/${N}/"
      printf '    %s└ log: %s%s\n' "$D" "$log" "$N"
      if [[ "$FAIL_FAST" == "1" ]]; then summary; exit 1; fi
    fi
  done
}

# case8_ <label> <harness args...>: fp8-Q only (the KV formats need fp8 Q)
case8_() { local DTYPES=fp8; case_ "$@"; }
# cased_ <label> <harness args...>: harness defaults (e2m3 KV on HND for fp8 Q,
# fp8 KV on HND for bf16 Q), every dtype in DTYPES
cased_() { local CASE_BASE=""; case_ "$@"; }

# ---------------------------------------------------------------- stages ---
stage_correctness() {
  hr "1/4 correctness  (dtypes: $DTYPES)"
  info "   each case: kernel vs bit-faithful emulation + LSE + poisoned-page NaN check"
  # basics — one sequence, chunked by the plan
  case_ "L=8192"                         --seqlen 8192
  case_ "L=128 (1 chunk, direct out)"    --seqlen 128
  case_ "L=1000 (partial tile mask)"     --seqlen 1000
  case_ "L=1 (minimal)"                  --seqlen 1
  case_ "L=65536 (ring wraps)"           --seqlen 65536
  # vLLM batches
  case_ "varlen 6x<=20000"               --batches 6  --seqlen 20000 --varlen
  case_ "varlen 32x<=2048 (>1 wave)"     --batches 32 --seqlen 2048  --varlen
  case_ "spec q_len=2 varlen 4x<=3000"   --batches 4  --seqlen 3000  --varlen --q_len 2
  case_ "spec q_len=4 2x777"             --batches 2  --seqlen 777   --q_len 4
  # paging
  case_ "page 16 varlen"                 --batches 4  --seqlen 1000  --varlen --page_size 16
  case_ "page 64 varlen"                 --batches 4  --seqlen 1000  --varlen --page_size 64
  case_ "contiguous pages"               --seqlen 8192 --contiguous_pages
  # scheduling knobs
  case_ "2 CTAs/SM (40-CTA plan)"        --seqlen 8192 --ctas_per_sm 2
  case_ "forced 7 chunks"                --seqlen 8192 --kv_splits 7
  case_ "padded work list (40)"          --batches 3 --seqlen 5000 --varlen --pad_work_to 40
  case_ "max_splits=1 (no reduction)"    --seqlen 8192 --max_splits 1
  # head dim
  case_ "headdim 128"                    --seqlen 8192 --headdim 128
  # vLLM memory layouts (stride-generic kernel)
  case_ "KV interleaved (B,H,N,2D)"      --batches 4 --seqlen 3000 --varlen --page_size 16 --kv_layout interleaved
  case_ "KV flashinfer (B,2,N,H,D)"      --batches 4 --seqlen 3000 --varlen --page_size 32 --kv_layout flashinfer
  case_ "strided Q/O (fused QKV view)"   --batches 4 --seqlen 3000 --varlen --strided_qo
  case_ "per-tensor descales"            --batches 4 --seqlen 3000 --varlen --descale per_tensor
  case_ "per-head descales"              --batches 4 --seqlen 3000 --varlen --descale per_head
  case_ "vLLM-like: interl+strided+tensor p16" --batches 8 --seqlen 6000 --varlen --page_size 16 \
                                         --kv_layout interleaved --strided_qo --descale per_tensor
  # launch / prefetch options
  case_ "PDL (decode + reduction)"       --batches 4 --seqlen 20000 --varlen --pdl
  case_ "L2 prefetch 2, page 128"        --batches 2 --seqlen 30000 --varlen --l2_prefetch 2
  case_ "L2 prefetch 4, page 16"         --batches 4 --seqlen 9000 --varlen --page_size 16 --l2_prefetch 4
  case_ "PDL + L2 prefetch 1, q_len=2"   --batches 3 --seqlen 5000 --varlen --q_len 2 --pdl --l2_prefetch 1
  # head-major KV layout + KV cache formats (fp6/fp4 direct MMA, nvfp4)
  case_  "HND layout p128"               --seqlen 8192 --kv_layout hnd
  case_  "HND layout p16 varlen"         --batches 4 --seqlen 3000 --varlen --page_size 16 --kv_layout hnd
  case8_ "fp8 from float (accuracy ref)" --batches 2 --seqlen 5000 --varlen --kv_from_float
  case8_ "e3m2 KV p128"                  --seqlen 8192 --kv_format e3m2
  case8_ "e2m3 KV p16 varlen"            --batches 4 --seqlen 3000 --varlen --page_size 16 --kv_format e2m3
  case8_ "e2m1 KV p128"                  --seqlen 8192 --kv_format e2m1
  case8_ "e2m1 KV HND p16 q_len=2"       --batches 3 --seqlen 2500 --varlen --page_size 16 --q_len 2 \
                                         --kv_layout hnd --kv_format e2m1
  case8_ "nvfp4 KV p128"                 --seqlen 8192 --kv_format nvfp4
  case8_ "nvfp4 KV p16 varlen"           --batches 4 --seqlen 3000 --varlen --page_size 16 --kv_format nvfp4
  case8_ "nvfp4 KV HND q_len=2 pad"      --batches 3 --seqlen 2500 --varlen --q_len 2 --kv_layout hnd \
                                         --kv_format nvfp4 --pad_work_to 40
  # the harness default: e2m3 KV on HND (fp8 Q); bf16 Q runs fp8 KV on HND
  cased_ "default: e2m3 HND L=8192"      --seqlen 8192
  cased_ "default: L=65536 (ring wraps)" --seqlen 65536
  cased_ "default: varlen 6x<=20000"     --batches 6 --seqlen 20000 --varlen
  cased_ "default: q_len=3 p16 pad"      --batches 4 --seqlen 2000 --varlen --q_len 3 --page_size 16 --pad_work_to 40
  cased_ "default: headdim 128 + PDL"    --batches 2 --seqlen 9000 --varlen --headdim 128 --pdl
  # L2 eviction hint + PDL (the vLLM fp8 defaults)
  case_ "kv evict_first"                 --batches 4 --seqlen 20000 --varlen --kv_evict_first
  case_ "evict + PDL L=8192"             --seqlen 8192 --kv_evict_first --pdl
  case_ "evict + PDL q_len=3 + pad"      --batches 4 --seqlen 2000 --varlen --q_len 3 --kv_evict_first --pdl --pad_work_to 40
  case_ "evict + PDL, interl p16"        --batches 4 --seqlen 9000 --varlen --page_size 16 \
                                         --kv_layout interleaved --kv_evict_first --pdl
  case_ "vLLM-like spec q_len=3 p16"     --batches 4 --seqlen 2000 --varlen --q_len 3 --page_size 16 \
                                         --kv_layout interleaved --strided_qo --descale per_tensor --pad_work_to 40
}

stage_timing() {
  hr "2/4 quick timing  (harness-timed, cold L2, no ncu; harness defaults: fp8 Q -> e2m3 HND, bf16 Q -> fp8 HND)"
  mkdir -p "$LOGDIR/timing"
  local dt L c
  printf ' %s%-6s %-8s %-9s %14s %14s%s\n' "$B" dtype L ctas/SM "median" "GB/s" "$N"
  for dt in $DTYPES; do
    for L in 8192 65536 262144; do
      for c in 1 2; do
        local log="$LOGDIR/timing/$dt-L$L-ctas$c.log"
        t5_run "$log" "$dt" --seqlen "$L" --skip_ref_check \
          --warmup_iterations 5 --iterations 50 --ctas_per_sm "$c"
        local line us gbps
        line="$(grep -m1 '^median' "$log")"
        us="$(sed -n 's/^median \([0-9.]*\) us.*/\1/p' <<< "$line")"
        gbps="$(sed -n 's/.* us, \([0-9.e+]*\) GB\/s.*/\1/p' <<< "$line")"
        if [[ -n "$us" ]]; then
          printf ' %-6s %-8s %-9s %11s us %9s GB/s\n' "$dt" "$L" "$c" "$us" "$gbps"
          TIMED+=("$dt L=$L ctas=$c: $us us, $gbps GB/s")
        else
          printf ' %-6s %-8s %-9s %s%s%s\n' "$dt" "$L" "$c" "$R" "failed — $log" "$N"
          FAILED+=("timing $dt L=$L ctas=$c|$log")
        fi
      done
    done
  done
}

run_sweep() {  # run_sweep <outdir> <env...>
  local out="$1"; shift
  local lsenv=(); [[ -n "$SWEEP_LS" ]] && lsenv=(LS="$SWEEP_LS")
  env "$@" "${lsenv[@]}" OUT="$out" bash "$SWEEP" 2>&1 | tee "$out.console.log" \
    | sed -n '/T5 decode L-sweep/,$p'
}

# Average T5-fp8 total time at L >= 64K from a sweep's summary.csv
t5_long_us() {
  local csv="$1/summary.csv"
  [[ -f "$csv" ]] || { echo ""; return; }
  awk -F, '$2=="t5fp8" && $3=="decode" && $1>=65536 { s+=$10; n++ }
           END { if (n) printf "%.3f", s/n }' "$csv"
}

stage_sweep() {
  hr "3/4 ncu sweeps vs FlashInfer"
  mkdir -p "$LOGDIR/sweeps"
  local full_ctas="${T5_CTAS_FULL:-2}"
  if [[ "${SWEEP_CTAS_AB:-0}" == "1" ]]; then
    info "   A: T5 at 2 CTAs/SM (FI-equivalent 40-CTA plan)"
    run_sweep "$LOGDIR/sweeps/A-ctas2" LEGS="fi t5fp8" T5_CTAS=2 ITERS="$SWEEP_ITERS"
    info "   B: T5 at 1 CTA/SM (20-CTA plan)"
    run_sweep "$LOGDIR/sweeps/B-ctas1" LEGS="fi t5fp8" T5_CTAS=1 ITERS="$SWEEP_ITERS"
    if [[ -z "${T5_CTAS_FULL:-}" ]]; then
      local a b
      a="$(t5_long_us "$LOGDIR/sweeps/A-ctas2")"; b="$(t5_long_us "$LOGDIR/sweeps/B-ctas1")"
      full_ctas=2
      if [[ -n "$a" && -n "$b" ]] && awk -v a="$a" -v b="$b" 'BEGIN{exit !(b < a)}'; then
        full_ctas=1
      fi
      info "   long-L T5-fp8 mean: 2/SM ${a:-?} us, 1/SM ${b:-?} us -> full sweep uses ctas/SM=$full_ctas"
    fi
  fi
  local legs="${SWEEP_LEGS:-fi t5fp8}"
  info "   full sweep ($legs) at ctas/SM=$full_ctas, $SWEEP_ITERS iters"
  run_sweep "$LOGDIR/sweeps/C-full" LEGS="$legs" T5_CTAS="$full_ctas" ITERS="$SWEEP_ITERS"
}

stage_diag() {
  hr "4/4 diagnostics at long L"
  mkdir -p "$LOGDIR/sweeps"
  info "   unlocked clocks: compute-latency vs memory limit"
  run_sweep "$LOGDIR/sweeps/D-noclock" LEGS="fi t5fp8" T5_CTAS=2 CLOCK=none LS="65536 262144"
  info "   shallower KV ring at 2 CTAs/SM"
  run_sweep "$LOGDIR/sweeps/E-stages4" LEGS="t5fp8" T5_CTAS=2 \
    T5_EXTRA="--kv_stages 4" LS="65536 262144"
}

stage_ring() {
  local stages="${RING_STAGES:-2 3 4 5 6}" ctas="${RING_CTAS:-2}"
  local ls="${RING_LS:-$ALL_LS}" leg="${RING_LEG:-t5fp8}"
  local iters="${RING_ITERS:-$SWEEP_ITERS}"
  case "$leg" in t5fp8|t5bf16) ;; *) echo "RING_LEG must be t5fp8|t5bf16"; return 2 ;; esac
  hr "ring-depth sweep  ($leg, ${ctas} CTAs/SM, stages: $stages, iters: $iters)"
  mkdir -p "$LOGDIR/sweeps"
  info "   FI reference"
  SWEEP_LS="$ls" run_sweep "$LOGDIR/sweeps/ring-fi" LEGS="fi" ITERS="$iters" > /dev/null
  local st
  for st in $stages; do
    info "   kv_stages=$st"
    SWEEP_LS="$ls" run_sweep "$LOGDIR/sweeps/ring-$leg-s$st" LEGS="$leg" ITERS="$iters" \
      T5_CTAS="$ctas" T5_EXTRA="--kv_stages $st" > /dev/null
  done
  echo
  # depth x L table of decode+merge totals, best per L, ratio vs FI
  sweep_table "$LOGDIR/sweeps" "$ls" "$stages" "$leg" "ring-$leg-s" "ring-fi" kv_stages "fastest depth"
}

# sweep_table <root> <Ls> <values> <leg> <dir-prefix> <fi-dir> <label> <best-text>
sweep_table() {
  python3 - "$@" <<'PY'
import csv, os, sys
root, ls, stages, leg = sys.argv[1], [int(x) for x in sys.argv[2].split()], sys.argv[3].split(), sys.argv[4]
prefix, fi_dir, label, best_txt = sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]
def totals(d, leg):
    out = {}
    path = os.path.join(root, d, "summary.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r["leg"] == leg:
                out[int(r["L"])] = float(r["leg_total_us"])
    return out
FI_LABEL = {"fi": "FI", "finhd": "FI-NHD", "fihnd": "FI-HND"}
fis = {k: totals(fi_dir, k) for k in FI_LABEL}
fis = {k: v for k, v in fis.items() if v}
fi = fis.get("fi") or next(iter(fis.values()), {})
hnd = fis.get("fihnd") if "fi" in fis else None   # second ratio: vs FlashInfer on HND
W = 26 if hnd else 18
rows = {st: totals(f"{prefix}{st}", leg) for st in stages}
fmt = lambda t: (f"{t/1000:.2f} ms" if t >= 1000 else f"{t:.1f} us") if t else "—"
print(f"  {label:>10} " + " ".join(f"{L:>{W},}" for L in ls))
for k, v in fis.items():
    print(f"  {FI_LABEL[k]:>10} " + " ".join(f"{fmt(v.get(L)):>{W}}" for L in ls))
best = {L: min((rows[st][L], st) for st in stages if L in rows[st]) for L in ls
        if any(L in rows[st] for st in stages)}
for st in stages:
    cells = []
    for L in ls:
        t = rows[st].get(L)
        if not t:
            cells.append(f"{'—':>{W}}"); continue
        r = f" {fi[L]/t:4.2f}x" if L in fi else ""
        if hnd is not None:
            r += f" /{hnd[L]/t:4.2f}x" if L in hnd else " /  —  "
        mark = "*" if best.get(L, (0, None))[1] == st else " "
        cells.append(f"{fmt(t) + r + mark:>{W}}")
    print(f"  {st:>10} " + " ".join(cells))
print(f"  (* = {best_txt} at that L; Nx = FI time / T5 time, >1 means T5 faster"
      + ("; second ratio /Nx is vs FI-HND" if hnd is not None else "") + ")")
PY
}

stage_prefetch() {
  local dists="${PF_DIST:-0 1 2 4}" ls="${PF_LS:-${RING_LS:-$ALL_LS}}"
  local leg="${PF_LEG:-t5fp8}" iters="${PF_ITERS:-${RING_ITERS:-$SWEEP_ITERS}}"
  case "$leg" in t5fp8|t5bf16) ;; *) echo "PF_LEG must be t5fp8|t5bf16"; return 2 ;; esac
  hr "L2-prefetch sweep  ($leg, distances: $dists tiles, iters: $iters)"
  mkdir -p "$LOGDIR/sweeps"
  info "   FI reference"
  SWEEP_LS="$ls" run_sweep "$LOGDIR/sweeps/pf-fi" LEGS="fi" ITERS="$iters" > /dev/null
  local k
  for k in $dists; do
    info "   l2_prefetch=$k"
    SWEEP_LS="$ls" run_sweep "$LOGDIR/sweeps/pf-$leg-k$k" LEGS="$leg" ITERS="$iters" \
      T5_EXTRA="--l2_prefetch $k" > /dev/null
  done
  echo
  sweep_table "$LOGDIR/sweeps" "$ls" "$dists" "$leg" "pf-$leg-k" "pf-fi" l2_pf "fastest distance"
}

stage_ab() {
  local variants="${AB_VARIANTS:-base:--kv_evict_first 0|evict:--kv_evict_first 1|pdl:--kv_evict_first 0 --pdl|both:--kv_evict_first 1 --pdl}"
  local ls="${AB_LS:-$ALL_LS}" leg="${AB_LEG:-t5fp8}" iters="${AB_ITERS:-$SWEEP_ITERS}"
  local wls="${AB_WALL_LS:-4096 8192 16384}" witers="${AB_WALL_ITERS:-200}"
  case "$leg" in t5fp8|t5bf16) ;; *) echo "AB_LEG must be t5fp8|t5bf16"; return 2 ;; esac
  local names=() v name flags
  IFS='|' read -ra _vs <<< "$variants"
  hr "A/B  ($leg: ${_vs[*]}; ncu iters $iters)"
  mkdir -p "$LOGDIR/sweeps" "$LOGDIR/ab"
  local filegs="${AB_FI_LEGS:-fi}"
  info "   FI reference ($filegs)"
  SWEEP_LS="$ls" run_sweep "$LOGDIR/sweeps/ab-fi" LEGS="$filegs" ITERS="$iters" > /dev/null
  for v in "${_vs[@]}"; do
    name="${v%%:*}"; flags="${v#*:}"; names+=("$name")
    info "   $name: ${flags:-(defaults)}"
    SWEEP_LS="$ls" run_sweep "$LOGDIR/sweeps/ab-$leg-$name" LEGS="$leg" ITERS="$iters" \
      T5_EXTRA="$flags" > /dev/null
  done
  echo
  info "   ncu: decode + merge kernel time (no launch gaps)"
  sweep_table "$LOGDIR/sweeps" "$ls" "${names[*]}" "$leg" "ab-$leg-" "ab-fi" variant "fastest variant"
  # wall clock: includes the launch gap a separate reduction kernel costs
  echo
  info "   wall clock (harness median, cold L2, b=1, $witers iters) — includes launch gaps"
  local dt=fp8; [[ "$leg" == t5bf16 ]] && dt=bf16
  printf ' %s%-8s' "$B" L; for name in "${names[@]}"; do printf ' %12s' "$name"; done; printf '%s\n' "$N"
  local L
  for L in $wls; do
    printf ' %-8s' "$L"
    local base=""
    for v in "${_vs[@]}"; do
      name="${v%%:*}"; flags="${v#*:}"
      local log="$LOGDIR/ab/$dt-L$L-$name.log" us
      t5_run "$log" "$dt" --seqlen "$L" --skip_ref_check --warmup_iterations 10 \
        --iterations "$witers" $flags
      us="$(sed -n 's/^median \([0-9.]*\) us.*/\1/p' "$log" | head -1)"
      if [[ -z "$us" ]]; then printf ' %s%12s%s' "$R" "fail" "$N"; FAILED+=("ab $name L=$L|$log"); continue; fi
      if [[ -z "$base" ]]; then base="$us"; printf ' %9s us' "$us"
      else
        local d; d="$(python3 -c "print(f'{($us-$base)/$base*100:+.0f}%')")"
        local col="$D"; [[ "$d" == -* ]] && col="$G"
        printf ' %s%7s %4s%s' "$col" "$us" "$d" "$N"
      fi
      TIMED+=("ab $dt L=$L $name: $us us")
    done
    printf '\n'
  done
  info "   (first variant is the baseline; negative % = faster)"
}

stage_kvfmt() {
  AB_VARIANTS="${KVFMT_VARIANTS:-fp8:--kv_format fp8 --kv_layout nhd|interl:--kv_format fp8 --kv_layout interleaved|hnd:--kv_format fp8 --kv_layout hnd|e2m3-hnd:--kv_format e2m3 --kv_layout hnd|e3m2-hnd:--kv_format e3m2 --kv_layout hnd|e2m1-hnd:--kv_format e2m1 --kv_layout hnd}" \
  AB_LS="${KVFMT_LS:-$ALL_LS}" AB_WALL_LS="${KVFMT_WALL_LS:-65536}" AB_LEG=t5fp8 \
  AB_FI_LEGS="${KVFMT_FI_LEGS:-fi finhd fihnd}" \
    stage_ab
  info "   (ncu GB/s columns assume fp8 KV bytes: compare TIMES across formats)"
}

# The fair long-context baseline: FlashInfer on NHD (vLLM-style driver and
# the same-data layout driver) and on HND, against T5-fp8 on HND, full L range.
stage_hndsweep() {
  hr "HND sweep  (FI-tc, FI-NHD, FI-HND vs T5 fp8 KV on HND${HNDSWEEP_EXTRA:+ $HNDSWEEP_EXTRA})"
  mkdir -p "$LOGDIR/sweeps"
  run_sweep "$LOGDIR/sweeps/H-hnd" LEGS="${HNDSWEEP_LEGS:-fi finhd fihnd t5fp8}" \
    T5_CTAS=2 ITERS="$SWEEP_ITERS" T5FP8_EXTRA="--kv_format fp8 --kv_layout hnd ${HNDSWEEP_EXTRA:-}"
  info "   'T5-fp8 vs best FI' = against the faster FlashInfer layout at each L"
}

# Full ncu profiles (one .ncu-rep per variant, openable in the Nsight Compute
# GUI) of the T5 decode kernel, then a side-by-side of the metrics that
# explain why fp6 saves fewer microseconds than bytes: L2 sectors read vs the
# ideal byte ratio, throughput %, stall reasons, and the metrics that moved most.
stage_fp6prof() {
  local variants="${PROF_VARIANTS:-hnd:--kv_format fp8 --kv_layout hnd|e2m3-hnd:--kv_format e2m3 --kv_layout hnd|e2m1-hnd:--kv_format e2m1 --kv_layout hnd}"
  local L="${PROF_L:-65536}" n="${PROF_COUNT:-3}" set="${PROF_SET:-full}"
  local to="${PROF_TIMEOUT:-900}" dir="$LOGDIR/fp6prof"
  local ncu_home="${NCU_HOME:-/opt/nvidia/nsight-compute/2025.3.0}"
  local secs="--section SpeedOfLight --section MemoryWorkloadAnalysis --section WarpStateStats --section SchedulerStats --section Occupancy --section LaunchStats --section ComputeWorkloadAnalysis"
  IFS='|' read -ra _vs <<< "$variants"
  hr "fp6 profile  (ncu --set $set, L=$L, $n launches each: ${_vs[*]})"
  mkdir -p "$dir"
  local v name flags names=() specs=()
  for v in "${_vs[@]}"; do
    name="${v%%:*}"; flags="${v#*:}"; names+=("$name"); specs+=("$name=$flags")
    local log="$dir/$name.log" t0=$SECONDS rc attempt
    for attempt in "--set $set" "$secs"; do
      local cname="t5prof-$$-$RANDOM"
      info "   $name: ${flags:-(defaults)}  [$attempt]"
      timeout "$to" docker run --rm --name "$cname" --gpus all \
        --security-opt seccomp=unconfined --cap-add SYS_ADMIN --entrypoint bash \
        -v "$KERNEL":/t5.py:ro -v "$ncu_home":/ncu:ro -v "$dir":/out \
        -e TMPDIR=/tmp/ncu-t5 "$IMG" -c "mkdir -p /tmp/ncu-t5 && \
          /ncu/ncu --target-processes all --clock-control base --cache-control all \
            --kernel-name regex:MixedInputFusedMultiHeadAttentionDecode \
            --launch-skip 1 --launch-count $n $attempt -f -o /out/$name \
            python3 /t5.py --q_dtype Float8E4M3FN --kv_dtype Float8E4M3FN --seqlen $L \
              --skip_ref_check --no_l2_flush --warmup_iterations 0 --iterations $((n + 2)) $flags && \
          /ncu/ncu --import /out/$name.ncu-rep --page raw --csv > /out/$name.raw.csv && \
          /ncu/ncu --import /out/$name.ncu-rep --page details --csv > /out/$name.details.csv && \
          chmod a+rw /out/$name.*" > "$log" 2>&1
      rc=$?
      reap "$rc" "$cname"
      (( rc == 0 )) && break
      echo " ${Y}⚠  $name: ncu exit $rc with [$attempt]${N} — $(grep -m1 -E 'ERR|rror' "$log")"
    done
    if (( rc == 0 )); then PASSED+=("fp6prof $name")
    else FAILED+=("fp6prof $name (exit $rc)|$log"); fi
    printf '   %s└ %s.ncu-rep (%ss)%s
' "$D" "$dir/$name" $(( SECONDS - t0 )) "$N"
  done
  echo
  prof_compare "$dir" "${specs[@]}"
}

# prof_compare <dir> <name=flags>... : side-by-side of the profiles
prof_compare() {
  python3 - "$@" <<'PY'
import csv, math, os, re, sys
d, specs = sys.argv[1], sys.argv[2:]
names = [s.split("=", 1)[0] for s in specs]
flags = {s.split("=", 1)[0]: s.split("=", 1)[1] for s in specs}
BITS = {"fp8": 8, "e3m2": 6, "e2m3": 6, "e2m1": 4, "nvfp4": 4.5}
def bits(n):
    m = re.search(r"--kv_format\s+(\S+)", flags[n])
    return BITS.get(m.group(1) if m else "e2m3", 8)   # no flag = harness default e2m3
def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None
raw, units, det = {}, {}, {}
for n in names:
    p = os.path.join(d, f"{n}.raw.csv")
    if not os.path.exists(p):
        print(f"  (no {n}.raw.csv)"); continue
    rows = list(csv.reader(open(p, errors="replace")))
    hi = next((i for i, r in enumerate(rows) if "Kernel Name" in r or "ID" in r), None)
    if hi is None or len(rows) < hi + 3:
        print(f"  ({n}: empty profile)"); continue
    hdr, unit, data = rows[hi], rows[hi + 1], rows[hi + 2:]
    acc = {}
    for r in data:
        for k, v in zip(hdr, r):
            x = num(v)
            if x is not None:
                acc.setdefault(k, []).append(x)
    raw[n] = {k: sum(v) / len(v) for k, v in acc.items()}
    units.update({k: u for k, u in zip(hdr, unit)})
    # durations -> ns whatever unit this ncu prints
    tu = units.get("gpu__time_duration.sum", "nsecond").strip().lower()
    k_ns = {"nsecond": 1, "ns": 1, "usecond": 1e3, "us": 1e3, "msecond": 1e6, "ms": 1e6,
            "second": 1e9, "s": 1e9}.get(tu, 1)
    if "gpu__time_duration.sum" in raw[n]:
        raw[n]["gpu__time_duration.sum"] *= k_ns
    dp = os.path.join(d, f"{n}.details.csv")
    if os.path.exists(dp):
        acc = {}
        for r in csv.DictReader(open(dp, errors="replace")):
            x = num(r.get("Metric Value", ""))
            if x is None:
                continue
            k = (r.get("Section Name", ""), r.get("Metric Name", ""), r.get("Metric Unit", ""))
            acc.setdefault(k, []).append(x)
        det[n] = {k: sum(v) / len(v) for k, v in acc.items()}
ok = [n for n in names if n in raw]
units["gpu__time_duration.sum"] = "ns (converted)"
if not ok:
    sys.exit(0)
b0 = ok[0]
W = 14
LW = 46
def row(label, vals, unit=""):
    print(f"  {label[:LW]:<{LW}} " + " ".join(f"{v:>{W}}" for v in vals) + (f"  {unit}" if unit else ""))
def f(x):
    if x is None:
        return "—"
    a = abs(x)
    return f"{x:,.0f}" if a >= 1000 else f"{x:.3g}" if a < 10 else f"{x:.1f}"
row("variant", ok)
print("  " + "─" * (47 + (W + 1) * len(ok)))

# 1) bytes vs time: the headline
T = "gpu__time_duration.sum"
S = next((k for k in ("lts__t_sectors_op_read.sum", "lts__t_sectors.sum") if k in raw[b0]), None)
row("KV bits / element", [f"{bits(n):g}" for n in ok])
row("ideal byte ratio vs " + b0, [f"{bits(n) / bits(b0):.2f}" for n in ok])
if T in raw[b0]:
    row("duration (us, avg)", [f(raw[n].get(T, 0) / 1e3) for n in ok])
    row("time ratio vs " + b0, [f"{raw[n][T] / raw[b0][T]:.2f}" for n in ok])
if S:
    row(f"L2 {S.split('.')[0].replace('lts__t_', '')}", [f(raw[n].get(S)) for n in ok])
    row("L2 sector ratio vs " + b0, [f"{raw[n][S] / raw[b0][S]:.2f}" for n in ok])
    if T in raw[b0]:
        row("L2 read GB/s (sectors*32 / time)", [f(raw[n][S] * 32 / raw[n][T]) for n in ok])
print()

# 2) curated metrics (whatever this ncu/Thor exposes)
CUR = [
    ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "memory throughput %"),
    ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM throughput %"),
    ("lts__throughput.avg.pct_of_peak_sustained_elapsed", "L2 throughput %"),
    ("lts__t_sector_hit_rate.pct", "L2 hit rate %"),
    ("l1tex__throughput.avg.pct_of_peak_sustained_active", "L1/TEX throughput %"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM throughput %"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved occupancy %"),
    ("sm__inst_executed.sum", "instructions executed"),
    ("smsp__cycles_active.avg", "SMSP active cycles"),
    ("launch__registers_per_thread", "registers / thread"),
    ("launch__shared_mem_per_block_dynamic", "dynamic smem / CTA"),
    ("launch__grid_size", "grid (CTAs)"),
]
print("  key metrics")
for k, lab in CUR:
    if any(k in raw[n] for n in ok):
        row(lab, [f(raw[n].get(k)) for n in ok])
# pipes: anything tensor / TMA related that this build exposes
pipes = sorted({k for n in ok for k in raw[n]
                if re.search(r"(pipe_(tc|tensor|tma|uniform)|tmem|utc|tma)", k)
                and "pct_of_peak" in k and not k.endswith("_per_second")})
if pipes:
    print("\n  tensor / TMA pipes (% of peak)")
    for k in pipes[:12]:
        row(k.replace(".avg.pct_of_peak_sustained_active", "").replace(".avg.pct_of_peak_sustained_elapsed", ""),
            [f(raw[n].get(k)) for n in ok])

# 3) stall reasons per issued instruction (top by max)
st = {k for n in ok for k in raw[n]
      if k.startswith("smsp__average_warps_issue_stalled_") and k.endswith("_per_issue_active.ratio")}
if st:
    top = sorted(st, key=lambda k: -max(raw[n].get(k, 0) for n in ok))[:10]
    print("\n  warp stalls (cycles per issued instruction, top 10)")
    for k in top:
        row(k.replace("smsp__average_warps_issue_stalled_", "").replace("_per_issue_active.ratio", ""),
            [f(raw[n].get(k)) for n in ok])

# 4) the metrics that moved most between the first two variants
if len(ok) > 1:
    a, b = ok[0], ok[1]
    moved = []
    for k in set(raw[a]) & set(raw[b]):
        x, y = raw[a][k], raw[b][k]
        if max(abs(x), abs(y)) < 1e-9 or k.startswith(("launch__", "device__", "ID", "Process")):
            continue
        r = (y + 1e-12) / (x + 1e-12)
        if r > 0:
            moved.append((abs(math.log(r)), k, x, y))
    moved.sort(reverse=True)
    print(f"\n  biggest relative changes {a} -> {b} (top 25 of {len(moved)}; full list in moved.csv)")
    LW = min(72, max(46, max((len(k) for _, k, _, _ in moved[:25]), default=46)))
    for _, k, x, y in moved[:25]:
        row(k, [f(raw[n].get(k)) for n in ok], units.get(k, ""))
    LW = 46
    with open(os.path.join(d, "moved.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "unit"] + ok + [f"{b}/{a}"])
        for _, k, x, y in moved:
            w.writerow([k, units.get(k, "")] + [raw[n].get(k, "") for n in ok] + [f"{y / x:.4f}" if x else ""])

# 5) ncu's own section summaries (details page), side by side
if det:
    print("\n  ncu section summaries")
    keys = []
    for n in ok:
        for k in det.get(n, {}):
            if k not in keys and k[0] in ("GPU Speed Of Light Throughput", "Memory Workload Analysis",
                                          "Warp State Statistics", "Scheduler Statistics", "Occupancy"):
                keys.append(k)
    cur = None
    for k in keys:
        if k[0] != cur:
            cur = k[0]; print(f"   [{cur}]")
        row("  " + k[1], [f(det.get(n, {}).get(k)) for n in ok], k[2])
print(f"\n  reports: {d}/<variant>.ncu-rep (open in Nsight Compute; compare with 'Add Baseline')")
print("  how to read it:")
print("   - L2 sector ratio above the ideal byte ratio: the TMA reads more than the packed bytes")
print("     (box padding / alignment of sub-byte rows) -> fix the box/layout")
print("   - sector ratio ≈ ideal, time ratio ≈ sector ratio: bandwidth-bound, the format pays off fully")
print("   - sector ratio ≈ ideal, time ratio well above it: something else limits (lower L2 read GB/s;")
print("     look at stalls, tensor/TMA pipes and the 'biggest changes' list)")
PY
}

# Accuracy of each KV format: kernel output vs attention over the
# UNQUANTIZED float KV (same queries), as relative L2 error. fp8 runs with
# --kv_from_float so every format starts from the same kind of data.
stage_quant() {
  local fmts="${QUANT_FORMATS:-fp8 e3m2 e2m3 e2m1 nvfp4}" ls="${QUANT_LS:-4096 32768}"
  hr "KV format accuracy  (rel L2 error vs unquantized KV; fp8 Q, P in e4m3)"
  mkdir -p "$LOGDIR/quant"
  printf ' %s%-7s' "$B" fmt; local L f; for L in $ls; do printf ' %22s' "L=$L"; done; printf '%s\n' "$N"
  for f in $fmts; do
    printf ' %-7s' "$f"
    for L in $ls; do
      local log="$LOGDIR/quant/$f-L$L.log" extra=""
      [[ "$f" == fp8 ]] && extra="--kv_format fp8 --kv_from_float" || extra="--kv_format $f"
      t5_run "$log" fp8 --batches 2 --seqlen "$L" $extra
      local rel mx
      rel="$(sed -n 's/.*rel_l2=\([0-9.e+-]*\).*/\1/p' "$log" | head -1)"
      mx="$(sed -n 's/.*unquantized| max=\([0-9.e+-]*\) .*/\1/p' "$log" | head -1)"
      if [[ -n "$rel" ]]; then
        printf ' %12s (max %6s)' "$rel" "$mx"
        grep -qx PASS "$log" || { FAILED+=("quant $f L=$L (kernel != emulator)|$log"); printf '%s!%s' "$R" "$N"; }
      else
        printf ' %s%22s%s' "$R" "fail" "$N"; FAILED+=("quant $f L=$L|$log")
      fi
    done
    printf '\n'
  done
  info "   (lower is better; fp8 row = today's KV cache. '!' = kernel disagrees with its emulator)"
}

# PDL only changes how launches overlap, which ncu (one kernel at a time)
# cannot see -> harness wall-clock medians, same process, on vs off.
stage_pdl() {
  local ls="${PDL_LS:-8192 65536 262144}" bs="${PDL_BATCHES:-1 8}" iters="${PDL_ITERS:-200}"
  hr "PDL A/B  (harness median, cold L2, $iters iters)"
  mkdir -p "$LOGDIR/pdl"
  printf ' %s%-5s %-4s %-8s %12s %12s %9s%s\n' "$B" dtype b L "off" "on" "delta" "$N"
  local dt b L
  for dt in $DTYPES; do
    for b in $bs; do
      for L in $ls; do
        local off on args="--batches $b --seqlen $L --skip_ref_check --warmup_iterations 10 --iterations $iters"
        (( b > 1 )) && args+=" --varlen"
        t5_run "$LOGDIR/pdl/$dt-b$b-L$L-off.log" "$dt" $args
        t5_run "$LOGDIR/pdl/$dt-b$b-L$L-on.log" "$dt" $args --pdl
        off="$(sed -n 's/^median \([0-9.]*\) us.*/\1/p' "$LOGDIR/pdl/$dt-b$b-L$L-off.log" | head -1)"
        on="$(sed -n 's/^median \([0-9.]*\) us.*/\1/p' "$LOGDIR/pdl/$dt-b$b-L$L-on.log" | head -1)"
        if [[ -n "$off" && -n "$on" ]]; then
          local d; d="$(python3 -c "print(f'{($on-$off)/$off*100:+.1f}%')")"
          local col="$D"; [[ "$d" == -* ]] && col="$G"
          printf ' %-5s %-4s %-8s %9s us %9s us %s%9s%s\n' "$dt" "$b" "$L" "$off" "$on" "$col" "$d" "$N"
          TIMED+=("pdl $dt b=$b L=$L: off $off us, on $on us ($d)")
        else
          printf ' %-5s %-4s %-8s %s(failed — see %s)%s\n' "$dt" "$b" "$L" "$R" "$LOGDIR/pdl" "$N"
          FAILED+=("pdl $dt b=$b L=$L|$LOGDIR/pdl")
        fi
      done
    done
  done
  info "   (negative delta = PDL faster)"
}

# vllm_run <logfile> <timeout> <cmd...>: package at /work/src/vllm_thor_t5
# (canonical kernel mounted over its t5_decode.py copy), tests at
# /work/tests/thor_t5, PYTHONPATH=/work/src
vllm_run() {
  local log="$1" to="$2"; shift 2
  mkdir -p "$HF_HOME_HOST"
  local overlay=()
  [[ "$(realpath "$KERNEL")" != "$(realpath "$PKG/t5_decode.py")" ]] \
    && overlay=(-v "$KERNEL":/work/src/vllm_thor_t5/t5_decode.py:ro)
  local name="t5vllm-$$-$RANDOM"
  timeout "$to" docker run --rm --name "$name" --gpus all --ipc=host --entrypoint bash \
    -v "$PKG":/work/src/vllm_thor_t5 "${overlay[@]}" \
    -v "$TESTS":/work/tests/thor_t5 \
    -v "$HF_HOME_HOST":/root/.cache/huggingface \
    -e PYTHONPATH=/work/src -e HF_TOKEN="${HF_TOKEN:-}" -e TMPDIR=/tmp \
    -w /work "$IMG" -c "$*" > "$log" 2>&1
  local rc=$?
  reap "$rc" "$name"
  return $rc
}

stage_vllm() {
  hr "vLLM plugin  ($PKG)"
  local miss=0
  [[ -f "$PKG/__init__.py" && -f "$PKG/backend.py" ]] \
    || { echo " ${R}❌ no vllm_thor_t5 package at $PKG${N} (set PKG=...)"; miss=1; }
  [[ -f "$TESTS/test_runner.py" ]] \
    || { echo " ${R}❌ no tests at $TESTS${N} (put test_runner.py + smoke_vllm.py there, or set TESTS=...)"; miss=1; }
  if (( miss )); then FAILED+=("vllm package/tests|$PKG $TESTS"); return; fi
  if ! cmp -s "$KERNEL" "$PKG/t5_decode.py"; then
    echo " ${Y}⚠  KERNEL=$KERNEL is not the package's t5_decode.py — testing KERNEL${N}"
  fi
  mkdir -p "$LOGDIR/vllm"
  # --- runner tests (no vLLM needed) ---
  local log="$LOGDIR/vllm/test_runner.log" t0=$SECONDS line
  info "   runner vs emulator ($TESTS/test_runner.py)"
  vllm_run "$log" "$TIMEOUT_VLLM" python3 tests/thor_t5/test_runner.py
  local rc=$?
  while IFS= read -r line; do
    case "$line" in
      *"✅ test_"*) PASSED+=("runner ${line##*✅ }"); printf '   %s\n' "$line" ;;
      *"❌ test_"*) local t="${line##*❌ }"; FAILED+=("runner ${t%%:*}|$log")
                    printf '   %s%s%s\n' "$R" "$line" "$N" ;;
    esac
  done < "$log"
  if (( rc != 0 )) && ! grep -q '❌ test_' "$log"; then
    FAILED+=("runner tests (exit $rc)|$log")
    echo " ${R}❌ runner tests exit $rc${N}"; tail -5 "$log" | sed "s/^/    ${D}│ /;s/\$/${N}/"
  fi
  printf '   %s└ %s (%ss)%s\n' "$D" "$log" $(( SECONDS - t0 )) "$N"
  # --- end-to-end smoke ---
  local m
  for m in $SMOKE_MODES; do
    local args="--model $SMOKE_MODEL --max_tokens $SMOKE_TOKENS"
    case "$m" in
      graph) ;;
      eager) args+=" --eager" ;;
      spec*) args+=" --spec ${m#spec}" ;;
      *) echo " ${Y}unknown smoke mode $m${N}"; continue ;;
    esac
    log="$LOGDIR/vllm/smoke-$m.log"; t0=$SECONDS
    info "   smoke [$m]: CUSTOM vs FLASHINFER, $SMOKE_MODEL, fp8 KV"
    vllm_run "$log" "$TIMEOUT_VLLM" python3 tests/thor_t5/smoke_vllm.py $args
    rc=$?
    sed -n '/=== comparison/,$p' "$log" | sed 's/^/   /'
    if (( rc == 0 )) && grep -qx 'PASS' "$log"; then
      PASSED+=("vllm smoke $m")
    else
      local why="exit $rc"; (( rc == 124 )) && why="TIMEOUT ${TIMEOUT_VLLM}s"
      FAILED+=("vllm smoke $m ($why)|$log")
      echo " ${R}❌ smoke $m: $why${N}"
      grep -E -m4 'Error|error:|Traceback|Thor T5' "$log" | sed "s/^/    ${D}│ /;s/\$/${N}/"
    fi
    grep -m3 'Thor T5' "$log" | sed "s/^/    ${D}│ /;s/\$/${N}/"
    printf '   %s└ %s (%ss)%s\n' "$D" "$log" $(( SECONDS - t0 )) "$N"
  done
}

stage_report() {  # stage_report <sweep-out-dir> [LEGS]
  local dir="${1:?usage: ./verify.sh report <sweep-dir>}"
  REPORT_ONLY=1 OUT="$dir" LEGS="${LEGS:-fi t5fp8}" bash "$SWEEP" \
    | sed -n '/T5 decode L-sweep/,$p'
}

# --------------------------------------------------------------- summary ---
summary() {
  echo
  hr "summary"
  local np=${#PASSED[@]} nf=${#FAILED[@]}
  if (( np + nf )); then
    if (( nf == 0 )); then
      echo " ${G}${B}✅ all $np checks passed${N}"
    else
      echo " ${R}${B}❌ $nf failed${N}, ${G}$np passed${N}"
      local f
      for f in "${FAILED[@]}"; do
        printf '   %s❌ %s%s  %s%s%s\n' "$R" "${f%%|*}" "$N" "$D" "${f#*|}" "$N"
      done
    fi
  fi
  [[ -d "$LOGDIR/sweeps" ]] && {
    local d
    for d in "$LOGDIR"/sweeps/*/; do
      [[ -f "$d/summary.md" ]] && printf '   📝 %s\n' "${d%/}/summary.md"
    done
  }
  printf '   📁 %s\n' "$LOGDIR"
  {
    echo "verify.sh $TS"
    echo "kernel: $KERNEL"
    echo "passed: ${#PASSED[@]}  failed: ${#FAILED[@]}"
    printf 'PASS %s\n' "${PASSED[@]:-}"
    printf 'FAIL %s\n' "${FAILED[@]:-}"
    printf 'TIME %s\n' "${TIMED[@]:-}"
  } > "$LOGDIR/verify-summary.txt"
}

# ------------------------------------------------------------------ main ---
main() {
  local stages=("$@")
  (( ${#stages[@]} )) || stages=(correctness timing sweep)
  if [[ "${stages[0]}" == "report" ]]; then
    stage_report "${stages[@]:1}"; exit $?
  fi
  [[ "${stages[0]}" == "all" ]] && stages=(correctness timing sweep diag)
  local s
  for s in "${stages[@]}"; do
    case "$s" in
      correctness|timing|sweep|diag|ring|vllm|prefetch|pdl|ab|kvfmt|quant|hndsweep|fp6prof) ;;
      -h|--help|help) sed -n '2,80p' "$0"; exit 0 ;;
      *) echo "unknown stage '$s' (correctness|timing|sweep|diag|ring|vllm|prefetch|pdl|ab|kvfmt|quant|hndsweep|fp6prof|all|report <dir>)"; exit 2 ;;
    esac
  done
  preflight
  for s in "${stages[@]}"; do "stage_$s"; echo; done
  summary
  (( ${#FAILED[@]} == 0 ))
}

main "$@"
