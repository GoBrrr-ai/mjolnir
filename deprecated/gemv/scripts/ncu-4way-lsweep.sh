#!/usr/bin/env bash
# 4-way ncu L-sweep — the full decode kernel comparison across the
# compute-bound → DRAM-bound transition:
#   FI-tc       : FlashInfer FA2 TC prefill (plan-once, split-KV ON) — the reference.
#   GEMV-t1ns20 : the shipped GEMV (tile1, 48 KiB ring, ns=20 -> 80 CTAs = 4/SM).
#   _spark-TC   : our mma.sync QK/PV TC (128-thread 1x4-warp, ns=20) — the 20x-L2 bug.
#   (c) cvt     : the tcgen05+TMEM+cvt decode (one-time fp8->bf16 cvt, 1/SM one-wave).
# Per L in LS, one ncu profile per kernel. The per-kernel gpu__time_duration.sum
# (base clock) + lts__t_sectors.sum (the memory-access check) + the launched CTAs
# (the one-wave check) are the robust metrics (dram__bytes is n/a on Thor).
#
# CO-TENANCY: --clock-control base locks the SM/memory clocks; DRAM BW is shared,
# so lean on the RELATIVE ratios + L2 sectors, not absolute wall-clock. Schedule
# when the box is idle (vLLM running/waiting == 0).
#
# Env: VFA, NCU_HOME, IMG, OUT, LS (space-separated, default "8192 16384 32768
# 65536 131072"), ITERS (the (c) kernel's --iterations, default 20).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK="${TASK:-$ROOT/task/fa4-hd256-fp8}"
GEMV_DIR="$ROOT/docker/vllm-thor/fa4-gemv-kernel"
VFA="${VFA:-$ROOT/vfa-tree}"
NCU_HOME="${NCU_HOME:-/opt/nvidia/nsight-compute/2025.3.0}"
IMG="${IMG:-narandill/mjolnir-thor:v15}"
OUT="${OUT:-$ROOT/benchmarks/raw/ncu-4way-lsweep}"
LS="${LS:-8192 16384 32768 65536 131072}"
NCU="/ncu/ncu"
ITERS="${ITERS:-20}"
mkdir -p "$OUT"

METRICS="gpu__time_duration.sum,lts__t_sectors.sum,sm__cycles_active.sum,sm__ctas_launched.sum"

# run_ncu <tag> <python-cmd> <env...>: one docker run, ncu --clock-control base.
run_ncu() {
  local tag="$1"; local cmd="$2"; shift 2
  local envargs=()
  for e in "$@"; do envargs+=(-e "$e"); done
  echo ""
  echo "############ ncu: $tag ############"
  docker run --rm --gpus all --network host \
    --security-opt seccomp=unconfined --cap-add SYS_ADMIN \
    --entrypoint bash \
    -v "$VFA/vllm_flash_attn":/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$TASK":/p \
    -v "$GEMV_DIR/sm100_hd256_decode_tc_spark_cvt.py":/cvt.py:ro \
    -v "$NCU_HOME":/ncu:ro \
    "${envargs[@]}" \
    -e TMPDIR=/tmp/ncu-4w \
    "$IMG" \
    -c "mkdir -p /tmp/ncu-4w && $NCU --target-processes all --clock-control base --cache-control none --print-summary per-kernel --metrics $METRICS $cmd" \
    | tee "$OUT/$tag.txt"
}

for L in $LS; do
  run_ncu "fi-tc-L$L"     "python3 /p/fi-ncu-driver.py" FI_MODE=tc FI_L=$L
  run_ncu "gemv-L$L"      "python3 /p/ncu-gemv-ring.py" NCU_L=$L NCU_NS=20 NCU_STAGES=16 VLLM_FA4_HD256_GEMV_TILE=1
  run_ncu "spark-tc-L$L"  "python3 /p/ncu-tc-spark-ring.py" NCU_L=$L NCU_NS=20
  run_ncu "cvt-L$L"       "python3 /cvt.py --seqlen $L --iterations $ITERS --warmup_iterations 5 --skip_ref_check"
done

echo ""
echo "==================== 4-way L-sweep summary (per-kernel time + sectors + CTAs) ===================="
extract() {
  local f="$1" pat="$2"
  if [[ -f "$f" ]]; then
    grep -A4 -E "$pat" "$f" | grep -E "$pat|Invocations|gpu__time_duration|lts__t_sectors|sm__ctas_launched" | head -9 || true
  else
    echo "  (missing $f)"
  fi
}
for L in $LS; do
  echo "=== L=$L ==="
  echo " FI-tc:";      extract "$OUT/fi-tc-L$L.txt"     "flash|FlashInfer|FA2|fmha|attention"
  echo " GEMV:";       extract "$OUT/gemv-L$L.txt"       "sm100_hd256_decode_gemv"
  echo " _spark_tc:";  extract "$OUT/spark-tc-L$L.txt"   "sm100_hd256_decode_tc_spark"
  echo " (c) cvt:";    extract "$OUT/cvt-L$L.txt"        "MixedInputFusedMultiHeadAttentionDecode"
done
echo ""
echo "Output in: $OUT"
