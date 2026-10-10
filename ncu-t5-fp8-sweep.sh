#!/usr/bin/env bash
# T5 4-way ncu L-sweep — the fully-fp8 T5 vs the baseline legs:
#   FI-tc   : FlashInfer FA2 TC decode (plan-once, split-KV ON) — the reference.
#   FI-NHD / FI-HND (legs finhd / fihnd): FlashInfer TC decode from
#             tests/thor_t5/fi_layout_driver.py on the SAME data in NHD vs HND
#             (head-major) layout — the fair baseline for T5 --kv_layout hnd.
#   GEMV    : the shipped pure-FMA GEMV (tile1, 48 KiB ring, ns=20 -> 80 CTAs).
#   T5-bf16 : T5 tcgen05 kernel, cvt path: Q bf16, KV fp8 -> bf16 cvt, bf16 mma.
#   T5-fp8  : T5 tcgen05 kernel, fp8-Q path, no cvt. KV per the harness default:
#             e2m3 (fp6) on head-major HND pages; pass T5FP8_EXTRA="--kv_format
#             fp8 --kv_layout nhd" for the old fp8/NHD leg. Labels show the KV
#             config and GB/s counts each leg's real KV bytes.
#
# Per L in LS, one ncu profile per leg. Metrics (dram__bytes is n/a on Thor):
# gpu__time_duration.sum, lts__t_sectors.sum (memory-access check),
# sm__ctas_launched.sum (wave check), sm__cycles_active.sum.
#
# FAIRNESS: every leg runs under the SAME cache policy. Default CACHE=all makes
# ncu flush L2 before every profiled kernel (cold, DRAM-bound — the decode
# steady state once the KV cache exceeds Thor's 32 MiB L2). CACHE=none keeps
# caches warm for every leg (KV <= ~16K tokens then lives in L2). T5's own
# 64 MB flush is disabled (--no_l2_flush) so it can't skew the comparison.
# Only the kernels of interest are profiled (FILTER_*), so fill/flush/warmup
# kernels never enter the averages.
#
# CO-TENANCY: --clock-control base locks clocks; DRAM BW is shared, so lean on
# RELATIVE ratios + L2 sectors. Run when the box is idle (`mjolnir gate --once`).
#
# Output: per-leg ncu logs, a coloured terminal report, and
#   $OUT/summary.md   (markdown report: comparison table + per-kernel detail)
#   $OUT/summary.csv  (one row per kernel per leg per L)
#
# Env:
#   VFA, NCU_HOME, IMG, OUT
#   T5_KERNEL   the T5 kernel/harness file (default $ROOT/src/vllm_thor_t5/t5_decode.py,
#               the copy vLLM loads)
#   LS          space-separated L values  (default "128 4096 8192 ... 262144")
#   ITERS       profiled iterations per leg (default 50: on this box single cells move
#               by ~±7% between runs at 20)
#   CACHE       all | none                (default all)
#   CLOCK       base | none               (default base)
#   HQ HK D     heads_q / heads_k / headdim passed to T5 (default 24 4 256;
#               keep in sync with the FI / GEMV drivers)
#   T5_CTAS     T5 --ctas_per_sm          (default 2 -> 40 CTAs on 20 SMs)
#   T5_EXTRA    extra args for both T5 legs (e.g. "--kv_splits 10")
#   T5FP8_EXTRA / T5BF16_EXTRA   extra args per T5 leg. Each leg already runs
#               with its vLLM default (the harness auto-enables --kv_evict_first
#               for fp8 Q only); e.g. T5BF16_EXTRA="--kv_evict_first 1" forces it
#   NOISE_PCT   ratio band drawn as 🟰 "within noise" (default 7)
#   LEGS        subset of "fi finhd fihnd gemv t5bf16 t5fp8" (default "fi t5fp8";
#               the pure-FMA GEMV is retired — ~0.5x FI everywhere — add it explicitly)
#   FI_DRIVER   NHD/HND FlashInfer driver (default $ROOT/tests/thor_t5/fi_layout_driver.py)
#   FI_PAGE     page size for finhd/fihnd (default 128, as the T5 harness)
#   BASE_LEG    leg the "vs FI" ratios/geomeans use (default: fi, else finhd, else fihnd)
#   FILTER_FI / FILTER_GEMV / FILTER_T5   ncu --kernel-name filters
#               (set to "" to profile every kernel of that leg)
#   PEAK_GBS    DRAM peak used for %-of-peak (default 262.8)
#   REPORT_ONLY=1  skip profiling, just rebuild the report from $OUT
#   NO_COLOR=1 / FORCE_COLOR=1   terminal colour control
set -uo pipefail   # no -e: one failing leg must not abort the sweep

# Repo root: $ROOT if set; else the script's own dir when it sits in the repo
# root (has src/vllm_thor_t5 or docker/vllm-thor); else two levels up
# (task/<name>/ layout).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${ROOT:-}" ]]; then
  if [[ -d "$SCRIPT_DIR/src/vllm_thor_t5" || -d "$SCRIPT_DIR/docker/vllm-thor" ]]; then
    ROOT="$SCRIPT_DIR"
  else ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"; fi
fi
TASK="${TASK:-$ROOT/task/fa4-hd256-fp8}"   # holds fi-ncu-driver.py, ncu-gemv-ring.py
T5_KERNEL="${T5_KERNEL:-$ROOT/src/vllm_thor_t5/t5_decode.py}"
FI_DRIVER="${FI_DRIVER:-$ROOT/tests/thor_t5/fi_layout_driver.py}"
FI_PAGE="${FI_PAGE:-128}"
VFA="${VFA:-$ROOT/vfa-tree}"
NCU_HOME="${NCU_HOME:-/opt/nvidia/nsight-compute/2025.3.0}"
IMG="${IMG:-narandill/mjolnir-thor:v15}"
OUT="${OUT:-$ROOT/benchmarks/raw/ncu-t5-fp8-sweep}"
LS="${LS:-128 4096 8192 16384 32768 65536 131072 262144}"
ITERS="${ITERS:-50}"
CACHE="${CACHE:-all}"
CLOCK="${CLOCK:-base}"
HQ="${HQ:-24}"; HK="${HK:-4}"; D="${D:-256}"
T5_CTAS="${T5_CTAS:-2}"
T5_EXTRA="${T5_EXTRA:-}"
T5FP8_EXTRA="${T5FP8_EXTRA:-}"
T5BF16_EXTRA="${T5BF16_EXTRA:-}"
NOISE_PCT="${NOISE_PCT:-7}"
LEGS="${LEGS:-fi t5fp8}"
FILTER_FI="${FILTER_FI-regex:BatchPrefillWithPagedKVCache|BatchDecode|MergeStates}"
FILTER_GEMV="${FILTER_GEMV-regex:gemv|combine}"
FILTER_T5="${FILTER_T5-regex:MixedInputFusedMultiHeadAttentionDecode|reduction}"
PEAK_GBS="${PEAK_GBS:-262.8}"
REPORT_ONLY="${REPORT_ONLY:-0}"
NCU="/ncu/ncu"
mkdir -p "$OUT"
if [[ "$REPORT_ONLY" != "1" && ( " $LEGS " == *" t5bf16 "* || " $LEGS " == *" t5fp8 "* ) \
      && ! -f "$T5_KERNEL" ]]; then
  echo "T5 kernel not found: $T5_KERNEL (set T5_KERNEL=...)" >&2; exit 2
fi
if [[ "$REPORT_ONLY" != "1" && ( " $LEGS " == *" finhd "* || " $LEGS " == *" fihnd "* ) \
      && ! -f "$FI_DRIVER" ]]; then
  echo "FI layout driver not found: $FI_DRIVER (set FI_DRIVER=...)" >&2; exit 2
fi

case "$CACHE" in all|none) ;; *) echo "CACHE must be all|none" >&2; exit 2 ;; esac
case "$CLOCK" in base|none) ;; *) echo "CLOCK must be base|none" >&2; exit 2 ;; esac

METRICS="gpu__time_duration.sum,lts__t_sectors.sum,sm__cycles_active.sum,sm__ctas_launched.sum"
FAILED=()

has_leg() { [[ " $LEGS " == *" $1 "* ]]; }

# run_ncu <tag> <kernel-filter> <python-cmd> <env...>
run_ncu() {
  local tag="$1" filter="$2" cmd="$3"; shift 3
  local envargs=()
  for e in "$@"; do envargs+=(-e "$e"); done
  local kf=""
  [[ -n "$filter" ]] && kf="--kernel-name '$filter'"
  echo ""
  echo "############ ncu: $tag (cache=$CACHE clock=$CLOCK) ############"
  docker run --rm --gpus all --network host \
    --security-opt seccomp=unconfined --cap-add SYS_ADMIN \
    --entrypoint bash \
    -v "$VFA/vllm_flash_attn":/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn \
    -v "$TASK":/p \
    -v "$T5_KERNEL":/t5.py:ro \
    -v "$FI_DRIVER":/fi_layout.py:ro \
    -v "$NCU_HOME":/ncu:ro \
    "${envargs[@]}" \
    -e TMPDIR=/tmp/ncu-t5 \
    "$IMG" \
    -c "mkdir -p /tmp/ncu-t5 && $NCU --target-processes all \
        --clock-control $CLOCK --cache-control $CACHE $kf \
        --print-summary per-kernel --metrics $METRICS $cmd" \
    2>&1 | tee "$OUT/$tag.txt"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "!!! leg $tag failed (exit $rc) — continuing" >&2
    FAILED+=("$tag")
  fi
}

T5_COMMON="--heads_q $HQ --heads_k $HK --headdim $D --ctas_per_sm $T5_CTAS \
  --iterations $ITERS --warmup_iterations 0 --skip_ref_check --no_l2_flush $T5_EXTRA"

if [[ "$REPORT_ONLY" != "1" ]]; then
  for L in $LS; do
    has_leg fi && run_ncu "fi-tc-L$L" "$FILTER_FI" \
      "python3 /p/fi-ncu-driver.py" FI_MODE=tc FI_L=$L FI_ITERS=$ITERS
    for lay in nhd hnd; do
      has_leg "fi$lay" && run_ncu "fi-$lay-L$L" "$FILTER_FI" \
        "python3 /fi_layout.py" FI_LAYOUT=$lay FI_L=$L FI_ITERS=$ITERS \
        FI_HQ=$HQ FI_HK=$HK FI_D=$D FI_PAGE=$FI_PAGE
    done
    has_leg gemv && run_ncu "gemv-L$L" "$FILTER_GEMV" \
      "python3 /p/ncu-gemv-ring.py" NCU_L=$L NCU_NS=20 NCU_STAGES=16 NCU_ITERS=$ITERS \
      VLLM_FA4_HD256_GEMV_TILE=1
    has_leg t5bf16 && run_ncu "t5-bf16-L$L" "$FILTER_T5" \
      "python3 /t5.py --seqlen $L --q_dtype BFloat16 --kv_dtype Float8E4M3FN $T5_COMMON $T5BF16_EXTRA"
    has_leg t5fp8 && run_ncu "t5-fp8-L$L" "$FILTER_T5" \
      "python3 /t5.py --seqlen $L --q_dtype Float8E4M3FN --kv_dtype Float8E4M3FN $T5_COMMON $T5FP8_EXTRA"
  done
fi

##############################################################################
# Report: terminal (colour + icons + bars) and markdown/csv export
##############################################################################
OUT="$OUT" LS="$LS" LEGS="$LEGS" HQ="$HQ" HK="$HK" D="$D" ITERS="$ITERS" \
CACHE="$CACHE" CLOCK="$CLOCK" PEAK_GBS="$PEAK_GBS" T5_CTAS="$T5_CTAS" \
T5_EXTRA="$T5_EXTRA" T5FP8_EXTRA="$T5FP8_EXTRA" T5BF16_EXTRA="$T5BF16_EXTRA" \
NOISE_PCT="$NOISE_PCT" BASE_LEG="${BASE_LEG:-}" FAILED_LEGS="${FAILED[*]:-}" \
python3 - <<'PY'
import csv, datetime, math, os, re, sys, unicodedata

OUT = os.environ["OUT"]
LS = [int(x) for x in os.environ["LS"].split()]
SEL = os.environ["LEGS"].split()
HQ, HK, D = (int(os.environ[k]) for k in ("HQ", "HK", "D"))
ITERS = int(os.environ["ITERS"])
CACHE, CLOCK = os.environ["CACHE"], os.environ["CLOCK"]
PEAK = float(os.environ["PEAK_GBS"])
FAILED = set(os.environ.get("FAILED_LEGS", "").split())
T5_CTAS, T5_EXTRA = os.environ.get("T5_CTAS", ""), os.environ.get("T5_EXTRA", "")
T5FP8_EXTRA = os.environ.get("T5FP8_EXTRA", "").strip()
T5BF16_EXTRA = os.environ.get("T5BF16_EXTRA", "").strip()
NOISE = float(os.environ.get("NOISE_PCT", "7")) / 100.0

LEG_DEFS = {
    "fi":     dict(label="FI-tc",   file="fi-tc-L{L}.txt",
                   main=("BatchPrefillWithPagedKVCache", "BatchDecode"),
                   merge=("MergeStates",), merge_label="merge"),
    "finhd":  dict(label="FI-NHD",  file="fi-nhd-L{L}.txt",
                   main=("BatchPrefillWithPagedKVCache", "BatchDecode"),
                   merge=("MergeStates",), merge_label="merge"),
    "fihnd":  dict(label="FI-HND",  file="fi-hnd-L{L}.txt",
                   main=("BatchPrefillWithPagedKVCache", "BatchDecode"),
                   merge=("MergeStates",), merge_label="merge"),
    "gemv":   dict(label="GEMV",    file="gemv-L{L}.txt",
                   main=("sm100_hd256_decode_gemv",),
                   merge=("flash_fwd_combine",), merge_label="combine"),
    "t5bf16": dict(label="T5-bf16", file="t5-bf16-L{L}.txt",
                   main=("MixedInputFusedMultiHeadAttentionDecode",),
                   merge=("kernel_cutlass_reduction",), merge_label="reduce"),
    "t5fp8":  dict(label="T5-fp8",  file="t5-fp8-L{L}.txt",
                   main=("MixedInputFusedMultiHeadAttentionDecode",),
                   merge=("kernel_cutlass_reduction",), merge_label="reduce"),
}
LEGS = [k for k in SEL if k in LEG_DEFS]
FI_LEGS = [k for k in LEGS if k in ("fi", "finhd", "fihnd")]
BASE = os.environ.get("BASE_LEG") or (FI_LEGS[0] if FI_LEGS else "fi")
BASE_LABEL = LEG_DEFS[BASE]["label"] if BASE in LEG_DEFS else BASE

# ---------------------------------------------------------------- colours ---
USE_COLOR = (sys.stdout.isatty() or os.environ.get("FORCE_COLOR")) and not os.environ.get("NO_COLOR")
def sgr(code):
    return (lambda s: f"\x1b[{code}m{s}\x1b[0m") if USE_COLOR else (lambda s: s)
bold, dim = sgr("1"), sgr("2")
green, yellow, red, cyan, mag = sgr("32"), sgr("33"), sgr("31"), sgr("36"), sgr("35")
bgreen = sgr("1;32")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
def vw(s):
    """Visible terminal width (ANSI-free, wide emoji = 2 columns)."""
    w = 0
    for ch in ANSI.sub("", s):
        if unicodedata.combining(ch) or ch in "‍️":
            continue
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w
def pad(s, width, right=False):
    gap = max(0, width - vw(s))
    return (" " * gap + s) if right else (s + " " * gap)

# ---------------------------------------------------------------- parsing ---
UNIT_US = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0,
           "ms": 1e3, "msecond": 1e3, "s": 1e6, "second": 1e6}
def num(x):
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return float("nan")

def parse_ncu(path):
    """Kernel rows from an ncu --print-summary per-kernel log."""
    rows, cur = [], None
    with open(path, errors="replace") as f:
        for line in f:
            if "Invocations" in line and "Device" in line:
                m = re.search(r"Invocations\s+(\d+)", line)
                cur = dict(header=line.strip(),
                           name=line.strip().split(" (")[0],
                           inv=int(m.group(1)) if m else 0)
                rows.append(cur)
                continue
            if cur is None:
                continue
            p = line.split()
            if len(p) >= 5 and p[0] in ("gpu__time_duration.sum", "lts__t_sectors.sum",
                                         "sm__ctas_launched.sum", "sm__cycles_active.sum"):
                unit, mn, avg = p[1], num(p[2]), num(p[-1])
                if p[0] == "gpu__time_duration.sum":
                    k = UNIT_US.get(unit, 1.0)
                    cur["avg_us"], cur["min_us"] = avg * k, mn * k
                elif p[0] == "lts__t_sectors.sum":
                    cur["sectors"] = avg
                elif p[0] == "sm__ctas_launched.sum":
                    cur["ctas"] = avg
    return [r for r in rows if "avg_us" in r]

KV_BITS = {"fp8": 8, "e3m2": 6, "e2m3": 6, "e2m1": 4, "nvfp4": 4.5}
def t5_kv(leg):
    """(format, layout) a T5 leg ran with: harness defaults, then T5_EXTRA,
    then the leg's own extra (last flag wins, as in argparse)."""
    fmt, lay = ("e2m3" if leg == "t5fp8" else "fp8"), "hnd"
    for args in (T5_EXTRA, T5FP8_EXTRA if leg == "t5fp8" else T5BF16_EXTRA):
        for m in re.finditer(r"--kv_format\s+(\S+)", args):
            fmt = m.group(1)
        for m in re.finditer(r"--kv_layout\s+(\S+)", args):
            lay = m.group(1)
    return fmt, lay
for _leg in ("t5fp8", "t5bf16"):
    _f, _l = t5_kv(_leg)
    LEG_DEFS[_leg]["label"] += f" {_f}·{_l}"
    LEG_DEFS[_leg]["bits"] = KV_BITS.get(_f, 8)

def kv_bytes(L, leg=None):
    """K + V bytes of one sequence (fp8 unless the leg reads a narrower format)."""
    bits = LEG_DEFS.get(leg, {}).get("bits", 8)
    return L * HK * D * 2 * bits / 8

data, warnings, inv_notes = {}, [], {}
for L in LS:
    for leg in LEGS:
        d = LEG_DEFS[leg]
        path = os.path.join(OUT, d["file"].format(L=L))
        tag = d["file"].format(L=L)[:-4]
        if tag in FAILED:
            warnings.append(f"❌ {d['label']} @ L={L:,}: leg failed (see {os.path.basename(path)})")
        if not os.path.exists(path):
            if tag not in FAILED:
                warnings.append(f"❌ {d['label']} @ L={L:,}: missing {os.path.basename(path)}")
            continue
        rows = parse_ncu(path)
        main = [r for r in rows if any(p in r["header"] for p in d["main"])]
        merge = [r for r in rows if any(p in r["header"] for p in d["merge"]) and r not in main]
        if not main:
            warnings.append(f"❌ {d['label']} @ L={L:,}: no matching kernel rows "
                            f"(check the ncu filter / kernel names)")
            continue
        total = sum(r["avg_us"] for r in main) + sum(r["avg_us"] for r in merge)
        gbps = kv_bytes(L, leg) / (total * 1e3) if total > 0 else 0.0
        data[(L, leg)] = dict(main=main, merge=merge, total=total, gbps=gbps)
        if CACHE == "all" and gbps > PEAK * 1.02:
            warnings.append(f"❗ {d['label']} @ L={L:,}: {gbps:.0f} GB/s exceeds the "
                            f"{PEAK:.0f} GB/s DRAM peak — L2 hits, not a cold measurement")
        for r in main + merge:
            if r["inv"] and r["inv"] != ITERS:
                inv_notes.setdefault(leg, []).append((L, r["inv"]))
                break

for leg, notes in inv_notes.items():
    counts = sorted({n for _, n in notes})
    where = ", ".join(f"{L:,}" for L, _ in notes)
    warnings.append(f"💬 {LEG_DEFS[leg]['label']}: kernels profiled "
                    f"{'/'.join(str(c) for c in counts)}× instead of {ITERS} at L = {where} "
                    f"(driver warmups/extra launches — averages include them)")

# ---------------------------------------------------------------- helpers ---
def fmt_us(t):
    return f"{t/1000:.2f} ms" if t >= 1000 else f"{t:.2f} µs"
def pct_color(p):
    return green if p >= 70 else yellow if p >= 40 else red
def ratio_color(r):
    return green if r >= 1.0 else yellow if r >= 0.9 else red
def ratio_icon(r):
    return "🚀" if r >= 1 + NOISE else "🟰" if r >= 1 - NOISE else "🐢"
BLOCKS = "▏▎▍▌▋▊▉█"
def bar(frac, width=22):
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    rem = frac * width - full
    s = "█" * full
    if full < width:
        s += BLOCKS[int(rem * 8) - 1] if rem >= 1 / 8 else ""
    return s + "·" * (width - vw(s))
def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None

def best_leg(L):
    present = [(data[(L, k)]["total"], k) for k in LEGS if (L, k) in data]
    return min(present)[1] if present else None
def worst_leg(L):
    present = [(data[(L, k)]["total"], k) for k in LEGS if (L, k) in data]
    return max(present)[1] if len(present) > 1 else None

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
cache_icon = "🧊 cold L2 (ncu flush)" if CACHE == "all" else "🔥 warm L2 (no flush)"
cfg = (f"{cache_icon} · clock={CLOCK} · HQ={HQ} HK={HK} D={D} · iters={ITERS} · "
       f"T5 ctas/SM={T5_CTAS}{(' · T5 ' + T5_EXTRA) if T5_EXTRA.strip() else ''}"
       f"{(' · fp8 ' + T5FP8_EXTRA) if T5FP8_EXTRA else ''}"
       f"{(' · bf16 ' + T5BF16_EXTRA) if T5BF16_EXTRA else ''} · "
       f"evict_first: fp8 on, bf16 off unless forced · "
       f"GB/s = each leg's real KV bytes (fp8 KV size shown per L) · "
       f"peak {PEAK:.1f} GB/s")

# ---------------------------------------------------------------- terminal --
W = 100
print()
print(bold(cyan("╔" + "═" * (W - 2) + "╗")))
print(bold(cyan("║")) + pad(bold("  📊 T5 decode L-sweep") + dim(f"   {now}"), W - 2) + bold(cyan("║")))
cfg_lines, cur = [], ""
for part in cfg.split(" · "):
    cand = part if not cur else cur + " · " + part
    if vw(cand) > W - 6 and cur:
        cfg_lines.append(cur)
        cur = part
    else:
        cur = cand
cfg_lines.append(cur)
for cl in cfg_lines:
    print(bold(cyan("║")) + pad("  " + cl, W - 2) + bold(cyan("║")))
print(bold(cyan("╚" + "═" * (W - 2) + "╝")))

for L in LS:
    present = [k for k in LEGS if (L, k) in data]
    head = f" L = {L:,} tokens · KV {kv_bytes(L)/1e6:.1f} MB "
    print()
    print(bold(mag("━━" + head + "━" * max(0, W - 2 - vw(head)))))
    if not present:
        print(dim("   (no data)"))
        continue
    best, worst = best_leg(L), worst_leg(L)
    LBLW = max(9, max(vw(LEG_DEFS[k]["label"]) for k in present))
    fi = data.get((L, BASE))
    for k in sorted(present, key=lambda k: data[(L, k)]["total"]):
        e, d = data[(L, k)], LEG_DEFS[k]
        icon = "🏆" if k == best else "🐢" if k == worst else "  "
        pctp = 100 * e["gbps"] / PEAK
        label = bgreen(d["label"]) if k == best else red(d["label"]) if k == worst else bold(d["label"])
        if fi and k != BASE:
            r = fi["total"] / e["total"]
            vs = ratio_color(r)(f"{r:5.2f}× vs {BASE_LABEL} {ratio_icon(r)}")
        elif k == BASE:
            vs = dim("   baseline")
        else:
            vs = ""
        ctas = int(e["main"][0].get("ctas", 0))
        gbps_s = "%.1f GB/s" % e["gbps"]   # no nested same-quote f-strings (py < 3.12)
        pct_s = "%4.0f%%" % pctp
        print(f" {icon} {pad(label, LBLW)} {pad(fmt_us(e['total']), 11, right=True)}  "
              f"{pct_color(pctp)(bar(e['gbps'] / PEAK))} "
              f"{pad(gbps_s, 11, right=True)} {pct_color(pctp)(pct_s)}  "
              f"{pad(vs, 18)} {dim(f'{ctas} CTAs')}")
        parts = [f"decode {fmt_us(r['avg_us'])} (min {fmt_us(r['min_us'])})" for r in e["main"]]
        parts += [f"{d['merge_label']} {fmt_us(r['avg_us'])}" for r in e["merge"]]
        print(dim("      └ " + " · ".join(parts)))

# comparison table
print()
cols = ["L"] + [LEG_DEFS[k]["label"] for k in LEGS]
has_ratio = BASE in LEGS and "t5fp8" in LEGS
has_best = len(FI_LEGS) > 1 and "t5fp8" in LEGS
def best_fi(L):
    ts = [data[(L, k)]["total"] for k in FI_LEGS if (L, k) in data]
    return min(ts) if ts else None
if has_ratio:
    cols.append(f"T5-fp8 vs {BASE_LABEL}")
if has_best:
    cols.append("T5-fp8 vs best FI")
cells = []
for L in LS:
    row = [f"{L:,}"]
    best = best_leg(L)
    for k in LEGS:
        e = data.get((L, k))
        if not e:
            row.append(dim("—"))
            continue
        s = f"{fmt_us(e['total'])} · {e['gbps']:.0f} GB/s"
        row.append(bgreen("🏆 " + s) if k == best else s)
    if has_ratio:
        fi, t5 = data.get((L, BASE)), data.get((L, "t5fp8"))
        if fi and t5:
            r = fi["total"] / t5["total"]
            row.append(ratio_color(r)(f"{ratio_icon(r)} {r:.2f}×"))
        else:
            row.append(dim("—"))
    if has_best:
        b, t5 = best_fi(L), data.get((L, "t5fp8"))
        if b and t5:
            r = b / t5["total"]
            row.append(ratio_color(r)(f"{ratio_icon(r)} {r:.2f}×"))
        else:
            row.append(dim("—"))
    cells.append(row)
geo = [f"geomean vs {BASE_LABEL}"]
for k in LEGS:
    g = geomean([data[(L, BASE)]["total"] / data[(L, k)]["total"]
                 for L in LS if (L, BASE) in data and (L, k) in data])
    geo.append(dim("—") if g is None else ratio_color(g)(f"{g:.2f}×"))
if has_ratio:
    geo.append("")
if has_best:
    g = geomean([best_fi(L) / data[(L, "t5fp8")]["total"]
                 for L in LS if best_fi(L) and (L, "t5fp8") in data])
    geo.append(dim("—") if g is None else ratio_color(g)(f"{g:.2f}×"))
widths = [max(vw(c[i]) for c in [cols] + cells + [geo]) for i in range(len(cols))]
def line(l, m, r):
    return dim(l + m.join("─" * (w + 2) for w in widths) + r)
def trow(row, hdr=False):
    out = []
    for i, c in enumerate(row):
        c = bold(c) if hdr else c
        out.append(" " + pad(c, widths[i], right=(i > 0 and not hdr)) + " ")
    return dim("│") + dim("│").join(out) + dim("│")
print(bold("  📋 Comparison — decode + merge per step (time · effective KV bandwidth)"))
print(line("┌", "┬", "┐"))
print(trow(cols, hdr=True))
print(line("├", "┼", "┤"))
for row in cells:
    print(trow(row))
print(line("├", "┼", "┤"))
print(trow(geo))
print(line("└", "┴", "┘"))
print(dim(f"  🚀 ≥{1 + NOISE:.2f}× faster than {BASE_LABEL}   🟰 within ±{NOISE * 100:.0f}% (box noise)   "
          f"🐢 slower   🏆 fastest at that L   "
          "bars = % of DRAM peak"))

if warnings:
    print()
    print(bold(yellow("  ⚠  Warnings")))
    for w in warnings:
        print("   " + w)

# ---------------------------------------------------------------- markdown --
md = []
md.append(f"# 📊 T5 decode L-sweep — {now}\n")
md.append(f"**Config:** {cfg}  \n")
md.append("Times are ncu `gpu__time_duration.sum` averages (decode + merge kernels); "
          "bandwidth is fp8 K+V bytes ÷ time.\n")
md.append("## 📋 Comparison\n")
hdr = (["L"] + [LEG_DEFS[k]["label"] for k in LEGS] + ([f"T5-fp8 vs {BASE_LABEL}"] if has_ratio else [])
       + (["T5-fp8 vs best FI"] if has_best else []))
md.append("| " + " | ".join(hdr) + " |")
md.append("|" + "|".join(["--:"] * len(hdr)) + "|")
for L in LS:
    row = [f"{L:,}"]
    best = best_leg(L)
    for k in LEGS:
        e = data.get((L, k))
        if not e:
            row.append("—")
            continue
        s = f"{fmt_us(e['total'])}<br>{e['gbps']:.0f} GB/s ({100*e['gbps']/PEAK:.0f}%)"
        row.append(f"🏆 **{s}**" if k == best else s)
    if has_ratio:
        fi, t5 = data.get((L, BASE)), data.get((L, "t5fp8"))
        row.append(f"{ratio_icon(fi['total']/t5['total'])} {fi['total']/t5['total']:.2f}×"
                   if fi and t5 else "—")
    if has_best:
        b, t5 = best_fi(L), data.get((L, "t5fp8"))
        row.append(f"{ratio_icon(b/t5['total'])} {b/t5['total']:.2f}×" if b and t5 else "—")
    md.append("| " + " | ".join(row) + " |")
georow = [f"**geomean vs {BASE_LABEL}**"]
for k in LEGS:
    g = geomean([data[(L, BASE)]["total"] / data[(L, k)]["total"]
                 for L in LS if (L, BASE) in data and (L, k) in data])
    georow.append("—" if g is None else f"**{g:.2f}×**")
if has_ratio:
    georow.append("")
if has_best:
    g = geomean([best_fi(L) / data[(L, "t5fp8")]["total"]
                 for L in LS if best_fi(L) and (L, "t5fp8") in data])
    georow.append("—" if g is None else f"**{g:.2f}×**")
md.append("| " + " | ".join(georow) + " |")
md.append(f"\n🚀 ≥{1 + NOISE:.2f}× faster than {BASE_LABEL} · 🟰 within ±{NOISE * 100:.0f}% (run-to-run "
          f"noise on this box) · 🐢 slower · 🏆 fastest at that L\n")

md.append("## 🔬 Per-kernel detail\n")
for L in LS:
    md.append(f"### L = {L:,} (KV {kv_bytes(L)/1e6:.1f} MB)\n")
    md.append("| Leg | Kernel | avg | min | L2 sectors | CTAs | calls |")
    md.append("|---|---|--:|--:|--:|--:|--:|")
    any_row = False
    for k in LEGS:
        e = data.get((L, k))
        if not e:
            continue
        for role, rows in (("decode", e["main"]), (LEG_DEFS[k]["merge_label"], e["merge"])):
            for r in rows:
                any_row = True
                name = r["name"] if len(r["name"]) <= 48 else r["name"][:45] + "…"
                md.append(f"| {LEG_DEFS[k]['label']} | {role}: `{name}` | {fmt_us(r['avg_us'])} | "
                          f"{fmt_us(r['min_us'])} | {r.get('sectors', float('nan')):,.0f} | "
                          f"{r.get('ctas', 0):.0f} | {r['inv']} |")
    if not any_row:
        md.append("| — | no data | | | | | |")
    md.append("")

if warnings:
    md.append("## ⚠️ Warnings\n")
    md.extend(f"- {w}" for w in warnings)
    md.append("")

md_path = os.path.join(OUT, "summary.md")
with open(md_path, "w") as f:
    f.write("\n".join(md) + "\n")

csv_path = os.path.join(OUT, "summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["L", "leg", "role", "kernel", "avg_us", "min_us", "l2_sectors",
                "ctas", "invocations", "leg_total_us", "leg_kv_gbps"])
    for L in LS:
        for k in LEGS:
            e = data.get((L, k))
            if not e:
                continue
            for role, rows in (("decode", e["main"]), ("merge", e["merge"])):
                for r in rows:
                    w.writerow([L, k, role, r["name"], f"{r['avg_us']:.3f}", f"{r['min_us']:.3f}",
                                f"{r.get('sectors', float('nan')):.0f}", f"{r.get('ctas', 0):.0f}",
                                r["inv"], f"{e['total']:.3f}", f"{e['gbps']:.2f}"])

print()
print(f"  📝 {bold('Markdown')}: {md_path}")
print(f"  📈 {bold('CSV')}:      {csv_path}")
print(f"  📁 {bold('Raw logs')}: {OUT}")
PY
