"""``verify correctness`` — the numerical cases: the kernel vs the emulator
and the reference paths, per dtype/layout/feature. The cases pin the vLLM
production config (fp8 KV, NHD) unless they pass their own flags; the Orin
runs its own runner-test matrix. Split from stages.py along the concern
seam (the file-length ratchet) — the shared helpers stay in stages."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from mjolnir_cli.verify.legs import slug as _slug

if TYPE_CHECKING:
    from mjolnir_cli.verify.stages import Ctx

# ── correctness ─────────────────────────────────────────────────────────────

# (label, args, q dtypes). Cases pin the vLLM production config (fp8 KV, NHD)
# unless they pass their own --kv_format / --kv_layout (later flags win);
# "default:" cases run the bare harness default (e2m3 on HND for fp8 Q).
_PIN = "--kv_format fp8 --kv_layout nhd"
BOTH = ("fp8", "bf16")
FP8 = ("fp8",)
CASES: list[tuple[str, str, tuple[str, ...]]] = [
    ("L=8192", "--seqlen 8192", BOTH),
    ("L=128 (1 chunk, direct out)", "--seqlen 128", BOTH),
    ("L=1000 (partial tile mask)", "--seqlen 1000", BOTH),
    ("L=1 (minimal)", "--seqlen 1", BOTH),
    ("L=65536 (ring wraps)", "--seqlen 65536", BOTH),
    ("varlen 6x<=20000", "--batches 6 --seqlen 20000 --varlen", BOTH),
    ("varlen 32x<=2048 (>1 wave)", "--batches 32 --seqlen 2048 --varlen", BOTH),
    ("spec q_len=2 varlen 4x<=3000", "--batches 4 --seqlen 3000 --varlen --q_len 2", BOTH),
    ("spec q_len=4 2x777", "--batches 2 --seqlen 777 --q_len 4", BOTH),
    ("page 16 varlen", "--batches 4 --seqlen 1000 --varlen --page_size 16", BOTH),
    ("page 64 varlen", "--batches 4 --seqlen 1000 --varlen --page_size 64", BOTH),
    ("contiguous pages", "--seqlen 8192 --contiguous_pages", BOTH),
    ("2 CTAs/SM (40-CTA plan)", "--seqlen 8192 --ctas_per_sm 2", BOTH),
    ("forced 7 chunks", "--seqlen 8192 --kv_splits 7", BOTH),
    ("padded work list (40)", "--batches 3 --seqlen 5000 --varlen --pad_work_to 40", BOTH),
    ("max_splits=1 (no reduction)", "--seqlen 8192 --max_splits 1", BOTH),
    ("headdim 128", "--seqlen 8192 --headdim 128", BOTH),
    ("KV interleaved (B,H,N,2D)", "--batches 4 --seqlen 3000 --varlen --page_size 16 --kv_layout interleaved", BOTH),
    ("KV flashinfer (B,2,N,H,D)", "--batches 4 --seqlen 3000 --varlen --page_size 32 --kv_layout flashinfer", BOTH),
    ("strided Q/O (fused QKV view)", "--batches 4 --seqlen 3000 --varlen --strided_qo", BOTH),
    ("per-tensor descales", "--batches 4 --seqlen 3000 --varlen --descale per_tensor", BOTH),
    ("per-head descales", "--batches 4 --seqlen 3000 --varlen --descale per_head", BOTH),
    (
        "vLLM-like: interl+strided+tensor p16",
        "--batches 8 --seqlen 6000 --varlen --page_size 16 --kv_layout interleaved --strided_qo --descale per_tensor",
        BOTH,
    ),
    ("PDL (decode + reduction)", "--batches 4 --seqlen 20000 --varlen --pdl", BOTH),
    ("L2 prefetch 2, page 128", "--batches 2 --seqlen 30000 --varlen --l2_prefetch 2", BOTH),
    ("L2 prefetch 4, page 16", "--batches 4 --seqlen 9000 --varlen --page_size 16 --l2_prefetch 4", BOTH),
    ("PDL + L2 prefetch 1, q_len=2", "--batches 3 --seqlen 5000 --varlen --q_len 2 --pdl --l2_prefetch 1", BOTH),
    ("HND layout p128", "--seqlen 8192 --kv_layout hnd", BOTH),
    ("HND layout p16 varlen", "--batches 4 --seqlen 3000 --varlen --page_size 16 --kv_layout hnd", BOTH),
    ("fp8 from float (accuracy ref)", "--batches 2 --seqlen 5000 --varlen --kv_from_float", FP8),
    ("e3m2 KV p128", "--seqlen 8192 --kv_format e3m2", FP8),
    ("e2m3 KV p16 varlen", "--batches 4 --seqlen 3000 --varlen --page_size 16 --kv_format e2m3", FP8),
    (
        "e2m3 KV vLLM layout (interl p128 q_len=4)",
        "--batches 4 --seqlen 5000 --varlen --page_size 128 --q_len 4 --kv_format e2m3 --kv_layout interleaved "
        "--strided_qo",
        FP8,
    ),
    ("e2m1 KV p128", "--seqlen 8192 --kv_format e2m1", FP8),
    (
        "e2m1 KV HND p16 q_len=2",
        "--batches 3 --seqlen 2500 --varlen --page_size 16 --q_len 2 --kv_layout hnd --kv_format e2m1",
        FP8,
    ),
    ("nvfp4 KV p128", "--seqlen 8192 --kv_format nvfp4", FP8),
    ("nvfp4 KV p16 varlen", "--batches 4 --seqlen 3000 --varlen --page_size 16 --kv_format nvfp4", FP8),
    (
        "nvfp4 KV HND q_len=2 pad",
        "--batches 3 --seqlen 2500 --varlen --q_len 2 --kv_layout hnd --kv_format nvfp4 --pad_work_to 40",
        FP8,
    ),
    ("default: e2m3 HND L=8192", "@default --seqlen 8192", BOTH),
    ("default: L=65536 (ring wraps)", "@default --seqlen 65536", BOTH),
    ("default: varlen 6x<=20000", "@default --batches 6 --seqlen 20000 --varlen", BOTH),
    (
        "default: q_len=3 p16 pad",
        "@default --batches 4 --seqlen 2000 --varlen --q_len 3 --page_size 16 --pad_work_to 40",
        BOTH,
    ),
    ("default: headdim 128 + PDL", "@default --batches 2 --seqlen 9000 --varlen --headdim 128 --pdl", BOTH),
    # parallel drafters (DSpark / DFlash): a non-causal block of 1 + 7 queries,
    # the DSpark drafter's heads (40 query / 8 KV, d = 128)
    (
        "non-causal q_len=8 (drafter block)",
        "@default --batches 4 --seqlen 5000 --varlen --q_len 8 --heads_q 40 --heads_k 8 --headdim 128 --non_causal",
        BOTH,
    ),
    (
        "non-causal q_len=8 fp8 p16 pad",
        "--batches 3 --seqlen 2500 --varlen --q_len 8 --heads_q 40 --heads_k 8 --headdim 128 --page_size 16 "
        "--non_causal --pad_work_to 40",
        BOTH,
    ),
    ("ring depth 2, fp8 (ring wraps)", "--seqlen 65536 --kv_layout hnd --kv_stages 2", BOTH),
    ("ring depth 2, e2m3 varlen p16", "@default --batches 4 --seqlen 9000 --varlen --page_size 16 --kv_stages 2", FP8),
    ("kv evict_first", "--batches 4 --seqlen 20000 --varlen --kv_evict_first", BOTH),
    ("evict + PDL L=8192", "--seqlen 8192 --kv_evict_first --pdl", BOTH),
    (
        "evict + PDL q_len=3 + pad",
        "--batches 4 --seqlen 2000 --varlen --q_len 3 --kv_evict_first --pdl --pad_work_to 40",
        BOTH,
    ),
    (
        "evict + PDL, interl p16",
        "--batches 4 --seqlen 9000 --varlen --page_size 16 --kv_layout interleaved --kv_evict_first --pdl",
        BOTH,
    ),
    (
        "vLLM-like spec q_len=3 p16",
        "--batches 4 --seqlen 2000 --varlen --q_len 3 --page_size 16 --kv_layout interleaved --strided_qo "
        "--descale per_tensor --pad_work_to 40",
        BOTH,
    ),
    # E1 fused last-CTA merge (counter self-reset covered by the replay case)
    ("fused merge e2m3 HND varlen", "@default --batches 2 --seqlen 16384 --varlen --fused_merge", FP8),
    ("fused merge L=65536, replay x8", "@default --seqlen 65536 --fused_merge --iterations 8", FP8),
    ("fused merge q_len=2 fp8 HND", "--batches 3 --seqlen 9000 --varlen --q_len 2 --kv_layout hnd --fused_merge", FP8),
]


# One representative case per group: the fast pre-check autoresearch runs on
# every experiment (~10 containers) before the full matrix is worth paying for.
QUICK_CASES = (
    "L=8192",
    "L=1000 (partial tile mask)",
    "L=65536 (ring wraps)",
    "varlen 6x<=20000",
    "spec q_len=2 varlen 4x<=3000",
    "page 16 varlen",
    "KV interleaved (B,H,N,2D)",
    "HND layout p16 varlen",
    "e2m3 KV p16 varlen",
    "default: e2m3 HND L=8192",
    "default: varlen 6x<=20000",
    "evict + PDL q_len=3 + pad",
    "non-causal q_len=8 (drafter block)",
    "fused merge e2m3 HND varlen",
)


def case_args(args: str) -> str:
    if args.startswith("@default"):
        return args[len("@default") :].strip()
    return f"{_PIN} {args}"


LABEL_W = max(len(label) for label, _, _ in CASES)


def _short_check(check: str | None) -> str:
    """'|kern-emu| max=0.03051 mean=0.0007534' -> 'max 3.05e-02  mean 7.53e-04'
    (fixed width, so the live correctness table never wraps)."""
    if not check:
        return ""
    m = re.search(r"max=([0-9.eE+-]+)\s+mean=([0-9.eE+-]+)", check)
    if not m:
        return check[:26]
    try:
        return f"max {float(m.group(1)):.2e}  mean {float(m.group(2)):.2e}"
    except ValueError:
        return check[:26]


_ORIN_CHECK = re.compile(
    r"check b=(\d+) M=(\d+) L<=(\d+) kv=(\w+) split=(\w+): rel_err=([0-9.na]+) \(tol [0-9.]+\) -> (OK|FAIL)"
)


def _orin_correctness(ctx: Ctx, timeout: float) -> dict:
    """The Orin correctness matrix: one container per KV format runs the
    harness sweep (page size x M x L against the fp32 reference) — fp8 and
    fp6 e2m3, both against the dequantized-pool reference. The mma kernel
    takes a bf16 query only (the fp8-Q pack retired with the GEMV; the M3
    work), so the loop axis is the KV format, not Q. The check lines parse
    into the shared case schema."""
    cases = []
    for kv in ("fp8", "e2m3"):
        log = ctx.logdir / "correctness" / f"orin-sweep-kv{kv}.log"
        t0 = time.time()
        rc, text = ctx.runner.harness(log, "bf16", f"--kv_dtype {kv} --sweep", timeout=timeout)
        for m in _ORIN_CHECK.finditer(text):
            b, M, L, kv_seen, split, rel, verdict = m.groups()
            cases.append(
                {
                    "label": f"b={b} M={M} L<={int(L):,} kv={kv_seen} split={split}",
                    "q": "bf16",
                    "args": ["--batches", b, "--q_len", M, "--seqlen", L, "--kv_dtype", kv_seen],
                    "passed": verdict == "OK",
                    "rc": rc,
                    "seconds": round(time.time() - t0, 1),
                    "check": f"rel_err={rel} (tol 0.05)",
                    "log": str(log),
                }
            )
    for c in cases:
        mark = "✅" if c["passed"] else "❌"
        ctx.out(f" {mark} {c['label']:<44} {c['check']}")
    if not cases and not ctx.runner.dry_run:
        ctx.out(f" ❌ no check lines parsed — {ctx.logdir / 'correctness'}")
    return {"cases": cases, "params": {"matrix": "ps x M x L x kv (orin harness --sweep)"}}


def stage_correctness(
    ctx: Ctx,
    dtypes: str = "fp8 bf16",
    match: str = "",
    fail_fast: bool = False,
    timeout: float = 180.0,
    quick: bool = False,
) -> dict:
    """Kernel vs the bit-faithful emulator: output + LSE + poisoned-page NaN
    check, for every case x Q dtype."""
    if ctx.runner.platform == "orin":
        # One container runs the whole ps x M x L matrix: well past the
        # 180 s Thor-per-case default.
        return _orin_correctness(ctx, max(timeout, 900.0))
    want = set(dtypes.split())
    cases = []
    for label, args, dts in CASES:
        if match and match.lower() not in label.lower():
            continue
        if quick and label not in QUICK_CASES:
            continue
        for dt in dts:
            if dt not in want:
                continue
            log = ctx.logdir / "correctness" / f"{len(cases) + 1:02d}-{dt}-{_slug(label)}.log"
            t0 = time.time()
            rc, text = ctx.runner.harness(log, dt, case_args(args), timeout=timeout)
            ok = rc == 0 and bool(re.search(r"^PASS$", text, re.MULTILINE))
            check = re.search(r"^check: (.*?)(;|$)", text, re.MULTILINE)
            plan = re.search(r"plan: SplitPlan\((.*)\)", text)
            c = {
                "label": label,
                "q": dt,
                "args": case_args(args),
                "passed": ok or ctx.runner.dry_run,
                "rc": rc,
                "seconds": round(time.time() - t0, 1),
                "check": check.group(1) if check else None,
                "plan": plan.group(1) if plan else None,
                "log": str(log),
            }
            cases.append(c)
            mark = "✅" if c["passed"] else "❌"
            ctx.out(f" {mark} {dt:<5} {label:<{LABEL_W}} {_short_check(c['check']):<26} {c['seconds']:>4.0f}s")
            if not c["passed"]:
                for line in [
                    ln for ln in text.splitlines() if re.search(r"Error|Traceback|Assertion|NaN|Mismatch", ln)
                ][:3]:
                    ctx.out(f"      │ {line}")
                if fail_fast:
                    return {"cases": cases, "params": {"dtypes": dtypes, "match": match, "quick": quick}}
    return {"cases": cases, "params": {"dtypes": dtypes, "match": match, "quick": quick}}
