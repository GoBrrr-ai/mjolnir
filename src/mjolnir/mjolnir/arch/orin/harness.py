"""The Orin (sm_87) harness: correctness + benchmark driver for the mma
decode kernel — the family sibling of ``mjolnir.arch.thor.harness``.

Run standalone: ``python -m mjolnir.arch.orin.harness --help``. The kernel
targets sm_87 (mma.sync m16n8k16 + cp.async), so this harness runs on any
sm_87+ device — on a Thor it is the development smoke; the Orin *numbers*
come from running it (and ``mjolnir verify``) on an Orin.

Correctness model: the kernel computes scores and P in fp32 (no P
quantization — unlike the t5 family's e4m3 P), with the descales applied
as qk_descale = q_descale·k_descale on the score scale and v_descale on
the output, the causal tail of the t5 contract (seq_lens include the M
query tokens; row m sees KV[0 .. kv_len − M + m)), and the
chunk-normalized split-KV combine. The reference is plain fp32 torch
attention over the dequantized cache; the check is relative error on O
(the t5 harness's 0.05 tolerance; the split path must match the direct
path because the combine is exact math, not an approximation).
"""

from __future__ import annotations

import argparse
import math

import torch

from .runner import OrinDecodeRunner


def ref_decode(
    q: torch.Tensor,  # (b, M, h_q, d) bf16
    k_pool: torch.Tensor,  # (pages, ps, h_k, d) e4m3 or bf16
    v_pool: torch.Tensor,
    block_table: torch.Tensor,  # (b, max_blocks) int32
    seq_lens: torch.Tensor,  # (b,) int (INCLUDING the M query tokens)
    q_descale: torch.Tensor | None,  # (b, h_k)
    k_descale: torch.Tensor,  # (b, h_k)
    v_descale: torch.Tensor,  # (b, h_k)
    softmax_scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """fp32 reference: dequantize, causal tail (or the whole context when
    ``causal`` is False — a parallel drafter's block), softmax, PV."""
    b, M, h_q, d = q.shape
    h_k = k_pool.shape[2]
    h_r = h_q // h_k
    ps = k_pool.shape[1]
    k_f = k_pool.float()  # torch converts e4m3 codes -> values natively
    v_f = v_pool.float()
    out = torch.zeros(b, M, h_q, d)
    qf = q.float()

    def gather(pool_f: torch.Tensor, bi: int, L: int) -> torch.Tensor:
        parts, remaining = [], L
        for page in block_table[bi].tolist():
            if remaining <= 0:
                break
            take = min(ps, remaining)
            parts.append(pool_f[int(page), :take])
            remaining -= take
        return torch.cat(parts, dim=0)  # (L, h_k, d)

    for bi in range(b):
        L = int(seq_lens[bi])
        k_seq = gather(k_f, bi, L)
        v_seq = gather(v_f, bi, L)
        kd = k_descale[bi].float().view(h_k, 1, 1)
        vd = v_descale[bi].float().view(h_k, 1, 1)
        qd = q_descale[bi].float().view(h_k, 1, 1) if q_descale is not None else 1.0
        for m in range(M):
            lim = L - M + 1 + m if causal else L  # the causal tail: keys 0 .. L-M+m
            q_m = qf[bi, m].view(h_k, h_r, d)  # (h_k, h_r, d)
            s = torch.einsum("hrd,hld->hrl", q_m, k_seq.transpose(0, 1)) * (softmax_scale * qd * kd)
            s[:, :, lim:] = -torch.inf
            p = s.softmax(dim=-1)  # (h_k, h_r, L)
            o = torch.einsum("hrl,hld->hrd", p, v_seq.transpose(0, 1)) * vd
            out[bi, m] = o.reshape(h_q, d)
    return out


def make_case(
    b: int,
    M: int,
    seqlen: int,
    h_q: int,
    h_k: int,
    d: int,
    page_size: int,
    kv_dtype: torch.dtype,
    fp8_q: bool,
    seed: int = 7,
    device: str = "cuda",
    layout: str = "nhd",
):
    """A random case: bf16 Q, e4m3/bf16 KV pool + block table + descales.

    Returns the unquantized float pool last — ``check`` uses it for the
    ``--kv_from_float`` quantization-error reference (the pool is always
    built from this float source; for bf16 the two are the same)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    max_pages = (seqlen + page_size - 1) // page_size + 1
    num_pages = b * max_pages + 8
    q = (torch.randn(b, M, h_q, d, generator=gen) * 0.5).to(torch.float8_e4m3fn if fp8_q else torch.bfloat16).to(device)
    raw = (torch.randn(num_pages, page_size, h_k, d, generator=gen) * 0.7).clamp(-3, 3)
    if kv_dtype == torch.float8_e4m3fn:
        kv = raw.to(torch.float8_e4m3fn).to(device)
    else:
        kv = raw.to(torch.bfloat16).to(device)
    table = torch.randperm(num_pages, generator=gen)[: b * max_pages].view(b, max_pages).to(torch.int32).to(device)
    seq_lens = torch.randint(M, seqlen + 1, (b,), generator=gen)
    seq_lens[0] = seqlen
    seq_lens = seq_lens.to(torch.int32).to(device)
    kd = (1.0 + torch.rand(b, h_k, generator=gen)).to(device)
    vd = (1.0 + torch.rand(b, h_k, generator=gen)).to(device)
    qd = (1.0 + torch.rand(b, h_k, generator=gen)).to(device) if fp8_q else None
    if kv_dtype == "e2m3":
        # Per-head scale = amax/7.5 (the kvfp6 calibration rule); the codes
        # pack 4-per-3-bytes LE. The ref dequantizes code_value * scale.
        from mjolnir.kvfp6 import e2m3_values

        lut = e2m3_values("cpu")  # (64,) fp32
        scale_h = (raw.amax(dim=(0, 1, 3)) / 7.5).clamp_min(1e-6)  # (h_k,)
        mag = (raw / scale_h.view(1, 1, h_k, 1)).clamp(-7.5, 7.5)
        # The LUT argmin is chunked over pages: the full broadcast builds a
        # (pages, 16, h_k, d, 64) fp32 intermediate (68.6 GB at L=262144 —
        # the OOM killer). Chunking is bit-identical (argmin is per element)
        # at a ~2 GB peak.
        codes = torch.empty(num_pages, page_size, h_k, d, dtype=torch.uint8)
        for p0 in range(0, num_pages, 512):
            sl = slice(p0, min(p0 + 512, num_pages))
            codes[sl] = (mag[sl].unsqueeze(-1) - lut.view(1, 1, 1, 1, 64)).abs().argmin(-1).to(torch.uint8)
        codes |= ((mag < 0).to(torch.uint8)) << 5  # sign bit (bit 5)
        c = codes.reshape(num_pages, page_size, h_k, d // 4, 4).to(torch.int32)
        b0 = c[..., 0] | ((c[..., 1] & 3) << 6)
        b1 = (c[..., 1] >> 2) | ((c[..., 2] & 0xF) << 4)
        b2 = (c[..., 2] >> 4) | ((c[..., 3] & 0x3F) << 2)
        kv = torch.stack((b0, b1, b2), -1).reshape(num_pages, page_size, h_k, d * 3 // 4).to(torch.uint8).to(device)
        kd = scale_h.view(1, h_k).to(device).expand(b, h_k).contiguous()
        vd = kd.clone()
        if layout == "hnd":
            # head-major STORAGE presented in the runner's logical shape
            # (the strides carry the physical order — split_kv_cache's contract)
            kv = kv.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
            raw = raw.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
        return q, kv, table, seq_lens, qd, kd, vd, raw
    if layout == "hnd":
        kv = kv.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
        raw = raw.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
    return q, kv, table, seq_lens, qd, kd, vd, raw


def _e2m3_dequant(kv: torch.Tensor, d: int, scale_h: torch.Tensor) -> torch.Tensor:
    """The packed pool (pages, ps, h_k, d*3/4) uint8 -> the (pages, ps, h_k, d)
    fp32 values (the code LUT x the per-head scale)."""
    from mjolnir.kvfp6 import e2m3_values

    idx = torch.arange(d)
    bb, oo = (idx * 6) // 8, (idx * 6) % 8
    hi = (bb + 1).clamp(max=kv.shape[-1] - 1)  # the last code ends in byte bb
    codes = ((kv[..., bb].int() >> oo) | (kv[..., hi].int() << (8 - oo))) & 0x3F
    return e2m3_values("cpu")[codes] * scale_h.view(1, 1, -1, 1)


def check(args) -> bool:
    dev = args.device
    q, kv, table, seq_lens, qd, kd, vd, raw = make_case(
        args.batches,
        args.q_len,
        args.seqlen,
        args.heads_q,
        args.heads_k,
        args.headdim,
        args.page_size,
        args.kv_dtype
        if args.kv_dtype == "e2m3"
        else (torch.float8_e4m3fn if args.kv_dtype == "fp8" else torch.bfloat16),
        args.q_dtype == "fp8",
        seed=args.seed,
        device=dev,
        layout=args.kv_layout,
    )
    b = args.batches
    scale = 1.0 / math.sqrt(args.headdim)
    runner = OrinDecodeRunner(
        num_heads=args.heads_q,
        num_kv_heads=args.heads_k,
        head_size=args.headdim,
        page_size=args.page_size,
        out_dtype=torch.bfloat16,
        max_num_seqs=b,
        max_q_len=args.q_len,
        q_mode=args.q_dtype,
        kv_format=args.kv_dtype,
        max_splits=args.max_splits,
        device=dev,
    )
    plan = runner.plan(seq_lens.tolist(), b, args.q_len, table, seq_lens, causal=not args.non_causal)
    q_tokens = q.view(b * args.q_len, args.heads_q, args.headdim)
    out = torch.empty_like(q_tokens, dtype=torch.bfloat16)
    runner.run(plan, q_tokens, out, kv, kv, kd, vd, scale, q_scale=qd)
    got = out.view(b, args.q_len, args.heads_q, args.headdim).float().cpu()
    # e2m3: the dequantized pool already carries the per-head scale, so the
    # ref sees identity descales (the kernel folds the scale via kd/vd).
    if args.kv_dtype == "e2m3":
        pool_ref = _e2m3_dequant(kv.cpu(), args.headdim, kd.cpu()[0])
        ref_kd = torch.ones_like(kd).cpu()
        ref_vd = torch.ones_like(vd).cpu()
    else:
        pool_ref = kv.cpu()
        ref_kd = kd.cpu()
        ref_vd = vd.cpu()
    ref = ref_decode(
        q.float().cpu(),
        pool_ref,
        pool_ref,
        table.cpu(),
        seq_lens.cpu(),
        qd.cpu() if qd is not None else None,  # fp8 Q: the kernel's scale_eff folds it
        ref_kd,
        ref_vd,
        scale,
        causal=not args.non_causal,
    )
    rel = (got - ref).norm() / ref.norm().clamp_min(1e-6)
    ok = rel.item() < args.tolerance
    print(
        f"check b={b} M={args.q_len} L<={args.seqlen} kv={args.kv_dtype} "
        f"split={runner.split}: rel_err={rel.item():.5f} (tol {args.tolerance}) -> {'OK' if ok else 'FAIL'}"
    )
    # What the cache format costs (the verify quant stage parses the rel_l2=):
    # the kernel vs attention over the UNQUANTIZED float pool. fp8 reports it
    # under --kv_from_float; e2m3 always (its pool is built from float);
    # bf16 is float — no error to report.
    if args.kv_dtype == "e2m3" or (args.kv_dtype == "fp8" and args.kv_from_float):
        if args.kv_dtype == "e2m3":
            float_pool, f_kd, f_vd = raw.cpu(), torch.ones_like(kd).cpu(), torch.ones_like(vd).cpu()
        else:  # fp8: the random descales stay — the kernel sees dequant(kv) * kd
            float_pool, f_kd, f_vd = raw.cpu(), kd.cpu(), vd.cpu()
        ref_true = ref_decode(
            q.float().cpu(),
            float_pool,
            float_pool,
            table.cpu(),
            seq_lens.cpu(),
            None,
            f_kd,
            f_vd,
            scale,
            causal=not args.non_causal,
        )
        d_true = (got - ref_true).abs()
        rel_q = (d_true.norm() / ref_true.norm().clamp_min(1e-6)).item()
        print(
            f"kv quantization error ({args.kv_dtype}): |kern-fp32 unquantized| "
            f"max={d_true.max().item():.4g} mean={d_true.mean().item():.4g} "
            f"rel_l2={rel_q:.4g}"
        )
    return ok


def bench(args) -> None:
    dev = args.device
    q, kv, table, seq_lens, qd, kd, vd, _raw = make_case(
        args.batches,
        args.q_len,
        args.seqlen,
        args.heads_q,
        args.heads_k,
        args.headdim,
        args.page_size,
        args.kv_dtype
        if args.kv_dtype == "e2m3"
        else (torch.float8_e4m3fn if args.kv_dtype == "fp8" else torch.bfloat16),
        args.q_dtype == "fp8",
        seed=args.seed,
        device=dev,
        layout=args.kv_layout,
    )
    b = args.batches
    scale = 1.0 / math.sqrt(args.headdim)
    runner = OrinDecodeRunner(
        num_heads=args.heads_q,
        num_kv_heads=args.heads_k,
        head_size=args.headdim,
        page_size=args.page_size,
        out_dtype=torch.bfloat16,
        max_num_seqs=b,
        max_q_len=args.q_len,
        q_mode=args.q_dtype,
        kv_format=args.kv_dtype,
        max_splits=args.max_splits,
        device=dev,
    )
    plan = runner.plan(seq_lens.tolist(), b, args.q_len, table, seq_lens)
    q_tokens = q.view(b * args.q_len, args.heads_q, args.headdim)
    out = torch.empty_like(q_tokens, dtype=torch.bfloat16)
    runner.run(plan, q_tokens, out, kv, kv, kd, vd, scale, q_scale=qd)  # compile
    torch.cuda.synchronize()
    # Time the compiled launch with PREBUILT args: the serving path's python
    # (arg packing, the combine) is real cost but dominates the measurement
    # at small L; the kernel number needs back-to-back replays.
    from mjolnir.arch.orin import runner as _R

    key = runner._key(args.q_len, not args.non_causal, kv.shape[0], b, table.shape[1])
    fn = _R._COMPILED[key]
    ns = runner.ns_cap if runner.split else 1
    launch = (*runner._torch_args(key, b, q_tokens, out, kv, kv, table), scale * math.log2(math.e), ns)
    times = []
    for i in range(args.warmup + args.iters):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(args.inner):
            fn(*launch)
        e.record()
        torch.cuda.synchronize()
        if i >= args.warmup:
            times.append(s.elapsed_time(e) / args.inner * 1000)  # us
    times.sort()
    med = times[len(times) // 2]
    bytes_per_elem = 0.75 if args.kv_dtype == "e2m3" else (1 if args.kv_dtype == "fp8" else 2)
    kv_gb = seq_lens.sum().item() * 2 * args.heads_k * args.headdim * bytes_per_elem / 1e9
    gbps = kv_gb / (med / 1e6)
    print(
        f"bench b={b} M={args.q_len} L={args.seqlen} kv={args.kv_dtype} "
        f"split={runner.split} ns={runner.ns_cap}: median {med:.2f} us "
        f"({gbps:.0f} GB/s KV)"
    )
    # The shared verify stage code parses this canonical line (stages._MEDIAN).
    print(f"median {med:.2f} us, {gbps:.3g} GB/s (prebuilt launches, b={b}, M={args.q_len})")

    # The full serving path (arg packing + the LSE combine): back-to-back
    # runner.run() calls, one event pair each -- the honest per-call cost
    # vLLM would pay, python included. Parsed by stages._FULL_MEDIAN.
    for _ in range(args.warmup):
        runner.run(plan, q_tokens, out, kv, kv, kd, vd, scale, q_scale=qd)
    torch.cuda.synchronize()
    full_times = []
    for _ in range(args.iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        runner.run(plan, q_tokens, out, kv, kv, kd, vd, scale, q_scale=qd)
        e.record()
        torch.cuda.synchronize()
        full_times.append(s.elapsed_time(e) * 1000)
    full_times.sort()
    full_med = full_times[len(full_times) // 2]
    print(
        f"full median {full_med:.2f} us, {kv_gb / (full_med / 1e6):.3g} GB/s (full serving path, b={b}, M={args.q_len})"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Orin mma decode harness (correctness + bench).")
    p.add_argument("--batches", "--b", type=int, default=1)
    p.add_argument("--seqlen", "--s", type=int, default=4096)
    p.add_argument("--q_len", "--M", type=int, default=1)
    p.add_argument("--heads_q", type=int, default=24)
    p.add_argument("--heads_k", type=int, default=4)
    p.add_argument("--headdim", "--d", type=int, default=256)
    p.add_argument("--page_size", "--ps", type=int, default=16, choices=(16, 32, 64, 128))
    p.add_argument("--kv_dtype", "--kv_format", default="fp8", choices=("fp8", "e2m3"))
    p.add_argument(
        "--kv_layout",
        default="nhd",
        choices=("hnd", "nhd"),
        help="the PHYSICAL pool arrangement the case builds: nhd = "
        "token-major (np, ps, h_k, row) — what vLLM allocates; hnd = "
        "head-major (np, h_k, ps, row) — contiguous per-head rows",
    )
    p.add_argument("--q_dtype", default="bf16", choices=("bf16", "fp8"))
    p.add_argument("--non_causal", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tolerance", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bench", action="store_true", help="time instead of (or after) --check")
    p.add_argument("--warmup", "--warmup_iterations", type=int, default=5, help="untimed launches before the window")
    p.add_argument("--iters", "--iterations", type=int, default=30)
    p.add_argument("--inner", type=int, default=1, help="runs per timing window")
    p.add_argument("--check", action="store_true")
    p.add_argument(
        "--kv_from_float",
        action="store_true",
        help="also report the KV quantization error: the kernel vs attention "
        "over the unquantized float pool (the verify quant stage parses it; "
        "e2m3 always reports it — its pool is built from float)",
    )
    p.add_argument("--skip_ref_check", action="store_true", help="bench only (no correctness pass)")
    p.add_argument("--sweep", action="store_true", help="the default correctness matrix")
    # Shared verify-stage flags that have no mma meaning: accepted so the
    # platform-dispatched stage code can pass one arg surface to both families.
    p.add_argument("--no_l2_flush", action="store_true", help="accepted; the prebuilt launch timing never flushes")
    p.add_argument("--ctas_per_sm", type=int, default=2, help="the Orin grid has no CTAs/SM knob; 2 only")
    p.add_argument("--max_splits", type=int, default=0, help="split-KV capacity (0 = the auto plan)")
    args = p.parse_args()
    assert args.q_dtype == "bf16", "the mma kernel takes a bf16 query (the fp8-Q pack path is not wired; M3)"
    assert args.ctas_per_sm == 2, "the Orin mma grid is pinned at 2 CTAs/SM (255 regs + the smem bind)"
    ok = True
    if args.sweep:
        for ps in (16, 32, 64, 128):
            for M in (1, 4):
                for L in (1024, 8192, 65536):
                    args.page_size, args.q_len, args.seqlen = ps, M, L
                    ok &= check(args)
    elif args.check or (not args.bench and not args.skip_ref_check):
        ok = check(args)
    if args.bench or args.skip_ref_check:
        bench(args)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
