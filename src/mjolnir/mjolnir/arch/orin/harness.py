"""The Orin (sm_87) harness: correctness + benchmark driver for the GEMV
decode kernel — the family sibling of ``mjolnir.arch.thor.harness``.

Run standalone: ``python -m mjolnir.arch.orin.harness --help``. The kernel
is arch-portable FMA (cp.async + shfl + exp2), so this harness runs on any
sm_80+ device — on a Thor it is the development smoke; the Orin *numbers*
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

from .runner import GEMVDecodeRunner


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
) -> torch.Tensor:
    """fp32 reference: dequantize, causal tail, softmax, PV."""
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
            lim = L - M + 1 + m  # the causal tail: keys 0 .. L-M+m
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
):
    """A random case: bf16 Q, e4m3/bf16 KV pool + block table + descales."""
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
        codes = (mag.unsqueeze(-1) - lut.view(1, 1, 1, 1, 64)).abs().argmin(-1).to(torch.uint8)
        codes |= ((mag < 0).to(torch.uint8)) << 5  # sign bit (bit 5)
        c = codes.reshape(num_pages, page_size, h_k, d // 4, 4).to(torch.int32)
        b0 = c[..., 0] | ((c[..., 1] & 3) << 6)
        b1 = (c[..., 1] >> 2) | ((c[..., 2] & 0xF) << 4)
        b2 = (c[..., 2] >> 4) | ((c[..., 3] & 0x3F) << 2)
        kv = torch.stack((b0, b1, b2), -1).reshape(num_pages, page_size, h_k, d * 3 // 4).to(torch.uint8).to(device)
        kd = scale_h.view(1, h_k).to(device).expand(b, h_k).contiguous()
        vd = kd.clone()
        return q, kv, table, seq_lens, qd, kd, vd
    return q, kv, table, seq_lens, qd, kd, vd


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
    q, kv, table, seq_lens, _qd, kd, vd = make_case(
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
    )
    b = args.batches
    scale = 1.0 / math.sqrt(args.headdim)
    runner = GEMVDecodeRunner(
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
    runner.run(plan, q_tokens, out, kv, kv, kd, vd, scale)
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
        None,  # bf16 Q: no q_descale on either side
        ref_kd,
        ref_vd,
        scale,
    )
    rel = (got - ref).norm() / ref.norm().clamp_min(1e-6)
    ok = rel.item() < args.tolerance
    print(
        f"check b={b} M={args.q_len} L<={args.seqlen} kv={args.kv_dtype} "
        f"split={runner.split}: rel_err={rel.item():.5f} (tol {args.tolerance}) -> {'OK' if ok else 'FAIL'}"
    )
    return ok


def bench(args) -> None:
    dev = args.device
    q, kv, table, seq_lens, _qd, kd, vd = make_case(
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
    )
    b = args.batches
    scale = 1.0 / math.sqrt(args.headdim)
    runner = GEMVDecodeRunner(
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
    runner.run(plan, q_tokens, out, kv, kv, kd, vd, scale)  # compile
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


def main() -> int:
    p = argparse.ArgumentParser(description="Orin GEMV decode harness (correctness + bench).")
    p.add_argument("--batches", "--b", type=int, default=1)
    p.add_argument("--seqlen", "--s", type=int, default=4096)
    p.add_argument("--q_len", "--M", type=int, default=1)
    p.add_argument("--heads_q", type=int, default=24)
    p.add_argument("--heads_k", type=int, default=4)
    p.add_argument("--headdim", "--d", type=int, default=256)
    p.add_argument("--page_size", "--ps", type=int, default=16, choices=(16, 32, 64, 128))
    p.add_argument("--kv_dtype", "--kv_format", default="fp8", choices=("fp8", "bf16", "e2m3"))
    p.add_argument("--kv_layout", default="hnd", choices=("hnd", "nhd"))
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
    p.add_argument("--skip_ref_check", action="store_true", help="bench only (no correctness pass)")
    p.add_argument("--sweep", action="store_true", help="the default correctness matrix")
    # Shared verify-stage flags that have no GEMV meaning: accepted so the
    # platform-dispatched stage code can pass one arg surface to both families.
    p.add_argument("--no_l2_flush", action="store_true", help="accepted; the prebuilt launch timing never flushes")
    p.add_argument("--ctas_per_sm", type=int, default=1, help="the GEMV grid has no CTAs/SM knob; 1 only")
    p.add_argument("--max_splits", type=int, default=0, help="split-KV capacity (0 = the auto plan)")
    args = p.parse_args()
    assert args.ctas_per_sm == 1, "the Orin GEMV grid has no CTAs/SM knob (one CTA per (head, chunk))"
    if args.kv_layout == "nhd":
        # The GEMV consumes the shared machinery's head-major LOGICAL views
        # (whatever the vLLM pool layout, detect/split hands the runner HND);
        # the layout question is the backend's, so this flag is informational.
        print("note: the Orin GEMV consumes HND logical views regardless of the vLLM pool layout")

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
