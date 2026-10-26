"""sm_87 gate-probe canary for the Orin family (self-contained; the image's mjolnir).

The Thor counterpart is kernel_test_sm110_gates.py (T6 GDN, T9 FA4, T10
draft-CG, T11 hd256); this is the Orin counterpart -- each decode kernel
against the reference on this device:

  O1 plugin   mjolnir.register() resolves (MJOLNIR on a patched vLLM, CUSTOM
              on stock); must not raise
  O2 mma      fp8-KV decode vs ref_decode -- the contiguous pool and the
              serving's K|V-interleaved views (vLLM 0.32 interleaves K|V in
              the last dim under every layout name), q_len 1 and 4 (the MTP
              verify shape)
  O3 mma fp6  the packed e2m3 pool + the code LUT + per-channel scales via
              the q_mul/o_mul fold (M1+M2) vs the dequantized reference,
              both layouts, q_len 1 and 4

Exit 0 iff every probe passes. GPU probes self-skip off sm_87 (and without
CUDA), so the same image can gate on any board.
"""

import torch

from mjolnir.arch.orin import OrinDecodeRunner
from mjolnir.arch.orin.harness import ref_decode
from mjolnir.kvfp6 import e2m3_values

H_Q, H_K, D = 24, 4, 256  # Qwen3.8-27B's attention: 24/4 heads, d 256
HR = H_Q // H_K
PS = 64  # the board's pool page size
SCALE = 1.0 / (D**0.5)


def _runner(b: int, M: int, **kw) -> OrinDecodeRunner:
    return OrinDecodeRunner(
        num_heads=H_Q,
        num_kv_heads=H_K,
        head_size=D,
        page_size=PS,
        out_dtype=torch.bfloat16,
        max_num_seqs=b,
        max_q_len=M,
        q_mode="bf16",
        device="cuda",
        **kw,
    )


def _interleaved(k_half: torch.Tensor, v_half: torch.Tensor):
    """Views into the serving's K|V-interleaved pool: the (np, ps, h_k, half)
    halves live in one (np, h_k, ps, 2*half) buffer; the kernel reads the
    strided (np, ps, h_k, half) slices back."""
    npages, ps, hk, kh = k_half.shape
    p2 = torch.zeros(npages, hk, ps, kh + v_half.shape[-1], dtype=k_half.dtype, device=k_half.device)
    p2[..., :kh] = k_half.permute(0, 2, 1, 3)
    p2[..., kh:] = v_half.permute(0, 2, 1, 3)
    return p2[..., :kh].transpose(1, 2), p2[..., kh:].transpose(1, 2)


def _verdict(tag: str, kind: str, mode: str, M: int, out: torch.Tensor, ref: torch.Tensor) -> str:
    o = out.view(1, M, H_Q, D).float().cpu()[0]
    rel = ((o - ref).norm() / ref.norm().clamp_min(1e-6)).item()
    ok = rel < (0.05 if kind == "fp6" else 0.03)
    print(f"{tag:5s} {kind:3s} {mode:8s} ql={M}: rel={rel:.4f} {'OK' if ok else 'FAIL'}", flush=True)
    return "OK" if ok else "FAIL"


def check_fp8(mode: str, M: int) -> str:
    import mjolnir.arch.orin.runner as OR

    OR._COMPILED.clear()
    from mjolnir.arch.orin.harness import make_case

    b, L = 1, 25
    q, kv, table, seq_lens, _qd, kd, vd, _raw = make_case(
        b, M, 1500, H_Q, H_K, D, PS, torch.float8_e4m3fn, False, seed=3, device="cuda"
    )
    seq_lens = torch.tensor([L], dtype=torch.int32, device="cuda")
    out = torch.empty(b * M, H_Q, D, dtype=torch.bfloat16, device="cuda")
    runner = _runner(b, M)
    plan = runner.plan(seq_lens.tolist(), b, M, table, seq_lens)
    if mode == "contig":
        runner.run(plan, q.view(b * M, H_Q, D), out, kv, kv, kd, vd, SCALE)
    else:
        k_view, v_view = _interleaved(kv, kv)
        runner.run(plan, q.view(b * M, H_Q, D), out, k_view, v_view, kd, vd, SCALE)
    torch.cuda.synchronize()
    ref = ref_decode(
        q.float().cpu(),
        kv.float().cpu(),
        kv.float().cpu(),
        table.cpu(),
        seq_lens.cpu(),
        None,
        kd.float().cpu(),
        vd.float().cpu(),
        SCALE,
    )[0]
    return _verdict("mma", "fp8", mode, M, out, ref)


def _pack_e2m3(mag: torch.Tensor) -> torch.Tensor:
    lut = e2m3_values("cpu")
    codes = (mag.unsqueeze(-1) - lut.view(*([1] * mag.dim()), 64)).abs().argmin(-1).to(torch.uint8)
    codes |= ((mag < 0).to(torch.uint8)) << 5  # sign bit (bit 5)
    d = mag.shape[-1]
    c = codes.reshape(*mag.shape[:-1], d // 4, 4).to(torch.int32)
    b0 = c[..., 0] | ((c[..., 1] & 3) << 6)
    b1 = (c[..., 1] >> 2) | ((c[..., 2] & 0xF) << 4)
    b2 = (c[..., 2] >> 4) | ((c[..., 3] & 0x3F) << 2)
    return torch.stack((b0, b1, b2), -1).reshape(*mag.shape[:-1], d * 3 // 4).to(torch.uint8)


def _dequant(half: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    """(np, ps, h_k, D*3//4) uint8 -> (np, ps, h_k, D) fp32: the codes x the
    per-channel scales."""
    idx = torch.arange(D)
    bb, oo = (idx * 6) // 8, (idx * 6) % 8
    hi = (bb + 1).clamp(max=half.shape[-1] - 1)
    codes = ((half[..., bb].int() >> oo) | (half[..., hi].int() << (8 - oo))) & 0x3F
    return e2m3_values("cpu")[codes] * ks.view(1, 1, half.shape[2], D)


def _build_per_channel(b: int, M: int, seqlen: int, seed: int, device: str):
    gen = torch.Generator("cpu").manual_seed(seed)
    max_pages = (seqlen + PS - 1) // PS + 1
    num_pages = b * max_pages + 8
    q = (torch.randn(b, M, H_Q, D, generator=gen) * 0.5).to(torch.bfloat16).to(device)
    raw = (torch.randn(num_pages, PS, H_K, D, generator=gen) * 0.7).clamp(-3, 3)
    ks = torch.rand(H_K, D, generator=gen) * 0.5 + 0.25
    vs = torch.rand(H_K, D, generator=gen) * 0.5 + 0.25
    kcodes = _pack_e2m3((raw / ks.view(1, 1, H_K, D)).clamp(-7.5, 7.5))
    vcodes = _pack_e2m3((raw / vs.view(1, 1, H_K, D)).clamp(-7.5, 7.5))
    pool = torch.cat((kcodes, vcodes), -1).to(device)  # (np, ps, h_k, 2*rb)
    table = torch.randperm(num_pages, generator=gen)[: b * max_pages].view(b, max_pages).to(torch.int32).to(device)
    seq_lens = torch.full((b,), seqlen, dtype=torch.int32, device=device)
    q_mul = ks.repeat_interleave(HR, dim=0).to(device)
    o_mul = vs.repeat_interleave(HR, dim=0).to(device)
    return q, pool, table, seq_lens, q_mul, o_mul, ks, vs


def check_fp6(mode: str, M: int) -> str:
    import mjolnir.arch.orin.runner as OR

    OR._COMPILED.clear()
    b, L = 1, 25
    q, pool, table, seq_lens, q_mul, o_mul, ks, vs = _build_per_channel(b, M, 1500, 3, "cuda")
    seq_lens = torch.tensor([L], dtype=torch.int32, device="cuda")
    out = torch.empty(b * M, H_Q, D, dtype=torch.bfloat16, device="cuda")
    runner = _runner(b, M, kv_format="e2m3")
    plan = runner.plan(seq_lens.tolist(), b, M, table, seq_lens)
    rb = D * 3 // 4
    ones = torch.ones(b, H_K, device="cuda")
    if mode == "contig":
        k_view, v_view = pool[..., :rb], pool[..., rb:]
    else:
        k_view, v_view = _interleaved(pool[..., :rb], pool[..., rb:])
    runner.run(plan, q.view(b * M, H_Q, D), out, k_view, v_view, ones, ones, SCALE, q_mul=q_mul, o_mul=o_mul)
    torch.cuda.synchronize()
    dek = _dequant(pool[..., :rb].cpu(), ks)
    dev = _dequant(pool[..., rb:].cpu(), vs)
    ref = ref_decode(
        q.float().cpu(),
        dek,
        dev,
        table.cpu(),
        seq_lens.cpu(),
        None,
        torch.ones(b, H_K),
        torch.ones(b, H_K),
        SCALE,
    )[0]
    return _verdict("mma", "fp6", mode, M, out, ref)


def probe_plugin() -> str:
    import mjolnir

    mjolnir.register()  # idempotent; must not raise on any vLLM
    native = mjolnir.has_native_member()
    print(f"O1 plugin: register() ok, native MJOLNIR member={native}", flush=True)
    return "OK"


def main() -> int:
    if not torch.cuda.is_available():
        print("[orin-gates] no CUDA -- GPU probes self-skip")
        return 0
    cc = torch.cuda.get_device_capability()
    if cc != (8, 7):
        print(f"[orin-gates] device is sm_{cc[0]}{cc[1]}, not sm_87 -- self-skip")
        return 0
    results = [probe_plugin()]
    results += [check_fp8(mode, m) for mode in ("contig", "strided") for m in (1, 4)]
    results += [check_fp6(mode, m) for mode in ("contig", "strided") for m in (1, 4)]
    ok = all(r == "OK" for r in results)
    print("ORIN GATES: ALL OK" if ok else "ORIN GATES: SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
