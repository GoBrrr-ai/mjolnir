"""Synthetic `mjolnir verify` documents for chart / README tests.

TEST DATA ONLY — never written into benchmarks/. The magnitudes follow the
shape of real Thor runs (so charts exercise realistic ranges) but every
document is marked ``host: "fixture"``.
"""

from __future__ import annotations

from mjolnir_cli.verify import results, stages

LS = list(stages.ALL_LS)
# µs at each L (ncu, cold L2) — shapes only
FI = [25.4, 64.2, 103.0, 191.5, 371.9, 763.4, 1370.0, 2560.0]
FA4 = [31.0, 118.0, 205.0, 380.0, 745.0, 1490.0, 2950.0, 5880.0]  # the default FA-family baseline (hd256 fp8)
MJ_FP8 = [13.6, 55.1, 87.7, 169.6, 354.7, 676.3, 1270.0, 2400.0]
MJ_E2M3 = [13.3, 49.8, 82.1, 172.7, 330.0, 617.3, 1190.0, 2180.0]
FI_HND = [25.3, 61.0, 96.0, 175.4, 344.8, 716.9, 1310.0, 2530.0]


def _pts(times, bits=8.0, hk=4, d=256):
    out = []
    for L, t in zip(LS, times, strict=True):
        kv = L * hk * d * 2 * bits / 8
        out.append(
            {
                "L": L,
                "total_us": t,
                "main_us": t * 0.97,
                "merge_us": t * 0.03,
                "gbps": kv / (t * 1e3),
                "ctas": 40,
                "invocations": 50,
                "l2_sectors": kv / 32,
            }
        )
    return out


def _series(sid, label, kind, times, bits=8.0, role="variant", flags=""):
    return {
        "id": sid,
        "label": label,
        "kind": kind,
        "role": role,
        "flags": flags,
        "kv_bits": bits,
        "points": _pts(times, bits),
    }


def _doc(stage, **body):
    d = results.header(
        stage,
        "20261010-120000",
        image="fixture",
        default=True,
        params={"ls": LS},
        kernel={"version": "0.1.0", "git": "abc1234"},
        gate={"server": "offline"},
    )
    d["host"] = "fixture"
    d.update(body)
    d["summary"] = stages.summarize(stage, d)
    return d


def _scale(times, f):
    return [t * f for t in times]


def all_docs() -> dict[str, dict]:
    fi = _series("fi", "FlashInfer (fp8 KV, NHD)", "fi", FI, role="baseline")
    fihnd = _series("fi-hnd", "FlashInfer (fp8 KV, HND)", "fi", FI_HND, role="baseline")
    return {
        "sweep": _doc(
            "sweep",
            series=[
                fi,
                _series("fa4", "FlashAttention 4 hd256 (fp8 Q+KV)", "fa4", FA4, 8.0, "baseline"),
                _series("mjolnir-fp8", "Mjolnir (fp8 KV, HND)", "t5", MJ_FP8, 8.0, "ours"),
                _series("mjolnir", "Mjolnir (e2m3 KV, HND)", "t5", MJ_E2M3, 6.0, "ours"),
            ],
        ),
        "hndsweep": _doc(
            "hndsweep", series=[fi, fihnd, _series("mjolnir-fp8", "Mjolnir (fp8 KV, HND)", "t5", MJ_FP8, 8.0, "ours")]
        ),
        "kvfmt": _doc(
            "kvfmt",
            series=[
                fi,
                fihnd,
                _series("t5:fp8-nhd", "fp8-nhd", "t5", _scale(MJ_FP8, 1.12)),
                _series("t5:fp8-interleaved", "fp8-interleaved", "t5", _scale(MJ_FP8, 1.1)),
                _series("t5:fp8-hnd", "fp8-hnd", "t5", MJ_FP8),
                _series("t5:e2m3-hnd", "e2m3-hnd", "t5", MJ_E2M3, 6.0),
                _series("t5:e3m2-hnd", "e3m2-hnd", "t5", _scale(MJ_E2M3, 1.01), 6.0),
                _series("t5:e2m1-hnd", "e2m1-hnd", "t5", _scale(MJ_FP8, 0.53), 4.0),
                _series("t5:nvfp4-hnd", "nvfp4-hnd", "t5", _scale(MJ_FP8, 1.6), 4.5),
            ],
            rows=[
                {"L": L, "series": s, "median_us": t * f, "gbps": None, "ok": True}
                for L, t in zip(LS, MJ_FP8, strict=True)
                for s, f in (("t5:fp8-nhd", 1.1), ("t5:e2m3-hnd", 0.82))
            ],
        ),
        "quant": _doc(
            "quant",
            rows=[
                {
                    "format": f,
                    "bits": b,
                    "L": L,
                    "rel_l2": e * (1 + 0.02 * i),
                    "max_abs": e * 3,
                    "emulator_pass": True,
                    "ok": True,
                }
                for f, b, e in (
                    ("fp8", 8, 0.045),
                    ("e2m3", 6, 0.051),
                    ("e3m2", 6, 0.080),
                    ("e2m1", 4, 0.20),
                    ("nvfp4", 4.5, 0.14),
                )
                for i, L in enumerate(LS)
            ],
        ),
        "ring": _doc(
            "ring",
            series=[fi]
            + [
                _series(f"t5:{n}", n, "t5", _scale(MJ_E2M3, f), 6.0)
                for n, f in (
                    ("default", 1.0),
                    ("stages=2", 1.08),
                    ("stages=3", 1.0),
                    ("stages=4", 1.04),
                    ("stages=5", 1.06),
                    ("stages=6", 1.12),
                )
            ],
        ),
        "prefetch": _doc(
            "prefetch",
            series=[fi]
            + [
                _series(f"t5:prefetch={k}", f"prefetch={k}", "t5", _scale(MJ_FP8, f))
                for k, f in ((0, 1.0), (1, 1.03), (2, 1.05), (4, 1.09))
            ],
        ),
        "ab": _doc(
            "ab",
            series=[fi]
            + [
                _series(f"t5:{n}", n, "t5", _scale(MJ_E2M3, f), 6.0)
                for n, f in (("base", 1.06), ("evict", 1.0), ("pdl", 1.05), ("both", 0.99))
            ],
            rows=[
                {"L": L, "series": s, "median_us": t, "gbps": None, "ok": True}
                for L, t in zip(LS, MJ_E2M3, strict=True)
                for s in ("t5:base", "t5:evict")
            ],
        ),
        "pdl": _doc(
            "pdl",
            rows=[
                {"q": "fp8", "batches": b, "L": L, "off_us": t, "on_us": t * (1 + d / 100), "delta_pct": d}
                for b in (1, 8)
                for L, t, d in zip(LS, MJ_E2M3, (-4, -2, -1, 1, -1, 0.5, 2, -0.5), strict=True)
            ],
        ),
        "fp6prof": _doc(
            "fp6prof",
            profiles=[
                {
                    "series": f"t5:{n}",
                    "label": n,
                    "kv_format": n.split("-")[0],
                    "kv_bits": bits,
                    "L": L,
                    "ok": True,
                    "metrics": {
                        "gpu__time_duration.sum": t * 1e3 * tf,
                        "lts__t_sectors_op_read.sum": L * 4 * 256 * 2 * bits / 8 / 32,
                        "lts__t_requests_srcunit_tex_op_read.sum": L * 16 * rq,
                    },
                    "stalls": {"long_scoreboard": 38.0 * tf, "barrier": 7.0 * tf, "wait": 0.9},
                    "pipes": {},
                    "details": {},
                }
                for n, bits, tf, rq in (
                    ("fp8-hnd", 8, 1.0, 1.0),
                    ("e2m3-hnd", 6, 0.95, 1.5),
                    ("e2m1-hnd", 4, 0.53, 1.0),
                )
                for L, t in zip(LS, MJ_FP8, strict=True)
            ],
        ),
        "correctness": _doc(
            "correctness",
            cases=[{"label": c[0], "q": q, "passed": True, "seconds": 20.0} for c in stages.CASES for q in c[2]],
        ),
    }


def write_all(verify_dir) -> None:
    for d in all_docs().values():
        results.save(verify_dir, d)
