"""Default legs of every benchmark — the one place that decides what gets
measured against what.

A **serving leg** is ``(side, config)``: ``official`` serves the config on the
upstream ``vllm/vllm-openai`` image (native backends are only ever measured
there), ``mjolnir`` on the Mjolnir image. Each command renders the table in
its own CLI syntax — ``verify`` stages take ``side:config``
(:func:`verify_legs`), ``bench`` commands take ``<image>:<config>`` or a bare
config served from ``--image`` (:func:`bench_legs`).

A **kernel leg** is a harness leg id from :mod:`mjolnir_cli.verify.legs`
(``fi``, ``fi-hnd``, ``fa2``, ``mjolnir-fp8``, ``mjolnir``), used by the ncu
stages.

Change a list here and every command, its ``--help`` default and the
"is this a headline (default) run" check follow. ``--legs`` / ``--backends``
on the command line still override one run.
"""

from __future__ import annotations

from mjolnir_cli.config import BASELINE_QUANT, DEFAULT_QUANT, FAST_QUANT, STOCK_IMAGE

OFFICIAL, MJOLNIR = "official", "mjolnir"
Leg = tuple[str, str]  # (side, config)

FP8 = "NVFP4_MJOLNIR"  # Mjolnir, fp8 KV
FP6 = "NVFP4_MJOLNIR_E2M3"  # Mjolnir, fp6 (e2m3) KV


def _uniq(legs: list[Leg]) -> tuple[Leg, ...]:
    """Drop repeats, keep order (DEFAULT_QUANT may equal FP6 or FAST_QUANT)."""
    return tuple(dict.fromkeys(legs))


# ── serving legs ────────────────────────────────────────────────────────────
SERVING: dict[str, tuple[Leg, ...]] = {
    # vLLM's KV sizing at startup: stock fp8 KV, Mjolnir fp8 / fp6 KV, and the
    # fast config (its fp8 lm_head copy costs KV memory)
    "verify.kv-cache-size": _uniq([(OFFICIAL, BASELINE_QUANT), (MJOLNIR, FP8), (MJOLNIR, FP6), (MJOLNIR, FAST_QUANT)]),
    # MTP acceptance and decode rate per draft count
    "verify.spec": _uniq([(OFFICIAL, BASELINE_QUANT), (MJOLNIR, FP8), (MJOLNIR, FP6), (MJOLNIR, FAST_QUANT)]),
    # torch-profiler trace of one decode window
    "verify.profile": _uniq([(OFFICIAL, BASELINE_QUANT), (MJOLNIR, FP6), (MJOLNIR, FAST_QUANT)]),
    # llama-benchy A/B, one leg after the other
    "bench.ab": _uniq([(OFFICIAL, BASELINE_QUANT), (MJOLNIR, DEFAULT_QUANT), (MJOLNIR, FAST_QUANT)]),
    # official vs Mjolnir end to end: the first official and the first mjolnir
    # leg are the headline pair (comparison table, README infographic); further
    # mjolnir legs are measured after it (`bench e2e --also`) and join the charts
    "bench.e2e": _uniq([(OFFICIAL, BASELINE_QUANT), (MJOLNIR, DEFAULT_QUANT), (MJOLNIR, FAST_QUANT)]),
    # model quality (tool-eval-bench)
    "bench.accuracy": _uniq(
        [
            (OFFICIAL, BASELINE_QUANT),
            (MJOLNIR, FP8),
            (MJOLNIR, FP6),
            (MJOLNIR, FAST_QUANT),
        ]
    ),
}

# ── kernel legs (ncu stages) ────────────────────────────────────────────────
KERNEL: dict[str, str] = {
    "verify.sweep": "fi,fa2,mjolnir-fp8,mjolnir",
    "verify.hndsweep": "fi,fi-hnd,mjolnir-fp8",
    "verify.kvfmt.refs": "fi,fi-hnd",
    # the autoresearch keep/discard harness: FlashInfer vs the kernel under test
    "autoresearch.evaluate": "fi,mjolnir",
}

# ── `verify vllm`: greedy MJOLNIR-vs-FlashInfer smoke modes ────────────────
SMOKE_MODES = "graph eager e2m3 lmh8"


def verify_legs(name: str) -> str:
    """``official:NVFP4,mjolnir:NVFP4_MJOLNIR,…`` — the ``verify --legs`` form."""
    return ",".join(f"{side}:{cfg}" for side, cfg in SERVING[name])


def bench_legs(name: str) -> str:
    """``vllm/vllm-openai:…:NVFP4,NVFP4_MJOLNIR_E2M3,…`` — the ``bench --backends``
    form (official legs pinned to the upstream image, Mjolnir legs bare)."""
    return ",".join(f"{STOCK_IMAGE}:{cfg}" if side == OFFICIAL else cfg for side, cfg in SERVING[name])


def side_configs(name: str, side: str) -> list[str]:
    return [cfg for s, cfg in SERVING[name] if s == side]
