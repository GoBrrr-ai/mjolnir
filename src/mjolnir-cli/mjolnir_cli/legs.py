"""Default legs of every benchmark — the one place that decides what gets
measured against what.

A **serving leg** is ``(side, config)``: ``official`` serves the config on the
upstream ``vllm/vllm-openai`` image (native backends are only ever measured
there), ``mjolnir`` on the Mjolnir image. Each command renders the table in
its own CLI syntax — ``verify`` stages take ``side:config``
(:func:`verify_legs`), ``bench`` commands take ``<image>:<config>`` or a bare
config served from ``--image`` (:func:`bench_legs`).

A **kernel leg** is a harness leg id from :mod:`mjolnir_cli.verify.legs`
(``fi``, ``fi-hnd``, ``fa2``, ``fa4``, ``mjolnir-fp8``, ``mjolnir``), used by
the ncu stages.

Change a list here and every command, its ``--help`` default and the
"is this a headline (default) run" check follow. ``--legs`` / ``--backends``
on the command line still override one run.
"""

from __future__ import annotations

from pathlib import Path

from mjolnir_cli.config import BASELINE_QUANT, DEFAULT_QUANT, FAST_QUANT, configs_root, user_configs_dir
from mjolnir_cli.overlays import config_exists_on
from mjolnir_cli.platforms import DEFAULT_PLATFORM, REGISTRY

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

# Opt-in serving legs: measured only when asked (`bench e2e --optional`), after
# the default legs — other speculative decoders on both images. They land in
# the history (`mjolnir history`, `bench compare --config <cfg>`) but never
# replace the headline pair, so the README stays on the default legs.
OPTIONAL: dict[str, tuple[Leg, ...]] = {
    "bench.e2e": (
        (OFFICIAL, "NVFP4_DSPARK"),  # stock vLLM: FlashInfer, fp8 KV, DSpark K=7
        (OFFICIAL, "NVFP4_DFLASH2"),  # stock vLLM: FlashInfer, fp8 KV, DFlash2 K=7
        (MJOLNIR, "NVFP4_MJOLNIR_E2M3_DSPARK"),  # Mjolnir fp6 KV, DSpark K=7
        (MJOLNIR, "NVFP4_MJOLNIR_E2M3_DFLASH2"),  # Mjolnir fp6 KV, DFlash2 K=7
    ),
}

# ── kernel legs (ncu stages) ────────────────────────────────────────────────
KERNEL: dict[str, str] = {
    # the FA-family baseline is FA4 hd256 — as the NVFP4_FA4hd256 config serves
    # it on the Mjolnir image (its patched FA4): 1-CTA decode carve-out, no
    # SplitKV (structural in the vendored hd256 kernel), fp8 Q+KV. FA2 stays
    # available via explicit --legs (bf16 KV; research platforms keep it as
    # _RESEARCH_KERNEL_LEGS — FA4 is sm_110-only).
    "verify.sweep": "fi,fa4,mjolnir-fp8,mjolnir",
    "verify.hndsweep": "fi,fi-hnd,mjolnir-fp8",
    "verify.kvfmt.refs": "fi,fi-hnd",
    # the autoresearch keep/discard harness: FlashInfer vs the kernel under test
    "autoresearch.evaluate": "fi,mjolnir",
}

# Platform overrides of the KERNEL tables: a platform whose headline stage
# measures differently names its own leg ids here (looked up before the
# default table). The Orin's headline sweep is the events-timed same-run
# comparison (the ratio that survives the board's noise); with ncu installed
# the HND stage profiles the GEMV for real — FA4 is absent (sm_110-only) and
# the kvfmt/quant variant lists need their own Orin sets (the GEMV harness
# speaks fp8/bf16/e2m3 only) before those stages open up.
_KERNEL_BY_PLATFORM: dict[str, dict[str, str]] = {
    "orin": {
        "verify.sweep": "mjolnir,mjolnir-full,fi",
        "verify.hndsweep": "fi,fi-hnd,mjolnir-fp8",
    },
}

# ── `verify vllm`: greedy MJOLNIR-vs-FlashInfer smoke modes ────────────────
SMOKE_MODES = "graph eager e2m3 lmh8"

# Research platforms measure the FlashInfer + FA2 baselines only until their
# kernel lands (the ncu stages need this repo's image to run the harness in).
_RESEARCH_KERNEL_LEGS = "fi,fa2"

# A research platform's own serving baseline: the one official leg its stage
# plans measure — the platform's native compute path, which the thor table
# does not carry. The default platform needs no entry (its table is the thor
# table); spark keeps the thor table's NVFP4 baseline.
_BASELINE: dict[str, str] = {
    "orin": "AWQ_FLASHINFER",  # int4 W4A16 (AWQ) — Marlin, native on sm_87
}


def legs_for_stage(name: str, platform: str = DEFAULT_PLATFORM) -> tuple[Leg, ...]:
    """The stage's legs on a platform: the full table on a supported one; on
    a research platform its official baseline only — the platform's own
    baseline config when it names one (``_BASELINE``), else the thor table's
    official side. The Mjolnir legs arrive with the platform's kernel."""
    ls = SERVING[name]
    if REGISTRY[platform].status == "supported":
        return ls
    baseline = _BASELINE.get(platform)
    if baseline is not None:
        return ((OFFICIAL, baseline),)
    return _uniq([(side, cfg) for side, cfg in ls if side == OFFICIAL])


def available_serving_legs(
    stage: str,
    platform: str = DEFAULT_PLATFORM,
    model: str | None = None,
    configs_dir: Path | None = None,
) -> tuple[Leg, ...]:
    """The stage's legs for a platform, filtered to configs that exist for the
    model — a config dir that is not there cannot be served, and per the
    overlay rule a config exists on a platform only when its
    ``platforms/<platform>.yaml`` overlay is there (no ``platforms/`` dir =
    legacy thor-only). No model (or no filter hit at all) skips the
    filesystem check; ``configs_dir`` defaults to the canonical + user-local
    roots."""
    ls = legs_for_stage(stage, platform)
    if model is None:
        return ls
    roots = [configs_dir] if configs_dir is not None else [configs_root(), user_configs_dir()]
    m = model  # narrowed; captured by the filter expression below

    def served(cfg: str) -> bool:
        return any(config_exists_on(root / m / cfg, platform) for root in roots)

    return tuple(leg for leg in ls if served(leg[1]))


def kernel_legs(stage: str, platform: str = DEFAULT_PLATFORM) -> str:
    """The ncu stage's harness-leg ids for a platform (research platforms get
    the FlashInfer + FA2 baselines). A platform override (e.g. the Orin's
    events-timed sweep series) wins over the default table."""
    per = _KERNEL_BY_PLATFORM.get(platform)
    if per and stage in per:
        return per[stage]
    if REGISTRY[platform].status == "supported":
        return KERNEL[stage]
    return _RESEARCH_KERNEL_LEGS


def verify_legs(name: str, platform: str = DEFAULT_PLATFORM) -> str:
    """``official:NVFP4,mjolnir:NVFP4_MJOLNIR,…`` — the ``verify --legs`` form
    for a platform (the default platform's table: the import-time typer
    contract)."""
    return ",".join(f"{side}:{cfg}" for side, cfg in legs_for_stage(name, platform))


def bench_legs(name: str, platform: str = DEFAULT_PLATFORM, model: str | None = None) -> str:
    """``vllm/vllm-openai:…:NVFP4,NVFP4_MJOLNIR_E2M3,…`` — the ``bench
    --backends`` form (official legs pinned to the platform's upstream image,
    Mjolnir legs bare). With a ``model``, the legs are filtered to configs
    that exist for it."""
    ls = available_serving_legs(name, platform, model) if model else legs_for_stage(name, platform)
    stock = REGISTRY[platform].stock_image
    return ",".join(f"{stock}:{cfg}" if side == OFFICIAL else cfg for side, cfg in ls)


def side_configs(name: str, side: str) -> list[str]:
    return [cfg for s, cfg in SERVING[name] if s == side]


def parse_leg(text: str, default_side: str = MJOLNIR) -> Leg:
    """``official:NVFP4_DSPARK`` / ``mjolnir:NVFP4_MJOLNIR_E2M3_DSPARK`` / a bare
    config (``default_side``) -> (side, config)."""
    side, sep, cfg = text.strip().partition(":")
    if sep and side.lower() in (OFFICIAL, MJOLNIR):
        return side.lower(), cfg.strip()
    return default_side, text.strip()
