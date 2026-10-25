"""The quant availability tables — the one in-code source of truth for what
loads and runs where (cycle 3a; content transcribed row by row from
``task/research/quants.md``, which is authoritative where a row is ambiguous).

Two tables of frozen :class:`Availability` cells:

* :data:`WEIGHTS_AVAILABILITY` — checkpoint format × platform;
* :data:`KV_AVAILABILITY` — kv-cache-dtype × attention-backend class × platform.

Each cell carries its ``state`` (:data:`NATIVE`, :data:`FALLBACK` or
:data:`UNAVAILABLE`) and a complete-sentence ``why`` naming the mechanism
(the vLLM kernel / fallback that decides at load time). Cells whose
on-device result is still a quants.md §gaps item carry ``probe=True`` —
:data:`PROBES` keys those items to their cells so a field result can flip
them trivially. The tables are dense where a combination is servable: an
*absent* cell means the combination cannot be served, and :func:`kv_state`
/:func:`weights_state` degrade to ``UNAVAILABLE`` with a why that names the
accepted keys (never a silent guess, never an exception).

Backend classes (:data:`MJOLNIR`, :data:`FLASHINFER`, :data:`FA2`, :data:`FA4`,
:data:`TRITON`) are addressed by class, never by backend-string soup:
:func:`backend_class` maps a config's ``attention-backend`` field value to
its class. Serve validation (``mjolnir_cli.overlays.validate``), leg
filtering and the cycle-3b renderers all read this one module.

Stdlib only — a pure data leaf (import-purity-pinned in
``tests/cli/test_host_import.py``); new formats/platforms are new rows,
never new code paths. Row order follows quants.md §1/§2 (weights) and the
cycle-2 matrix (KV), with ``bf16`` — the trivially-servable baseline — last
in the KV table.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjolnir_cli.platforms import DEFAULT_PLATFORM

# ── the vocabularies ────────────────────────────────────────────────────────

NATIVE = "native"
FALLBACK = "loads_with_fallback"
UNAVAILABLE = "unavailable"

# The attention-backend classes (the config field maps onto these; see
# :func:`backend_class`). FA4 is FLASH_ATTN with flash_attn_version 4 —
# its hd256 kernel is sm_100-only upstream, its fp8-KV route is thor-only here.
MJOLNIR = "MJOLNIR"
FLASHINFER = "FLASHINFER"
FA2 = "FA2"
FA4 = "FA4"
TRITON = "TRITON"

PLATFORMS = ("thor", "orin", "spark")


@dataclass(frozen=True)
class Availability:
    """One (format-or-dtype, backend, platform) cell: the state plus *why*.

    ``why`` is a complete sentence naming the mechanism — renderers quote it,
    serve validation embeds it in its error. ``probe`` marks the six
    quants.md §gaps items pending an on-device result."""

    state: str  # NATIVE | FALLBACK | UNAVAILABLE
    why: str
    probe: bool = False


@dataclass(frozen=True)
class Probe:
    """One quants.md §gaps item: its number (1–6), the table cells its
    on-device result can flip (``None`` for the two items that are not a
    weights/KV cell) and the transcribed note."""

    number: int
    key: tuple[tuple[str, str, str | None, str], ...] | None
    note: str


# ── weights: format × platform (quants.md §1) ───────────────────────────────

# Identical why-sentences shared by several platforms are hoisted here (the
# emitted cells are byte-identical to the per-platform literals they replaced).
_UNQUANTIZED = (
    "UnquantizedLinearMethod serves bf16 and fp16 everywhere; bf16 itself needs compute capability 8.0 or newer."
)
_FP8_NATIVE = "CUTLASS fp8 scaled-mm GEMM runs on the native fp8 tensor cores (compute capability 89 and up)."
_INT8_W8A8 = (
    "CutlassInt8ScaledMM runs W8A8 on the int8 tensor cores every one of these platforms has "
    "(the c2x kernels are built for sm_87 too)."
)
_MARLIN_W4A16 = (
    "Marlin serves W4A16 (GPTQ/AWQ) from compute capability 7.5 up — the bandwidth-optimized classic edge format."
)
_NO_FP4 = (
    "No supported native FP4 GEMM: vLLM falls back to weight-only Marlin W4A16 — the memory win only, "
    "never W4A4 compute."
)
_HUMMING = (
    "The Humming kernel (SM75+) serves OCP MXFP6 weights (e2m3/e3m2) when the optional humming "
    "package is installed in the image."
)
_MXFP8_MARLIN = "ModelOpt MXFP8 rides the Marlin kernel on SM80+ (OCP microscaling with E8M0 scales)."
_NO_FP6_WEIGHTS = (
    "No stock fp6 weights path exists in vLLM 0.32.0: Mjolnir's fp6 scheme is a KV-cache format, not a weights format."
)

WEIGHTS_AVAILABILITY: dict[str, dict[str, Availability]] = {
    "bf16": {p: Availability(NATIVE, _UNQUANTIZED) for p in PLATFORMS},
    "fp8-w8a8": {
        "thor": Availability(NATIVE, _FP8_NATIVE),
        # no fp8 tensor cores on sm_87: vLLM loads the checkpoint anyway and
        # logs the Marlin weight-only warning — a memory win, never fp8 speed
        "orin": Availability(
            FALLBACK,
            "No fp8 tensor cores here: vLLM auto-falls back to weight-only W8A16 via Marlin, "
            "so the checkpoint loads with a memory win but no fp8 compute.",
        ),
        "spark": Availability(NATIVE, _FP8_NATIVE),
    },
    "int8-w8a8": {p: Availability(NATIVE, _INT8_W8A8) for p in PLATFORMS},
    "int4-w4a16": {
        "thor": Availability(NATIVE, _MARLIN_W4A16),
        # probe gap 2: Marlin's effective perf on sm_87 (8.0+PTX JIT on the iGPU)
        "orin": Availability(NATIVE, _MARLIN_W4A16, probe=True),
        "spark": Availability(NATIVE, _MARLIN_W4A16),
    },
    "nvfp4": {
        # probe gap 4: which NVFP4 GEMM kernel actually wins on thor 0.32.0
        "thor": Availability(
            NATIVE,
            "Native fp4 GEMM (compute capability 100 to 121) — though our 0.32.0 runs picked Marlin W4A16 "
            "for dense GEMM, the per-load log line decides.",
            probe=True,
        ),
        "orin": Availability(FALLBACK, _NO_FP4),
        "spark": Availability(NATIVE, "SM12x scaled-mm kernels plus FlashInfer's 120f cubins run NVFP4 W4A4 natively."),
    },
    # OCP microscaling weights: both ride the optional Humming kernel (probe gap 5:
    # humming is not in the image requirements, so its presence is per image)
    "mxfp6": {p: Availability(NATIVE, _HUMMING, probe=True) for p in PLATFORMS},
    "mxfp8": {p: Availability(NATIVE, _MXFP8_MARLIN, probe=True) for p in PLATFORMS},
    # quants.md correction 1: Mjolnir's fp6 is a KV-cache scheme, not weights —
    # there is no stock fp6-weights path on any platform
    "mjolnir-fp6": {p: Availability(UNAVAILABLE, _NO_FP6_WEIGHTS) for p in PLATFORMS},
}

# ── KV cache: dtype × backend class × platform (quants.md §2) ────────────────
# Only servable combinations are rows; an absent (backend, platform) cell means
# the combination cannot be served and kv_state composes its why from _KV_WHYS.

TRITON_ONLY = (
    "Only the Triton attention backend reads {dtype} KV (dynamic per-token-head scales computed at cache-write time)."
)
_FLASHINFER_FP8 = "The FlashInfer backend serves fp8 KV across its whole sm_80 to sm_121 support window."
_FA2_NVFP4_KV = (
    "The FA2 pathway serves NVFP4 KV on its sm_80/90/120 families (storage-level: packed fp4 pages, fp8 block scales)."
)
_FI_NVFP4_KV = "The FlashInfer backend enables NVFP4 KV on every family the FA2 path covers."
_INT4_KV = "Only the Triton attention backend reads int4 KV, the same per-token-head shape as int8."
_BF16_KV = (
    "BF16 (and fp16) KV is the unquantized baseline: every backend class serves it wherever that backend itself runs."
)

KV_AVAILABILITY: dict[str, dict[str, dict[str, Availability]]] = {
    "fp8": {
        FLASHINFER: {p: Availability(NATIVE, _FLASHINFER_FP8) for p in PLATFORMS},
        # the proven fa4-fp8kv route (hd<=128); FA2 itself rejects quantized KV
        FA4: {
            "thor": Availability(
                NATIVE,
                "FA4 serves fp8 KV on sm_110 (the proven fa4-fp8kv route); FA2 still rejects every quantized KV dtype.",
            )
        },
        MJOLNIR: {
            "thor": Availability(NATIVE, "The Mjolnir kernel reads e4m3 KV natively on thor as its default dtype."),
            # the Orin GEMV + mma.sync decode kernels: e4m3 KV, head_dim 256,
            # q_len <= 8 — measured against FlashInfer FA2-TC across the
            # M/L band (research/runs/orin1/notes/mma-kernel-design-2.md)
            "orin": Availability(
                NATIVE,
                "The Orin Mjolnir kernels (the FMA GEMV and the mma.sync decode: hd 256, q_len <= 8) read e4m3 KV natively.",
            ),
        },
    },
    "fp6_e2m3": {
        MJOLNIR: {
            "thor": Availability(
                NATIVE,
                "The Mjolnir kernel is the only fp6 reader, with per-(layer, KV-head, channel) calibrated scales.",
            )
        },
    },
    "nvfp4": {
        # probe gap 1: NVFP4-KV decode on Ampere (flashinfer-native wrapper,
        # uint8 pages) — code path exists, no field report found
        FA2: {
            "orin": Availability(NATIVE, _FA2_NVFP4_KV, probe=True),
            "spark": Availability(NATIVE, _FA2_NVFP4_KV),
        },
        FLASHINFER: {
            "orin": Availability(NATIVE, _FI_NVFP4_KV, probe=True),
            "spark": Availability(NATIVE, _FI_NVFP4_KV),
        },
    },
    "int8": {
        TRITON: {p: Availability(NATIVE, TRITON_ONLY.format(dtype="int8")) for p in PLATFORMS},
    },
    "int4": {
        TRITON: {p: Availability(NATIVE, _INT4_KV) for p in PLATFORMS},
    },
    "bf16": {
        cls: {p: Availability(NATIVE, _BF16_KV) for p in PLATFORMS} for cls in (MJOLNIR, FLASHINFER, FA2, FA4, TRITON)
    },
}

# The dtype-level whys, transplanted verbatim from the cycle-2 overlay matrix
# (mjolnir_cli.overlays._WHYS): serve validation embeds them in its error
# messages byte-identically, so they deliberately carry NO trailing period.
# A table cell's why ends with one; these do not.
_KV_WHYS: dict[str, str] = {
    "fp6_e2m3": "fp6 (e2m3) KV has no fallback reader — only the Mjolnir kernel (MJOLNIR) on thor serves it",
    "fp8": "fp8 KV is served by FlashInfer on every platform, by FA4 on thor, and by the Mjolnir kernels on thor and Orin (hd 256, q_len <= 8)",
    "nvfp4": "nvfp4 KV rides the FA2 path (fa2-nvfp4 families are sm_80/90/120, never sm_110) and has no FlashInfer path",
    "int8": "int8 KV is int8_per_token_head — served by the Triton attention backend only",
    "int4": "int4 KV is int4_per_token_head — served by the Triton attention backend only",
    "bf16": "bf16 (and fp16) KV is the unquantized baseline every backend serves",
}

# The quants.md §gaps items, keyed to the cells they can flip. Gaps 3 (GDN
# layers on Orin) and 6 (the CUDA-programming-guide fetch) are not a
# weights/KV cell: their key is None. Cell keys are (kind, format, backend
# class or None, platform) — the same tuples the probe scan in
# tests/cli/test_availability.py collects.
PROBES: tuple[Probe, ...] = (
    Probe(
        1,
        (("kv", "nvfp4", FA2, "orin"), ("kv", "nvfp4", FLASHINFER, "orin")),
        "NVFP4-KV decode on Ampere (the flashinfer-native wrapper reading uint8 pages) — the code path exists, "
        "no field report found.",
    ),
    Probe(
        2,
        (("weights", "int4-w4a16", None, "orin"),),
        "Marlin's effective perf on sm_87 (the 8.0+PTX JIT path on the iGPU) — decides W4A16 vs INT8-W8A8 "
        "as the Orin default.",
    ),
    Probe(
        3,
        None,
        "GDN layers on Orin: the fused Gated-DeltaNet decode kernel has no sm_87 gencode — expect the "
        "Triton/FlashInfer mamba fallback.",
    ),
    Probe(
        4,
        (("weights", "nvfp4", None, "thor"),),
        "Which NVFP4 GEMM kernel actually wins on thor under vLLM 0.32.0 (a 35B run observed Marlin W4A16) — "
        "the 'Using ... for NVFP4 GEMM' log line is the oracle.",
    ),
    Probe(
        5,
        tuple(("weights", fmt, None, plat) for fmt in ("mxfp6", "mxfp8") for plat in PLATFORMS),
        "Humming availability inside the vLLM images (an optional dependency) — gates the MXFP6/MXFP8 weight paths.",
    ),
    Probe(
        6,
        None,
        "The CUDA programming guide's per-arch feature table could not be fetched (cookie wall); the per-arch "
        "claims rest on vLLM build + runtime code, the FlashInfer audit and TRT-Edge-LLM's platform statement.",
    ),
)

# The attention-backend field values -> their class. Unknown values pass
# through uppercased so a validation error can name them verbatim.
_FIELD_TO_CLASS: dict[str, str] = {
    "MJOLNIR": MJOLNIR,
    "FLASHINFER": FLASHINFER,
    "FLASH_ATTN": FA2,
    "FLASH_ATTN@4": FA4,
    "TRITON": TRITON,
    "TRITON_ATTN": TRITON,
}

# With no attention-backend declared, the platform's default serving class
# (vLLM auto-selects; on thor that is our backend, elsewhere FlashInfer).
_PLATFORM_DEFAULT: dict[str, str] = {"thor": MJOLNIR, "orin": FLASHINFER, "spark": FLASHINFER}


def backend_class(field: str | None, platform: str = DEFAULT_PLATFORM) -> str:
    """The backend class a config's ``attention-backend`` field value selects.

    ``None`` (nothing declared) infers the platform's default class; an
    unknown field value passes through uppercased so an error can name it."""
    if not field:
        return _PLATFORM_DEFAULT.get(platform, FLASHINFER)
    f = str(field).upper()
    return _FIELD_TO_CLASS.get(f, f)


def weights_state(fmt: str, platform: str) -> Availability:
    """The availability cell for checkpoint format ``fmt`` on ``platform`` —
    ``UNAVAILABLE`` with a why naming the accepted keys when either is
    unknown (never an exception: renderers quote the why)."""
    row = WEIGHTS_AVAILABILITY.get(fmt)
    if row is None:
        return Availability(
            UNAVAILABLE,
            f"Weights format {fmt!r} is not in the availability table — known formats: "
            f"{', '.join(sorted(WEIGHTS_AVAILABILITY))}.",
        )
    if platform not in PLATFORMS:
        return Availability(
            UNAVAILABLE,
            f"Platform {platform!r} is not a shipped platform — the availability table covers {', '.join(PLATFORMS)}.",
        )
    return row[platform]


def kv_state(dtype: str, backend: str, platform: str) -> Availability:
    """The availability cell for ``kv-cache-dtype`` ``dtype`` served by
    backend class ``backend`` on ``platform``.

    A dtype the table does not know, or a platform that is not shipped,
    returns ``UNAVAILABLE`` naming the accepted keys. A known dtype/platform
    whose backend class has no cell there returns ``UNAVAILABLE`` with the
    dtype-level why (the transplanted cycle-2 string — no trailing period,
    embedded verbatim in serve-validation errors)."""
    row = KV_AVAILABILITY.get(dtype)
    if row is None:
        return Availability(
            UNAVAILABLE,
            f"kv-cache-dtype {dtype!r} is not in the availability table — known dtypes: "
            f"{', '.join(sorted(KV_AVAILABILITY))}.",
        )
    if platform not in PLATFORMS:
        return Availability(
            UNAVAILABLE,
            f"Platform {platform!r} is not a shipped platform — the availability table covers {', '.join(PLATFORMS)}.",
        )
    cells = row.get(backend)
    if cells is None or platform not in cells:
        return Availability(UNAVAILABLE, _KV_WHYS[dtype])
    return cells[platform]
