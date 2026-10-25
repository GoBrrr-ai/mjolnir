"""The quant availability tables (``mjolnir_cli.availability``, new in cycle 3a).

One in-code source of truth (task/research/quants.md, transcribed row by row):
``WEIGHTS_AVAILABILITY`` (format x platform) and ``KV_AVAILABILITY``
(kv-cache-dtype x backend class x platform), each cell a frozen
:class:`Availability` with its ``state`` (native / loads_with_fallback /
unavailable), a complete-sentence ``why`` naming the mechanism, and a
``probe`` flag for the six quants.md §gaps items pending on-device results.
Lookups are pure functions: ``weights_state``, ``kv_state`` and
``backend_class`` (the config ``attention-backend`` field -> class, via the
``BACKEND_LABELS`` prefixes). The tables are stdlib-only data — renderers
(cycle 3b) and serve-validation (``overlays.validate``) all read this one
module; new formats/platforms are new rows, never new code paths.

States are compared by value everywhere (the module defines the three
constants), mechanism tokens are matched case-insensitively, and nothing in
this file touches the GPU stack — the module is import-purity-tested in
``test_host_import.py``.
"""

from __future__ import annotations

import pytest

from mjolnir_cli import availability
from mjolnir_cli.availability import (
    FA2,
    FA4,
    FLASHINFER,
    MJOLNIR,
    TRITON,
    FALLBACK,
    NATIVE,
    UNAVAILABLE,
    KV_AVAILABILITY,
    WEIGHTS_AVAILABILITY,
    backend_class,
    kv_state,
    weights_state,
)

THOR, ORIN, SPARK = "thor", "orin", "spark"
PLATFORMS = (THOR, ORIN, SPARK)

# quants.md §1 rows, transcribed: (format, platform, state, mechanism tokens
# the why must name). The tokens are the vLLM mechanisms quants.md names for
# the cell (kernel / fallback names); the full sentences are pinned by the
# style lint below, not here.
WEIGHTS_ROWS: list[tuple[str, str, str, tuple[str, ...]]] = [
    # bf16 / fp16: UnquantizedLinearMethod, bf16 needs cc >= 8.0 — all native
    ("bf16", THOR, NATIVE, ("UnquantizedLinearMethod",)),
    ("bf16", ORIN, NATIVE, ("UnquantizedLinearMethod",)),
    ("bf16", SPARK, NATIVE, ("UnquantizedLinearMethod",)),
    # fp8 e4m3 W8A8: fp8 tensor cores on thor (CUTLASS scaled-mm, cc >= 89)
    # and spark; orin auto-falls back to weight-only W8A16 via Marlin
    ("fp8-w8a8", THOR, NATIVE, ("CUTLASS",)),
    ("fp8-w8a8", ORIN, FALLBACK, ("Marlin", "W8A16")),
    ("fp8-w8a8", SPARK, NATIVE, ("CUTLASS",)),
    # INT8 W8A8: int8 tensor cores on all three (CUTLASS c2x built incl. 8.7)
    ("int8-w8a8", THOR, NATIVE, ("CutlassInt8ScaledMM",)),
    ("int8-w8a8", ORIN, NATIVE, ("CutlassInt8ScaledMM",)),
    ("int8-w8a8", SPARK, NATIVE, ("CutlassInt8ScaledMM",)),
    # INT4 W4A16 (AWQ/GPTQ): Marlin, cc >= 7.5 — the classic edge format
    ("int4-w4a16", THOR, NATIVE, ("Marlin",)),
    ("int4-w4a16", ORIN, NATIVE, ("Marlin",)),
    ("int4-w4a16", SPARK, NATIVE, ("Marlin",)),
    # NVFP4: native W4A4 where fp4 GEMM exists (cc 100-121) — thor + spark;
    # orin auto-falls back to W4A16 Marlin (memory win only)
    ("nvfp4", THOR, NATIVE, ("fp4",)),
    ("nvfp4", ORIN, FALLBACK, ("Marlin W4A16", "memory")),
    ("nvfp4", SPARK, NATIVE, ("SM12x",)),
    # MXFP6 / MXFP8 weights (OCP microscaling): Humming (SM75+) for mxfp6;
    # ModelOpt MXFP8 rides Marlin on SM80+ — all three platforms
    ("mxfp6", THOR, NATIVE, ("Humming",)),
    ("mxfp6", ORIN, NATIVE, ("Humming",)),
    ("mxfp6", SPARK, NATIVE, ("Humming",)),
    ("mxfp8", THOR, NATIVE, ("Marlin",)),
    ("mxfp8", ORIN, NATIVE, ("Marlin",)),
    ("mxfp8", SPARK, NATIVE, ("Marlin",)),
    # mjolnir-fp6 weights: no such stock path anywhere — Mjolnir's fp6 is a
    # KV-cache scheme, not a weights format (quants.md correction 1)
    ("mjolnir-fp6", THOR, UNAVAILABLE, ("KV",)),
    ("mjolnir-fp6", ORIN, UNAVAILABLE, ("KV",)),
    ("mjolnir-fp6", SPARK, UNAVAILABLE, ("KV",)),
]

# ── 1. the weights matrix (quants.md §1, row by row) ────────────────────────


@pytest.mark.parametrize(("fmt", "platform", "state", "tokens"), WEIGHTS_ROWS)
def test_weights_table_matches_the_quants_rows(fmt, platform, state, tokens):
    cell = weights_state(fmt, platform)
    assert cell.state == state, (fmt, platform)
    for token in tokens:
        assert token.lower() in cell.why.lower(), (fmt, platform, token)


def test_weights_table_covers_every_format_on_every_platform():
    assert set(WEIGHTS_AVAILABILITY) == {
        "bf16",
        "nvfp4",
        "fp8-w8a8",
        "int8-w8a8",
        "int4-w4a16",
        "mxfp6",
        "mxfp8",
        "mjolnir-fp6",
    }
    for fmt, row in WEIGHTS_AVAILABILITY.items():
        assert set(row) == set(PLATFORMS), fmt  # a dense table: no missing platform


def test_nvfp4_on_orin_is_the_marlin_w4a16_memory_win_only_fallback():
    # the flagship quants.md cell (feature.md AC1 names its why)
    cell = weights_state("nvfp4", ORIN)
    assert cell.state == FALLBACK
    assert "Marlin W4A16" in cell.why
    assert "memory" in cell.why.lower()


# ── 2. the KV matrix (quants.md §2, via the class vocabulary) ────────────────

# (dtype, backend class, platform, state, mechanism tokens). Unavailable
# combos return the dtype-level why — the string transplanted verbatim from
# the cycle-2 overlay matrix, pinned byte-identical in test_overlays.py.
KV_ROWS: list[tuple[str, str, str, str, tuple[str, ...]]] = [
    # fp8: FlashInfer serves it on every platform; FA4 on thor; the Mjolnir
    # kernels on thor and Orin (the Orin GEMV + mma.sync, hd 256, q_len <= 8);
    # FA2 rejects quantized KV everywhere (family-110 included)
    ("fp8", FLASHINFER, THOR, NATIVE, ("FlashInfer",)),
    ("fp8", FLASHINFER, ORIN, NATIVE, ("FlashInfer",)),
    ("fp8", FLASHINFER, SPARK, NATIVE, ("FlashInfer",)),
    ("fp8", FA4, THOR, NATIVE, ("FA4", "sm_110")),
    ("fp8", MJOLNIR, THOR, NATIVE, ("Mjolnir",)),
    ("fp8", MJOLNIR, ORIN, NATIVE, ("Mjolnir", "hd 256", "q_len")),
    ("fp8", FA2, THOR, UNAVAILABLE, ()),
    ("fp8", FA2, ORIN, UNAVAILABLE, ()),
    ("fp8", FA2, SPARK, UNAVAILABLE, ()),
    ("fp8", FA4, ORIN, UNAVAILABLE, ()),
    ("fp8", FA4, SPARK, UNAVAILABLE, ()),
    ("fp8", MJOLNIR, SPARK, UNAVAILABLE, ()),
    # fp6 (e2m3): the Mjolnir kernel on thor — no fallback reader anywhere
    ("fp6_e2m3", MJOLNIR, THOR, NATIVE, ("Mjolnir", "calibrated scales")),
    ("fp6_e2m3", MJOLNIR, ORIN, UNAVAILABLE, ()),
    ("fp6_e2m3", MJOLNIR, SPARK, UNAVAILABLE, ()),
    ("fp6_e2m3", FLASHINFER, THOR, UNAVAILABLE, ()),
    # nvfp4: the FA2 path (families 80/90/120) and the FlashInfer backend on
    # orin + spark — never sm_110, which falls through both
    ("nvfp4", FA2, ORIN, NATIVE, ("FA2",)),
    ("nvfp4", FA2, SPARK, NATIVE, ("FA2",)),
    ("nvfp4", FLASHINFER, ORIN, NATIVE, ("FlashInfer",)),
    ("nvfp4", FLASHINFER, SPARK, NATIVE, ("FlashInfer",)),
    ("nvfp4", FA2, THOR, UNAVAILABLE, ()),
    ("nvfp4", FLASHINFER, THOR, UNAVAILABLE, ()),
    # int8 / int4 per-token-head: the Triton backend only, on all three
    ("int8", TRITON, THOR, NATIVE, ("Triton",)),
    ("int8", TRITON, ORIN, NATIVE, ("Triton",)),
    ("int8", TRITON, SPARK, NATIVE, ("Triton",)),
    ("int8", FLASHINFER, THOR, UNAVAILABLE, ()),
    ("int4", TRITON, THOR, NATIVE, ("Triton",)),
    ("int4", TRITON, ORIN, NATIVE, ("Triton",)),
    ("int4", TRITON, SPARK, NATIVE, ("Triton",)),
    ("int4", MJOLNIR, THOR, UNAVAILABLE, ()),
    # bf16 / fp16: trivially everywhere, every backend
    ("bf16", FA2, THOR, NATIVE, ()),
    ("bf16", FA2, ORIN, NATIVE, ()),
    ("bf16", FA2, SPARK, NATIVE, ()),
    ("bf16", FA4, THOR, NATIVE, ()),
    ("bf16", FA4, ORIN, NATIVE, ()),
    ("bf16", FA4, SPARK, NATIVE, ()),
    ("bf16", FLASHINFER, THOR, NATIVE, ()),
    ("bf16", FLASHINFER, ORIN, NATIVE, ()),
    ("bf16", FLASHINFER, SPARK, NATIVE, ()),
    ("bf16", MJOLNIR, THOR, NATIVE, ()),
    ("bf16", MJOLNIR, ORIN, NATIVE, ()),
    ("bf16", MJOLNIR, SPARK, NATIVE, ()),
    ("bf16", TRITON, THOR, NATIVE, ()),
    ("bf16", TRITON, ORIN, NATIVE, ()),
    ("bf16", TRITON, SPARK, NATIVE, ()),
]


@pytest.mark.parametrize(("dtype", "backend", "platform", "state", "tokens"), KV_ROWS)
def test_kv_table_matches_the_quants_rows(dtype, backend, platform, state, tokens):
    cell = kv_state(dtype, backend, platform)
    assert cell.state == state, (dtype, backend, platform)
    for token in tokens:
        assert token.lower() in cell.why.lower(), (dtype, backend, platform, token)


def test_kv_table_rows_carry_every_declared_cell():
    # the tables are dense where a class serves a platform: each dtype row
    # holds at least its native class -> platform cells
    assert set(KV_AVAILABILITY) == {"fp8", "fp6_e2m3", "nvfp4", "int8", "int4", "bf16"}
    for dtype, row in KV_AVAILABILITY.items():
        assert row, dtype  # every dtype names the classes that serve it
        for backend, cells in row.items():
            assert cells, (dtype, backend)  # and those classes their platforms


# ── 3. unknown keys degrade to UNAVAILABLE naming the accepted keys ─────────


def test_weights_state_on_an_unknown_format_is_unavailable_naming_the_formats():
    cell = weights_state("gguf", THOR)  # real-world format, deliberately not a row
    assert cell.state == UNAVAILABLE
    for known in sorted(WEIGHTS_AVAILABILITY):
        assert known in cell.why, known


def test_weights_state_on_an_unknown_platform_is_unavailable_naming_the_platforms():
    cell = weights_state("bf16", "jetson-nano")
    assert cell.state == UNAVAILABLE
    for plat in PLATFORMS:
        assert plat in cell.why, plat


def test_kv_state_on_an_unknown_dtype_is_unavailable_naming_the_dtypes():
    cell = kv_state("turboquant_v2", TRITON, THOR)
    assert cell.state == UNAVAILABLE
    for known in sorted(KV_AVAILABILITY):
        assert known in cell.why, known


def test_kv_state_on_an_unknown_platform_is_unavailable_naming_the_platforms():
    cell = kv_state("fp8", FLASHINFER, "bogus")
    assert cell.state == UNAVAILABLE
    for plat in PLATFORMS:
        assert plat in cell.why, plat


def test_kv_state_on_an_unmapped_backend_keeps_the_transplanted_dtype_why():
    # validate's unmapped-backend path: the why is the dtype-level string
    # (byte-identical to the cycle-2 matrix — pinned fully in test_overlays)
    cell = kv_state("fp8", "TRITON_KITCHEN_SINK", THOR)
    assert cell.state == UNAVAILABLE
    assert cell.why == "fp8 KV is served by FlashInfer on every platform, by FA4 on thor, and by the Mjolnir kernels on thor and Orin (hd 256, q_len <= 8)"


# ── 4. style lint over the whole table (plan §4: enforced in test) ──────────


def _all_cells():
    for fmt, row in WEIGHTS_AVAILABILITY.items():
        for platform, cell in row.items():
            yield (fmt, platform), cell
    for dtype, row in KV_AVAILABILITY.items():
        for backend, cells in row.items():
            for platform, cell in cells.items():
                yield (dtype, backend, platform), cell


def test_every_cell_state_is_one_of_the_three_states():
    for key, cell in _all_cells():
        assert cell.state in (NATIVE, FALLBACK, UNAVAILABLE), key


def test_every_why_is_a_complete_sentence_naming_a_mechanism():
    for key, cell in _all_cells():
        why = cell.why
        assert why == why.strip(), key  # no stray whitespace
        assert why[0].isupper(), key  # a sentence starts capitalized
        assert why.endswith("."), key  # and ends with a period
        assert len(why) >= 30, key  # long enough to name a mechanism
        assert "  " not in why, key
        assert "TODO" not in why.upper(), key


# ── 5. the probe flags are exactly the six quants.md §gaps items ────────────

# (kind, format/dtype, backend class or None, platform) — the cells whose
# on-device result can flip a quants.md §gaps item.
EXPECTED_PROBES: set[tuple[str, str, str | None, str]] = {
    # gap 2: Marlin-on-sm_87 effective perf (JIT path + iGPU)
    ("weights", "int4-w4a16", None, ORIN),
    # gap 4: which NVFP4 GEMM kernel actually wins on Thor 0.32.0
    ("weights", "nvfp4", None, THOR),
    # gap 5: Humming availability in the images — gates the MX paths
    ("weights", "mxfp6", None, THOR),
    ("weights", "mxfp6", None, ORIN),
    ("weights", "mxfp6", None, SPARK),
    ("weights", "mxfp8", None, THOR),
    ("weights", "mxfp8", None, ORIN),
    ("weights", "mxfp8", None, SPARK),
    # gap 1: NVFP4-KV decode on Ampere (flashinfer-native wrapper + uint8 pages)
    ("kv", "nvfp4", FA2, ORIN),
    ("kv", "nvfp4", FLASHINFER, ORIN),
}


def test_probe_flagged_cells_are_exactly_the_quants_gap_items():
    flagged: set[tuple[str, str, str | None, str]] = set()
    for fmt, row in WEIGHTS_AVAILABILITY.items():
        for platform, cell in row.items():
            if cell.probe:
                flagged.add(("weights", fmt, None, platform))
    for dtype, row in KV_AVAILABILITY.items():
        for backend, cells in row.items():
            for platform, cell in cells.items():
                if cell.probe:
                    flagged.add(("kv", dtype, backend, platform))
    assert flagged == EXPECTED_PROBES


def test_probe_items_are_six_and_keyed_to_their_cells():
    probes = availability.PROBES
    assert len(probes) == 6
    assert [p.number for p in probes] == [1, 2, 3, 4, 5, 6]
    # gaps 3 (GDN layers on Orin) and 6 (the CUDA guide fetch) are not
    # weights/KV cells: they stay probe items without a cell
    assert probes[2].key is None
    assert probes[5].key is None
    for probe in probes:
        assert probe.note  # the quants.md §gaps text, transcribed


# ── 7. backend_class: the config field -> class (plan §1) ───────────────────


@pytest.mark.parametrize(
    ("field", "want"),
    [
        ("MJOLNIR", MJOLNIR),
        ("FLASHINFER", FLASHINFER),
        ("FLASH_ATTN", FA2),
        ("FLASH_ATTN@4", FA4),  # the overlays attention-config FA4 variant
        ("TRITON", TRITON),
        ("TRITON_ATTN", TRITON),
        ("flashinfer", FLASHINFER),  # the field is uppercased first
        ("mjolnir", MJOLNIR),
        # unknown values pass through (uppercased) so validate's error can
        # name them — never a silent class guess
        ("TRITON_KITCHEN_SINK", "TRITON_KITCHEN_SINK"),
    ],
)
def test_backend_class_maps_the_attention_backend_field(field, want):
    assert backend_class(field) == want


@pytest.mark.parametrize(
    ("platform", "want"),
    [(THOR, MJOLNIR), (ORIN, FLASHINFER), (SPARK, FLASHINFER)],
)
def test_backend_class_with_no_field_infers_the_platform_default(platform, want):
    # no attention-backend declared: the platform's default serving class
    assert backend_class(None, platform=platform) == want
