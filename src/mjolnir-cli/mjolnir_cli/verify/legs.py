"""Benchmark legs: what one ncu-profiled series runs, and how to read it.

A *leg* is one kernel configuration swept over the L axis:

* ``mjolnir``     — the Mjolnir kernel at the harness default (fp8 Q, e2m3 KV
                    on head-major HND pages, 2 CTAs/SM, evict-first, the
                    2-stage fp6 ring) — the kernel's best measured config;
* ``mjolnir-fp8`` — the Mjolnir kernel on fp8 KV on HND pages (the
                    `NVFP4_MJOLNIR` config; the default serves e2m3);
* ``fi``          — FlashInfer tensor-core decode as vLLM runs it (fp8 KV,
                    NHD pages) — the strongest stock baseline on Thor;
* ``fi-hnd``      — FlashInfer on the same data in HND pages (fair layout check);
* ``fa2``         — FlashAttention 2 (bf16 KV — FA2 has no fp8 KV path, so 2x
                    the bytes); on demand via ``--legs fa2`` (research
                    platforms keep it as their FA-family baseline);
* ``fa4``         — FlashAttention 4 hd256 as the patched image serves it
                    (fp8 Q+KV, NHD paged-128, 1CTA decode carve-out, no
                    SplitKV) — the default FA-family baseline; runs on the
                    Mjolnir image, not the stock one;
* T5 variants     — the Mjolnir harness with extra flags (ab / ring / ...).

Every leg profiles only its own kernels (``filter``) and splits them into
the main decode kernel(s) and the split-KV merge (``main`` / ``merge``
substrings of the ncu kernel name).
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

# Bits per K/V element of each KV format (nvfp4 adds one e4m3 scale per 16).
KV_BITS = {"fp8": 8.0, "e3m2": 6.0, "e2m3": 6.0, "e2m1": 4.0, "nvfp4": 4.5, "bf16": 16.0}

# The harness default the `mjolnir` leg runs (kept in sync with
# mjolnir.t5_decode: HARNESS_KV_FORMAT_FP8Q / HARNESS_KV_LAYOUT).
HARNESS_KV_FORMAT = "e2m3"
HARNESS_KV_LAYOUT = "hnd"

FILTERS = {
    "fi": "regex:BatchPrefillWithPagedKVCache|BatchDecode|MergeStates",
    "fa2": "regex:flash_fwd",
    "fa4": "regex:sm100_hd256_2cta_fmha_forward",
    "t5": "regex:MixedInputFusedMultiHeadAttentionDecode|reduction",
}
MAIN = {
    "fi": ("BatchPrefillWithPagedKVCache", "BatchDecode"),
    "fa2": ("flash_fwd_kernel", "flash_fwd_splitkv_kernel"),
    "fa4": ("sm100_hd256_2cta_fmha_forward",),
    "t5": ("MixedInputFusedMultiHeadAttentionDecode",),
}
MERGE = {
    "fi": ("MergeStates",),
    "fa2": ("combine",),
    "fa4": (),  # hd256: no SplitKV (vLLM pins num_splits=1), no merge kernel
    "t5": ("reduction",),
}


@dataclass(frozen=True)
class Leg:
    id: str  # stable series id (file names, chart colors, README lookups)
    label: str  # human label
    kind: str  # "fi" | "fa2" | "fa4" | "t5"
    role: str = "variant"  # "ours" | "baseline" | "variant"
    flags: str = ""  # harness flags (t5) — appended after the common args
    env: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # driver env (fi / fa2)
    q: str = "fp8"  # t5 Q dtype: fp8 | bf16

    @property
    def kv_format(self) -> str:
        if self.kind == "fa2":
            return "bf16"
        if self.kind in ("fi", "fa4"):
            return "fp8"
        fmt = _last_flag(self.flags, "--kv_format")
        if fmt in (None, "auto"):
            return HARNESS_KV_FORMAT if self.q == "fp8" else "fp8"
        return fmt

    @property
    def kv_layout(self) -> str:
        if self.kind == "fi":
            return dict(self.env).get("FI_LAYOUT", "NHD").lower()
        if self.kind in ("fa2", "fa4"):
            return "nhd"
        return _last_flag(self.flags, "--kv_layout") or HARNESS_KV_LAYOUT

    @property
    def kv_bits(self) -> float:
        return KV_BITS.get(self.kv_format, 8.0)

    def describe(self) -> str:
        return f"{self.kv_format} KV · {self.kv_layout.upper()}" + (f" · {self.flags}" if self.flags.strip() else "")


def _last_flag(flags: str, name: str) -> str | None:
    """Value of the last ``--name value`` in a flag string (argparse: last wins)."""
    out = None
    toks = shlex.split(flags or "")
    for i, t in enumerate(toks):
        if t == name and i + 1 < len(toks):
            out = toks[i + 1]
        elif t.startswith(name + "="):
            out = t.split("=", 1)[1]
    return out


# ── the canonical legs ──────────────────────────────────────────────────────
MJOLNIR = Leg("mjolnir", "Mjolnir (e2m3 KV, HND)", "t5", role="ours")
MJOLNIR_FP8 = Leg("mjolnir-fp8", "Mjolnir (fp8 KV, HND)", "t5", role="ours", flags="--kv_format fp8 --kv_layout hnd")
FI = Leg("fi", "FlashInfer (fp8 KV, NHD)", "fi", role="baseline", env=(("FI_LAYOUT", "NHD"),))
FI_HND = Leg("fi-hnd", "FlashInfer (fp8 KV, HND)", "fi", role="baseline", env=(("FI_LAYOUT", "HND"),))
FA2 = Leg("fa2", "FlashAttention 2 (bf16 KV)", "fa2", role="baseline")
FA4 = Leg("fa4", "FlashAttention 4 hd256 (fp8 Q+KV)", "fa4", role="baseline")

CANONICAL = {leg.id: leg for leg in (MJOLNIR, MJOLNIR_FP8, FI, FI_HND, FA2, FA4)}


def t5_variant(name: str, flags: str, q: str = "fp8") -> Leg:
    return Leg(f"t5:{name}", name, "t5", role="variant", flags=flags, q=q)


def parse_variants(spec: str) -> list[Leg]:
    """``"name:flags|name:flags"`` (the verify.sh AB_VARIANTS syntax) → legs."""
    out = []
    for part in (spec or "").split("|"):
        part = part.strip()
        if not part:
            continue
        name, _, flags = part.partition(":")
        out.append(t5_variant(name.strip(), flags.strip()))
    return out


def legs_from_ids(ids: str | list[str]) -> list[Leg]:
    if isinstance(ids, str):
        ids = [x for x in re.split(r"[,\s]+", ids) if x]
    unknown = [i for i in ids if i not in CANONICAL]
    if unknown:
        raise ValueError(f"unknown leg(s) {unknown}; known: {', '.join(CANONICAL)}")
    return [CANONICAL[i] for i in ids]
