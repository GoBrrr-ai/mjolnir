"""Config platform overlays (``mjolnir_cli.overlays``, new in cycle 2).

The spec (plan §0.3 / §1), pinned as tests:

* layout: ``<quant>/platforms/<platform>.yaml`` — a config dir with no
  ``platforms/`` is a legacy thor-only config; with the dir, only the
  listed overlays exist (``overlay_platforms``);
* ``load_overlay`` — the parsed overlay dict, ``None`` when the platform
  has none, a clear ``ValueError`` naming the file when the yaml is
  malformed;
* ``merge_config`` — deep merge over the base config: nested dicts merge
  recursively, lists replace, scalars override;
* ``validate`` — the ``kv-cache-dtype`` x ``attention-backend`` availability
  matrix (task/research/spark-stack.md): fp6 (e2m3) KV is Mjolnir-on-thor
  only; fp8 KV goes via FlashInfer anywhere and FA4/Mjolnir on thor; nvfp4
  KV goes via the FA2 path on orin/spark but never on sm_110; an unmapped
  backend is an error. Every error names the platform, the key and the
  value (and why), so a bad overlay is fixable from the message;
* ``config_exists_on`` — the leg-availability rule: no config.yaml, no
  config; legacy dirs are thor-only; explicit overlays list their platforms.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mjolnir_cli import overlays
from mjolnir_cli.config import configs_root


def _config_dir(tmp_path: Path, overlays_: tuple[str, ...] = (), body: str = "gpu-memory-utilization: 0.8\n") -> Path:
    d = tmp_path / "NVFP4"
    d.mkdir()
    (d / "config.yaml").write_text("model: Qwen/Qwen3.8-27B\n")
    for p in overlays_:
        (d / "platforms").mkdir(exist_ok=True)
        (d / "platforms" / f"{p}.yaml").write_text(body)
    return d


# ── layout ──────────────────────────────────────────────────────────────────


def test_overlay_platforms_empty_when_there_is_no_dir(tmp_path):
    assert overlays.overlay_platforms(_config_dir(tmp_path)) == ()


def test_overlay_platforms_lists_the_existing_overlays(tmp_path):
    d = _config_dir(tmp_path, ("spark", "thor"))
    assert overlays.overlay_platforms(d) == ("spark", "thor")


# ── load + merge ────────────────────────────────────────────────────────────


def test_load_overlay_is_none_when_the_platform_has_none(tmp_path):
    assert overlays.load_overlay(_config_dir(tmp_path, ("thor",)), "spark") is None


def test_load_overlay_parses_the_yaml(tmp_path):
    d = _config_dir(tmp_path, ("spark",), body="gpu-memory-utilization: 0.6\nkv-cache-dtype: fp8\n")
    assert overlays.load_overlay(d, "spark") == {"gpu-memory-utilization": 0.6, "kv-cache-dtype": "fp8"}


def test_load_overlay_on_a_malformed_yaml_names_the_file(tmp_path):
    d = _config_dir(tmp_path, ("spark",), body=":::: not yaml ::::")
    with pytest.raises(ValueError) as ei:
        overlays.load_overlay(d, "spark")
    assert "spark.yaml" in str(ei.value)


def test_merge_config_deep_merges_dicts_replaces_lists_and_scalars():
    base = {
        "model": "Qwen/Qwen3.8-27B",
        "gpu-memory-utilization": 0.8,
        "attention-config": {"backend": "flash_attn", "flash_attn_version": 4, "window_size": 512},
        "kv-cache-dtype": "auto",
        "speculative-config": {"method": "mtp", "tokens": 4},
    }
    overlay = {
        "gpu-memory-utilization": 0.6,
        "attention-config": {"backend": "flashinfer", "window_size": None},
        "speculative-config": {"tokens": 7},  # the dict merges, the key not mentioned survives
    }
    merged = overlays.merge_config(base, overlay)
    assert merged["model"] == "Qwen/Qwen3.8-27B"  # untouched base keys survive
    assert merged["gpu-memory-utilization"] == 0.6  # scalars override
    assert merged["kv-cache-dtype"] == "auto"
    assert merged["attention-config"] == {"backend": "flashinfer", "flash_attn_version": 4, "window_size": None}
    assert merged["speculative-config"] == {"method": "mtp", "tokens": 7}


def test_merge_config_lists_replace_wholesale():
    base = {"env": ["A=1", "B=2"], "flags": ["--x"]}
    assert overlays.merge_config(base, {"env": ["C=3"]})["env"] == ["C=3"]  # not ["A=1", "B=2", "C=3"]
    assert overlays.merge_config(base, {"flags": []})["flags"] == []


def test_merge_config_does_not_mutate_its_inputs():
    base = {"a": {"b": 1}}
    overlay = {"a": {"c": 2}}
    merged = overlays.merge_config(base, overlay)
    assert merged == {"a": {"b": 1, "c": 2}}
    assert base == {"a": {"b": 1}}  # the base dict is untouched
    assert overlay == {"a": {"c": 2}}


# ── the availability matrix ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("platform", "merged"),
    [
        # fp6 (e2m3) KV: the Mjolnir kernels on thor and orin (the Orin's
        # prefill rides the FlashInfer fallback on an e4m3 scratch)
        ("thor", {"kv-cache-dtype": "fp6_e2m3", "attention-backend": "MJOLNIR"}),
        ("orin", {"kv-cache-dtype": "fp6_e2m3", "attention-backend": "MJOLNIR"}),
        # fp8 KV: FlashInfer serves it on every platform; FA4 and Mjolnir are thor-only
        ("thor", {"kv-cache-dtype": "fp8", "attention-backend": "FLASHINFER"}),
        ("spark", {"kv-cache-dtype": "fp8", "attention-backend": "FLASHINFER"}),
        ("orin", {"kv-cache-dtype": "fp8", "attention-backend": "FLASHINFER"}),
        ("thor", {"kv-cache-dtype": "fp8", "attention-backend": "MJOLNIR"}),
        ("thor", {"kv-cache-dtype": "fp8", "attention-config": {"backend": "flash_attn", "flash_attn_version": 4}}),
        # nvfp4 KV: the FA2 path covers orin (family 80) and spark (family 120)
        ("spark", {"kv-cache-dtype": "nvfp4", "attention-backend": "FLASH_ATTN"}),
        ("orin", {"kv-cache-dtype": "nvfp4", "attention-backend": "FLASH_ATTN"}),
        # no kv-cache-dtype: nothing to contradict
        ("spark", {"attention-backend": "FLASH_ATTN"}),
    ],
)
def test_validate_accepts_the_available_combinations(platform, merged):
    overlays.validate(platform, merged)  # does not raise


@pytest.mark.parametrize(
    ("platform", "merged", "named"),
    [
        # fp6 needs the Mjolnir kernel reader: thor + orin + MJOLNIR only
        (
            "spark",
            {"kv-cache-dtype": "fp6_e2m3", "attention-backend": "MJOLNIR"},
            ("spark", "kv-cache-dtype", "fp6_e2m3"),
        ),
        (
            "thor",
            {"kv-cache-dtype": "fp6_e2m3", "attention-backend": "FLASHINFER"},
            ("thor", "kv-cache-dtype", "fp6_e2m3"),
        ),
        # fp8 via FA2 is not a supported path on any platform
        ("orin", {"kv-cache-dtype": "fp8", "attention-backend": "FLASH_ATTN"}, ("orin", "kv-cache-dtype", "fp8")),
        ("thor", {"kv-cache-dtype": "fp8", "attention-backend": "FLASH_ATTN"}, ("thor", "kv-cache-dtype", "fp8")),
        # nvfp4 KV never happens on sm_110 (fa2-nvfp4 families are 80/90/120)
        ("thor", {"kv-cache-dtype": "nvfp4", "attention-backend": "FLASH_ATTN"}, ("thor", "kv-cache-dtype", "nvfp4")),
        ("thor", {"kv-cache-dtype": "nvfp4", "attention-backend": "FLASHINFER"}, ("thor", "kv-cache-dtype", "nvfp4")),
        # FA4 is sm_100-only upstream: not on orin
        (
            "orin",
            {"kv-cache-dtype": "fp8", "attention-config": {"backend": "flash_attn", "flash_attn_version": 4}},
            ("orin", "kv-cache-dtype", "fp8"),
        ),
    ],
)
def test_validate_rejects_unavailable_combinations_naming_platform_key_and_value(platform, merged, named):
    with pytest.raises(ValueError) as ei:
        overlays.validate(platform, merged)
    for fragment in named:
        assert fragment in str(ei.value), fragment


def test_validate_error_for_an_unmapped_backend_names_the_backend_value():
    with pytest.raises(ValueError) as ei:
        overlays.validate("thor", {"kv-cache-dtype": "fp8", "attention-backend": "TRITON_KITCHEN_SINK"})
    assert "TRITON_KITCHEN_SINK" in str(ei.value)
    assert "thor" in str(ei.value) and "kv-cache-dtype" in str(ei.value)


# ── cycle 3a: the refactor contract ─────────────────────────────────────────
# validate is about to be rewritten over the KV_AVAILABILITY table
# (mjolnir_cli.availability); these tests are the pin that the user-facing
# half does not move. The byte-identical pin is a regression guard like
# ``test_thor_legs_are_unchanged``: green against today's data by design.


@pytest.mark.parametrize(
    ("platform", "merged", "message"),
    [
        (
            "spark",
            {"kv-cache-dtype": "fp6_e2m3", "attention-backend": "MJOLNIR"},
            "spark: kv-cache-dtype 'fp6_e2m3' cannot be served with attention-backend 'MJOLNIR'"
            " — fp6 (e2m3) KV is read by the Mjolnir kernels on thor and Orin (the Orin's prefill rides"
            " the FlashInfer fallback on an e4m3 scratch)",
        ),
        (
            "thor",
            {"kv-cache-dtype": "fp6_e2m3", "attention-backend": "FLASHINFER"},
            "thor: kv-cache-dtype 'fp6_e2m3' cannot be served with attention-backend 'FLASHINFER'"
            " — fp6 (e2m3) KV is read by the Mjolnir kernels on thor and Orin (the Orin's prefill rides"
            " the FlashInfer fallback on an e4m3 scratch)",
        ),
        (
            "orin",
            {"kv-cache-dtype": "fp8", "attention-backend": "FLASH_ATTN"},
            "orin: kv-cache-dtype 'fp8' cannot be served with attention-backend 'FLASH_ATTN'"
            " — fp8 KV is served by FlashInfer on every platform, by FA4 on thor, and by the Mjolnir kernels on thor and Orin (hd 256, q_len <= 8)",
        ),
        (
            "thor",
            {"kv-cache-dtype": "fp8", "attention-backend": "FLASH_ATTN"},
            "thor: kv-cache-dtype 'fp8' cannot be served with attention-backend 'FLASH_ATTN'"
            " — fp8 KV is served by FlashInfer on every platform, by FA4 on thor, and by the Mjolnir kernels on thor and Orin (hd 256, q_len <= 8)",
        ),
        (
            "spark",
            {"kv-cache-dtype": "fp8", "attention-backend": "FLASH_ATTN"},
            "spark: kv-cache-dtype 'fp8' cannot be served with attention-backend 'FLASH_ATTN'"
            " — fp8 KV is served by FlashInfer on every platform, by FA4 on thor, and by the Mjolnir kernels on thor and Orin (hd 256, q_len <= 8)",
        ),
        (
            "orin",
            {"kv-cache-dtype": "fp8", "attention-config": {"backend": "flash_attn", "flash_attn_version": 4}},
            "orin: kv-cache-dtype 'fp8' cannot be served with attention-backend 'FLASH_ATTN@4'"
            " — fp8 KV is served by FlashInfer on every platform, by FA4 on thor, and by the Mjolnir kernels on thor and Orin (hd 256, q_len <= 8)",
        ),
        (
            "thor",
            {"kv-cache-dtype": "nvfp4", "attention-backend": "FLASH_ATTN"},
            "thor: kv-cache-dtype 'nvfp4' cannot be served with attention-backend 'FLASH_ATTN'"
            " — nvfp4 KV rides the FA2 path (fa2-nvfp4 families are sm_80/90/120, never sm_110)"
            " and has no FlashInfer path",
        ),
        (
            "thor",
            {"kv-cache-dtype": "nvfp4", "attention-backend": "FLASHINFER"},
            "thor: kv-cache-dtype 'nvfp4' cannot be served with attention-backend 'FLASHINFER'"
            " — nvfp4 KV rides the FA2 path (fa2-nvfp4 families are sm_80/90/120, never sm_110)"
            " and has no FlashInfer path",
        ),
        (
            "thor",
            {"kv-cache-dtype": "fp8", "attention-backend": "TRITON_KITCHEN_SINK"},
            "thor: kv-cache-dtype 'fp8' cannot be served with attention-backend 'TRITON_KITCHEN_SINK'"
            " — fp8 KV is served by FlashInfer on every platform, by FA4 on thor, and by the Mjolnir kernels on thor and Orin (hd 256, q_len <= 8)",
        ),
    ],
)
def test_validate_error_messages_stay_byte_identical_across_the_availability_refactor(platform, merged, message):
    """PINNED CONTRACT (cycle 3a): ``validate`` messages are byte-identical
    before and after the KV_AVAILABILITY refactor (feature.md AC4). Regression
    guard — green against today's validate by design; any wording, quoting or
    ordering change to a user-facing message fails here."""
    with pytest.raises(ValueError) as ei:
        overlays.validate(platform, merged)
    assert str(ei.value) == message


def test_validate_routes_through_the_availability_table(monkeypatch):
    """The refactor's point: validate consults ``kv_state`` — a sentinel cell
    it returns must drive both the decision and the message's why. Duck-typed
    cell on purpose: the routing contract is what's pinned here, not the
    availability types."""
    from types import SimpleNamespace

    seen: list[tuple[str, str, str]] = []

    def fake_kv_state(dtype: str, backend: str, platform: str) -> SimpleNamespace:
        seen.append((dtype, backend, platform))
        return SimpleNamespace(state="unavailable", why="sentinel why routed from the availability table.")

    monkeypatch.setattr(overlays, "kv_state", fake_kv_state, raising=False)

    # a combination validate accepts today must now be refused, with the
    # sentinel's why in the message — proving the cell came from kv_state
    with pytest.raises(ValueError) as ei:
        overlays.validate("thor", {"kv-cache-dtype": "fp8", "attention-backend": "FLASHINFER"})
    assert seen == [("fp8", "FLASHINFER", "thor")]
    assert "sentinel why routed from the availability table." in str(ei.value)


def test_validate_accepts_when_the_availability_table_says_native(monkeypatch):
    """The routing contract's other half: a native sentinel cell keeps validate
    silent — the accept decision also comes from the table."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        overlays,
        "kv_state",
        lambda dtype, backend, platform: SimpleNamespace(state="native", why="sentinel native."),
        raising=False,
    )
    overlays.validate("spark", {"kv-cache-dtype": "fp6_e2m3", "attention-backend": "MJOLNIR"})  # refused today


# ── fix round 1: an undeclared backend has nothing to contradict ────────────
# (review cycle 2, IMPROVEMENT 2: with no attention-backend declared, vLLM
# auto-selects a servable one — the matrix validates declared pairs, so it
# must not claim a dtype "is not available on this platform" when it is)


@pytest.mark.parametrize(
    ("platform", "merged"),
    [
        # fp8 KV is servable on spark — the false claim the review reproduced
        ("spark", {"kv-cache-dtype": "fp8"}),
        ("orin", {"kv-cache-dtype": "fp8"}),
        # an attention-config without a backend key declares nothing either
        ("spark", {"kv-cache-dtype": "fp8", "attention-config": {"window_size": 512}}),
        # and even a dtype the platform cannot serve at all: with no declared
        # backend there is no pair to reject (an undeclared backend is like a
        # dtype the matrix does not know — nothing to contradict)
        ("spark", {"kv-cache-dtype": "fp6_e2m3"}),
    ],
)
def test_validate_accepts_a_config_without_a_declared_backend(platform, merged):
    overlays.validate(platform, merged)  # does not raise


def test_validate_still_rejects_a_declared_unavailable_backend_on_spark():
    with pytest.raises(ValueError) as ei:
        overlays.validate("spark", {"kv-cache-dtype": "fp8", "attention-backend": "FLASH_ATTN"})
    msg = str(ei.value)
    assert "spark" in msg and "fp8" in msg and "FLASH_ATTN" in msg


# ── availability (the legs filter reads this) ───────────────────────────────


def test_config_exists_on_legacy_configs_are_thor_only(tmp_path):
    d = _config_dir(tmp_path)  # no platforms/ dir: the pre-overlay layout
    assert overlays.config_exists_on(d, "thor")
    assert not overlays.config_exists_on(d, "spark")
    assert not overlays.config_exists_on(d, "orin")


def test_config_exists_on_explicit_overlays_list_their_platforms(tmp_path):
    d = _config_dir(tmp_path, ("spark",))
    assert overlays.config_exists_on(d, "spark")
    assert not overlays.config_exists_on(d, "thor")  # the dir present means only the listed ones
    assert not overlays.config_exists_on(d, "orin")


def test_config_exists_on_is_false_without_a_config_yaml(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "platforms").mkdir()
    (d / "platforms" / "spark.yaml").write_text("gpu-memory-utilization: 0.6\n")
    assert not overlays.config_exists_on(d, "spark")  # an overlay without a config is not a config
    assert not overlays.config_exists_on(tmp_path / "missing", "thor")


# ── the real Orin AWQ baseline (spec §Directory Layout's example, made real) ─


def test_the_orin_awq_config_exists_only_on_orin():
    """The repo's Orin AWQ serving config: a ``platforms/`` dir carrying only
    ``orin.yaml`` — the overlay-only rule makes thor and spark em-dashes
    (the legs filter reads exactly this)."""
    d = configs_root() / "Qwen" / "Qwen3.8-27B" / "AWQ_FLASHINFER"
    assert overlays.overlay_platforms(d) == ("orin",)
    assert not overlays.config_exists_on(d, "thor")
    assert not overlays.config_exists_on(d, "spark")
    assert overlays.config_exists_on(d, "orin")


def test_the_orin_awq_config_validates_on_orin_with_native_rows():
    """The merged config (base + the orin overlay) passes the matrix: fp8 KV
    via FlashInfer is a native orin row, and the AWQ weights are int4-w4a16 —
    native on orin too (Marlin from compute capability 7.5 up)."""
    from mjolnir_cli.availability import FLASHINFER, NATIVE, kv_state, weights_state

    d = configs_root() / "Qwen" / "Qwen3.8-27B" / "AWQ_FLASHINFER"
    base = yaml.safe_load((d / "config.yaml").read_text())
    merged = overlays.merge_config(base, overlays.load_overlay(d, "orin") or {})
    overlays.validate("orin", merged)  # does not raise
    assert kv_state("fp8", FLASHINFER, "orin").state == NATIVE
    assert weights_state("int4-w4a16", "orin").state == NATIVE
