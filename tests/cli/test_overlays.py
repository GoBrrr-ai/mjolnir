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

from mjolnir_cli import overlays


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
        # fp6 (e2m3) KV: Mjolnir on thor, and nothing else anywhere
        ("thor", {"kv-cache-dtype": "fp6_e2m3", "attention-backend": "MJOLNIR"}),
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
        # fp6 needs the Mjolnir kernel reader: thor + MJOLNIR only
        (
            "spark",
            {"kv-cache-dtype": "fp6_e2m3", "attention-backend": "MJOLNIR"},
            ("spark", "kv-cache-dtype", "fp6_e2m3"),
        ),
        (
            "orin",
            {"kv-cache-dtype": "fp6_e2m3", "attention-backend": "MJOLNIR"},
            ("orin", "kv-cache-dtype", "fp6_e2m3"),
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
