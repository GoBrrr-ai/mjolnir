"""The opt-in W4A4 routed-experts switch (``mjolnir.moew4a4``).

``MJOLNIR_NVFP4_MOE_W4A4=1`` wraps vLLM's
``ModelOptMixedPrecisionConfig.get_quant_method`` so routed experts declared
``W4A16_NVFP4`` load as NVFP4 (W4A4) with the checkpoint's own input scales.
Pinned here, against stubs of the two vLLM modules the hook imports (no vLLM,
no torch):

* unset (or a non-truthy value) installs nothing, and a missing vLLM is a
  warning, not a raise;
* installing twice wraps once;
* a ``RoutedExperts`` layer resolving to ``W4A16_NVFP4`` gets
  ``ModelOptNvFp4FusedMoE`` built from the **NVFP4** sub-config;
* linear layers, excluded layers and every other algorithm fall through to
  the original ``get_quant_method``;
* the input scales are created NaN and the load refuses while one is still
  NaN (a checkpoint without calibrated expert input scales), and proceeds
  once the loader has filled both.
"""

from __future__ import annotations

import math
import sys
import types
from typing import ClassVar

import pytest

import mjolnir.moew4a4 as moew4a4

EXPERTS = "model.layers.0.mlp.experts"


class _Scale:
    """A per-expert input-scale parameter: just what the hook touches."""

    def __init__(self) -> None:
        self.values = [0.01, 0.01]
        self.data = self

    def fill_(self, value: float) -> None:
        self.values = [value] * len(self.values)


class RoutedExperts:
    moe_config = "moe-config"
    w13_input_scale: _Scale  # registered by create_weights
    w2_input_scale: _Scale


class Linear:
    pass


class ModelOptNvFp4FusedMoE:
    def __init__(self, quant_config, moe_config) -> None:
        self.quant_config, self.moe_config = quant_config, moe_config
        self.created: list[tuple] = []
        self.processed: list[object] = []

    def create_weights(self, layer, *args, **kwargs) -> None:
        layer.w13_input_scale, layer.w2_input_scale = _Scale(), _Scale()
        self.created.append((args, kwargs))

    def process_weights_after_loading(self, layer) -> None:
        self.processed.append(layer)


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


@pytest.fixture
def config_cls(monkeypatch):
    """A fresh stub of vLLM's ModelOptMixedPrecisionConfig (the hook mutates
    the class), importable from where the hook looks for it; switch on."""

    class ModelOptMixedPrecisionConfig:
        nvfp4_config = "nvfp4-config"
        w4a16_nvfp4_config = "w4a16-config"
        original_calls: ClassVar[list[tuple]] = []  # per test: the class is created per test

        def __init__(self, algos: dict[str, str], excluded: tuple[str, ...] = ()) -> None:
            self.algos, self.excluded = algos, excluded

        def is_layer_excluded(self, prefix: str) -> bool:
            return prefix in self.excluded

        def _resolve_quant_algo(self, prefix: str) -> str | None:
            return self.algos.get(prefix)

        def get_quant_method(self, layer, prefix):
            self.original_calls.append((layer, prefix))
            return "original-method"

    modelopt = _module(
        "vllm.model_executor.layers.quantization.modelopt",
        ModelOptMixedPrecisionConfig=ModelOptMixedPrecisionConfig,
        ModelOptNvFp4FusedMoE=ModelOptNvFp4FusedMoE,
    )
    stubs = [
        _module("vllm"),
        _module("vllm.model_executor"),
        _module("vllm.model_executor.layers"),
        _module("vllm.model_executor.layers.fused_moe", RoutedExperts=RoutedExperts),
        _module("vllm.model_executor.layers.quantization", modelopt=modelopt),
        modelopt,
        # _require_input_scales only asks torch whether a scale holds a NaN
        _module("torch", isnan=lambda t: types.SimpleNamespace(any=lambda: any(math.isnan(v) for v in t.values))),
    ]
    for mod in stubs:
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setenv(moew4a4.ENV_MOE_W4A4, "1")
    return ModelOptMixedPrecisionConfig


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), (" YES ", True), (None, False), ("", False), ("0", False), ("false", False)],
)
def test_wanted_reads_the_env_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(moew4a4.ENV_MOE_W4A4, raising=False)
    else:
        monkeypatch.setenv(moew4a4.ENV_MOE_W4A4, value)
    assert moew4a4.wanted() is expected


def test_noop_when_the_switch_is_unset(config_cls, monkeypatch):
    monkeypatch.delenv(moew4a4.ENV_MOE_W4A4)
    original = config_cls.get_quant_method

    moew4a4.install_hooks()

    assert config_cls.get_quant_method is original
    assert not hasattr(config_cls, "_mjolnir_moe_w4a4_hooked")


def test_missing_vllm_does_not_raise(monkeypatch):
    """Opted in on a host without vLLM: a logged warning, never an exception
    out of the plugin entry point."""
    monkeypatch.setenv(moew4a4.ENV_MOE_W4A4, "1")
    monkeypatch.setitem(sys.modules, "vllm", None)  # `import vllm...` raises ImportError

    moew4a4.install_hooks()


def test_install_is_idempotent(config_cls):
    moew4a4.install_hooks()
    wrapped = config_cls.get_quant_method
    moew4a4.install_hooks()

    assert config_cls.get_quant_method is wrapped  # not wrapped a second time
    layer = Linear()
    assert config_cls({}).get_quant_method(layer, "model.lm_head") == "original-method"
    assert config_cls.original_calls == [(layer, "model.lm_head")]


def test_w4a16_routed_experts_get_the_nvfp4_method(config_cls):
    moew4a4.install_hooks()
    layer = RoutedExperts()

    method = config_cls({EXPERTS: "W4A16_NVFP4"}).get_quant_method(layer, EXPERTS)

    assert isinstance(method, ModelOptNvFp4FusedMoE)
    assert method.quant_config == "nvfp4-config"  # the W4A4 sub-config, not w4a16_nvfp4_config
    assert method.moe_config == "moe-config"
    assert config_cls.original_calls == []


@pytest.mark.parametrize(
    ("layer_cls", "algo", "excluded"),
    [
        (Linear, "W4A16_NVFP4", False),  # linear layers (shared experts, attention, lm_head) keep their algo
        (RoutedExperts, "W4A16_NVFP4", True),  # excluded routed experts stay excluded
        (RoutedExperts, "NVFP4", False),  # already W4A4
        (RoutedExperts, "FP8", False),
        (RoutedExperts, None, False),  # not in quantized_layers
    ],
)
def test_everything_else_falls_through_to_the_original(config_cls, layer_cls, algo, excluded):
    moew4a4.install_hooks()
    layer = layer_cls()
    config = config_cls({EXPERTS: algo} if algo else {}, excluded=(EXPERTS,) if excluded else ())

    assert config.get_quant_method(layer, EXPERTS) == "original-method"
    assert config_cls.original_calls == [(layer, EXPERTS)]


def _w4a4_method(config_cls):
    """The hooked method for a W4A16 routed-experts layer, weights created."""
    moew4a4.install_hooks()
    layer = RoutedExperts()
    method = config_cls({EXPERTS: "W4A16_NVFP4"}).get_quant_method(layer, EXPERTS)
    method.create_weights(layer, 256, hidden_size=2048)
    return layer, method


def test_input_scales_are_created_nan(config_cls):
    layer, method = _w4a4_method(config_cls)

    assert method.created == [((256,), {"hidden_size": 2048})]  # the arguments reach the real create_weights
    for scale in (layer.w13_input_scale, layer.w2_input_scale):
        assert all(math.isnan(v) for v in scale.values)


@pytest.mark.parametrize("missing", ["w13_input_scale", "w2_input_scale"])
def test_load_refuses_while_an_input_scale_is_nan(config_cls, missing):
    """The checkpoint loader overwrites the NaN fill; a scale it never wrote
    (no calibrated expert input scales in the checkpoint) stops the load."""
    layer, method = _w4a4_method(config_cls)
    for name in ("w13_input_scale", "w2_input_scale"):
        if name != missing:
            getattr(layer, name).fill_(0.006)

    with pytest.raises(RuntimeError, match=rf"{moew4a4.ENV_MOE_W4A4}=1.*{EXPERTS}.*{missing}"):
        method.process_weights_after_loading(layer)
    assert method.processed == []  # the real conversion never ran


def test_load_proceeds_once_both_input_scales_are_filled(config_cls):
    layer, method = _w4a4_method(config_cls)
    layer.w13_input_scale.fill_(0.006)
    layer.w2_input_scale.fill_(0.011)

    method.process_weights_after_loading(layer)

    assert method.processed == [layer]
