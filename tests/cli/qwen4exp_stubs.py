"""Import the ``mjolnir.qwen4exp`` modules on a host without torch / vLLM.

The modules run inside the vLLM image; their tests here exercise the plain
Python around vLLM's classes (env switches, target regexes, vocabulary files,
config composition). ``install`` registers permissive stand-ins for torch,
triton and every vLLM module they import — any attribute is a dummy class
(subclassable, callable, usable as a decorator) unless a test sets a real
one — and ``fresh`` imports a module against them.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

QWEN4EXP = ("draft_head", "side_fp8", "hc_fp8", "screened_head")
VLLM_MODULES = (
    "vllm",
    "vllm.envs",
    "vllm.config",
    "vllm.config.quantization",
    "vllm.logger",
    "vllm.triton_utils",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.quantization",
    "vllm.model_executor.layers.quantization.online",
    "vllm.model_executor.layers.quantization.online.base",
    "vllm.model_executor.layers.quantization.online.fp8",
    "vllm.model_executor.layers.quantization.online.mxfp8",
    "vllm.model_executor.layers.vocab_parallel_embedding",
    "vllm.model_executor.models",
    "vllm.model_executor.models.utils",
    "vllm.models",
    "vllm.models.qwen4_exp",
    "vllm.models.qwen4_exp.nvidia",
    "vllm.models.qwen4_exp.nvidia.qsa",
    "vllm.v1",
    "vllm.v1.sample",
    "vllm.v1.sample.sampler",
    "vllm.v1.worker",
    "vllm.v1.worker.gpu",
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu.sample",
    "vllm.v1.worker.gpu.sample.sampler",
)


class _AnyAttr(type):
    """Class-level attribute access on a stand-in class (``triton.jit``)."""

    def __getattr__(cls, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy()


class _Dummy(metaclass=_AnyAttr):
    """Stands in for any class, function, decorator or constant."""

    def __init__(self, *args, **kwargs) -> None:
        self.args, self.kwargs = args, kwargs

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]  # used as a decorator
        return _Dummy(*args, **kwargs)

    def __getattr__(self, name: str):
        return _Dummy()

    def __getitem__(self, key):
        return _Dummy()

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self


class _AnyModule(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        value = type(name, (_Dummy,), {})
        setattr(self, name, value)
        return value


def install(monkeypatch: pytest.MonkeyPatch, *, torch: bool = True) -> dict[str, types.ModuleType]:
    """Register the stand-ins (and drop cached qwen4exp imports)."""
    names = [*VLLM_MODULES, "triton", "triton.language"] + (["torch", "torch.nn"] if torch else [])
    mods = {}
    for name in names:
        mod = _AnyModule(name)
        mods[name] = mod
        monkeypatch.setitem(sys.modules, name, mod)
    for name, mod in mods.items():  # `import a.b as c` binds through the parent
        parent, _, leaf = name.rpartition(".")
        if parent in mods:
            mods[parent].__dict__[leaf] = mod
    mods["vllm.logger"].__dict__["init_logger"] = lambda name: _Logger()
    for leaf in (*QWEN4EXP, "lmhead"):
        monkeypatch.delitem(sys.modules, f"mjolnir.qwen4exp.{leaf}", raising=False)
    monkeypatch.delitem(sys.modules, "mjolnir.lmhead", raising=False)
    return mods


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def _log(self, msg, *args) -> None:
        self.lines.append(msg % args if args else msg)

    info = warning = debug = warning_once = info_once = _log


def stand_ins(monkeypatch: pytest.MonkeyPatch):
    """Generator body of a test's ``stubs`` fixture: the stand-ins for one
    test; the qwen4exp / lmhead modules it imported against them are dropped
    afterwards."""
    yield install(monkeypatch)
    for leaf in QWEN4EXP:
        sys.modules.pop(f"mjolnir.qwen4exp.{leaf}", None)
    sys.modules.pop("mjolnir.lmhead", None)


def fresh(name: str):
    """Import ``name`` against the installed stand-ins."""
    sys.modules.pop(name, None)
    return importlib.import_module(name)
