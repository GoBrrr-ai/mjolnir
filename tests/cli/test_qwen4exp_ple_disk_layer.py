"""The disk PLE layer's vLLM 0.34 integration (``mjolnir.qwen4exp.ple_disk``),
against stand-ins for torch and vLLM.

* the weight loader keeps only where each table shard's loader view lives
  (file and byte offset from ``/proc/self/maps``), loads the small tensors
  through the base class and refuses a shard whose dtype differs from the
  table's;
* ``bind_file_shards`` (vLLM 0.34 calls it after loading) opens the slot-pool
  lookup over exactly those shards and allocates one GPU pool per table
  (packed rows, block scales) with ``MJOLNIR_PLE_DISK_CACHE_GB``'s slot count;
  the opened tables read the same rows as the checkpoint;
* more than one PLE table rank refuses.

The GPU half (pools, kernels) runs only on the device; the host half is
covered by ``test_qwen4exp_ple_disk_{table,cache}.py``.
"""

from __future__ import annotations

import sys
import tempfile

import numpy as np
import pytest
import qwen4exp_ple_fixtures as fx
from qwen4exp_stubs import _AnyModule, fresh, stand_ins

import mjolnir.qwen4exp.ple_disk_cache as pdc
import mjolnir.qwen4exp.ple_disk_table as pdt

EXTRA_MODULES = (
    "vllm.compilation",
    "vllm.compilation.breakable_cudagraph",
    "vllm.utils",
    "vllm.utils.torch_utils",
    "vllm.models.qwen4_exp.common",
    "vllm.models.qwen4_exp.common.ngram_embedding",
)
ROW_BYTES = {"weight": fx.EMBED_DIM // 2, "weight_scale": fx.EMBED_DIM // 16}
MAX_TOKENS = 8


class FakeParam:
    def __init__(self, width: int, dtype: str) -> None:
        self.shape = (0, width)
        self.dtype = dtype

    def element_size(self) -> int:
        return 1


class FakeView:
    """A loader view: a NumPy view of the mapped checkpoint file."""

    def __init__(self, array: np.ndarray, dtype: str) -> None:
        self.array, self.dtype, self.shape = array, dtype, array.shape

    def data_ptr(self) -> int:
        return self.array.ctypes.data


class FakeNvFp4Method:
    pass


class FakeEmbedding:
    """Stands in for vLLM's ``Qwen4ExpPLEEmbedding`` base (ETP=TP=1)."""

    tp_size = 1

    def __init__(self, num_embeddings, embedding_dim, *, prefix, **kwargs) -> None:
        self.prefix = prefix
        self.org_vocab_size = num_embeddings
        self.etp_data_parallel_size = 1
        self.shard_indices = type("Indices", (), {"org_vocab_start_index": 0, "org_vocab_end_index": num_embeddings})
        self.weight = FakeParam(embedding_dim // 2, "uint8")
        self.weight_scale = FakeParam(embedding_dim // 16, "float8_e4m3fn")
        self.base_loaded: list = []

    def weight_loader(self, param, loaded_weight) -> None:
        self.base_loaded.append((param, loaded_weight))


class FakePool:
    def __init__(self, *shape) -> None:
        self.shape = shape
        self.zeroed: list[int] = []

    def __getitem__(self, index):
        pool = self

        class Row:
            def zero_(self) -> None:
                pool.zeroed.append(index)

        return Row()


@pytest.fixture
def stubs(monkeypatch):
    yield from stand_ins(monkeypatch)
    sys.modules.pop("mjolnir.qwen4exp.ple_disk", None)


@pytest.fixture
def ple(stubs, monkeypatch):
    for name in EXTRA_MODULES:
        stubs[name] = _AnyModule(name)
        monkeypatch.setitem(sys.modules, name, stubs[name])
    common = stubs["vllm.models.qwen4_exp.common.ngram_embedding"]
    common.Qwen4ExpPLEEmbedding = FakeEmbedding
    common.Qwen4ExpPLENvFp4EmbeddingMethod = FakeNvFp4Method
    stubs["torch"].empty = lambda *shape, **kwargs: FakePool(*shape)
    monkeypatch.setenv("MJOLNIR_PLE_DISK_CACHE_GB", "0.001")
    return fresh("mjolnir.qwen4exp.ple_disk")


def make_layer(ple):
    return ple.Qwen4ExpPLEDiskEmbedding(
        fx.NUM_ROWS,
        fx.EMBED_DIM,
        params_dtype="bfloat16",
        padding_size=64,
        prefix=fx.PREFIX,
        embedding_method=FakeNvFp4Method(),
        num_ngram_heads=16,
        max_total_tokens=MAX_TOKENS,
    )


def load_shards(layer, directory):
    """Hand the layer its shards like ``Qwen4ExpNGramEmbedding.load_weights``."""
    shard_size = (fx.NUM_ROWS + fx.SPLIT_PARTS - 1) // fx.SPLIT_PARTS
    prefix = fx.PREFIX + ".shard_"
    for name, view, _, _ in fx.loader_views(directory):
        if not name.startswith(prefix):
            continue
        index, suffix = name[len(prefix) :].split(".")
        param = getattr(layer, suffix)
        layer.weight_loader(param, FakeView(view, param.dtype), checkpoint_start=int(index) * shard_size)


def test_loader_records_shards_and_bind_opens_the_cache(ple):
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as directory:
        weight, scale = fx.build_checkpoint(directory, rng)
        layer = make_layer(ple)
        assert layer.lookup is None and not layer._pools

        scale_2 = object()
        loaded = FakeView(np.ones(1, np.float32), "float32")
        layer.weight_loader(scale_2, loaded)
        assert layer.base_loaded == [(scale_2, loaded)]

        load_shards(layer, directory)
        assert layer._shards == fx.loader_shards(directory, pdt.file_shard)
        assert not layer.base_loaded[1:], "table shards must not be copied"

        layer.bind_file_shards()
        slots = pdc.ple_cache_slots(sum(ROW_BYTES.values()))
        assert isinstance(layer.lookup, pdc.PLECachedLookup)
        assert layer.lookup.cache.num_slots == slots
        assert layer.lookup.max_rows == MAX_TOKENS * 16
        assert [pool.shape for pool in layer._pools] == [(slots, 80), (slots, 10)]
        assert all(pool.zeroed == [0] for pool in layer._pools)
        assert len(layer.lookup.reader.fds) == 2

        # The opened tables read the checkpoint's rows.
        ids = rng.integers(0, fx.NUM_ROWS, MAX_TOKENS * 16)
        staging = np.zeros(layer.lookup.staging_bytes, np.uint8)
        got_weight, got_scale = fx.expand(_Uncached(layer.lookup), staging, ids)
        want_weight, want_scale = fx.direct(weight, scale, ids, 0, fx.NUM_ROWS)
        assert np.array_equal(got_weight, want_weight)
        assert np.array_equal(got_scale, want_scale)


class _Uncached:
    """The uncached staging over a cached lookup's tables (the reference)."""

    def __init__(self, lookup) -> None:
        self.lookup, self.tables = lookup, lookup.tables

    def stage(self, ids, staging):
        return pdt.PLEDiskLookup.stage(self.lookup, ids, staging)


def test_shard_dtype_must_match_the_table(ple):
    layer = make_layer(ple)
    view = FakeView(np.zeros((3, 10), np.uint8), "uint8")
    with pytest.raises(ValueError, match="dtype"):
        layer.weight_loader(layer.weight_scale, view, checkpoint_start=0)


def test_one_table_rank_only(ple, monkeypatch):
    monkeypatch.setattr(FakeEmbedding, "tp_size", 2)
    with pytest.raises(RuntimeError, match="ETP=TP=1"):
        make_layer(ple)
