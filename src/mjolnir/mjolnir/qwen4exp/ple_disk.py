# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp PLE embedding served from the checkpoint files through a GPU
slot-pool cache (``disk_offload`` + ``MJOLNIR_PLE_DISK_CACHE_GB``).

Reimplements the PLE serving recipe of NemoClaw-Thor
(https://github.com/pastoriomarco/NemoClaw-Thor, MIT, Copyright Marco
Pastorio; see NOTICE).

Switch: vLLM's ``--engram-config '{"disk_offload": true}'`` keeps the table
in the checkpoint files. With ``MJOLNIR_PLE_DISK_CACHE_GB`` (or
``MJOLNIR_PLE_DISK_CACHE_SLOTS``) unset or 0, vLLM's stock reader
(``Qwen4ExpPLEDiskOffloadEmbedding``) serves it; set, the image patch
``thor-qwen4-ple-disk`` builds this layer instead. It is not a subclass of
the stock class, so neither the per-step stream sync of
``stage_checkpoint_rows`` nor the stock load branch applies to it.

The table weights stay on disk: the layer registers zero-row placeholders,
and the weight loader records each checkpoint shard's file and byte offset
from the loader's zero-copy view (``/proc/self/maps``, as the stock class
does) instead of copying it. ``bind_file_shards`` (vLLM 0.34 calls it from
the method's ``process_weights_after_loading``, inside ``load_model``) opens
the tables and allocates the slot pool. Only the small tensors
(``weight_scale_2``, hash buffers) load normally.

Slot-pool cache (``ple_disk_cache.py``): the GPU keeps a CLOCK-managed pool
of rows in their stored form. The host stages only the missed rows (read
with parallel ``pread``) plus a per-id int32 slot map; the GPU scatters the
missed rows into their slots and decodes through the slot map with the
lookup kernels of ``common/ngram_embedding.py``, so the output equals the
device and pinned-host paths. The host work cannot be captured in a CUDA
graph: the synchronous lookup is the custom op
``vllm::qwen4_exp_ple_disk_lookup`` (opaque to torch.compile) and an eager
break of the breakable CUDA graph, so it needs ``cudagraph_mode=PIECEWISE``
(or ``--enforce-eager``), not FULL.

Prefetch (default; ``MJOLNIR_PLE_DISK_PREFETCH=0`` restores the synchronous
lookup above): the model calls ``start_prefetch`` before the decoder layer
that precedes the PLE layer (``nvidia/model.py`` ``_start_layer_ple_prefetch``).
It enqueues a non-blocking D2H copy of the ids and an event, and hands the
host staging to a background thread, so the main thread goes on launching
that layer. ``finalize_prefetch`` in the PLE layer waits for the staging,
then enqueues the H2D copy, the slot scatter and the decode kernel. Both are
eager breaks.
"""

import time
from concurrent.futures import Future

import torch

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLENvFp4EmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
    _lookup_nvfp4_ple_embedding_kernel,
    _lookup_ple_embedding_from_pinned_kernel,
)
from .ple_disk_cache import open_ple_cached_lookup, ple_cache_slots
from .ple_disk_table import (
    STATS,
    STATS_ZERO,
    env_int,
    file_shard,
)

logger = init_logger("vllm." + __name__)  # under vLLM's handler, like vllm.mjolnir

_LAYERS: dict[str, "Qwen4ExpPLEDiskEmbedding"] = {}
_STATS_SEC = env_int("MJOLNIR_PLE_DISK_STATS_SEC", 30)
_PREFETCH = env_int("MJOLNIR_PLE_DISK_PREFETCH", 1) != 0
_stats_last = [0.0]


class Qwen4ExpPLEDiskEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table read from the checkpoint files through a GPU slot pool."""

    supports_prefetch = _PREFETCH

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        reduce_results: bool = True,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
            reduce_results=reduce_results,
        )
        self.num_ngram_heads = num_ngram_heads
        self.is_nvfp4 = isinstance(embedding_method, Qwen4ExpPLENvFp4EmbeddingMethod)
        if self.tp_size != 1:
            raise RuntimeError("MJOLNIR_PLE_DISK_CACHE_GB supports one PLE table rank (ETP=TP=1) only")
        # Staging order: packed rows, then (NVFP4) their block scales.
        self._table_row_bytes = {"weight": self.weight.shape[1] * self.weight.element_size()}
        if self.is_nvfp4:
            self._table_row_bytes["weight_scale"] = self.weight_scale.shape[1]
        self._max_rows = max_total_tokens * self.etp_data_parallel_size * num_ngram_heads
        # Checkpoint shards per table, {checkpoint_start: (path, offset, rows)},
        # recorded by the weight loader and opened in bind_file_shards.
        self._shards: dict[str, dict[int, tuple[str, int, int]]] = {name: {} for name in self._table_row_bytes}
        self.lookup = None
        # GPU slot pools, one per table (packed rows, block scales), slot 0 zero.
        self._pools: list[torch.Tensor] = []
        self._staging: torch.Tensor | None = None
        self._device_staging: torch.Tensor | None = None
        # Prefetch state: pinned ids, the event after their D2H copy, the
        # running staging job and its row count.
        self._prefetch_ids: torch.Tensor | None = None
        self._prefetch_event: torch.cuda.Event | None = None
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._prefetch_job = None
        self._prefetch_rows = 0
        _LAYERS[prefix] = self

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate a zero-row placeholder; the rows stay in the checkpoint."""
        del num_embeddings
        return torch.empty(0, embedding_dim, dtype=dtype, device="cpu")

    def weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        checkpoint_start: int | None = None,
    ) -> None:
        if checkpoint_start is None:
            super().weight_loader(param, loaded_weight)
            return
        # Table shards are read from disk at lookup time: keep only where the
        # loader's view of this shard lives.
        name = "weight" if param is self.weight else "weight_scale"
        if loaded_weight.dtype != param.dtype:
            raise ValueError(
                f"PLE shard {name} dtype {loaded_weight.dtype} must match the "
                f"embedding dtype {param.dtype}; the disk table serves rows as stored"
            )
        self._shards[name][checkpoint_start] = file_shard(loaded_weight.data_ptr(), loaded_weight.shape[0])

    def bind_file_shards(self) -> None:
        """Open the recorded checkpoint shards and allocate the slot pool.

        Runs inside ``load_model``, so vLLM counts the pool as model memory
        before it sizes the KV cache.
        """
        cache_slots = ple_cache_slots(sum(self._table_row_bytes.values()))
        self.lookup = open_ple_cached_lookup(
            self._shards,
            self._table_row_bytes,
            num_rows=self.org_vocab_size,
            max_rows=self._max_rows,
            vocab_start=self.shard_indices.org_vocab_start_index,
            vocab_end=self.shard_indices.org_vocab_end_index,
            num_slots=cache_slots,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        for row_bytes in self._table_row_bytes.values():
            pool = torch.empty(cache_slots, row_bytes, dtype=torch.uint8, device=device)
            pool[0].zero_()
            self._pools.append(pool)
        logger.info(
            "PLE disk: %s reads %d rows x %d B (%.1f GiB) from %d checkpoint files, "
            "%d B pinned staging, %d gather workers, prefetch %s, "
            "slot-pool cache %d slots (%.2f GiB GPU, %.2f GiB host index), %d readers",
            self.prefix,
            self.org_vocab_size,
            self.lookup.row_bytes,
            self.org_vocab_size * self.lookup.row_bytes / 2**30,
            len(self.lookup.reader.fds),
            self.lookup.staging_bytes,
            self.lookup.tables[0].pool._max_workers,
            "on" if self.supports_prefetch else "off",
            cache_slots,
            cache_slots * self.lookup.row_bytes / 2**30,
            self.org_vocab_size * 4 / 2**30,
            self.lookup.reader.readers,
        )

    # ETP reduction is identical to the pinned-host table's.
    _reduce_etp_embeddings = Qwen4ExpPLEPinnedHostEmbedding._reduce_etp_embeddings

    def _allocate_staging(self, device: torch.device) -> None:
        # First use is the eager profile run, before any graph capture.
        self._staging = torch.empty(self.lookup.staging_bytes, dtype=torch.uint8, pin_memory=True)
        self._device_staging = torch.empty(self.lookup.staging_bytes, dtype=torch.uint8, device=device)
        if self.supports_prefetch:
            self._prefetch_ids = torch.empty(self.lookup.max_rows, dtype=torch.int64, pin_memory=True)
            self._prefetch_event = torch.cuda.Event()
            # The id copy and its event live on a side stream: the prefetch
            # thread waits on that event while the main stream is already
            # capturing the next graph segment, and synchronizing on work of a
            # capturing stream invalidates the capture.
            self._prefetch_stream = torch.cuda.Stream(device=device)

    @eager_break_during_capture
    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Copy the ids to the host and stage their rows on the prefetch thread."""
        start_time = time.perf_counter()
        _refuse_capture()
        if self._staging is None:
            self._allocate_staging(ngram_ids.device)
        ids_host = self._prefetch_ids
        event = self._prefetch_event
        assert ids_host is not None and event is not None and self._staging is not None
        slot_size, _ = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        num_ids = gathered_ids.numel()
        # The previous lookup's finalize has joined its job and enqueued its
        # H2D copy before this copy and event, so both host buffers are free
        # once the event fires.
        if BreakableCUDAGraphCapture.is_active():
            # While a graph is being captured, stage on this thread: a host
            # wait from the prefetch thread during the next segment's capture
            # invalidates it. Replays and eager steps take the async path.
            job: Future = Future()
            job.set_result(self.lookup.stage(gathered_ids.reshape(-1).cpu().numpy(), self._staging.numpy()))
            self._prefetch_rows = gathered_ids.shape[0]
            self._prefetch_job = job
            STATS["op_ms"] += (time.perf_counter() - start_time) * 1e3
            return
        prefetch_stream = self._prefetch_stream
        prefetch_stream.wait_stream(torch.cuda.current_stream())
        gathered_ids.record_stream(prefetch_stream)
        with torch.cuda.stream(prefetch_stream):
            ids_host[:num_ids].copy_(gathered_ids.reshape(-1), non_blocking=True)
            event.record(prefetch_stream)
        self._prefetch_rows = gathered_ids.shape[0]
        self._prefetch_job = self.lookup.stage_async(
            ids_host[:num_ids].numpy(), self._staging.numpy(), event.synchronize
        )
        STATS["op_ms"] += (time.perf_counter() - start_time) * 1e3

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Decode the prefetched rows for this DP rank's ``total_tokens``."""
        assert self._device_staging is not None
        # Allocated in the graph segment so the eager finalize writes a static
        # buffer that later segments read.
        output = torch.empty(
            (total_tokens, self.num_ngram_heads * self.embedding_dim),
            dtype=self.embedding_method.lookup_dtype(self),
            device=self._device_staging.device,
        )
        self._finalize_prefetch(output)
        return output

    @eager_break_during_capture
    def _finalize_prefetch(self, output: torch.Tensor) -> None:
        start_time = time.perf_counter()
        _refuse_capture()
        job = self._prefetch_job
        assert job is not None, "PLE disk finalize requires a prior start_prefetch"
        self._prefetch_job = None
        STATS["hit" if job.done() else "miss"] += 1
        staged = job.result()
        STATS["wait_ms"] += (time.perf_counter() - start_time) * 1e3
        rows = self._prefetch_rows
        if self.tp_size == 1:
            # No ETP reduction or DP selection: decode straight into output.
            self._upload_and_decode(staged, output.view(rows, self.num_ngram_heads, self.embedding_dim))
        else:
            embeddings = output.new_empty((rows, self.num_ngram_heads, self.embedding_dim))
            self._upload_and_decode(staged, embeddings)
            _, slot_offset = self._get_dp_gather_slot(output.shape[0])
            if self.reduce_results:
                embeddings = self._reduce_etp_embeddings(embeddings)
            else:
                # The outer TP reduce-scatter requires ETP=TP.
                assert self.etp_data_parallel_size == 1
            embeddings = self._select_embeddings(embeddings, output.shape[0], slot_offset)
            output.copy_(embeddings.flatten(-2))
        STATS["calls"] += 1
        STATS["op_ms"] += (time.perf_counter() - start_time) * 1e3
        _log_stats()

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Gather ETP IDs, look up rows from disk, reduce, and select local rows."""
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
        # Allocated in the graph segment so the eager lookup writes a static
        # buffer that later segments read.
        embeddings = torch.empty(
            (*gathered_ids.shape, self.embedding_dim),
            dtype=self.embedding_method.lookup_dtype(self),
            device=gathered_ids.device,
        )
        torch.ops.vllm.qwen4_exp_ple_disk_lookup(gathered_ids, embeddings, self.prefix)
        if self.reduce_results:
            embeddings = self._reduce_etp_embeddings(embeddings)
        else:
            # The outer TP reduce-scatter requires ETP=TP.
            assert self.etp_data_parallel_size == 1
        return self._select_embeddings(
            embeddings,
            ngram_ids.shape[0],
            slot_offset,
        )

    def lookup_into(self, ngram_ids: torch.Tensor, output: torch.Tensor) -> None:
        """Read the rows for ``ngram_ids`` from disk and decode them to ``output``."""
        _refuse_capture()
        ids = ngram_ids.reshape(-1)
        if ids.numel() == 0:
            return
        if self._staging is None:
            self._allocate_staging(output.device)
        staging = self._staging
        assert staging is not None
        # The blocking D2H copy also waits for the previous lookup's H2D copy
        # on this stream, so the pinned staging buffer is free to reuse.
        start_time = time.perf_counter()
        ids_host = ids.cpu().numpy()
        STATS["id_sync_ms"] += (time.perf_counter() - start_time) * 1e3
        self._upload_and_decode(self.lookup.stage(ids_host, staging.numpy()), output)

    def _upload_and_decode(self, staged, output: torch.Tensor) -> None:
        """Upload the slot map and missed rows, fill their slots, decode.

        One H2D copy; ``index_copy_`` per table writes the missed rows into
        their slots; the lookup kernel then reads the pool through the slot map
        (slot 0 is the zero row). The step's pins are released once this is
        enqueued: a later step's scatter into an evicted slot runs after this
        decode on the same stream.
        """
        lookup = self.lookup
        pools = self._pools
        staging = self._staging
        device_staging = self._device_staging
        assert pools is not None
        assert staging is not None and device_staging is not None
        num_ids, num_misses, slot_offset, table_offsets, _ = staged
        try:
            if num_ids == 0:
                return
            start_time = time.perf_counter()
            last = lookup.tables[-1].row_bytes
            num_bytes = table_offsets[-1] + num_misses * last
            device_staging[:num_bytes].copy_(staging[:num_bytes], non_blocking=True)
            if num_misses:
                slots = device_staging[slot_offset : slot_offset + 8 * num_misses].view(torch.int64)
                for pool, offset in zip(pools, table_offsets, strict=False):
                    row_bytes = pool.shape[1]
                    pool.index_copy_(
                        0,
                        slots,
                        device_staging[offset : offset + num_misses * row_bytes].view(num_misses, row_bytes),
                    )
            launch_time = time.perf_counter()
            STATS["h2d_ms"] += (launch_time - start_time) * 1e3
            slot_map = device_staging[: 4 * num_ids].view(torch.int32)
            num_slots = pools[0].shape[0]
            if self.is_nvfp4:
                _lookup_nvfp4_ple_embedding_kernel[(num_ids,)](
                    pools[0],
                    pools[1],
                    self.weight_scale_2,
                    slot_map,
                    output,
                    self.embedding_dim,
                    0,
                    num_slots,
                    BLOCK_D=triton.next_power_of_2(self.embedding_dim),
                )
            else:
                row_bytes = pools[0].shape[1]
                _lookup_ple_embedding_from_pinned_kernel[(num_ids,)](
                    pools[0],
                    slot_map,
                    output,
                    row_bytes,
                    0,
                    num_slots,
                    BLOCK_D=triton.next_power_of_2(row_bytes),
                )
            STATS["launch_ms"] += (time.perf_counter() - launch_time) * 1e3
        finally:
            lookup.release(staged)


def _refuse_capture() -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "PLE disk lookups run on the host and cannot be captured in a "
            "FULL CUDA graph; use -cc.cudagraph_mode=PIECEWISE or "
            "--enforce-eager"
        )


def _log_stats() -> None:
    now = time.monotonic()
    if _STATS_SEC <= 0 or now - _stats_last[0] < _STATS_SEC:
        return
    elapsed = now - _stats_last[0] if _stats_last[0] else float(_STATS_SEC)
    _stats_last[0] = now
    calls = STATS["calls"]
    if not calls:
        return
    per_call = {key: STATS[key] / calls for key in STATS if key.endswith("_ms")}
    cache_text = ""
    caches = [layer.lookup.cache for layer in _LAYERS.values()]
    if caches:
        hits, misses = STATS["cache_hits"], STATS["cache_misses"]
        cache_text = (
            f"; cache: {hits} hit rows, {misses} miss rows "
            f"({100.0 * hits / max(1, hits + misses):.1f}% hit), "
            f"{STATS['evictions']} evictions, {STATS['read_calls']} preads "
            f"({STATS['disk_reads']} waited for the disk), "
            f"{STATS['read_bytes'] / 2**20:.1f} MiB read, "
            f"{STATS['mmap_rows']} rows via mmap, "
            f"{sum(cache.used for cache in caches)}/"
            f"{sum(cache.num_slots - 1 for cache in caches)} slots used"
        )
    logger.info(
        "PLE disk stats (last %.0fs): %d lookups (%d sync, %d prefetch hit, "
        "%d miss), %d rows (%d unique); ms/lookup: %.2f host op, %.2f prefetch "
        "wait, %.2f id sync, %.2f dedup, %.2f gather, %.3f H2D, %.3f decode "
        "launch%s",
        elapsed,
        calls,
        STATS["sync"],
        STATS["hit"],
        STATS["miss"],
        STATS["rows"],
        STATS["unique"],
        per_call["op_ms"],
        per_call["wait_ms"],
        per_call["id_sync_ms"],
        per_call["dedup_ms"],
        per_call["gather_ms"],
        per_call["h2d_ms"],
        per_call["launch_ms"],
        cache_text,
    )
    STATS.update(STATS_ZERO)


def _ple_disk_lookup(
    ngram_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    start_time = time.perf_counter()
    _LAYERS[layer_name].lookup_into(ngram_ids, output)
    STATS["calls"] += 1
    STATS["sync"] += 1
    STATS["op_ms"] += (time.perf_counter() - start_time) * 1e3
    _log_stats()


def _ple_disk_lookup_fake(
    ngram_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen4_exp_ple_disk_lookup",
    op_func=eager_break_during_capture(_ple_disk_lookup),
    mutates_args=["output"],
    fake_impl=_ple_disk_lookup_fake,
)
