"""CPU test of the disk-backed Qwen4Exp PLE lookup (``mjolnir.qwen4exp.ple_disk_table``).

Builds a tiny checkpoint with an NVFP4-shaped PLE table split into shards over
two safetensors files (plus an index and distractor tensors), locates the
shards from loader-style mmap views (``file_shard``, as vLLM 0.34's
``disk_offload`` does) and checks that staged rows equal direct indexing of
the full table: decode-sized (inline) and
prefill-sized (threaded) batches, repeated ids, a TP vocabulary range that
leaves rows unowned, and the prefetch staging (``stage_async``) against the
synchronous ``stage``. With torch installed it also decodes both through a
reference NVFP4 decoder. Ported from the contributor's overlay self-test.
"""

import tempfile
import threading

import numpy as np
import qwen4exp_ple_fixtures as fx

import mjolnir.qwen4exp.ple_disk_table as pdt


def test_disk_lookup_stages_the_direct_rows():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as directory:
        weight, scale = fx.build_checkpoint(directory, rng)
        row_bytes = {"weight": fx.EMBED_DIM // 2, "weight_scale": fx.EMBED_DIM // 16}
        shards = fx.loader_shards(directory, pdt.file_shard)
        cases = [
            ("decode, TP=1", 4 * 16, 0, fx.NUM_ROWS),
            ("prefill, TP=1", 300 * 16, 0, fx.NUM_ROWS),
            ("prefill, TP rank 1 of 2", 300 * 16, fx.NUM_ROWS // 2, fx.NUM_ROWS),
        ]
        for label, num_ids, vocab_start, vocab_end in cases:
            lookup = pdt.open_ple_disk_lookup(
                shards,
                row_bytes,
                num_rows=fx.NUM_ROWS,
                max_rows=num_ids,
                vocab_start=vocab_start,
                vocab_end=vocab_end,
            )
            for table in lookup.tables:
                if num_ids > 64:
                    table.fast_rows = 16  # force the threaded path
                table.chunk = 64  # several tasks per shard on the threaded path
            staging = np.empty(lookup.staging_bytes, np.uint8)
            # Repeats are common; include the first and last row explicitly.
            ids = rng.integers(0, fx.NUM_ROWS, num_ids // 2)
            ids = np.concatenate((ids, ids[: num_ids // 2 - 2], [0, fx.NUM_ROWS - 1]))
            rng.shuffle(ids)
            got_weight, got_scale = fx.expand(lookup, staging, ids)
            want_weight, want_scale = fx.direct(weight, scale, ids, vocab_start, vocab_end)
            assert np.array_equal(got_weight, want_weight), label
            assert np.array_equal(got_scale, want_scale), label
            print(f"ok  {label}: {num_ids} ids, rows equal fx.direct indexing")

            # Prefetch: the job waits for the ids (the D2H event), then stages
            # exactly the bytes the synchronous path stages.
            want_staged = lookup.stage(ids, staging)
            want_bytes = staging.copy()
            ids_host = np.zeros(num_ids, np.int64)  # not yet "copied"
            async_staging = np.zeros_like(staging)
            ready = threading.Event()
            job = lookup.stage_async(ids_host, async_staging, ready.wait)
            ids_host[:] = ids  # the D2H copy lands before the event fires
            assert not job.done(), label
            ready.set()
            assert job.result(timeout=30) == want_staged, label
            num_bytes = want_staged[2][-1] + want_staged[1] * lookup.tables[-1].row_bytes
            assert np.array_equal(async_staging[:num_bytes], want_bytes[:num_bytes]), label
            print(f"ok  {label}: prefetch staging equals synchronous staging")

            try:
                import torch  # ty: ignore[unresolved-import]
            except ImportError:
                continue
            got = fx.nvfp4_reference(torch, got_weight, got_scale, 0.5)
            want = fx.nvfp4_reference(torch, want_weight, want_scale, 0.5)
            assert torch.equal(got, want), label
            print(f"ok  {label}: NVFP4 decode equal ({got.shape[1]} values/row)")

        # Too many ids for the staging buffer must refuse, not overrun.
        try:
            lookup.stage(np.zeros(lookup.max_rows + 1, np.int64), staging)
        except ValueError:
            print("ok  oversize batch refused")
        else:
            raise AssertionError("oversize batch accepted")
        job = lookup.stage_async(np.zeros(lookup.max_rows + 1, np.int64), staging, lambda: None)
        try:
            job.result(timeout=30)
        except ValueError:
            print("ok  oversize prefetch refused (raised to the caller)")
        else:
            raise AssertionError("oversize prefetch accepted")
        timed = ("id_sync_ms", "dedup_ms", "gather_ms")
        assert all(pdt.STATS[key] > 0 for key in timed), pdt.STATS
        assert set(pdt.STATS) == set(pdt.STATS_ZERO)
        print("ok  stats: " + ", ".join(f"{k}={pdt.STATS[k]:.2f}" for k in timed))


def test_file_shard_needs_a_file_backed_view():
    anonymous = np.zeros(4096, np.uint8)  # heap or anonymous mapping
    try:
        pdt.file_shard(anonymous.ctypes.data, 1)
    except ValueError:
        print("ok  heap view refused (not file-backed)")
    else:
        raise AssertionError("anonymous view accepted")


def test_missing_shards_refuse():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as directory:
        fx.build_checkpoint(directory, rng)
        row_bytes = {"weight": fx.EMBED_DIM // 2, "weight_scale": fx.EMBED_DIM // 16}
        shards = fx.loader_shards(directory, pdt.file_shard)
        shard_size = shards["weight"][0][2]
        for start in (0, 3 * shard_size):
            partial = {suffix: dict(table) for suffix, table in shards.items()}
            del partial["weight_scale"][start]
            try:
                pdt.open_ple_disk_lookup(
                    partial, row_bytes, num_rows=fx.NUM_ROWS, max_rows=16, vocab_start=0, vocab_end=fx.NUM_ROWS
                )
            except ValueError as error:
                assert "missing" in str(error), error
            else:
                raise AssertionError(f"table without shard {start // shard_size} accepted")
        print("ok  a missing checkpoint shard refuses")
