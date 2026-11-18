"""Helpers for the disk-backed Qwen4Exp PLE lookup tests (host half).

Builds a tiny checkpoint with an NVFP4-shaped PLE table split into shards over
two safetensors files (plus an index and distractor tensors), hands the
layer's shards over like vLLM's safetensors loader (zero-copy views of the
mmapped files, located through ``/proc/self/maps``), maps them with
``ple_disk_table`` and checks that staged rows equal direct indexing of the
full table: for decode-sized (inline) and prefill-sized (threaded) batches,
with repeated ids, and with a TP vocabulary range that leaves rows unowned.
The prefetch staging (``stage_async``) must wait for its ids and write the
same staging bytes as the synchronous ``stage``.
With torch installed it also decodes both through a reference NVFP4 decoder
written like the 0.33 Triton kernel and compares the BF16 outputs.

Ported from the contributor's overlay self-test; runs on the CPU with NumPy
(torch optional).
"""

import json
import mmap
import os
import re
import struct

import numpy as np


EMBED_DIM = 160  # Qwen3.8-Flash-Next head_dim: 2560 / 16 n-gram heads
NUM_ROWS = 1000  # padded n-gram vocabulary (org_vocab_size)
SPLIT_PARTS = 7  # split_ngram_parts
LAYER = 1
PREFIX = f"model.language_model.layers.{LAYER}.ple.ple_embedding.ngram_embedding"


def write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        data = np.ascontiguousarray(array).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for data in blobs:
            f.write(data)


def build_checkpoint(directory, rng):
    weight = rng.integers(0, 256, (NUM_ROWS, EMBED_DIM // 2), dtype=np.uint8)
    # E4M3 codes 0x00..0x7e (positive finite scales).
    scale = rng.integers(0, 0x7F, (NUM_ROWS, EMBED_DIM // 16), dtype=np.uint8)
    shard_size = (NUM_ROWS + SPLIT_PARTS - 1) // SPLIT_PARTS
    files = ({}, {})
    for shard in range(SPLIT_PARTS):
        rows = slice(shard * shard_size, min(NUM_ROWS, (shard + 1) * shard_size))
        # Interleave shards over two files, weight and scale apart.
        files[shard % 2][f"{PREFIX}.shard_{shard}.weight"] = ("U8", weight[rows])
        files[(shard + 1) % 2][f"{PREFIX}.shard_{shard}.weight_scale"] = (
            "F8_E4M3",
            scale[rows],
        )
    files[0][f"{PREFIX}.weight_scale_2"] = ("F32", np.array([0.5], np.float32))
    # Distractors: other layer and MTP tensors with the same suffixes.
    files[1][f"mtp.layers.{LAYER}.ple.ple_embedding.ngram_embedding.shard_0.weight"] = (
        "U8",
        weight[:3],
    )
    files[1]["model.language_model.layers.11.ple.ple_embedding.ngram_embedding.shard_0.weight"] = (
        "U8",
        weight[:3],
    )
    weight_map = {}
    for index, tensors in enumerate(files):
        name = f"model-0000{index + 1}-of-00002.safetensors"
        write_safetensors(os.path.join(directory, name), tensors)
        weight_map.update(dict.fromkeys(tensors, name))
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    return weight, scale


def parse_safetensors_header(path):
    """Return (tensor header, absolute data start) of a safetensors file."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


def loader_views(directory):
    """Yield ``(name, view, path, offset)`` per tensor: the view is what
    vLLM's safetensors loader hands over (zero-copy, into the file mapped
    whole, as ``safe_open`` maps it); path and offset are the header's."""
    for file_name in sorted(os.listdir(directory)):
        if not file_name.endswith(".safetensors"):
            continue
        path = os.path.join(directory, file_name)
        header, data_start = parse_safetensors_header(path)
        with open(path, "rb") as f:
            mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = np.frombuffer(mapped, np.uint8)
        for name, meta in header.items():
            start, end = meta["data_offsets"]
            rows = meta["shape"][0]
            yield name, data[data_start + start : data_start + end].reshape(rows, -1), path, data_start + start


def loader_shards(directory, file_shard, suffixes=("weight", "weight_scale")):
    """The ``{suffix: {checkpoint_start: shard}}`` the disk layer's weight
    loader records for this layer's table shards (``file_shard`` on each
    view), checked against the header offsets."""
    pattern = re.compile(rf"^{re.escape(PREFIX)}\.shard_(\d+)\.({'|'.join(suffixes)})$")
    shard_size = (NUM_ROWS + SPLIT_PARTS - 1) // SPLIT_PARTS
    tables = {suffix: {} for suffix in suffixes}
    for name, view, path, offset in loader_views(directory):
        match = pattern.match(name)
        if match is None:
            continue  # the loader routes other tensors elsewhere
        shard = file_shard(view.ctypes.data, view.shape[0])
        assert shard == (path, offset, view.shape[0]), (name, shard)
        tables[match.group(2)][int(match.group(1)) * shard_size] = shard
    return tables


def expand(lookup, staging, ids):
    """Rebuild per-id rows from the staging buffer like the GPU side does."""
    num_ids, num_unique, offsets = lookup.stage(ids, staging)
    inverse = staging[: num_ids * 8].view(np.int64)
    rows = []
    for table, offset in zip(lookup.tables, offsets, strict=False):
        unique_rows = staging[offset : offset + num_unique * table.row_bytes].reshape(num_unique, table.row_bytes)
        out = np.zeros((num_ids, table.row_bytes), np.uint8)
        owned = inverse >= 0
        out[owned] = unique_rows[inverse[owned]]
        rows.append(out)
    return rows


def direct(weight, scale, ids, vocab_start, vocab_end):
    owned = (ids >= vocab_start) & (ids < vocab_end)
    safe = np.where(owned, ids, 0)
    return (
        np.where(owned[:, None], weight[safe], 0).astype(np.uint8),
        np.where(owned[:, None], scale[safe], 0).astype(np.uint8),
    )


def nvfp4_reference(torch, packed, scale, global_scale):
    """Decode like _lookup_nvfp4_ple_embedding_kernel (low nibble = even)."""
    e2m1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    packed = torch.from_numpy(packed).long()
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    scales = torch.from_numpy(scale).view(torch.float8_e4m3fn).float()
    values = e2m1[codes] * scales.repeat_interleave(16, dim=-1) * global_scale
    return values.to(torch.bfloat16)
