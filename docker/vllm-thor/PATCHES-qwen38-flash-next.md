# Qwen3.8-Flash-Next (qwen4_exp) serving stack (2026-11-17)

The deep-dive part of [PATCHES.md](PATCHES.md) for these patches (their rows are in
the PATCHES.md table); split out to keep PATCHES.md under the file-length limit.

The patches and `mjolnir.qwen4exp` modules that serve `nvidia/Qwen3.8-Flash-Next-NVFP4`
and `nvidia/Qwen3.8-Flash-Next-FP4-Local` on Thor (configs under
`configs/nvidia/Qwen3.8-Flash-Next/`), on vLLM 0.34.0. Each Thor-authored
piece is inert unless a config opts in through its `.env`; its vLLM hunks
import `mjolnir.qwen4exp` only inside the switched branch. The loader backport
is always on and changes nothing for checkpoints without nested shards. The
measured effects quoted below come from the contributor's Thor runs on the
official **v0.33.0** image with the same code as file overlays
([NemoClaw-Thor](https://github.com/pastoriomarco/NemoClaw-Thor) `69df863`,
`serving/docs/QWEN38-FLASH-NEXT-FP4-LOCAL-THOR.md`, `FLASH-NEXT-SPEED-PLAN.md`);
they are not Mjolnir evidence until `mjolnir bench` reproduces them, and they
are re-measured on 0.34. vLLM 0.34 carries the CSF-compressed NVFP4 scales
FP4-Local needs (#64312 + #64869), so no backport is needed for them.

**Index subdirectory (`64739`).** Upstream backport (open PR, split out of
#64312), test files left out. Without it the FP4-Local checkpoint needs a
hand-made directory with the shards linked at the top level (how it was
served before this port). **Drop when** the base carries #64739.

**PLE slot-pool cache in front of the disk table (`thor-qwen4-ple-disk`, `mjolnir/qwen4exp/ple_disk*.py`).**
Both checkpoints carry one PLE n-gram table (layer 1, ~320 M rows; 26.8 GiB
NVFP4 in FP4-Local, 47.7 GiB FP8 in NVFP4). Kept in pinned host memory or on
the GPU, it allocated ~112 GB on Thor's unified pool before loading and the
model could not start. vLLM 0.34 (#59958) keeps it in the checkpoint files
with `--engram-config '{"disk_offload": true}'` (set in the YAML), the on/off
switch for the disk table: each step syncs the stream once, then reads that
step's distinct rows with one serial `pread` each on the main thread, with no
cache and no prefetch. With `disk_offload` on and `MJOLNIR_PLE_DISK_CACHE_GB` > 0
(`.env`), the patch builds Mjolnir's layer instead (not a subclass of
upstream's, so neither the per-step sync nor upstream's staging applies): it
finds the shard files and offsets from the loader's views the way upstream
does (`/proc/self/maps`), and `bind_file_shards` opens a GPU CLOCK slot pool
of that size (design after TensorRT Edge-LLM 0.12 `--ple-residency cached`)
inside `load_model`, so the pool counts as model memory before KV sizing. A
step reads only its missed rows (page-aligned merged `pread`s, parallel
readers for the ones that wait for the disk; prefill-sized miss sets via the
mmap gather), uploads them with an int32 slot map and decodes through the
slot map with upstream's lookup kernels; prefetch
(`MJOLNIR_PLE_DISK_PREFETCH`, default on) stages a step on a background thread
while the preceding decoder layer runs. With the cache at 0 or unset,
upstream's reader runs unchanged. Needs PIECEWISE graphs (the lookup is an
eager break) and ETP = TP = 1. Measured effect (contributor's Thor runs,
FP4-Local, recipe R7b → R8a in the contributor's doc; **measured on 0.33 with
Mjolnir's own disk reader underneath, against that reader without the pool;
to be re-measured on 0.34 against upstream's `disk_offload`**): decode GPU
time per step 51.7 → 49.3–50.0 ms, warm 7.6K prefill 3.2 → 3.04 s, KV 311,928
tokens at share 0.70 → 475,281 at 0.75 (the pool removes the page-cache
reservation that held the share at 0.70). CPU tests:
`tests/cli/test_qwen4exp_ple_disk_{table,cache,layer}.py`. **Drop when**
upstream's disk table gains a row pool and asynchronous prefetch (e.g. vLLM
#66941, #66464); the switch then stays `disk_offload` alone.
