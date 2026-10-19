# Design: platform × model matrix

**Status:** Phase 0 (the data/platform layer) landed — the platform registry,
`mjolnir platform`, record fields and per-platform legs; commit ref to be
added at delivery. Phases 1–3 open. Proposal — extends the repo from one
platform + one primary model to a matrix of (platform × model × config).
Absorbs and generalizes
[#31](https://github.com/thor-arena/mjolnir/issues/31) (per-model rendering).

> **Revision B (2026-10-19, research-informed — supersedes the parts it names):**
> Research in `task/research/` (detection.md, jetpack-containers.md, spark-stack.md) settled:
> 1. **Detection-first platform layer** — no remembered platform state, no `platform set`.
>    Every command auto-detects (two-source rule: `nvidia-smi` name + compute cap, cross-checked
>    against device-tree `compatible` / DMI product name / `/etc/nv_tegra_release`). `mjolnir
>    platform` stays read-only (prints detection + provenance); `MJOLNIR_PLATFORM` env and
>    `--platform` are override hatches for tests/CI. §4.1's state persistence is dropped.
> 2. **One image, three platforms** — JetPack 7.2 added Orin (L4T 39.2.1, CUDA 13.2.2);
>    container compat follows the JP/L4T release + arch family, not the board. The cu130 arm64
>    track (our base) covers Thor + Orin (sm_80 bin-compat → sm_87) AND DGX Spark (sm_120 →
>    sm_121). So: ONE `narandill/mjolnir-vllm` image, arch resolved at runtime (Cycle 1's
>    guard), Mjolnir hard-disabled on sm_87/sm_121, Thor patches probe-gated, native baselines
>    always on the official image. §5's per-platform image repos are dropped; `docker/`
>    restructure becomes optional cleanup, and host requirements are documented per platform
>    (Orin: JP 7.2.1+; Spark: DGX OS + CUDA 13).
> 3. **Config platform overlays confirmed** — deltas are ~6–8 keys per platform (§6);
>    `performance-mode`/`optimization-level` and the TORCH_SDPA ViT workaround are model-layer.
>    A `kv-cache-dtype × attention-backend` availability matrix gets validated at serve time.
> 4. **Per-config docs** — every `<quant>/` gets generated `README.md` + `BENCHMARKS.md`
>    (per-platform sections from raw data); the model dir gets a `README.md` index (§8).
> Stock attention reality per platform: Thor = FA4/FlashInfer; Orin = FA2 (no FA3/FA4 on
> Ampere); Spark = FlashInfer (no FA4 on sm_121 upstream). fp6 KV is Thor/Mjolnir-only.

> **Revision C (2026-10-19, interview-approved — supersedes §8 rendering + adds §4a):**
> The Cycle-3 documentation/benchmark-presentation architecture is spec'd in
> `task/interviewer/spec-docs-benchmarks.md` (16 locked decisions): per-platform kernel case
> studies move `docs/kernel/` → `docs/platforms/<platform>/kernel/`; every config gets a
> generated `README.md` + `BENCHMARKS.md` (per-platform sections, "no data yet" placeholders);
> per-model + per-platform index pages (j2, generated); quant availability = ONE in-code
> three-state table (✅ native / ⚠️ loads-with-fallback / ❌ not loadable — per
> `task/research/quants.md`) read by serve-validation, legs filtering, and every renderer;
> calibration becomes `kv_scales/<format>.json` at model root; no `model.yaml` this cycle
> (metadata derived from config scan); static pages = citable permalinks, site = filters +
> compare + search, no duplicated tables.

---

## 1. Goal

1. **Platforms** — dedicated, tailored Mjolnir kernels, each with its own Docker
   image, shipped in the **single `mjolnir-kernel` PyPI module**:

   | Platform | Arch | Status | Kernel family |
   |---|---|---|---|
   | Jetson AGX Thor | `sm_110` | **done** — tcgen05/TMEM/TMA, 20 SMs, 2-CTA clusters | Blackwell tensor-core decode |
   | DGX Spark (GB10) | `sm_120/121` | research HW online | port of the Thor template: TMA + tcgen05, different SM count / SMEM / cluster support (probe) |
   | Jetson Orin | `sm_87` | planned | **new family**: Ampere — `mma.sync` + `cp.async` ring, no TMA / TMEM / clusters / fp8 tensor cores |
   | Clusters of DGX Sparks | n/a | planned | not a kernel — DP serving over ConnectX-7; a *benchmark* target |

2. **Platform × model measurement** — every benchmark series is keyed
   `(platform, model, config)`; raw data and logs stay self-describing so the
   render layer can generate **per-model `README.md` + `BENCHMARK.md` inside
   `configs/<vendor>/<model>/`**. The root README becomes model-agnostic and
   pins the default model (Qwen3.8-27B) for the headline.

3. **GH Pages v2** — one site that can filter by platform, model, config,
   backend family, and stage; search across everything; and compare the same
   (model, config) **across platforms** side by side.

---

## 2. Where the single-platform assumption lives today

| Location | Hard-coded thing | Becomes |
|---|---|---|
| `src/mjolnir/mjolnir/t5_*.py` | the sm_110 kernel itself (20 SMs, 2-CTA, TMEM) | the `thor` arch implementation; new `spark` / `orin` siblings |
| `src/mjolnir/mjolnir/backend.py` | assumes the T5 class | resolves a kernel **per device arch** at `builder()` time |
| `src/mjolnir-cli/.../config.py` | `DEFAULT_MODEL`, `DEFAULT_QUANT`, `IMAGE_REPO`, `VLLM_VERSION`, `DEFAULT_IMAGE`, `STOCK_IMAGE` — module singletons | `DEFAULT_MODEL` stays (the pin); image/version constants move **into a `Platform` record** |
| `src/mjolnir-cli/.../legs.py` | one global leg list, implicitly Thor + DEFAULT_MODEL | `LEGS[platform]`, model-aware via existence checks + `model.yaml` |
| `history.py` records | `host`, `image`, `model`, `config` — **no platform** | add `"platform"` (legacy rows infer `"thor"`) |
| `verify/results.py` `mjolnir.verify/1` | `host`, `image` — no platform | add optional `"platform"` field (schema stays `/1`; optional fields are backward-compatible) |
| `charts_e2e.e2e_rows(layout, model=)` | model hook exists; platform missing | `e2e_rows(layout, model=, platform=)`; pairing matches on **both** |
| `readme.py` `ReadmeContext(model=DEFAULT_MODEL)` | one target | multi-target: `(platform, model)` contexts |
| `interactive.py` | one page, hard-coded series lists (`["mjolnir", …]`, `["NVFP4", …]`) | shell + manifest + lazy per-pair data; filters in JS |
| `docker/vllm-thor/` | the only image | `docker/vllm-<platform>/` per platform |
| `configs/<vendor>/<model>/<quant>/config.yaml` | Thor-specific flags inline | platform-neutral base + `platforms/<platform>.yaml` overlay |

Groundwork already merged (#24 / a4da31b / 373fb56): `config.model_of` /
`by_model` / `model_short`, `e2e_rows(model=)`, `bench compare --model`. This
plan builds directly on those hooks and adds the platform dimension beside the
model dimension with the same philosophy.

---

## 3. Design principles

1. **One rule for a record's identity** — mirrors `model_of`:
   ```python
   def platform_of(rec) -> str | None: ...   # rec.get("platform")
   def series_key(rec) -> tuple:             # (platform, model, config)
   ```
   Unlike `model_of`, `platform_of` gets a **legacy inference**: the repo's
   entire history is Thor, so a platform-less record resolves to `"thor"` when
   its image matches the Thor images — documented in the schema, plus a
   one-time `mjolnir migrate --backfill-platform` that rewrites `history.jsonl`
   and the verify raws in a dedicated commit (append-only discipline applies to
   *results*, not to a schema migration; the commit is reviewable and revertible).

2. **The index is the query layer, not the directory tree.** The ticket's
   proposed `benchmarks/results/<org>/<model>/<config>/` tree is *not* adopted
   for the raw data — see §7.2. Raw files already carry their own keys; the
   generated per-model pages land in `configs/…` instead.

3. **Kernels are JIT — the wheel stays pure Python.** Everything is compiled
   on-device by `nvidia-cutlass-dsl`, so one sdist/wheel can carry all arch
   implementations; per-arch modules are lazy-imported so an Orin machine never
   touches Thor code (and vice versa). No `[tool.hatch.build]` changes, no
   binary wheels, `release-kernel.yml` unchanged.

4. **`legs.py` stays "the one place".** It gains a platform dimension and stays
   the single source of what is measured; per-model deltas come from
   `model.yaml` + config-dir existence checks, not from a second leg registry.

5. **A guessed platform is how two platforms get compared** — the renderer
   refuses to pair legs across `(platform, model)` boundaries, exactly as #24
   fixed for models.

---

## 4. Platform layer

### 4.1 The `Platform` record

> **Superseded by Revision B (1):** no remembered platform state — `platform set` and the
> state file are gone; resolution is flag > `MJOLNIR_PLATFORM` > two-source detection.

New `src/mjolnir-cli/mjolnir_cli/platforms.py` (and a mirrored, dependency-free
descriptor in the kernel package — the kernel needs the same registry for its
own `can_implement` gating):

```python
@dataclass(frozen=True)
class Platform:
    key: str                 # "thor" | "spark" | "orin"
    name: str                # "Jetson AGX Thor"
    sm: tuple[int, int]      # (11, 0) / (12, 0) or (12, 1) / (8, 7)
    sm_family: str           # "sm_110"
    sm_count: int
    image_repo: str          # "narandill/mjolnir-vllm" (thor) / "narandill/mjolnir-spark" / ...
    stock_image: str         # the per-platform "official vLLM" reference image
    vllm_version: str        # per-platform pinned upstream (Orin will pin its own)
    ref_image_env: str       # per-platform MJOLNIR_REF_IMAGE override name
    hw_setup: str | None     # "scripts/hw/setup-thor.sh" for now; per-platform setup scripts
```

- `DEFAULT_IMAGE`, `STOCK_IMAGE`, `IMAGE_REPO`, `VLLM_VERSION` become
  derived-from-`Platform` (kept as module-level aliases for the default
  platform during the transition, deleted once callers move).
- Detection: `nvidia-smi --query-gpu=compute_cap` (host) and
  `torch.cuda.get_device_capability()` (in-container); remembered via
  `mjolnir platform set <key>` into `$MJOLNIR_STATE` next to `mjolnir model`,
  overridable per command (`--platform`) and per env (`MJOLNIR_PLATFORM`).
- `Settings` gains `platform`; `resolve()` threads it; every `make_record` and
  verify save writes `platform`.

### 4.2 Kernel package layout

```
src/mjolnir/mjolnir/
  __init__.py          # register() — unchanged shape, now arch-aware
  platforms.py         # NEW: descriptor + detect() + kernel_for(arch)
  arch/
    __init__.py        # registry: (major, minor) → impl, lazy imports
    thor.py            # re-exports t5_* — the sm_110 implementation
    spark.py           # Phase 1: sm_120/121 (port of the Thor template)
    orin.py            # Phase 2: sm_87 (new Ampere family)
  t5_*.py              # untouched (BSD-3-Clause header intact, DSL body unsplittable)
```

- `backend.py` `builder()` calls `platforms.kernel_for(device_cc)`; unknown arch
  → a clear error naming the arch and the platforms this wheel ships, and vLLM
  falls back (the backend already degrades to the fallback backend on
  `can_implement` failures — reuse that path).
- `MJOLNIR_ARCH` env override for bring-up on research HW.
- **Shared vs per-arch:** `t5_const.py` math helpers and the plan/stride logic
  (`t5_plan.py`) get platform constants extracted (SM count, SMEM budget, ring
  depths, cluster support flag) so `spark` reuses the Thor template with
  different constants and `orin` shares only the plan/merge structure.
- **Orin reality check** (drives expectations, not just code): `sm_87` has no
  TMA, no TMEM/tcgen05, no thread-block clusters, and **no fp8 tensor cores** —
  fp8/fp6 KV is a non-starter there. The Orin kernel is bf16 KV (+ bf16 Q),
  `mma.sync.m16n8k16`, `cp.async` double-buffered SMEM ring, occupancy-pinned
  split-KV on a bandwidth-limited part (AGX Orin 64GB: 16 SMs, ~205 GB/s
  LPDDR5). The win to chase is bandwidth utilization and split-KV quality vs
  FlashInfer/FA2 — the fp6 story simply does not transfer.
- **Spark reality check:** `sm_120/121` has TMA + tcgen05 + fp8/fp6, but
  consumer-derived Blackwell — **cluster support must be probed first**; the
  plan must work without the 2-CTA path (1-CTA × more CTAs/SM variant). SM
  count (48) and SMEM/launch-limits differ from Thor; those become platform
  constants, not conditionals sprinkled in the kernel body.

### 4.3 PyPI distribution

- Single `mjolnir-kernel` wheel; **no** per-arch extras (pure Python — extras
  would only gate `nvidia-cutlass-dsl`, which stays the existing `[dsl]` extra).
- Package metadata: drop "Jetson AGX Thor (sm_110)" from the description →
  "for NVIDIA Jetson Thor (sm_110), DGX Spark (sm_120/121) and Jetson Orin
  (sm_87)"; keywords add `jetson-orin`, `dgx-spark`, `sm87`, `sm120`.
- CI gains a no-GPU smoke: import `mjolnir.platforms`, assert the registry
  resolves every descriptor and that unknown-arch raises the designed error.

---

## 5. Images & workflows

> **Superseded by Revision B (2):** ONE image, three platforms — no per-platform image
> repos/dirs; host requirements are documented per platform (Orin: JP 7.2.1+; Spark: DGX OS
> + CUDA 13). Native baselines always on the official image.

### 5.1 One image directory per platform

```
docker/
  vllm-thor/     # existing — BASE_IMAGE vllm/vllm-openai:v0.32.0-ubuntu2404
  vllm-spark/    # Phase 1 — arm64 Blackwell-consumer base (probe: does the official
                 # arm64 image cover sm_120? else a DGX OS CUDA 13 base + vLLM build)
  vllm-orin/     # Phase 2 — JetPack 6.x base; per-platform pinned vLLM (see risks)
  _shared/       # extracted common pieces: the MJOLNIR enum + general_plugins
                 # patch, verify_patches/apply_patches machinery, canary runners
```

- The platform-neutral patches (MJOLNIR/CUSTOM enum + `vllm.general_plugins`
  entry point) move to `_shared/patches/`; per-platform patch dirs keep only
  what is arch-specific (FA4 fallback selection, arch workarounds).
- `PATCHES.md` per image dir, as today.
- Tags follow upstream vLLM **per platform**: `narandill/mjolnir-vllm:v0.32.0`
  (thor, unchanged), `narandill/mjolnir-spark:v0.32.0`, `narandill/mjolnir-orin:v<x>`
  (Orin's `x` will differ — JetPack pins it). The existing Thor tag/repo names
  do not move; nothing is renamed out from under committed history rows.

### 5.2 Workflow changes

- `docker-image.yml` → **matrix build** over platforms (`docker/vllm-*/`),
  each job building its own Dockerfile on `ubuntu-24.04-arm` (all three are
  arm64), pushing to its own repo, each syncing its own Hub description.
  GPU canaries keep self-skipping on CPU runners; per-platform GPU canaries
  run from the bring-up hosts (as today on Thor).
- `release-kernel.yml` — unchanged (pure-Python wheel).
- `pages.yml` — unchanged mechanically (`mjolnir plot --site` output grows, §9).
- `release.yml` — release notes gain a section per image.

---

## 6. Configs: platform overlays + model metadata

> **Revision B note:** the overlay mechanics below are implemented as specified
> (`<quant>/platforms/<platform>.yaml`, deep-merge, lists replace, absence ⇒ not on the
> platform); `model.yaml` is a later cycle, and there is no per-command `--platform` CLI
> option — the flag layer is programmatic (`resolve()`).

```
configs/Qwen/Qwen3.8-27B/
  model.yaml                  # NEW — model metadata (below)
  kv_scales.json, draft_vocab.json, chat templates …   # unchanged
  NVFP4_MJOLNIR_E2M3/
    config.yaml               # platform-NEUTRAL base (model behavior, quant, spec decoding)
    platforms/
      thor.yaml               # Thor deltas (performance-mode, TORCH_SDPA ViT workaround, …)
      spark.yaml              # Spark deltas
    .env                      # unchanged (opt-in env switches)
```

- `model.yaml` (hand-written, one per model):
  ```yaml
  hf-id: QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4
  served-name: Qwen/Qwen3.8-27B
  heads: {q: 24, kv: 4, headdim: 256}   # feeds the kernel harness/verify stages
  page-size: 128
  default-quant: NVFP4_MJOLNIR_E2M3_LMH8_VT64K
  legs: default                          # "default" → legs.py; or an explicit list
  ```
  This is what lets `verify sweep` measure *this model's* head layout and lets
  the renderer describe the model without importing the config YAML.
- **Overlay merge rule:** deep-merge `platforms/<platform>.yaml` over
  `config.yaml` (lists replace, dicts merge); `mjolnir serve --platform spark`
  renders the merged argv and prints which file contributed each flag. A config
  without an overlay for platform P **does not exist** on P (legs skip it) —
  that is the per-platform leg filter, for free.
- `BACKEND_LABELS` stays platform-neutral (a quant's label doesn't change per
  platform; the *pairing* does).

---

## 7. Data model & raw layout

### 7.1 Records

`history.jsonl` record gains:

```jsonc
{
  "platform": "thor",          // NEW — see legacy rule in §3
  "gpu": "NVIDIA IGX Thor",    // optional — nvidia-smi name, disambiguates host farms
  // host, image, model, config, backend, spec, gate, cells … unchanged
}
```

`mjolnir.verify/1` gains the same optional `platform` (+ `gpu`) field. Schema
stays `/1`; readers treat a missing field via the legacy rule. The `default`
flag on a verify run remains per-stage — but `results.latest()` grows
`platform=` / `model=` filters so "the latest default sweep" means *of that
pair*.

### 7.2 Raw tree: keep, don't move

Decision: **do not** restructure `benchmarks/raw/` into
`results/<org>/<model>/<config>/` (the ticket's sketch). Rationale:

- `history.jsonl` is already the join key — one line per run carrying
  `(platform, model, config, backend, raw → path)`; every renderer (charts,
  README, site, compare) already queries it. Adding `platform` completes the
  key.
- Verify raws are self-describing JSON (`stage`, `platform`, `model`, `params`)
  and stage-scoped by path — a model dimension inside a stage directory would
  fight the `latest(stage)` contract.
- Moving committed raws rewrites provenance (git history, `raw` relpaths inside
  records, chart manifests) for zero query benefit.

The **generated** per-model outputs (§8) are what lands in `configs/…` — that
satisfies the ticket's intent ("per-model README / BENCHMARK") without moving
data. If file-tree browsing of raw results ever matters more than stability,
`mjolnir plot --site` can additionally *emit* the results tree as generated
files (git-ignored), but that is a site concern, not a data concern.

What *does* change for logs: `benchmarks/raw/perf-<ts>/` keeps its name but the
record's `platform` field is what locates it; benchy's server log naming gains
the platform tag only inside the log header, not the path. New raw dirs are
cheap to disambiguate later because `history.jsonl` is the lookup.

### 7.3 Pairing rules (the heart of correctness here)

- `e2e.latest_pair(history, model=, platform=)` — both sides must match **both**
  dimensions. The `is_mjolnir` / `is_official` image sniffing becomes
  per-platform (`Platform.stock_image` prefix vs the platform's image repos).
- `bench compare` / site compare: same `(model, config)` across platforms is a
  *first-class* comparison (the platform matrix story); same platform across
  models is *not* a comparison the UI should imply by default.

---

## 8. Rendering: root README, per-model pages

### 8.1 Targets of `mjolnir plot --readme`

| Output | Source template | Content |
|---|---|---|
| `README.md` (root) | `README.md.j2` (reworked) | model-agnostic intro + **platform matrix** (what is supported where, status) + headline from the **default pair** (Qwen3.8-27B × Thor) + an index of all measured (platform, model) pairs linking into `configs/…` |
| `configs/<vendor>/<model>/README.md` | `MODEL_README.md.j2` (new) | the model page: what it is, its configs table, per-platform quickstart, links to BENCHMARK.md |
| `configs/<vendor>/<model>/BENCHMARK.md` | `MODEL_BENCHMARK.md.j2` (new) | **per platform section**: e2e table, spec/kv-cache/profile summaries, accuracy, links to per-pair charts; generated only for pairs with data |
| `docs/kernel/*.md` | unchanged templates | **stays the kernel case study for the default pair**; per-pair numbers live in BENCHMARK.md and the site. (Avoids exploding docs/ into platform×model trees.) |
| `assets/benchmarks/` | — | new charts go to `assets/benchmarks/<platform>/<model-short>/*.png` (+ `-dark`); existing flat PNGs stay (manifest v2 keeps resolving them) |
| `docker/vllm-*/README.md` | existing `.md.j2` | per-image Hub descriptions, now platform-aware |

- `ReadmeContext` gains `platform` and `model`; `render_all` iterates
  `platforms × by_model(records)` and renders a target only when the pair has
  data. Root README renders from the pinned `DEFAULT_MODEL` (and default
  platform) — the "pinned to default model" requirement — while linking to
  everything else.
- Findings sentences stay data-driven (`readme.py` findings), now parameterized
  by pair — no hand-typed numbers anywhere.
- CI keeps the "generated files reproduce" check; it now checks the per-model
  outputs too (render in CI and diff).
- **File-length:** `readme.py` is 702/1000 lines; the multi-target loop and new
  context bits go into a new `readme_targets.py` (target discovery + context
  building) so `readme.py` stays under the ratchet. Same for `interactive.py`
  (§9): split `site/`.

### 8.2 Charts

- Chart identity: `(platform, model, chart-id)`. `manifest.json` v2:
  entries carry `platform`, `model`, `path`; legacy entries (flat files) imply
  the default pair.
- `charts.py` render functions take a `(platform, model)` context; the series
  colors stay the design-system's (Mjolnir orange / FlashInfer blue / FA2
  aqua) — platforms don't get new colors, they get **separate charts** (and the
  site's compare view plots same-config-different-platform as a table, not a
  recolor).

---

## 9. The site (GH Pages v2)

### 9.1 Architecture

Keep it dependency-free (no node build step in CI):

```
_site/
  index.html          # the shell: filters, search, plot container
  manifest.json       # pair list + chart ids + stage summaries + sizes
  data/<platform>/<model-short>.json   # per-pair payload, lazy-loaded
  app.js, app.css     # real files (not embedded strings)
```

- `mjolnir plot --site` writes the shell + manifest + one payload per pair.
  The manifest loads first (small); selecting a pair fetches its payload.
  Keeps initial load fast as the matrix grows.
- `interactive.py` splits: `site/data.py` (payload builders), `site/page.py`
  (HTML shell), `app.js` template — `interactive.py` is 383 lines today and the
  filter UI will blow past the ratchet if it stays one file.

### 9.2 Filters (the full set)

| Filter | Control | Behavior |
|---|---|---|
| Platform | tabs: All / Thor / Spark / Orin (+ cluster) | narrows pairs; "All" enables compare-across-platforms |
| Model | searchable dropdown | one model at a time (or "All" → model index table) |
| Config / quant | multiselect chips | e.g. Mjolnir fp6, fp8, FlashInfer, FA2, drafters |
| Backend family | toggle: Mjolnir only / all | the headline view hides native legs unless asked |
| Stage | nav: Kernel (sweep, kvfmt, quant, …) / E2E / Spec / KV size / Accuracy / Profile | the existing per-stage views, per pair |
| Metric | decode / prefill / TTFT / ITL (e2e); time / gbps / speedup (kernel) | |
| Host / run age | dropdown + date range | multi-host farms (thor-01, thor-02, spark-lab-01) |
| Compare mode | switch | same (model, config) across platforms — side-by-side bars + a delta table; also same platform across drafters |
| Search | global text box | matches model / config / backend / chart titles; deep-links via URL hash |

- URL hash state (`#platform=thor&model=Qwen/Qwen3.8-27B&stage=e2e`) — every
  view shareable; no router library.
- The hero view stays: default pair (Qwen3.8 × Thor) when nothing is selected.
- Per-pair accuracy charts replace the single mixed accuracy page (ticket ask).

---

## 10. Spark clusters (Phase 3)

A **benchmark target**, not a kernel:

- Topology: N× DGX Spark nodes, vLLM data-parallel serving behind an LB
  (or vLLM's internal DP), interconnect = ConnectX-7 200GbE. No NVLink —
  tensor parallelism across nodes is not the story; DP throughput scaling is.
- `mjolnir bench cluster --nodes <hosts>`: orchestrates one server per node
  (reusing `dockerctl`/`gate` per host), benches the LB endpoint with
  llama-benchy, writes records with `platform: "spark-cluster"`, `nodes: N`.
- The platform record gets `"spark-cluster"` as a *derived* platform (same
  arch descriptor for kernel work; the bench dimension is the cluster).
- Deliverable: scaling curve (1→2→4 nodes) on the site + BENCHMARK.md section.
  Gate discipline: the clean-window gate runs **per host**; a cluster run is
  valid only when every node is clean — the gate aggregator checks all.

---

## 11. Phasing

### Phase 0 — groundwork (no new hardware; unblocks everything)

1. `platforms.py` (CLI + kernel) + detection + `mjolnir platform` + record
   fields + legacy inference + `migrate --backfill-platform`.
2. `legs.py` per platform (`LEGS[platform]`); model existence checks +
   `model.yaml`; `e2e_rows`/`latest_pair` gain `platform=`.
3. Charts per pair + manifest v2 + `render_all` multi-target; root README
   rework; `MODEL_README.md.j2` / `MODEL_BENCHMARK.md.j2`; per-model outputs in
   `configs/…` (+ CI reproduction check).
4. Site v2 (shell + manifest + lazy data + filters + compare).
5. Config overlays (`platforms/<platform>.yaml`) + overlay merge in `serve`.
6. Tests: two models same config → separate series; two platforms same model →
   separate series; pairing refuses cross-(platform, model); legacy rows
   (platform-less) resolve `thor`; unknown-arch error message.
7. AGENTS.md + docs/README.md index + `docs/faq/` task guides updated.

**Exit:** the site and configs/ pages already render Qwen3.8 (Thor) and
Qwen3.6 (Thor) as separate pairs; nothing about the kernel changed.

### Phase 1 — DGX Spark (`sm_120/121`)

1. Probe matrix (research HW): clusters on/off, TMEM columns, SMEM per SM,
   fp8/fp6 `tcgen05` shapes → platform constants + a probe report doc.
2. `arch/spark.py` — port of the Thor template with Spark constants; correctness
   vs `emulate_decode` first, then `verify sweep` vs FlashInfer/FA2 on Spark.
3. `docker/vllm-spark/` (base-image probe: official arm64 vLLM image on GB10?),
   canaries, `mjolnir hw setup` variant.
4. First committed Spark raws → the site's Spark tab lights up.

### Phase 2 — Jetson Orin (`sm_87`)

1. `arch/orin.py` — Ampere family (bf16 KV first; fp8/6 explicitly unsupported
   there — document why in `docs/faq/orin-fp8.md`).
2. `docker/vllm-orin/` — JetPack 6 base; pin the vLLM version Orin can actually
   build (likely ≠ 0.32.0); `Platform.vllm_version` per platform absorbs this.
3. Verify + first Orin raws; expectations set as bandwidth-bound wins.

### Phase 3 — Spark clusters

`mjolnir bench cluster` + per-host gating + scaling charts + docs.

---

## 12. Risks & open questions

| Risk | Mitigation |
|---|---|
| `sm_120/121` cluster support unknown → the 2-CTA plan may be unusable | probe first (Phase 1 step 1); plan already supports 1-CTA × N CTAs/SM (Thor runs it as the alternative) |
| vLLM on Orin: JetPack base + version pin is its own project; fp8 impossible | Phase 2 scoped to bf16 KV; per-platform `vllm_version`; treat the image as a deliverable with its own gate |
| DGX Spark base image availability (official arm64 vLLM image on GB10?) | Phase 1 probe; fallback = DGX OS CUDA 13 base + vLLM from source with the shared patch set |
| `history.jsonl` backfill touches a committed file | one dedicated migration commit, script is idempotent, CI reproduces renders after |
| `interactive.py` / `readme.py` near the file-length ratchet | split along natural seams **before** adding features (repo rule: separate commit) |
| CuTe DSL `sm_87` maturity (Ampere support in the DSL) | Phase 2 entry probe: compile a trivial `mma.sync` kernel on Orin before committing to the family design |
| Legacy Thor tag `narandill/mjolnir:v0.32.0` vs new per-platform repos | keep Thor names as-is; new repos are additive; `Platform.image_repo` removes the hard-coding |
| More platforms → more shared-GPU discipline | `gate.py` is host-agnostic already; per-host state files; cluster gate aggregates per-host cleanliness |

**Open questions for review:**
1. Image repo naming — one repo per platform (`mjolnir-spark`, `mjolnir-orin`)
   as proposed, or one repo with platform-prefixed tags?
2. Adopt the ticket's `benchmarks/results/…` tree as *additional generated
   output* for browsability, or stay metadata-only as proposed?
3. Orin scope: is bf16-KV-only acceptable for v1 of the Orin kernel, or is a
   software-dequant fp8 path worth exploring (likely a loss at 205 GB/s)?
4. Does `docs/kernel/` stay default-pair-only (proposed), or should the deep
   case study also split per platform eventually?
