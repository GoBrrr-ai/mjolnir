# Working with platforms (thor / spark / orin)

**Story:** "I want to run the CLI on — or bench for — another GPU (DGX
Spark, Jetson Orin) without breaking the Thor workflow — and without
remembering a platform anywhere."

## Detection, not state

There is no `platform set` and no remembered platform state (the Cycle-1
state model is gone). Every command resolves its platform at invocation,
in one fixed order:

```
flag > $MJOLNIR_PLATFORM env > host detection
```

Why: a remembered state is how a stale platform silently mislabels every
record written after it. Detection names the host a run actually happened
on, and an explicit operator is never second-guessed — an env override
means zero probes run at all. Detection is cached per process: each
source is asked at most once.

There is deliberately no per-command `--platform` CLI option; the flag
layer is the programmatic `resolve_platform(cli_flag)` /
`resolve(cli_platform=)` parameter. `MJOLNIR_PLATFORM` is the user-facing
override (tests and CI use it to pin a fake host).

## The two-source rule

Detection asks four source classes and maps every raw value through a
marker table (name substrings, compute capability, board files, L4T
release majors):

| source | reads | in-container |
|---|---|---|
| `nvidia-smi` | GPU name + compute cap, queried **separately** (a combined query dies on drivers that don't know `compute_cap`) | yes — the one signal always visible in a container |
| `jetson-files` | `/proc/device-tree/{compatible,model}`, `/etc/nv_tegra_release` | host only — absence in a container is not evidence |
| `dmi` | `/sys/class/dmi/id/product_name`, `/etc/dgx-release` | yes |
| `nv_jetson_model` | `/tmp/nv_jetson_model` (the jetson-containers model copy) | yes — **corroboration only** |

Two trusted classes agreeing on a platform is `confirmed`; exactly one is
`single-source`. Names match by substring, never equality: the Tegra
driver appends ` (nvgpu)`, and Thor's device-tree model string differs
between factory and BSP firmware — match markers, not model strings.
NVIDIA's own normalization rule (`[_-]+` → space) makes `NVIDIA_DGX_Spark`
and an OEM `DGX Spark` map alike. The nvidia-smi CSV cross-checks the GPU
name against the compute capability; a contradiction is a hard error
naming both raw values — never a silent wrong-platform guess.

`/tmp/nv_jetson_model` is world-writable, so it is operator-controlled
(the security-audit LOW): it probes last and can only corroborate another
class. Alone it is provenance-only — it can neither confirm, contradict
nor steer detection.

## `mjolnir platform`

Read-only — the platform is detected, never stored:

```bash
mjolnir platform list        # the registry: arch, images, maturity
mjolnir platform show        # resolved platform + confidence + provenance
```

`show` on a real Thor (abridged):

```
platform: thor — Jetson AGX Thor (sm_110, supported)
confidence: confirmed
  nvidia-smi: NVIDIA Thor, 11.0
  jetson-files: nvidia,p4071-0000+… tegra264 …
  dmi: NVIDIA Jetson AGX Thor Developer Kit
gpu: NVIDIA Thor
image: narandill/mjolnir-vllm:v0.32.0
```

Under `MJOLNIR_PLATFORM=spark` the same command prints
`confidence: env-override` and `(no probes ran — env-override)`, and the
image default switches to the official one. `platform set` exits 2 ("No
such command") and writes nothing — not even a state file.

## What a platform changes

- **Image default** — `thor` defaults to `narandill/mjolnir-vllm`; a
  research platform defaults to upstream `vllm/vllm-openai` (there is no
  Mjolnir image for it yet). An explicit `--image` / `$MJOLNIR_IMAGE`
  always wins.
- **Legs** — `bench ab`, `bench accuracy`, `bench e2e` and the `--legs`
  verify stages (`sweep`, `hndsweep`, `profile`, `spec`,
  `kv-cache-size`) all re-resolve on a non-default platform — but only
  when the flag was left *unset* (the typer `ParameterSource.DEFAULT`
  sentinel): an explicit `--backends` / `--legs` / `--config` / `--also`
  that merely spells the default-platform table still wins. On a research
  platform only official-side baselines remain, filtered to configs that
  exist there (the overlay rule, below); `bench e2e`'s Mjolnir side
  resolves to no leg. `verify bench` / `verify all` keep stage-default
  semantics, so `is_default` stays per-platform truthful.
- **Records** — `platform` is written from the resolved detection (`gpu`
  is the nvidia-smi name); platform-less legacy records resolve to
  `thor`.
- **Pairing** — `bench e2e` / `bench compare` never pair runs across a
  (platform, model) boundary.

## Config overlays

`configs/<vendor>/<model>/<quant>/platforms/<platform>.yaml` deep-merges
over the config's `config.yaml`: dicts merge recursively, **lists replace
wholesale**, scalars override; neither input is mutated.

```yaml
# configs/Qwen/Qwen3.8-27B/NVFP4/platforms/spark.yaml
{}   # an availability marker — the config exists on spark, no deltas
```

The absence rule: a config dir **without** a `platforms/` dir is a legacy
**thor-only** config (the pre-overlay layout — zero data churn); with the
dir, only the listed overlays exist. `available_serving_legs` honours it:
a config that does not exist on the platform is skipped, never guessed
into a leg. Today only Qwen3.8's `NVFP4` carries overlays (empty deltas —
its fp8 + FlashInfer pair is matrix-valid everywhere); the Mjolnir and
drafter configs are deliberately thor-only (fp6 has no fallback reader,
and the kernel is sm_110).

## The availability matrix

`kv-cache-dtype × attention-backend` is validated when a leg resolves and
at serve time (`serve up` gates the merged config after the dry-run
return, so dry-runs still print the plan). Backend classes come from the
config's `attention-backend` / `attention-config.backend` field; FA4 is
`FLASH_ATTN` with `flash_attn_version: 4`.

| kv-cache-dtype | thor | spark | orin |
|---|---|---|---|
| `fp6_e2m3` | MJOLNIR | — | — |
| `fp8` | FlashInfer, FA4, MJOLNIR | FlashInfer | FlashInfer |
| `nvfp4` | — | FA2 | FA2 |

A config with no `kv-cache-dtype`, an unknown dtype, or **no declared
attention backend** has nothing to contradict — the matrix validates
declared (dtype, backend) pairs only (with no backend declared, vLLM
auto-selects a servable one). A violation is actionable: it names the
platform, the key, the value and why:

```
spark: kv-cache-dtype 'fp6_e2m3' cannot be served with attention-backend
'FLASHINFER' — fp6 (e2m3) KV has no fallback reader — only the Mjolnir
kernel (MJOLNIR) on thor serves it
```

## One image, three platforms — host requirements

The target is ONE `narandill/mjolnir-vllm` image serving all three
platforms: arch resolved at runtime (the existing guard), Mjolnir
hard-disabled on sm_87/sm_121, Thor patches probe-gated, native baselines
always on the official image. Container compat follows the JP/L4T release
+ arch family, not the board — the cu130 arm64 track covers Thor + Orin
(sm_80 bin-compat → sm_87) and Spark (sm_120 → sm_121). The image
restructure itself is a later cycle; the host requirements are:

| platform | host needs |
|---|---|
| `thor` | Jetson AGX Thor, JetPack 7.x (the benchmark host runs L4T R38) |
| `orin` | JetPack **7.2.1+** (JP 7.2 unified the Jetson line: L4T 39.2.1, CUDA 13.2.2) |
| `spark` | DGX OS with **CUDA 13** |

## Unknown hosts

A host where every probe comes up empty is an explicit error, not a
silent thor guess. Commands that need a platform print the probe
instructions and the escape hatch on clean stderr (no traceback) and exit
2 via the central handler:

```
could not detect the platform on this host — probes that find nothing: nvidia-smi --query-gpu=name --format=csv,noheader (and --query-gpu=compute_cap separately); jetson-files (/proc/device-tree/compatible, /proc/device-tree/model, /etc/nv_tegra_release); dmi (/sys/class/dmi/id/product_name, /etc/dgx-release); nv_jetson_model (/tmp/nv_jetson_model — operator-controlled: corroborates another source class, never one alone). Set MJOLNIR_PLATFORM to one of: thor, spark, orin
```

`platform show` renders the same message and exits 1 (its own handler).
Two deliberate asymmetries: a bad *value* — a flag or env naming no
shipped platform — keeps its `ValueError` traceback, whose message names
the shipped platforms (`unknown platform 'bogus' — shipped platforms:
thor, spark, orin`); and `mjolnir version` never resolves a platform at
all, so it runs on any host.

## `mjolnir hw`

`hw setup` dispatches on the detected platform: the Jetson family (thor,
orin) runs the existing `scripts/hw/setup-thor.sh` — orin with an
explicit "untested on this platform" note — anything else refuses:

```
mjolnir hw setup is not implemented for spark yet — what exists today is
the Jetson setup (thor, orin; scripts/hw/setup-thor.sh).
```

exit 2. On an unknown host it prints the probe instructions (exit 2).

## What does not work yet

- **The MJOLNIR kernel only on thor** — no kernel exists for spark
  (`sm_121`) or orin (`sm_87`) (`PlatformNotSupported`); the vLLM builder
  degrades to the fallback backend instead. fp6 (`e2m3`) KV has no reader
  but the Mjolnir kernel, so it is thor-only by construction.
- **No per-platform image dirs** — the one-image restructure is a later
  cycle; until then research platforms default to the official image.
- **Per-pair rendering** — charts, README and the site render the default
  platform only; per-config `README.md` / `BENCHMARKS.md` and site
  filters are Cycle 3 of [the matrix design](../design/platform-model-matrix.md).
