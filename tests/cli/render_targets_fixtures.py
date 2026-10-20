"""A two-model fixture repo for the cycle-3b render-target tests.

TEST DATA ONLY — everything is built under a ``tmp_path`` repo; the real
``docs/`` tree is only ever *read* (templates are copied out of it, guarded so
the RED state — before ``docs/templates/`` exists — builds a template-less
repo and every render contract fails loud). The magnitudes follow the shape of
real Thor runs (``verify_fixtures`` provides the kernel raws, stamped
``host: "fixture"``); history / accuracy records carry explicit ``platform`` /
``model`` fields per the cycle-1 schema.

Layout (plan §2: 2 models × multi configs × overlays × platforms):

    Qwen/Qwen3.8-27B/
      NVFP4_MJOLNIR_E2M3_LMH8_VT64K/  legacy thor-only (no platforms/), .env sidecar
      NVFP4/                          overlays {thor, spark}
      AWQ_FLASHINFER/                 overlay {orin} — the orin-only config
    nvidia/Qwen3.6-35B-A3B/
      NVFP4_MJOLNIR_E2M3/             overlay {thor}
      NVFP4_FA2/                      legacy thor-only — nvfp4 KV on FA2: the
                                      exists-but-unavailable ❌ cell
      NVFP4_FA4/                      legacy thor-only — FA2's @4 sibling:
                                      FLASH_ATTN + flash_attn_version 4 (the
                                      only input where the old inline backend
                                      derivation and overlays.backend_class
                                      disagree — the review-round-2 pin)

Data: the pinned (thor, Qwen3.8-27B) e2e A/B pair (+30.8% decode at 1 request)
plus a spark pair (+28.9%) the root headline must ignore; verify raws for
(thor, Qwen3.8-27B) only — spec + kv-cache-size docs feed the merged config
pages' per-platform sections; one accuracy run with two legs. A v2
chart manifest (+ dummy PNGs) is seeded so pages' ``chart()`` references
resolve without matplotlib.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from mjolnir_cli.config import DEFAULT_MODEL
from mjolnir_cli.verify import results, stages

import verify_fixtures

REPO = Path(__file__).resolve().parents[2]

M38 = DEFAULT_MODEL  # the pinned thor headline model
M36 = "nvidia/Qwen3.6-35B-A3B"
MJOR_IMAGE = "narandill/mjolnir-vllm:v0.32.0"
OFFICIAL_IMAGE = "vllm/vllm-openai:v0.32.0-ubuntu2404"
SPARK_IMAGE = "narandill/mjolnir-spark:v0.1.0"  # a future spark image: is_mjolnir

FP6_QUANT = "NVFP4_MJOLNIR_E2M3_LMH8_VT64K"  # == DEFAULT_QUANT: the pinned A/B side

# thor vs spark A/B decode gains at 1 request (the root-headline pin)
THOR_GAIN = "+30.8%"  # 52.7 / 40.3
SPARK_GAIN = "+28.9%"  # 58.0 / 45.0 — must never reach the root headline

# quant -> (overlay platforms (None = legacy thor-only), config.yaml body, .env lines)
CONFIGS: dict[str, dict[str, tuple[tuple[str, ...] | None, dict, dict[str, str]]]] = {
    M38: {
        FP6_QUANT: (
            None,
            {
                "model": "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4",
                "served-model-name": M38,
                "kv-cache-dtype": "fp6_e2m3",
                "kv-cache-dtype-skip-layers": "sliding_window",
                "attention-backend": "MJOLNIR",
                "max-model-len": "256K",
                "max-num-seqs": 8,
                "max-num-batched-tokens": "16K",
                "gpu-memory-utilization": 0.80,
                "tool-call-parser": "qwen3_coder",
                "reasoning-parser": "qwen3",
                "speculative-config": {"method": "mtp", "num_speculative_tokens": 5},
                # the dict-valued / template keys the README must NEVER show
                # (cycle 3e: they live in config.yaml, one click away)
                "compilation-config": {"cudagraph_num_of_warmups": 5},
                "override-generation-config": {"temperature": 1.0},
                "default-chat-template-kwargs": {"enable_thinking": True},
                "chat-template": "/configs/Qwen/Qwen3.8_chat_template.jinja",
            },
            {"MJOLNIR_FP8_LM_HEAD": "1", "MJOLNIR_DRAFT_VOCAB": "65536"},
        ),
        "NVFP4": (
            ("thor", "spark"),
            {
                "model": "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4",
                "served-model-name": M38,
                "kv-cache-dtype": "fp8",
                "attention-backend": "FLASHINFER",
                "max-model-len": "200K",
                "gpu-memory-utilization": 0.80,
            },
            {},
        ),
        "AWQ_FLASHINFER": (
            ("orin",),
            {
                "model": "Qwen/Qwen3.8-27B-AWQ",
                "served-model-name": M38,
                "kv-cache-dtype": "auto",
                "attention-backend": "FLASHINFER",
            },
            {},
        ),
    },
    M36: {
        "NVFP4_MJOLNIR_E2M3": (
            ("thor",),
            {
                "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
                "served-model-name": "nvidia/Qwen3.6-35B-A3B-NVFP4",
                "kv-cache-dtype": "fp6_e2m3",
                "attention-backend": "MJOLNIR",
                # int on purpose: pins the L_label path (the real Qwen3.6 shape)
                "max-model-len": 262144,
            },
            {},
        ),
        "NVFP4_FA2": (
            None,
            {
                "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
                "served-model-name": "nvidia/Qwen3.6-35B-A3B-NVFP4",
                "kv-cache-dtype": "nvfp4",
                "attention-backend": "FLASH_ATTN",
                "attention-config": {"flash_attn_version": 2},
            },
            {},
        ),
        # NVFP4_FA2's @4 sibling — byte-identical body except the one key the
        # two backend derivations disagree on (review cycle-3e round 2):
        # the old inline two-liner renders plain FLASH_ATTN, the canonical
        # overlays.backend_class renders FLASH_ATTN@4.
        "NVFP4_FA4": (
            None,
            {
                "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
                "served-model-name": "nvidia/Qwen3.6-35B-A3B-NVFP4",
                "kv-cache-dtype": "nvfp4",
                "attention-backend": "FLASH_ATTN",
                "attention-config": {"flash_attn_version": 4},
            },
            {},
        ),
    },
}

# the spark overlay's observable delta (the config README's per-platform
# "which file contributed what" contract)
SPARK_DELTA = {"max-model-len": "128K", "gpu-memory-utilization": 0.85}

OVERVIEW = (
    "Thor is the platform this repo is built for: the kernel, the image and the\n"
    "calibration flow all target its 20 SMs and 273 GB/s of LPDDR5X.\n"
)

# charts the seeded v2 manifest offers (identity = the pinned thor pair);
# chart() resolves them without matplotlib ever being imported. e2e-decode
# rides along from cycle 3f: the merged README's open A/B chart.
SEEDED_CHARTS = {
    "kernel-speedup": "sweep",
    "kernel-latency": "sweep",
    "ring-heatmap": "ring",
    "spec-drafts": "spec",
    "kv-cache-capacity": "kv-cache-size",
    "e2e-decode": "e2e",
}


def _cell(conc: int, ctx: int, tg: float, pp: float, ttft: float, itl: float) -> dict:
    return {
        "concurrency": conc,
        "context": ctx,
        "tg_tps": {"mean": tg},
        "pp_tps": {"mean": pp},
        "ttft_ms": {"mean": ttft},
        "itl_ms": {"mean": itl},
    }


def _rec(image: str, config: str, epoch: int, cells: list, model: str, platform: str) -> dict:
    return {
        "model": model,
        "image": image,
        "config": config,
        "epoch": epoch,
        "platform": platform,
        "gpu": "fixture",
        "backend": "x",
        "cells": cells,
    }


def _official_cells() -> list:
    return [
        _cell(1, 0, 40.3, 9000.0, 320.0, 24.8),
        _cell(1, 8192, 40.3, 9000.0, 320.0, 24.8),
        _cell(8, 8192, 322.4, 9000.0, 320.0, 24.8),
    ]


def _mjolnir_cells() -> list:
    return [
        _cell(1, 0, 52.7, 9600.0, 280.0, 19.0),
        _cell(1, 8192, 52.7, 9600.0, 280.0, 19.0),
        _cell(8, 8192, 421.6, 9600.0, 280.0, 19.0),
    ]


def history_records() -> list[dict]:
    """The pinned (thor, Qwen3.8) A/B pair + a spark pair whose gain must
    never leak into the root headline (the pin is (thor, Qwen3.8-27B)), and
    — cycle 3f — a solo NVFP4_FA2 run (own rows, no official-image partner:
    the merged README's solo end-to-end state)."""
    return [
        _rec(OFFICIAL_IMAGE, "NVFP4", 10, _official_cells(), M38, "thor"),
        _rec(MJOR_IMAGE, FP6_QUANT, 11, _mjolnir_cells(), M38, "thor"),
        _rec(OFFICIAL_IMAGE, "NVFP4", 20, [_cell(1, 8192, 45.0, 8000.0, 300.0, 22.2)], M38, "spark"),
        _rec(SPARK_IMAGE, "NVFP4_MJOLNIR", 21, [_cell(1, 8192, 58.0, 8600.0, 260.0, 17.2)], M38, "spark"),
        _rec(OFFICIAL_IMAGE, "NVFP4_FA2", 5, [_cell(1, 8192, 44.0, 8800.0, 310.0, 21.5)], M36, "thor"),
    ]


def _spec_doc() -> dict:
    d = results.header(
        "spec",
        "20261011-090000",
        image="fixture",
        default=True,
        params={"drafts": [1, 2, 3, 4]},
        kernel={"version": "0.1.0", "git": "abc1234"},
        gate={"server": "up"},
        platform="thor",
    )
    d["host"] = "fixture"
    d["model"] = M38
    d["rows"] = [
        {"config": FP6_QUANT, "concurrency": 1, "k": 3, "decode_tps": 45.2, "acceptance_length": 3.41},
        {"config": FP6_QUANT, "concurrency": 1, "k": 4, "decode_tps": 46.9, "acceptance_length": 4.05},
        {"config": "NVFP4", "concurrency": 1, "k": 3, "decode_tps": 41.0, "acceptance_length": 3.05},
    ]
    d["summary"] = stages.summarize("spec", d)
    return d


def _kvcache_doc() -> dict:
    d = results.header(
        "kv-cache-size",
        "20261011-093000",
        image="fixture",
        default=True,
        params={},
        kernel={"version": "0.1.0", "git": "abc1234"},
        gate={"server": "up"},
        platform="thor",
    )
    d["host"] = "fixture"
    d["model"] = M38
    d["rows"] = [
        {
            "id": "official-fp8",
            "side": "official",
            "kv_dtype": "fp8",
            "kv_tokens": 328_000,
            "max_concurrency": 9.8,
            "max_model_len": 262_144,
        },
        {
            "id": "mjolnir-fp6",
            "side": "mjolnir",
            "config": FP6_QUANT,
            "kv_dtype": "fp6_e2m3",
            "kv_tokens": 410_000,
            "max_concurrency": 12.2,
            "max_model_len": 262_144,
        },
    ]
    d["summary"] = stages.summarize("kv-cache-size", d)
    return d


def _accuracy_doc() -> dict:
    def leg(image: str, config: str, kv: str, tools: float, hard: float) -> dict:
        return {
            "id": f"{'mjolnir' if image.startswith('narandill') else 'official'}-{config}",
            "model": M38,
            "image": image,
            "config": config,
            "kv_cache_dtype": kv,
            "suites": {"tools": {"score": tools, "n": 69}, "hardmode": {"score": hard, "n": 28}},
        }

    return {
        "format": "mjolnir.accuracy/1",
        "run_id": "20261012-100000",
        "created": "2026-10-12T10:00:00Z",
        "tool": {"repo": "https://github.com/SeraphimSerapis/tool-eval-bench", "commit": "b60e2f0"},
        "args": ["run", "--hardmode", "--seed", "42", "--timeout", "600"],
        "legs": [
            leg(OFFICIAL_IMAGE, "NVFP4", "fp8", 0.86, 0.55),
            leg(MJOR_IMAGE, FP6_QUANT, "fp6_e2m3", 0.855, 0.54),
        ],
    }


def _patch_verify_raws(verify_dir: Path) -> None:
    """verify_fixtures' docs predate the platform/model fields: stamp the
    (thor, Qwen3.8-27B) identity onto every saved document."""
    for p in verify_dir.glob("*/*.json"):
        doc = json.loads(p.read_text())
        doc["platform"] = "thor"
        doc["model"] = M38
        p.write_text(json.dumps(doc, indent=1) + "\n")


def _seed_charts(root: Path) -> None:
    """A v2 manifest + dummy PNGs for the charts the pages reference — the
    reader side of manifest v2, no matplotlib needed. Entries carry a single
    ``dark`` path."""
    manifest = {}
    for name, stage in SEEDED_CHARTS.items():
        dark = f"assets/benchmarks/thor/Qwen3.8-27B/{name}.png"
        p = root / dark
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x89PNG fixture")
        manifest[name] = {"stage": stage, "platform": "thor", "model": M38, "dark": dark}
    (root / "assets" / "benchmarks" / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (root / "assets" / "hero.png").write_bytes(b"\x89PNG fixture")


def _copy_templates(root: Path) -> None:
    """The template home (docs/templates/ — the root README included, cycle
    3k) — guarded: RED builds without it."""
    src = REPO / "docs" / "templates"
    if src.is_dir():
        shutil.copytree(src, root / "docs" / "templates", ignore=shutil.ignore_patterns("*.md"))


def build_repo(tmp_path: Path, monkeypatch, *, with_data: bool = True, overview: bool = True) -> Path:
    """The fixture checkout: templates + configs (overlays, .env) + data.
    Hermetic by construction: MJOLNIR_CONFIGS_DIR points at an empty dir so
    the user-local config scan never leaks real machines into discovery."""
    root = tmp_path / "repo"
    (root / "docker" / "vllm-thor").mkdir(parents=True)
    for name in ("Dockerfile", "PATCHES.md", "README.md.j2"):
        shutil.copy(REPO / "docker" / "vllm-thor" / name, root / "docker" / "vllm-thor" / name)
    _copy_templates(root)
    gh = root / ".github"
    gh.mkdir()
    shutil.copy(REPO / ".github" / "CONTRIBUTING.md", gh / "CONTRIBUTING.md")  # chapters link it
    for pkg, init in (("mjolnir/mjolnir", "__init__.py"), ("mjolnir-cli/mjolnir_cli", "__init__.py")):
        (root / "src" / pkg).mkdir(parents=True)
        shutil.copy(REPO / "src" / pkg / init, root / "src" / pkg / init)
    # every repo file the generated pages link (the cross-linking contract
    # walks README.md + docs/platforms/** + configs/**): LICENSE/NOTICE, the
    # FAQ/methodology docs, the research index, SECURITY, the kernel source
    for rel in (
        "LICENSE",
        "NOTICE",
        ".github/SECURITY.md",
        "docs/README.md",
        "docs/faq/new-model-config.md",
        "docs/faq/default-model-quant.md",
        "docs/faq/serve-server.md",
        "docs/faq/settings-precedence.md",
        "docs/faq/fp6-kv.md",
        "docs/methodology/benchmarking.md",
        "research/README.md",
        "research/program.md",
        "src/mjolnir/mjolnir/t5_decode.py",
    ):
        src = REPO / rel
        if src.is_file():
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

    # the configs tree: quant dirs with platform overlays + .env sidecars
    for model, quants in CONFIGS.items():
        for quant, (overlays, body, env) in quants.items():
            d = root / "configs" / model / quant
            d.mkdir(parents=True)
            (d / "config.yaml").write_text(yaml.safe_dump(body, sort_keys=False))
            if overlays is not None:
                (d / "platforms").mkdir()
                for p in overlays:
                    delta = SPARK_DELTA if p == "spark" else {}
                    (d / "platforms" / f"{p}.yaml").write_text(yaml.safe_dump(delta, sort_keys=False))
            if env:
                (d / ".env").write_text("".join(f"{k}={v}\n" for k, v in env.items()))

    # a hand-written platform narrative fragment (inlined by PLATFORM_README)
    if overview:
        od = root / "docs" / "platforms" / "thor"
        od.mkdir(parents=True)
        (od / "overview.md").write_text(OVERVIEW)

    if with_data:
        _seed_charts(root)
    (root / "benchmarks").mkdir()
    if with_data:
        (root / "benchmarks" / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history_records()))
        verify_dir = root / "benchmarks" / "raw" / "verify"
        verify_fixtures.write_all(verify_dir)
        _patch_verify_raws(verify_dir)
        results.save(verify_dir, _spec_doc())
        results.save(verify_dir, _kvcache_doc())
        acc = root / "benchmarks" / "raw" / "accuracy" / "20261012-100000"
        acc.mkdir(parents=True)
        (acc / "accuracy.json").write_text(json.dumps(_accuracy_doc(), indent=1) + "\n")

    monkeypatch.setenv("MJOLNIR_REPO", str(root))
    monkeypatch.setenv("MJOLNIR_CONFIGS_DIR", str(tmp_path / "local-configs"))
    (tmp_path / "local-configs").mkdir(exist_ok=True)
    for v in ("MJOLNIR_BENCHMARKS", "MJOLNIR_HISTORY", "MJOLNIR_STATE"):
        monkeypatch.delenv(v, raising=False)
    return root
