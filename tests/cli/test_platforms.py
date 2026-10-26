"""The CLI-side platform registry and detection (``mjolnir_cli.platforms``).

Cycle-2 spec (plan §0.2 / §1, detection-first — no remembered state):

* ``Platform`` / ``REGISTRY`` / ``get()`` — unchanged from cycle 1: one
  frozen descriptor per shipped platform; ``thor`` byte-identical to the
  image constants; research rows fall back to the upstream image.
* Detection is source-class based: ``PlatformSource`` (``name`` +
  ``probe() -> str | None``, never raises) with ``DEFAULT_SOURCES`` =
  nvidia-smi, jetson-files, dmi and the ``/tmp/nv_jetson_model`` model copy
  (probed last). ``platform_for_raw`` maps every research raw value through
  the marker table (normalized substring match; the nvidia-smi CSV's compute
  capability cross-checks the name).
* ``detect(sources)`` -> ``Detection(platform, confidence, provenance)``:
  two source classes agreeing = ``confirmed``; one class =
  ``single-source``; no evidence = ``UnknownPlatform`` naming the probes;
  contradictory evidence = ``UnknownPlatform`` naming both raw values.
  ``MJOLNIR_PLATFORM`` beats detection (``env-override``, zero probes).
  Security-audit LOW (cycle 2): the model copy lives in world-writable
  ``/tmp`` — any local user can plant one — so its vote is
  corroboration-only: it confirms a platform only alongside another class
  and can never steer or contradict detection on its own.
* ``resolve_platform(cli_flag)`` — flag > env > detection; unknown flag /
  env values raise ``ValueError`` naming the shipped keys; an unknown
  host raises ``UnknownPlatform`` with the probe instructions. There is
  deliberately no silent fallback to thor.
* ``gpu_name()`` — unchanged: the GPU model string for records (never
  raises, ``""`` when there is none).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

import pytest

from mjolnir_cli import platforms
from platform_fixtures import JETSON_PATHS, FakeSource, fake_reader, fake_sources, fake_smi as _fake_smi, install

# Field-by-field contract per shipped platform. thor's values are today's
# config.py constants; spark/orin research rows fall back to the upstream
# image until an image of ours exists.
EXPECTED: dict[str, dict] = {
    "thor": {
        "name": "Jetson AGX Thor",
        "sm": (11, 0),
        "sm_family": "sm_110",
        "image_repo": "narandill/mjolnir-vllm",
        "default_image": "narandill/mjolnir-vllm:v0.32.0",
        "stock_image": "vllm/vllm-openai:v0.32.0-ubuntu2404",
        "vllm_version": "0.32.0",
        "status": "supported",
        "host_requirements": "JetPack 7.x",
    },
    "spark": {
        "name": "DGX Spark",
        "sm": (12, 1),
        "sm_family": "sm_121",
        "image_repo": "narandill/mjolnir-spark",
        "default_image": "vllm/vllm-openai:v0.32.0-ubuntu2404",
        "stock_image": "vllm/vllm-openai:v0.32.0-ubuntu2404",
        "vllm_version": None,
        "status": "research",
        "host_requirements": "DGX OS + CUDA 13",
    },
    "orin": {
        "name": "Jetson Orin",
        "sm": (8, 7),
        "sm_family": "sm_87",
        "image_repo": "narandill/mjolnir-vllm",
        "default_image": "narandill/mjolnir-vllm:v0.32.0",
        "stock_image": "vllm/vllm-openai:v0.32.0-ubuntu2404",
        "vllm_version": None,
        "status": "research",
        "host_requirements": "JetPack 7.2.1+ (L4T 39.2.1, CUDA 13.2)",
    },
}


@pytest.fixture(autouse=True)
def _detection_file_opts_out(monkeypatch):
    # this file exercises the detection path itself, so the suite's thor pin
    # (conftest) is cleared here; a test that wants an explicit platform sets
    # its own env value, and platform_fixtures.install() clears it as well
    monkeypatch.delenv("MJOLNIR_PLATFORM", raising=False)


def _smi_run(answers: dict[str, str], calls: list[list[str]]) -> Callable[..., subprocess.CompletedProcess]:
    """A fake subprocess.run answering nvidia-smi queries by argv substring."""

    def run(argv, **kw):
        calls.append(list(argv))
        joined = " ".join(argv)
        for key, out in answers.items():
            if key in joined:
                return subprocess.CompletedProcess([], 0, stdout=out)
        return subprocess.CompletedProcess([], 1, stdout="")

    return run


# ── 1. the registry (cycle-1 pins, unchanged) ───────────────────────────────


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_registry_resolves_every_shipped_platform(key):
    p = platforms.get(key)
    assert p.key == key
    for field, want in EXPECTED[key].items():
        assert getattr(p, field) == want, f"{key}.{field}"
    assert platforms.REGISTRY[key] is p  # the registry holds the same frozen records


def test_unknown_platform_key_names_the_shipped_platforms():
    with pytest.raises(KeyError) as ei:
        platforms.get("bogus")
    msg = str(ei.value)
    assert "bogus" in msg
    assert all(k in msg for k in ("thor", "spark", "orin"))


# ── 1b. host_requirements (cycle 3a — renderer-facing data, no behavior) ────


@pytest.mark.parametrize(
    ("key", "requirements"),
    [
        ("thor", "JetPack 7.x"),
        ("orin", "JetPack 7.2.1+ (L4T 39.2.1, CUDA 13.2)"),
        ("spark", "DGX OS + CUDA 13"),
    ],
)
def test_host_requirements_present_and_correct_for_every_platform(key, requirements):
    assert platforms.get(key).host_requirements == requirements


def test_host_requirements_is_a_documented_string_on_every_record():
    for key, record in platforms.REGISTRY.items():
        assert isinstance(record.host_requirements, str) and record.host_requirements, key


# ── 2. the sources ──────────────────────────────────────────────────────────


def test_default_sources_probe_the_tmp_model_copy_last():
    # the /tmp copy is its own source class (not part of jetson-files) and
    # probes last: it is operator-controlled (security audit LOW)
    assert [s.name for s in platforms.DEFAULT_SOURCES] == ["nvidia-smi", "jetson-files", "dmi", "nv_jetson_model"]


def test_nvidia_smi_source_parses_name_and_cc_into_the_raw_value():
    calls: list[list[str]] = []
    src = platforms.NvidiaSmiSource(
        run=_smi_run({"--query-gpu=name": "NVIDIA Thor\n", "--query-gpu=compute_cap": "11.0\n"}, calls)
    )
    assert src.name == "nvidia-smi"
    assert src.probe() == "NVIDIA Thor, 11.0"


def test_nvidia_smi_source_never_combines_the_name_and_cc_queries():
    # research rule 1: a combined `--query-gpu=name,compute_cap` dies entirely
    # on drivers that do not know the compute_cap field — the cc query stands alone
    calls: list[list[str]] = []
    src = platforms.NvidiaSmiSource(
        run=_smi_run({"--query-gpu=name": "NVIDIA Thor\n", "--query-gpu=compute_cap": "11.0\n"}, calls)
    )
    src.probe()
    assert calls, "the source never queried nvidia-smi"
    assert not any("name" in " ".join(a) and "compute_cap" in " ".join(a) for a in calls)


def test_nvidia_smi_source_degrades_when_the_cc_query_is_unsupported():
    # a driver that rejects compute_cap must cost the cc only, never the name
    src = platforms.NvidiaSmiSource(run=_smi_run({"--query-gpu=name": "Orin (nvgpu)\n"}, []))
    assert src.probe() == "Orin (nvgpu)"


def test_nvidia_smi_source_is_none_when_nvidia_smi_is_absent():
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    assert platforms.NvidiaSmiSource(run=boom).probe() is None  # never raises


def test_nvidia_smi_source_probes_at_most_once_per_instance():
    # the per-process cache: `platform show` shares one source between
    # detect() and gpu_name() — one nvidia-smi spawn, however many calls
    calls: list[list[str]] = []
    src = platforms.NvidiaSmiSource(
        run=_smi_run({"--query-gpu=name": "NVIDIA Thor\n", "--query-gpu=compute_cap": "11.0\n"}, calls)
    )
    assert src.probe() == "NVIDIA Thor, 11.0"
    assert src.probe() == "NVIDIA Thor, 11.0"
    assert len(calls) == 2  # the name query + the cc query, asked once each


def test_jetson_files_source_returns_the_readables_nul_stripped():
    asked: list[str] = []
    files = {
        "/proc/device-tree/compatible": "nvidia,p4071-0000+p3834-0008\0nvidia,tegra264\0",
        "/etc/nv_tegra_release": "# R38 (release), REVISION: 2.2\n",
    }
    src = platforms.JetsonFilesSource(read=fake_reader(files, asked))
    raw = src.probe()
    assert raw is not None and "tegra264" in raw and "R38" in raw
    assert "\0" not in raw  # device-tree strings are NUL-terminated; the raw value is not
    assert set(asked) == set(JETSON_PATHS)  # exactly the research probe files


def test_jetson_files_source_maps_the_host_fixtures():
    # factory firmware model string, BSP compatible and the L4T release line
    # all identify their platform through the table
    cases = [
        ({"/proc/device-tree/model": "NVIDIA Jetson Thor Developer Kit\0"}, "thor"),
        ({"/proc/device-tree/compatible": "nvidia,p3737-0000+p3701-0000 nvidia,tegra234\0"}, "orin"),
        ({"/etc/nv_tegra_release": "# R36 (release), REVISION: 5.0\n"}, "orin"),
        ({"/etc/nv_tegra_release": "# R39.2 (release), REVISION: 1.0\n"}, "thor"),
    ]
    for files, want in cases:
        src = platforms.JetsonFilesSource(read=fake_reader(files))
        assert platforms.platform_for_raw(src.probe()) == want, files


def test_the_model_file_source_probes_only_the_tmp_copy():
    # jetson-containers' model copy is its own source class now
    src = platforms.JetsonModelFileSource(
        read=fake_reader({"/tmp/nv_jetson_model": "NVIDIA Jetson AGX Thor Developer Kit"})
    )
    assert platforms.platform_for_raw(src.probe()) == "thor"
    assert platforms.JetsonModelFileSource(read=fake_reader({})).probe() is None


def test_a_planted_tmp_file_alone_is_not_evidence():
    # any local user can write the world-writable /tmp copy: alone it must not
    # steer detection — provenance only, the probe instructions are raised
    with pytest.raises(platforms.UnknownPlatform) as ei:
        platforms.detect(fake_sources(smi=None, jetson=None, dmi=None, model_file="DGX Spark"))
    msg = str(ei.value)
    assert "could not detect the platform" in msg and "MJOLNIR_PLATFORM" in msg


def test_a_planted_tmp_file_cannot_contradict_trusted_evidence():
    # the hint cannot force a contradiction DoS either: the trusted classes decide
    d = platforms.detect(fake_sources(smi="NVIDIA Thor, 11.0", model_file="DGX Spark"))
    assert d.platform == "thor"
    assert d.confidence == "single-source"  # the planted hint is no corroboration for another platform
    assert ("nv_jetson_model", "DGX Spark") in d.provenance  # shown, never acted on


def test_a_tmp_file_that_agrees_corroborates_to_confirmed():
    d = platforms.detect(fake_sources(smi=None, dmi="NVIDIA_DGX_Spark", model_file="DGX Spark"))
    assert d.platform == "spark"
    assert d.confidence == "confirmed"  # hint + trusted class = two agreeing sources
    assert ("nv_jetson_model", "DGX Spark") in d.provenance


def test_jetson_files_source_is_none_when_nothing_is_readable():
    src = platforms.JetsonFilesSource(read=fake_reader({}))
    assert src.probe() is None


def test_dmi_source_reads_product_name_then_dgx_release():
    assert (
        platforms.platform_for_raw(
            platforms.DmiSource(read=fake_reader({"/sys/class/dmi/id/product_name": "NVIDIA_DGX_Spark\n"})).probe()
        )
        == "spark"
    )
    dgx = {"/etc/dgx-release": 'DGX_NAME="DGX Spark"\nDGX_PRETTY_NAME="NVIDIA DGX Spark"\n'}
    assert platforms.platform_for_raw(platforms.DmiSource(read=fake_reader(dgx)).probe()) == "spark"
    # an OEM unit: the Lenovo ThinkStation PGX normalizes to the same marker
    assert (
        platforms.platform_for_raw(
            platforms.DmiSource(read=fake_reader({"/sys/class/dmi/id/product_name": "DGX Spark"})).probe()
        )
        == "spark"
    )
    assert platforms.DmiSource(read=fake_reader({})).probe() is None


# ── 3. the mapping table ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        # nvidia-smi CSV: the name substring decides, the cc confirms
        ("NVIDIA Thor, 11.0", "thor"),
        ("Thor (nvgpu), 11.0", "thor"),
        ("NVIDIA IGX Thor, 11.0", "thor"),
        ("Orin (nvgpu), 8.7", "orin"),
        ("Orin, 8.7", "orin"),
        ("NVIDIA GB10, 12.1", "spark"),
        ("NVIDIA GB10", "spark"),
        # device-tree: factory-vs-BSP Thor strings differ — match on markers, never exact
        ("NVIDIA Jetson Thor Developer Kit", "thor"),
        ("NVIDIA Jetson AGX Thor Developer Kit", "thor"),
        ("nvidia,p4071-0080+p3834-0008 nvidia,p3834-0008 nvidia,tegra264", "thor"),
        ("nvidia,p4071-0000+p3834-0008 nvidia,tegra264", "thor"),
        ("nvidia,p3737-0000+p3701-0000 nvidia,tegra234", "orin"),
        # L4T release lines: JetPack 7 (R38/R39) is Thor, R36 is Orin
        ("# R38 (release), REVISION: 2.2, GCID: 12345678", "thor"),
        ("# R39.2 (release), REVISION: 1.0", "thor"),
        ("# R36 (release), REVISION: 5.0, GCID: 1", "orin"),
        # DMI / dgx-release: [_-]+ normalizes to space (OEM units included)
        ("NVIDIA_DGX_Spark", "spark"),
        ("DGX Spark", "spark"),
        ('DGX_NAME="DGX Spark"', "spark"),
    ],
)
def test_the_mapping_table_maps_every_research_raw_value(raw, want):
    assert platforms.platform_for_raw(raw) == want


@pytest.mark.parametrize(
    "raw",
    ["", "Tesla H100, 9.0", "GH200", "DGX Server for KVM", "Virtual Machine"],
)
def test_the_mapping_table_returns_none_for_unmapped_raw_values(raw):
    assert platforms.platform_for_raw(raw) is None


def test_the_mapping_cross_checks_the_smi_name_against_the_cc():
    # a thor name with a spark compute capability is contradictory evidence,
    # never a silent guess: the error names both raw values
    with pytest.raises(platforms.UnknownPlatform) as ei:
        platforms.platform_for_raw("NVIDIA Thor, 12.1")
    msg = str(ei.value)
    assert "NVIDIA Thor" in msg and "12.1" in msg


# ── 4. detect() ─────────────────────────────────────────────────────────────


def test_detect_two_source_classes_agree_is_confirmed():
    srcs = fake_sources(smi="NVIDIA Thor, 11.0", jetson="nvidia,p4071-0000 nvidia,tegra264")
    d = platforms.detect(srcs)
    assert d.platform == "thor"
    assert d.confidence == "confirmed"
    assert d.provenance == (
        ("nvidia-smi", "NVIDIA Thor, 11.0"),
        ("jetson-files", "nvidia,p4071-0000 nvidia,tegra264"),
    )


def test_detect_one_source_class_is_single_source():
    d = platforms.detect(fake_sources(smi=None, dmi="NVIDIA_DGX_Spark"))
    assert d.platform == "spark"
    assert d.confidence == "single-source"
    assert d.provenance == (("dmi", "NVIDIA_DGX_Spark"),)


def test_detect_without_evidence_raises_unknown_platform_naming_the_probes():
    with pytest.raises(platforms.UnknownPlatform) as ei:
        platforms.detect(fake_sources(smi=None, jetson=None, dmi=None))
    msg = str(ei.value)
    assert "nvidia-smi" in msg  # the probe instructions name the probes...
    assert "MJOLNIR_PLATFORM" in msg  # ...and the escape hatch


def test_detect_disagreement_raises_naming_both_raw_values():
    with pytest.raises(platforms.UnknownPlatform) as ei:
        platforms.detect(fake_sources(smi="NVIDIA Thor, 11.0", dmi="NVIDIA_DGX_Spark"))
    msg = str(ei.value)
    assert "NVIDIA Thor, 11.0" in msg
    assert "NVIDIA_DGX_Spark" in msg


def test_detect_env_beats_detection_and_probes_nothing(monkeypatch):
    monkeypatch.setenv("MJOLNIR_PLATFORM", "spark")
    srcs = fake_sources(smi="NVIDIA Thor, 11.0")
    d = platforms.detect(srcs)
    assert d.platform == "spark"
    assert d.confidence == "env-override"
    assert all(s.calls == 0 for s in srcs)  # an explicit operator is never second-guessed


def test_detect_rejects_a_bad_env_value_naming_the_shipped_platforms(monkeypatch):
    monkeypatch.setenv("MJOLNIR_PLATFORM", "bogus")
    with pytest.raises(ValueError) as ei:
        platforms.detect(fake_sources())
    assert all(k in str(ei.value) for k in ("thor", "spark", "orin"))


def test_detect_uses_the_default_sources_when_none_are_given(monkeypatch):
    install(monkeypatch, smi="NVIDIA GB10, 12.1")
    assert platforms.detect().platform == "spark"


# ── 5. resolve_platform: flag > env > detection ─────────────────────────────


def test_resolve_platform_flag_beats_env_beats_detection(monkeypatch):
    install(monkeypatch, smi="NVIDIA Thor, 11.0")  # detection would say thor
    monkeypatch.setenv("MJOLNIR_PLATFORM", "spark")
    assert platforms.resolve_platform("orin") == "orin"  # the flag wins
    assert platforms.resolve_platform(None) == "spark"  # then the env
    monkeypatch.delenv("MJOLNIR_PLATFORM")
    assert platforms.resolve_platform(None) == "thor"  # then detection


def test_resolve_platform_rejects_an_unknown_flag_naming_the_shipped_platforms():
    with pytest.raises(ValueError) as ei:
        platforms.resolve_platform("bogus")
    assert all(k in str(ei.value) for k in ("thor", "spark", "orin"))


def test_resolve_platform_on_an_unknown_host_raises_unknown_platform(monkeypatch):
    install(monkeypatch, smi=None, jetson=None, dmi=None)  # a GPU-less CI host
    with pytest.raises(platforms.UnknownPlatform) as ei:
        platforms.resolve_platform(None)
    assert "nvidia-smi" in str(ei.value) and "MJOLNIR_PLATFORM" in str(ei.value)


# ── 6. record identity + the reference image (cycle-1 pins, unchanged) ──────


def test_platform_of_explicit_wins_legacy_resolves_thor():
    assert platforms.platform_of({"platform": "spark"}) == "spark"
    assert platforms.platform_of({"image": "img", "config": "c"}) == "thor"  # legacy record
    assert platforms.platform_of({"platform": "bogus"}) == "bogus"  # passes through unvalidated


def test_platform_of_is_orthogonal_to_model_of():
    # model-less is not platform-less: a record without a model still resolves one
    assert platforms.platform_of({"config": "c"}) == "thor"


def test_series_key_is_platform_model_config():
    assert platforms.series_key({"platform": "spark", "model": "m", "config": "c"}) == ("spark", "m", "c")
    assert platforms.series_key({"model": "m", "config": "c"}) == ("thor", "m", "c")  # legacy


def test_gpu_name_parses_the_nvidia_smi_csv(monkeypatch):
    _fake_smi(monkeypatch, "NVIDIA IGX Thor, 11.0\n")
    assert platforms.gpu_name() == "NVIDIA IGX Thor"


def test_gpu_name_is_empty_when_nvidia_smi_is_absent(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", boom)
    assert platforms.gpu_name() == ""  # never raises


def test_stock_image_env_override_wins():
    assert platforms.stock_image("thor", "ref:custom") == "ref:custom"


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_stock_image_default_is_the_record_stock_image(key):
    assert platforms.stock_image(key, None) == EXPECTED[key]["stock_image"]


def test_fake_sources_satisfy_the_source_protocol():
    # the test fakes are structural PlatformSource implementations
    for s in fake_sources():
        assert isinstance(s.name, str)
        assert callable(s.probe)
    assert FakeSource("x", None).probe() is None
