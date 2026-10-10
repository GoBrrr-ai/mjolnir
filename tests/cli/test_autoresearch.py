"""The autoresearch ledger + the fixed keep/discard rule (the part of the loop
the agents cannot influence)."""

from __future__ import annotations

import json
import subprocess

from mjolnir_cli import autoresearch as ar


def test_decide_is_noise_aware():
    assert ar.decide(None, True, 1.2, 0.02) == "crash"
    assert ar.decide(1.5, False, 1.2, 0.02) == "discard"  # wrong answers are never kept
    assert ar.decide(1.2, True, None, 0.02) == "baseline"
    assert ar.decide(1.23, True, 1.2, 0.02) == "keep"
    assert ar.decide(1.22, True, 1.2, 0.02) == "discard"  # +1.7%: inside the required gain


def test_ledger_roundtrip_and_best(tmp_path):
    rows = [
        {"id": 1, "ts": "2026-10-10T10:00:00+00:00", "commit": "a", "status": "baseline", "score": 1.20},
        {"id": 2, "ts": "2026-10-10T10:20:00+00:00", "commit": "b", "status": "discard", "score": 1.30},
        {"id": 3, "ts": "2026-10-10T10:40:00+00:00", "commit": "c", "status": "keep", "score": 1.25},
    ]
    for r in rows:
        ar.append_ledger(tmp_path, {**r, "hypothesis": "h\twith tab", "agent": "x"})
    back = ar.load_ledger(tmp_path)
    assert [r["id"] for r in back] == [1, 2, 3]
    assert ar.best_kept(back)["commit"] == "c"  # a discarded higher score never becomes "best"
    tsv = (tmp_path / "research" / "ledger.tsv").read_text().splitlines()
    assert tsv[0].split("\t") == ar.TSV_COLUMNS and len(tsv) == 4
    assert "\t".join(["h with tab"]) in tsv[1]


def test_export_writes_summary(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", "-b", "autoresearch/t1", str(tmp_path)], check=True)
    (tmp_path / "docker" / "vllm-thor").mkdir(parents=True)
    (tmp_path / "docker" / "vllm-thor" / "Dockerfile").write_text("#")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "x"], check=True
    )
    monkeypatch.setenv("MJOLNIR_REPO", str(tmp_path))
    ar.append_ledger(
        tmp_path,
        {
            "id": 1,
            "ts": "t",
            "commit": "a",
            "status": "baseline",
            "score": 1.2,
            "hypothesis": "baseline",
            "agent": "human",
            "sources": [],
        },
    )
    ar.append_ledger(
        tmp_path,
        {
            "id": 2,
            "ts": "t",
            "commit": "b",
            "status": "keep",
            "score": 1.3,
            "hypothesis": "3-stage ring",
            "agent": "python-engineer",
            "sources": ["https://arxiv.org/abs/2407.08608"],
        },
    )
    ar.export(tag=None)
    out = tmp_path / "research" / "runs" / "t1"
    assert json.loads((out / "ledger.jsonl").read_text().splitlines()[1])["status"] == "keep"
    summary = (out / "SUMMARY.md").read_text()
    assert "best: **1.3×**" in summary and "3-stage ring" in summary
