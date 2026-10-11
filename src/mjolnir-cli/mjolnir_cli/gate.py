"""Reusable clean-window gate for Thor GPU benchmarks.

The Thor GPU is shared with the live vLLM server. A valid benchmark window
requires the vLLM request queue to be empty (no running, no waiting) for
several consecutive samples — the "clean window". This module is the single
source of truth for that gate; the kernel bench scripts carry inline copies
of the same logic.

The vLLM server is NEVER stopped or restarted to get a clean window — the
gate only waits for one to open on its own.

Metrics source: the vLLM server's Prometheus endpoint (``/metrics`` on its
serving port). We read ``vllm:num_requests_running`` and
``vllm:num_requests_waiting``; the window is clean when both are 0.0.
"""

from __future__ import annotations

import threading
import re
import time
import urllib.request

DEFAULT_POLL_S = 2.0
DEFAULT_CONFIRM = 6
DEFAULT_TIMEOUT_S = 3600

_RUNNING = "vllm:num_requests_running"
_WAITING = "vllm:num_requests_waiting"


class PreflightError(RuntimeError):
    """The vLLM metrics endpoint was unreachable (server not up / wrong port)."""


def server_load(metrics_url: str) -> tuple[float, float] | None:
    """Return ``(num_running, num_waiting)`` from the vLLM server, or ``None``
    if the endpoint is unreachable or the metrics are missing."""
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as r:
            text = r.read().decode()
    except Exception:  # noqa: BLE001
        return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        for key in (_RUNNING, _WAITING):
            if line.startswith(key + "{") or line.startswith(key + " "):
                out[key.split(":", 1)[1]] = float(line.rsplit(" ", 1)[1])
    if "num_requests_running" not in out or "num_requests_waiting" not in out:
        return None
    return (out["num_requests_running"], out["num_requests_waiting"])


_SPEC = {
    "vllm:spec_decode_num_drafts_total": "drafts",
    "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "accepted",
    "vllm:num_preemptions_total": "preemptions",
    "vllm:generation_tokens_total": "generation_tokens",
}
_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


def server_counters(metrics_url: str) -> dict | None:
    """Speculative-decoding / scheduler counters (summed over label sets), for
    before/after deltas around a bench. None if unreachable."""
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as r:
            text = r.read().decode()
    except Exception:  # noqa: BLE001
        return None
    out: dict = {"per_pos": {}}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        if name in _SPEC:
            out[_SPEC[name]] = out.get(_SPEC[name], 0.0) + val
        elif name == _POS:
            m = re.search(r'position="(\d+)"', line)
            if m:
                k = int(m.group(1))
                out["per_pos"][k] = out["per_pos"].get(k, 0.0) + val
    return out


def spec_delta(before: dict | None, after: dict | None) -> dict | None:
    """Acceptance stats between two :func:`server_counters` samples:
    mean acceptance length (1 + accepted / drafts: tokens per verify step),
    per-position acceptance rate, preemptions."""
    if not before or not after:
        return None
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in set(_SPEC.values())}
    if d.get("drafts", 0) <= 0:
        return {"preemptions": int(d.get("preemptions", 0))} if "preemptions" in after else None
    counts = {
        k: after["per_pos"].get(k, 0.0) - before.get("per_pos", {}).get(k, 0.0)
        for k in sorted(after.get("per_pos", {}))
    }
    pos = {k: v / d["drafts"] for k, v in counts.items()}
    return {
        "drafts": int(d["drafts"]),
        "draft_tokens": int(d.get("draft_tokens", 0)),
        "accepted": int(d.get("accepted", 0)),
        "acceptance_length": round(1 + d.get("accepted", 0) / d["drafts"], 3),
        "acceptance_rate": round(d.get("accepted", 0) / d["draft_tokens"], 4) if d.get("draft_tokens") else None,
        "per_pos": [round(v, 4) for v in pos.values()],
        "per_pos_accepted": [int(v) for v in counts.values()],  # raw counts, for pooling
        "preemptions": int(d.get("preemptions", 0)),
    }


def wait_for_idle(
    metrics_url: str,
    poll_s: float = DEFAULT_POLL_S,
    confirm: int = DEFAULT_CONFIRM,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> float | None:
    """Block until ``confirm`` consecutive samples read (0,0); return the
    ``time.time()`` of the sample that opened the window, or ``None`` on
    timeout."""
    t_start = time.perf_counter()
    streak = 0
    while time.perf_counter() - t_start < timeout_s:
        load = server_load(metrics_url)
        if load == (0.0, 0.0):
            streak += 1
            if streak >= confirm:
                return time.time()
        else:
            streak = 0
        time.sleep(poll_s)
    return None


class _Monitor(threading.Thread):
    """Background thread that records ``(time, load)`` samples."""

    def __init__(self, metrics_url: str, poll_s: float) -> None:
        super().__init__(daemon=True)
        self.metrics_url = metrics_url
        self.poll_s = poll_s
        self.trace: list[tuple[float, tuple[float, float] | None]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            self.trace.append((time.time(), server_load(self.metrics_url)))
            self._stop.wait(self.poll_s)

    def stop(self) -> None:
        self._stop.set()


class CleanWindow:
    """Context manager: open a gated clean window, then report whether it
    stayed clean for the whole body.

    ``open_ts`` is set once the window opens. ``clean`` is True iff every
    in-window sample read (0,0). Raises :class:`PreflightError` if the
    metrics endpoint is unreachable at start, :class:`TimeoutError` if the
    window never opens within ``timeout_s``.
    """

    def __init__(
        self,
        metrics_url: str,
        confirm: int = DEFAULT_CONFIRM,
        poll_s: float = DEFAULT_POLL_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.metrics_url = metrics_url
        self.confirm = confirm
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self.open_ts: float | None = None
        self.clean: bool | None = None
        self.dirty_samples: list = []

    def __enter__(self) -> "CleanWindow":
        self._mon = _Monitor(self.metrics_url, self.poll_s)
        self._mon.start()
        time.sleep(self.poll_s)
        if all(load is None for _, load in self._mon.trace):
            self._mon.stop()
            raise PreflightError(
                f"vLLM metrics endpoint {self.metrics_url} is unreachable — is the server up? (mjolnir serve up)"
            )
        self._streak = 0
        self._consumed = 0
        t_start = time.perf_counter()
        while True:
            if time.perf_counter() - t_start > self.timeout_s:
                self._mon.stop()
                raise TimeoutError(
                    f"timed out after {self.timeout_s:.0f}s waiting for {self.confirm} consecutive 0/0 samples"
                )
            self._drain()
            if self._streak >= self.confirm:
                self.open_ts = self._mon.trace[-1][0]
                return self
            time.sleep(1.0)

    def _drain(self) -> None:
        while self._consumed < len(self._mon.trace):
            _ts, load = self._mon.trace[self._consumed]
            self._consumed += 1
            self._streak = self._streak + 1 if load == (0.0, 0.0) else 0

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._mon.stop()
        self._drain()
        in_window = [(ts, load) for (ts, load) in self._mon.trace if self.open_ts is not None and ts >= self.open_ts]
        self.dirty_samples = [[ts, load] for (ts, load) in in_window if load is None or load != (0.0, 0.0)]
        self.clean = not self.dirty_samples
        return False

    def summary(self) -> dict:
        return {
            "metrics_url": self.metrics_url,
            "poll_s": self.poll_s,
            "confirm": self.confirm,
            "open_ts": self.open_ts,
            "clean": self.clean,
            "n_in_window": (
                sum(1 for ts, _ in self._mon.trace if ts >= self.open_ts) if self.open_ts is not None else 0
            ),
            "dirty_samples": self.dirty_samples,
        }
