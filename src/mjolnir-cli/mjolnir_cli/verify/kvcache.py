"""``mjolnir verify kv-cache-size`` — how many tokens of KV each config holds.

Serves every config in turn (the vLLM server is restarted per leg, like
``bench e2e``), waits for the engine to size its KV cache, and reads the
numbers vLLM logs at startup::

    Available KV cache memory: 38.12 GiB
    GPU KV cache size: 2,656,033 tokens
    Maximum concurrency for 204,800 tokens per request: 12.97x

The leg's server is stopped as soon as the numbers are in (no graph capture
or health wait needed). Charts turn the token count into the number of
concurrent requests each config fits at the e2e bench's request shapes.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from mjolnir_cli import dockerctl
from mjolnir_cli.config import DEFAULT_IMAGE, STOCK_IMAGE, config_field, resolve

# official image + stock backend, the Mjolnir image on fp8 KV, on fp6 KV
DEFAULT_LEGS = "official:NVFP4,mjolnir:NVFP4_MJOLNIR,mjolnir:NVFP4_MJOLNIR_E2M3"

_AVAIL = re.compile(r"Available KV cache memory:\s*([0-9.]+)\s*GiB")
_TOKENS = re.compile(r"GPU KV cache size:\s*([0-9,]+)\s*tokens")
_CONC = re.compile(r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:\s*([0-9.]+)x")
_FAILED = re.compile(r"EngineCore failed to start|Traceback \(most recent call last\)")

# the e2e bench's request: context depth + 2048 prompt + 128 generated tokens
E2E_PP, E2E_TG = 2048, 128


def parse_log(text: str) -> dict:
    """The KV numbers from a vLLM startup log (absent ones are None)."""
    out: dict = {"kv_gib": None, "kv_tokens": None, "max_model_len": None, "max_concurrency": None}
    m = _AVAIL.search(text)
    if m:
        out["kv_gib"] = float(m.group(1))
    m = _TOKENS.search(text)
    if m:
        out["kv_tokens"] = int(m.group(1).replace(",", ""))
    m = _CONC.search(text)
    if m:
        out["max_model_len"] = int(m.group(1).replace(",", ""))
        out["max_concurrency"] = float(m.group(2))
    return out


def fits(kv_tokens: int, tokens_per_request: int) -> int:
    """Whole requests of ``tokens_per_request`` that fit in ``kv_tokens``."""
    return kv_tokens // max(1, tokens_per_request)


def _legs(spec: str) -> list[tuple[str, str]]:
    out = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        side, _, quant = part.partition(":")
        if side not in ("official", "mjolnir") or not quant:
            raise ValueError(f"leg {part!r}: expected official:<config> or mjolnir:<config>")
        out.append((side, quant))
    return out


def _docker_logs() -> str:
    import subprocess

    p = subprocess.run([dockerctl._docker(), "logs", dockerctl.CONTAINER_NAME], capture_output=True, text=True)
    return dockerctl._strip_ansi((p.stdout or "") + (p.stderr or ""))


def _container_state() -> str:
    rc, out = dockerctl._run_quiet(
        [dockerctl._docker(), "inspect", "-f", "{{.State.Status}}", dockerctl.CONTAINER_NAME], ok_codes=(0, 1)
    )
    return out.strip() if rc == 0 else "absent"


def stage_kvcache(
    ctx,
    legs: str = DEFAULT_LEGS,
    model: str | None = None,
    restart: bool = False,
    timeout: float = 1800.0,
) -> dict:
    """Serve each leg, read its KV cache size, stop it."""
    rows, cases = [], []
    mj_image = ctx.runner.image or DEFAULT_IMAGE
    if dockerctl.container_running() and not restart and not ctx.runner.dry_run:
        ctx.out(
            "[red]   a vLLM server is running — this stage restarts the server for every leg. "
            "Stop it in a window you own (mjolnir serve down), or pass --restart.[/red]"
        )
        return {
            "rows": [],
            "cases": [{"label": "server busy (not restarted)", "passed": False}],
            "params": {"legs": legs},
        }
    (ctx.logdir / "kv-cache-size").mkdir(parents=True, exist_ok=True)
    for side, quant in _legs(legs):
        s = resolve(model, quant, STOCK_IMAGE if side == "official" else mj_image, None)
        leg_id = f"{side}-{quant}"
        log = ctx.logdir / "kv-cache-size" / f"{leg_id}.log"
        kv_dtype = config_field(s.config_path, "kv-cache-dtype") or "auto"
        row = {
            "id": leg_id,
            "side": side,
            "config": quant,
            "image": s.image,
            "model": s.model,
            "backend": s.attention_backend,
            "kv_dtype": kv_dtype,
            "gpu_memory_utilization": config_field(s.config_path, "gpu-memory-utilization"),
            "log": str(log),
        }
        ctx.out(f"   ▸ {leg_id}  ({s.image}, kv {kv_dtype})")
        if ctx.runner.dry_run:
            dockerctl.serve_up(s, wait=False, dry_run=True)
            continue
        if not s.config_path.exists():
            cases.append({"label": leg_id, "passed": False, "error": f"config not found: {s.config_path}"})
            continue
        stale = dockerctl.stale_image(s.image) if side == "mjolnir" else None
        if stale:
            ctx.out(f"[yellow]     warning: {stale}[/yellow]")
        t0 = time.time()
        text, err = "", None
        try:
            dockerctl.serve_up(s, wait=False)
            while time.time() - t0 < timeout:
                time.sleep(5)
                text = _docker_logs()
                if _CONC.search(text) and _TOKENS.search(text):
                    break
                if _FAILED.search(text) or _container_state() in ("exited", "dead", "absent"):
                    err = "the server failed before sizing its KV cache"
                    break
            else:
                err = f"no KV cache size after {timeout:.0f}s"
        except dockerctl.DockerError as e:
            err = str(e)
        finally:
            text = text or _docker_logs()
            Path(log).write_text(text)
            dockerctl.serve_down(s)
            # an engine that crashed leaves an exited container behind, whose
            # name would block the next leg's docker run
            dockerctl._run_quiet([dockerctl._docker(), "rm", "-f", dockerctl.CONTAINER_NAME], ok_codes=(0, 1))
        row.update(parse_log(text))
        row["startup_s"] = round(time.time() - t0, 1)
        ok = err is None and row["kv_tokens"] is not None
        rows.append(row)
        cases.append({"label": leg_id, "passed": ok, **({"error": err} if err else {}), "log": str(log)})
        if ok:
            ctx.out(
                f"     ✅ {row['kv_tokens']:,} tokens · {row['kv_gib'] or 0:.2f} GiB · "
                f"{row['max_concurrency']:.2f}x at {row['max_model_len']:,} tokens/request"
            )
        else:
            ctx.out(f"     ❌ {err or 'KV cache size not found in the log'} — {log}")
    return {"rows": rows, "cases": cases, "params": {"legs": legs, "model": model, "e2e_pp": E2E_PP, "e2e_tg": E2E_TG}}
