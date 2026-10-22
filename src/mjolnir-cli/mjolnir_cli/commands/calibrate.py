"""Root command ``calibrate`` — fp6 KV scales / drafter vocabulary in a
one-shot container."""

from __future__ import annotations

from pathlib import Path

import typer

from mjolnir_cli import dockerctl
from mjolnir_cli.commands._common import settings


def calibrate(
    model: str | None = typer.Option(None, "--model", help="vendor/Model (default: the active one)"),
    config: str | None = typer.Option(None, "--config", help="the config to calibrate (default: the active one)"),
    image: str | None = typer.Option(None, "--image", help="the Mjolnir image"),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="default: the model's kv_scales/<format>.json next to the config dirs — kv_scales/<drafter>.kv_scales.json "
        "for a config whose own drafter (DSpark) runs on Mjolnir",
    ),
    prompts: int = typer.Option(48, "--prompts", help="short prompts"),
    long: int = typer.Option(6, "--long", help="long-context prompts"),
    max_model_len: int = typer.Option(32768, "--max-model-len"),
    draft_vocab: bool = typer.Option(
        False,
        "--draft-vocab",
        help="instead: count the target's generated tokens -> draft_vocab.json (VocabTrim, MJOLNIR_DRAFT_VOCAB)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the docker command, run nothing"),
):
    """Calibrate per-(layer, KV head, channel) K/V scales for the fp6 (e2m3) KV cache
    (or, with --draft-vocab, the drafter's vocabulary for VocabTrim).

    Runs the config's model on fp8 KV in the Mjolnir image, records the
    per-channel amax of every K/V written to the cache over a varied prompt set,
    and writes the per-format ``kv_scales/<format>.json`` next to the config —
    what an ``kv-cache-dtype: fp6_e2m3`` config is served with. Needs the GPU to
    itself: refuses while the vLLM server is up (stop it with
    ``mjolnir serve down`` in a window you own)."""
    s = settings(model, config, image, None)
    if not dry_run and dockerctl.container_running():
        typer.secho(
            "the vLLM server is running — calibration loads the model itself. Stop the server "
            "first (mjolnir serve down), in a window you own.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(3)
    target = dockerctl.calibrate_targets(s, out, draft_vocab)
    target.parent.mkdir(parents=True, exist_ok=True)
    mounts = [(str(target.parent.resolve()), "/out")]
    env = []
    try:  # a checkout's kernel package over the image's (calibrate.py may be newer)
        from mjolnir_cli.config import find_repo_root

        src = find_repo_root() / "src" / "mjolnir"
        if (src / "mjolnir" / ("draftvocab.py" if draft_vocab else "calibrate.py")).exists():
            mounts.append((str(src), "/opt/mjolnir-src:ro"))
            env.append("PYTHONPATH=/opt/mjolnir-src")
    except FileNotFoundError:
        pass
    cmd = (
        f"python3 -m mjolnir.{'draftvocab' if draft_vocab else 'calibrate'} "
        f"--config {dockerctl.config_model_dir(s)}/{s.quant}/config.yaml --out /out/{target.name} "
        f"--download-dir /data/models/huggingface --max-model-len {max_model_len} "
        f"--prompts {prompts} --long {long}"
    )
    argv = dockerctl.build_run_args(
        s,
        name="mjolnir-calibrate",
        detach=False,
        command=cmd,
        extra_env=tuple(env),
        extra_mounts=tuple(mounts),
        chown=(f"/out/{target.name}",),
    )
    if dry_run:
        typer.echo(" ".join(argv))
        return
    typer.secho(f"calibrating {s.model}/{s.quant} on {s.image} -> {target}", fg=typer.colors.CYAN)
    rc = dockerctl.run_once(argv, "mjolnir-calibrate")
    if rc == 0:
        typer.secho(f"wrote {target}", fg=typer.colors.GREEN)
    raise typer.Exit(rc)
