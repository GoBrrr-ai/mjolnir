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
