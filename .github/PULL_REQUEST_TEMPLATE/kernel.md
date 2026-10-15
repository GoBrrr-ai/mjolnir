<!-- Use for manual kernel changes: the loop is
     .opencode/skills/kernel-iteration/SKILL.md. The order is fixed:
     correctness first, then the stage that isolates the change, then a
     same-run A/B, then (only with a win) the default. -->

## The change

- harness flag (new code paths go behind a flag): `--...`
- files (usually only `src/mjolnir/mjolnir/t5_decode.py` — keep NVIDIA's BSD header):

## Evidence

- [ ] `mjolnir verify correctness` — full matrix passes (never change `emulate_decode` to make a test pass)
- [ ] the stage that isolates the change: `mjolnir verify <stage>` — raw committed
- [ ] same-run A/B vs the reference legs (Mjolnir ÷ FlashInfer ÷ FA2), legs measured in the **same run**
- [ ] differences under ±3% treated as noise; **L=16K was not used as the sole decision** (bimodal)
- [ ] default flipped: yes / no — if yes, a win across L, not a single point
- [ ] re-rendered: `uv run mjolnir plot --readme`

## Notes

<!-- what surprised you; negative results belong in the docs/kernel/<stage>.md.j2
     chapter ("open ideas" section) as well. -->
