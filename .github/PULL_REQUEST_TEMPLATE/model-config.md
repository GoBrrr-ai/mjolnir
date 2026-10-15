<!-- Use for new or changed serving configs. Full guide:
     docs/faq/new-model-config.md -->

## The config

- dir: `configs/<vendor>/<model>/<quant>/` (config.yaml + `.env` sidecar if it has env switches)
- `.env` sidecar included: yes / no
- calibration: `mjolnir calibrate` and/or `mjolnir calibrate --draft-vocab` — which committed files it produced
  (`kv_scales.json`, `draft_vocab.json` — next to the model's config dirs, or per-config for drafters)

## Verified

- [ ] `mjolnir serve up --dry-run` shows the config dir mounted
- [ ] serve smoke passed — first token sane (config **and** its calibration files as a pair)
- [ ] legs it was measured with: `mjolnir bench perf|e2e` / `mjolnir verify ...` — raw committed
- [ ] KV-cache fit: `mjolnir verify kv-cache-size` or the serve log attached

## Notes

<!-- what this config is for: default serving, an opt-in experiment leg, a drafter?
     If it should be measured by default, say so (that is a legs.py change). -->
