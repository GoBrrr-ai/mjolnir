# thinkingmachines/Inkling-2-Mini

`NVFP4_FP8KV` — NVFP4_FP8KV · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `thinkingmachines/Inkling-2-Mini` |
| Format | NVFP4_FP8KV |
| Max context | 200K |
| Max concurrency | 8 |
| Prefill chunk | 8K |
| KV cache | fp8_e4m3 |
| Spec decode | DSPARK ×7 |
| GPU memory util | 0.95 |
| Parsing | inkling / inkling |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config NVFP4_FP8KV
```
Availability: ✅ The Mjolnir kernel reads e4m3 KV natively on thor as its default dtype.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4_FP8KV).

## Benchmarks
### End to end (NVFP4_FP8KV, 1 request · 128K context)


| metric | `NVFP4_FP8KV` |
|---|--:|
| decode (t/s) | 34.9 |
| prefill (t/s) | 1,612 |
| TTFT (ms) | 82,752 |
| ITL (ms) | 28.8 |

<details>
<summary>Model quality (per suite)</summary>

| suite | score |
|---|--:|
| Tool calls | 86.2 |
| Hard Mode | 85.7 |

<details>
<summary>How the % and × here are calculated</summary>

- **Score** = points ÷ maximum points × 100, each scenario 2 (pass), 1 (partial) or 0 (fail); GSM8K / MMLU = accuracy in %; IFEval = prompt-level accuracy in %.
- **(±)** = this row's score − the baseline row's score, in percentage **points**, not a relative change: 86.5 vs 88.0 is −1.5 points (a relative change would be −1.7%).
- **n** = runs averaged (`--trials`); **completion** = the share of scenarios the endpoint answered — compare rows only at equal completion.

</details><br />
</details>

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#-kernel), one chapter per verify stage.
