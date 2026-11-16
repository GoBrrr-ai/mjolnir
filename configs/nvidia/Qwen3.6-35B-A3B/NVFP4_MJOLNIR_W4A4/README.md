# nvidia/Qwen3.6-35B-A3B

`NVFP4_MJOLNIR_W4A4` — Mjolnir + W4A4 experts · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Served as | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Format | Mjolnir + W4A4 experts |
| Max context | 256K |
| Max concurrency | 1 |
| Prefill chunk | 16K |
| KV cache | fp8 · skip: sliding_window |
| Attention backend | MJOLNIR |
| Spec decode | MTP ×3 |
| GPU memory util | 0.45 |
| Parsing | qwen3_coder / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config NVFP4_MJOLNIR_W4A4
```
Availability: ✅ The Mjolnir kernel reads e4m3 KV natively on thor as its default dtype.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=NVFP4_MJOLNIR_W4A4).

## Benchmarks

<details>
<summary>Model quality (per suite)</summary>

| suite | score |
|---|--:|
| Tool calls | 90.9 |
| Hard Mode | 78.6 |

<details>
<summary>How the % and × here are calculated</summary>

- **Score** = points ÷ maximum points × 100, each scenario 2 (pass), 1 (partial) or 0 (fail); GSM8K / MMLU = accuracy in %; IFEval = prompt-level accuracy in %.
- **(±)** = this row's score − the baseline row's score, in percentage **points**, not a relative change: 86.5 vs 88.0 is −1.5 points (a relative change would be −1.7%).
- **n** = runs averaged (`--trials`); **completion** = the share of scenarios the endpoint answered — compare rows only at equal completion.

</details><br />
</details>

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#-kernel), one chapter per verify stage.
