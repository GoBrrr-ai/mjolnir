# nvidia/Qwen3.8-Flash-Next

`FP4LOCAL` — QSA + FP4-Local (MXFP8 side layers) + FP8 hyperconnections · serving profile. Flags live in
[config.yaml](config.yaml), Mjolnir env switches in `.env`; every number on
this page derives from the committed raw data ([history](../../../../benchmarks/history.jsonl), [raw](../../../../benchmarks/raw/)). The model's [README](../README.md).

## Serving profile

| parameter | Thor |
|---|---|
| Checkpoint | `nvidia/Qwen3.8-Flash-Next-FP4-Local` |
| Format | QSA + FP4-Local (MXFP8 side layers) + FP8 hyperconnections |
| Max context | 256K |
| Max concurrency | 1 |
| Prefill chunk | 8K |
| KV cache | auto |
| Spec decode | MTP ×3 |
| GPU memory util | 0.75 |
| Parsing | qwen3_xml / qwen3 |

## Serve

**Jetson AGX Thor**
```bash
mjolnir serve up --config FP4LOCAL
```
Availability: ✅ BF16 (and fp16) KV is the unquantized baseline: every backend class serves it wherever that backend itself runs.

Explore the numbers in [the interactive charts](https://GoBrrr-ai.github.io/mjolnir/#config=FP4LOCAL).

## Benchmarks
### End to end (FP4LOCAL, 1 request · 128K context)


| metric | `FP4LOCAL` |
|---|--:|
| decode (t/s) | 46.7 |
| prefill (t/s) | 3,186 |
| TTFT (ms) | 38,131 |
| ITL (ms) | 21.7 |

<details>
<summary>Model quality (per suite)</summary>

| suite | score |
|---|--:|
| Tool calls | 90.9 |
| Hard Mode | 86.6 |

<details>
<summary>How the % and × here are calculated</summary>

- **Score** = points ÷ maximum points × 100, each scenario 2 (pass), 1 (partial) or 0 (fail); GSM8K / MMLU = accuracy in %; IFEval = prompt-level accuracy in %.
- **(±)** = this row's score − the baseline row's score, in percentage **points**, not a relative change: 86.5 vs 88.0 is −1.5 points (a relative change would be −1.7%).
- **n** = runs averaged (`--trials`); **completion** = the share of scenarios the endpoint answered — compare rows only at equal completion.

</details><br />
</details>

The per-stage kernel numbers — findings, charts and raw provenance — live in the [Jetson AGX Thor kernel case study](../../../../docs/platforms/thor/README.md#-kernel), one chapter per verify stage.
