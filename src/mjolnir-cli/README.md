# mjolnir-cli

The `mjolnir` command for the [Mjolnir](https://github.com/thor-arena/mjolnir) project on NVIDIA Jetson AGX Thor (`sm_110`).

```bash
uv tool install mjolnir-cli      # or: pipx install mjolnir-cli
mjolnir --help
```

| command | what it does |
|---|---|
| `mjolnir serve up` | serve a model from the Mjolnir vLLM image (`--attention-backend MJOLNIR`) |
| `mjolnir verify <stage>` | correctness matrix + ncu benchmarks of the Mjolnir kernel vs FlashInfer and FlashAttention 2 |
| `mjolnir plot --readme` | re-render every chart, table and the README from the raw benchmark data |
| `mjolnir autoresearch` | run the autonomous kernel-research loop (OpenCode agents, ledger-tracked) |

The distribution is named `mjolnir-cli`; the command it installs is `mjolnir`.
The kernel itself ships separately as [`mjolnir-kernel`](https://pypi.org/project/mjolnir-kernel/).
