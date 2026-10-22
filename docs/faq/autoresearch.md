# Running the autoresearch loop

**Story:** "I want agents to try kernel ideas overnight — with every attempt
recorded — and find the good ones in the morning."

The loop follows [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
agents change the Thor kernel family (`src/mjolnir/mjolnir/arch/thor/`), a fixed harness
scores it (geomean speedup over FlashInfer at the eight headline L, after a quick
correctness check), and the change is kept only if it beats the best score by
more than 2%. The full design is in [`research/README.md`](../../research/README.md);
the agents' instructions are [`research/program.md`](../../research/program.md).

## Prerequisites

- a Thor with the image pulled (`mjolnir image`), and a checkout of this repo;
- [OpenCode](https://opencode.ai) (`npm i -g opencode-ai`) with a model provider
  configured — the orchestrator lives in [`.opencode/agent/autoresearch.md`](../../.opencode/agent/autoresearch.md);
  the helpers it delegates to (`arxiv-researcher`, `web-researcher`,
  `python-engineer`, `debugger`) are optional agents you define yourself in
  `~/.config/opencode/agent/` (see [the team](../../research/README.md#the-team)).

## Run it

```bash
mjolnir autoresearch init oct10     # branch autoresearch/oct10, scores the baseline
mjolnir autoresearch run            # opencode run --agent autoresearch …
mjolnir autoresearch run --model anthropic/claude-opus-5-5   # pick the orchestrator's model
```

or inside an interactive `opencode` session: `/autoresearch`.

Score one experiment by hand (what the agents call each iteration):

```bash
git commit -am "3-stage ring for fp8 too"
mjolnir autoresearch evaluate -m "3-stage ring for fp8 too" --agent human
```

## In the morning

```bash
mjolnir autoresearch status             # best score, counts per status
mjolnir autoresearch ledger --tail 40   # every experiment, newest last
mjolnir autoresearch export             # research/runs/<tag>/ ledger + SUMMARY.md
```

The ledger (`research/ledger.jsonl`) is untracked while the run is live so a
`git reset --hard` on a discarded experiment never rewinds it; `export` makes the
committed copy. Each row links its raw verify runs, saved with
`"default": false` so experiments never touch the headline charts.

## Turning a win into a PR

Cherry-pick the kept commits onto a branch from `main`, run
`mjolnir verify bench` and `uv run mjolnir plot --readme`, and open a PR with
`research/runs/<tag>/` attached. Ledgers of discarded ideas are welcome too —
they tell the next person what not to try.
