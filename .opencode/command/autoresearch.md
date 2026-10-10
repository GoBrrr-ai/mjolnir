---
description: Start (or resume) the Mjolnir autoresearch loop
agent: autoresearch
---

Read `research/program.md` and run the loop. Current state:

!`mjolnir autoresearch status`

!`mjolnir autoresearch ledger --tail 10`

If the ledger is empty, stop and tell the human to run `mjolnir autoresearch init <tag>`
first. Otherwise continue from the last row and never stop to ask. $ARGUMENTS
