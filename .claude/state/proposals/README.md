# Proposals

A proposal is a structured request for a material change, produced by a specialist agent and reviewed by the CEO (and `risk-officer`, if required).

## Lifecycle

```
[draft] --submit--> open/*.md
                     |
         review by   |  review by
         risk-officer|  code-reviewer
                     v
                    open/*.md (updated with verdicts)
                     |
              ceo-decide
                     |
        +------------+------------+
        v                         v
 approved/*.md              rejected/*.md
        |
   execute (by owner)
        |
        v
 approved/*.md (with `executed: YYYY-MM-DDThh:mm:ssZ` in frontmatter)
```

## File naming

`YYYY-MM-DD-NN-short-slug.md` — e.g., `2026-04-21-01-promote-btc-model-v4.md`. `NN` is a daily counter. The id stays the same when the file moves between directories.

## Template

```markdown
---
id: 2026-04-21-01-promote-btc-model-v4
from: ml-engineer
to: ceo
status: open
risk_review_required: true
risk_verdict: null         # null | approve | approve-with-conditions | reject
code_review_required: false
board_required: true       # if true, ceo must escalate to human
created: 2026-04-21T09:30:00Z
updated: 2026-04-21T09:30:00Z
---

## Ask

One-sentence summary of what's being proposed.

## Context

Why now? What problem does this solve?

## Proposed change

Concrete description. Files, configs, parameters, commands.

## Evidence

- Backtest results: `docs/research/experiments/...`
- Paper runtime: N hours since YYYY-MM-DD
- Metrics vs baseline: …

## How this could lose money

Adversarial self-review — at least 3 scenarios.

## Rollback plan

Exact steps and commands.

## Verdicts

### risk-officer
(filled by risk-officer; leave blank if review not yet run)

### code-reviewer
(filled by code-reviewer; leave blank if review not yet run)

### ceo
(filled by ceo; one of: approve / reject / escalate-to-board)
```

## Rules

- **Never edit a rejected proposal in-place** — submit a new proposal with a reference to the prior one.
- **`status` is the single authoritative field.** If frontmatter and directory disagree, frontmatter wins — and `ceo` should log a cleanup decision.
- **`board_required: true` blocks auto-execution.** The ceo produces a Board Brief; the human moves the file to `approved/` to unblock.
