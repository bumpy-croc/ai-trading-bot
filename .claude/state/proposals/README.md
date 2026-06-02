# Proposals

A proposal is a structured request for a material change, produced by a specialist agent and reviewed by the PM (and `risk-officer`, if required).

## Lifecycle

Flat directory. Status lives in frontmatter; files never move between subdirs.

```
status: open        (submitted, awaiting reviews)
  → risk-officer and/or code-reviewer fill their verdict sections
  → pm decides
status: approved    (board_required: false, or human approval recorded)
  → owner executes; adds `executed: YYYY-MM-DDThh:mm:ssZ` to frontmatter
status: rejected    (any reject verdict, or pm reject rationale appended)
```

The companion GitHub Issue tracks the card through `state:*` labels; the proposal file is the concrete artifact reviewed in the PR.

## File naming

`YYYY-MM-DD-NN-short-slug.md` — e.g., `2026-04-21-01-promote-btc-model-v4.md`. `NN` is a daily counter. The filename and `id` never change.

## Template

```markdown
---
id: 2026-04-21-01-promote-btc-model-v4
from: ml-engineer
to: pm
status: open
risk_review_required: true
risk_verdict: null         # null | approve | approve-with-conditions | reject
code_review_required: false
board_required: true       # if true, pm must escalate to human
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

### pm
(filled by pm; one of: approve / reject / escalate-to-board)
```

## Rules

- **Never edit a rejected proposal in-place** — submit a new proposal with a reference to the prior one.
- **`status` is the single authoritative field.** PM transitions it; no subdirs to keep in sync.
- **`board_required: true` blocks auto-execution.** PM produces the Board Brief; the human unblocks by commenting `approved` on the linked GitHub Issue (or setting `status: approved` in the file). PM records the transition in `log.md`.
