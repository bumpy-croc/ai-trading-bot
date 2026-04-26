# Board — agent operating conventions

Schema and label taxonomy live in `.claude/state/README.md` and `.claude/docs/project-setup.md`. This file is about the **behavior** every agent must follow when reading from or writing to the board.

The board is GitHub Issues + the "Agent PM Board" Project (`bumpy-croc/projects/6`). The card issue is the conversation thread; the linked PR (when one exists) is the diff under review.

## Transitions

Every state change on a card is a comment, then a label flip. Two-step so the comment exists at the moment of transition.

Comment template:

```
moving state:<old> → state:<new>
next action: <one sentence>
owner: @<owned-by-label>
wake at: <ISO timestamp or n/a>
```

Then update the labels on the issue: remove the old `state:*`, add the new one; update `owned-by:*`; add or clear `needs:*` as appropriate.

Never silently flip a label. The comment is what makes the transition auditable when scrolling through the issue later.

## WIP limits (PM enforces)

Hard caps. When a lane is full, no new card may enter that state until something clears.

| Lane | Cap | Why |
|---|---|---|
| `state:researching` | 3 | Compute and attention bound |
| `state:paper` | 2 | Real-time observation cost |
| `needs:human-approval` open | 2 | Don't drown the human |
| `state:building` per `area:*` | 1 | Avoid stomping on the same files |

If the PM's `/tick` finds a violation, it does not start new work; it drives one of the in-flight items toward completion or to `state:icebox` instead.

## What each specialist writes to a card

Each entry below: who owns the card, what comments and labels they post, what they hand off.

**`market-analyst`**
- Opens new cards with `state:idea + source:market-anomaly + area:*` when the brief surfaces something exploitable. Don't bury opportunities inside a brief no one will re-read.
- On existing cards: short comment with regime/sentiment context relevant to the card.
- Never sets `state:*` past `idea` — the proposing specialist does.

**`quant-researcher`**
- Owns cards with `area:strategy`, `area:backtest`. Carries `owned-by:quant-researcher` while researching.
- On finishing investigation: comment with experiment file link; if action warranted, opens the proposal PR and transitions card `state:researching → state:proposed`; sets `needs:risk-review` for any live-affecting change.
- Ready for review: hand off to `risk-officer` or `code-reviewer` by leaving the `needs:*` labels and switching `owned-by:*`.

**`ml-engineer`**
- Owns cards with `area:ml-model`. Same flow as quant.
- For promotions: opens a proposal PR, transitions card to `state:proposed`, applies `type:model-promotion + needs:risk-review`. Live-affecting → also `needs:human-approval`.

**`risk-officer`**
- Owns any card carrying `needs:risk-review`.
- Posts verdict as a PR review comment; on the card, comments the summary + removes `needs:risk-review`. On reject, leaves a recommendation that the PM transitions to `state:icebox`.
- Live-monitor mode: opens `type:incident` cards directly with the right `priority:*`. Pages human per charter for P0.

**`live-ops`**
- Opens `type:incident` cards on anomaly detection. Restarts paper-only services per its agent rules.
- Comments with the snapshot link; updates `state:*` only on incident closure (after post-mortem under `docs/post-mortems/`).

**`code-reviewer` / `architecture-reviewer`**
- Operate on PRs, not cards. Verdict is a PR review.
- On approve: clear `needs:code-review` from the linked card.
- On request-changes: leave the `needs:code-review` label; the proposer keeps `owned-by:*`.

**`pm`** (orchestrator / daemon session)
- Picks one card per `/tick`, advances it one step, ends with a transition comment.
- Synthesizes risk + code reviews into a card-level decision; transitions `state:proposed → state:building` (or `→ icebox`).
- Never clears `needs:human-approval`. Never merges a PR while that label is on the card.

## Wakeups

The Project's `Wake At` field is the human-facing schedule. The daemon also keeps a local mirror at `.claude/state/wakeups.jsonl` so `/tick` can answer "what's due now?" with one file read instead of scanning every issue.

Format — one JSON object per line:

```
{"card": "https://github.com/bumpy-croc/ai-trading-bot/issues/N", "wake_at": "2026-04-24T16:00:00Z", "reason": "paper window ends"}
```

Discipline:
- When a card sets a future `Wake At`, append a line here too.
- `/tick` first action: read this file, find lines where `wake_at <= now`, process those cards, then drop those lines (rewrite the file with the rest).
- The Project field is canonical for humans; this file is canonical for the daemon. They must not drift — if they do, the Project field wins and the daemon rewrites the file from the Project.

## Logging

After any card transition the daemon makes, append to `.claude/state/log.md`:

```
## YYYY-MM-DD HH:MM · decision · pm
Moved card #N to state:X.
Rationale: ...
Ref: github.com/bumpy-croc/ai-trading-bot/issues/N
```

Logs are for audit. The card on GitHub is the operational state.
