# Triage — sweep open queues

Process every open proposal and incident. Unlike `/standup` which surveys the world, `/triage` clears the backlog.

## Step 1 — Enumerate

Read, no subagents yet:

- `.claude/state/incidents/*.md` — filter `status: open` from frontmatter.
- `.claude/state/proposals/*.md` — filter `status: open`.

Sort: P0 incidents → P1 incidents → proposals by age (oldest first).

## Step 2 — Per incident

- **P0**: verify still active. If mitigated and stable > 2h, dispatch the relevant specialist for a post-mortem, fill in the file's post-mortem section, set `status: closed`, append an `incident-close` entry to `log.md`. If still active, ensure human is paged and freeze further triage work.
- **P1**: dispatch `live-ops` to recheck state. If mitigated stable > 2h, write post-mortem, close.
- **P2 / P3**: one-line update if something changed; else skip.

## Step 3 — Per proposal

Check what's outstanding:

- `risk_verdict: null` and the proposal is live-affecting → dispatch `risk-officer` with a narrow prompt pointing to the proposal file.
- Code diff attached and no review → dispatch `code-reviewer`.
- All required reviews complete → PM decides (step 4).

Dispatch independent reviewers **in parallel** in one message.

## Step 4 — Decide

For each proposal with reviews complete:

- **Any `reject` verdict** → `status: rejected`, append rejection rationale to the file, append a `decision` entry to `log.md`.
- **All approve** and `board_required: false` → `status: approved`, notify the owning agent to execute, log decision.
- **All approve** and `board_required: true` → leave `status: open`, append an `escalation` entry to `log.md`, include in next brief.
- **Disagreement**: do not flatten. Log your rationale for going one way explicitly.

## Step 5 — Housekeeping

- Proposals `status: open` > 14 days with no updates → `status: rejected` with reason "timed out".
- Compare `mtime` on `.claude/state/risk-limits.json` vs `src/config/constants.py`. If constants is newer, open a P1 incident "risk-limits sync drift — reconcile".

## Output

```
## Triage — YYYY-MM-DD HH:MM UTC

Incidents processed: N
Proposals processed: M

Approved: [ids]
Rejected: [ids]
Escalated: [ids]
Reviews dispatched: [ids]
Still pending: [ids with reason]
```

## Guardrails

- **Never approve a proposal authored in this same session.** Flag it for extra scrutiny — the freshness of context creates bias.
- If > 5 approvals in one triage run, pause and summarize for the human before writing the decisions.
