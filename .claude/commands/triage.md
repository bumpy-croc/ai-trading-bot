# Triage — sweep the open queues

Process every open proposal and incident. Unlike `/standup` which surveys the world, `/triage` clears the backlog.

## Step 1 — Enumerate

Read (no subagents yet):

- `ls .claude/state/proposals/open/` — open proposals
- `ls .claude/state/incidents/open/` — open incidents
- Parse frontmatter of each file (id, severity/priority, ages, required reviews pending)

Sort: P0 incidents first, then P1 incidents, then proposals by age descending.

## Step 2 — Per-incident

For each open incident:

1. Read the file.
2. **P0**: re-verify the incident is still active. If mitigated, gather post-mortem input (dispatch `live-ops` + the affected specialist) and move to `closed/`. If still active, ensure human is paged and freeze further triage work.
3. **P1**: dispatch `live-ops` to check current state. If mitigated and stable for > 2h, write post-mortem and close. Else, note and continue.
4. **P2/P3**: one-line update if something changed; otherwise skip.

## Step 3 — Per-proposal

For each open proposal, check what reviews are outstanding:

- `risk_review_required: true` and `risk_verdict: null` → dispatch `risk-officer` (stress-test mode) with the proposal file path.
- `code_review_required: true` and no review done → dispatch `code-reviewer` on the diff referenced.
- All required reviews done → CEO decides (see step 4).

Dispatch reviewers **in parallel** when they are independent.

## Step 4 — Decide

For each proposal with all reviews complete:

- **Any `reject` verdict from risk-officer** → move to `rejected/`, append a rejection rationale to the file, log to `decisions.jsonl`. Done.
- **All `approve` (possibly with conditions)** → examine `board_required`:
  - `false` → move to `approved/`, log decision, notify the owning agent to execute.
  - `true` → leave in `open/`, log an `escalation` entry in `decisions.jsonl`, include in next Board Brief.
- **Mixed / unclear** → explicitly surface the disagreement in the decision log with your rationale for the call. Do not flatten dissent.

## Step 5 — Housekeeping

- Remove stale draft files (any `open/` file > 14 days with no updates → move to `rejected/` with reason "timed out; re-submit if still relevant").
- Verify every file in `approved/executed/` (if present) has an `executed` timestamp.
- Check `risk-limits.json` mtime vs `src/config/constants.py` mtime. If constants.py is newer, dispatch `code-reviewer` with: "constants changed since last risk-limits sync — reconcile or flag P0 divergence."

## Output

Print a summary:

```
## Triage — YYYY-MM-DD HH:MM UTC

Processed: N incidents, M proposals
Actions:
- Approved: [ids]
- Rejected: [ids]
- Escalated to Board: [ids]
- Reviews dispatched: [ids]
- Post-mortems written: [ids]

Still pending: [ids with reason]
```

## Rules

- **Never approve your own proposals.** Although the CEO is the default decision-maker, if a proposal was authored by a subagent invoked from this same session, flag it for extra scrutiny — the freshness of context creates bias.
- **Don't mass-approve.** Each approval writes its own `decisions.jsonl` line. If the triage run has > 5 approvals, pause and produce a summary for the human before writing them.
