# Standup — CEO daily cycle

Full situational cycle. Dispatches specialists, synthesizes a Board Brief, records decisions. Run on schedule or on demand.

**Precondition**: this session is acting as the CEO (daemon session, or main session under CLAUDE.md).

## Step 1 — Situate

Read (cheap, sequential):

1. `.claude/state/charter.md` — if any `TODO` remains in mission / autonomy / escalation, **STOP**, print a one-line message telling the human which sections to fill, exit.
2. `.claude/state/risk-limits.json`.
3. Tail of `.claude/state/log.md` (last ~50 lines).
4. `ls .claude/state/proposals/*.md` and `ls .claude/state/incidents/*.md`; filter `status: open` from frontmatter.
5. Yesterday's brief if present under `docs/research/daily-briefs/` — note deltas.

**Halt rule**: if any P0 is open, scope the session to `/triage` for that incident. Don't continue standup.

## Step 2 — Dispatch (parallel)

Send these in ONE message as three Agent tool calls:

- `market-analyst`: "Run your standard pre-market protocol for symbols listed in `.claude/state/charter.md`. Save brief to `docs/research/market-briefs/`. Return a 5-bullet summary."
- `live-ops`: "Standard health snapshot. Save to `docs/research/ops-snapshots/`. Return severity + anomalies."
- `risk-officer`: "Live-monitor mode. Use `.claude/state/risk-limits.json` as the canonical thresholds. Return verdict + top 3 open risks."

Wait for all three.

## Step 3 — Synthesize

Write the brief to `docs/research/daily-briefs/YYYY-MM-DD-HHMM.md`:

```
# Board Brief — YYYY-MM-DD HH:MM UTC

## State
[green / yellow / red — one line]

## Market
[3 bullets]

## Health
[3 bullets]

## Risk
- Drawdown: X% (limit Y%, %-of-limit: Z%)
- Top risks: …

## Open queue
- Proposals: N open (ids, one-line asks, board_required flags)
- Incidents: N open (ids, severity, status)

## Decisions this cycle (autonomous)
- …

## Needs human approval
- …

## Deltas since yesterday
- …

## Next check-in
[when]
```

## Step 4 — Record

For each decision or escalation in the brief, append a section to `.claude/state/log.md`:

```
## YYYY-MM-DD HH:MM · decision|escalation · ceo
One-line summary.
Rationale: ...
Ref: proposals/... or incidents/...
```

If any item is `board_required: true`, fire the charter's notification webhook with the brief path.

## Guardrails

- Read-only except for the brief file and `log.md` appends.
- No kill-switch, no production deploys, no model-`latest` promotions in standup.
- If a subagent runs > 10 minutes or loops, abort and log a P2 incident "standup cost cap exceeded".
