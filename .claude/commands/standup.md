# Standup — CEO daily cycle

The daemon's primary workflow. Superset of `/daily-brief`: situational awareness + queue triage + synthesis + board brief. Run on schedule (e.g., every 6 hours) or on demand.

**Precondition**: this session is acting as the CEO. The daemon (or user) is the main session; `ceo.md` subagent is available but need not be invoked — the persona is inherited from `CLAUDE.md`.

## Step 1 — Situate (read-only)

Read, in parallel:

1. `.claude/state/charter.md` — reconfirm mandate. If TODOs remain, note it in the brief.
2. `.claude/state/risk-limits.json` — know the lines.
3. Last 20 lines of `.claude/state/decisions.jsonl`.
4. `ls .claude/state/proposals/open/` and `ls .claude/state/incidents/open/`.
5. `docs/project_status.md` (current focus).

**Guardrail**: if `charter.md` contains unfilled `TODO` markers for mission or autonomy envelope, STOP. Produce a one-paragraph message: "Cannot run standup — charter is incomplete. Please fill: [list of TODOs]." Exit.

**Guardrail**: if any P0 incident is in `incidents/open/`, STOP normal standup. Invoke `/triage` scoped to that incident. Standup does not continue until P0 is mitigated or explicitly acknowledged by the human.

## Step 2 — Dispatch specialists (parallel)

Send these in ONE message with multiple Agent tool calls:

- `market-analyst`: "Run your standard pre-market protocol for symbols listed in `charter.md`. Save to `docs/research/market-briefs/`. Return a 5-bullet summary AND append a call to `.claude/state/track-records/market-analyst.jsonl`."
- `live-ops`: "Standard health snapshot. Read `.claude/state/baselines.json` to know normal. Save to `docs/research/ops-snapshots/`. Return severity + anomalies AND append to your track record."
- `risk-officer`: "Live-monitor mode risk snapshot. Read `.claude/state/risk-limits.json` for thresholds. Return verdict + top 3 open risks AND append to your track record."

Wait for all three.

## Step 3 — Synthesize

Produce a **Board Brief** written to `docs/research/daily-briefs/YYYY-MM-DD-HHMM.md`:

```
# Board Brief — YYYY-MM-DD HH:MM UTC

## State
[green / yellow / red — one line]

## Market
[3 bullets]

## Bot health
[3 bullets]

## Risk
- Drawdown: X% (limit Y%, %-of-limit: Z%)
- Concentration: …
- Top risks: …

## Open queue
- Proposals: N open (list ids, one-line asks, board_required flag)
- Incidents: N open (list ids, severity, status)

## Decisions made this cycle (autonomous)
- [bullets; each becomes a decisions.jsonl entry]

## Recommendations needing human approval
- [bullets with rationale; each becomes a board_required decision]

## Open risks / follow-ups
- …

## Next check-in
[when]
```

## Step 4 — Write back to state

For every material item in "Decisions made" and "Recommendations needing approval":

```bash
# Append to decisions.jsonl (one line per decision)
echo '{"ts":"...","actor":"ceo","kind":"decision|escalation","ref":"...","summary":"...","rationale":"...","board_required":false|true}' >> .claude/state/decisions.jsonl
```

Also update `docs/project_status.md` if focus changed.

## Step 5 — Escalation

If any recommendation is `board_required: true`:
- The Board Brief already contains it.
- Do NOT execute the underlying action.
- If the charter specifies a notification webhook, fire it with the brief URL.

## Cost cap

This workflow should complete within one cycle. If it's been running > 10 minutes or a subagent is looping, abort and log a P2 incident: "standup exceeded cost cap".
