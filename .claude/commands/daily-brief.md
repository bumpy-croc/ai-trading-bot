# Daily Brief

Produce the morning situational brief for the trading bot. Combines market read, ops health, and open risks into a single page the human can skim in 60 seconds.

## Instructions

Dispatch these subagents **in parallel** (one message, multiple Agent tool calls):

1. `market-analyst` — produce today's market brief. Prompt: "Run your standard pre-market protocol for the symbols currently traded by the bot. Save the brief under `docs/research/market-briefs/` and return a 5-bullet summary."
2. `live-ops` — produce the standard health snapshot. Prompt: "Run your standard health snapshot. Save under `docs/research/ops-snapshots/` and return severity + any anomalies."
3. `risk-officer` — in live-monitor mode. Prompt: "Produce a current risk snapshot. Focus on drawdown vs limits, concentration, and kill-switch readiness. Return verdict + top 3 open risks."

Wait for all three. Then:

4. Read `docs/project_status.md` for in-flight work context.
5. Synthesize into a **Daily Brief** written to `docs/research/daily-briefs/YYYY-MM-DD.md` and printed to the user.

## Output format

```
# Daily Brief — YYYY-MM-DD

## TL;DR
[One sentence on the state of the world]

## Market
[3 bullets from market-analyst]

## Bot health
- Overall: green/yellow/red
- [any anomalies]

## Risk
- Drawdown: X% (limit Y%)
- Concentration: …
- [top risks]

## In flight
[What the engineering desk is working on, from project_status.md]

## Needs human attention today
[Bulleted list — empty is fine]
```

## When to run

- Daily, pre-market (suggest scheduling via cron invoking `claude -p /daily-brief`).
- Ad-hoc before any material decision.

## Guardrails

- If `live-ops` returns red severity, STOP the brief and escalate directly to the user before continuing.
- If `risk-officer` returns a `reject` or flags an active breach, same — escalate first.
- Do not make any changes to code, config, or live state during this command. It is strictly read-only reporting.
