# Heartbeat — dead-man's-switch

The most important automation is the one that catches the automation being broken. Run on a short schedule (e.g., every 30 minutes). If this command has not run in > 36 hours, the human should be paged and live trading halted.

## Step 1 — Write heartbeat

```bash
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) heartbeat ok pid=$$" >> .claude/state/heartbeat.log
```

## Step 2 — Self-check

Verify the state system is sane:

- `.claude/state/charter.md` exists and is non-empty.
- `.claude/state/risk-limits.json` parses as JSON and has all required keys.
- `.claude/state/decisions.jsonl` is append-only (last-mtime > first-mtime; size monotonic via a size-snapshot file — optional but recommended).
- No P0 incident has been in `open/` for > 2 hours (if so, human contact has failed — rotate to a louder alert).

On ANY failure: this is itself a P0. Open an incident `.claude/state/incidents/open/` with id `heartbeat-self-check-failed` and page the human.

## Step 3 — Liveness check on the bot

Use the real health mechanisms, do NOT spawn subagents (heartbeat must be cheap):

```bash
# Process check
pgrep -af "atb live" || echo "no live process"
# Health endpoint if configured
curl -sf -m 5 http://localhost:8000/health || echo "health endpoint fail"
# DB check (read-only)
atb db verify 2>&1 | head -5
```

## Step 4 — Classify

- **All green** → append `status=green` to heartbeat.log, exit silently.
- **Bot process missing but charter says paper-only** → append `status=yellow`, open a P1 incident if no existing one.
- **Bot process missing in live mode, or health endpoint down** → append `status=red`, open a P0 incident, page human.
- **DB unreachable** → P0. Do not proceed. Page.

## Step 5 — Cost discipline

The heartbeat must be cheap: no LLM calls, no subagents, just bash. Budget < 1 second of wall clock and < 5KB of context. If this ever starts spawning agents, something is wrong — simplify it.

## The dead-man mechanism (external)

This command is the *internal* heartbeat. The dead-man side (external to Claude) should be:

1. A cron job or Railway scheduled task that checks `mtime` on `.claude/state/heartbeat.log`.
2. If mtime is > 36h old → it fires an alert via whatever channel the charter specifies.
3. If `status=red` was the last entry and no recovery in 1h → it halts live trading (e.g., `atb live-control halt` or stops the Railway deploy).

Without the external piece, this command is only half of a dead-man switch. Document the external wiring in the `charter.md` "Escalation" section.
