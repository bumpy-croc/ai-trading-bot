# Heartbeat — dead-man's switch

Pure bash. No LLM calls, no subagents. Cheap, frequent, and includes a drawdown tripwire.

## Step 1 — Heartbeat

```bash
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "$ts ok" >> .claude/state/heartbeat.log
```

## Step 2 — Bot liveness

```bash
# Process
pgrep -af "atb live" >/dev/null || echo "$ts WARN no-live-process" >> .claude/state/heartbeat.log
# Health endpoint (if configured)
curl -sf -m 5 http://localhost:8000/health >/dev/null 2>&1 || echo "$ts WARN health-endpoint-down" >> .claude/state/heartbeat.log
# DB
atb db verify >/dev/null 2>&1 || echo "$ts ERROR db-unreachable" >> .claude/state/heartbeat.log
```

## Step 3 — Drawdown tripwire

```bash
# Replace with real query. Must exit cleanly on error, not crash the heartbeat.
current_dd=$(atb risk drawdown --json 2>/dev/null | jq -r '.current_pct // empty')
limit=$(jq -r '.portfolio.max_drawdown_pct' .claude/state/risk-limits.json)
warn_frac=$(jq -r '.escalation.warning_at_pct_of_limit' .claude/state/risk-limits.json)

if [ -n "$current_dd" ] && [ -n "$limit" ]; then
  warn_at=$(awk -v a="$limit" -v b="$warn_frac" 'BEGIN{print a*b}')
  if awk -v c="$current_dd" -v l="$limit" 'BEGIN{exit !(c>=l)}'; then
    # Breach — open P0 incident
    slug=".claude/state/incidents/$(date -u +%Y-%m-%dT%H%M)-P0-drawdown-breach.md"
    cat > "$slug" <<EOF
---
id: $(basename "$slug" .md)
severity: P0
status: open
opened_by: heartbeat
opened_at: $ts
human_paged: true
---
## What happened
Drawdown $current_dd breached limit $limit at $ts.
EOF
    echo "$ts ERROR drawdown-breach dd=$current_dd limit=$limit" >> .claude/state/heartbeat.log
    # Fire webhook per charter (example; real command from charter.md)
    # curl -X POST "$ALERT_WEBHOOK" -d "P0 drawdown breach: $current_dd / $limit"
  elif awk -v c="$current_dd" -v w="$warn_at" 'BEGIN{exit !(c>=w)}'; then
    echo "$ts WARN drawdown-approach dd=$current_dd warn=$warn_at" >> .claude/state/heartbeat.log
  fi
fi
```

## Step 4 — Self-check

```bash
test -s .claude/state/charter.md     || echo "$ts ERROR charter-missing"     >> .claude/state/heartbeat.log
test -s .claude/state/risk-limits.json || echo "$ts ERROR risk-limits-missing" >> .claude/state/heartbeat.log
```

## External dead-man

This command is the *internal* heartbeat. The external watchdog (cron, systemd, Railway scheduler) must:

1. Check `mtime` of `.claude/state/heartbeat.log`. If > 36h stale → page human and run `atb live-control halt`.
2. Tail last 10 lines; if any `ERROR` with no subsequent `ok` in 1h → same response.

Document the external wiring in `charter.md` "Escalation" section.

## Cost discipline

No LLM calls. No subagents. Budget < 2s wall clock. If this command ever starts spawning agents, revert — something is wrong.
