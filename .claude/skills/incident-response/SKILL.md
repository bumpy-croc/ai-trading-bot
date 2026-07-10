---
name: incident-response
description: P0/P1 incident playbook for the live trading bot — classify by log signature, capture evidence BEFORE containment destroys it, apply the pre-committed containment for that class (entry-pause / close-only / kill-switch), open the incident record, escalate per charter. Use when a monitor, standup, or human reports capital at risk, a down bot, an order storm, auth failures, or DB/exchange divergence.
---

# Incident Response

The playbook for "something is wrong in production." Detection lives in `bot-monitor-live`
(signatures: `.claude/LESSONS.md` §5); this skill is what the RESPONDER does next. Memory layers
per `docs/architecture/memory_system.md`: you append to layer 2 (incident file, log.md, GH issue);
lessons reach LESSONS.md via `weekly-retro`, not mid-incident edits.

## 1. Evidence FIRST — containment actions destroy it

`railway logs` serves ONLY the current deployment, and `railway variables --set` (any flag flip)
triggers a redeploy — so the act of containing erases the logs of the thing you're containing
(learned the hard way in the #913 forensics: no `--since` value reaches prior containers). Never
run `railway domain` during evidence-gathering (or any other pass) — it's get-or-create, not a
read, and created an unauthorized public domain on prod when run "to check a URL" (2026-07-08
incident, GH #941; full safe/prohibited Railway CLI list: `.claude/LESSONS.md` §3).

```bash
railway logs -e production -s "Trading Bot" -n 1000 > /tmp/incident-$(date -u +%Y%m%dT%H%M).log 2>&1
# Read-only DB snapshot (positions, balances, recent events) — psql via the PUBLIC proxy URL:
# first statement ALWAYS: SET default_transaction_read_only = on;
```

Snapshot: open `positions` rows, last 5 `account_balances` rows, last 20 `system_events`, the
`account_history` heartbeat timestamps. Seconds of work; do it before ANY mutation. Exception:
if capital is actively bleeding (repeated emergency-close churn), contain first — a lost log is
cheaper than a lost account. Forensics then run from Postgres (`prod-forensics` skill).

## 2. Classify by signature

| Class | Signature (grep the log dump) | Real precedent |
|---|---|---|
| SL-fail cascade | `emergency.close` / "Stop-loss placement failed", repeating | 2026-06-02: ~15% capital erosion ($100→$84) while reporting phantom $99.89 |
| Close-only trip | `CLOSE-ONLY MODE ACTIVATED` | reconcile/DB problem; entries already halted |
| Circuit breaker | `ACCOUNT_CIRCUIT_BREAKER_TRIP` / `risk_event=account_circuit_breaker_trip` | #807 daily-loss/drawdown halt; operator reviews & clears |
| Order storm | Same CRITICAL repeating per-cycle (e.g. "Order UNKNOWN… manual intervention") | 06-01: 714 CRITICAL rows in a 33-min storm, paged nobody |
| Auth failure | `-2015` / signature/IP-restriction errors with open positions | kill-switch auto-trigger condition per risk-limits.json |
| Precision regression | `code=-1111` / `code=51077` (ANCHORED grep — bare digits match timestamps) | LESSONS §1.1; recurrence = regression |
| DB divergence | "No active trading session for balance update"; balance ≠ exchange equity | #693; phantom-balance era |
| Zombie bot | Deploy API SUCCESS but no `Decision:` lines and no hourly `account_history` row | 2026-05-19: both bots dead for days, API said SUCCESS |
| Degraded-not-down | WS churn + REST fallback, fills still polling, SL intact | 2026-07-05 IP transition: correctly NOT treated as P0 |

## 3. Contain — pre-committed action per class

- **Entries are the risk, protection intact** → `FEATURE_ENTRY_PAUSE`:
  `railway variables --set "FEATURE_ENTRY_PAUSE=true" -e production -s "Trading Bot"`
  (this restarts the bot — restart-with-position is safe by design: re-adoption #677 + dedup
  guards; never wait for a "flat window", you can't catch one — LESSONS §2.4).
- **State integrity suspect (DB divergence, reconcile errors)** → close-only mode. It trips
  itself via `_enter_close_only_mode`; if it hasn't, escalate rather than hand-rolling writes.
- **Kill-switch criteria** (risk-limits.json `auto_trigger_conditions`): db/memory divergence,
  duplicate-order storm, auth failure with open positions, data corruption. Kill-switch is
  `authorized_actors: ["human"]` — you recommend, the human pulls it. NOTE: the documented
  `atb live-control halt` does not exist (see `kill-switch-drill`); the real levers are
  entry-pause, close-only, and stopping the Railway service.
- **Degraded-not-down** → no midnight heroics on a protected account (2026-07-06 nightcap
  precedent): designed-degraded + monitoring + a daylight fix issue.
- **NEVER** manually flatten to "help", never repay/sell margin dust without `free` vs
  `borrowed` vs `netAsset` analysis (LESSONS §1.5 — selling held dust creates a naked short).

## 4. Verify the premise before paging (P-phantom class)

Cron/relayed premises have repeatedly evaporated (LESSONS §5.5, §2.5; the 2026-06-05 "orphan"
was a phantom DB row, not double-exposure). Confirm against live state: `get_open_orders`, the
actual order id, tracked `Positions:` count, heartbeat row, UTC clock math (`date -u` — a GMT+1
scheduler makes a live bot look 1h stale).

## 5. Record + escalate

1. `.claude/state/incidents/YYYY-MM-DDTHHMM-P<n>-<slug>.md` — `status: open` frontmatter,
   timeline (UTC), evidence paths, containment applied, severity rationale.
2. GH issue, labels `type:incident` + `priority:p<n>` + `area:live-ops`. P0 scopes the whole
   session to it (CLAUDE.md daemon rule).
3. `log.md` append via `decision-record` (`[D-…]` id, kind `incident-open`).
4. Escalate per charter.md Escalation section (method + SLA); while waiting: freeze new
   entries, maintain stops, keep monitoring. One escalation per state, not per tick.

## 6. Postmortem (before `status: closed`)

Template: impact ($ and duration) · timeline · root cause (5-whys until a code/process defect)
· why detection missed it (compare vs the 2026-06-08 observability audit — most misses were
"emitted but never delivered") · fixes as GH issues with owners (the #626–#631 hardening series
is the reference shape) · corrections to any wrong mid-incident claims, appended per the
phantom-peak pattern (memory_system.md discipline 1). Feed durable rules to `weekly-retro`.
