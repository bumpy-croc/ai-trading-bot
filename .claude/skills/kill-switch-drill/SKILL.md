---
name: kill-switch-drill
description: Quarterly staging fire-drill for the safety stack — prove the drawdown guard trips close-only, alerts actually reach Slack, entry-pause round-trips, and rollback works, then record the results. Use on the quarterly cadence, after any change to guards/alerting/close-only code, or when anyone asks "would the kill switch actually fire?".
---

# Kill-Switch Drill

**Safety code rots invisibly unless fired.** June 2026 proved it four ways: the 20% max-drawdown
"kill switch" (`check_drawdown`) was dead code with zero callers; `EmergencyControls` was never
wired into any engine; the alert webhook was unset in prod → 0 of 20 system_events delivered in
14 days including 2 criticals (the double-blind finding, `docs/observability_audit_2026-06-08.md`);
and the drawdown guard's seed defect (peak=$100 vs true $84.42) only surfaced the day the guard
first armed (2026-07-04). None of these were visible in CI. A drill fires each mechanism on
staging (paper, ~$1,000) and watches it actually work.

Run quarterly, and after any change under `src/engines/live/monitoring/` or to alerting.
All mutations are STAGING-ONLY. Results append to layer 2 (log.md + issues); see
`docs/architecture/memory_system.md`.

## Drill checklist

**0. Baseline.** `deploy-staging` boot verification passes; note deployment id for rollback.

**1. Entry-pause round trip.**
```bash
railway variables --set "FEATURE_ENTRY_PAUSE=true" -e staging -s "Trading Bot"   # = a redeploy
```
Verify the boot log shows the pause armed and entry evaluation logs the paused skip (EntryPauseGate
also gates scale-ins — that was a review catch on #835, confirm it still holds). Flip back off;
verify entries resume. Each `--set` is a restart — batch flags into one command.

**2. Drawdown guard trips close-only.** The guard recomputes its peak from session
`account_history` on boot (`drawdown_guard.py`; #850/#851 seed fix). Simulate breach-distance by
re-baselining with the purpose-built flag rather than faking balances:
`FEATURE_MAX_DRAWDOWN_RESET_PEAK=true` re-baselines the peak (misuse-safe: it can only ever make
the guard MORE likely to be correct after a reconciled reset — never fabricate DB rows to force a
trip). Verify: arm banner shows the expected peak + 20% cap and the 10%/16% warning tiers; if a
paper drawdown can be arranged, confirm the trip enters close-only and emits the alert event.
At minimum, assert the wiring: the guard is constructed, armed, and its trip path calls
`_enter_close_only_mode` (the June lesson was precisely that this call-path check never happened).

**3. Alert delivery, end-to-end.** A drill-triggered alert (or the startup `NO_ALERT_CHANNEL`
guard warning if the webhook is deliberately unset) must produce: a `system_events` row with
`alert_sent=true` AND a visible Slack message. `alert_sent` records the real webhook outcome
(2xx-only), so a `true` row is proof of delivery. Staging webhook parity is tracked in #915 —
if staging has no webhook, that's a drill FINDING, not a skip. Verify the alert self-check:
prod's 07-04/07-05 startups fired `NO_ALERT_CHANNEL` loudly; a silent unset channel = regression.

**4. Documented manual trigger exists.** risk-limits.json names
`manual_trigger_command: "atb live-control halt"`. As of 2026-07-06 that command DOES NOT EXIST,
and `atb live-control emergency-stop` prints "(simulated)" and exits 0 — a vapor kill-switch.
Run the documented command; if it doesn't exist or doesn't halt, file the issue and flag the
risk-limits.json line to `risk-ratification`. The real levers today: FEATURE_ENTRY_PAUSE,
close-only mode, Railway service stop.

**5. Circuit breakers.** `account_circuit_breakers` flag (`off`/`dry_run`/`on`): in dry_run,
verify the `🟡 … WOULD HALT (dry_run)` line appears on a simulated-threshold day (LESSONS §5.2
lists the live trip signatures: `ACCOUNT_CIRCUIT_BREAKER_TRIP`).

**6. Rollback/redeploy drill.** Redeploy the previous SUCCESS deployment via Railway, verify
re-boot: session reuse, position re-adoption (`Positions:` count survives, new opens = 0), guard
re-arms at the correct peak. This is the same muscle `deploy-prod` rollback needs; exercise it
where it's free.

## Record

- log.md entry (kind `note`, via `decision-record`, `[D-…]` id): date, each mechanism
  PASS/FAIL, evidence lines (log timestamps, system_events ids, Slack permalink).
- Every FAIL → GH issue (`priority:p1` if it's a delivery or trip-path failure — a dead alert
  channel is the double-blind incident again), linked from the log entry.
- Restore ALL staging flags to their pre-drill values; verify with a final boot check.

## Red flags

- A mechanism "passes" without an artifact (no event row, no Slack message, no log line) —
  that's a fail. The audit's core finding was mechanisms that *looked* wired.
- Drill left staging flags dirty (next trial inherits them silently).
- Anyone proposing to drill on production. Never. Prod validation is boot verification
  (`deploy-prod`) + the standup tripwires, not induced failures.
