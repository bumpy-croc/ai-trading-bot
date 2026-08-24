---
id: 2026-08-24T1200-P1-close-only-4day-silent-halt
opened_by: live-ops
severity: P1
status: open        # open | mitigated | closed
opened_at: 2026-08-24T12:00:00Z
mitigated_at: null
closed_at: null
human_paged: true
affected_components: [live-engine, reconciliation, stop-loss-placement, daily-trading-standup]
affected_symbols: [ETHUSDT]
---

## What happened

Production (session 20, HyperGrowth/ETHUSDT, commit `fba64281` deployed 2026-08-13, no
restart since) latched **close-only mode** at **2026-08-20 16:48:51 UTC** after a stop-loss
re-placement failure, and has been blocking all new entries for **~3d19h** as of first
detection (2026-08-24 ~11:50 UTC). The book is flat (0 open positions, balance ≈ $87.27) and
the halt is fail-safe — no capital was at risk during the window — but it went **undetected
by the human operator for the full duration**, despite the daily automated standup running
on schedule every day in that window.

## Detection

PM-led forensic sweep, 2026-08-24, triggered by a manual review (not by any automated
alarm). `max(strategy_executions.timestamp)` = 2026-08-20 16:49:43 was the first hard
signal — 4 days with zero strategy-execution rows on a bot that normally logs one per
`check_entry_conditions` call.

**This was a surprise finding, not a designed detection.** The daily-trading-standup
scheduled task (`0 9 * * *`) fired on time every day 08-20 through 08-24 and reported
**NOMINAL every single time** (see Root cause).

## Impact

Paper/live: **live**, real capital ($87.27). No loss — book was flat for the entire halt
window; the halt is fail-safe by design (exits/stops remain active, only new entries are
blocked). Opportunity cost only:

- Zero entry orders since the halt (`orders` table: 0 `ENTRY` rows after 2026-08-20 16:49;
  `trades` table: 0 new positions after that time).
- The engine's **signal-generation loop never stopped** — `Decision:` log lines kept flowing
  at the normal ~60–140s cadence throughout (confirmed for the last several hours; older
  Railway log history has rotated out and is unrecoverable — see Root cause note on log
  retention). Today's 400-line sample (06:00–11:53 UTC) shows 32 `BUY` decisions, **all with
  `Size: 0.00`** (near-zero confidence/strength in a `trend_down/low_vol` regime) — consistent
  with the pre-existing, already-tracked #700/#1045 sub-minimum-sizing pattern. On that one
  day's evidence the suppressed entries would likely have been rejected by the sizing floor
  regardless of close-only, so realized opportunity cost looks low — but this cannot be proven
  for 08-20→08-23 because `strategy_executions` is empty for that whole window (the logging
  call sits *after* the close-only early-return in `entry_coordinator.check_entry_conditions`)
  and Railway `logs` only retains the current deployment's recent tail. **Do not extrapolate
  the one-day sample to the full 4-day window as fact.**

## Root cause

**Two independent failures, not one:**

### 1. The SL re-placement failure that triggered the latch
`src/engines/live/reconciliation.py:3985` — the periodic reconciler detected the exchange
had cancelled/rejected/expired the resting stop-loss for the then-open ETHUSDT position
(position DB id 27 / trade id 18), attempted to re-place it via
`exchange.place_stop_loss_order(...)`, and got back `None` — logged only as "exchange
returned no order id" (`reconciliation.py:4187`). This generic string is **all that survives**
in `system_events`/`reconciliation_audit_events`; the actual Binance error text is swallowed
inside `BinanceProvider.place_stop_loss_order` (`src/data_providers/binance_provider.py:1923-1929`
— both `BinanceOrderException` and the general `Exception` handler do `logger.error(...); return
None`, never propagating the code/message to the caller). That `logger.error` line would only
have existed in application stdout at 2026-08-20 16:48 — four days before this investigation —
and `railway logs` never retains history beyond the current deployment's live tail (confirmed:
`-n 400` on 2026-08-24 only reaches back to 05:39 UTC that day). **The precise Binance error
code (precision/-1111/51077, -2010 insufficient-balance, or something new) is unrecoverable.**
This is a structural gap, not a one-off: any silent `place_stop_loss_order` failure loses its
root cause the moment the log line scrolls off Railway's retention, because the code path that
turns it into a persisted DB record (`_audit_unprotected`) never receives the underlying
exception.

A near-identical churn (5 rejections in ~90min, orders 49076124513→49076355916) happened the
day before (2026-08-19 15:52–17:19) on a *different* position (trade id 16) and self-resolved
without triggering close-only (that position's SL eventually re-placed successfully or the
position exited before exhausting retries). The 08-20 16:48 failure was the one that finally
hit the reconciler's CRITICAL threshold.

Separately, immediately after the close-only latch (16:48:51–16:48:52), the position itself
exited one second later (16:49:49) via `exit_reason = "Stop loss"` — i.e. the exchange-resident
stop order (the one the reconciler couldn't confirm re-placing) appears to have actually filled
and protected the position correctly; the failure was in **our tracking/re-placement
confirmation**, not necessarily in real exchange-side protection. This is consistent with all
three of the most recent exits (trades 16, 17, 18) being profitable "Stop loss" exits.

### 2. Why nothing escalated for 4 days despite the alert firing
**Correction to the initial working hypothesis:** an operator alert *was* sent. Both
close-only system_events (`id=91` "Close-only mode activated…", `id=92` "Reconciliation cycle
detected a CRITICAL discrepancy…") have `alert_sent=true, alert_method=webhook` in
`system_events` — `ALERT_WEBHOOK_URL` is configured in prod (a Slack webhook) and the
alert-delivery code path (`_record_event(..., alert=True)` → `_send_alert`, shipped in commit
`20788b72`, already live in the deployed `fba64281` build) fired successfully at 16:48:51 and
16:48:52 UTC on 2026-08-20. **Whether that Slack message was seen/actioned is outside this
agent's visibility (no Slack tool access) — flagging to the human to confirm channel
delivery/visibility, since this changes the finding from "no alert fired" to "an alert fired
and was not actioned for 4 days."**

Independent of the webhook question, the **daily-trading-standup** (the sole scheduled
watchdog since the 6-hourly alert-monitor was retired 2026-07-29) ran on schedule at 09:01 UTC
every day 08-20 through 08-24 (confirmed via session-transcript directory mtimes — no missed
slots, unlike the #1085 class of failure) and reported **NOMINAL** every time. It structurally
could not have caught this:
- It has no check for `CLOSE-ONLY MODE ACTIVATED` / `error_code=CLOSE_ONLY` in logs or
  `system_events`, and does not check `max(strategy_executions.timestamp)` recency.
- Its "engine alive" positive-state assertion greps for `Decision:` log lines — but those are
  emitted by the strategy/signal layer *before* `entry_coordinator.check_entry_conditions`'s
  close-only early return, so they keep flowing at normal cadence even with entries fully
  blocked. This check cannot distinguish "healthy and quiet" from "latched close-only."
  Confirmed empirically today: current logs still show 60–140s-cadence `Decision:` lines while
  close-only remains active.
  - Each of the four daily runs (08-21, 08-22, 08-23, 08-24) individually noted the bot was
    "flat" for a growing duration and attributed it to the pre-existing, already-triaged
    #1045/#700 zero-size-decision pattern — a plausible-looking known cause that happened to
    match the observed symptom (near-zero-size `Decision:` lines) and was never cross-checked
    against `system_events`/`strategy_executions` for a *different*, new cause.

## Timeline

```
2026-08-19 15:52–17:19 UTC — 5x stop-loss order cancel/reject/expire for trade-16's position
                              (self-resolved, no close-only trigger)
2026-08-19 17:20:56    UTC — trade 16 exits, "Stop loss", +8.00%
2026-08-19 17:21:05    UTC — trade 17 opens
2026-08-19 21:10:29    UTC — trade 17 exits, "Stop loss", +10.46%
2026-08-19 22:25:43    UTC — trade 18 opens
2026-08-20 16:48:49    UTC — reconciliation_audit_events id=796: SL re-placement failed
                              for position 27 (trade 18), CRITICAL
2026-08-20 16:48:51    UTC — [detection/self] system_events id=91 ALERT: "Close-only mode
                              activated" — alert_sent=true, webhook
2026-08-20 16:48:52    UTC — [detection/self] system_events id=92 ALERT: reconciliation
                              CRITICAL discrepancy — alert_sent=true, webhook
2026-08-20 16:49:49    UTC — trade 18 exits, "Stop loss", +3.63% — book goes flat
2026-08-20 → 08-24     UTC — max(strategy_executions.timestamp) frozen at 08-20 16:49:43;
                              zero ENTRY orders; zero new trades; 0 open positions throughout
2026-08-21..24 09:01   UTC — daily-trading-standup runs, reports NOMINAL each day (misses halt)
2026-08-24 ~11:50      UTC — [escalation] PM-led manual sweep detects the halt via
                              strategy_executions staleness
2026-08-24 12:00       UTC — this incident filed; human paged per charter escalation method
```

## Actions taken

- Read-only forensic sweep only. No writes to prod DB, no config/flag changes, no restart.
- Connected read-only to prod Postgres (`DATABASE_PUBLIC_URL` via `railway variables -e
  production -s Postgres --json`, `SET default_transaction_read_only = on` first statement).
- Pulled `railway logs -e production -s "Trading Bot" -n 400` (only reaches back to 08-24
  05:39 UTC — confirms the log-retention gap noted above).
- Queried `system_events`, `reconciliation_audit_events`, `orders`, `trades`, `positions`,
  `system_control_flags` (empty — confirms `_close_only_mode` is in-process memory only, not
  persisted anywhere) for the incident window and current state.
- Read `src/engines/live/entry_coordinator.py`, `reconciliation.py`, `trading_engine.py`,
  `binance_provider.py`, `recovery.py` to trace the close-only latch, the swallowed exchange
  error, and the alert-delivery path.
- Read all five 09:01 UTC `daily-trading-standup` session transcripts (08-20 through 08-24)
  to confirm what was actually reported.

## Current state

Bleeding stopped (was never bleeding — fail-safe halt, book flat). System is currently:
close-only mode still latched (no `resume_trading()`/deactivation event exists, and the
Trading Bot service has not restarted since the 08-13 deploy that predates this incident),
zero open positions, heartbeat current (account_history rows still landing normally — the
heartbeat writer is not gated by close-only).

## Recovery requirements (NOT executed — human/authorized-agent action required)

1. **Confirm no orphaned exchange state**: verify via exchange (not just DB) that ETHUSDT has
   no resting orders and no held/borrowed base asset — the reconciler's own SL-placement
   trouble on 08-20 is reason enough to independently re-verify, not just trust the DB's "0 open
   positions."
2. **Root-cause the swallowed exchange error** before trusting the SL path again: the specific
   Binance rejection code from 08-20 16:48 is gone from logs; consider whether
   `place_stop_loss_order`'s exception handlers should propagate the code/message into
   `_audit_unprotected`'s `reason` field (currently just "exchange returned no order id") so a
   future occurrence is diagnosable without racing log retention.
3. **Clear close-only mode**: `_close_only_mode` is a plain in-process bool
   (`trading_engine.py:782`) with **no CLI/API to clear it remotely** — `resume_trading()`
   (`trading_engine.py:1314`) has zero call sites outside the class itself. The only way to
   clear it on the *running* process is a **full restart** of the "Trading Bot" Railway
   service, which re-runs `reconciler.reconcile_startup(...)` (`recovery.py`) — since the book
   is currently flat, that reconciliation has nothing to protect and should not immediately
   re-latch on the same cause, but this is inference, not a guarantee, and restarting a
   live-capital process is outside this agent's authorization (Authorization matrix: "Restart
   process — Live: no — escalate").
4. **Fix the standup's blind spot** before relying on it again as the sole watchdog (see
   companion GitHub issue) — otherwise a repeat of this exact failure mode will again read as
   NOMINAL for days.

## Post-mortem (filled after close)

### Root cause
(see above — pending human sign-off / close)
### Contributing factors
### What went well
### What went poorly
### Action items (each links to a proposal or tracker)
