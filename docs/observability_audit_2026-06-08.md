# Observability & Alerting Hardening Audit — Live Trading Bot

**Date:** 2026-06-08 · **Scope:** production live-trading path (`src/engines/live/`, `src/database/manager.py`, `src/risk/`, `src/position_management/`, `src/data_providers/binance_provider.py`) · **Method:** read-only code audit by 5 parallel reviewers (money trail, orders, positions, reconciliation, risk/connectivity) + production-DB evidence. Every `file:line` was verified by reading code.

---

## TL;DR

The bot is **double-blind**: the events most likely to need an operator (silent money corrections, unprotected positions, external closes/liquidations, loop crashes, WS circuit-open, reconciliation drift) are **(A) not emitted** to the structured event/audit stream, and **(B) not delivered** even when they are — because the only alert channel (a single webhook) is **unset in production**.

**Production proof (last 14 days):** 20 `system_events` total, **`alert_sent=true` on 0 of them** — including the 2 `critical` events that did fire (`CLOSE_ONLY`, `EMERGENCY_CLOSE`). The reconciliation audit table holds **714 `CRITICAL` "manual intervention required"** rows that paged nobody, plus 2 "position unprotected" rows mis-classified `MEDIUM`. Both `margin_equity_sync_correction` balance write-downs (−$15.75, −$1.37) have **zero** audit/event rows.

---

## Root causes (fix these and most findings collapse)

### RC-A — Alert delivery is dead in production (P0)
The only operator-paging path is `_send_alert(message)` → a single `alert_webhook_url` (`trading_engine.py:5024`). It returns `False` when the URL is unset (`:5033`). Production starts the engine via `atb live-health hyper_growth --max-position 0.5`, which forwards straight to `runner.py` (`cli/commands/live_health.py:187`); `--webhook-url` is a CLI flag with **no env/config fallback** (`runner.py`), so `alert_webhook_url is None`. **Every `alert=True` writes a DB row but pages no one.** Worse, the one channel is DB+webhook, so the **DB-outage** close-only trip (`trading_engine.py:2708`) can't even persist its own `system_events` row (the DB is down) — there is no out-of-band path.

### RC-B — No event/alert sink below the engine layer
`reconciliation.py` (whole file: **0** `_record_event`/`_send_alert`/`log_event` calls, ~20 `logger.critical("MANUAL INTERVENTION REQUIRED")` sites), `OrderTracker`, and `LiveExecutionEngine` have no path to `system_events` or the alert webhook — they only `logger.*` and write state tables. So every order-failure / orphan / reconciliation-drift / liquidation event below `trading_engine` lands in application logs only. **Highest-leverage fix:** inject a structured event/alert sink (mirror the existing `on_critical` callback wiring at `trading_engine.py:1515`) into these classes.

### RC-C — Incident-driven, asymmetric instrumentation
Each historical fix was added to one path, leaving its twin silent:
| Instrumented path | Silent twin |
|---|---|
| Startup margin-sync emits `BALANCE_OVERWRITE` (`trading_engine.py:1443`) | **Periodic** margin-sync — nothing (`:2670`, `account_sync.py:240`) |
| Startup balance correction audits (`reconciliation.py:2034`) | **Periodic** balance correction — no audit (`:2972`) |
| Spot external-close audits (`:2675`) | **Margin** external-close/liquidation — no audit, no severity bump (`:2553`) |
| Entry SL-fail → emergency close alerts (`trading_engine.py:3950`) | **Reconciler** emergency-sell twin — nothing (`reconciliation.py:640-709`); **exit/close** paths — nothing |
| Close-only routed via `_enter_close_only_mode` (`:5617`) | Twin branch sets flag silently (`:5583`) |

### RC-D — Two siloed tables + a coarse event taxonomy
`system_events` (engine-emitted, ~13 sites) and `reconciliation_audit_events` (reconciler-emitted) never cross-reference; HIGH/CRITICAL **audit** events do **not** emit a `system_event`, so an operator watching one table is blind to the other. `EventType` has only 9 coarse members (`models.py:110`: ENGINE_START/STOP, STRATEGY_CHANGE, MODEL_UPDATE, ERROR, WARNING, ALERT, BALANCE_ADJUSTMENT, TEST) — no ORDER/POSITION/TRADE/RECONCILIATION/RISK/CONNECTIVITY/DRAWDOWN/LIQUIDATION types. The `orders` FAILED journal update has no `reason`/`error_code` column, so rejection codes (−2010/−1111/51077) are lost.

### RC-E — Advertised safety/alerts that don't exist
- **Hard max-drawdown kill-switch is dead code.** `PortfolioRiskManager.check_drawdown` (20% default, surfaced as `--max-drawdown` "Maximum drawdown before stopping") is **never called** in live (`risk_manager.py:846`; only reference is a docstring). Live drawdown response is graduated position-sizing only (`dynamic_risk.py`, 5/10/15% → ×0.8/0.6/0.4), which never halts and tops out at 15%.
- **`EmergencyControls` subsystem not wired.** `emergency_controls.py` (AlertType, HIGH_DRAWDOWN=15%, CONSECUTIVE_LOSSES=5, cooldowns, rate-limiting) is only instantiated by `performance_monitoring_system.py:55`, which has **zero** references in `src/engines/`/`cli/`. Its alerts only call in-process `alert_callbacks` (none registered) — they never page even in principle.

---

## P1 findings — capital-at-risk, silent (deduped)

| # | Finding | Where | What's recorded today | Fix |
|---|---|---|---|---|
| 1 | **Unprotected position** (SL cancelled / re-place failed) — bare `_send_alert`, no `system_event`, dead in prod. Found by 3 reviewers; also the reconciler SL-re-place-fail twin. | `trading_engine.py:4667-4729`, `:4100` (SL-cancel callback never matches SL order id); `reconciliation.py:2853-2873`, `:1474` | `logger.critical` + bare `_send_alert` (no-op in prod) | `_record_event(ALERT, severity=critical, error_code=POSITION_UNPROTECTED, alert=True)`; re-place SL in the cancel callback; consider auto close-only |
| 2 | **Periodic margin-equity sync correction** (the seed gap) — books overwritten to exchange equity; no audit, no event, no alert. | `account_sync.py:240`; `trading_engine.py:2670` | `account_balances` row + INFO log only | `log_audit_event(balance, HIGH)` + mirror startup `_record_event(BALANCE_OVERWRITE, alert=True)` on the periodic path |
| 3 | **Realized-PnL balance-update failure** → trade logged but balance NOT booked → silent ledger divergence (how phantom balances accrue). | `trading_engine.py:4389-4396`, `:5779-5785` | `logger.error` only | `_record_event(ERROR, severity=critical, error_code=PNL_BALANCE_DESYNC, exc=…, alert=True)` |
| 4 | **Close half-completes** (exchange closed but DB/balance write throws) → DB stays OPEN while flat on exchange. | `trading_engine.py:4525` (catch-all) | `logger.error` only | `_record_event(ERROR, critical, error_code=CLOSE_PARTIAL_FAILURE, alert=True)` |
| 5 | **Stop-loss fill** (dominant real exit) + drain failure — no event/alert. | `trading_engine.py:3993`/`:4036`, drain `:1686`/`:1726` | file log + `stop_loss` trade row | `_record_event(ALERT, STOP_LOSS_FILLED)`; critical alert on drain failure |
| 6 | **External close / margin liquidation auto-close** — audit-only (spot) or fully silent + no severity bump (margin). | `reconciliation.py:2553-2573` (margin), `:2675` (spot, audit-only); `_remove_phantom_position:1919` | warning + `close_position` (+ audit on spot only) | `log_audit_event(position, HIGH/CRITICAL)` + `log_event(ALERT)` + bump severity |
| 7 | **Reconciler emergency-sell twin** of the "good" 3950 template — un-instrumented; "EMERGENCY SELL FAILED, manual intervention required" emits nothing. | `reconciliation.py:640-709` | `logger.critical` only | Same ALERT/critical event as the engine template, via injected sink |
| 8 | **Position adoption/recovery on restart** — silent; the reconciler-adopt path isn't even risk-registered. | `reconciliation.py:534` (`track_recovered_position`); `trading_engine.py:5377` | INFO/WARN log + `positions` row | `log_event(ALERT, POSITION_ADOPTED, alert=True)` |
| 9 | **Loop crash / self-shutdown** indistinguishable from a clean stop; crash-loop is silent (the 2026-05-19 zombie-bot class). | `trading_engine.py:2742-2750`, `:2341`, `:2367`; abnormal `ENGINE_STOP` == clean | `logger.critical` + generic `ENGINE_STOP` | `_record_event(ALERT, LOOP_CRASH, exc, alert=True)`; distinct abnormal-stop event |
| 10 | **WS user-stream circuit-open (REST_DEGRADED)** — bot blind to real-time fills/balance, possibly forever; only `logger.warning`, no escalation. (Live now, #717.) | `trading_engine.py:1824-1950` (0 event calls); `binance_provider.mark_user_degraded` | `logger.warning` only | `_record_event(ALERT, USER_WS_DEGRADED, alert=True)` on circuit-open + recovery + duration-based re-escalation |
| 11 | **HIGH-severity reconciliation drift** (entry-price/qty mismatch, orphaned SL) never trips close-only or emits an event — only CRITICAL does. | `reconciliation.py` severity HIGH at `:1381,1659,1767,1794,1949`; gate `:2984` | `logger.critical` only | Emit `system_events` for every result ≥ HIGH; decide if HIGH trips close-only |
| 12 | **Order rejections** (−2010/−1111/51077) & failed protective orders in the execution layer — log-only, no event, `orders` FAILED row has no reason. | `execution_engine.py:738-752`, `:980`, `:633-701`; `order_tracker.py:404` (orphan force-remove) | `logger.error` + `update_order_journal("FAILED")` (no reason) | Inject sink → `log_event(ERROR/ALERT, parsed code)`; add `reason`/`error_code` column |
| 13 | **Orphan-borrow sweep** active repay (real money out) + over-cap "manual review" — audited but **no `system_event`, no alert** (sweep has no alert channel). | `reconciliation.py:3357-3456` | `reconciliation_audit_events` + log | `log_event(ALERT)` on active repay + over-cap, via injected sink |
| 14 | **Hard max-drawdown kill-switch is dead code** (control gap, not just observability). | `risk_manager.py:846` (never called); flag `runner.py:137` | nothing (never runs) | Wire `check_drawdown` into the loop → close-only + `_record_event(ALERT, MAX_DRAWDOWN)`, or remove the method+flag and fix help text |

## P2 findings — material events recorded-but-not-paged, or wrong tier
- **Non-margin exchange-sync overwrite** evented (startup only) but `alert` defaults False (`trading_engine.py:1443`).
- **Offline stop-loss PnL booking** (stop fired while bot was down) — no `system_event` (`trading_engine.py:5765-5831`).
- **Periodic reconciliation balance correction** — money corrected, no audit row (`reconciliation.py:2972`).
- **`manual_balance_adjustment`** (dashboard POST /api/balance) evented (`BALANCE_ADJUSTMENT`) but never alerted; `updated_by` is client-supplied (`manager.py:2367`, `dashboard.py:452`).
- **Partial stop-loss fill** ("manual monitoring required") uses bare `_send_alert` — no row, no page (`trading_engine.py:4066-4098`).
- **Trailing-stop activation / breakeven trigger** — risk-state change, no event (only a mutated `positions` column) (`exit_handler.py:681`, `position_tracker.py:749`).
- **Estimated-vs-actual fill/commission divergence** — `logger.debug` only; silent P&L/fee-model drift (`execution_engine.py:374-417`, `:531-572`).
- **Order-journal write failures** (crash-recovery anchor lost) swallowed to WARNING (`execution_engine.py:723-727`, …).
- **Kline-stream degradation → REST fallback** — `logger.warning` only (`trading_engine.py:1749-1799`).
- **DB-outage close-only** trip can't persist its own event (DB is the trigger) — needs the out-of-band channel (`trading_engine.py:2708`).
- **Dynamic-risk drawdown/perf reductions** → `risk_adjustments` table, not `system_events`, never alerted (`trading_engine.py:793-835`).
- **Bare `_send_alert` pattern** (Position Opened `:3863`, hot-swap `:2428/:5942`, model update `:5973`) — intended alerts that no-op in prod and write no row.

## P3 findings — minor
- Margin interest silently booked $0 on API failure → balance biased up over time (`margin_interest_tracker.py:73`).
- `update_balance` returns `False` silently on no-session/DB failure; most callers ignore it (`manager.py:2221`).
- Unknown WS order status dropped with only a warning (`order_tracker.py:687`).
- `_normalize_quantity` precision rejections return 0.0 (looks like "no signal") (`execution_engine.py:980`).
- IP-ban / `SUSPENDED` WS state not evented (`binance_provider.py:112-186`).
- Account-sync failure at startup is `logger.warning` only (`trading_engine.py:1454`).
- Scale-in DB-failure swallowed (in-memory size diverges from DB) while partial-exit failure raises — asymmetric (`position_tracker.py:741` vs `:668`).

## Confirmed properly covered (do not re-flag)
- SL-placement-failure → emergency close: `_record_event(ALERT, EMERGENCY_CLOSE, alert=True)` (`trading_engine.py:3950`) — the reference template.
- Close-only activation: `_enter_close_only_mode` (`:1560-1574`), edge-guarded, reused by ambiguous-entry, DB-outage, CRITICAL-reconciliation.
- `_record_event`/`_send_alert` are fault-isolated and record the **real** alert outcome (`alert_sent=True` only on a 2xx webhook), so the prod "nobody paged" state is at least truthfully stored.
- `atomic_balance_update` / `atomic_position_reconciliation`: atomic ledger + AccountHistory + rollback (`manager.py:2398-2747`).
- `log_trade` transactional insert + position-close (`manager.py:702`). Orphan-borrow sweep audits every branch. Startup position/balance corrections are audited. `EventDeduplicator` cannot drop audit/system events (verified safe).

---

## Production evidence (read-only SELECTs, 14-day window)
- `system_events`: **20 rows**, types only `ENGINE_START`(11)/`ENGINE_STOP`(7)/`ALERT`(2); **`alert_sent=true` count = 0**.
- The 2 `critical` ALERTs — `CLOSE_ONLY` (2026-06-05 16:04, "no new entries until manual review") and `EMERGENCY_CLOSE` (2026-06-05 21:04) — both `alert_sent=f`.
- `reconciliation_audit_events`: **714 CRITICAL** "Order UNKNOWN … manual intervention required" (714 distinct orders, 33-min storm on 06-01), 51 LOW, 24 MEDIUM (orphan-borrow dry-run), **2 MEDIUM "position is unprotected, needs SL re-placement"** (06-07 & 06-08; the 06-08 one was on the live position and self-healed).
- Both `margin_equity_sync_correction` write-downs: 0 audit rows, 0 system_events within ±30 min.

---

## Recommended hardening plan (sequenced)

**P0 — make delivery work (else every fix below pages no one):**
1. Wire a real, delivering alert channel in prod: add an `ALERT_WEBHOOK_URL` env fallback and pass it in the `atb live-health` start command. Add a **secondary out-of-band channel** for the DB-down case.
2. On startup, if no alert channel is configured, emit a loud WARNING `system_event` and log it; treat "no alert channel" as a **deploy-gate** for live trading.
3. Add a periodic **alerting self-test/heartbeat** so a silently-dead channel is detected.

**P1 — the two structural fixes that collapse most findings:**
4. **Inject an event/alert sink** (mirror `on_critical`) into `reconciliation.py`, `OrderTracker`, `LiveExecutionEngine`; convert the ~20 `logger.critical("MANUAL…")` + bare `_send_alert` sites to `_record_event(…, alert=True)`.
5. **Make the asymmetric twins symmetric** (periodic margin-sync, periodic balance correction, margin liquidation, close-only `:5583`, reconciler emergency-sell).
6. **Fix severities:** unprotected-position = CRITICAL + alert (not MEDIUM); HIGH reconciliation drift → event + alert.
7. **Loop-crash / abnormal-stop** distinct event + alert; distinguish from clean `ENGINE_STOP`.
8. **WS circuit-open** event + alert + duration-based re-escalation.
9. **Wire (or remove) the hard max-drawdown kill-switch** and fix the misleading `--max-drawdown` help text.

**P2 — taxonomy + dedup:**
10. Add `EventType` members (ORDER_REJECTED, POSITION_OPENED/CLOSED, STOP_LOSS_FILLED, RECONCILIATION_*, DRAWDOWN_*, CONNECTIVITY_*, LIQUIDATION).
11. Emit a `system_event` alongside every HIGH/CRITICAL `log_audit_event` so the two tables unify for operators; add a `reason`/`error_code` column to the order-journal FAILED update.
12. Rate-limit/dedup **and escalate** repeated criticals (the 714-storm) instead of re-flagging per cycle.
13. Add a **coverage test/matrix**: every state transition must declare its observability tier (log / event / alert), failing CI if a money/position/order transition has none.
