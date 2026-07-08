# Forensics: Latency-abort phantom-short bug — did production ever trade on it?

**Date**: 2026-07-06
**Investigator**: live-ops (read-only forensics)
**Scope**: Determine whether PRODUCTION (or staging, as secondary signal) ever opened a position on
an error `PredictionResult` (price=0.0, confidence≈1.0, predicted_return≈-1.0) caused by
`PredictionEngine.predict()`'s 0.1s latency-abort/invalidation bug (`src/prediction/engine.py:299-318`,
`src/strategies/components/ml_signal_generator.py:1061`), with special attention to the SHORT
position opened in prod around 2026-07-02/03.

## Access paths and their limits

| Path | Status | Notes |
|---|---|---|
| Railway CLI, prod service logs | **Available but window is short** | `railway logs --service <Trading Bot> --environment production` only returns logs from the **current running deployment**. The current prod deployment (`a9d3001f-3d2c-42e1-91c7-5dac6fbd12cd`, commit `53d41f31` / PR #911, "WS subscription hold fix") started at **2026-07-06 12:30:30 UTC** — i.e. ~1h45m before this investigation. `--since 7d/14d/30d` did not extend the window; the earliest line returned was always the process-start "Non-US location detected" log. **Railway log retention for this incident is effectively zero** — the container that was running on 2026-07-02/03 has long since been replaced by later deploys, and Railway does not appear to retain logs across deployments/restarts via this CLI path. |
| Railway CLI, staging service logs | Same limitation | Staging's current deployment also started today (~12:22 UTC). No historical reach into July 1-6 via logs either. |
| Production Postgres (read-only, via `DATABASE_PUBLIC_URL` proxy `trolley.proxy.rlwy.net:10722`) | **Available, full history** | This was the primary evideence source. `psql`, SELECT-only. |
| Staging Postgres (read-only, via `DATABASE_PUBLIC_URL` proxy `switchyard.proxy.rlwy.net:12631`) | **Available, full history** | Used as secondary signal per instructions. |

**Conclusion on log access**: Railway's `logs` command could not reach the 2026-07-01–07-06 window at all (verified empirically — earliest log line returned was the current container's own startup line, regardless of `--since` value). All findings below come from the production/staging databases, which retain full history.

## Config values confirmed

- `DEFAULT_MAX_PREDICTION_LATENCY = 0.1` seconds (`src/config/constants.py:10`).
- Production env vars (`railway variables --service <Trading Bot> --environment production`) contain **no `MAX_PREDICTION_LATENCY` override** — confirms prod runs on the 0.1s default, exactly as described in the bug report.
- Confirmed by direct code read: `engine.py:299-318` returns `PredictionResult(price=0.0, confidence=0.0, ..., error="Prediction timeout after {t}s (max: {max}s)", metadata={"error_type": "PredictionTimeoutError", ...})` when `inference_time > max_prediction_latency`, silently (no log line at this call site). `ml_signal_generator.py:1061` does `pred = float(result.price)` with no check of `result.error` before returning `pred` for use in `predicted_return` / `_calculate_confidence` (`ml_signal_generator.py:1073-1084`: `confidence = min(1.0, abs(predicted_return) * self.confidence_multiplier)`).

## Queries run (production DB)

1. `\dt` — table discovery (23 tables incl. `trades`, `positions`, `orders`, `strategy_executions`, `system_events`, `prediction_cache`, `prediction_performance`).
2. `SELECT ... FROM trades WHERE entry_time >= '2026-07-01' AND entry_time < '2026-07-07'` → **0 rows**. No trade in `trades` closed in this window (trades only land in this table on close).
3. `SELECT ... FROM positions WHERE entry_time >= '2026-07-01' AND entry_time < '2026-07-07'` → **1 row**: position `id=22`, ETHUSDT SHORT, `status=OPEN`, `entry_price=1696.83`, `size=0.15955605`, `entry_time=2026-07-02 13:34:24.610464`, `strategy_name=HyperGrowth`, `entry_order_id=48096837652`. This is the short position referenced in the task ("opened around 2026-07-03" — actual entry timestamp is 2026-07-02 13:34 UTC, still open as of this writing, unrealized PnL ≈ -$0.30 to -$0.40 per the live status logs, consistent with a normal small ETHUSDT short).
4. `SELECT ... FROM orders WHERE created_at BETWEEN '2026-07-02 13:20' AND '2026-07-02 13:45'` → found the exact `ENTRY` order (`id=7427`, `filled_price=1696.83`, `filled_at=2026-07-02 13:34:24.948128`) that opened position 22, immediately preceded by a `FULL_EXIT` closing the prior LONG position 21.
5. `SELECT ... FROM strategy_executions WHERE timestamp BETWEEN '2026-07-02 13:00' AND '2026-07-02 14:00'` — pulled the **complete** decision trail around the entry (44 rows). Every `opened_short` row in this run shows a **gradually increasing** confidence (0.05 → 0.36) and signal_strength tracking price moving further against the existing thesis — a normal accumulating-signal pattern, not a single-bar spike to 1.0. The row nearest the actual fill (`id=141042`, `timestamp=2026-07-02 13:34:18.067`, `price=1695.14`, `confidence_score=0.2500`, `signal_strength=0.2083`) is unremarkable and consistent with legitimate model output.
6. `ml_predictions` column check: `SELECT count(*), count(ml_predictions) FROM strategy_executions` → 151113 / 151113 (looked fully populated), **but** `SELECT ml_predictions::text, count(*) ... GROUP BY 1` and `SELECT count(*) WHERE ml_predictions::text != 'null'` → **0 rows ever have non-null JSON content; the column is the JSON literal `null` in every single row in the table's entire history.** This column is effectively dead/unwired in the current code path and could not be used as direct evidence either way — noted as a data-quality gap, not evidence of the bug.
7. Signature search across **all of `strategy_executions` history** (not just the window):
   - `reasons::text ILIKE '%prediction_failed%' OR ILIKE '%PredictionTimeout%' OR ILIKE '%predicted_return%'` → **0 rows, ever.**
   - `reasons::text ILIKE '%risk_signal_confidence_1.0000%' OR ILIKE '%risk_signal_strength_1.0000%'` → **5 rows total, in all of prod history**: ids 1 (2026-03-29), 270 (2026-04-03), 70490/70872 (2026-06-02), 71007 (2026-06-03). **None fall in the 2026-07-01–07-06 window.** All 5 are `opened_short` entries with plausible surrounding context (varying `regime_confidence`, sane `price` values, no `price=0.0` anywhere) — consistent with the strategy's normal confidence-clamping behavior on strong signals (~once every 2-4 weeks), not with a repeating timeout-driven artifact. Flagged as worth a closer look by quant/ML but not tied to this incident window.
   - Direct numeric check: `SELECT max(confidence_score), max(signal_strength) FROM strategy_executions WHERE timestamp BETWEEN '2026-07-01' AND '2026-07-07' AND action_taken LIKE 'opened%'` → **max confidence_score = 0.371, max signal_strength = 0.309**. No `opened_*` action anywhere in the investigation window came remotely close to confidence=1.0.
8. `system_events` — `SELECT count(*) WHERE message ILIKE '%timeout%' OR message ILIKE '%prediction%' OR event_type ILIKE '%prediction%'` → **0 rows in the entire table's history.** Only 4 `event_type` values exist at all: `ALERT`, `ENGINE_START`, `ENGINE_STOP`, `WARNING`. Prediction-timeout events are not routed to this table by current code, so its absence is expected and not itself exculpatory — but it also means this table cannot corroborate either way.
9. `system_events` in window (`2026-07-01` to `2026-07-07`) — 29 rows, **all unrelated to prediction/timeout**: repeated `WARNING`/`ALERT` "User data stream circuit-open after N reconnects — REST-degraded" (N climbing 3→10→40→...→1360 from 2026-07-04 14:38 through 2026-07-06 11:39, ongoing) plus "No alert channel configured — operator alerts will not be delivered" (2026-07-04 14:38, 2026-07-05 18:39). **This is a separate, currently-live issue** — flagged below, out of scope for the phantom-short question but material and unaddressed.

## Queries run (staging DB, secondary signal)

- `strategy_executions` full range: 2026-04-02 to 2026-07-06, 204,384 rows.
- `opened_*` with `confidence_score >= 0.9` in `2026-07-01`–`2026-07-07` → **0 rows.**
- `reasons::text ILIKE '%risk_signal_confidence_1.0000%'` across all history → **45,524 rows**, but `GROUP BY date_trunc('day', timestamp)` shows this is concentrated almost entirely in **2026-04-10 through 2026-04-25** (a distinct historical episode, ~2,400/day during that stretch, tapering off after 04-25) — well outside and unrelated to the investigation window. **Zero occurrences in the 07-01–07-06 window.** (The April episode itself may be worth a separate look by quant/ML given its volume, but it predates and is unrelated to today's confirmed defect discovery — not chased further here per scope.)
- `system_events` prediction/timeout search → 0 rows.
- Railway logs for staging (`--since 5d`, filtered) → 0 matches, but same log-retention caveat as production applies (staging's current deployment also started today ~12:22 UTC).

## Verdict

**No evidence found in the inspectable window** that production (or staging) ever opened a trade, position, or high-confidence signal on a `price=0.0` / `PredictionTimeoutError` result during 2026-07-01 through 2026-07-06.

Specifically for the SHORT position the task flagged (`positions.id=22`, entered 2026-07-02 13:34:24 UTC, not 07-03 as approximately recalled — still OPEN as of this report):
- Entry confidence (0.25) and signal_strength (0.21) are unremarkable and far below the 1.0/1.0 phantom-bug signature.
- The full decision trail in the preceding ~30 minutes shows a smooth, monotonically strengthening short signal as price ran up — the normal pattern of a strategy re-affirming a directional thesis, not a single-bar artifact.
- No `price=0.0` predictions, no `predicted_return≈-1.0`, no `PredictionTimeoutError` string, anywhere in `strategy_executions.reasons` for the window.

Caveats / what could not be fully ruled out:
- **Railway log retention could not be used to corroborate or refute** — the current production and staging deployments both restarted today (12:22-12:30 UTC), well after the window in question, and `railway logs` only serves the current deployment's log stream. There is no accessible path from this environment to the actual application logs that were live during 2026-07-01–07-06. If Railway retains logs centrally beyond the deployment lifecycle (e.g. via a log-drain/export not wired up here), it was not reachable via `railway logs` CLI in this session.
- The `ml_predictions` JSON column on `strategy_executions` — the most direct place raw model output would be recorded — is **always JSON `null`, for every row in the table's entire history, in both prod and staging**. This is a pre-existing instrumentation gap (not caused by today's bug) that removes what would otherwise be the cleanest evidence source. Recommend flagging to ml-engineer/quant-researcher as a fix so a future occurrence of this exact bug (or a recurrence) would be directly visible instead of inferred from `reasons` text.
- `system_events` does not capture prediction-timeout events at all (only 4 event types exist), so its silence is expected, not proof of absence.
- The 5 historical max-confidence (1.0/1.0) `opened_short` rows in prod (Mar 29, Apr 3, Jun 2 x2, Jun 3) and the large staging cluster (Apr 10-25) were not exhaustively verified against price-feed ground truth to rule out this same bug firing on *those* dates — they are out of the requested window (2026-07-01 to 07-06) and out of scope for this report, but worth a follow-up if the team wants full-history remediation triage, since a `confidence_multiplier >= 1.0` makes a `predicted_return = -1.0` (the timeout artifact) mathematically indistinguishable in `confidence_score` alone from a genuine max-strength real signal — only the `price` field (0.0) or `reasons`/`ml_predictions` would disambiguate, and neither is populated reliably enough pre-fix.

## Out-of-scope finding surfaced during this investigation (flagging, not acting)

Production `system_events` shows an **active, unresolved incident**, separate from the phantom-short question:
- Recurring `WARNING`/`ALERT`, `component=connectivity`, `error_code=USER_WS_DEGRADED`: "User data stream circuit-open after N reconnects — REST-degraded; real-time fills/balance updates unavailable", first seen 2026-07-04 14:38:02 UTC, recurring roughly hourly with exponentially growing reconnect counts (3 → 10 → 40 → 80 → 160 → 280 → 400 → 520 → 640 → 760 → 880 → 1120 → 1240 → 1360), last seen in the queried window at 2026-07-06 11:39:25 UTC (the current deployment restarted ~1h after that, at 12:30 UTC, which may have cleared it — not confirmed).
- Companion event: `WARNING`, `error_code=NO_ALERT_CHANNEL`, "No alert channel configured — operator alerts will not be delivered" (2026-07-04 14:38:02 and 2026-07-05 18:39:43 UTC). **This means the above circuit-open alerts were never delivered to a human** — they only exist as DB rows. This matches the known lesson `project_binance_margin_ws_churn.md` (user stream degrades to REST until restart — issues #723/#724 were opened for this class of problem previously) and should be cross-checked against whether #723/#724 are still open or reintroduced by a recent change (the current deploy is PR #911 / #908, "hold user-data WS subscription fix" — plausibly related; worth checking if this fix actually addresses the reconnect-storm or is orthogonal).
- Not evaluated further here (out of scope for this forensics task) — recommend a dedicated live-ops/on-call pass to determine current status post-restart and whether #723/#724 (or a new incident) should be opened for the alert-channel gap specifically, since "no alert channel configured" is itself a P1-grade observability hole regardless of the WS issue's current state.
