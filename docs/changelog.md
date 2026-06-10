# Changelog

All notable changes to the AI Trading Bot project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Maintainer Note**: This is a living document. Update after completing features, bug fixes, or significant changes. Use the `/update-docs` command to auto-populate entries.

---

## [Unreleased]

### Fixed
- Periodic reconciler now books realized P&L when it detects a filled
  stop-loss. Both detection paths previously corrupted tracked capital:
  the stop-verification branch closed the DB row with NO balance update
  (and no trade record), and the margin holdings check misclassified a
  just-filled short stop-loss (AUTO_REPAY zeroes the borrow) as
  "externally closed" (the spot holdings check had the identical flaw:
  a filled stop also empties the held balance), closing the row with no
  exit price at all. Every
  SL loss the reconciler processed before the engine's deferred-exit
  drain (~equal ~2-minute cadences, so a large fraction) silently never
  hit the balance → overstated capital → oversized subsequent positions.
  Both the margin and spot holdings checks now consult the tracked stop
  order before classifying an external close.
  Both paths now delegate to the startup reconciler's filled-SL handler
  (#731): DB close first (a failed close leaves the position tracked for
  retry), P&L with USD-normalized commission and margin interest, plus a
  deduplicated `trades` row. The periodic wrapper skips when the engine's
  deferred-exit drain already processed the fill (no double-booking) and
  defers classification (fail-closed) when the stop's state cannot be
  confirmed.
- OrderTracker no longer converts an API outage into a position deletion.
  After `MAX_API_ERROR_RETRIES` (10) consecutive failed/`None` polls
  (~50 s at the live 5 s interval) the tracker fired `on_cancel`, and
  `_handle_order_cancel` popped the (possibly live) position from the
  tracker and refunded its entry fee — manufacturing untracked exchange
  exposure, a corrupted balance, and room for a double entry on the next
  signal, exactly during exchange API degradations (LESSONS §1.8 fail-open
  class). Polling give-up now routes to a new `on_tracking_lost` callback;
  the engine's `_handle_order_tracking_lost` keeps the position tracked,
  leaves the balance untouched, and escalates with a critical
  `system_events` row (`ORDER_TRACKING_LOST`) + webhook alert so the
  periodic reconciler resolves the order's true state from the exchange.
  `on_cancel` now fires only for exchange-confirmed terminal states.
- Closed live `trades` rows now persist `commission` and `quantity` (previously
  always `0` / `NULL`). The live close path (`LiveTradingEngine._close_position`
  and the offline stop-loss reconciliation path) now passes `commission` and
  `quantity` to `DatabaseManager.log_trade`, which already supported both.
  `trades.commission` is the round-trip fee in **USD** (`entry_fee + exit_fee`) —
  the same values booked to `account_balances` (entry as the `entry_fee_<symbol>`
  ledger event, exit folded into `realized_pnl_<symbol>`), **not** the raw
  `orders.actual_commission` (which is denominated in the received asset and
  populated asynchronously, so unit-ambiguous and unreliable at close time).
  `trades.quantity` is the actual filled base quantity, scaled by
  `current_size/original_size` for partially-exited positions (NULL for scale-in
  positions, whose held quantity is not derivable, and for corrupt sizing).
  `DatabaseManager._trade_net_pnl` now also subtracts `commission`, so true net P&L
  (`pnl - commission - margin_interest_cost`) flows through performance metrics and
  `recover_last_balance` reconstruction — correcting a latent overstatement now that
  commission is populated (historical rows carry `commission = 0` and are unaffected).
  For positions recovered after a restart, the entry-fee leg is reconstructed from the
  fee model (the `positions` table does not persist entry fee) rather than dropped, and
  scaled to the closed portion so a partial final close's commission matches its
  portion-level pnl/quantity. The `PositionReconciler` offline stop-loss path
  (`_realize_pnl_on_close`) now also inserts a `trades` row — previously it
  balance-corrected and DB-closed the position but recorded **no trade at all** (deduped
  via the exit order id + `uq_trade_order_session`). `LiveExecutionEngine` now converts an
  exchange fill commission to USD via its `commission_asset` (a base-asset commission on a
  buy, e.g. ETH, is priced into USD; an unconvertible asset like BNB falls back to the
  modelled fee) — fixing a latent bug where a base-asset commission could be booked as if
  it were USD. Relatedly, `_recover_active_positions` now hydrates
  `original_size`/`current_size` and partial-operation counters from the DB, so a position
  partially exited before a restart closes at its remaining size. The commission→USD
  conversion is shared via `src/engines/shared/commission.py` and applied on the
  reconciler offline-SL path too (a short's stop-loss is a base-asset buy), so it is
  never booked wrong-unit. The reconciler logs its trade row only after the DB position
  is actually closed and with a stable, non-NULL dedup key (real exit order id, else a
  synthetic id from the position) so a re-run cannot insert a duplicate
  (`uq_trade_order_session`; NULL≠NULL in Postgres) — guarding the #657/#668 phantom-trade
  class; a failure to persist the row after the balance was corrected now escalates to
  CRITICAL rather than a silent warning. See the "Trade fee accounting" note in
  `docs/live_trading.md`.
- Reconciler accounting hardening (review follow-ups): the offline stop-loss close now
  realizes P&L **only after** the DB position is actually closed (a failed close no longer
  double-subtracts P&L on the next reconcile), and a failed balance write skips the audit +
  trade row (no `trades`/`account_balances` divergence). Fees route through the shared
  `CostCalculator` (no duplicated fee modelling); the SL exit-fee fallback and the recovered
  entry/exit reconciler bookings now normalize commission to USD via `commission_asset` like
  the rest of the change. A scaled-in position closed by the reconciler stores NULL quantity
  and an un-inflated entry fee, matching the engine close path. `_extract_base_asset` now
  delegates to the shared `split_base_quote`. The mock DB enforces `uq_trade_order_session`
  so the dedup path is unit-tested.
- Live engine hard-disables partial exits / scale-ins behind the default-OFF
  `live_partial_operations` feature flag (#734). The live engine executed
  partial operations as bookkeeping only — `_execute_partial_exit` /
  `_execute_scale_in` mutate the tracker/DB but **never place an exchange
  order** — and with mismatched units (policy fractions of the original
  position applied to fraction-of-balance state), so on a real account a
  winner reaching the default +2%/+3% triggers desynced tracked size from
  actual holdings (stranded inventory, un-repaid margin borrows, -2010 close
  failures), booked phantom realized PnL, and freed daily-risk budget that
  was still deployed. All three activation paths are gated (constructor,
  strategy hot-swap overrides, runtime policy hydration via the existing
  opt-in state). Re-enable only for development of the #734 fix.
- Reconciler no longer places a DUPLICATE stop-loss when an order lookup
  fails transiently (#713). `BinanceProvider.get_order` swallows every
  exception into `None`, and both stop-loss verifiers (startup
  `PositionReconciler._verify_stop_loss` and the periodic reconciler's
  stop-verification loop) treated `None` as "stop missing" — clearing the
  tracked `stop_loss_order_id` and re-placing a new stop while the original
  could still be resting on the exchange (reserving base/margin, able to
  cause -2010 on a later close, and able to flip the position if both
  stops fill). Added a fail-closed `ExchangeInterface.get_order_checked`
  (Binance override returns `None` only on a confirmed -2013
  "order does not exist" and raises `OrderLookupError` on any unconfirmed
  lookup), and both verifiers now skip the cycle on an unconfirmed lookup
  instead of re-placing. Confirmed-missing stops are still re-placed.
- Live trade recovery on the `emergency_sync` path no longer silently fails.
  `AccountSynchronizer.recover_missing_trades` called
  `DatabaseManager.log_trade(order_id=...)`, but `log_trade` has no `order_id`
  parameter (the field is `exit_order_id`) and no `**kwargs`, so every recovered
  trade raised `TypeError` — swallowed by the per-trade `except` — and was never
  persisted to the ledger. Maps `trade.order_id` onto `exit_order_id` (which feeds
  the `Trade.order_id` column). Adds a regression test that drives
  `recover_missing_trades` with an autospec'd `DatabaseManager`, so the real
  `log_trade` signature is enforced. Also clears pre-existing mypy loop-variable
  and ruff `UP038` debt on `account_sync.py` (behaviour-neutral).
- Margin-equity balance corrections are now audited and alertable.
  `margin_equity_sync_correction` book-downs (written by
  `AccountSynchronizer._sync_margin_equity`) previously updated the balance ledger
  without recording a `reconciliation_audit_events` row or a warning-level
  `system_events` row, so the single largest capital event a margin session can
  produce was invisible to monitoring/auditing — a −$15.75 (−15.8%) production
  book-down on 2026-06-03 (and a second −$1.37 on 2026-06-05) left zero audit
  trail. The path now emits both records via a new best-effort
  `_record_equity_correction_audit` helper: an immutable audit row
  (`entity_type='balance'`, `field='total_balance'`, before/after values, severity
  `HIGH`, escalating to `CRITICAL` when divergence ≥ 5%) and a `BALANCE_ADJUSTMENT`
  system event at `warning` severity (`critical` when ≥ 5%) so alerting can see large
  book-downs. Both writes are independently guarded so a logging failure can neither
  raise into the sync loop nor unwind the already-persisted correction; emission is
  skipped entirely if `update_balance` itself reports failure (no audit for a
  correction that never persisted). The audit binds to the same session the balance
  write used — resolved via `update_balance`'s own `_current_session_id` fallback — so
  the first post-restart correction (when `AccountSynchronizer.session_id` has not yet
  been assigned) is captured too, not just steady-state periodic syncs.
- Live position/trade recovery no longer crashes on `Decimal`-vs-`float`
  arithmetic. `DatabaseManager.get_active_positions` and `get_recent_trades`
  now coerce SQLAlchemy `Numeric(18,8)` columns (which read back from
  PostgreSQL as `Decimal`) to `float` at the source — `float()` for
  non-nullable columns, `_to_optional_float()` for nullable ones — mirroring
  the existing `orders_data` block and `LivePositionTracker.recover_positions`.
  Previously these raw `Decimal`s flowed through `_recover_active_positions`
  into recovered `Position` objects and raised `unsupported operand type(s)
  for *: 'decimal.Decimal' and 'float'` in reconciliation's default
  stop-loss branches (`entry_price * (1.0 ± DEFAULT_STOP_LOSS_PCT)`), which
  run *before* the `place_stop_loss_order` boundary that PR #653 had patched.
  Also keeps dashboard consumers JSON-serializable (`json.dumps` raises on
  `Decimal`).
- Live restart balance recovery no longer crashes or silently resets on
  `Decimal`-vs-`float` arithmetic (same `Numeric` class as above, balance path).
  `DatabaseManager.recover_last_balance`'s trades fallback computed
  `initial_balance + net PnL`, raising `TypeError` on `Decimal + float` — swallowed
  by `_recover_existing_session`, which then returned `None` and reset the engine to
  its default balance on restart. With no trades it returned a raw `Decimal` that
  later broke `_print_final_stats`' float arithmetic on shutdown
  (`unsupported operand -: Decimal and float`). The fallback now coerces
  `float(initial_balance)`; `_recover_existing_session` coerces the recovered value
  to `float` and fails fast (raises) on a non-finite balance *before* its `> 0`
  positivity filter, so corrupt persisted state can never reach position sizing or
  silently fall back to the default balance.
- Backtest-live engine parity: closed nine silent divergences. Backtest now
  propagates `TimeExitPolicy`-specific exit reasons (`"Max holding period"`,
  `"Weekend flat"`, etc.) instead of hardcoding `"Time limit"`; gained an
  optional `annual_margin_interest_rate` parameter on `Backtester` mirroring
  live's `MarginInterestTracker` (default `0.0` preserves spot-mode
  behaviour); now sums `entry_fee + exit_fee + margin_interest_cost` into
  `PerformanceTracker.record_trade` matching live's total-fee semantics;
  persists `margin_interest_cost` to the `trades` DB column via
  `EventLogger.log_completed_trade`. Live now wires `CorrelationHandler`
  into `LiveEntryHandler` and threads the full `symbol/timeframe/df/index`
  context through `_check_entry_conditions` so correlation-driven sizing
  reduction actually fires; backfills historical sentiment over the full
  buffer before overlaying the live snapshot so ML strategies get
  equivalent inputs; sweeps the position tracker after reconciliation to
  register reconciler-created positions with the risk manager (using
  `current_size` to preserve partial-exit accounting); passes the live
  positions list to the direct `ComponentStrategy.process_candle` path.
  Documented tick-size rounding, margin interest, and single-vs-multi-position
  as known parity caveats on the `Backtester` docstring.
- Live trading engine no longer shuts itself down during transient database
  outages. Transient DB-connectivity errors (DNS resolution failures, dropped
  connections, brief Postgres unavailability) are now classified and *ridden
  out* with a bounded backoff instead of counting toward
  `max_consecutive_errors`. This was the root cause of the 2026-05-19 incident:
  a multi-hour Railway internal-DNS outage made `postgres.railway.internal`
  unresolvable, every loop iteration raised `OperationalError`, the
  consecutive-error limit tripped, and **both the staging and production bots
  went offline — silently — for ~12 days**. `pool_pre_ping` reconnects
  automatically once the database returns. Permanent faults (bad credentials,
  missing role/database, permission denied) are excluded and still fail fast,
  and an outage lasting more than 30 minutes drops the engine into close-only
  mode (new entries suspended; exits and server-side stop-losses continue).
- Prediction-cache performance test (`test_cache_performance_characteristics`)
  is no longer timing-flaky on loaded CI runners. It previously took the *mean*
  of `time.time()` over 100 cold, fully-mocked operations and asserted cache-hit
  was within 5× a tiny (~0.13ms/op) noise-dominated cache-miss baseline, so a
  single GC/scheduler pause inflating the mean would trip it (it failed twice on
  PR #637). It now warms up, samples many ops with `perf_counter`, and asserts
  the *median* (immune to those outliers) against a generous absolute budget. It
  is also marked `@pytest.mark.performance` so it runs in the nightly performance
  workflow rather than the blocking PR integration gate.

### Added
- Heartbeat staleness monitor (`scripts/check_heartbeat.py` +
  `.github/workflows/heartbeat-monitor.yml`): a scheduled, read-only CI job that
  fails (notifying maintainers) when an active trading session's
  `account_history` snapshot goes stale beyond a threshold (default 2h) — the
  canonical liveness signal. Requires the `RAILWAY_STAGING_DATABASE_URL` /
  `RAILWAY_PRODUCTION_DATABASE_URL` repository secrets.

### Changed
- `railway.json`: raised the Trading Bot `restartPolicyMaxRetries` from 3 to 10
  so Railway keeps retrying through longer transient infrastructure failures.
- `/deploy-staging` and `/deploy-prod` slash-command skills rewritten to match
  the actual `develop` → `staging` → `main` promotion workflow (Railway
  development → staging → production). `/deploy-prod` no longer does
  `git reset --hard origin/staging` + force-push to `main` (which would rewrite
  production history and drop the "Promote to production" commits that live only
  on `main`); it now opens an additive **"Promote to production" PR**
  (`staging`→`main`), reconciles changelog conflicts by merging `main` into
  `staging`, waits for green CI, and merges with a merge commit. `/deploy-staging`
  syncs the long-running `staging` branch to `develop`.
- Protected the `staging` branch (`allow_deletions: false`, force-push still
  allowed) so the repo-wide `delete_branch_on_merge: true` no longer
  auto-deletes it when a `staging`→`main` promotion PR is merged. `staging` is a
  long-running branch bound to the Railway staging environment and must persist.

### Security
- Hardened a batch of security findings from a repo-wide scan (bandit + manual
  audit):
  - Monitoring dashboard: added a token auth guard (`MONITORING_DASHBOARD_TOKEN`)
    on state-changing/data-leaking endpoints (`POST /api/balance`,
    `POST /api/config`, `POST /api/debug/fix-positions`, `GET /api/debug/positions`).
    Fails closed in production when no token is set; warns-and-allows only in
    explicit dev/test envs. Restricted Socket.IO CORS from `"*"` to same-origin
    (override via `MONITORING_CORS_ALLOWED_ORIGINS`).
  - SageMaker artifact extraction now validates tar members (rejects path
    traversal / zip-slip and escaping symlinks). Model-registry sync validates
    `version_id`/`model_type` from `metadata.json` and asserts the resolved path
    stays inside the registry before any `rmtree`/`copytree`/`symlink`.
    S3 artifact download skips object keys that escape the target directory.
  - `get_secret_key()` now fails closed: an unset `ENV`/`FLASK_ENV` is treated as
    production instead of silently returning the public dev key.
  - Admin UI login compares the username in constant time (`hmac.compare_digest`).
  - `atb data cache-manager --detailed` uses a restricted unpickler (allowlisted
    pandas/numpy types) instead of raw `pickle.load` on legacy `.pkl` files.
  - JUnit XML parsing uses `defusedxml` (XXE / billion-laughs hardening).
  - Tightened temp-shim permissions to `0o700`; quoted/validated table
    identifiers in the DB integrity check; marked the dashboard's `0.0.0.0`
    bind intentional.

### Added
- **Monitoring dashboard mobile layout**: V2 dashboard reflows below 768px to a
  bottom tab bar + stacked content + inline inspector. Reuses the same React
  store and data flow; layout swap driven by `useIsMobile()` hook backed by
  `window.matchMedia` with a resize listener so it adapts live. iOS safe-area
  insets respected via `viewport-fit=cover` + `env(safe-area-inset-*)`.
- **Monitoring dashboard V2 redesign**: chart-led layout with left-rail nav
  (Dash / Pos / Strat / Trades / Risk / Logs), KPI strip, hero equity chart
  with overlay toggles (benchmark / trades / drawdown), positions strip, and
  a swappable right inspector. Light + dark themes (toggle persisted to
  `localStorage`). Tech stack swap: Bootstrap + Chart.js → React 18 (UMD) +
  Babel-standalone + socket.io-client. CDN scripts pinned with SRI hashes.
- New `GET /api/dashboard/state` endpoint bundles metrics + positions +
  trades + bot meta in a single request to keep first paint snappy. Accepts
  `?trades_limit=` (clamped to 1..500). Falls back to per-resource fetches
  in the JS adapter if the bundled endpoint is unavailable.
- New `MonitoringDashboard._get_bot_meta()` reads strategy / symbols /
  timeframe / mode / `max_open_positions` from the most recent **running**
  `trading_sessions` row (falls back to the most recent overall row),
  matching the "Exchange Mode & Account Type Safety" guidance so a stale
  paper-mode session can't mask an active live one.
- `.claude/launch.json` — preview-server configurations for all three
  dashboards plus live-health.
- Experimentation framework (`src/experiments/`) with declarative YAML suites,
  `atb experiment run|list|show|promote` CLI, ranked reporter with statistical
  verdicts, file-based ledger under `experiments/.history/`, and promotion
  writer for `StrategyVersionRecord`/`ChangeRecord` plus patch YAML emission.
- ML signal generators now expose `long_entry_threshold`, `short_entry_threshold`,
  `confidence_multiplier`, and regime-specific thresholds as overridable instance
  attributes (class constants remain as defaults).
- `ConfidenceWeightedSizer` gained `min_confidence_floor` parameter.
- `create_ml_basic_strategy`, `create_ml_adaptive_strategy`, and
  `create_ml_sentiment_strategy` accept the new tuning knobs.

### Removed
- Deleted the unused first-attempt optimizer layer: `src/optimizer/analyzer.py`,
  `validator.py`, `strategy_drift.py`, the `atb optimizer` CLI, the
  `OptimizationCycle` DB model/table, `DatabaseManager.record_optimization_cycle`,
  `fetch_optimization_cycles`, and the `/api/optimizer/cycles` dashboard route.
  Alembic migration `0011_drop_optimization_cycles` drops the table.
- Renamed `src/optimizer/` → `src/experiments/` now that the package reflects
  its actual purpose. `atb walk-forward` continues to work via
  `src/experiments/walk_forward.py`.

### Fixed
- Binance margin-WS keepalive noise + user-stream watchdog gap (#608).
  `python-binance==1.0.36` multiplexes margin user-data subscriptions over a
  shared `ws_api` connection that Binance closes every ~2 min with WS code
  1011 'keepalive ping timeout'. The library's reconnect machinery recovers
  but each cycle surfaces an unretrieved-task exception on the asyncio
  default handler (~720/day on prod). Added
  `BinanceWSKeepaliveFilter` (rate-limits to one full traceback per 60s
  window with a periodic suppression summary) and extended
  `BinanceProvider.ws_healthy` to fail when the user/margin stream is
  configured but stale or non-PRIMARY (was previously kline-only, masking a
  permanently-dark user stream). New `user_ws_healthy` property exposes the
  user-stream status directly.
- `BinanceWSKeepaliveFilter` now also matches the ws_api subscribe-timeout
  signature (GH #608 follow-up). #609's filter only matched the 1011
  'keepalive ping timeout' close code, which never fires on prod — the
  actual ~2-min churn is the margin `userDataStream.subscribe` request
  timing out after 10s (`BinanceWebsocketUnableToConnect: Request timed
  out`), which carries no 'keepalive ping timeout' text and so was never
  suppressed. Replaced the single fingerprint with `KEEPALIVE_MARKER_GROUPS`
  (match all markers in any group); a `binance/ws/` anchor prevents
  swallowing connection errors raised by our own code.
- Add ban-aware retry to Binance client startup — parses `-1003` ban expiry and sleeps until lifted instead of crashing (#590)
- `hyper_growth`: fix silent-SELL bug caused by feature-shape mismatch
  (#603). The factory wired `MLBasicSignalGenerator(model_type="sentiment")`
  but fed the sentiment model the 5-column price-only feature tensor
  instead of the 10 columns it was trained on. The model returned 0.0 on
  every bar, which the generator converted to `predicted_return=-1.0` and
  emitted as a constant SELL with confidence=1.0. Swapped to
  `model_type="basic"` (real directional edge of 55-57% BUY accuracy at
  12-24h horizons). Also tightened the default `stop_loss_pct` from 0.20
  to 0.10. On BTCUSDT 1h 2024: 14.16% → 99.80% return, 7.24% → 4.74% max
  drawdown, 0.055 → 0.259 Sharpe.

---

## 2026-02-18

### Infrastructure
- Added minimal CI dependencies and enabled tests in Claude GitHub workflow (#551)
- Added Claude Code GitHub Workflow (#543)

---

## 2026-01-15

### Added
- Automated cloud training with auto data download/upload (#532)
- CoinGecko data provider as Binance alternative (#538)
- Feature schema saving with trained ML models (#530)
- `--changed` flag to run quality checks only on modified files (#529)
- Code review agents and deployment slash commands
- Automated quality checks hook for Python files
- Side utilities and validation utility modules (#500)
- Order-type execution modeling for live and backtest (#493)

### Changed
- Consolidated backtesting and live engines into unified architecture (#527)
- Removed deprecated `src/indicators` directory (#515)
- Refactored strategies for improved code quality and maintainability (#501)
- Improved ML training and cloud module code quality (#502)
- Used shared `pnl_percent` function for engine parity (#505)

### Fixed
- Prevented race conditions in position tracking (#528)
- Addressed infrastructure code quality and safety issues (#513)
- Resolved database manager bugs and improved financial data safety (#512)
- Comprehensive position management code quality and safety improvements (#507)
- Critical issues in risk management module (#509)
- Comprehensive input validation for performance module (#508)
- Made regime regression test deterministic with dependency injection (#504)
- Used relative comparison in cache performance test (#540)

### Documentation
- Comprehensive risk management architecture documentation (#518)
- Updated docs and CLI commands for cache and migrations (#533)
- Added common PR review issues to CLAUDE.md (#499)
- Added instructions to run review agents after significant changes

---

## 2025-12-28

### Added
- Stop hook with completion detection for Claude Code Web
- PSB system analysis documentation (`docs/PSB_SYSTEM_ANALYSIS.md`)
- Automated documentation system (changelog.md, project_status.md, architecture.md)
- `/update-docs` slash command for documentation maintenance
- Shared entry utilities and validation helpers for consistent engine behavior
- Comprehensive engine parity test coverage (#487)
- Correlation sizing adjustments for runtime entries (#483)

### Changed
- Enhanced CLAUDE.md with Railway environment guidelines
- Unified backtest/live entry and partial-exit logic via shared helpers
- Refactored live entry execution to use LiveEntryHandler & LiveExecutionEngine (#482)
- Routed filled live exits through LiveExitHandler (#485)
- Completed shared engine models consolidation (#475)

### Fixed
- Fixed post-fee entry balance in live entry paths (#491)
- Aligned live engine dynamic risk handling (#490)
- Honored take-profit limit pricing (#489)
- Added missing order tracking columns to positions table migration
- Recorded live exits even when filled prices exceed deviation thresholds

### Documentation
- Updated documentation links in READMEs (#488)
- Added comprehensive backtesting engine audit report (#476)
- Added performance tracker integration execplan (#467)

---

## 2025-12-22

### Changed
- Removed outdated workflows for cursor reviews and nightly code quality

---

## 2025-12-21

### Added
- Nightly performance test workflow for CI (#438)

### Changed
- Optimized ML training pipeline with performance improvements (#439)
  - Batch processing enhancements
  - Memory efficiency improvements

### Documentation
- Clarified merge-develop command in documentation
- Updated AGENTS.md with detailed execplan storage guidelines
- Enhanced PR creation guidelines for clarity

---

## 2025-12-20

### Changed
- Refactored trading bot for better code quality (#437)
  - Code organization improvements
  - Enhanced maintainability

### Documentation
- Updated CLI command consistency and accuracy across docs
- Clarified live-health invocation across guides (#429)
- Fixed broken link in prediction README (#428)

---

## 2025-12-19

### Changed
- Refactored prediction model registry and usage (#421)
  - Improved model loading patterns
  - Enhanced registry structure

### Documentation
- Updated data pipeline and model registry docs (#416)
- Refreshed nightly documentation set (#427)
- Changed documentation scan workflow from nightly to weekly

---

## Earlier Changes

For changes prior to December 2025, see the git history:
```bash
git log --oneline --since="2025-01-01"
```

---

## Categories

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Features to be removed in future versions
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes
- **Documentation**: Documentation-only changes
- **Infrastructure**: CI/CD, deployment, and tooling changes
