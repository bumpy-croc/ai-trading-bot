# Live trading

> **Last Updated**: 2026-06-15
> **Related Documentation**: [Backtesting](backtesting.md), [Monitoring](monitoring.md), [Database](database.md)

`src/engines/live/trading_engine.py` powers the real-time execution stack. It shares core building blocks with the backtester while adding
continuous polling, account synchronisation, and resilience features required for production trading.

## Engine highlights

- **Safety first** – the runner defaults to paper trading. Passing `--live-trading --i-understand-the-risks` is required to send
  orders to the exchange. Consecutive failures trigger configurable cooldowns and the engine stops after repeated errors.
- **Exchange adapters** – `BinanceProvider` and `CoinbaseProvider` implement the `DataProvider` interface. The runner can load
  either via the `--provider` flag, or switch to `MockDataProvider` for dry runs.
- **Risk controls** – `RiskManager`, `DynamicRiskManager`, trailing stops, correlation limits, and partial exit policies are
  available just like in the backtester. Position updates emit structured events through `log_engine_event`, `log_order_event`,
  and `log_risk_event`.
- **Account synchronisation** – `AccountSynchronizer` periodically reconciles balances, open positions, and open orders using the
  exchange API (`src/engines/live/account_sync.py`). It stores the results through `DatabaseManager` so restarts can resume from the last
  known state.
- **Sentiment and regime inputs** – pass a `SentimentDataProvider` (Fear & Greed) or enable the `RegimeStrategySwitcher` to swap
  strategies when market conditions change.

## Handler decomposition & lock ownership (#486)

`LiveTradingEngine` orchestrates; exchange- and observability-facing work is delegated:

| Component | Module | Responsibility | Lock ownership |
|-----------|--------|----------------|----------------|
| `LiveStopLossManager` | `engines/live/execution/stop_loss_manager.py` | All exchange-facing stop-loss calls: placement (with retry), cancel, fill/held queries, re-protect, offline-fill detection | None — stateless; reads `enable_live_trading`/`exchange_interface`/`order_tracker` off the engine at call time; position mutations go through `LivePositionTracker`'s internal lock |
| `LiveAccountMonitor` | `engines/live/monitoring/account_monitor.py` | Balance/equity snapshots, status lines, performance summaries | None — stateless; reads positions via `LivePositionTracker.positions` (thread-safe snapshot) |
| `LiveSessionRecoverer` | `engines/live/recovery.py` | Startup recovery: session balance, persisted-position reload, risk-manager re-registration, startup exchange reconciliation | None — runs on the startup path before the trading loop; engine state it mutates (session id, balance, close-only flag) is written through the engine as before |
| `LiveStartupSequencer` | `engines/live/startup.py` | Bootstrap orchestration (`start()` delegates to `run()`): session recover/create + wiring, #668 carry-forward, #657 self-heal, account sync, runtime-service startup, main-loop launch | None — runs once on the startup path (main thread) before the trading loop; all engine state written through the backref as before |
| `StrategyRuntimeCoordinator` | `engines/live/strategy_runtime.py` | Strategy normalization, component risk-context provider, runtime dataframe prep + `RuntimeContext` construction, per-candle runtime decision processing, risk-param merge/clone | None — reads/writes engine strategy-runtime state (`strategy`, `_runtime`, `_runtime_dataset`, context cache) at call time; all touched only on the single trading-loop thread (per-candle + loop-applied hot-swap) |
| `StrategyHotSwapCoordinator` | `engines/live/strategy_hot_swap.py` | Public `hot_swap_strategy`/`update_model`, StrategyManager callbacks, loop-applied pending-update application, post-swap refresh of trailing-stop/partial-ops/time-exit policies + component risk re-binding | None — entry points/callbacks only queue a `StrategyManager`-locked pending update (caller thread); all engine-state mutation runs in `apply_pending_strategy_update` on the trading-loop thread |
| `WebSocketHealthMonitor` | `engines/live/ws_health.py` | WS stream startup, the background health-monitor thread, kline/user-stream staleness + reconnect/probe decisions, degraded-user hard-reconnect, draining the order-fill exit queue on the loop thread | None — the monitor holds no locks. It owns a single daemon thread (handle on the engine, `state._ws_health_thread`) and writes its reconnect-failure counters / `_ws_kline_active` flag only from that one thread (GIL-atomic single-writer); the trading loop reads them. Cross-thread order-fill handoff goes through the thread-safe `_pending_fill_exits` `SimpleQueue`, drained on the loop thread. Provider-owned WS connection state is read-only here |
| `LiveEntryCoordinator` | `engines/live/execution/entry_coordinator.py` | Entry decision (signal/sizing/SL-TP derivation) + base-asset-locked order execution: duplicate/limit guards, balance+fee accounting, position tracking, risk re-registration, stop-loss placement, emergency-close fallbacks | Serialises each entry on the symbol's base-asset lock (`state._base_asset_locks`) across submit → track → SL placement; the lock lives on the engine and the emergency-close fallback re-acquires it re-entrantly via `state._execute_exit` (#703). Runs on the trading-loop thread; all engine state mutated through the backref as before |
| `LiveExitCoordinator` | `engines/live/execution/exit_coordinator.py` | Exit decision (`check_exit_conditions`: per-position SL/TP/runtime evaluation + strategy-execution logging) and base-asset-locked close (`execute_exit` → `execute_exit_locked`): resting-stop cancel-before-close (#710), realized-PnL + margin-interest accounting, balance update, trade persistence/CLOSED flip (#657), re-protect on failed close | Serialises each close on the symbol's base-asset lock (`state._base_asset_locks`, #703); the lock lives on the engine and is re-entrant (an entry's failed-SL emergency close routes through the engine's `_execute_exit` wrapper on the same thread). `check_exit_conditions` invokes the close via `state._execute_exit` (the engine wrapper) so engine-level test mocks still intercept. Runs on the trading-loop thread; all engine state mutated through the backref as before |
| `LiveDynamicRiskCoordinator` | `engines/live/dynamic_risk_coordinator.py` | Per-entry dynamic-risk position-size adjustment (drawdown/peak-aware, via the shared `DynamicRiskHandler`) + its observability/audit logging | None — holds no state of its own. Runs on the trading-loop thread; called by the loop and by `LiveEntryCoordinator` via the engine `_apply_dynamic_risk_adjustment` wrapper |
| `LiveLoopTimingCoordinator` | `engines/live/loop_timing.py` | Trading-loop cadence + data-freshness helpers: interruptible sleep, activity/time-of-day-aware poll interval, candle-age / WS-buffer freshness gate | None — leaf helpers, no order placement/balance mutation/engine-method calls. Runs on the trading-loop thread; `is_data_fresh` is also reached by `LiveMarketDataCoordinator` via the engine `_is_data_fresh` wrapper. Holds no state of its own |
| `LiveMarketDataCoordinator` | `engines/live/execution/market_data_coordinator.py` | Per-candle read path: latest-frame fetch (WS-cache vs REST, resync handling), sentiment enrichment, the strategy-context readiness gate, and correlation-sizing context | None — read path, no order placement/balance mutation (only writes `last_data_update`). Runs on the trading-loop thread; `build_correlation_context` is also called by `StrategyRuntimeCoordinator` via the engine wrapper. Holds no state of its own; defers freshness to the engine's `_is_data_fresh` via `state` |
| `LiveOrderFillCoordinator` | `engines/live/execution/order_fill_coordinator.py` | The `OrderTracker` callbacks: full fill (queues stop-loss-fill closes), partial fill (critical SL-partial alert), cancel/reject (entry-fee refund for the unfilled fraction + stop-loss-cancel escalation #741), tracking-lost (fail-closed: keep position, escalate) | None — holds no coordinator-local state. Runs on the **OrderTracker poll thread**: a stop-loss fill is handed to the loop via the thread-safe `state._pending_fill_exits` SimpleQueue (never closed inline, #631); position reads/mutations use `LivePositionTracker`'s thread-safe copy + atomic `pop_position`/`set_stop_loss_order_id`; the cancel refund uses `db_manager.atomic_balance_update`. `_record_event`/`_send_alert` invoked via `state` so engine mocks still intercept |
| `LivePositionTracker` | `engines/live/execution/position_tracker.py` | Position state | Owns `_positions_lock`; `positions` property returns a defensive copy |
| `OrderTracker` | `engines/live/order_tracker.py` | Order fill polling + callbacks | Owns its internal lock; engine callbacks (now `LiveOrderFillCoordinator`) defer closes to the trading loop via `_pending_fill_exits` |
| `SharedEntryHandlerMixin` | `engines/shared/execution/entry_handler_mixin.py` | Entry-plan extraction + dynamic-risk sizing, identical for backtest and live | None — delegates to shared `DynamicRiskHandler` |

The engine itself keeps thin private wrappers (`_cancel_stop_loss_order`, `_check_stop_loss_filled`, `_log_account_snapshot`, …) so
call sites and test mock points are stable while the implementations live in the modules above.

## State recovery & account sync

- The engine resumes balances and open positions from the last `trading_sessions` snapshot when `resume_from_last_balance=True`
  (the default). Balance updates feed into risk sizing so restarts continue with the correct exposure.
- `account_snapshot_interval` controls periodic reconciliations (default: 3600 seconds). Each pass checks balances, positions,
  and order status against the exchange and records adjustments for auditing.
- Trigger an emergency reconciliation whenever you suspect drift (for example after manual exchange trades):

    ```python
    from src.data_providers.binance_provider import BinanceProvider
    from src.database.manager import DatabaseManager
    from src.engines.live.account_sync import AccountSynchronizer

    sync = AccountSynchronizer(BinanceProvider(), DatabaseManager(), session_id=<current_session_id>)
    sync.emergency_sync()
    ```

## Max-drawdown hard cap (close-only halt)

The live engine enforces `portfolio.max_drawdown_pct` from `.claude/state/risk-limits.json`
(0.20, mirrored by `DEFAULT_MAX_DRAWDOWN` / `RiskParameters.max_drawdown`). On every
trading-loop iteration `MaxDrawdownEnforcer` (`src/engines/live/monitoring/drawdown_guard.py`)
measures the drawdown of the current balance from the **session peak balance** — the same
numbers `account_history.drawdown` is derived from.

- **Peak baseline**: the `account_history` session max is **authoritative** — seeded on boot
  from `max(account_history.balance)` for the active session (plus the recovered inactive
  session on clean restarts), never below the current recovered balance. The in-memory
  performance-tracker peak is deliberately **not** a seed candidate: it can initialize from
  the configured `INITIAL_BALANCE` book value, which mis-seeded the prod guard at $100 vs
  true equity ~$84 (2026-07-04) and produced a phantom 15.6% drawdown warning. A failed DB
  read defers seeding to the next loop cycle (bounded by `MAX_SEED_ATTEMPTS`, then falls
  back to the current balance with a WARNING). A restart therefore never resets — or
  inflates — the drawdown baseline.
  By policy the baseline is the peak **true equity since the last reconciled reset** —
  pre-reset ledger history is excluded because the Mar–Jun 2026 rows carry a phantom-era
  book peak (see the capital-erosion postmortem; measuring from it would falsely report an
  immediate breach on deploy). Known limitation: a clean restart that creates a NEW session
  re-baselines the peak ("20% per session", not rolling); dormant while prod reuses the
  active session across restarts — a durable cross-session peak is tracked in #847.
- **Escalation tiers** (risk-limits.json `escalation`): WARNING log at 50% of the cap
  (10% drawdown), CRITICAL log at 80% (16% drawdown). Tier logs are rate-limited
  (`DRAWDOWN_GUARD_LOG_INTERVAL_SECONDS`) but escalations log immediately.
- **Breach (drawdown >= cap)**: the engine enters the existing **close-only mode**. The flag
  gates every exposure increase: entry evaluation, the `execute_entry_locked` chokepoint
  (covers the legacy short path and any direct caller), and scale-ins. Exits, partial exits,
  stop-loss management, and reconciliation keep running. Nothing is liquidated.
  A CRITICAL `system_events` row (`error_code=MAX_DRAWDOWN_BREACH`), a structured
  `risk_event`, and the alert webhook (if configured) fire once. The trip is **latched**: it
  survives balance recovery below the cap, does not re-spam, and re-trips on restart via the
  boot-time peak recompute.
- **Clearing a trip (operator only)**: `resume_trading()` alone will not stick — the guard
  re-trips on the next iteration while the breach persists. To accept the loss and resume,
  restart with `FEATURE_MAX_DRAWDOWN_RESET_PEAK=true`, which re-baselines the peak to the
  current balance (the guard stays armed from the new baseline). **Remove the flag after the
  restart** — leaving it set re-baselines the peak on every future restart, weakening the cap.

## Position management features

- Dynamic risk adjustment (`DynamicRiskManager`) tapers exposure after drawdowns and relaxes limits during recoveries. Configure
  thresholds via `DynamicRiskConfig` and inspect changes through the monitoring dashboards.
- Correlation controls (`CorrelationEngine`) review active exposure across symbols before approving new trades. Set
  `max_correlated_exposure` to cap aggregate risk when assets move together.
- Partial exits and scale-ins (`PartialExitPolicy`) automate laddered profit-taking and controlled averaging strategies with
  explicit percentage targets and sizes.
- Time-based exits (`TimeExitPolicy`) enforce maximum holding periods, end-of-day flattening, or weekend shutdowns for markets
  with gaps.
- Trailing stops (`TrailingStopPolicy`) and breakeven rules lock in gains once price moves in favour of the position.
- MFE/MAE tooling (`MfeMaeAnalyzer`) feeds analytics back into strategy tuning so component strategies can adjust thresholds over
  time.

## Performance Tracking

The live trading engine uses the unified `PerformanceTracker` from `src/performance/tracker.py` to calculate real-time performance metrics. All metrics use the same calculation logic as the backtest engine, ensuring consistent validation.

### Available Metrics

The live engine tracks 30+ comprehensive metrics in real-time:

| Category | Metrics | Description |
| -------- | ------- | ----------- |
| **Returns** | `total_return_pct`, `annualized_return` | Overall profitability |
| **Risk-Adjusted** | `sharpe_ratio`, `sortino_ratio`, `calmar_ratio` | Returns adjusted for volatility and drawdown risk |
| **Risk** | `max_drawdown`, `current_drawdown`, `var_95` | Real-time risk exposure |
| **Trade Quality** | `win_rate`, `profit_factor`, `expectancy` | Trade effectiveness |
| **Efficiency** | `avg_trade_duration_hours`, `consecutive_wins`, `consecutive_losses` | Streak tracking and frequency |
| **Costs** | `total_fees_paid`, `total_slippage_cost` | Transaction cost tracking |

### Accessing Performance Metrics

Retrieve current performance via the `get_performance_summary()` method:

```python
from src.engines.live.trading_engine import LiveTradingEngine

engine = LiveTradingEngine(...)
summary = engine.get_performance_summary()

# Access risk-adjusted metrics
print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {summary['sortino_ratio']:.2f}")
print(f"Calmar Ratio: {summary['calmar_ratio']:.2f}")
print(f"VaR (95%): {summary['var_95']:.4f}")

# Check trade quality
print(f"Expectancy: {summary['expectancy']:.2f}")
print(f"Win Rate: {summary['win_rate'] * 100:.1f}%")
print(f"Consecutive Wins: {summary['consecutive_wins']}")
```

### Database Persistence

All performance metrics are persisted to PostgreSQL tables:
- **account_history** - Balance snapshots with Sharpe, Sortino, Calmar, VaR
- **performance_metrics** - Aggregated metrics including consecutive streaks, fees, slippage

The database schema supports historical analysis and comparison with backtest results.

### Trade fee accounting (`trades.commission` unit convention)

Each closed `trades` row stores fees so consumers can compute true net P&L:

- **`trades.pnl`** — GROSS dollar P&L (price movement only), for parity with the
  backtest engine. Fees are **not** netted into `pnl`.
- **`trades.commission`** — total round-trip fee in **USD**, equal to
  `entry_fee + exit_fee`. These are the **same values booked to
  `account_balances`**: the entry leg is the `entry_fee_<symbol>` ledger event
  (deducted at open), and the exit leg is folded into the `realized_pnl_<symbol>`
  balance update at close. The entry leg is reconciled to the actual exchange fill
  commission where available (`LiveExecutionEngine.execute_entry/_exit`). For a
  position **recovered after a restart** (the `positions` table does not persist entry
  fee), the entry leg is reconstructed from the fee model applied to the recovered
  entry notional, so it is not silently dropped.
- **`trades.margin_interest_cost`** — borrow interest in USD (short margin
  positions), from `MarginInterestTracker`.
- **`trades.quantity`** — actual filled base-asset quantity for the closed portion.

**Net P&L = `pnl - commission - margin_interest_cost`.**

> `trades.commission` is deliberately **not** the raw `orders.actual_commission`.
> That column stores the exchange commission in the *received asset* (base on buys,
> quote on sells) with no `commission_asset` column to disambiguate, and it is
> populated asynchronously by reconciliation — so it is both unit-ambiguous and
> unreliable at close time. Booking `commission` from the engine's USD fee
> accounting keeps `trades` consistent with the `account_balances` ledger.

> **Known limitation — partial exits.** Partial exits book their realized P&L to the
> `account_balances` ledger but do **not** write a `trades` row (only the final close
> does, recording the remaining slice). So `recover_last_balance` — the degraded
> fallback that reconstructs balance as `initial_balance + Σ _trade_net_pnl` when the
> ledger is unavailable — reconciles *exactly* for full round trips but is **approximate**
> for positions that took partial exits (their intermediate P&L and fees are in the ledger,
> not in `trades`). Logging partial-exit trade rows is tracked as a follow-up.

> **Reconciler-closed trades.** When a position is closed during reconciliation rather than by
> the engine, a `trades` row is still written so fees/quantity/P&L are not lost. Two kinds:
> - **Closes the bot can price** — offline stop-loss (`_close_position_from_filled_sl`) and
>   crash-recovery `FULL_EXIT` (`_reconcile_filled_exit`) — know the exit fill, so they realize
>   P&L **and** log the trade (only after the DB position is confirmed closed; deduped by the
>   exit order id or a synthetic `reconcile_exit_<position_id>` key).
> - **External/manual closes** (operator sells on the exchange UI, or a liquidation) detected by
>   a holdings/borrow check (`_verify_asset_holdings` spot, `_remove_phantom_position` margin)
>   carry no fill. They log a **balance-neutral** trade row only (commission = reconstructed
>   entry leg, GROSS pnl priced mark-to-market from the data provider, falling back to entry
>   price → pnl 0), keyed by a synthetic `reconcile_ext_<position_id>`. They do **not** touch the
>   balance: that capital is reconciled by startup Step C (`_reconcile_balance`, spot) or
>   `AccountSynchronizer._sync_margin_equity` (margin), so realizing P&L here too would
>   double-book the ledger. Because that GROSS pnl is a mark-to-market **estimate** (not the real
>   external fill), it is an approximation wherever `trades.pnl` is summed — session metrics and
>   the degraded `recover_last_balance` fallback — in the same spirit as the partial-exit caveat
>   above; the authoritative `account_balances` ledger is unaffected. (`PeriodicReconciler`'s
>   runtime external-close detection does not yet log a trade row — tracked as a follow-up.)

## CLI usage

`atb live` forwards arguments to `src/engines/live/runner.py`:

```bash
# Paper trading session (Binance, 60 second polling)
atb live ml_basic --symbol BTCUSDT --timeframe 1h --paper-trading --check-interval 60

# Live trading with explicit acknowledgement (be careful!)
atb live ml_basic --symbol BTCUSDT --live-trading --i-understand-the-risks --provider binance
```

Useful flags:

- `--balance`, `--max-position` – tune initial balance and maximum position size fraction.
- `--risk-per-trade`, `--max-risk-per-trade`, `--max-drawdown` – inject custom `RiskParameters` values.
- `--no-cache` – disable `CachedDataProvider` wrapping when live candles must always be fresh.
- `--mock-data` – run the engine loop without touching the exchange (useful in CI).

The control surface lives under `atb live-control`:

- `atb live-control train --symbol BTCUSDT --days 365` – runs the standard `atb train` pipeline from the live console and updates the
  registry’s `latest` symlink automatically so the live engine picks up the new model.
- `atb live-control deploy-model --model-path <staging-dir> --close-positions` – promote a staged bundle into the live strategy
  directory.
- `atb live-control list-models` / `status` / `emergency-stop` – quick operational actions when supervising a running engine.

## Programmatic usage

```python
import os

from src.engines.live.trading_engine import LiveTradingEngine
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider
from src.strategies.ml_basic import create_ml_basic_strategy

engine = LiveTradingEngine(
    strategy=create_ml_basic_strategy(),
    data_provider=CachedDataProvider(BinanceProvider(), cache_ttl_hours=1),
    check_interval=60,
    max_position_size=0.1,
    enable_live_trading=False,  # keep paper trading unless explicitly enabled
    database_url=os.environ["DATABASE_URL"],  # LiveTradingEngine requires PostgreSQL
)
# engine.start("BTCUSDT", "1h")  # blocking loop (prefer running via `atb live` / `atb live-health`)
```

In production deployments wrap the engine in a supervisor (systemd, Docker, Kubernetes) so that `SIGTERM` triggers the graceful
shutdown path implemented in the runner.
