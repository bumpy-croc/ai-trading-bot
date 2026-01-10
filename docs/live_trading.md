# Live trading

> **Last Updated**: 2025-12-26
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
