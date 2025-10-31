# Backtesting

> **Last Updated**: 2025-10-31  
> **Related Documentation**: [Live trading](live_trading.md), [Data pipeline](data_pipeline.md)

The vectorised backtesting engine in `src/backtesting/engine.py` replays historical candles, applies the strategy lifecycle, and
records trades, risk metrics, and optional database logs. It mirrors the live engine behaviour: partial exits, trailing stops,
regime-aware strategy switching, and dynamic risk controls all share the same helpers.

## Key components

- `Backtester` orchestrates the run, handles warm-up periods, and computes summary metrics using
  `src/backtesting/utils.compute_performance_metrics`.
- Strategies use component-based architecture with `Strategy` class that composes `SignalGenerator`, `RiskManager`, 
  and `PositionSizer` components. The engine calls `strategy.process_candle()` for each candle to get trading decisions.
- Risk controls rely on `RiskManager`, `DynamicRiskManager`, `TrailingStopPolicy`, and optional partial exit policies – the same
  classes used by the live engine.
- When `log_to_database=True`, the engine persists trades, strategy executions, and session records through
  `DatabaseManager` so dashboards can visualise results next to live data.

## CLI usage

The `atb backtest` command (`cli/commands/backtest.py`) is the fastest way to run a simulation:

```bash
# 90 day simulation with cached Binance data
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 90 --initial-balance 10000
```

Important flags:

| Flag | Purpose |
| ---- | ------- |
| `--provider {binance,coinbase}` | Switches the market data source. |
| `--no-cache` / `--cache-ttl` | Control the `CachedDataProvider` wrapper. |
| `--use-sentiment` | Adds `FearGreedProvider` data to the candle frame. |
| `--risk-per-trade` / `--max-risk-per-trade` | Override `RiskParameters` for the run. |
| `--log-to-db` | Persist the session to PostgreSQL for later inspection. |
| `--start` / `--end` | Explicit date boundaries (override `--days`). |

Strategies available via the CLI loader today: `ml_basic`, `ml_sentiment`, `ml_adaptive`, `ensemble_weighted`, and
`momentum_leverage`. Add new strategies under `src/strategies` and register them in `_load_strategy` to expose them through the
command.

## Built-in strategies

- `ml_basic`, `ml_sentiment`, and `ml_adaptive` are tuned for `1h` candles – daily candles underperform because ML features lose
  resolution.
- `ensemble_weighted` mixes the ML strategies and shares the same timeframe expectations.
- `momentum_leverage` targets high-volatility regimes; prefer shorter lookbacks (≤ 180 days) when comparing against ML baselines.

## Safety limits

Backtests stop early when max drawdown exceeds 50% to surface unbounded risk profiles. The run prints the stop reason, time, and
candle index so you can inspect the raw data or adjust risk parameters (`--risk-per-trade`, `--max-drawdown`) before re-running.

## Best practices

- Prefill data with `atb data prefill-cache` before running long simulations to avoid partial years being fetched mid-run.
- Benchmark new strategy ideas on 90–365 day windows first, then extend to multi-year ranges once the signal is stable.
- Use `--no-cache` when validating fixes against freshly downloaded data, and re-enable caching for regular workflows.
- Capture DB logs (`--log-to-db`) when comparing against live trading so dashboards show apples-to-apples metrics.

## Regime detection

- Regime-aware behaviour is handled inside each strategy configuration. When a strategy supports multiple regimes it will
  manage component switching automatically without additional CLI flags.
- Regime detectors live under `src/regime/` and expose reusable analyzers such as `VolatilityRegimeDetector` and
  `TrendRegimeDetector`. Combine them with the prediction engine or ML features to feed richer context into switching logic.
- Use the CLI under `atb regime ...` to profile historical regime labels, export summaries, or validate detector thresholds before
  enabling live.

## Programmatic execution

Backtests can also run from Python modules or notebooks:

```python
from datetime import datetime, timedelta

from src.backtesting.engine import Backtester
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider
from src.strategies.ml_basic import create_ml_basic_strategy

strategy = create_ml_basic_strategy()
provider = CachedDataProvider(BinanceProvider(), cache_ttl_hours=24)
backtester = Backtester(strategy=strategy, data_provider=provider, initial_balance=10_000)
start = datetime.utcnow() - timedelta(days=120)
results = backtester.run(symbol="BTCUSDT", timeframe="1h", start=start)
print(results["total_return"], results["max_drawdown"])
```

The returned dictionary includes cumulative metrics, yearly breakdowns, and a trade list. When sentiment or ML prediction columns
are present they are captured in the trade audit entries to support detailed analysis.

## Optimisation loop

`atb optimizer` (`cli/commands/optimizer.py`) runs a baseline backtest, feeds the results into the
`src/optimizer/analyzer.PerformanceAnalyzer`, and optionally evaluates a candidate configuration suggested by the analyzer. When
`--persist` is supplied the cycle records an `OptimizationCycle` row via `DatabaseManager`, creating an auditable trail of proposed
risk or strategy adjustments.
