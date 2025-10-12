# Strategy Migration Baseline Benchmarking

_Phase 0 deliverables for the strategy migration proposal._

## Overview

Phase 0 captures a reproducible snapshot of the legacy execution stack
before introducing the new strategy runtime. The `benchmark_legacy_baseline`
script exercises both the vectorised backtester and the live trading engine
(paper mode) against deterministic synthetic market data supplied by
`MockDataProvider`. The artefacts produced here serve as regression
targets for later phases of the migration.

All outputs are written to `artifacts/strategy-migration/baseline/` and
are committed to version control so they can be diffed in future phases.

## Running the Benchmarks

```bash
# Backtest and live (paper) baselines for ML Basic and ML Adaptive
python scripts/benchmark_legacy_baseline.py

# Backtest only (skip live) for ML Basic
python scripts/benchmark_legacy_baseline.py --strategies ml_basic --skip-live

# Custom timeframe / dataset length
python scripts/benchmark_legacy_baseline.py \
  --strategies ml_basic ml_adaptive \
  --timeframe 4h \
  --backtest-days 60 \
  --live-steps 30
```

The script uses in-memory SQLite for the live engine so no external
database is required. The committed baseline was generated with::

    python scripts/benchmark_legacy_baseline.py --strategies ml_basic --backtest-days 30 --live-steps 20

The resulting live-paper snapshot did not open any positions during the
20-step observation window; this provides a clean control sample focused
on runtime throughput and indicator preparation costs.

Artefacts include:

- `baseline_backtest_<strategy>.json` – core performance metrics and
  runtime timings from the backtester.
- `baseline_backtest_<strategy>_trades.csv` – raw trade logs from the
  backtest run.
- `baseline_live_<strategy>.json` – live engine performance summary,
  runtime timings, and captured trades (paper trading mode).
- `baseline_live_<strategy>_trades.csv` – live engine trade logs.
- `baseline_summary.json` and `baseline_summary.md` – aggregated summary
  for quick comparison between scenarios.

## Dataset Notes

- **Data source**: `MockDataProvider` generates deterministic OHLCV
  sequences (random walk seeded per scenario) to ensure repeatable runs
  without external API calls.
- **Timeframes**: Default `1h` candles over 30 days for backtests and
  `1h` candles for 20 live loop iterations in the committed baseline.
- **Initial balance**: $10,000 with conservative risk parameters to
  mirror existing defaults in the legacy engines.

## Using the Artefacts

1. **Regression comparisons** – future runtime implementations can be
   benchmarked against the JSON summaries to confirm parity on total
   trades, PnL, drawdown, and runtime throughput.
2. **Trade sequence auditing** – CSV trade logs capture entry/exit
   timing, allowing diff-based regression testing.
3. **Performance monitoring** – aggregated Markdown summary offers a
   quick dashboard for spotting regressions during development.

These artefacts should be regenerated and compared after significant
changes to the strategy runtime or engine integration layers to validate
behavioural parity. The aggregated results are summarised in
`artifacts/strategy-migration/baseline/baseline_summary.md`.
