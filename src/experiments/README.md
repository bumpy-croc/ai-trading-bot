# Experiments

Declarative experimentation framework for incrementally improving a trading
strategy. A YAML file describes a baseline plus one or more variants; the
runner backtests each, a reporter ranks them with statistical tests, and a
promotion step records the winner in the strategy lineage.

## Overview

The goal is a tight feedback loop: *propose a variant → backtest vs baseline →
compare → promote if it wins*. Strategies are tuned via hand-picked variants
rather than random search.

## Modules

- `schemas.py` — `ExperimentConfig`, `ParameterSet`, `ExperimentResult`.
- `runner.py` — `ExperimentRunner` loads a strategy, applies dotted-path
  parameter overrides, and drives `Backtester.run`.
- `suite.py` — `SuiteConfig`, `VariantSpec`, `SuiteResult`,
  `ExperimentSuiteRunner` — expands a suite into one baseline + N variant
  experiments.
- `suite_loader.py` — loads and validates YAML suite files.
- `reporter.py` — `ExperimentReporter` renders ranked tables with
  Δ-vs-baseline and statistical significance.
- `ledger.py` — append-only JSONL history of completed suites.
- `promotion.py` — `PromotionManager` writes `StrategyVersionRecord` and
  `ChangeRecord` for winning variants and emits a patch YAML.
- `walk_forward.py` — rolling IS/OOS walk-forward analysis (retained from the
  previous optimizer module).

## CLI Usage

```bash
# Run a suite
atb experiment run --config experiments/signal_thresholds.yaml

# Inspect past suites
atb experiment list
atb experiment show <suite_id>

# Promote a variant (records version + lineage, emits patch YAML)
atb experiment promote <suite_id> <variant_name>

# Walk-forward robustness (separate pipeline)
atb walk-forward --strategy ml_basic --train-days 180 --test-days 30 --folds 6
```

## Programmatic Usage

```python
from src.experiments.runner import ExperimentRunner
from src.experiments.schemas import ExperimentConfig, ParameterSet
from datetime import UTC, datetime, timedelta

end = datetime.now(UTC)
start = end - timedelta(days=30)

config = ExperimentConfig(
    strategy_name="ml_basic",
    symbol="BTCUSDT",
    timeframe="1h",
    start=start,
    end=end,
    initial_balance=1000,
    parameters=ParameterSet(
        name="tight_stops",
        values={"ml_basic.stop_loss_pct": 0.02},
    ),
    provider="fixture",
    use_cache=False,
)

runner = ExperimentRunner()
result = runner.run(config)
print(result.sharpe_ratio, result.total_return, result.total_trades)
```

## Documentation

See [docs/backtesting.md](../../docs/backtesting.md) for detailed information
on the experimentation framework and usage patterns.
