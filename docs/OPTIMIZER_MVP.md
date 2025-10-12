# Optimizer MVP

A minimal closed-loop component to run a backtest, analyze results, and emit bounded improvement suggestions.

## How it works
- Runs a single backtest (defaults: `ml_basic`, `BTCUSDT`, `1h`, last 30 days)
- Computes key KPIs from the backtester output
- Analyzer proposes small, safe parameter adjustments (risk bounds, SL/TP tweaks) if thresholds are not met
- Writes a JSON report with results and suggestions

## Run
```bash
atb optimizer --strategy ml_basic --symbol BTCUSDT --timeframe 1h --days 60
```

Flags:
- `--provider {binance,coinbase}`: data source
- `--no-cache`: disable cached provider wrapper
- `--output path`: JSON report path (default `artifacts/optimizer_report.json`)

## Output
- JSON report with experiment settings, KPIs, and an array of suggestions (target, change, rationale, expected deltas, confidence)

## Next steps
- Add sweep support and Bayesian search (Optuna)
- Wire into CI schedule to run daily and attach artifacts
- Auto-PR for config changes guarded by validation jobs
