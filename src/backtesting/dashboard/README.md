# Backtesting Dashboard

Simple Flask dashboard for visualizing backtest results.

## Run
```bash
# Preferred: use the dashboards CLI to discover and launch the UI
atb dashboards run backtesting --port 8050

# Direct module execution (if you need custom flags)
python -m src.dashboards.backtesting.dashboard --host 0.0.0.0 --port 8050
```
