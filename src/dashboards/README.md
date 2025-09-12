# Dashboards

Web-based dashboards for monitoring and analyzing trading system performance.

## Available Dashboards

- `monitoring/`: Real-time trading performance monitoring
- `backtesting/`: Backtest results visualization
- `market_prediction/`: ML model prediction analysis

## Usage

```bash
# List available dashboards
atb dashboards list

# Run monitoring dashboard
atb dashboards run monitoring --port 8000

# Run backtesting dashboard  
atb dashboards run backtesting --port 8001
```

## Technology

- Flask + Socket.IO for real-time updates
- Bootstrap for responsive UI
- Chart.js for data visualization