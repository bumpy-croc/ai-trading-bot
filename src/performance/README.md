# Performance Utilities

Comprehensive utilities for computing performance metrics used across backtesting, live trading, and monitoring.

## Overview

This module provides a complete suite of performance analysis tools for evaluating trading strategies. Metrics include risk-adjusted returns, drawdown analysis, win/loss statistics, and advanced portfolio analytics.

## Contents
- `metrics.py` - Core performance metrics and analysis functions

## Available Metrics

### Risk-Adjusted Returns
- **Sharpe Ratio** - Return per unit of volatility
- **Sortino Ratio** - Return per unit of downside volatility
- **Calmar Ratio** - Return per unit of maximum drawdown
- **Information Ratio** - Excess return per unit of tracking error

### Drawdown Analysis
- **Maximum Drawdown** - Largest peak-to-trough decline
- **Average Drawdown** - Mean of all drawdowns
- **Drawdown Duration** - Time spent in drawdown
- **Recovery Time** - Time to recover from drawdowns

### Win/Loss Statistics
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Gross profit / gross loss
- **Average Win** - Mean profit of winning trades
- **Average Loss** - Mean loss of losing trades
- **Expectancy** - Expected value per trade

### Portfolio Analytics
- **Total Return** - Cumulative return over period
- **Annualized Return** - CAGR (Compound Annual Growth Rate)
- **Volatility** - Standard deviation of returns
- **Beta** - Correlation with benchmark
- **Alpha** - Excess return over benchmark

## Usage

### Basic Metrics
```python
from src.performance.metrics import (
    perf_sharpe, 
    perf_max_drawdown,
    perf_sortino,
    perf_calmar
)

# Calculate metrics from daily balance series
sharpe = perf_sharpe(daily_balance_series)
max_dd = perf_max_drawdown(daily_balance_series)
sortino = perf_sortino(daily_balance_series)
calmar = perf_calmar(daily_balance_series)

print(f"Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2%}")
```

### Trade-Based Metrics
```python
from src.performance.metrics import (
    calculate_win_rate,
    calculate_profit_factor,
    calculate_expectancy
)

trades = [
    {'pnl': 100, 'exit_time': '2024-01-01'},
    {'pnl': -50, 'exit_time': '2024-01-02'},
    {'pnl': 150, 'exit_time': '2024-01-03'}
]

win_rate = calculate_win_rate(trades)
profit_factor = calculate_profit_factor(trades)
expectancy = calculate_expectancy(trades)

print(f"Win Rate: {win_rate:.1%}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Expectancy: ${expectancy:.2f}")
```

### Comprehensive Performance Report
```python
from src.performance.metrics import generate_performance_report

report = generate_performance_report(
    balance_series=daily_balances,
    trades=trade_history,
    benchmark_returns=sp500_returns  # Optional benchmark
)

print(report)
```

### Rolling Metrics
```python
from src.performance.metrics import calculate_rolling_sharpe

# Calculate 30-day rolling Sharpe ratio
rolling_sharpe = calculate_rolling_sharpe(
    daily_returns,
    window=30
)
```

## Performance Visualization

```python
from src.performance.metrics import plot_performance

# Generate comprehensive performance plots
plot_performance(
    balance_series=daily_balances,
    trades=trade_history,
    output_path='performance_report.png'
)
```

## Integration

### Backtesting
Performance metrics are automatically calculated and displayed in backtest results:
```bash
atb backtest ml_basic --symbol BTCUSDT --days 365
```

### Live Trading
Real-time performance metrics are available through the monitoring dashboard:
```bash
atb dashboards run monitoring --port 8000
```

### Database
Historical performance metrics are stored in the `performance_metrics` table for trend analysis.

## Best Practices

1. **Use annualized metrics** for comparing strategies across different time periods
2. **Consider risk-adjusted returns** (Sharpe, Sortino) over raw returns
3. **Monitor drawdown metrics** to ensure risk tolerance is maintained
4. **Track multiple timeframes** (daily, weekly, monthly) for complete picture
5. **Compare to benchmarks** (buy-and-hold, index) for context
