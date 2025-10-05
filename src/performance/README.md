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
from src.performance.metrics import sharpe, max_drawdown

# Calculate metrics from daily balance series
sharpe_ratio = sharpe(daily_balance_series)
max_dd = max_drawdown(daily_balance_series)

print(f"Sharpe: {sharpe_ratio:.2f}, Max DD: {max_dd:.2%}")
```

### Trade-Based Metrics
```python
import pandas as pd
import numpy as np

# Calculate win rate from trades
trades_df = pd.DataFrame([
    {'pnl': 100, 'exit_time': '2024-01-01'},
    {'pnl': -50, 'exit_time': '2024-01-02'},
    {'pnl': 150, 'exit_time': '2024-01-03'}
])

winning_trades = len(trades_df[trades_df['pnl'] > 0])
total_trades = len(trades_df)
win_rate = winning_trades / total_trades if total_trades > 0 else 0

print(f"Win Rate: {win_rate:.1%}")
print(f"Total Trades: {total_trades}")
print(f"Average PnL: ${trades_df['pnl'].mean():.2f}")
```

### Rolling Metrics
```python
from src.performance.metrics import sharpe
import pandas as pd

# Calculate rolling Sharpe ratio
window = 30
rolling_returns = daily_balance_series.pct_change()
rolling_sharpe = rolling_returns.rolling(window=window).apply(
    lambda x: sharpe(pd.Series(x)) if len(x) == window else np.nan
)

print(f"Current Rolling Sharpe (30d): {rolling_sharpe.iloc[-1]:.2f}")
```

## Performance Analysis

```python
import matplotlib.pyplot as plt

# Plot balance curve
plt.figure(figsize=(12, 6))
plt.plot(daily_balance_series.index, daily_balance_series.values)
plt.title('Balance Over Time')
plt.xlabel('Date')
plt.ylabel('Balance ($)')
plt.grid(True)
plt.savefig('balance_curve.png')
plt.close()

# Calculate cumulative returns
cumulative_returns = (daily_balance_series / daily_balance_series.iloc[0] - 1) * 100
print(f"Total Return: {cumulative_returns.iloc[-1]:.2f}%")
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
