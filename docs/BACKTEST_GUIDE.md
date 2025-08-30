# Backtest Guide for AI Trading Bot

## Overview

The AI trading bot includes a comprehensive backtesting system that allows you to test trading strategies against historical data. This guide covers how to run backtests effectively and interpret the results.

## Quick Start

### Basic Backtest Command

```bash
atb backtest <strategy_name> --symbol <symbol> --timeframe <timeframe> --days <number_of_days> --initial-balance <amount>
```

### Example: 5-Year Backtest with MLBasic Strategy

```bash
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 1825 --initial-balance 10000
```

## Available Strategies

- `ml_basic` - Machine Learning Basic Strategy (recommended for 1h timeframe)

## Timeframe Recommendations

### For ML Strategies (ml_basic)
**Always use 1-hour candles (`--timeframe 1h`)**

- ML strategies perform significantly better with 1-hour data
- Daily candles often result in poor performance
- 1-hour timeframe provides better signal resolution for ML predictions

### For Traditional Strategies
- `1h` - Good for short-term strategies
- `1d` - Good for long-term trend following
- `4h` - Balanced approach

## Command Line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `strategy` | Strategy name | Required | `ml_basic` |
| `--symbol` | Trading symbol | Required | `--symbol BTCUSDT` |
| `--days` | Number of days to backtest | 30 | `--days 1825` |
| `--timeframe` | Candle timeframe | `1h` | `--timeframe 1h` |
| `--initial-balance` | Starting balance | 10000 | `--initial-balance 10000` |

## Important Notes

### Early Stop Mechanism

The backtest includes a safety mechanism that stops the test early if:
- **Maximum drawdown exceeds 50%** - This prevents unrealistic results

When a backtest stops early, you'll see a warning message:
```
⚠️  BACKTEST STOPPED EARLY ⚠️
Reason: Maximum drawdown exceeded (50.1%)
Date: 2021-05-15 14:00:00
Candle: 8760 of 43780
```

### Data Fetching

- Data is automatically cached by the system for faster subsequent runs
- Cache can be managed using: `atb data cache-manager info|list|clear-old`
- Prefill cache for faster backtests: `atb data prefill-cache --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --years 4`

### Performance Expectations

#### MLBasic Strategy (1h timeframe)
- **Expected trades**: 200-300 over 5 years
- **Win rate**: 40-50%
- **Annualized return**: 10-20% (good years)
- **Max drawdown**: 30-50%

#### Daily vs Hourly Performance
- **1h timeframe**: Much better performance, more trades
- **1d timeframe**: Often negative returns, fewer trades

## Best Practices

1. **Always use 1h timeframe for ML strategies**
2. **Use `--no-cache` for fresh data**
3. **Start with 1-year tests before running 5-year tests**
4. **Monitor the early stop warnings**

## Example Commands

### Quick 1-Year Test
```bash
python scripts/run_backtest.py ml_basic --days 365 --timeframe 1h --initial-balance 10000
```

### Full 5-Year Test with Fresh Data
```bash
python scripts/run_backtest.py ml_basic --days 1825 --timeframe 1h --initial-balance 10000 --no-cache
```

## Troubleshooting

### Backtest Stops Early
- This is normal and expected for risky strategies
- The 50% drawdown limit prevents unrealistic results
- Consider using a different strategy or adjusting risk parameters

### No Data Available
- Check your internet connection
- Ensure Binance API is accessible
- Try using cached data (remove `--no-cache`)

### Poor Performance
- Ensure you're using 1h timeframe for ML strategies
- Check that you have sufficient historical data
- Consider testing with a shorter period first

## Understanding Results

### Key Metrics
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Overall percentage gain/loss
- **Annualized Return**: Yearly average return
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure

### Yearly Returns
Shows performance for each calendar year in the test period.

### Early Stop Information
If present, shows why the backtest stopped early and when.
