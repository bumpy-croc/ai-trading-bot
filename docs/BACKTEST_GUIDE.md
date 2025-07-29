# Backtest Guide for AI Trading Bot

## Overview

The AI trading bot includes a comprehensive backtesting system that allows you to test trading strategies against historical data. This guide covers how to run backtests effectively and interpret the results.

## Quick Start

### Basic Backtest Command

```bash
python scripts/run_backtest.py <strategy_name> --days <number_of_days> --timeframe <timeframe> --initial-balance <amount>
```

### Example: 5-Year Backtest with MLBasic Strategy

```bash
python scripts/run_backtest.py ml_basic --days 1825 --timeframe 1h --initial-balance 10000 --no-cache
```

## Available Strategies

- `ml_basic` - Machine Learning Basic Strategy (recommended for 1h timeframe)
- `ml_with_sentiment` - ML Strategy with sentiment analysis

## Timeframe Recommendations

### For ML Strategies (ml_basic, ml_with_sentiment)
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
| `--days` | Number of days to backtest | 30 | `--days 1825` |
| `--timeframe` | Candle timeframe | `1h` | `--timeframe 1h` |
| `--initial-balance` | Starting balance | 10000 | `--initial-balance 10000` |
| `--no-cache` | Disable data caching | False | `--no-cache` |
| `--use-sentiment` | Enable sentiment analysis | False | `--use-sentiment` |
| `--no-db` | Disable database logging | False | `--no-db` |

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

- Use `--no-cache` to fetch fresh data from Binance
- Without `--no-cache`, the system uses cached data (faster but may be outdated)
- Data is automatically cached for 24 hours by default

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
5. **Compare multiple strategies with the same parameters**

## Example Commands

### Quick 1-Year Test
```bash
python scripts/run_backtest.py ml_basic --days 365 --timeframe 1h --initial-balance 10000
```

### Full 5-Year Test with Fresh Data
```bash
python scripts/run_backtest.py ml_basic --days 1825 --timeframe 1h --initial-balance 10000 --no-cache
```

### Test with Sentiment Analysis
```bash
python scripts/run_backtest.py ml_with_sentiment --days 365 --timeframe 1h --initial-balance 10000 --use-sentiment
```

### Compare Strategies
```bash
# Test multiple strategies
python scripts/run_backtest.py ml_basic --days 365 --timeframe 1h --initial-balance 10000
python scripts/run_backtest.py ml_with_sentiment --days 365 --timeframe 1h --initial-balance 10000
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