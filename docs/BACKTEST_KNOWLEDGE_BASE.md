# Backtest Knowledge Base

## Key Findings and Best Practices

### 1. Timeframe is Critical for ML Strategies

**CRITICAL: Always use 1-hour candles for ML strategies**

#### Performance Comparison (MLBasic Strategy)

| Timeframe | 5-Year Performance | Trades | Win Rate | Annualized Return | Max Drawdown |
|-----------|-------------------|--------|----------|-------------------|--------------|
| 1h        | +101.85%         | 269    | 43.12%   | +15.08%          | 51.69%       |
| 1d        | -20.03%          | 31     | 41.94%   | -4.37%           | 51.13%       |

**Key Insight:** ML strategies perform dramatically better with 1-hour data due to:
- Better signal resolution for ML predictions
- More frequent trading opportunities
- Reduced noise compared to daily data

### 2. Early Stop Mechanism

The backtest includes a safety mechanism that stops when:
- **Maximum drawdown exceeds 50%**

This prevents unrealistic results and protects against excessive losses.

**Warning Display:**
```
⚠️  BACKTEST STOPPED EARLY ⚠️
Reason: Maximum drawdown exceeded (51.1%)
Date: 2021-06-24 00:00:00
Candle: 337 of 1825
```

### 3. Data Management

#### Fresh Data vs Cached Data
- **Use `--no-cache`** for fresh data from Binance
- **Without `--no-cache`** uses cached data (faster but may be outdated)
- Data is automatically cached for 24 hours

#### Data Availability
- Binance provides 5+ years of historical data
- 1-hour data: ~43,800 candles for 5 years
- 1-day data: ~1,825 candles for 5 years

### 4. Strategy Performance Expectations

#### MLBasic Strategy (1h timeframe)
- **Expected trades**: 200-300 over 5 years
- **Win rate**: 40-50%
- **Annualized return**: 10-20% (good years)
- **Max drawdown**: 30-50%

#### Performance Patterns
- **2020**: Generally positive (bull market)
- **2021**: High volatility, often triggers early stop
- **2022-2025**: Varies by market conditions

### 5. Command Line Best Practices

#### Standard 5-Year Backtest
```bash
python scripts/run_backtest.py ml_basic --days 1825 --timeframe 1h --initial-balance 10000 --no-cache
```

#### Quick 1-Year Test
```bash
python scripts/run_backtest.py ml_basic --days 365 --timeframe 1h --initial-balance 10000
```

### 6. Available Strategies

| Strategy | Best Timeframe | Use Case |
|----------|----------------|----------|
| `ml_basic` | 1h | Primary ML strategy |

### 7. Troubleshooting

#### Backtest Stops Early
- **Normal behavior** for risky strategies
- 50% drawdown limit prevents unrealistic results
- Consider using different strategy or adjusting parameters

#### Poor Performance
- **Ensure 1h timeframe for ML strategies**
- Check data availability
- Test with shorter periods first

#### No Data Available
- Check internet connection
- Ensure Binance API access
- Try cached data (remove `--no-cache`)

### 8. Understanding Results

#### Key Metrics
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Overall percentage gain/loss
- **Annualized Return**: Yearly average return
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure

#### Yearly Returns
Shows performance for each calendar year in the test period.

#### Early Stop Information
If present, shows why the backtest stopped early and when.

### 9. Database Integration

#### Session Tracking
- Each backtest creates a unique session ID
- All trades and decisions are logged to database
- Enables detailed analysis and monitoring

#### Logging Options
- `--no-db`: Disable database logging
- Default: Database logging enabled

### 10. Risk Management

#### Built-in Protections
- 50% maximum drawdown limit
- Position sizing based on balance
- Stop-loss mechanisms

#### Manual Risk Control
- Adjust initial balance
- Use different strategies
- Test with shorter periods

## Summary

1. **Always use 1h timeframe for ML strategies** - This is the most critical factor
2. **Use `--no-cache` for fresh data** - Ensures up-to-date results
3. **Monitor early stop warnings** - Understand when and why backtests stop
4. **Start with 1-year tests** - Before running 5-year tests

The MLBasic strategy with 1-hour candles consistently outperforms daily candles and provides a solid foundation for ML-based trading strategies. 