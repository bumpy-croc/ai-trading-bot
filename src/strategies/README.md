# Trading Strategies

This directory contains all available trading strategies for the AI Trading Bot. Each strategy implements the `BaseStrategy` interface and can be used for backtesting, paper trading, and live trading.

## Available Strategies

### Machine Learning Strategies

#### ML Basic (`ml_basic.py`)
A foundational ML strategy using price-only data for reliable predictions.

**Key Features:**
- LSTM neural network trained on OHLCV data
- 120-day sequence length for pattern recognition
- 2% stop loss, 4% take profit risk management
- No external API dependencies
- Position sizing: 5-25% of balance

**Best For:**
- Consistent, reliable trading signals
- Historical backtesting
- Simple deployment scenarios
- Environments without sentiment data

**Usage:**
```bash
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 90
atb live ml_basic --symbol BTCUSDT --paper-trading
```

#### ML Adaptive (`ml_adaptive.py`)
An advanced ML strategy with regime detection and adaptive parameters.

**Key Features:**
- Dynamic threshold adjustment based on market regimes
- Regime-aware position sizing
- Enhanced risk management during volatile periods
- Automatic adaptation to market conditions
- Position sizing: 5-25% of balance

**Best For:**
- Volatile market conditions
- Long-term trading strategies
- Adaptive risk management
- Regime-aware trading

**Usage:**
```bash
atb backtest ml_adaptive --symbol BTCUSDT --timeframe 1h --days 90
atb live ml_adaptive --symbol BTCUSDT --paper-trading
```

#### ML Sentiment (`ml_sentiment.py`)
ML strategy enhanced with sentiment analysis for improved prediction accuracy.

**Key Features:**
- Models trained with price data and Fear & Greed Index
- Adaptive position sizing based on sentiment confidence
- Enhanced accuracy during volatile market conditions
- Robust fallback when sentiment data unavailable
- Supports BTC and ETH models with sentiment integration

**Best For:**
- Volatile market conditions
- Sentiment-driven trading
- Enhanced prediction accuracy
- Market psychology analysis

**Usage:**
```bash
atb backtest ml_sentiment --symbol BTCUSDT --timeframe 1h --days 90
atb live ml_sentiment --symbol BTCUSDT --paper-trading
```

### Ensemble Strategies

#### Ensemble Weighted (`ensemble_weighted.py`)
An aggressive ensemble approach designed to beat buy-and-hold returns.

**Key Features:**
- Combines multiple strategies with performance-based weighting
- Leveraged position sizing (up to 80% per trade)
- Momentum-based entry timing
- Dynamic risk scaling based on market volatility
- Multi-timeframe confirmation
- Aggressive profit-taking and re-entry

**Risk Management:**
- Wider stops (3.5%) to avoid premature exits
- Higher profit targets (8%) to capture trends
- Trailing stops to protect profits
- Dynamic position sizing based on confidence

**Best For:**
- Beating buy-and-hold performance
- Aggressive growth strategies
- Multi-strategy diversification
- High-confidence trading

**Usage:**
```bash
atb backtest ensemble_weighted --symbol BTCUSDT --timeframe 1h --days 90
atb live ensemble_weighted --symbol BTCUSDT --paper-trading
```

#### Regime Adaptive (`regime_adaptive.py`)
A meta-strategy that automatically switches between strategies based on market regimes.

**Key Features:**
- Automatic strategy switching based on market conditions
- Regime detection using trend and volatility analysis
- Optimal strategy selection for each market phase
- Seamless transitions between strategies
- Enhanced performance across market cycles

**Component Strategies:**
- Momentum Leverage for trending markets
- ML Basic for stable conditions
- Ensemble Weighted for volatile periods
- Bear Strategy for downtrends

**Best For:**
- All market conditions
- Long-term trading strategies
- Automatic adaptation
- Optimal performance across cycles

**Usage:**
```bash
atb backtest regime_adaptive --symbol BTCUSDT --timeframe 1h --days 90
atb live regime_adaptive --symbol BTCUSDT --paper-trading
```

### Market-Specific Strategies

#### Bull Strategy (`bull.py`)
Optimized for bull market conditions with trend confirmation.

**Key Features:**
- Trend confirmation using MA(50) > MA(200)
- Momentum confirmation using MACD and RSI
- Volatility-aware sizing using ATR
- Simple exit rules via stop-loss/take-profit
- Position sizing: 5-25% of balance

**Best For:**
- Strong uptrends
- Bull market conditions
- Trend-following strategies
- Momentum trading

**Usage:**
```bash
atb backtest bull --symbol BTCUSDT --timeframe 1h --days 90
atb live bull --symbol BTCUSDT --paper-trading
```

#### Bear Strategy (`bear.py`)
Designed for bear market conditions with defensive positioning.

**Key Features:**
- Defensive positioning in downtrends
- Short-term trading opportunities
- Risk management focused on capital preservation
- Position sizing: 4-20% of balance
- 2% stop loss, 4% take profit

**Best For:**
- Bear market conditions
- Defensive trading
- Capital preservation
- Short-term opportunities

**Usage:**
```bash
atb backtest bear --symbol BTCUSDT --timeframe 1h --days 90
atb live bear --symbol BTCUSDT --paper-trading
```

#### Momentum Leverage (`momentum_leverage.py`)
Aggressive momentum strategy designed to beat buy-and-hold returns.

**Key Features:**
- Pseudo-leverage through concentrated position sizing (up to 95%)
- Pure momentum following with trend confirmation
- Volatility-based position scaling
- Extended profit targets to capture full moves
- Quick re-entry after exits

**Risk Management:**
- 10% stop loss (wide to capture full moves)
- 35% take profit (capture massive moves)
- Position sizing: 40-95% of balance

**Best For:**
- Beating buy-and-hold performance
- Aggressive growth strategies
- Momentum trading
- High-conviction positions

**Usage:**
```bash
atb backtest momentum_leverage --symbol BTCUSDT --timeframe 1h --days 90
atb live momentum_leverage --symbol BTCUSDT --paper-trading
```

## Strategy Selection Guide

### For Beginners
- **ML Basic**: Simple, reliable, no external dependencies
- **Bull Strategy**: Good for trending markets

### For Intermediate Users
- **ML Adaptive**: Adaptive to market conditions
- **ML Sentiment**: Enhanced with sentiment analysis

### For Advanced Users
- **Ensemble Weighted**: Aggressive growth strategy
- **Regime Adaptive**: Automatic strategy switching
- **Momentum Leverage**: High-risk, high-reward

### For Specific Market Conditions
- **Bull Markets**: Bull Strategy, ML Basic
- **Bear Markets**: Bear Strategy, ML Adaptive
- **Volatile Markets**: ML Sentiment, Ensemble Weighted
- **All Conditions**: Regime Adaptive

## Creating Custom Strategies

To create your own strategy:

1. **Subclass BaseStrategy**:
```python
from src.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, name="MyStrategy"):
        super().__init__(name)
        # Initialize your strategy parameters
```

2. **Implement Required Methods**:
- `calculate_indicators()`: Calculate technical indicators
- `check_entry_conditions()`: Determine when to enter positions
- `check_exit_conditions()`: Determine when to exit positions
- `calculate_position_size()`: Calculate position size
- `calculate_stop_loss()`: Calculate stop loss level
- `get_parameters()`: Return strategy parameters

3. **Example Implementation**:
```python
def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    # Add your technical indicators
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    return df

def check_entry_conditions(self, df: pd.DataFrame, i: int) -> bool:
    # Define your entry logic
    return (df.iloc[i]['close'] > df.iloc[i]['sma_20'] and 
            df.iloc[i]['rsi'] < 70)
```

## Performance Considerations

- **Backtesting**: Always backtest strategies before live trading
- **Risk Management**: Never risk more than you can afford to lose
- **Position Sizing**: Start with smaller position sizes
- **Market Conditions**: Choose strategies appropriate for current market conditions
- **Monitoring**: Continuously monitor strategy performance

## Support

For questions about strategies or custom implementations, refer to the base strategy documentation or create an issue in the project repository.