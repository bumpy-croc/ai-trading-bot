---
description: Trading Bot Strategy Development & Implementation Guide
globs: 
alwaysApply: false
---

# 📊 Trading Bot Strategy Development

## Strategy System Overview

All strategies inherit from `BaseStrategy` and implement a standardized interface for signal generation, position sizing, and risk management.

---

## Strategy Base Class

### Required Interface
```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.trading_pair = 'BTCUSDT'  # Default, can be overridden
        
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific indicators on the data"""
        pass
        
    @abstractmethod
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if entry conditions are met at the given index"""
        pass
        
    @abstractmethod
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check if exit conditions are met at the given index"""
        pass
        
    @abstractmethod
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate the position size for a new trade"""
        pass
        
    @abstractmethod
    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = 'long') -> float:
        """Calculate stop loss level for a position"""
        pass
        
    @abstractmethod
    def get_parameters(self) -> dict:
        """Return strategy parameters for logging"""
        pass
```

### Strategy Execution Logging
```python
def log_execution(
    self,
    signal_type: str,           # 'entry', 'exit', 'hold'
    action_taken: str,          # 'opened_long', 'closed_position', 'no_action'
    price: float,
    signal_strength: Optional[float] = None,
    confidence_score: Optional[float] = None,
    indicators: Optional[Dict] = None,
    sentiment_data: Optional[Dict] = None,
    ml_predictions: Optional[Dict] = None,
    position_size: Optional[float] = None,
    reasons: Optional[List[str]] = None
):
    """Log strategy execution details to database"""
    pass
```

---

## Available Strategies

### 1. Adaptive Strategy (`adaptive.py`)
**Purpose**: Adaptive EMA crossover with market regime detection

**Key Features**:
- Adaptive Parameters: Adjusts based on market volatility
- Market Regime Detection: Different logic for trending vs ranging markets
- Dynamic Stop Losses: ATR-based stop loss calculation

**Logic**:
```python
def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
    # Adaptive EMA crossover
    ema_short = df.iloc[index]['ema_9']
    ema_long = df.iloc[index]['ema_21']
    
    # Market regime detection
    volatility = df.iloc[index]['atr'] / df.iloc[index]['close']
    is_trending = volatility < 0.02  # Low volatility = trending
    
    if is_trending:
        # More aggressive in trending markets
        return ema_short > ema_long and df.iloc[index]['rsi'] < 70
    else:
        # More conservative in ranging markets
        return ema_short > ema_long and df.iloc[index]['rsi'] < 60
```

**Best For**: Trend-following in various market conditions

---

### 2. Enhanced Strategy (`enhanced.py`)
**Purpose**: Multi-indicator confluence for high confirmation

**Key Features**:
- Multi-Indicator Confirmation: RSI + EMA + MACD
- Conservative Approach: Requires multiple confirmations
- Volume Confirmation: Uses volume for signal validation

**Logic**:
```python
def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
    # Multiple indicator confluence
    rsi = df.iloc[index]['rsi']
    ema_trend = df.iloc[index]['ema_9'] > df.iloc[index]['ema_21']
    macd_bullish = df.iloc[index]['macd'] > df.iloc[index]['macd_signal']
    volume_high = df.iloc[index]['volume'] > df.iloc[index]['volume'].rolling(20).mean()
    
    return (rsi < 70 and ema_trend and macd_bullish and volume_high)
```

**Best For**: Conservative trading with high confirmation

---

### 3. ML Basic Strategy (`ml_basic.py`)
**Purpose**: Uses ML price predictions for entry/exit decisions

**Key Features**:
- ML Model Integration: Real-time ONNX inference
- Confidence-Based Sizing: Position size based on prediction confidence
- Price Prediction: Uses trained neural network for price forecasting

**Logic**:
```python
def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
    # Get ML prediction
    prediction = df.iloc[index]['ml_prediction']
    confidence = df.iloc[index]['prediction_confidence']
    current_price = df.iloc[index]['close']
    
    # Entry conditions
    price_increase = (prediction - current_price) / current_price
    high_confidence = confidence > 0.7
    
    return price_increase > 0.01 and high_confidence  # 1% predicted increase
```

**Best For**: Data-driven trading decisions

---

### 4. ML with Sentiment Strategy (`ml_with_sentiment.py`)
**Purpose**: Combines ML predictions with sentiment analysis

**Key Features**:
- Multi-Modal Predictions: Price + sentiment data (13 features)
- Graceful Fallback: Works without sentiment data
- Sentiment Confidence: Weights predictions based on sentiment quality

**Logic**:
```python
def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
    # Get ML prediction with sentiment
    prediction = df.iloc[index]['ml_prediction']
    confidence = df.iloc[index]['prediction_confidence']
    sentiment_freshness = df.iloc[index]['sentiment_freshness']
    
    # Sentiment-enhanced conditions
    price_increase = (prediction - current_price) / current_price
    sentiment_positive = df.iloc[index]['sentiment_score'] > 0.1
    fresh_sentiment = sentiment_freshness > 0.8  # 80% fresh data
    
    return (price_increase > 0.01 and confidence > 0.7 and 
            (sentiment_positive or not fresh_sentiment))
```

**Best For**: Maximum prediction accuracy with sentiment context

---

### 5. High Risk High Reward Strategy (`high_risk_high_reward.py`)
**Purpose**: Aggressive trading with higher risk tolerance

**Key Features**:
- Larger Position Sizes: Up to 25% of balance per position
- Tighter Stops: More aggressive stop loss levels
- Higher Risk Tolerance: Designed for experienced traders

**Logic**:
```python
def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
    # Aggressive entry conditions
    rsi = df.iloc[index]['rsi']
    ema_trend = df.iloc[index]['ema_9'] > df.iloc[index]['ema_21']
    volume_spike = df.iloc[index]['volume'] > df.iloc[index]['volume'].rolling(10).mean() * 1.5
    
    return rsi < 75 and ema_trend and volume_spike
```

**Best For**: Experienced traders seeking higher returns

---

## Strategy Development Workflow

### 1. Create New Strategy
```python
# Create new file: src/strategies/my_strategy.py
from src.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, name: str):
        super().__init__(name)
        # Strategy-specific initialization
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate your indicators
        df['my_indicator'] = df['close'].rolling(20).mean()
        return df
        
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        # Your entry logic
        return df.iloc[index]['my_indicator'] > df.iloc[index]['close']
        
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        # Your exit logic
        current_price = df.iloc[index]['close']
        return current_price < entry_price * 0.95  # 5% stop loss
        
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        # Position sizing logic
        return balance * 0.02  # 2% risk per trade
        
    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = 'long') -> float:
        # Stop loss calculation
        return price * 0.95  # 5% stop loss
        
    def get_parameters(self) -> dict:
        return {"strategy_type": "custom", "version": "1.0"}
```

### 2. Add to Strategy Registry
```python
# Add to src/strategies/__init__.py
from .my_strategy import MyStrategy

STRATEGIES = {
    'my_strategy': MyStrategy,
    # ... other strategies
}
```

### 3. Test Strategy
```bash
# Quick backtest
python scripts/run_backtest.py my_strategy --days 30 --no-db

# Paper trading
python scripts/run_live_trading.py my_strategy --paper-trading
```

---

## Strategy Testing Best Practices

### 1. Backtesting Requirements
- Test on at least 6 months of data
- Use multiple market conditions (bull, bear, sideways)
- Validate with out-of-sample data
- Check for overfitting

### 2. Risk Management
- Always implement stop losses
- Limit position size to 2-5% of balance
- Monitor drawdown (max 20%)
- Test with realistic slippage and fees

### 3. Performance Metrics
- Sharpe ratio > 1.0
- Maximum drawdown < 20%
- Win rate > 50%
- Profit factor > 1.5

---

## Common Patterns

### Trend Following
```python
def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
    # Simple trend following
    ema_short = df.iloc[index]['ema_9']
    ema_long = df.iloc[index]['ema_21']
    return ema_short > ema_long
```

### Mean Reversion
```python
def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
    # Mean reversion
    rsi = df.iloc[index]['rsi']
    return rsi < 30  # Oversold condition
```

### Breakout Trading
```python
def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
    # Breakout trading
    current_price = df.iloc[index]['close']
    resistance = df.iloc[index-20:index]['high'].max()
    return current_price > resistance
```

---

**For detailed implementation guides, use:**
- `fetch_rules(["architecture"])` - Complete system architecture
- `fetch_rules(["project-structure"])` - Directory structure & organization
- `fetch_rules(["ml-models"])` - ML model training & integration
- `fetch_rules(["commands"])` - Complete command reference