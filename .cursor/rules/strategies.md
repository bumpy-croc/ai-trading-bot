---
description: Trading Bot Strategy Development & Implementation Guide
globs: 
alwaysApply: false
---

# 📊 Trading Bot Strategy Development

## 🎯 Strategy System Overview

All strategies inherit from `BaseStrategy` and implement a standardized interface for signal generation, position sizing, and risk management.

---

## 🏗️ Strategy Base Class

### **Required Interface**
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

### **Strategy Execution Logging**
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

## 📈 Available Strategies

### **1. Adaptive Strategy** (`adaptive.py`)
**Purpose**: Adaptive EMA crossover with market regime detection

**Key Features**:
- **Adaptive Parameters**: Adjusts based on market volatility
- **Market Regime Detection**: Different logic for trending vs ranging markets
- **Dynamic Stop Losses**: ATR-based stop loss calculation

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

### **2. Enhanced Strategy** (`enhanced.py`)
**Purpose**: Multi-indicator confluence for high confirmation

**Key Features**:
- **Multi-Indicator Confirmation**: RSI + EMA + MACD
- **Conservative Approach**: Requires multiple confirmations
- **Volume Confirmation**: Uses volume for signal validation

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

### **3. ML Basic Strategy** (`ml_basic.py`)
**Purpose**: Uses ML price predictions for entry/exit decisions

**Key Features**:
- **ML Model Integration**: Real-time ONNX inference
- **Confidence-Based Sizing**: Position size based on prediction confidence
- **Price Prediction**: Uses trained neural network for price forecasting

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

### **4. ML with Sentiment Strategy** (`ml_with_sentiment.py`)
**Purpose**: Combines ML predictions with sentiment analysis

**Key Features**:
- **Multi-Modal Predictions**: Price + sentiment data (13 features)
- **Graceful Fallback**: Works without sentiment data
- **Sentiment Confidence**: Weights predictions based on sentiment quality

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

### **5. High Risk High Reward Strategy** (`high_risk_high_reward.py`)
**Purpose**: Aggressive trading with higher risk tolerance

**Key Features**:
- **Larger Position Sizes**: Up to 25% of balance per position
- **Tighter Stops**: More aggressive stop loss levels
- **Higher Risk Tolerance**: Designed for experienced traders

**Logic**:
```python
def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
    # Aggressive momentum-based entry
    rsi = df.iloc[index]['rsi']
    momentum = df.iloc[index]['close'] / df.iloc[index-1]['close'] - 1
    volume_spike = df.iloc[index]['volume'] > df.iloc[index]['volume'].rolling(10).mean() * 1.5
    
    return rsi > 30 and rsi < 80 and momentum > 0.005 and volume_spike
```

**Best For**: Experienced traders seeking higher returns

---

## 🛠️ Creating New Strategies

### **Step 1: Create Strategy Class**
```python
from strategies.base import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):
    def __init__(self, name="MyStrategy"):
        super().__init__(name)
        self.trading_pair = 'BTCUSDT'
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add your custom indicators
        df['my_indicator'] = self._calculate_my_indicator(df)
        df['my_signal'] = self._calculate_signal_strength(df)
        return df
        
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        # Your entry logic
        signal_strength = df.iloc[index]['my_signal']
        indicator_value = df.iloc[index]['my_indicator']
        
        return signal_strength > 0.7 and indicator_value > threshold
        
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        # Your exit logic
        current_price = df.iloc[index]['close']
        signal_strength = df.iloc[index]['my_signal']
        
        # Exit on signal reversal or profit target
        profit_pct = (current_price - entry_price) / entry_price
        return signal_strength < 0.3 or profit_pct >= self.take_profit_pct
        
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        # Your position sizing logic
        signal_strength = df.iloc[index]['my_signal']
        base_size = balance * 0.02  # 2% base risk
        
        # Scale position size with signal strength
        return base_size * signal_strength
        
    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = 'long') -> float:
        # Your stop loss logic
        atr = df.iloc[index]['atr']
        
        if side == 'long':
            return price - (atr * 2)  # 2 ATR below entry
        else:
            return price + (atr * 2)  # 2 ATR above entry
            
    def get_parameters(self) -> dict:
        return {
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'trading_pair': self.trading_pair
        }
        
    def _calculate_my_indicator(self, df: pd.DataFrame) -> pd.Series:
        # Your custom indicator calculation
        return df['close'].rolling(20).mean() / df['close'] - 1
        
    def _calculate_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        # Your signal strength calculation
        return df['my_indicator'].rolling(5).mean()
```

### **Step 2: Add to Strategy Registry**
```python
# In strategies/__init__.py
from .my_strategy import MyStrategy

__all__ = [
    'AdaptiveStrategy',
    'EnhancedStrategy', 
    'MlBasic',
    'MlWithSentiment',
    'HighRiskHighRewardStrategy',
    'MyStrategy',  # Add your strategy
]
```

### **Step 3: Test with Backtesting**
```bash
# Quick backtest (development)
python scripts/run_backtest.py my_strategy --days 30 --no-db

# Production backtest (with database logging)
python scripts/run_backtest.py my_strategy --days 365
```

### **Step 4: Test with Paper Trading**
```bash
# Paper trading validation
python scripts/run_live_trading.py my_strategy --paper-trading
```

---

## 📊 Strategy Performance Analysis

### **Key Metrics to Monitor**
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Average Trade**: Average P&L per trade
- **Consecutive Losses**: Maximum losing streak

### **Strategy Validation**
```bash
# Run strategy tests
python tests/run_tests.py --file test_strategies.py

# Performance analysis
python scripts/analyze_btc_data.py --strategy my_strategy

# Risk analysis
python tests/run_tests.py --file test_risk_management.py
```

---

## 🔧 Strategy Optimization

### **Parameter Optimization**
```python
# Use grid search or genetic algorithms
def optimize_strategy_parameters(strategy_class, data, param_ranges):
    best_params = None
    best_sharpe = -999
    
    for params in generate_param_combinations(param_ranges):
        strategy = strategy_class(**params)
        results = backtest_strategy(strategy, data)
        
        if results['sharpe_ratio'] > best_sharpe:
            best_sharpe = results['sharpe_ratio']
            best_params = params
    
    return best_params
```

### **Market Regime Adaptation**
```python
def adapt_to_market_regime(df: pd.DataFrame) -> str:
    """Detect market regime and adjust strategy parameters"""
    volatility = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    trend_strength = abs(df['close'] - df['close'].rolling(50).mean()) / df['close']
    
    if volatility.mean() > 0.03:
        return 'volatile'
    elif trend_strength.mean() > 0.05:
        return 'trending'
    else:
        return 'ranging'
```

---

## 🚨 Strategy Safety Checklist

### **Before Live Trading**
- [ ] **Backtesting**: Minimum 6 months of historical data
- [ ] **Paper Trading**: At least 1 week of paper trading validation
- [ ] **Risk Limits**: Verify position sizing and stop losses
- [ ] **Error Handling**: Test with network failures and API errors
- [ ] **Performance**: Validate against multiple market conditions

### **Ongoing Monitoring**
- [ ] **Daily P&L**: Monitor daily performance
- [ ] **Drawdown**: Track maximum drawdown
- [ ] **Win Rate**: Monitor win rate consistency
- [ ] **Market Conditions**: Adapt to changing market regimes
- [ ] **Model Drift**: Retrain ML models if performance degrades

---

**For detailed implementation guides, use:**
- `fetch_rules(["architecture"])` - Complete system architecture
- `fetch_rules(["project-structure"])` - Directory structure & organization
- `fetch_rules(["ml-models"])` - ML model training & integration
- `fetch_rules(["commands"])` - Complete command reference