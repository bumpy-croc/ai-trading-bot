# Strategies

Component-based trading strategies using composable signal generators, risk managers, and position sizers.

## Architecture

Strategies are built using a component-based architecture that promotes reusability and testability:

- **Strategy**: Orchestrates components and produces `TradingDecision` objects
- **SignalGenerator**: Analyzes market data and generates trading signals
- **RiskManager**: Calculates risk-adjusted position sizes
- **PositionSizer**: Determines final position sizes based on confidence and risk
- **RegimeDetector**: Detects market regimes for adaptive behavior

## Built-in Strategies

- `ml_basic.py`: ONNX price model predictions (price-only)
- `ml_sentiment.py`: ONNX model predictions with sentiment analysis
- `ml_adaptive.py`: Adaptive ML strategy with regime detection
- `ensemble_weighted.py`: Weighted ensemble strategy combining multiple strategies
- `momentum_leverage.py`: Aggressive momentum-based strategy with pseudo-leverage

## Usage (with backtester)

```bash
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 90
atb backtest ml_sentiment --symbol BTCUSDT --timeframe 1h --days 90
atb backtest ml_adaptive --symbol BTCUSDT --timeframe 1h --days 90
atb backtest ensemble_weighted --symbol BTCUSDT --timeframe 1h --days 90
atb backtest momentum_leverage --symbol BTCUSDT --timeframe 1h --days 90
```

## Strategy Details

### ML Basic Strategy
- Uses ONNX models for price predictions
- Fixed risk management (2% per trade)
- Confidence-weighted position sizing
- Simple and reliable for trending markets

### ML Sentiment Strategy
- Combines price predictions with Fear & Greed Index sentiment
- Adaptive position sizing based on sentiment confidence
- Enhanced prediction accuracy during volatile market conditions
- Robust fallback when sentiment data is unavailable
- Supports both BTC and ETH models with sentiment integration

### ML Adaptive Strategy
- Regime-aware ML predictions with adaptive thresholds
- Regime-specific risk management (conservative in volatile markets)
- Regime-adaptive position sizing
- Optimized for different market conditions

### Ensemble Weighted Strategy
- Combines multiple signal generators using weighted voting
- Performance-based dynamic weighting that adapts over time
- Aggressive position sizing (up to 80% allocation) with pseudo-leverage
- Advanced momentum and trend indicators for enhanced entry timing
- Wide stop losses (6%) and high profit targets (20%) to capture major moves

### Momentum Leverage Strategy
- Pure momentum-based approach with pseudo-leverage for beating buy-and-hold
- Ultra-aggressive position sizing (40-95% allocation) based on momentum strength
- Multi-timeframe momentum analysis (3, 7, 20 periods)
- Trend confirmation using exponential moving averages
- Volatility-based position scaling for optimal risk management

## Creating Custom Strategies

Strategies are created by composing components. Here's how to create your own:

### Example: Simple Moving Average Strategy

```python
from src.strategies.components.strategy import Strategy
from src.strategies.components.signal_generator import SignalGenerator
from src.strategies.components.risk_manager import FixedRiskManager
from src.strategies.components.position_sizer import FixedFractionSizer
from src.strategies.components.types import Signal, SignalDirection
import pandas as pd

class SimpleMASignalGenerator(SignalGenerator):
    """Generate signals based on moving average crossover"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signal(self, df: pd.DataFrame, index: int, 
                       regime: Optional[RegimeContext] = None) -> Signal:
        """Generate signal based on MA crossover"""
        if index < self.slow_period:
            return Signal(direction=SignalDirection.HOLD, confidence=0.0)
        
        # Calculate moving averages
        fast_ma = df['close'].iloc[index - self.fast_period:index].mean()
        slow_ma = df['close'].iloc[index - self.slow_period:index].mean()
        
        # Previous values for crossover detection
        prev_fast = df['close'].iloc[index - self.fast_period - 1:index - 1].mean()
        prev_slow = df['close'].iloc[index - self.slow_period - 1:index - 1].mean()
        
        # Detect crossover
        if fast_ma > slow_ma and prev_fast <= prev_slow:
            return Signal(
                direction=SignalDirection.BUY,
                confidence=0.7,
                strength=1.0,
                metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma}
            )
        elif fast_ma < slow_ma and prev_fast >= prev_slow:
            return Signal(
                direction=SignalDirection.SELL,
                confidence=0.7,
                strength=1.0,
                metadata={'fast_ma': fast_ma, 'slow_ma': slow_ma}
            )
        
        return Signal(direction=SignalDirection.HOLD, confidence=0.0)

def create_simple_ma_strategy(name: str = "simple_ma") -> Strategy:
    """Create a simple moving average crossover strategy"""
    signal_generator = SimpleMASignalGenerator(fast_period=10, slow_period=30)
    risk_manager = FixedRiskManager(risk_per_trade=0.02)
    position_sizer = FixedFractionSizer(fraction=0.1)
    
    return Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer
    )
```

### Component Interface

#### SignalGenerator

```python
class SignalGenerator(ABC):
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, index: int,
                       regime: Optional[RegimeContext] = None) -> Signal:
        """Generate a trading signal for the given candle"""
        pass
```

#### RiskManager

```python
class RiskManager(ABC):
    @abstractmethod
    def calculate_position_size(self, signal: Signal, balance: float,
                               current_price: float,
                               regime: Optional[RegimeContext] = None) -> float:
        """Calculate risk-adjusted position size"""
        pass
```

#### PositionSizer

```python
class PositionSizer(ABC):
    @abstractmethod
    def calculate_position_size(self, signal: Signal, balance: float,
                               current_price: float,
                               regime: Optional[RegimeContext] = None) -> float:
        """Calculate final position size"""
        pass
```

### Strategy Workflow

When the backtesting or live trading engine calls `strategy.process_candle()`:

1. **Regime Detection** (optional): Detect current market regime
2. **Signal Generation**: Generate trading signal based on market data
3. **Risk Management**: Calculate risk-adjusted position size
4. **Position Sizing**: Calculate final position size based on confidence
5. **Return Decision**: Return `TradingDecision` with all context

```python
decision = strategy.process_candle(df, index=100, balance=10000.0)

# decision contains:
# - signal: Signal with direction, confidence, strength
# - position_size: Final calculated position size
# - regime: Market regime context (if available)
# - risk_metrics: Risk-related metrics
# - metadata: Additional decision context
```

## Testing Strategies

Test strategies by testing components individually and then testing composition:

```python
def test_signal_generator():
    """Test signal generator in isolation"""
    signal_gen = SimpleMASignalGenerator(fast_period=10, slow_period=30)
    df = create_test_data()
    
    signal = signal_gen.generate_signal(df, index=50)
    assert signal.direction in [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]
    assert 0 <= signal.confidence <= 1

def test_strategy_composition():
    """Test complete strategy"""
    strategy = create_simple_ma_strategy()
    df = create_test_data()
    
    decision = strategy.process_candle(df, index=50, balance=10000.0)
    assert decision.signal is not None
    assert decision.position_size >= 0
```
