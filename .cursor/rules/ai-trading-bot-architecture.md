---
description: Complete AI Trading Bot Architecture & Development Guide
globs: 
alwaysApply: true
---

# 🤖 AI Trading Bot - Complete Architecture Guide

## 🎯 **System Overview**

This is a sophisticated cryptocurrency trading system inspired by Ray Dalio's principles, focusing on **trend-following with risk containment**. The system supports both **backtesting** and **live trading** with multiple data sources, ML models, and trading strategies.

**Core Philosophy**: Trade with the trend, not against it. Protect capital above all else.

---

## 🏗️ **System Architecture**

### **High-Level Data Flow**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │───▶│ Indicator Layer │───▶│ Strategy Layer  │
│                 │    │                 │    │                 │
│ • Binance API   │    │ • RSI, EMA      │    │ • Signal Gen    │
│ • Sentiment API │    │ • Bollinger     │    │ • ML Models     │
│ • CSV Cache     │    │ • MACD, ATR     │    │ • Risk Logic    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Risk Manager   │    │ Execution Layer │
                       │                 │    │                 │
                       │ • Position Size │    │ • Live Trading  │
                       │ • Stop Loss     │    │ • Backtesting   │
                       │ • Exposure      │    │ • Order Mgmt    │
                       └─────────────────┘    └─────────────────┘
```

### **Component Architecture**
```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Trading Bot                           │
├─────────────────────────────────────────────────────────────────┤
│  Data Providers  │  Indicators  │  Strategies   │  Risk Mgmt    │
│  ┌─────────────┐ │ ┌──────────┐ │ ┌──────────┐  │ ┌──────────┐  │
│  │ Binance     │ │ │ RSI      │ │ │ Adaptive │  │ │ Position │  │
│  │ SentiCrypt  │ │ │ EMA      │ │ │ Enhanced │  │ │ Sizing   │  │
│  │ CryptoComp  │ │ │ MACD     │ │ │ ML Basic │  │ │ Stop Loss│  │
│  │ Cached      │ │ │ ATR      │ │ │ ML+Sent  │  │ │ Exposure │  │
│  └─────────────┘ │ └──────────┘ │ └──────────┘  │ └──────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Execution Layer  │  Database   │  Monitoring   │  ML Models    │
│  ┌─────────────┐  │ ┌─────────┐ │ ┌──────────┐  │ ┌──────────┐  │
│  │ Live Trading│  │ │ Trades  │ │ │ Dashboard│  │ │ ONNX     │  │
│  │ Backtesting │  │ │ Positions│ │ │ Metrics  │  │ │ Keras    │  │
│  │ Paper Trade │  │ │ Sessions│ │ │ Alerts   │  │ │ Metadata │  │
│  └─────────────┘  │ └─────────┘ │ └──────────┘  │ └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 **Directory Structure & Purpose**

### **Core Application (`src/`)**
```
src/
├── data_providers/          # Market & sentiment data adapters
│   ├── data_provider.py     # Abstract base class
│   ├── binance_data_provider.py    # Live/historical price data
│   ├── senticrypt_provider.py      # Sentiment data (SentiCrypt API)
│   ├── cryptocompare_sentiment.py  # Alternative sentiment source
│   ├── cached_data_provider.py     # API response caching
│   └── mock_data_provider.py       # Test data provider
│
├── indicators/              # Technical indicator calculations
│   └── technical.py         # RSI, EMA, MACD, ATR, Bollinger Bands
│
├── strategies/              # Trading strategy implementations
│   ├── base.py              # Abstract base class
│   ├── adaptive.py          # Adaptive EMA strategy
│   ├── enhanced.py          # Multi-indicator strategy
│   ├── ml_basic.py          # ML price prediction strategy
│   ├── ml_with_sentiment.py # Advanced ML + sentiment strategy
│   └── high_risk_high_reward.py # Aggressive trading strategy
│
├── risk/                    # Risk management system
│   └── risk_manager.py      # Position sizing, stop-loss, exposure limits
│
├── live/                    # Live trading engine
│   ├── trading_engine.py    # Main live trading orchestrator
│   └── strategy_manager.py  # Strategy hot-swapping & management
│
├── backtesting/             # Historical simulation engine
│   └── engine.py            # Vectorized backtesting with sentiment
│
├── database/                # Database management & models
│   ├── manager.py           # Database connection & operations
│   └── models.py            # SQLAlchemy models (trades, positions, etc.)
│
├── config/                  # Configuration management
│   ├── config_manager.py    # Multi-provider config system
│   ├── constants.py         # System constants
│   └── providers/           # Config providers (env, Railway, etc.)
│
├── performance/             # Performance metrics calculation
│   └── metrics.py           # Sharpe ratio, drawdown, returns
│
└── monitoring/              # Real-time monitoring dashboard
    ├── dashboard.py         # Web dashboard application
    └── templates/           # Dashboard HTML templates
```

### **Supporting Directories**
```
├── ml/                      # Trained ML models & metadata
│   ├── btcusdt_price.*      # Price prediction models (.h5/.keras/.onnx)
│   ├── btcusdt_sentiment.*  # Sentiment-enhanced models
│   └── *_metadata.json      # Model training metadata
│
├── data/                    # Cached market & sentiment data
│   ├── BTCUSDT_1d.csv       # Historical price data
│   └── senticrypt_sentiment_data.csv  # Cached sentiment data
│
├── scripts/                 # CLI utilities & automation
│   ├── run_backtest.py      # Backtesting runner
│   ├── run_live_trading.py  # Live trading launcher
│   ├── train_model.py       # ML model training
│   ├── cache_manager.py     # Data cache management
│   └── start_dashboard.py   # Monitoring dashboard launcher
│
├── tests/                   # Comprehensive test suite
│   ├── run_tests.py         # Enhanced test runner
│   ├── test_live_trading.py # Live trading tests (CRITICAL)
│   ├── test_risk_management.py # Risk management tests
│   └── test_strategies.py   # Strategy logic tests
│
├── docs/                    # Documentation & guides
├── migrations/              # Database schema migrations
└── logs/                    # Application logs
```

---

## 🚀 **Live Trading Engine (Most Important Component)**

### **Core Architecture**
The `LiveTradingEngine` is the heart of the system, orchestrating real-time trading execution.

```python
class LiveTradingEngine:
    """
    Advanced live trading engine with:
    - Real-time data streaming from Binance API
    - Strategy execution with ML model integration
    - Risk management with position sizing & stop-losses
    - Sentiment data integration (SentiCrypt API)
    - Database logging for all trades & positions
    - Graceful error handling & recovery
    - Hot-swapping strategies without stopping
    - Performance monitoring & alerts
    """
```

### **Key Features**

#### **1. Safety First Design**
- **Paper Trading Mode** (default) - No real money at risk
- **Explicit Risk Acknowledgment** - Must confirm for live trading
- **Position Size Limits** - Maximum 10% of balance per position
- **Stop Loss Protection** - Automatic loss limiting
- **Error Recovery** - Graceful handling of API failures

#### **2. Real-Time Data Processing**
```python
# Data flow in trading loop
while self.is_running:
    # 1. Fetch latest market data
    df = self._get_latest_data(symbol, timeframe)
    
    # 2. Add sentiment data (if available)
    df = self._add_sentiment_data(df, symbol)
    
    # 3. Calculate indicators
    df = self.strategy.calculate_indicators(df)
    
    # 4. Check entry/exit conditions
    self._check_entry_conditions(df, current_index, symbol, current_price)
    self._check_exit_conditions(df, current_index, current_price)
    
    # 5. Update positions & performance
    self._update_position_pnl(current_price)
    self._update_performance_metrics()
```

#### **3. Strategy Integration**
- **Hot-Swapping**: Change strategies without stopping trading
- **ML Model Integration**: Real-time ONNX model inference
- **Sentiment Analysis**: Live sentiment data from SentiCrypt API
- **Multi-Strategy Support**: Adaptive, Enhanced, ML-based strategies

#### **4. Risk Management**
```python
# Position sizing with risk management
position_size = self.risk_manager.calculate_position_size(
    price=current_price,
    atr=atr_value,
    balance=self.current_balance,
    regime=market_regime
)

# Stop loss calculation
stop_loss = self.risk_manager.calculate_stop_loss(
    entry_price=entry_price,
    atr=atr_value,
    side=position_side
)
```

#### **5. Database Logging**
- **Trade Logging**: All completed trades with P&L
- **Position Tracking**: Active positions with unrealized P&L
- **Performance Metrics**: Real-time Sharpe ratio, drawdown
- **Strategy Execution**: Detailed decision logs

### **Live Trading Commands**
```bash
# Paper trading (safe - no real money)
python scripts/run_live_trading.py adaptive --paper-trading

# Live trading (requires explicit confirmation)
python scripts/run_live_trading.py ml_with_sentiment --live-trading --i-understand-the-risks

# With custom settings
python scripts/run_live_trading.py adaptive --balance 5000 --max-position 0.05

# Monitor trading dashboard
python scripts/start_dashboard.py
```

---

## 🧠 **Machine Learning Integration**

### **ML Model Architecture**

#### **Model Types**
1. **Price Prediction Models** (`btcusdt_price.*`)
   - Input: 120 time steps × 5 features (OHLCV)
   - Architecture: CNN + LSTM + Dense layers
   - Output: Single price prediction

2. **Sentiment-Enhanced Models** (`btcusdt_sentiment.*`)
   - Input: 120 time steps × 13 features (5 price + 8 sentiment)
   - Architecture: CNN + LSTM + Dense layers
   - Output: Price prediction with confidence score

#### **Feature Engineering**
```python
# Price Features (MinMax normalization)
price_features = ['close', 'volume', 'high', 'low', 'open']

# Sentiment Features (StandardScaler)
sentiment_features = [
    'sentiment_score', 'sentiment_momentum', 'sentiment_volatility',
    'extreme_positive', 'extreme_negative',
    'sentiment_ma_3', 'sentiment_ma_7', 'sentiment_ma_14'
]
```

#### **Model Training Workflow**
```bash
# Train new models
python scripts/train_model.py BTCUSDT --force-sentiment

# Safe training (staging environment)
python scripts/safe_model_trainer.py

# Validate models
python scripts/simple_model_validator.py
```

#### **Live Trading Integration**
```python
# Real-time prediction in strategies
def generate_prediction(self, df: pd.DataFrame, index: int):
    # Prepare input sequence
    sequence = self._prepare_sequence(df, index)
    
    # Run ONNX inference
    prediction = self.ort_session.run(
        None, {self.input_name: sequence}
    )[0][0]
    
    # Calculate confidence
    confidence = self._calculate_confidence(prediction, df, index)
    
    return prediction, confidence
```

---

## 📊 **Strategy System**

### **Strategy Base Class**
All strategies inherit from `BaseStrategy` and implement:

```python
class BaseStrategy(ABC):
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific indicators"""
        pass
        
    @abstractmethod
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if entry conditions are met"""
        pass
        
    @abstractmethod
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check if exit conditions are met"""
        pass
        
    @abstractmethod
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size for new trade"""
        pass
```

### **Available Strategies**

#### **1. Adaptive Strategy** (`adaptive.py`)
- **Logic**: Adaptive EMA crossover with market regime detection
- **Features**: Adjusts parameters based on volatility
- **Best For**: Trend-following in various market conditions

#### **2. Enhanced Strategy** (`enhanced.py`)
- **Logic**: Multi-indicator confluence (RSI + EMA + MACD)
- **Features**: Confirmation from multiple technical indicators
- **Best For**: Conservative trading with high confirmation

#### **3. ML Basic Strategy** (`ml_basic.py`)
- **Logic**: Uses ML price predictions for entry/exit decisions
- **Features**: Confidence-based position sizing
- **Best For**: Data-driven trading decisions

#### **4. ML with Sentiment Strategy** (`ml_with_sentiment.py`)
- **Logic**: Combines ML predictions with sentiment analysis
- **Features**: Graceful fallback when sentiment data unavailable
- **Best For**: Maximum prediction accuracy with sentiment context

#### **5. High Risk High Reward Strategy** (`high_risk_high_reward.py`)
- **Logic**: Aggressive trading with higher risk tolerance
- **Features**: Larger position sizes, tighter stops
- **Best For**: Experienced traders seeking higher returns

---

## 🛡️ **Risk Management System**

### **Risk Parameters**
```python
@dataclass
class RiskParameters:
    base_risk_per_trade: float = 0.02      # 2% risk per trade
    max_risk_per_trade: float = 0.03       # 3% maximum risk per trade
    max_position_size: float = 0.25        # 25% maximum position size
    max_daily_risk: float = 0.06           # 6% maximum daily risk
    max_drawdown: float = 0.20             # 20% maximum drawdown
```

### **Position Sizing Logic**
```python
def calculate_position_size(self, price: float, atr: float, balance: float, regime: str):
    # Adjust risk based on market regime
    if regime == 'trending':
        risk = base_risk * 1.5  # More aggressive
    elif regime == 'volatile':
        risk = base_risk * 0.6  # More conservative
    
    # Calculate position size based on ATR
    risk_amount = balance * risk
    atr_stop = atr * self.params.position_size_atr_multiplier
    position_size = risk_amount / atr_stop
    
    # Ensure position size doesn't exceed maximum
    max_position_value = balance * self.params.max_position_size
    position_size = min(position_size, max_position_value / price)
    
    return max(0.0, position_size)
```

---

## 🗄️ **Database Architecture**

### **Core Tables**
```sql
-- Trading sessions
trading_sessions (id, strategy_name, symbol, mode, initial_balance, start_time)

-- Completed trades
trades (id, symbol, side, entry_price, exit_price, pnl, strategy_name, session_id)

-- Active positions
positions (id, symbol, side, entry_price, current_price, unrealized_pnl, session_id)

-- Account history
account_history (id, timestamp, balance, equity, drawdown, session_id)

-- Performance metrics
performance_metrics (id, period, total_return, sharpe_ratio, max_drawdown, session_id)

-- Strategy execution logs
strategy_executions (id, strategy_name, signal_type, indicators, sentiment_data, ml_predictions)
```

### **Database Features**
- **ACID Transactions**: Critical for financial data integrity
- **Connection Pooling**: Efficient resource management
- **Indexed Queries**: Fast performance for time-series data
- **JSONB Support**: Flexible storage for strategy configurations

---

## 🔧 **Key Commands & Workflows**

### **Backtesting**
```bash
# Quick backtest (development)
python scripts/run_backtest.py adaptive --days 30 --no-db

# Production backtest (with database logging)
python scripts/run_backtest.py ml_with_sentiment --days 365

# Custom parameters
python scripts/run_backtest.py enhanced --symbol ETHUSDT --days 100 --initial-balance 50000
```

### **Live Trading**
```bash
# Paper trading (safe)
python scripts/run_live_trading.py adaptive --paper-trading

# Live trading (requires confirmation)
python scripts/run_live_trading.py ml_with_sentiment --live-trading --i-understand-the-risks

# With custom balance
python scripts/run_live_trading.py adaptive --balance 10000 --max-position 0.05
```

### **Model Training**
```bash
# Train new models
python scripts/train_model.py BTCUSDT --force-sentiment

# Safe training (staging)
python scripts/safe_model_trainer.py

# Validate models
python scripts/simple_model_validator.py
```

### **Testing**
```bash
# Quick smoke tests
python tests/run_tests.py smoke

# Critical tests (live trading + risk management)
python tests/run_tests.py critical

# All tests with coverage
python tests/run_tests.py all --coverage

# Specific test file
python tests/run_tests.py --file test_strategies.py
```

### **Monitoring**
```bash
# Start monitoring dashboard
python scripts/start_dashboard.py

# Health check
python scripts/health_check.py

# Cache management
python scripts/cache_manager.py --check
```

---

## 🔄 **Data Flow Diagrams**

### **Live Trading Data Flow**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Binance API │───▶│ Data Cache  │───▶│ Indicators  │───▶│ Strategy    │
│             │    │             │    │             │    │             │
│ Real-time   │    │ Reduce API  │    │ RSI, EMA,   │    │ Signal Gen  │
│ OHLCV Data  │    │ Calls       │    │ MACD, ATR   │    │ ML Predict  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ SentiCrypt  │───▶│ Sentiment   │───▶│ Risk Mgmt   │◀───│ Position    │
│ API         │    │ Processing  │    │             │    │ Sizing      │
│             │    │             │    │ Stop Loss   │    │             │
│ Sentiment   │    │ Feature Eng │    │ Exposure    │    │ Entry/Exit  │
│ Scores      │    │             │    │ Limits      │    │ Logic       │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Database    │◀───│ Order Exec  │◀───│ Position    │◀───│ Trade       │
│             │    │             │    │ Management  │    │ Execution   │
│ Log Trades  │    │ Binance API │    │ P&L Update  │    │             │
│ Track P&L   │    │             │    │ Stop Loss   │    │ Real Orders │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### **ML Model Training Flow**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Historical  │───▶│ Feature     │───▶│ Model       │───▶│ Model       │
│ Price Data  │    │ Engineering │    │ Training    │    │ Validation  │
│             │    │             │    │             │    │             │
│ OHLCV +     │    │ Normalize   │    │ CNN + LSTM  │    │ Backtest    │
│ Sentiment   │    │ Features    │    │ + Dense     │    │ Performance │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Model       │◀───│ ONNX Export │◀───│ Model       │◀───│ Performance │
│ Deployment  │    │             │    │ Selection   │    │ Analysis    │
│             │    │ Optimize    │    │             │    │             │
│ Live        │    │ Inference   │    │ Best Model  │    │ Sharpe,     │
│ Trading     │    │ Speed       │    │ Selection   │    │ Drawdown    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 🎯 **Development Workflows**

### **Creating a New Strategy**
1. **Create Strategy Class**
```python
from strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, name="MyStrategy"):
        super().__init__(name)
        self.trading_pair = 'BTCUSDT'
        self.stop_loss_pct = 0.02
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add your indicators
        df['my_indicator'] = self._calculate_my_indicator(df)
        return df
        
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        # Your entry logic
        return df.iloc[index]['my_indicator'] > threshold
        
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        # Your exit logic
        return df.iloc[index]['my_indicator'] < exit_threshold
        
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        # Your position sizing logic
        return balance * 0.02  # 2% risk
```

2. **Add to Strategy Registry**
```python
# In strategies/__init__.py
from .my_strategy import MyStrategy

__all__ = [
    'AdaptiveStrategy',
    'EnhancedStrategy', 
    'MyStrategy',  # Add your strategy
    # ...
]
```

3. **Test with Backtesting**
```bash
python scripts/run_backtest.py my_strategy --days 30 --no-db
```

4. **Test with Paper Trading**
```bash
python scripts/run_live_trading.py my_strategy --paper-trading
```

### **Adding New Data Sources**
1. **Create Data Provider**
```python
from data_providers.data_provider import DataProvider

class MyDataProvider(DataProvider):
    def get_historical_data(self, symbol: str, timeframe: str, start: datetime, end: datetime):
        # Your data fetching logic
        return pd.DataFrame(...)
        
    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100):
        # Your live data logic
        return pd.DataFrame(...)
```

2. **Use in Backtesting/Live Trading**
```python
provider = MyDataProvider()
backtester = Backtester(strategy=strategy, data_provider=provider)
```

---

## 🔍 **Troubleshooting & Debugging**

### **Common Issues**

#### **1. API Rate Limits**
```bash
# Check API status
python scripts/health_check.py

# Use cached data provider
provider = CachedDataProvider(BinanceDataProvider())
```

#### **2. Model Loading Errors**
```bash
# Validate model files
python scripts/simple_model_validator.py

# Check model metadata
cat ml/btcusdt_sentiment_metadata.json
```

#### **3. Database Connection Issues**
```bash
# Test database connection
python scripts/test_database.py

# Check database schema
python scripts/verify_database_connection.py
```

#### **4. Strategy Performance Issues**
```bash
# Run strategy tests
python tests/run_tests.py --file test_strategies.py

# Check strategy logs
tail -f logs/trading.log
```

### **Debug Commands**
```bash
# Verbose backtesting
python scripts/run_backtest.py adaptive --days 7 --no-db --verbose

# Debug live trading
python scripts/run_live_trading.py adaptive --paper-trading --debug

# Check cache status
python scripts/cache_manager.py --status

# Monitor system resources
python scripts/health_check.py --detailed
```

---

## 📈 **Performance Monitoring**

### **Key Metrics Tracked**
- **Total P&L**: Cumulative profit/loss
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Total Trades**: Number of completed trades
- **Average Trade**: Average P&L per trade

### **Real-Time Dashboard**
```bash
# Start monitoring dashboard
python scripts/start_dashboard.py

# Access at: http://localhost:8080
```

### **Performance Analysis**
```python
# Get performance summary
summary = trading_engine.get_performance_summary()
print(f"Total P&L: ${summary['total_pnl']:.2f}")
print(f"Win Rate: {summary['win_rate']:.2f}%")
print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
```

---

## 🚨 **Safety & Risk Management**

### **Critical Safety Features**
1. **Paper Trading Default**: No real money at risk by default
2. **Explicit Risk Confirmation**: Must acknowledge risks for live trading
3. **Position Size Limits**: Maximum 10% of balance per position
4. **Stop Loss Protection**: Automatic loss limiting
5. **Daily Risk Limits**: Maximum 6% daily risk
6. **Drawdown Protection**: Stop trading at 20% drawdown

### **Emergency Procedures**
```bash
# Stop live trading immediately
python scripts/run_live_trading.py --stop

# Check current positions
python scripts/health_check.py --positions

# Emergency database backup
python scripts/backup_database.py
```

---

## 🔮 **Future Enhancements**

### **Planned Features**
- **Multi-Exchange Support**: Binance, Coinbase, Kraken
- **Portfolio Management**: Multi-asset trading
- **Advanced ML Models**: Transformer-based models
- **Real-Time Sentiment**: Multiple sentiment sources
- **Automated Model Retraining**: Scheduled model updates
- **Advanced Risk Models**: VaR, CVaR calculations

### **Scalability Considerations**
- **Microservices Architecture**: Separate services for data, ML, execution
- **Message Queues**: Redis/RabbitMQ for async processing
- **Load Balancing**: Multiple trading instances
- **Database Sharding**: Horizontal scaling for high-frequency trading

---

**Remember**: This is real money. Always validate changes thoroughly. When in doubt, backtest more. 🛡️