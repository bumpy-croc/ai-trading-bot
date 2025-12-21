---
description: Trading Bot Core Information & Essential Guidelines
globs:
alwaysApply: true
---

# 🤖 Trading Bot Core

## System Overview
Cryptocurrency trading system with trend-following risk containment. Supports backtesting, live trading, ML models, and multiple data sources.

**Philosophy**: Trade with the trend, not against it. Protect capital above all else.

---

## Core Architecture
```
Data Providers → Indicators → Strategies → Risk Manager → Execution Layer
```

### Key Directories
- `src/strategies/` - Trading strategy implementations
- `src/live/` - Live trading engine (core component)
- `src/backtesting/` - Historical simulation engine
- `src/risk/` - Risk management system
- `src/data_providers/` - Market & sentiment data adapters
- `src/ml/` - Trained ML models (.h5/.keras/.onnx)
- `scripts/` - CLI utilities & automation
- `tests/` - Comprehensive test suite

---

## Live Trading Engine

### Features
- Real-time data streaming from Binance API
- Strategy execution with ML model integration
- Risk management with position sizing & stop-losses
- Sentiment data integration (removed)
- Database logging for all trades & positions
- Hot-swapping strategies without stopping

### Safety Features
- **Paper Trading Mode** (default) - No real money at risk
- **Explicit Risk Confirmation** - Must confirm for live trading
- **Position Size Limits** - Maximum 10% of balance per position
- **Stop Loss Protection** - Automatic loss limiting

---

## Strategy Component Interface
All strategies use component-based architecture with `Strategy` class that composes:
- `SignalGenerator`: Generates trading signals (BUY/SELL/HOLD) with confidence
- `RiskManager`: Calculates risk-based position sizes and stop losses
- `PositionSizer`: Determines final position size based on signal and risk
- `RegimeDetector` (optional): Detects market regimes for adaptive behavior

Main interface: `strategy.process_candle(df, index, balance, positions) -> TradingDecision`

---

## Available Strategies (as implemented)
- `MlBasic` (`src/strategies/ml_basic.py`)
- `MlAdaptive` (`src/strategies/ml_adaptive.py`)
- `Bull` (`src/strategies/bull.py`)
- `Bear` (`src/strategies/bear.py`)
Registry exports in `src/strategies/__init__.py`.

---

## ML Model Integration
- **Price Prediction Models** (`btcusdt_price.*`) - 5 features (OHLCV)
- **Sentiment-Enhanced Models** (`btcusdt_sentiment.*`) - 13 features (5 price + 8 sentiment)
- **Real-time ONNX inference** with confidence-based position sizing

---

## Risk Management (defaults in `RiskParameters`)
```python
base_risk_per_trade = 0.02
max_risk_per_trade = 0.03
max_position_size = 0.25
max_daily_risk = 0.06
max_drawdown = 0.20
```

---

## Critical Safety Rules

### Live Trading Safety
- **Paper Trading Default**: All live trading tests run in paper trading mode by default
- **Explicit Risk Confirmation**: Must use `--live-trading --i-understand-the-risks` for real money
- **Position Size Limits**: Maximum 10% of balance per position
- **Stop Loss Protection**: Automatic loss limiting on all positions
- **Daily Risk Limits**: Maximum 6% daily risk exposure
- **Drawdown Protection**: Stop trading at 20% maximum drawdown

### Emergency Procedures
```bash
# Stop live trading immediately
atb live-control emergency-stop

# Check current positions and health
PORT=8000 atb live-health -- ml_basic --symbol BTCUSDT --paper-trading

# Emergency database backup
python scripts/backup_database.py --emergency
```

### Warning Signs (STOP IMMEDIATELY)
- **Drawdown > 20%**: Stop all trading
- **Consecutive losses > 5**: Review strategy
- **API errors > 10**: Check connectivity
- **Position size > 25%**: Reduce immediately

---

## Essential Commands

### Quick Development
```bash
# Quick backtest (development)
atb backtest ml_adaptive --symbol BTCUSDT --timeframe 1h --days 30

# Paper trading (safe)
atb live ml_adaptive --symbol BTCUSDT --paper-trading

# Quick tests
python tests/run_tests.py smoke
```

### Production
```bash
# Production backtest
atb backtest ml_adaptive --symbol BTCUSDT --timeframe 1h --days 365

# Live trading (requires confirmation)
atb live ml_adaptive --symbol BTCUSDT --live-trading --i-understand-the-risks

# Monitor dashboard
atb dashboards run monitoring --port 8000
```

### Safety
```bash
# Health check
python scripts/health_check.py

# Critical tests
python tests/run_tests.py critical

# Emergency stop
atb live-control emergency-stop
```

---

## Natural Language Commands

### Testing
- "run smoke tests" → `python tests/run_tests.py smoke`
- "run unit tests" → `python tests/run_tests.py unit`
- "run critical tests" → `python tests/run_tests.py critical`
- "run all tests" → `python tests/run_tests.py all`

### Backtesting
- "run backtest" → `atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30`
- "run backtest for [strategy]" → `atb backtest [strategy] --symbol BTCUSDT --timeframe 1h --days 30`
- "run production backtest" → `atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 365`

### Live Trading
- "start paper trading" → `atb live ml_basic --symbol BTCUSDT --paper-trading`
- "start live trading" → `atb live ml_basic --symbol BTCUSDT --live-trading --i-understand-the-risks`
- "start dashboard" → `atb dashboards run monitoring --port 8000`

### Health & Monitoring
- "check health" → `PORT=8000 atb live-health -- ml_basic --symbol BTCUSDT --paper-trading`
- "check positions" → `atb db verify`
- "check cache" → `atb data cache-manager info`

---

## Detailed Guides
- `fetch_rules(["architecture"])` - Complete system architecture
- `fetch_rules(["strategies"])` - Strategy development details
- `fetch_rules(["ml-models"])` - ML model training & integration
- `fetch_rules(["commands"])` - Complete command reference

---

**Remember**: This is real money. Always validate changes thoroughly. When in doubt, backtest more. 🛡️
