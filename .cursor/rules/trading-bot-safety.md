---
description: Critical Safety & Risk Management for Trading Bot
globs: 
alwaysApply: true
---

# 🛡️ Trading Bot Safety & Risk Management

## 🚨 Critical Safety Rules (ALWAYS APPLIED)

### **Live Trading Safety**
- **Paper Trading Default**: All live trading tests run in paper trading mode by default
- **Explicit Risk Confirmation**: Must use `--live-trading --i-understand-the-risks` for real money
- **Position Size Limits**: Maximum 10% of balance per position
- **Stop Loss Protection**: Automatic loss limiting on all positions
- **Daily Risk Limits**: Maximum 6% daily risk exposure
- **Drawdown Protection**: Stop trading at 20% maximum drawdown

### **Risk Management Parameters**
```python
# Default risk limits (DO NOT EXCEED without explicit approval)
base_risk_per_trade: float = 0.02      # 2% risk per trade
max_risk_per_trade: float = 0.03       # 3% maximum risk per trade
max_position_size: float = 0.25        # 25% maximum position size
max_daily_risk: float = 0.06           # 6% maximum daily risk
max_drawdown: float = 0.20             # 20% maximum drawdown
```

### **Emergency Procedures**
```bash
# Stop live trading immediately
python scripts/run_live_trading.py --stop

# Check current positions
python scripts/health_check.py --positions

# Emergency database backup
python scripts/backup_database.py
```

### **Testing Safety**
- **Always run tests** before deploying changes
- **Critical tests first**: `python tests/run_tests.py critical`
- **Paper trading validation** before live trading
- **Backtesting validation** before strategy changes

### **Code Safety**
- **Never commit API keys** or secrets
- **Validate all inputs** from external sources
- **Use environment variables** for sensitive data
- **Implement rate limiting** for API calls
- **Graceful error handling** for all external dependencies

### **Database Safety**
- **ACID transactions** for all financial operations
- **Backup before migrations** or schema changes
- **Validate data integrity** after operations
- **Connection pooling** to prevent resource exhaustion

---

## ⚠️ Warning Signs (STOP IMMEDIATELY)

### **Risk Thresholds**
- **Drawdown > 20%**: Stop all trading
- **Consecutive losses > 5**: Review strategy
- **API errors > 10**: Check connectivity
- **Position size > 25%**: Reduce immediately

### **System Health**
- **Database connection failures**: Check connectivity
- **Model loading errors**: Validate model files
- **Strategy execution errors**: Review strategy logic
- **Memory/CPU spikes**: Check for resource leaks

---

## 🔧 Safety Commands

### **Health Checks**
```bash
# Quick health check
python scripts/health_check.py

# Detailed system check
python scripts/health_check.py --detailed

# Check API connectivity
python scripts/test_database.py
```

### **Safe Testing**
```bash
# Run critical safety tests
python tests/run_tests.py critical

# Test risk management
python tests/run_tests.py --file test_risk_management.py

# Validate models
python scripts/simple_model_validator.py
```

### **Safe Development**
```bash
# Paper trading only
python scripts/run_live_trading.py adaptive --paper-trading

# Backtesting with no database
python scripts/run_backtest.py adaptive --days 30 --no-db

# Cache validation
python scripts/cache_manager.py --check
```

---

**Remember**: This is real money. When in doubt, err on the side of caution. 🛡️