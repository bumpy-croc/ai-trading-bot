---
description: Trading Bot Commands & Workflows Reference
globs: 
alwaysApply: false
---

# 🔧 Trading Bot Commands & Workflows

## 🚀 Quick Start Commands

### **Backtesting**
```bash
# Quick backtest (development - no database logging)
python scripts/run_backtest.py adaptive --days 30 --no-db

# Production backtest (with database logging)
python scripts/run_backtest.py ml_with_sentiment --days 365

# Custom parameters
python scripts/run_backtest.py enhanced --symbol ETHUSDT --days 100 --initial-balance 50000

# With sentiment analysis
python scripts/run_backtest.py ml_with_sentiment --use-sentiment --days 90
```

### **Live Trading**
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

### **Model Training**
```bash
# Train new models
python scripts/train_model.py BTCUSDT --force-sentiment

# Safe training (staging environment)
python scripts/safe_model_trainer.py

# Validate models
python scripts/simple_model_validator.py
```

---

## 🧪 Testing Commands

### **Test Categories**
```bash
# Quick smoke tests (fastest)
python tests/run_tests.py smoke

# Unit tests (individual components)
python tests/run_tests.py unit

# Integration tests (component interactions)
python tests/run_tests.py integration

# Critical tests (live trading + risk management)
python tests/run_tests.py critical

# All tests (complete test suite)
python tests/run_tests.py all

# Validate test environment
python tests/run_tests.py validate
```

### **Specific Test Files**
```bash
# Test specific components
python tests/run_tests.py --file test_strategies.py
python tests/run_tests.py --file test_data_providers.py
python tests/run_tests.py --file test_risk_management.py
python tests/run_tests.py --file test_live_trading.py
python tests/run_tests.py --file test_config_system.py
python tests/run_tests.py --file test_integration.py

# Short form
python tests/run_tests.py -f test_strategies.py
```

### **Test with Coverage**
```bash
# Generate coverage report
python tests/run_tests.py --coverage
python tests/run_tests.py -c

# Coverage for specific files
python tests/run_tests.py --file test_strategies.py --coverage
```

### **Test Output Control**
```bash
# Verbose output (default)
python tests/run_tests.py unit --verbose
python tests/run_tests.py unit -v

# Quiet output
python tests/run_tests.py unit --quiet
python tests/run_tests.py unit -q
```

### **Advanced Testing**
```bash
# Run tests with markers
python tests/run_tests.py --markers "not integration"
python tests/run_tests.py -m "live_trading or risk_management"
python tests/run_tests.py -m "strategy and not slow"

# Exclude slow tests
python tests/run_tests.py -m "not slow"

# Skip dependency checks (for CI/CD)
python tests/run_tests.py smoke --no-deps-check
```

---

## 📊 Monitoring & Health Checks

### **System Health**
```bash
# Quick health check
python scripts/health_check.py

# Detailed system check
python scripts/health_check.py --detailed

# Check current positions
python scripts/health_check.py --positions

# Check API connectivity
python scripts/test_database.py
```

### **Monitoring Dashboard**
```bash
# Start monitoring dashboard
python scripts/start_dashboard.py

# Access at: http://localhost:8080

# With custom settings
python scripts/start_dashboard.py --host 0.0.0.0 --port 8080
```

### **Cache Management**
```bash
# Check cache status
python scripts/cache_manager.py --check

# Cache info
python scripts/cache_manager.py --status

# Clear cache
python scripts/cache_manager.py --clear

# Cache validation
python scripts/cache_manager.py --validate
```

---

## 🗄️ Database Commands

### **Database Setup**
```bash
# Test database connection
python scripts/test_database.py

# Verify database schema
python scripts/verify_database_connection.py

# Database backup
python scripts/backup_database.py

# Emergency database backup
python scripts/backup_database.py --emergency
```

### **Database Migration**
```bash
# Run database migrations
python scripts/migrate_database.py migrate

# Check migration status
python scripts/migrate_database.py current

# Rollback migration
python scripts/migrate_database.py downgrade
```

---

## 🔄 Data Management

### **Data Download**
```bash
# Download Binance data
python scripts/download_binance_data.py BTCUSDT --days 365

# Download sentiment data
python scripts/download_binance_data.py --sentiment --days 90

# Update cached data
python scripts/cache_manager.py --update
```

### **Data Analysis**
```bash
# Analyze BTC data
python scripts/analyze_btc_data.py

# Analyze with specific strategy
python scripts/analyze_btc_data.py --strategy adaptive

# Generate data report
python scripts/analyze_btc_data.py --report
```

---

## 🧠 ML Model Commands

### **Model Training**
```bash
# Basic model training
python scripts/train_model.py BTCUSDT

# Train with sentiment
python scripts/train_model.py BTCUSDT --force-sentiment

# Custom training parameters
python scripts/train_model.py BTCUSDT --epochs 100 --batch-size 32

# Safe training (staging)
python scripts/safe_model_trainer.py
```

### **Model Validation**
```bash
# Validate model performance
python scripts/simple_model_validator.py

# Check model files
ls -la ml/btcusdt_*

# View model metadata
cat ml/btcusdt_sentiment_metadata.json

# Model performance analysis
python scripts/simple_model_validator.py --detailed
```

### **Model Management**
```bash
# List available models
ls -la ml/

# Check model compatibility
python scripts/simple_model_validator.py --compatibility

# Backup models
cp ml/btcusdt_sentiment.* ml/backup/

# Restore models
cp ml/backup/btcusdt_sentiment.* ml/
```

---

## 🛡️ Safety & Emergency Commands

### **Emergency Procedures**
```bash
# Stop live trading immediately
python scripts/run_live_trading.py --stop

# Check current positions
python scripts/health_check.py --positions

# Emergency database backup
python scripts/backup_database.py --emergency

# Force stop all processes
pkill -f "run_live_trading"
```

### **Risk Management**
```bash
# Check risk metrics
python scripts/health_check.py --risk

# Validate risk parameters
python tests/run_tests.py --file test_risk_management.py

# Check position sizes
python scripts/health_check.py --positions --detailed
```

---

## 🔧 Development Commands

### **Environment Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Setup local development
python scripts/setup_local_development.py

# Verify environment
python scripts/verify_database_connection.py
```

### **Code Quality**
```bash
# Run linting
python -m flake8 src/

# Run type checking
python -m mypy src/

# Run security checks
python scripts/security_demo.sh
```

### **Debugging**
```bash
# Verbose backtesting
python scripts/run_backtest.py adaptive --days 7 --no-db --verbose

# Debug live trading
python scripts/run_live_trading.py adaptive --paper-trading --debug

# Check logs
tail -f logs/trading.log

# Monitor system resources
python scripts/health_check.py --detailed
```

---

## 📈 Performance Analysis

### **Backtest Analysis**
```bash
# Run performance analysis
python scripts/analyze_btc_data.py --performance

# Compare strategies
python scripts/run_backtest.py adaptive --days 90 --no-db
python scripts/run_backtest.py enhanced --days 90 --no-db
python scripts/run_backtest.py ml_with_sentiment --days 90 --no-db

# Generate performance report
python scripts/analyze_btc_data.py --report --output performance_report.html
```

### **Live Performance**
```bash
# Monitor live performance
python scripts/start_dashboard.py

# Check performance metrics
python scripts/health_check.py --performance

# Export performance data
python scripts/analyze_btc_data.py --export --format csv
```

---

## 🚀 Deployment Commands

### **Railway Deployment**
```bash
# Deploy to Railway
railway up

# Check deployment status
railway status

# View logs
railway logs

# Connect to Railway database
railway connect
```

### **Local Deployment**
```bash
# Start with Docker
docker-compose up -d

# Check container status
docker-compose ps

# View container logs
docker-compose logs trading-bot

# Stop containers
docker-compose down
```

---

## 🔍 Troubleshooting Commands

### **Common Issues**
```bash
# Check API connectivity
python scripts/test_database.py

# Validate configuration
python scripts/test_config_system.py

# Check file permissions
ls -la scripts/run_live_trading.py

# Verify Python path
python -c "import sys; print(sys.path)"
```

### **Debug Mode**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python scripts/run_live_trading.py adaptive --paper-trading

# Show print statements in tests
python -m pytest tests/test_strategies.py -s

# Debug specific test
python -m pytest tests/test_strategies.py::TestAdaptiveStrategy::test_entry_conditions --pdb
```

---

## 📋 Command Reference by Category

### **Quick Development Workflow**
```bash
# 1. Test changes
python tests/run_tests.py smoke

# 2. Backtest strategy
python scripts/run_backtest.py my_strategy --days 30 --no-db

# 3. Paper trading validation
python scripts/run_live_trading.py my_strategy --paper-trading

# 4. Monitor performance
python scripts/start_dashboard.py
```

### **Production Deployment Workflow**
```bash
# 1. Run all tests
python tests/run_tests.py all --coverage

# 2. Production backtest
python scripts/run_backtest.py strategy --days 365

# 3. Validate models
python scripts/simple_model_validator.py

# 4. Deploy to production
python scripts/run_live_trading.py strategy --live-trading --i-understand-the-risks
```

### **Emergency Response Workflow**
```bash
# 1. Stop trading
python scripts/run_live_trading.py --stop

# 2. Check positions
python scripts/health_check.py --positions

# 3. Backup data
python scripts/backup_database.py --emergency

# 4. Investigate issue
python scripts/health_check.py --detailed
```

---

## 🎯 Natural Language Commands

When you say these phrases, I'll run the corresponding commands:

### **Testing**
- "run smoke tests" → `python tests/run_tests.py smoke`
- "run unit tests" → `python tests/run_tests.py unit`
- "run critical tests" → `python tests/run_tests.py critical`
- "run all tests" → `python tests/run_tests.py all`
- "test strategies" → `python tests/run_tests.py --file test_strategies.py`
- "test with coverage" → `python tests/run_tests.py --coverage`

### **Backtesting**
- "run backtest" → `python scripts/run_backtest.py adaptive --days 30 --no-db`
- "run backtest for [strategy]" → `python scripts/run_backtest.py [strategy] --days 30 --no-db`
- "run production backtest" → `python scripts/run_backtest.py adaptive --days 30`

### **Live Trading**
- "start paper trading" → `python scripts/run_live_trading.py adaptive --paper-trading`
- "start live trading" → `python scripts/run_live_trading.py adaptive --live-trading --i-understand-the-risks`
- "start dashboard" → `python scripts/start_dashboard.py`

### **Health & Monitoring**
- "check health" → `python scripts/health_check.py`
- "check positions" → `python scripts/health_check.py --positions`
- "check cache" → `python scripts/cache_manager.py --check`

---

**Remember**: Always run tests before deploying changes, and use paper trading for validation! 🛡️

---

**For detailed implementation guides, use:**
- `fetch_rules(["architecture"])` - Complete system architecture
- `fetch_rules(["project-structure"])` - Directory structure & organization
- `fetch_rules(["strategies"])` - Strategy development details
- `fetch_rules(["ml-models"])` - ML model training & integration