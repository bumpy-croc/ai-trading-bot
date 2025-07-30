---
description: Trading Bot Commands & Workflows Reference
globs: 
alwaysApply: false
---

# 🔧 Trading Bot Commands

## Backtesting Commands

### Quick Development
```bash
# Quick backtest (development - no database logging)
python scripts/run_backtest.py ml_basic --days 30 --no-db

# Production backtest (with database logging)
python scripts/run_backtest.py ml_with_sentiment --days 365

# Custom parameters
python scripts/run_backtest.py ml_basic --symbol ETHUSDT --days 100 --initial-balance 50000

# With sentiment analysis
python scripts/run_backtest.py ml_with_sentiment --use-sentiment --days 90
```

### Advanced Backtesting
```bash
# Multi-strategy comparison
python scripts/run_backtest.py ml_basic,ml_adaptive --days 100 --compare

# Custom risk parameters
python scripts/run_backtest.py ml_basic --risk-per-trade 0.01 --max-drawdown 0.15

# Export results
python scripts/run_backtest.py ml_basic --days 30 --export-results --format csv
```

---

## Live Trading Commands

### Paper Trading (Safe)
```bash
# Paper trading (no real money)
python scripts/run_live_trading.py ml_basic --paper-trading

# Paper trading with custom settings
python scripts/run_live_trading.py ml_basic --balance 5000 --max-position 0.05

# Paper trading with specific strategy config
python scripts/run_live_trading.py ml_with_sentiment --paper-trading --config custom_config.json
```

### Live Trading (Real Money)
```bash
# Live trading (requires explicit confirmation)
python scripts/run_live_trading.py ml_with_sentiment --live-trading --i-understand-the-risks

# Live trading with custom balance
python scripts/run_live_trading.py ml_basic --live-trading --balance 1000 --i-understand-the-risks

# Emergency stop
python scripts/run_live_trading.py --stop
```

### Monitoring
```bash
# Start monitoring dashboard
python scripts/start_dashboard.py

# Dashboard with custom settings
python scripts/start_dashboard.py --host 0.0.0.0 --port 8080

# Access at: http://localhost:8080
```

---

## Model Training Commands

### Training
```bash
# Train new models
python scripts/train_model.py BTCUSDT --force-sentiment

# Safe training (staging environment)
python scripts/safe_model_trainer.py

# Train with custom parameters
python scripts/train_model.py BTCUSDT --epochs 100 --batch-size 32 --validation-split 0.2
```

### Validation
```bash
# Validate models
python scripts/simple_model_validator.py

# Model performance analysis
python scripts/analyze_model_performance.py

# Compare model versions
python scripts/compare_models.py --model1 v1 --model2 v2
```

---

## Testing Commands

### Test Categories
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

### Specific Test Files
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

### Test with Coverage
```bash
# Generate coverage report
python tests/run_tests.py --coverage
python tests/run_tests.py -c

# Coverage for specific files
python tests/run_tests.py --file test_strategies.py --coverage
```

### Advanced Testing
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

## Monitoring & Health Commands

### System Health
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

### Cache Management
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

## Database Commands

### Database Setup
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

### Database Migration
```bash
# Run database migrations
python scripts/migrate_database.py migrate

# Check migration status
python scripts/migrate_database.py current

# Rollback migration
python scripts/migrate_database.py downgrade
```

---

## Data Management Commands

### Data Providers
```bash
# Test data providers
python scripts/test_data_providers.py

# Update cached data
python scripts/update_cached_data.py

# Validate data quality
python scripts/validate_data_quality.py
```

### Sentiment Data
```bash
# Fetch sentiment data
python scripts/fetch_sentiment_data.py

# Update sentiment cache
python scripts/update_sentiment_cache.py

# Validate sentiment data
python scripts/validate_sentiment_data.py
```

---

## Performance Analysis Commands

### Performance Metrics
```bash
# Generate performance report
python scripts/generate_performance_report.py

# Analyze strategy performance
python scripts/analyze_strategy_performance.py --strategy ml_basic

# Compare strategies
python scripts/compare_strategies.py --strategies ml_basic,ml_adaptive
```

### Risk Analysis
```bash
# Risk analysis report
python scripts/risk_analysis.py

# Drawdown analysis
python scripts/drawdown_analysis.py

# Position sizing analysis
python scripts/position_sizing_analysis.py
```

---

## Configuration Commands

### Config Management
```bash
# Validate configuration
python scripts/validate_config.py

# Update configuration
python scripts/update_config.py

# Backup configuration
python scripts/backup_config.py
```

### Environment Setup
```bash
# Setup environment
python scripts/setup_environment.py

# Install dependencies
python scripts/install_dependencies.py

# Update dependencies
python scripts/update_dependencies.py
```

---

## Debugging Commands

### Debug Mode
```bash
# Run with debug logging
python scripts/run_live_trading.py ml_adaptive --debug

# Debug specific component
python scripts/debug_component.py --component strategy

# Memory usage analysis
python scripts/memory_analysis.py
```

### Log Analysis
```bash
# Analyze logs
python scripts/analyze_logs.py

# Log cleanup
python scripts/cleanup_logs.py

# Log rotation
python scripts/rotate_logs.py
```

---

**Remember**: Always run tests before deploying changes, and use paper trading for validation! 🛡️

---

**For detailed implementation guides, use:**
- `fetch_rules(["architecture"])` - Complete system architecture
- `fetch_rules(["project-structure"])` - Directory structure & organization
- `fetch_rules(["strategies"])` - Strategy development details
- `fetch_rules(["ml-models"])` - ML model training & integration