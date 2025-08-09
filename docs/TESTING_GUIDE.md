# Trading Bot Testing Guide

## Overview

This document provides comprehensive guidance on testing the crypto trading bot system. The testing suite is designed to ensure reliability, especially for mission-critical live trading components.

## Testing Philosophy

### Risk-Based Testing Approach

Our testing strategy prioritizes components based on risk to capital and system stability:

1. **CRITICAL (P0)**: Live trading engine, risk management
2. **HIGH (P1)**: Strategy logic, data providers, hot-swapping
3. **MEDIUM (P2)**: ML models, backtesting, indicators
4. **LOW (P3)**: Utilities, documentation, examples

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing  
- **Live Trading Tests**: Real-money simulation tests
- **Performance Tests**: Load and timing tests
- **Risk Tests**: Capital protection validation

## Test Structure

### Test Organization

```
tests/
├── __init__.py                 # Test package
├── conftest.py                 # Shared fixtures and configuration
├── test_live_trading.py        # Live trading engine tests (CRITICAL)
├── test_risk_management.py     # Risk management tests (CRITICAL)
├── test_strategies.py          # Strategy testing
├── test_data_providers.py      # Data provider tests
├── test_strategy_manager.py    # Hot-swapping tests
├── test_account_sync.py        # Account synchronization tests
└── test_integration.py         # End-to-end tests
```

### Test Infrastructure Best Practices

**⚠️ CRITICAL: Preserve Existing Test Infrastructure**

When working with existing test infrastructure:

• **Never replace** comprehensive test setup files (like `conftest.py`)
• **Always add** new fixtures to existing infrastructure
• **Understand dependencies** before making changes
• **Test the impact** of changes on existing tests
• **Preserve database setup** (PostgreSQL containers, connection handling)
• **Maintain existing fixtures** (OHLCV data, strategies, risk parameters)
• **Follow established patterns** for consistency
• **Document new fixtures** with clear docstrings

### Test Markers

Tests are categorized using pytest markers:

```python
@pytest.mark.live_trading      # Tests live trading components
@pytest.mark.risk_management   # Tests risk management
@pytest.mark.strategy          # Tests strategy logic
@pytest.mark.data_provider     # Tests data providers
@pytest.mark.integration       # Integration tests (slower)
```

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock pytest-asyncio

# Install project dependencies
pip install -r requirements.txt  # or requirements-server.txt for CI
  ```

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai-trading-bot --cov-report=html

# Run specific test categories
pytest -m live_trading          # Live trading tests only
pytest -m risk_management       # Risk management tests only
pytest -m "not integration"     # Skip slower integration tests

# Run specific test files
pytest tests/test_live_trading.py
pytest tests/test_risk_management.py
pytest tests/test_account_sync.py

# Run account sync tests with coverage
pytest tests/test_account_sync.py --cov=src.live.account_sync --cov-report=term

# Run with verbose output
pytest -v

# Run with detailed output for failures
pytest -vvv --tb=short
```

### Development Testing

```bash
# Watch mode for development (requires pytest-watch)
ptw

# Run tests on file changes
pytest --looponfail

# Run only failed tests from last run
pytest --lf

# Run tests in parallel (requires pytest-xdist)
pytest -n 4
```

## Critical Test Categories

### 1. Live Trading Engine Tests

**Purpose**: Ensure the live trading engine handles real money safely.

**Key Test Areas**:
- Position opening/closing logic
- Risk limit enforcement
- Error handling and recovery
- Thread safety
- Graceful shutdown
- Hot-swapping capability

**Critical Tests**:
```python
# Test position management
test_position_opening_paper_trading()
test_position_closing()
test_stop_loss_trigger()
test_take_profit_trigger()

# Test risk integration
test_risk_manager_integration()
test_maximum_position_limits()
test_drawdown_monitoring()

# Test error handling
test_error_handling_in_trading_loop()
test_graceful_shutdown()
test_api_rate_limit_handling()

# Test thread safety
test_concurrent_position_updates()
test_stop_event_handling()
```

### 2. Risk Management Tests

**Purpose**: Validate capital protection mechanisms.

**Key Test Areas**:
- Position sizing calculations
- Stop loss placement
- Drawdown monitoring
- Daily risk limits
- Portfolio exposure limits

**Critical Tests**:
```python
# Test position sizing
test_position_size_calculation_normal_regime()
test_position_size_calculation_volatile_regime()
test_maximum_position_size_enforcement()

# Test risk limits
test_drawdown_monitoring()
test_daily_risk_limit_enforcement()
test_stop_loss_calculation()

# Test edge cases
test_zero_balance()
test_zero_atr()
test_very_large_values()
```

### 3. Strategy Tests

**Purpose**: Ensure strategies generate reliable signals.

**Key Test Areas**:
- Indicator calculations
- Entry/exit signal logic
- Position sizing logic
- Market condition adaptation
- Parameter validation

**Critical Tests**:
```python
# Test core functionality
test_indicator_calculation()
test_entry_conditions_with_valid_data()
test_exit_conditions_with_valid_data()
test_position_size_calculation()

# Test edge cases
test_insufficient_data_handling()
test_entry_conditions_out_of_bounds()
test_missing_indicator_data()

# Test market conditions
test_bull_market_conditions()
test_bear_market_conditions()
test_volatile_market_conditions()
```

### 4. Account Synchronization Tests

**Purpose**: Ensure data integrity between exchange and database.

**Key Test Areas**:
- Balance synchronization
- Position synchronization
- Order synchronization
- Trade recovery
- Emergency sync procedures
- Error handling and recovery

**Critical Tests**:
```python
# Test core synchronization
test_sync_account_data_success()
test_sync_balances_with_discrepancy()
test_sync_positions_new_position()
test_sync_orders_status_update()

# Test recovery mechanisms
test_recover_missing_trades_with_missing()
test_emergency_sync_success()
test_emergency_sync_failure()

# Test error handling
test_sync_balances_exception()
test_sync_positions_exception()
test_sync_orders_exception()
test_recover_missing_trades_exception()

# Test integration
test_full_sync_integration()
test_emergency_sync_integration()
```

### 4. Data Provider Tests

**Purpose**: Ensure data integrity and availability.

**Key Test Areas**:
- API connectivity
- Data validation
- Error handling
- Caching mechanisms
- Rate limiting

**Critical Tests**:
```python
# Test data integrity
test_data_format_consistency()
test_data_type_consistency()
test_timestamp_consistency()

# Test error handling
test_api_error_handling()
test_rate_limit_handling()
test_network_timeout_handling()

# Test caching
test_cached_provider_first_call()
test_cached_provider_subsequent_calls()
test_caching_performance()
```

## Test Environment Setup

### Mock Data and Fixtures

The test suite includes comprehensive fixtures for:

- **Sample OHLCV Data**: Realistic market data for testing
- **Mock Strategies**: Controllable strategy behavior
- **Mock Data Providers**: Predictable data sources
- **Risk Parameters**: Standard risk configurations
- **Market Conditions**: Various market scenarios

### Test Configuration

```python
# conftest.py provides shared configuration
@pytest.fixture
def sample_ohlcv_data():
    """Generate realistic OHLCV data for testing"""
    # Returns 100 periods of realistic price data

@pytest.fixture
def mock_strategy():
    """Create a controllable mock strategy"""
    # Returns strategy with predictable behavior

@pytest.fixture
def risk_parameters():
    """Standard risk parameters for testing"""
    # Returns conservative risk settings
```

## Safety Measures

### Paper Trading Mode

All live trading tests run in paper trading mode by default:

```python
# Always test with paper trading first
engine = LiveTradingEngine(
    strategy=strategy,
    data_provider=provider,
    enable_live_trading=False  # CRITICAL: Never True in tests
)
```

### Risk Validation

Tests validate risk management is working:

```python
def test_maximum_position_limits():
    """Ensure position limits are enforced"""
    # Test that positions never exceed configured limits
    
def test_stop_loss_enforcement():
    """Ensure stop losses are triggered"""
    # Test that losses are limited as configured
```

### Data Validation

Tests ensure data integrity:

```python
def test_data_consistency():
    """Validate OHLCV data is logical"""
    assert (df['high'] >= df['low']).all()
    assert (df['high'] >= df['open']).all()
    assert (df['high'] >= df['close']).all()
    assert (df['volume'] >= 0).all()
```

## Production Testing Guidelines

### Pre-Deployment Checklist

Before deploying to live trading:

1. **All Tests Pass**: `pytest --cov=ai-trading-bot`
2. **Performance Tests**: `pytest tests/test_performance.py`
3. **Integration Tests**: `pytest -m integration`
4. **Manual Risk Testing**: Verify risk limits manually
5. **Paper Trading**: Run strategies in paper mode first

### Continuous Testing

Set up automated testing:

```bash
# GitHub Actions / CI example
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
              run: pytest --cov=ai-trading-bot
    - name: Check critical tests
      run: pytest -m "live_trading or risk_management" --tb=short
```

### Live Trading Validation

For live trading deployment:

```python
# Validate live trading configuration
def validate_live_config():
    """Validate configuration before live trading"""
    assert config.enable_live_trading == True
    assert config.initial_balance > 0
    assert config.max_position_size <= 0.25  # Max 25%
    assert config.max_risk_per_trade <= 0.03  # Max 3%
    
    # Validate API keys are set
    assert os.getenv('BINANCE_API_KEY')
    assert os.getenv('BINANCE_API_SECRET')
```

## Test Data Management

### Historical Data

Tests use cached historical data to ensure consistency:

```python
# Use cached data for reproducible tests
@pytest.fixture
def btc_test_data():
    """Load cached BTC data for testing"""
    return pd.read_csv('tests/data/btc_test_data.csv')
```

### Sentiment Data

Mock sentiment data for testing ML strategies:

```python
# Generate realistic sentiment data
@pytest.fixture
def sentiment_data():
    """Generate test sentiment data"""
    return pd.DataFrame({
        'sentiment_score': np.random.uniform(-1, 1, 100),
        'sentiment_confidence': np.random.uniform(0, 1, 100)
    })
```

## Performance Testing

### Load Testing

Test system under load:

```python
def test_concurrent_trading():
    """Test multiple trading engines"""
    engines = []
    for i in range(10):
        engine = create_test_engine()
        engines.append(engine)
    
    # Run all engines concurrently
    # Verify no data corruption or deadlocks
```

### Memory Testing

Monitor memory usage:

```python
def test_memory_usage():
    """Test for memory leaks"""
    import psutil
    process = psutil.Process()
    
    initial_memory = process.memory_info().rss
    
    # Run trading simulation
    run_trading_simulation(hours=24)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory should not increase significantly
    assert memory_increase < 100 * 1024 * 1024  # 100MB limit
```

## Debugging Tests

### Test Debugging

When tests fail:

```bash
# Run with detailed output
pytest -vvv --tb=long test_that_failed.py::test_function

# Drop into debugger on failure
pytest --pdb test_that_failed.py

# Capture output
pytest -s test_that_failed.py  # Don't capture stdout
```

### Log Analysis

Tests include comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

def test_with_logging():
    """Test with detailed logging"""
    logger = logging.getLogger(__name__)
    logger.info("Starting test")
    # Test implementation
    logger.info("Test completed")
```

## Test Coverage

### Coverage Requirements

Minimum coverage requirements:
- **Live Trading Engine**: 95%
- **Risk Management**: 95%
- **Strategies**: 85%
- **Data Providers**: 80%
- **Overall**: 85%

### Coverage Analysis

```bash
# Generate coverage report
pytest --cov=ai-trading-bot --cov-report=html

# View coverage report
open htmlcov/index.html

# Check coverage of specific module
pytest --cov=live.trading_engine --cov-report=term-missing
```

## Contributing Tests

### Test Writing Guidelines

1. **Test One Thing**: Each test should verify one specific behavior
2. **Clear Names**: Test names should describe what is being tested
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use Fixtures**: Leverage shared fixtures for setup
5. **Mock External Dependencies**: Don't rely on external APIs

### Example Test Template

```python
def test_specific_behavior(fixture1, fixture2):
    """Test that specific behavior works correctly"""
    # Arrange
    setup_data = fixture1
    expected_result = calculate_expected()
    
    # Act
    actual_result = function_under_test(setup_data)
    
    # Assert
    assert actual_result == expected_result
    assert other_condition_is_true()
```

### Test Review Checklist

Before submitting tests:

- [ ] Test names are descriptive
- [ ] Tests are isolated and don't depend on each other
- [ ] Edge cases are covered
- [ ] Error conditions are tested
- [ ] Tests are fast (< 1 second each for unit tests)
- [ ] Mocks are used appropriately
- [ ] Test data is realistic but predictable

## Maintenance

### Updating Tests

When updating the codebase:

1. **Update Tests First**: Write tests for new functionality
2. **Maintain Compatibility**: Ensure existing tests still pass
3. **Update Documentation**: Keep test docs current
4. **Review Coverage**: Maintain coverage standards

### Test Cleanup

Regularly review and clean up:

- Remove obsolete tests
- Update outdated test data
- Refactor duplicate test code
- Improve test performance

---

## Quick Reference

### Running Key Test Suites

```bash
# Critical tests only
pytest -m "live_trading or risk_management" -v

# Before deployment
pytest --cov=ai-trading-bot --cov-min-percentage=85

# Performance tests
pytest tests/test_performance.py -v

# Strategy validation
pytest -m strategy -v

# Data provider validation
pytest -m data_provider -v
```

### Emergency Test Commands

If system issues occur:

```bash
# Test risk systems immediately
pytest tests/test_risk_management.py -v

# Test live trading safety
pytest tests/test_live_trading.py::TestRiskIntegration -v

# Validate data integrity
pytest tests/test_data_providers.py::TestDataConsistency -v
```

Remember: **When in doubt, test more.** The test suite is designed to catch issues before they affect real trading capital.