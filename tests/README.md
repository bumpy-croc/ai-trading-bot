# 🧪 AI Trading Bot Test Suite

This directory contains the comprehensive test suite for the AI Trading Bot. The tests ensure reliability, safety, and correctness of all trading components.

---


## 🏁 Quick Reference: Test Runner Options & Markers

### **Test Runner Options**
- `smoke` — Run smoke tests (fastest)
- `unit` — Run all unit tests
- `integration` — Run integration tests
- `critical` — Run critical (live trading + risk) tests
- `all` — Run the full test suite
- `validate` — Validate test environment
- `--file <file>` or `-f <file>` — Run a specific test file
- `--markers <expr>` or `-m <expr>` — Run tests with specific markers (e.g., `-m "strategy and not slow"`)
- `--coverage` or `-c` — Generate coverage report
- `--quiet` or `-q` — Quiet output
- `--verbose` or `-v` — Verbose output
- `--no-deps-check` — Skip dependency checking
- `--interactive` or `-i` — Force interactive mode
- `--help` or `-h` — Show help

### **Pytest Markers**
- `@pytest.mark.unit` — Unit tests
- `@pytest.mark.integration` — Integration tests
- `@pytest.mark.live_trading` — Live trading tests
- `@pytest.mark.risk_management` — Risk management tests
- `@pytest.mark.strategy` — Strategy tests
- `@pytest.mark.slow` — Slow-running tests
- `@pytest.mark.network` — Tests requiring network access

### **Marker Usage Examples**
```bash
python tests/run_tests.py -m "unit"
python tests/run_tests.py -m "not integration"
python tests/run_tests.py -m "live_trading or risk_management"
python tests/run_tests.py -m "strategy and not slow"
```

---

## 🚀 Quick Start

```bash
# Run basic smoke test
python tests/run_tests.py smoke

# Run all unit tests
python tests/run_tests.py unit

# Run specific test file
python tests/run_tests.py --file test_strategies.py

# Get help
python tests/run_tests.py --help
```

## 📁 Test Structure

```
tests/
├── README.md                    # This file
├── run_tests.py                 # Enhanced test runner
├── conftest.py                  # Pytest configuration and fixtures
├── test_config_system.py        # Configuration system tests
├── test_data_providers.py       # Data provider tests
├── test_integration.py          # End-to-end integration tests
├── test_live_trading.py         # Live trading engine tests
├── test_risk_management.py      # Risk management tests
├── test_strategies.py           # Trading strategy tests
└── test_strategy_manager.py     # Strategy management tests
```

## 🎯 Test Categories

### **Smoke Tests**
Quick validation that core components work:
```bash
python tests/run_tests.py smoke
```

### **Unit Tests**
Test individual components in isolation:
```bash
python tests/run_tests.py unit
```

### **Integration Tests**
Test component interactions and workflows:
```bash
python tests/run_tests.py integration
```

### **Critical Tests**
Focus on live trading and risk management:
```bash
python tests/run_tests.py critical
```

### **All Tests**
Run the full test suite:
```bash
python tests/run_tests.py all
```

### **Validate Environment**
Check test setup:
```bash
python tests/run_tests.py validate
```

## 🛠️ Test Runner Commands

(See Quick Reference above for all options)

## 🏷️ Test Markers

(See Quick Reference above for all markers and usage)

## 📊 Coverage Reports

Coverage reports are generated in multiple formats:

```bash
# Generate coverage report
python tests/run_tests.py --coverage

# Reports generated:
# - Terminal output (immediate)
# - htmlcov/index.html (detailed HTML report)
```

View the HTML report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 🔧 Test Configuration

### **pytest.ini**
The project uses `pytest.ini` for test configuration:
- Test discovery patterns
- Marker definitions
- Output formatting
- Coverage settings

### **conftest.py**
Contains shared fixtures and test utilities:
- Mock data providers
- Test databases
- Common test data
- Setup/teardown helpers

## 🚨 Safety Features

### **Paper Trading Mode**
All live trading tests run in paper trading mode by default:
- No real money at risk
- Simulated order execution
- Safe testing environment

### **Isolated Test Environment**
Tests use:
- Separate test database
- Mock API responses
- Isolated configuration
- Temporary files

## 📈 Test Types by Component

### **Data Providers** (`test_data_providers.py`)
- API connectivity
- Data validation
- Caching behavior
- Error handling
- Rate limiting

### **Strategies** (`test_strategies.py`)
- Indicator calculations
- Entry/exit conditions
- Position sizing
- Risk parameters
- Market conditions

### **Risk Management** (`test_risk_management.py`)
- Position size limits
- Stop loss calculations
- Drawdown monitoring
- Risk scenarios
- Edge cases

### **Live Trading** (`test_live_trading.py`)
- Engine initialization
- Position management
- Order execution
- Error recovery
- Performance tracking

### **Integration** (`test_integration.py`)
- End-to-end workflows
- Component interactions
- Real-time scenarios
- Production readiness

### **Configuration** (`test_config_system.py`)
- Environment variables
- AWS Secrets Manager
- Configuration providers
- Fallback mechanisms

### **Strategy Management** (`test_strategy_manager.py`)
- Strategy lifecycle
- Switching logic
- Performance aggregation
- Error handling

## 🐛 Debugging Failed Tests

### **Verbose Output**
```bash
python tests/run_tests.py --file test_strategies.py --verbose
```

### **Run Single Test**


### **Debug Mode**
```bash
python -m pytest tests/test_strategies.py --pdb
```

### **Show Print Statements**
```bash
python -m pytest tests/test_strategies.py -s
```

## 📝 Writing New Tests

### **Test File Naming**
- `test_*.py` - Test files
- `test_*()` - Test functions
- `Test*` - Test classes

### **Common Patterns**
```python
import pytest
from unittest.mock import Mock, patch

class TestYourComponent:
    def test_basic_functionality(self):
        # Arrange
        component = YourComponent()
        
        # Act
        result = component.do_something()
        
        # Assert
        assert result is not None
    
    @pytest.mark.unit
    def test_with_mock_data(self, mock_data_provider):
        # Use fixtures from conftest.py
        pass
    
    @pytest.mark.slow
    def test_expensive_operation(self):
        # Mark slow tests
        pass
```

### **Best Practices**
1. **Use descriptive test names** - `test_position_size_calculation_with_high_volatility`
2. **Test edge cases** - Empty data, invalid inputs, network failures
3. **Use appropriate markers** - Mark tests by category and speed
4. **Mock external dependencies** - APIs, databases, file systems
5. **Test error conditions** - Exception handling, recovery mechanisms

## 🔄 Continuous Integration

### **CI/CD Commands**
```bash
python tests/run_tests.py smoke --no-deps-check
python tests/run_tests.py all --coverage
python tests/run_tests.py critical --quiet
```

### **GitHub Actions Example**
```yaml
- name: Run Tests
  run: |
    python tests/run_tests.py smoke --no-deps-check
    python tests/run_tests.py unit --coverage
```

## 📞 Troubleshooting

### **Common Issues**

**Missing Dependencies:**
```bash
pip install -r requirements.txt
```

**Import Errors:**
```bash
# Run from project root
cd /path/to/ai-trading-bot
python tests/run_tests.py smoke
```

**Permission Errors:**
```bash
# Check file permissions
chmod +x tests/run_tests.py
```

### **Environment Issues**
```bash
# Validate test environment
python tests/run_tests.py validate

# Check Python path
python -c "import sys; print(sys.path)"
```

## 📚 Additional Resources

- **pytest Documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **Trading Bot Documentation**: `../docs/`
- **Strategy Development**: `../strategies/README.md`

---

**Remember**: Tests are your safety net. Run them frequently, especially before deploying to production! 🛡️ 