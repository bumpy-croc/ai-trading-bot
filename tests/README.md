# Test Suite

> **Last Updated**: 2025-10-21  
> **Related Documentation**: See [Development workflow](../docs/development.md#tests-and-diagnostics)

Comprehensive tests for reliability and correctness across components, including the new component-based strategy system.

## Quick Start

### Component System Tests
```bash
# Test individual components
pytest tests/unit/strategies/components/ -v

# Test complete trading workflows
pytest tests/integration/test_component_trading_workflows.py -v

# Run performance regression tests
pytest tests/performance/test_component_performance_regression.py -v -m performance

# Full component test suite
pytest tests/unit/strategies/components/ tests/integration/test_component_trading_workflows.py -v
```

### Legacy Compatibility Tests (archived)
These commands exercise the archived `BaseStrategy` pipeline to ensure historical parity while the
component-based runtime remains the production path.
```bash
python tests/run_tests.py smoke
python tests/run_tests.py unit
python tests/run_tests.py integration
python tests/run_tests.py all --coverage
```

### Migration and Compatibility Tests
```bash
# Test migration compatibility
pytest tests/unit/strategies/test_component_migration.py -v

# Test strategy conversion utilities
pytest tests/unit/strategies/migration/ -v
```

## Test Categories

### Component Tests (`tests/unit/strategies/components/`)
- **Signal Generators**: `test_signal_generator.py`, `test_ml_signal_generator.py`, `test_technical_signal_generator.py`
- **Risk Managers**: `test_risk_manager.py`
- **Position Sizers**: `test_position_sizer.py`
- **Complete Strategies**: `test_strategy.py`
- **Strategy Management**: `test_strategy_manager.py`, `test_strategy_registry.py`

### Integration Tests (`tests/integration/`)
- **Trading Workflows**: `test_component_trading_workflows.py`
- **Error Handling**: `test_error_handling_workflows.py`
- **Performance Monitoring**: `test_integration.py`

### Performance Tests (`tests/performance/`)
- **Regression Testing**: `test_component_performance_regression.py`
- **Baseline Management**: `performance_baseline_manager.py`
- **Automated Monitoring**: `automated_performance_monitor.py`

### Migration Tests (`tests/unit/strategies/`)
- **Compatibility**: `test_component_migration.py`
- **Conversion Utilities**: `migration/test_strategy_converter.py`

## Test Markers

### Component System Markers
- `unit` - Fast unit tests for individual components
- `integration` - Integration tests for component interactions
- `performance` - Performance regression tests
- `migration` - Migration and compatibility tests
- `regression` - Regression detection tests

### Legacy Compatibility Markers (archived)
- `live_trading` - Live trading system tests
- `risk_management` - Risk management tests
- `strategy` - Strategy-specific tests
- `slow` - Long-running tests
- `network` - Tests requiring network access

## Performance Monitoring

### Automated Performance Monitoring
```bash
# Run full performance monitoring cycle
python tests/performance/automated_performance_monitor.py --full-cycle

# Generate performance dashboard
python tests/performance/automated_performance_monitor.py --dashboard-only

# Check for performance regressions
python tests/performance/automated_performance_monitor.py --analysis-only
```

### Performance Dashboard
Access the performance dashboard at `tests/performance/results/performance_dashboard.html` for:
- Real-time performance metrics
- Trend analysis and regression detection
- Component-level performance breakdown
- Historical performance data

### Baseline Management
```bash
# View current performance baselines
python tests/performance/performance_baseline_manager.py --report

# Export performance data
python tests/performance/performance_baseline_manager.py --export performance_data.csv

# Clean old performance data
python tests/performance/performance_baseline_manager.py --cleanup 30
```

## Running Tests

### Component-Focused Commands
```bash
# All component tests
pytest tests/unit/strategies/components/ -v

# Specific component type
pytest tests/unit/strategies/components/test_signal_generator.py -v

# Integration workflows
pytest tests/integration/test_component_trading_workflows.py -v

# Performance tests
pytest tests/performance/ -v -m performance

# Migration tests
pytest tests/unit/strategies/ -v -m migration
```

### Legacy Commands (compatibility harness)
```bash
python tests/run_tests.py --file tests/test_strategies.py
python tests/run_tests.py -m "strategy and not slow"
```

### Parallel Execution
```bash
# Run component tests in parallel
pytest tests/unit/strategies/components/ -n auto

# Parallel integration tests
pytest tests/integration/ -n 4
```

## Coverage

### Component System Coverage
```bash
# Component coverage
pytest tests/unit/strategies/components/ --cov=src/strategies/components --cov-report=html

# Integration coverage
pytest tests/integration/ --cov=src/strategies/components --cov-report=html

# Combined coverage
pytest tests/unit/strategies/components/ tests/integration/ --cov=src/strategies/components --cov-report=html
```

### Legacy Coverage (compatibility)
```bash
python tests/run_tests.py --coverage
# HTML at htmlcov/index.html
```

## Documentation

### Comprehensive Guides
- **Component Testing Guide**: `tests/COMPONENT_TESTING_GUIDE.md`
- **Test Troubleshooting Guide**: `tests/TEST_TROUBLESHOOTING_GUIDE.md`
- **Migration Guide**: `tests/unit/strategies/TEST_MIGRATION_GUIDE.md`

### Quick References
- **Performance Dashboard**: `tests/performance/results/performance_dashboard.html`
- **Test Results**: `tests/performance/results/`
- **Baseline Data**: `tests/performance/component_baselines.json`

## Troubleshooting

### Common Issues
1. **Component Interface Errors**: Check `tests/TEST_TROUBLESHOOTING_GUIDE.md#component-interface-errors`
2. **Performance Regressions**: Run `python tests/performance/automated_performance_monitor.py --analysis-only`
3. **Migration Failures**: Check `tests/unit/strategies/TEST_MIGRATION_GUIDE.md`
4. **Integration Problems**: See `tests/TEST_TROUBLESHOOTING_GUIDE.md#integration-problems`

### Debug Commands
```bash
# Verbose output with full tracebacks
pytest tests/unit/strategies/components/test_strategy.py -v -s --tb=long

# Drop into debugger on failure
pytest tests/unit/strategies/components/test_signal_generator.py --pdb

# Show performance timing
pytest tests/performance/ --durations=10
```

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run Component Tests
  run: |
    pytest tests/unit/strategies/components/ -v --cov=src/strategies/components
    pytest tests/integration/test_component_trading_workflows.py -v
    python tests/performance/automated_performance_monitor.py --full-cycle
```

### Performance Monitoring in CI
The automated performance monitor can be integrated into CI/CD pipelines to:
- Detect performance regressions automatically
- Generate performance reports
- Update performance baselines
- Alert on critical performance issues

## Notes

- **Component System**: New architecture with pluggable components for signals, risk, and position sizing
- **Legacy Compatibility**: Migration tests ensure backward compatibility during transition
- **Performance Focus**: Comprehensive performance monitoring and regression detection
- **PostgreSQL Testcontainers**: Used for database integration tests
- **Parallel Execution**: Tests designed for parallel execution where possible
- **See**: `pytest.ini` and `tests/run_tests.py` for additional configuration options
