# Component System Testing Guide

This comprehensive guide covers testing procedures, methodologies, and best practices for the new component-based strategy system.

## Table of Contents

1. [Overview](#overview)
2. [Test Architecture](#test-architecture)
3. [Test Categories](#test-categories)
4. [Running Tests](#running-tests)
5. [Performance Testing](#performance-testing)
6. [Test Data Management](#test-data-management)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)
9. [Maintenance Procedures](#maintenance-procedures)

## Overview

The component system testing framework provides comprehensive coverage for:

- **Individual Components**: Signal generators, risk managers, position sizers
- **Component Integration**: How components work together in strategies
- **Performance Regression**: Ensuring performance doesn't degrade over time
- **Migration Compatibility**: Backward compatibility with legacy systems
- **Error Handling**: Graceful failure and recovery mechanisms

### Testing Philosophy

- **Test Pyramid**: Unit tests (fast) → Integration tests (medium) → End-to-end tests (slow)
- **Performance First**: Performance is a feature, not an afterthought
- **Regression Prevention**: Automated detection of performance and functional regressions
- **Real-world Scenarios**: Tests reflect actual trading conditions

## Test Architecture

### Directory Structure

```
tests/
├── unit/
│   └── strategies/
│       ├── test_component_migration.py      # Migration compatibility tests
│       └── migration/
│           └── test_strategy_converter.py   # Strategy conversion utilities
├── integration/
│   ├── test_component_trading_workflows.py # End-to-end trading workflows
│   └── test_error_handling_workflows.py    # Error handling and recovery
├── performance/
│   ├── test_component_performance_regression.py  # Performance regression tests
│   ├── performance_baseline_manager.py           # Baseline management
│   └── automated_performance_monitor.py          # Automated monitoring
└── strategies/
    └── components/
        ├── test_signal_generator.py        # Signal generator tests
        ├── test_risk_manager.py           # Risk manager tests
        ├── test_position_sizer.py         # Position sizer tests
        └── test_strategy.py               # Complete strategy tests
```

### Test Markers

Use pytest markers to categorize and run specific test groups:

```python
@pytest.mark.unit          # Fast unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.performance   # Performance tests
@pytest.mark.regression    # Regression tests
@pytest.mark.slow          # Long-running tests
@pytest.mark.migration     # Migration-related tests
```

## Test Categories

### 1. Unit Tests

**Purpose**: Test individual components in isolation

**Location**: `tests/strategies/components/`

**Examples**:
```bash
# Test signal generators
pytest tests/strategies/components/test_signal_generator.py -v

# Test risk managers
pytest tests/strategies/components/test_risk_manager.py -v

# Test position sizers
pytest tests/strategies/components/test_position_sizer.py -v
```

**Key Test Areas**:
- Component initialization and parameter validation
- Core functionality (signal generation, risk calculation, position sizing)
- Edge cases and boundary conditions
- Error handling and fallback behavior

### 2. Integration Tests

**Purpose**: Test component interactions and complete workflows

**Location**: `tests/integration/`

**Examples**:
```bash
# Test complete trading workflows
pytest tests/integration/test_component_trading_workflows.py -v

# Test error handling workflows
pytest tests/integration/test_error_handling_workflows.py -v
```

**Key Test Areas**:
- End-to-end trading decision cycles
- Component data flow and communication
- Regime transition handling
- Multi-component error recovery

### 3. Performance Tests

**Purpose**: Ensure performance meets requirements and detect regressions

**Location**: `tests/performance/`

**Examples**:
```bash
# Run performance regression tests
pytest tests/performance/test_component_performance_regression.py -v -m performance

# Run automated performance monitoring
python tests/performance/automated_performance_monitor.py --full-cycle
```

**Key Test Areas**:
- Individual component performance
- Complete decision cycle timing
- Memory usage and stability
- Batch processing performance
- Legacy system compatibility

### 4. Migration Tests

**Purpose**: Ensure compatibility during migration from legacy system

**Location**: `tests/unit/strategies/`

**Examples**:
```bash
# Test migration compatibility
pytest tests/unit/strategies/test_component_migration.py -v

# Test strategy conversion
pytest tests/unit/strategies/migration/test_strategy_converter.py -v
```

**Key Test Areas**:
- Legacy vs component system equivalence
- Parameter mapping and conversion
- Backward compatibility maintenance
- Test fixture compatibility

## Running Tests

### Quick Test Commands

```bash
# Run all component tests
pytest tests/strategies/components/ -v

# Run integration tests
pytest tests/integration/ -v -m integration

# Run performance tests
pytest tests/performance/ -v -m performance

# Run migration tests
pytest tests/unit/strategies/ -v -m migration

# Run all tests with coverage
pytest --cov=src/strategies/components --cov-report=html
```

### Test Execution Strategies

#### Development Workflow
```bash
# Fast feedback during development
pytest tests/strategies/components/test_signal_generator.py::TestSignalGenerator::test_signal_creation -v

# Test specific component
pytest tests/strategies/components/test_risk_manager.py -v

# Test with specific marker
pytest -m "unit and not slow" -v
```

#### CI/CD Pipeline
```bash
# Full test suite for CI
pytest tests/ --cov=src/strategies/components --cov-report=xml --junit-xml=test-results.xml

# Performance monitoring in CI
python tests/performance/automated_performance_monitor.py --full-cycle
```

#### Pre-deployment Validation
```bash
# Comprehensive validation
pytest tests/ -v --tb=short
python tests/performance/automated_performance_monitor.py --analysis-only
```

### Parallel Test Execution

```bash
# Run tests in parallel for speed
pytest tests/strategies/components/ -n auto

# Parallel with specific worker count
pytest tests/integration/ -n 4

# Parallel with load balancing
pytest tests/ -n auto --dist=loadgroup
```

## Performance Testing

### Performance Baselines

Performance tests use established baselines to detect regressions:

```python
# Example baseline configuration
baselines = {
    'signal_generation': {'target_ms': 5.0, 'max_ms': 10.0},
    'risk_calculation': {'target_ms': 2.0, 'max_ms': 5.0},
    'position_sizing': {'target_ms': 1.0, 'max_ms': 3.0},
    'complete_decision': {'target_ms': 15.0, 'max_ms': 30.0}
}
```

### Running Performance Tests

```bash
# Basic performance test
pytest tests/performance/test_component_performance_regression.py -v

# Performance monitoring with baseline updates
python tests/performance/automated_performance_monitor.py --full-cycle

# Generate performance dashboard
python tests/performance/automated_performance_monitor.py --dashboard-only
```

### Performance Regression Detection

The system automatically detects performance regressions:

- **Warning Threshold**: 20% slower than baseline
- **Critical Threshold**: 50% slower than baseline
- **Trend Analysis**: Tracks performance over time
- **Automated Alerts**: Notifications for significant regressions

### Performance Dashboard

Access the performance dashboard at `tests/performance/results/performance_dashboard.html` for:

- Real-time performance metrics
- Trend analysis and visualization
- Regression alerts and warnings
- Historical performance data

## Test Data Management

### Test Data Creation

```python
def create_test_data(scenario="trending_up", length=100):
    """Create test data for specific market scenarios"""
    if scenario == "trending_up":
        # Upward trending market data
        base_prices = np.linspace(50000, 55000, length)
        noise = np.random.normal(0, 200, length)
        closes = base_prices + noise
    # ... other scenarios
    
    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes,
        'volume': volumes, 'onnx_pred': predictions
    })
```

### Shared Test Fixtures

```python
@pytest.fixture
def sample_market_data():
    """Shared market data fixture"""
    return create_test_data("trending_up", 100)

@pytest.fixture
def test_strategy():
    """Shared strategy fixture"""
    return Strategy(
        name="test_strategy",
        signal_generator=MLBasicSignalGenerator(),
        risk_manager=FixedRiskManager(),
        position_sizer=ConfidenceWeightedSizer()
    )
```

### Test Data Guidelines

1. **Reproducible**: Use fixed random seeds for consistent results
2. **Realistic**: Data should reflect actual market conditions
3. **Comprehensive**: Cover various market scenarios (bull, bear, sideways, volatile)
4. **Efficient**: Minimize data size while maintaining test coverage
5. **Isolated**: Each test should use independent data

## Troubleshooting

### Common Test Failures

#### 1. Performance Regression Failures

**Symptom**: Performance tests fail with "regression detected"

**Diagnosis**:
```bash
# Check performance trends
python tests/performance/performance_baseline_manager.py --report

# Run regression analysis
python tests/performance/performance_baseline_manager.py --regression-check
```

**Solutions**:
- Review recent code changes for performance impact
- Update baselines if performance change is intentional
- Optimize slow components identified in the analysis

#### 2. Component Integration Failures

**Symptom**: Integration tests fail with component communication errors

**Diagnosis**:
```bash
# Run with verbose output
pytest tests/integration/test_component_trading_workflows.py -v -s

# Check component interfaces
pytest tests/strategies/components/ -v --tb=long
```

**Solutions**:
- Verify component interface compatibility
- Check data flow between components
- Validate component initialization parameters

#### 3. Migration Compatibility Failures

**Symptom**: Migration tests show differences between old and new systems

**Diagnosis**:
```bash
# Run migration tests with detailed output
pytest tests/unit/strategies/test_component_migration.py -v -s

# Compare specific strategy behaviors
pytest tests/unit/strategies/test_component_migration.py::TestComponentMigrationCompatibility::test_ml_basic_strategy_component_equivalence -v
```

**Solutions**:
- Review parameter mapping between systems
- Adjust tolerance levels for acceptable differences
- Update component logic to match legacy behavior

#### 4. Memory or Resource Issues

**Symptom**: Tests fail with memory errors or timeouts

**Diagnosis**:
```bash
# Run with memory profiling
pytest tests/performance/test_component_performance_regression.py::TestComponentPerformanceRegression::test_memory_usage_performance -v -s

# Check for resource leaks
pytest tests/integration/ -v --tb=short
```

**Solutions**:
- Review component cleanup and resource management
- Reduce test data size if appropriate
- Check for memory leaks in component implementations

### Debug Mode Testing

```bash
# Run tests with debug output
pytest tests/strategies/components/test_strategy.py -v -s --tb=long

# Run single test with maximum verbosity
pytest tests/integration/test_component_trading_workflows.py::TestEndToEndTradingWorkflows::test_complete_trading_session_trending_market -v -s --tb=long --capture=no

# Run with Python debugger
pytest tests/strategies/components/test_signal_generator.py::TestSignalGenerator::test_generate_signal -v -s --pdb
```

### Log Analysis

```bash
# Enable component logging during tests
export LOG_LEVEL=DEBUG
pytest tests/integration/test_component_trading_workflows.py -v -s

# Check test logs
tail -f tests/logs/test_execution.log
```

## Best Practices

### Test Writing Guidelines

1. **Descriptive Names**: Test names should clearly indicate what's being tested
   ```python
   def test_ml_signal_generator_produces_valid_buy_signal_with_high_confidence():
   ```

2. **Single Responsibility**: Each test should verify one specific behavior
   ```python
   def test_risk_manager_calculates_correct_position_size():
       # Test only position size calculation
   ```

3. **Arrange-Act-Assert**: Structure tests clearly
   ```python
   def test_strategy_processes_candle_successfully():
       # Arrange
       strategy = create_test_strategy()
       df = create_test_data()
       
       # Act
       decision = strategy.process_candle(df, 50, 10000.0)
       
       # Assert
       assert decision is not None
       assert isinstance(decision.signal.direction, SignalDirection)
   ```

4. **Independent Tests**: Tests should not depend on each other
   ```python
   def test_component_initialization():
       # Create fresh component for each test
       component = SignalGenerator("test")
   ```

5. **Meaningful Assertions**: Assert specific expected behaviors
   ```python
   # Good
   assert 0 <= signal.confidence <= 1
   assert signal.direction in [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]
   
   # Avoid
   assert signal is not None
   ```

### Performance Testing Guidelines

1. **Warm-up Runs**: Always include warm-up iterations
   ```python
   # Warm up
   for _ in range(5):
       component.process_data(test_data)
   
   # Measure
   times = []
   for _ in range(20):
       start = time.perf_counter()
       component.process_data(test_data)
       times.append(time.perf_counter() - start)
   ```

2. **Statistical Significance**: Use multiple measurements
   ```python
   avg_time = statistics.mean(times)
   p95_time = np.percentile(times, 95)
   ```

3. **Baseline Comparison**: Always compare against established baselines
   ```python
   result = baseline_manager.check_performance('test_name', avg_time)
   assert result['status'] != 'regression'
   ```

### Integration Testing Guidelines

1. **Realistic Scenarios**: Use realistic market data and conditions
2. **Error Injection**: Test error handling with controlled failures
3. **State Verification**: Verify system state after operations
4. **Resource Cleanup**: Ensure proper cleanup after tests

### Test Maintenance Guidelines

1. **Regular Review**: Review and update tests quarterly
2. **Baseline Updates**: Update performance baselines when appropriate
3. **Test Coverage**: Maintain >90% test coverage for critical components
4. **Documentation**: Keep test documentation current

## Maintenance Procedures

### Weekly Maintenance

```bash
# Update performance baselines
python tests/performance/performance_baseline_manager.py --cleanup 30

# Run comprehensive test suite
pytest tests/ --cov=src/strategies/components --cov-report=html

# Generate performance report
python tests/performance/automated_performance_monitor.py --dashboard-only
```

### Monthly Maintenance

```bash
# Export performance data for analysis
python tests/performance/performance_baseline_manager.py --export tests/performance/monthly_data.csv

# Run regression analysis
python tests/performance/performance_baseline_manager.py --regression-check

# Review test coverage
pytest --cov=src/strategies/components --cov-report=term-missing
```

### Quarterly Maintenance

1. **Test Review**: Review all tests for relevance and accuracy
2. **Baseline Adjustment**: Update performance baselines based on infrastructure changes
3. **Documentation Update**: Update test documentation and procedures
4. **Tool Updates**: Update testing tools and dependencies

### Performance Baseline Management

```bash
# View current baselines
python tests/performance/performance_baseline_manager.py --report

# Clean old data
python tests/performance/performance_baseline_manager.py --cleanup 60

# Export historical data
python tests/performance/performance_baseline_manager.py --export performance_history.csv
```

### Test Data Refresh

```bash
# Update test datasets
python tests/data/refresh_test_data.py

# Validate test data integrity
python tests/data/validate_test_data.py

# Archive old test data
python tests/data/archive_old_data.py --days 90
```

## Continuous Integration Integration

### GitHub Actions Example

```yaml
name: Component Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist
      
      - name: Run unit tests
        run: pytest tests/strategies/components/ -v --cov=src/strategies/components
      
      - name: Run integration tests
        run: pytest tests/integration/ -v -m integration
      
      - name: Run performance tests
        run: pytest tests/performance/ -v -m performance
      
      - name: Performance monitoring
        run: python tests/performance/automated_performance_monitor.py --full-cycle
      
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    stages {
        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'pytest tests/strategies/components/ -v --junit-xml=unit-results.xml'
                    }
                }
                stage('Integration Tests') {
                    steps {
                        sh 'pytest tests/integration/ -v --junit-xml=integration-results.xml'
                    }
                }
                stage('Performance Tests') {
                    steps {
                        sh 'python tests/performance/automated_performance_monitor.py --full-cycle'
                    }
                }
            }
        }
        stage('Performance Analysis') {
            steps {
                sh 'python tests/performance/performance_baseline_manager.py --regression-check'
            }
        }
    }
    post {
        always {
            junit 'unit-results.xml,integration-results.xml'
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'tests/performance/results',
                reportFiles: 'performance_dashboard.html',
                reportName: 'Performance Dashboard'
            ])
        }
    }
}
```

## Getting Help

### Resources

- **Component Documentation**: `src/strategies/components/README.md`
- **Migration Guide**: `tests/unit/strategies/TEST_MIGRATION_GUIDE.md`
- **Performance Dashboard**: `tests/performance/results/performance_dashboard.html`
- **Test Results**: `tests/performance/results/`

### Common Commands Reference

```bash
# Quick test run
pytest tests/strategies/components/ -v

# Performance check
python tests/performance/automated_performance_monitor.py --analysis-only

# Generate dashboard
python tests/performance/automated_performance_monitor.py --dashboard-only

# Full monitoring cycle
python tests/performance/automated_performance_monitor.py --full-cycle

# Test with coverage
pytest --cov=src/strategies/components --cov-report=html

# Debug specific test
pytest tests/path/to/test.py::test_name -v -s --pdb
```

### Support Contacts

- **Component System**: Check component documentation and tests
- **Performance Issues**: Review performance dashboard and baseline manager
- **Migration Questions**: Consult migration guide and compatibility tests
- **Test Failures**: Use troubleshooting section and debug commands