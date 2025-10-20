# Test Troubleshooting Guide

This guide provides solutions for common test failures and debugging procedures for the component-based strategy system.

## Table of Contents

1. [Quick Diagnosis](#quick-diagnosis)
2. [Common Test Failures](#common-test-failures)
3. [Performance Issues](#performance-issues)
4. [Integration Problems](#integration-problems)
5. [Migration Issues](#migration-issues)
6. [Environment Problems](#environment-problems)
7. [Debugging Techniques](#debugging-techniques)
8. [Recovery Procedures](#recovery-procedures)

## Quick Diagnosis

### Test Failure Triage

When tests fail, follow this triage process:

1. **Check the error type**:
   ```bash
   pytest tests/strategies/components/ -v --tb=short
   ```

2. **Identify the failure category**:
   - `AssertionError`: Logic or expectation issue
   - `ImportError`: Missing dependencies or module issues
   - `TimeoutError`: Performance or hanging issue
   - `MemoryError`: Resource exhaustion
   - `ConnectionError`: External dependency issue

3. **Check recent changes**:
   ```bash
   git log --oneline -10
   git diff HEAD~1 tests/
   ```

4. **Verify environment**:
   ```bash
   python --version
   pip list | grep -E "(pytest|pandas|numpy)"
   ```

### Quick Fixes Checklist

- [ ] Run tests in clean environment: `python -m pytest tests/ --cache-clear`
- [ ] Update dependencies: `pip install -r requirements.txt`
- [ ] Clear Python cache: `find . -name "*.pyc" -delete && find . -name "__pycache__" -delete`
- [ ] Check disk space: `df -h`
- [ ] Verify test data: `ls -la tests/data/`

## Common Test Failures

### 1. Component Interface Errors

**Symptoms**:
```
AttributeError: 'MLBasicSignalGenerator' object has no attribute 'generate_signal'
TypeError: generate_signal() missing 1 required positional argument: 'regime'
```

**Diagnosis**:
```bash
# Check component interface
python -c "from src.strategies.components.signal_generator import MLBasicSignalGenerator; print(dir(MLBasicSignalGenerator()))"

# Verify method signatures
pytest tests/strategies/components/test_signal_generator.py::TestSignalGenerator::test_signal_generator_initialization -v -s
```

**Solutions**:

1. **Missing Method Implementation**:
   ```python
   # Ensure all abstract methods are implemented
   class MySignalGenerator(SignalGenerator):
       def generate_signal(self, df, index, regime=None):
           # Implementation required
           pass
       
       def get_confidence(self, df, index):
           # Implementation required
           pass
   ```

2. **Incorrect Method Signature**:
   ```python
   # Correct signature
   def generate_signal(self, df: pd.DataFrame, index: int, regime: Optional[RegimeContext] = None) -> Signal:
   ```

3. **Import Issues**:
   ```python
   # Ensure correct imports
   from src.strategies.components.signal_generator import SignalGenerator, Signal, SignalDirection
   ```

### 2. Data Validation Errors

**Symptoms**:
```
ValueError: DataFrame missing required columns: ['open', 'high', 'low', 'close', 'volume']
IndexError: Index 50 is out of bounds for DataFrame of length 10
ValueError: strength must be between 0.0 and 1.0, got 1.5
```

**Diagnosis**:
```bash
# Check test data structure
pytest tests/strategies/components/test_signal_generator.py -v -s --tb=long

# Verify data creation
python -c "
from tests.strategies.components.test_signal_generator import TestSignalGenerator
test = TestSignalGenerator()
df = test.create_test_dataframe()
print(df.columns.tolist())
print(f'Length: {len(df)}')
print(df.head())
"
```

**Solutions**:

1. **Missing Columns**:
   ```python
   def create_test_dataframe(self):
       return pd.DataFrame({
           'open': [100, 101, 102],
           'high': [101, 102, 103],
           'low': [99, 100, 101],
           'close': [100.5, 101.5, 102.5],
           'volume': [1000, 1100, 1200],
           # Add any additional required columns
           'onnx_pred': [101, 102, 103],
           'atr': [1.0, 1.0, 1.0]
       })
   ```

2. **Index Out of Bounds**:
   ```python
   def test_signal_generation(self):
       df = self.create_test_dataframe()
       # Ensure index is within bounds
       index = min(50, len(df) - 1)
       signal = generator.generate_signal(df, index)
   ```

3. **Value Range Validation**:
   ```python
   # Ensure values are within expected ranges
   signal = Signal(
       direction=SignalDirection.BUY,
       strength=max(0.0, min(1.0, calculated_strength)),  # Clamp to [0, 1]
       confidence=max(0.0, min(1.0, calculated_confidence)),
       metadata={}
   )
   ```

### 3. Mock and Fixture Issues

**Symptoms**:
```
AttributeError: Mock object has no attribute 'generate_signal'
TypeError: 'Mock' object is not callable
AssertionError: Expected call not found
```

**Diagnosis**:
```bash
# Run with mock debugging
pytest tests/strategies/components/test_strategy.py -v -s --tb=long

# Check mock setup
python -c "
from unittest.mock import Mock
mock_gen = Mock()
mock_gen.generate_signal.return_value = 'test'
print(mock_gen.generate_signal())
"
```

**Solutions**:

1. **Proper Mock Configuration**:
   ```python
   from unittest.mock import Mock, patch
   
   def test_with_mock_component(self):
       mock_generator = Mock(spec=SignalGenerator)
       mock_generator.generate_signal.return_value = Signal(
           direction=SignalDirection.BUY,
           strength=0.8,
           confidence=0.9,
           metadata={}
       )
       mock_generator.get_confidence.return_value = 0.9
       
       strategy = Strategy(
           name="test",
           signal_generator=mock_generator,
           risk_manager=FixedRiskManager(),
           position_sizer=ConfidenceWeightedSizer()
       )
   ```

2. **Fixture Setup**:
   ```python
   @pytest.fixture
   def test_strategy():
       return Strategy(
           name="test_strategy",
           signal_generator=MLBasicSignalGenerator(),
           risk_manager=FixedRiskManager(),
           position_sizer=ConfidenceWeightedSizer()
       )
   
   def test_strategy_functionality(test_strategy):
       # Use fixture
       result = test_strategy.process_candle(df, 50, 10000.0)
   ```

### 4. Async and Threading Issues

**Symptoms**:
```
RuntimeError: This event loop is already running
asyncio.TimeoutError: Task timed out after 30 seconds
```

**Diagnosis**:
```bash
# Check for async issues
pytest tests/integration/test_error_handling_workflows.py::TestErrorHandlingWorkflows::test_concurrent_access_safety -v -s

# Run with threading debug
python -c "
import threading
print(f'Active threads: {threading.active_count()}')
for thread in threading.enumerate():
    print(f'  {thread.name}: {thread.is_alive()}')
"
```

**Solutions**:

1. **Proper Thread Management**:
   ```python
   import threading
   import time
   
   def test_concurrent_processing(self):
       threads = []
       results = []
       
       def worker():
           result = strategy.process_candle(df, 50, 10000.0)
           results.append(result)
       
       # Create and start threads
       for _ in range(3):
           thread = threading.Thread(target=worker)
           threads.append(thread)
           thread.start()
       
       # Wait for completion with timeout
       for thread in threads:
           thread.join(timeout=10.0)
           if thread.is_alive():
               pytest.fail("Thread did not complete within timeout")
   ```

2. **Avoid Async in Sync Tests**:
   ```python
   # Don't mix async/await with sync test functions
   def test_sync_function(self):
       # Use synchronous calls only
       result = strategy.process_candle(df, 50, 10000.0)
   ```

## Performance Issues

### 1. Slow Test Execution

**Symptoms**:
```
tests/performance/test_component_performance_regression.py::test_signal_generation_performance FAILED
Performance regression: 15.23ms > 10.00ms
```

**Diagnosis**:
```bash
# Profile test execution
pytest tests/performance/ -v --durations=10

# Check system resources
top -p $(pgrep -f pytest)
free -h
```

**Solutions**:

1. **Optimize Test Data**:
   ```python
   def create_test_data(self, size=100):  # Reduce from 1000
       # Use smaller datasets for unit tests
       np.random.seed(42)  # Ensure reproducibility
       return pd.DataFrame({
           'close': np.random.uniform(100, 110, size)
           # ... other columns
       })
   ```

2. **Use Caching**:
   ```python
   @pytest.fixture(scope="session")
   def large_test_dataset():
       # Create expensive data once per session
       return create_large_dataset()
   ```

3. **Parallel Execution**:
   ```bash
   # Run tests in parallel
   pytest tests/strategies/components/ -n auto
   ```

### 2. Memory Issues

**Symptoms**:
```
MemoryError: Unable to allocate array
pytest: internal error: killed by signal 9 (SIGKILL)
```

**Diagnosis**:
```bash
# Monitor memory usage
pytest tests/performance/test_component_performance_regression.py::test_memory_usage_performance -v -s

# Check memory limits
ulimit -a
```

**Solutions**:

1. **Reduce Memory Usage**:
   ```python
   def test_with_memory_management(self):
       # Process data in chunks
       chunk_size = 100
       for i in range(0, len(large_df), chunk_size):
           chunk = large_df.iloc[i:i+chunk_size]
           result = strategy.process_candle(chunk, 50, 10000.0)
           # Process result immediately, don't accumulate
   ```

2. **Clean Up Resources**:
   ```python
   def test_with_cleanup(self):
       try:
           # Test logic
           pass
       finally:
           # Explicit cleanup
           strategy.clear_history()
           del large_data
           import gc
           gc.collect()
   ```

### 3. Performance Regression Detection

**Symptoms**:
```
Performance regression detected: signal_generation 20.5% slower than baseline
```

**Diagnosis**:
```bash
# Check performance trends
python tests/performance/performance_baseline_manager.py --report

# Run regression analysis
python tests/performance/performance_baseline_manager.py --regression-check
```

**Solutions**:

1. **Update Baselines** (if change is intentional):
   ```bash
   # Record new baseline
   python tests/performance/performance_baseline_manager.py --update-baseline signal_generation
   ```

2. **Optimize Performance**:
   ```python
   # Profile slow components
   import cProfile
   
   def test_with_profiling(self):
       profiler = cProfile.Profile()
       profiler.enable()
       
       # Test code
       strategy.process_candle(df, 50, 10000.0)
       
       profiler.disable()
       profiler.print_stats(sort='cumulative')
   ```

## Integration Problems

### 1. Component Communication Failures

**Symptoms**:
```
TypeError: 'NoneType' object has no attribute 'direction'
AttributeError: 'Signal' object has no attribute 'metadata'
```

**Diagnosis**:
```bash
# Test component integration
pytest tests/integration/test_component_trading_workflows.py::TestEndToEndTradingWorkflows::test_multi_component_integration -v -s --tb=long

# Check data flow
python -c "
from src.strategies.components.strategy import Strategy
from src.strategies.components.signal_generator import MLBasicSignalGenerator
# ... test component creation and interaction
"
```

**Solutions**:

1. **Validate Component Outputs**:
   ```python
   def generate_signal(self, df, index, regime=None):
       # Always return valid Signal object
       try:
           # Signal generation logic
           signal = Signal(...)
           return signal
       except Exception as e:
           # Return safe fallback
           return Signal(
               direction=SignalDirection.HOLD,
               strength=0.0,
               confidence=0.0,
               metadata={'error': str(e)}
           )
   ```

2. **Check Component Interfaces**:
   ```python
   def test_component_interface_compatibility(self):
       signal_gen = MLBasicSignalGenerator()
       risk_mgr = FixedRiskManager()
       pos_sizer = ConfidenceWeightedSizer()
       
       # Verify interfaces match
       assert hasattr(signal_gen, 'generate_signal')
       assert hasattr(risk_mgr, 'calculate_position_size')
       assert hasattr(pos_sizer, 'calculate_size')
   ```

### 2. Regime Detection Issues

**Symptoms**:
```
AttributeError: 'NoneType' object has no attribute 'trend'
ValueError: Invalid regime type: 'unknown'
```

**Diagnosis**:
```bash
# Test regime detection
pytest tests/strategies/components/test_regime_context.py -v -s

# Check regime data
python -c "
from src.strategies.components.regime_context import EnhancedRegimeDetector
detector = EnhancedRegimeDetector()
# Test with sample data
"
```

**Solutions**:

1. **Handle Missing Regime Data**:
   ```python
   def process_candle(self, df, index, balance):
       regime = self._detect_regime(df, index)
       
       # Handle None regime gracefully
       if regime is None:
           regime = self._create_default_regime()
       
       signal = self.signal_generator.generate_signal(df, index, regime)
   ```

2. **Validate Regime Context**:
   ```python
   def _create_default_regime(self):
       return RegimeContext(
           trend=TrendLabel.RANGE,
           volatility=VolLabel.LOW,
           confidence=0.5,
           duration=1,
           strength=0.5
       )
   ```

## Migration Issues

### 1. Legacy Compatibility Failures

**Symptoms**:
```
AssertionError: Component strategy produces different results than legacy
ValueError: Parameter mapping failed for 'stop_loss_pct'
```

**Diagnosis**:
```bash
# Test migration compatibility
pytest tests/unit/strategies/test_component_migration.py -v -s

# Compare specific behaviors
pytest tests/unit/strategies/test_component_migration.py::TestComponentMigrationCompatibility::test_ml_basic_strategy_component_equivalence -v -s --tb=long
```

**Solutions**:

1. **Parameter Mapping**:
   ```python
   def convert_legacy_parameters(legacy_params):
       component_params = {}
       
       # Map legacy parameters to component parameters
       if 'stop_loss_pct' in legacy_params:
           component_params['risk_manager'] = {
               'type': 'FixedRiskManager',
               'stop_loss_pct': legacy_params['stop_loss_pct']
           }
       
       return component_params
   ```

2. **Tolerance for Differences**:
   ```python
   def test_legacy_compatibility(self):
       legacy_result = legacy_strategy.calculate_position_size(df, 50, 10000.0)
       component_result = component_strategy.process_candle(df, 50, 10000.0).position_size
       
       # Allow small differences due to implementation details
       assert abs(legacy_result - component_result) < 0.01 * legacy_result
   ```

### 2. Test Fixture Migration

**Symptoms**:
```
AttributeError: 'Strategy' object has no attribute 'check_entry_conditions'
TypeError: process_candle() takes 4 positional arguments but 5 were given
```

**Solutions**:

1. **Update Test Fixtures**:
   ```python
   from src.strategies.ml_basic import create_ml_basic_strategy
   
   # Updated fixture using component factory
   @pytest.fixture
   def component_strategy():
       return create_ml_basic_strategy()
   ```

2. **Adapter Pattern for Tests**:
   ```python
   class TestStrategyAdapter:
       def __init__(self, component_strategy):
           self.strategy = component_strategy
       
       def check_entry_conditions(self, df, index):
           decision = self.strategy.process_candle(df, index, 10000.0)
           return decision.signal.direction != SignalDirection.HOLD
   ```

## Environment Problems

### 1. Dependency Issues

**Symptoms**:
```
ImportError: No module named 'src.strategies.components'
ModuleNotFoundError: No module named 'pandas'
```

**Diagnosis**:
```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Verify installations
pip list | grep -E "(pandas|numpy|pytest)"

# Check module imports
python -c "from src.strategies.components.strategy import Strategy; print('Import successful')"
```

**Solutions**:

1. **Fix Python Path**:
   ```bash
   # Add project root to PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   
   # Or run tests from project root
   cd /path/to/project
   pytest tests/
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov pytest-xdist
   ```

3. **Virtual Environment**:
   ```bash
   python -m venv test_env
   source test_env/bin/activate  # Linux/Mac
   # test_env\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

### 2. Data File Issues

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'tests/data/test_data.feather'
PermissionError: [Errno 13] Permission denied: 'tests/data/'
```

**Diagnosis**:
```bash
# Check data files
ls -la tests/data/

# Check permissions
ls -ld tests/data/
```

**Solutions**:

1. **Create Missing Data**:
   ```bash
   mkdir -p tests/data
   python -c "
   import pandas as pd
   import numpy as np
   df = pd.DataFrame({'close': np.random.uniform(100, 110, 100)})
   df.to_feather('tests/data/test_data.feather')
   "
   ```

2. **Fix Permissions**:
   ```bash
   chmod -R 755 tests/data/
   ```

## Debugging Techniques

### 1. Verbose Test Output

```bash
# Maximum verbosity
pytest tests/strategies/components/test_signal_generator.py -v -s --tb=long --capture=no

# Show local variables in tracebacks
pytest tests/strategies/components/test_signal_generator.py --tb=long --showlocals

# Stop on first failure
pytest tests/strategies/components/ -x
```

### 2. Interactive Debugging

```bash
# Drop into debugger on failure
pytest tests/strategies/components/test_signal_generator.py --pdb

# Drop into debugger on specific test
pytest tests/strategies/components/test_signal_generator.py::TestSignalGenerator::test_specific_case --pdb
```

### 3. Logging and Print Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

def test_with_logging(self):
    logger = logging.getLogger(__name__)
    
    logger.debug("Starting test")
    signal = generator.generate_signal(df, 50)
    logger.debug(f"Generated signal: {signal}")
    
    # Or use print for simple debugging
    print(f"Signal direction: {signal.direction}")
    print(f"Signal confidence: {signal.confidence}")
```

### 4. Component State Inspection

```python
def test_with_state_inspection(self):
    strategy = Strategy(...)
    
    # Process some data
    decision = strategy.process_candle(df, 50, 10000.0)
    
    # Inspect component states
    print(f"Decision history length: {len(strategy.decision_history)}")
    print(f"Component info: {strategy.get_component_info()}")
    print(f"Performance metrics: {strategy.get_performance_metrics()}")
```

## Recovery Procedures

### 1. Reset Test Environment

```bash
# Clean Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

# Clear pytest cache
pytest --cache-clear

# Reset virtual environment
deactivate
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Reset Performance Baselines

```bash
# Backup current baselines
cp tests/performance/component_baselines.json tests/performance/component_baselines.json.backup

# Reset baselines
python tests/performance/performance_baseline_manager.py --reset-baselines

# Or restore from backup
cp tests/performance/component_baselines.json.backup tests/performance/component_baselines.json
```

### 3. Regenerate Test Data

```bash
# Remove old test data
rm -rf tests/data/*.feather

# Regenerate test data
python tests/data/generate_test_data.py

# Verify test data
python tests/data/validate_test_data.py
```

### 4. Emergency Test Bypass

If tests are blocking critical work, you can temporarily bypass them:

```bash
# Skip performance tests
pytest tests/ -m "not performance"

# Skip slow tests
pytest tests/ -m "not slow"

# Skip integration tests
pytest tests/ -m "not integration"

# Run only unit tests
pytest tests/strategies/components/ -m unit
```

### 5. Rollback to Known Good State

```bash
# Find last known good commit
git log --oneline --grep="tests passing"

# Create branch from good state
git checkout -b fix-tests <good-commit-hash>

# Cherry-pick specific changes
git cherry-pick <commit-hash>

# Run tests to verify
pytest tests/
```

## Getting Additional Help

### Log Analysis

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
pytest tests/ -v -s > test_output.log 2>&1

# Analyze logs
grep -i error test_output.log
grep -i failed test_output.log
```

### System Information

```bash
# Collect system info for bug reports
python -c "
import sys, platform, pandas, numpy
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Pandas: {pandas.__version__}')
print(f'NumPy: {numpy.__version__}')
"

# Check resource usage
free -h
df -h
top -n 1 -b | head -20
```

### Contact Information

- **Test Issues**: Check this troubleshooting guide first
- **Performance Problems**: Review performance dashboard and baseline manager
- **Component Bugs**: Check component documentation and unit tests
- **Migration Issues**: Consult migration guide and compatibility tests
