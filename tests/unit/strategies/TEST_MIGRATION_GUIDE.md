# Test Migration Guide

This guide explains how to migrate existing unit tests to work with the new component-based strategy system.

## Overview

The component-based strategy system is now the default runtime, and this guide captures how we
updated existing tests while preserving archived compatibility checks from the migration period.

## Migration Strategy

### Phase 1: Compatibility Tests
- Create tests that verify component strategies integrate with existing backtesting and live engine fixtures
- Validate error handling compatibility across component boundaries

### Phase 2: Component-Level Tests
- Convert strategy-level tests to component-level tests
- Test individual components in isolation
- Create comprehensive component interaction tests

### Phase 3: Integration Tests
- Test complete trading workflows with new components
- Validate regime-aware behavior
- Test performance under various market conditions

## Test Categories

### 1. Migration Compatibility Tests
Located in: `tests/unit/strategies/test_component_migration.py`

These tests ensure that:
- Component strategies stay aligned with archived legacy baselines
- Existing test fixtures work with component strategies
- Error handling is consistent between old and new systems
- Performance metrics maintain the same structure

### 2. Component Unit Tests
Located in: `tests/strategies/components/`

Individual component tests:
- `test_signal_generator.py` - Signal generation logic
- `test_risk_manager.py` - Risk management and position sizing
- `test_position_sizer.py` - Position sizing algorithms
- `test_strategy.py` - Complete strategy composition

### 3. Legacy Adapter Tests
Located in: `tests/unit/strategies/migration/`

Tests for migration utilities:
- `test_strategy_converter.py` - Strategy conversion utilities
- Parameter mapping validation
- Component creation from archived legacy parameter snapshots

## Migration Checklist

### For Each Existing Strategy Test:

1. **Identify Test Purpose**
   - [ ] What specific behavior is being tested?
   - [ ] Which component(s) would handle this in the new system?
   - [ ] Is this a unit test or integration test?

2. **Create Component Test**
   - [ ] Extract the core logic being tested
   - [ ] Create equivalent test for appropriate component
   - [ ] Ensure test covers edge cases from original test

3. **Update Test Fixtures**
   - [ ] Ensure fixtures work with component interfaces
   - [ ] Add any new fixtures needed for components
   - [ ] Update mock objects for new interfaces

### Example Migration

**Original Test:**
```python
def test_ml_basic_entry_conditions(self):
    strategy = create_ml_basic_strategy()
    df = create_test_data()
    decision = strategy.process_candle(df, 50, balance=10_000.0)
    
    assert decision.signal.direction is not None
    assert decision.position_size >= 0
```

**Migrated Component Test:**
```python
def test_ml_basic_signal_generation(self):
    signal_generator = MLBasicSignalGenerator()
    df = create_test_data()
    
    signal = signal_generator.generate_signal(df, 50)
    assert signal.direction in [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]
    assert 0 <= signal.confidence <= 1
```

## Test Data Management

### Shared Test Data
- Use consistent test datasets across old and new tests
- Create helper functions for generating test data
- Ensure test data includes all required indicators

### Mock Objects
- Update mocks to work with component interfaces
- Create component-specific mocks
- Maintain backward compatibility for existing mocks

## Performance Considerations

### Test Execution Speed
- Component tests should be faster than full strategy tests
- Use appropriate test markers (`@pytest.mark.unit`, `@pytest.mark.integration`)
- Parallelize independent component tests

### Test Coverage
- Maintain or improve test coverage during migration
- Focus on critical paths and edge cases
- Use coverage reports to identify gaps

## Common Migration Patterns

### 1. Strategy Method → Component Method
```python
# Old
strategy.process_candle(df, index, balance) → TradingDecision

# New  
signal_generator.generate_signal(df, index) → Signal
```

### 2. Parameter Access
```python
# Old
strategy.stop_loss_pct

# New
risk_manager.get_parameters()['stop_loss_pct']
```

### 3. Error Handling
```python
# Old
try:
    result = strategy.some_method()
except Exception:
    result = default_value

# New
# Components handle errors internally and return safe defaults
result = component.some_method()  # Always returns valid result
```

## Validation Steps

### Before Migration
1. Run existing test suite and record results
2. Identify all tests that need migration
3. Document expected behavior for each test

### During Migration
1. Run both old and new tests in parallel
2. Compare results and investigate differences
3. Update tests incrementally, not all at once

### After Migration
1. Verify all tests pass with new system
2. Check test coverage hasn't decreased
3. Run performance benchmarks
4. Validate integration with CI/CD pipeline

## Troubleshooting

### Common Issues

1. **Test Data Incompatibility**
   - Ensure test data includes all required columns
   - Check that indicators are calculated correctly
   - Verify data types and ranges

2. **Mock Object Issues**
   - Update mock interfaces to match components
   - Ensure mocks return expected data types
   - Check that mock behavior is consistent

3. **Assertion Failures**
   - Component results may differ slightly from archived legacy outputs
   - Use appropriate tolerance for floating-point comparisons
   - Focus on behavioral equivalence, not exact matches

4. **Performance Regressions**
   - Component tests should be faster, not slower
   - Check for unnecessary computation in tests
   - Use appropriate test fixtures and caching

### Getting Help

- Check existing component tests for examples
- Review migration compatibility tests
- Consult the component documentation
- Run tests with verbose output for debugging

## Best Practices

1. **Test One Thing at a Time**
   - Focus component tests on single responsibilities
   - Use integration tests for component interactions
   - Keep tests simple and focused

2. **Use Descriptive Names**
   - Test names should clearly indicate what's being tested
   - Include component name and behavior in test names
   - Group related tests in classes

3. **Maintain Test Independence**
   - Each test should be able to run independently
   - Don't rely on test execution order
   - Clean up any state changes in teardown

4. **Document Complex Tests**
   - Add comments explaining complex test logic
   - Document expected behavior and edge cases
   - Include references to requirements or design docs

## Timeline

### Week 1: Assessment and Planning
- Inventory existing tests
- Identify migration priorities
- Set up compatibility test framework

### Week 2-3: Core Component Tests
- Migrate signal generator tests
- Migrate risk manager tests
- Migrate position sizer tests

### Week 4: Integration and Validation
- Create integration tests
- Validate compatibility
- Performance testing and optimization

### Week 5: Documentation and Cleanup
- Update test documentation
- Remove deprecated tests
- Final validation and sign-off
