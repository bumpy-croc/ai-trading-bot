# Strategy System Migration - Complete Cutover Implementation Plan

## Overview

This implementation plan completes the migration from the legacy `BaseStrategy` system to the component-based `Strategy` system by removing all legacy code, adapters, and migration utilities.

## Implementation Tasks

- [x] 1. Convert concrete strategy implementations to component-based
  - Convert ml_basic.py to use component composition
  - Convert ml_adaptive.py to use component composition
  - Convert ml_sentiment.py to use component composition
  - Convert ensemble_weighted.py to use component composition
  - Convert momentum_leverage.py to use component composition
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 1.1 Convert ml_basic.py to component-based strategy
  - Create factory function `create_ml_basic_strategy()` that returns configured `Strategy` instance
  - Use `MLBasicSignalGenerator` for signal generation
  - Use `FixedRiskManager` for risk management
  - Use `ConfidenceWeightedSizer` for position sizing
  - Remove `BaseStrategy` inheritance and legacy methods
  - _Requirements: 6.1, 6.6_

- [x] 1.2 Convert ml_adaptive.py to component-based strategy
  - Create factory function `create_ml_adaptive_strategy()` that returns configured `Strategy` instance
  - Use `MLSignalGenerator` with regime-aware thresholds
  - Use `RegimeAdaptiveRiskManager` for regime-specific risk management
  - Use `RegimeAdaptiveSizer` for regime-specific position sizing
  - Remove `BaseStrategy` inheritance and legacy methods
  - _Requirements: 6.2, 6.6_

- [x] 1.3 Convert ml_sentiment.py to component-based strategy
  - Create factory function `create_ml_sentiment_strategy()` that returns configured `Strategy` instance
  - Create or use sentiment-aware signal generator
  - Use appropriate risk manager and position sizer
  - Remove `BaseStrategy` inheritance and legacy methods
  - _Requirements: 6.3, 6.6_

- [x] 1.4 Convert ensemble_weighted.py to component-based strategy
  - Create factory function `create_ensemble_weighted_strategy()` that returns configured `Strategy` instance
  - Use `WeightedVotingSignalGenerator` to combine multiple signal sources
  - Use appropriate risk manager and position sizer
  - Remove `BaseStrategy` inheritance and legacy methods
  - _Requirements: 6.4, 6.6_

- [x] 1.5 Convert momentum_leverage.py to component-based strategy
  - Create factory function `create_momentum_leverage_strategy()` that returns configured `Strategy` instance
  - Use `MomentumSignalGenerator` for momentum-based signals
  - Use appropriate risk manager and position sizer for aggressive sizing
  - Remove `BaseStrategy` inheritance and legacy methods
  - _Requirements: 6.5, 6.6_

- [x] 2. Update backtesting engine to use component-based interface
  - Remove upfront `calculate_indicators()` call
  - Replace legacy interface calls with `process_candle()`
  - Use `TradingDecision` objects for entry/exit logic
  - Update position sizing to use `decision.position_size`
  - Update stop loss logic to use `strategy.get_stop_loss_price()`
  - Update regime switching to work with component strategies
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 2.1 Remove calculate_indicators() from backtesting engine
  - Find and remove all calls to `strategy.calculate_indicators(df)`
  - Remove any DataFrame column expectations from legacy interface
  - Ensure DataFrame contains only OHLCV data
  - _Requirements: 4.2_

- [x] 2.2 Replace entry/exit checks with process_candle()
  - Replace `check_entry_conditions()` calls with `process_candle()`
  - Replace `check_exit_conditions()` calls with `should_exit_position()`
  - Use `decision.signal.direction` to determine entry/exit
  - Handle HOLD signals appropriately
  - _Requirements: 4.1, 4.3_

- [x] 2.3 Update position sizing in backtesting engine
  - Remove `calculate_position_size()` calls
  - Use `decision.position_size` from `TradingDecision`
  - Validate position size bounds
  - _Requirements: 4.4_

- [x] 2.4 Update stop loss logic in backtesting engine
  - Remove `calculate_stop_loss()` calls
  - Use `strategy.get_stop_loss_price()` method
  - Pass `decision.signal` and `decision.regime` to stop loss calculation
  - _Requirements: 4.5_

- [x] 2.5 Update regime switching in backtesting engine
  - Remove `calculate_indicators()` calls during strategy switches
  - Ensure regime switching works with component strategies
  - Test regime transitions with `TradingDecision` objects
  - _Requirements: 4.6_

- [x] 3. Update live trading engine to use component-based interface
  - Remove upfront `calculate_indicators()` call
  - Replace legacy interface calls with `process_candle()`
  - Use `TradingDecision` objects for entry/exit logic
  - Update database logging to capture `TradingDecision` data
  - Update strategy hot-swapping to work with component strategies
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 3.1 Remove calculate_indicators() from live trading engine
  - Find and remove all calls to `strategy.calculate_indicators(df)`
  - Remove any DataFrame column expectations from legacy interface
  - Ensure DataFrame contains only OHLCV data
  - _Requirements: 5.2_

- [x] 3.2 Replace entry/exit checks with process_candle() in live trading
  - Replace `check_entry_conditions()` calls with `process_candle()`
  - Replace `check_exit_conditions()` calls with `should_exit_position()`
  - Use `decision.signal.direction` to determine entry/exit
  - Handle HOLD signals appropriately
  - _Requirements: 5.1, 5.3_

- [x] 3.3 Update position sizing in live trading engine
  - Remove `calculate_position_size()` calls
  - Use `decision.position_size` from `TradingDecision`
  - Validate position size bounds
  - _Requirements: 5.4_

- [x] 3.4 Update database logging for TradingDecision
  - Modify database logging to capture `decision.to_dict()`
  - Log signal direction, confidence, strength
  - Log regime context if available
  - Log risk metrics and metadata
  - _Requirements: 5.6_

- [x] 3.5 Update strategy hot-swapping for component strategies
  - Ensure hot-swapping works with component-based strategies
  - Remove legacy interface dependencies from hot-swap logic
  - Test strategy transitions with `TradingDecision` objects
  - _Requirements: 5.5_

- [x] 4. Update strategy unit tests
  - Update all strategy tests to use component-based interface
  - Remove calls to `calculate_indicators()`
  - Test `process_candle()` and validate `TradingDecision` objects
  - Test individual components in isolation
  - Plenty of tests already exist, only write new tests if they don't exist. Modify existing tests first.
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 4.1 Update ml_basic strategy tests
  - Remove tests that call `calculate_indicators()`
  - Add tests for `create_ml_basic_strategy()` factory function
  - Test `process_candle()` returns valid `TradingDecision`
  - Test signal generation, risk management, position sizing
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 4.2 Update ml_adaptive strategy tests
  - Remove tests that call `calculate_indicators()`
  - Add tests for `create_ml_adaptive_strategy()` factory function
  - Test regime-aware behavior
  - Test `process_candle()` returns valid `TradingDecision`
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 4.3 Update ml_sentiment strategy tests
  - Remove tests that call `calculate_indicators()`
  - Add tests for `create_ml_sentiment_strategy()` factory function
  - Test sentiment integration
  - Test `process_candle()` returns valid `TradingDecision`
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 4.4 Update ensemble_weighted strategy tests
  - Remove tests that call `calculate_indicators()`
  - Add tests for `create_ensemble_weighted_strategy()` factory function
  - Test signal combination logic
  - Test `process_candle()` returns valid `TradingDecision`
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 4.5 Update momentum_leverage strategy tests
  - Remove tests that call `calculate_indicators()`
  - Add tests for `create_momentum_leverage_strategy()` factory function
  - Test momentum signal generation
  - Test `process_candle()` returns valid `TradingDecision`
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 4.6 Add component isolation tests
  - Test signal generators in isolation
  - Test risk managers in isolation
  - Test position sizers in isolation
  - Test component composition
  - _Requirements: 7.4_

- [ ] 5. Update backtesting integration tests
  - Update all backtesting tests to use component-based strategies
  - Remove legacy interface references
  - Validate `TradingDecision` objects in tests
  - Ensure all backtesting integration tests pass
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 5.1 Update single strategy backtest tests
  - Use component-based strategies in backtest tests
  - Validate `TradingDecision` objects
  - Test entry/exit logic with new interface
  - _Requirements: 8.1, 8.2_

- [ ] 5.2 Update regime switching backtest tests
  - Test regime switching with component strategies
  - Validate strategy transitions
  - Ensure regime detection works correctly
  - _Requirements: 8.1, 8.2_

- [ ] 5.3 Update performance metrics tests
  - Test performance calculation with component strategies
  - Validate metrics accuracy
  - Test regime-specific performance metrics
  - _Requirements: 8.1, 8.2_

- [ ] 5.4 Remove legacy interface references from backtest tests
  - Remove all calls to `calculate_indicators()`
  - Remove all calls to `check_entry_conditions()`, `check_exit_conditions()`
  - Remove DataFrame column assertions for legacy columns
  - _Requirements: 8.3_

- [ ] 6. Update live trading integration tests
  - Update all live trading tests to use component-based strategies
  - Remove legacy interface references
  - Validate `TradingDecision` objects in tests
  - Ensure all live trading integration tests pass
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ] 6.1 Update order execution tests
  - Use component-based strategies in order execution tests
  - Validate `TradingDecision` objects
  - Test order placement with new interface
  - _Requirements: 9.1, 9.2_

- [ ] 6.2 Update position management tests
  - Test position management with component strategies
  - Validate exit logic using `should_exit_position()`
  - Test stop loss updates
  - _Requirements: 9.1, 9.2_

- [ ] 6.3 Update database logging tests
  - Test database logging with `TradingDecision` objects
  - Validate logged data structure
  - Test signal, regime, and risk metrics logging
  - _Requirements: 9.1, 9.2_

- [ ] 6.4 Remove legacy interface references from live trading tests
  - Remove all calls to `calculate_indicators()`
  - Remove all calls to `check_entry_conditions()`, `check_exit_conditions()`
  - Remove DataFrame column assertions for legacy columns
  - _Requirements: 9.3_

- [ ] 7. Delete legacy code and adapters
  - Delete `src/strategies/adapters/` directory
  - Delete `src/strategies/migration/` directory
  - Delete `src/strategies/base.py` (BaseStrategy)
  - Delete adapter tests
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4_

- [ ] 7.1 Delete adapter directory
  - Delete `src/strategies/adapters/legacy_adapter.py`
  - Delete `src/strategies/adapters/adapter_factory.py`
  - Delete `src/strategies/adapters/__init__.py`
  - Delete entire `src/strategies/adapters/` directory
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 7.2 Delete migration utilities directory
  - Delete `src/strategies/migration/strategy_converter.py`
  - Delete `src/strategies/migration/validation_utils.py`
  - Delete `src/strategies/migration/cross_validation.py`
  - Delete `src/strategies/migration/regression_testing.py`
  - Delete `src/strategies/migration/difference_analysis.py`
  - Delete `src/strategies/migration/rollback_manager.py`
  - Delete `src/strategies/migration/rollback_validation.py`
  - Delete entire `src/strategies/migration/` directory
  - _Requirements: 3.1, 3.2_

- [ ] 7.3 Delete BaseStrategy class
  - Delete `src/strategies/base.py`
  - Remove imports of `BaseStrategy` from other files
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 7.4 Delete adapter tests
  - Delete tests for `LegacyStrategyAdapter`
  - Delete tests for `adapter_factory`
  - Delete any migration-related tests
  - _Requirements: 2.4_

- [ ] 8. Update documentation and remove migration references
  - Delete or archive `MIGRATION.md`
  - Update `README.md` to remove legacy references
  - Update code comments to remove "legacy", "componentised", "new" terminology
  - Update docstrings to reflect component-based interface only
  - _Requirements: 3.3, 3.4, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [ ] 8.1 Delete migration documentation
  - Delete `src/strategies/MIGRATION.md`
  - Remove migration references from other documentation files
  - _Requirements: 3.3_

- [ ] 8.2 Update README files
  - Update `src/strategies/README.md` to document component-based approach only
  - Remove references to `BaseStrategy` and legacy interface
  - Add examples of creating strategies using component composition
  - _Requirements: 3.4, 10.4, 10.5_

- [ ] 8.3 Clean up code comments
  - Remove comments referring to "legacy" strategies
  - Remove comments referring to "componentised" or "new" strategies
  - Remove comments referring to "migration" or "adapter"
  - Update comments to describe component-based architecture
  - _Requirements: 10.1, 10.2, 10.3_

- [ ] 8.4 Update docstrings
  - Update docstrings to reflect component-based interface
  - Remove references to legacy methods
  - Document `process_candle()` and `TradingDecision` usage
  - _Requirements: 10.4, 10.6_

- [ ] 9. Verify all tests pass
  - Run all unit tests and ensure they pass
  - Run all integration tests and ensure they pass
  - Run backtesting tests with component strategies
  - Run live trading tests with component strategies
  - _Requirements: 7.6, 8.4, 9.4_

- [ ] 9.1 Run unit test suite
  - Execute all unit tests
  - Verify no tests reference legacy interface
  - Fix any failing tests
  - _Requirements: 7.6_

- [ ] 9.2 Run integration test suite
  - Execute all integration tests
  - Verify backtesting integration tests pass
  - Verify live trading integration tests pass
  - Fix any failing tests
  - _Requirements: 8.4, 9.4_

- [ ] 9.3 Run end-to-end validation
  - Run complete backtests with all strategies
  - Verify results are reasonable
  - Test live trading in paper mode
  - _Requirements: 8.4, 9.4_

- [ ] 10. Final cleanup and validation
  - Search codebase for any remaining legacy references
  - Verify no imports of deleted modules
  - Verify no calls to legacy methods
  - Update any remaining documentation
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [ ] 10.1 Search for legacy references
  - Search for "BaseStrategy" in codebase
  - Search for "calculate_indicators" in codebase
  - Search for "check_entry_conditions" in codebase
  - Search for "check_exit_conditions" in codebase
  - Search for "LegacyStrategyAdapter" in codebase
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2_

- [ ] 10.2 Verify no imports of deleted modules
  - Check for imports from `src.strategies.base`
  - Check for imports from `src.strategies.adapters`
  - Check for imports from `src.strategies.migration`
  - Fix any remaining imports
  - _Requirements: 1.1, 2.1, 3.1_

- [ ] 10.3 Verify no calls to legacy methods
  - Search for calls to legacy interface methods
  - Verify all code uses `process_candle()`
  - Verify all code uses `TradingDecision` objects
  - _Requirements: 1.2, 1.3, 1.4_

- [ ] 10.4 Final documentation review
  - Review all documentation for accuracy
  - Ensure component-based architecture is well documented
  - Remove any remaining migration references
  - _Requirements: 3.3, 3.4, 10.4, 10.5, 10.6_
