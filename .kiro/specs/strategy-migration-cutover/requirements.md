# Strategy System Migration - Complete Cutover Requirements

## Introduction

The strategy system has been partially migrated to a component-based architecture. The component system is fully implemented with `Strategy`, `SignalGenerator`, `RiskManager`, and `PositionSizer` classes. However, the codebase still maintains the legacy `BaseStrategy` system alongside adapters and migration utilities, creating significant technical debt.

This specification defines the requirements for **completing the migration** by removing all legacy code, adapters, and migration utilities, leaving only the clean component-based system.

## Current State

### What Exists (Component-Based System)
- ✅ `Strategy` class that composes components via `process_candle()`
- ✅ `SignalGenerator` implementations (ML, Technical, Ensemble)
- ✅ `RiskManager` implementations (Fixed, Volatility-based, Regime-adaptive)
- ✅ `PositionSizer` implementations (Fixed fraction, Kelly, Confidence-weighted)
- ✅ `TradingDecision` dataclass with rich context
- ✅ `RegimeContext` and `EnhancedRegimeDetector`

### What Needs Removal (Legacy System)
- ❌ `BaseStrategy` abstract class with legacy interface methods
- ❌ `LegacyStrategyAdapter` (563 lines) - bridges component strategies to legacy interface
- ❌ `adapter_factory.py` (573 lines) - creates adapters
- ❌ Migration utilities directory (6,666 lines) - conversion, validation, cross-validation tools
- ❌ Concrete strategy implementations extending `BaseStrategy` (ml_basic.py, ml_adaptive.py, etc.)
- ❌ Backtesting engine code that calls `calculate_indicators()`, `check_entry_conditions()`, etc.
- ❌ Live trading engine code that uses legacy interface
- ❌ Tests that depend on legacy interface

## Requirements

### Requirement 1: Remove Legacy BaseStrategy Interface

**User Story:** As a developer, I want to remove the legacy `BaseStrategy` abstract class and all its interface methods, so that there is only one way to implement strategies.

#### Acceptance Criteria

1. WHEN the migration is complete THEN the `BaseStrategy` class SHALL be deleted
2. WHEN the migration is complete THEN no code SHALL reference `calculate_indicators()`, `check_entry_conditions()`, `check_exit_conditions()`, `calculate_position_size()`, or `calculate_stop_loss()` methods
3. WHEN the migration is complete THEN all strategies SHALL be implemented using the component-based `Strategy` class
4. WHEN the migration is complete THEN no strategy SHALL extend `BaseStrategy`

### Requirement 2: Remove Adapter Layer

**User Story:** As a developer, I want to remove all adapter code that bridges between legacy and component systems, so that the codebase is simpler and more maintainable.

#### Acceptance Criteria

1. WHEN the migration is complete THEN the `src/strategies/adapters/` directory SHALL be deleted
2. WHEN the migration is complete THEN no code SHALL reference `LegacyStrategyAdapter`
3. WHEN the migration is complete THEN no code SHALL reference `adapter_factory`
4. WHEN the migration is complete THEN all adapter tests SHALL be deleted

### Requirement 3: Remove Migration Utilities

**User Story:** As a developer, I want to remove all migration utility code that was used during the transition period, so that the codebase contains only production code.

#### Acceptance Criteria

1. WHEN the migration is complete THEN the `src/strategies/migration/` directory SHALL be deleted
2. WHEN the migration is complete THEN no code SHALL reference migration utilities
3. WHEN the migration is complete THEN the `MIGRATION.md` file SHALL be deleted or archived
4. WHEN the migration is complete THEN migration-related documentation SHALL be removed from README files

### Requirement 4: Update Backtesting Engine

**User Story:** As a developer, I want the backtesting engine to use only the component-based strategy interface, so that backtests work with `TradingDecision` objects.

#### Acceptance Criteria

1. WHEN running a backtest THEN the engine SHALL call `strategy.process_candle()` for each candle
2. WHEN running a backtest THEN the engine SHALL NOT call `calculate_indicators()` upfront
3. WHEN running a backtest THEN the engine SHALL use `TradingDecision.signal.direction` to determine entry/exit
4. WHEN running a backtest THEN the engine SHALL use `TradingDecision.position_size` for position sizing
5. WHEN running a backtest THEN the engine SHALL use `TradingDecision.risk_metrics` for stop loss information
6. WHEN running a backtest THEN regime switching SHALL work with component strategies

### Requirement 5: Update Live Trading Engine

**User Story:** As a developer, I want the live trading engine to use only the component-based strategy interface, so that live trading works with `TradingDecision` objects.

#### Acceptance Criteria

1. WHEN running live trading THEN the engine SHALL call `strategy.process_candle()` for each candle
2. WHEN running live trading THEN the engine SHALL NOT call `calculate_indicators()` upfront
3. WHEN running live trading THEN the engine SHALL use `TradingDecision.signal.direction` to determine entry/exit
4. WHEN running live trading THEN the engine SHALL use `TradingDecision.position_size` for position sizing
5. WHEN running live trading THEN strategy hot-swapping SHALL work with component strategies
6. WHEN running live trading THEN database logging SHALL capture `TradingDecision` data

### Requirement 6: Convert Concrete Strategy Implementations

**User Story:** As a developer, I want all concrete strategy implementations (ml_basic, ml_adaptive, etc.) to be converted to component-based strategies, so that they use the new architecture.

#### Acceptance Criteria

1. WHEN the migration is complete THEN ml_basic SHALL be a component-based strategy
2. WHEN the migration is complete THEN ml_adaptive SHALL be a component-based strategy
3. WHEN the migration is complete THEN ml_sentiment SHALL be a component-based strategy
4. WHEN the migration is complete THEN ensemble_weighted SHALL be a component-based strategy
5. WHEN the migration is complete THEN momentum_leverage SHALL be a component-based strategy
6. WHEN the migration is complete THEN all strategies SHALL be instantiated using `Strategy` class with composed components

### Requirement 7: Update Strategy Tests

**User Story:** As a developer, I want all strategy tests to work with the component-based interface, so that tests validate the new architecture.

#### Acceptance Criteria

1. WHEN running tests THEN no test SHALL call `calculate_indicators()`
2. WHEN running tests THEN no test SHALL call `check_entry_conditions()` or `check_exit_conditions()`
3. WHEN running tests THEN tests SHALL call `process_candle()` and validate `TradingDecision` objects
4. WHEN running tests THEN component tests SHALL test individual components in isolation
5. WHEN running tests THEN integration tests SHALL test complete strategy workflows
6. WHEN running tests THEN all tests SHALL pass with the component-based system

### Requirement 8: Update Backtesting Tests

**User Story:** As a developer, I want backtesting integration tests to work with component-based strategies, so that backtesting is validated with the new architecture.

#### Acceptance Criteria

1. WHEN running backtesting tests THEN tests SHALL use component-based strategies
2. WHEN running backtesting tests THEN tests SHALL validate `TradingDecision` objects
3. WHEN running backtesting tests THEN tests SHALL NOT reference legacy interface methods
4. WHEN running backtesting tests THEN all backtesting integration tests SHALL pass

### Requirement 9: Update Live Trading Tests

**User Story:** As a developer, I want live trading integration tests to work with component-based strategies, so that live trading is validated with the new architecture.

#### Acceptance Criteria

1. WHEN running live trading tests THEN tests SHALL use component-based strategies
2. WHEN running live trading tests THEN tests SHALL validate `TradingDecision` objects
3. WHEN running live trading tests THEN tests SHALL NOT reference legacy interface methods
4. WHEN running live trading tests THEN all live trading integration tests SHALL pass

### Requirement 10: Clean Up Documentation and Comments

**User Story:** As a developer, I want all documentation and comments to reflect the component-based architecture only, so that there is no confusion about which system to use.

#### Acceptance Criteria

1. WHEN the migration is complete THEN no comment SHALL refer to "legacy" strategies
2. WHEN the migration is complete THEN no comment SHALL refer to "componentised" or "new" strategies
3. WHEN the migration is complete THEN no comment SHALL refer to "migration" or "adapter"
4. WHEN the migration is complete THEN documentation SHALL describe only the component-based architecture
5. WHEN the migration is complete THEN README files SHALL be updated to remove legacy references
6. WHEN the migration is complete THEN there SHALL be only one way to create strategies (component-based)
