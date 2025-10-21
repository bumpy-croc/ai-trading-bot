# Strategy System Migration - Complete Cutover Requirements

## Introduction

The strategy system has been partially migrated to a component-based architecture, but currently maintains both the legacy `BaseStrategy` system and the new component-based system with adapters bridging between them. This creates ~7,229 lines of technical debt including adapters, migration utilities, and duplicate testing infrastructure.

This specification defines the requirements for **completing the migration** by removing all legacy code, adapters, and migration utilities, leaving only the clean component-based system.

## Current System Analysis

The codebase currently has:

1. **Component-Based System** (Target Architecture)
   - `Strategy` class that composes `SignalGenerator`, `RiskManager`, `PositionSizer`
   - `TradingDecision` objects with rich context
   - Clean separation of concerns, testable components

2. **Legacy System** (To Be Removed)
   - `BaseStrategy` abstract class with `calculate_indicators()`, `check_entry_conditions()`, etc.
   - Concrete strategy implementations extending `BaseStrategy`
   - DataFrame pollution with strategy-specific columns

3. **Adapter Layer** (To Be Removed)
   - `LegacyStrategyAdapter` (563 lines) - bridges component strategies to legacy interface
   - `adapter_factory.py` (573 lines) - creates adapters
   - Migration utilities (6,666 lines) - conversion, validation, cross-validation tools

4. **Dual Interface Support** (To Be Removed)
   - Backtesting engine supports both interfaces
   - Live trading engine supports both interfaces
   - Tests written for both interfaces

The goal is to remove items 2, 3, and 4, keeping only the component-based system.


## Requirements

### Requirement 1: Unified Strategy Architecture

**User Story:** As a trading system developer, I want a single, coherent strategy architecture that eliminates redundancy between regime-aware strategies, ensemble strategies, and hot-swapping systems, so that I can focus on strategy logic rather than architectural complexity.


#### Acceptance Criteria

1. WHEN the system is redesigned THEN there SHALL be a single primary strategy execution path

2. WHEN a strategy needs regime awareness THEN it SHALL implement regime logic internally rather than relying on external hot-swapping

3. WHEN multiple strategies need to be combined THEN it SHALL be done through composition patterns rather than separate ensemble classes

4. WHEN the system runs THEN it SHALL execute one primary strategy instance that can internally adapt to market conditions


### Requirement 2: Modular Strategy Components

**User Story:** As a strategy developer, I want to build strategies from reusable, testable components (signal generators, risk managers, position sizers), so that I can experiment with different combinations and test each component independently.


#### Acceptance Criteria

1. WHEN creating a strategy THEN it SHALL be composed of pluggable components for signal generation, risk management, and position sizing

2. WHEN testing a strategy THEN each component SHALL be testable in isolation
3. WHEN a component is updated THEN it SHALL not require changes to other components

4. WHEN regime conditions change THEN individual components SHALL be able to adapt their behavior independently


### Requirement 3: Regime-Aware Component Testing

**User Story:** As a strategy developer, I want to test how my strategy performs in specific market regimes (bull, bear, range, high/low volatility), so that I can optimize each regime's behavior separately and measure regime-specific performance.


#### Acceptance Criteria

1. WHEN backtesting a strategy THEN the system SHALL provide regime-specific performance metrics

2. WHEN testing in a specific regime THEN the system SHALL allow filtering backtests to only that regime period

3. WHEN a strategy switches regime behavior THEN the system SHALL log the regime transition and performance impact

4. WHEN optimizing a strategy THEN the system SHALL allow separate optimization for each regime type


### Requirement 4: Strategy Experimentation Framework

**User Story:** As a trading system operator, I want to easily experiment with new strategies, duplicate successful ones for iteration, and promote the best-performing strategy to production, so that I can continuously improve trading performance.


#### Acceptance Criteria

1. WHEN creating a new strategy THEN the system SHALL provide templates and scaffolding for rapid development

2. WHEN duplicating a strategy THEN the system SHALL create a copy with versioning and lineage tracking

3. WHEN comparing strategies THEN the system SHALL provide side-by-side performance comparisons

4. WHEN promoting a strategy THEN the system SHALL safely transition from experimental to production status


### Requirement 5: Strategy Versioning and Lineage

**User Story:** As a trading system operator, I want to track strategy versions, their performance history, and their evolutionary lineage, so that I can understand which changes improved performance and roll back if needed.


#### Acceptance Criteria

1. WHEN a strategy is modified THEN the system SHALL create a new version with timestamp and change description

2. WHEN strategies are related THEN the system SHALL track parent-child relationships and branching

3. WHEN viewing strategy history THEN the system SHALL show performance evolution over time

4. WHEN rolling back THEN the system SHALL allow reverting to any previous version safely


### Requirement 6: Simplified Strategy Interface

**User Story:** As a strategy developer, I want a clean, intuitive interface for creating strategies that focuses on trading logic rather than infrastructure concerns, so that I can rapidly prototype and test new ideas.


#### Acceptance Criteria

1. WHEN implementing a strategy THEN the developer SHALL only need to define core trading logic methods

2. WHEN the strategy needs regime awareness THEN it SHALL use built-in regime detection utilities

3. WHEN the strategy needs risk management THEN it SHALL use configurable risk management components

4. WHEN the strategy needs position sizing THEN it SHALL use pluggable position sizing algorithms


### Requirement 7: Performance-Based Strategy Selection

**User Story:** As a trading system operator, I want the system to automatically identify and use the best-performing strategy based on recent performance metrics, so that the system continuously optimizes itself without manual intervention.


#### Acceptance Criteria

1. WHEN multiple strategy versions exist THEN the system SHALL track their comparative performance

2. WHEN performance metrics are updated THEN the system SHALL evaluate if a strategy change is warranted

3. WHEN switching strategies THEN the system SHALL do so safely without disrupting active positions

4. WHEN performance degrades THEN the system SHALL automatically revert to a previously successful strategy


### Requirement 8: Component-Level Backtesting

**User Story:** As a strategy developer, I want to backtest individual strategy components (signal generators, risk managers) in isolation, so that I can identify which components contribute most to performance and optimize them independently.


#### Acceptance Criteria

1. WHEN backtesting a signal generator THEN the system SHALL test it independently of risk management and position sizing

2. WHEN backtesting a risk manager THEN the system SHALL test it with standardized signals and position sizes

3. WHEN backtesting position sizing THEN the system SHALL test it with standardized signals and risk parameters

4. WHEN combining components THEN the system SHALL predict combined performance based on individual component performance