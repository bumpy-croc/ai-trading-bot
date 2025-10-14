# Strategy Migration Completion Proposal

## Background

Recent work introduced a component-oriented `Strategy` class that produces a `TradingDecision` per candle by orchestrating signal generation, risk management, position sizing, and regime analysis components.【F:src/strategies/components/strategy.py†L1-L200】 The interim `LegacyStrategyAdapter` currently wraps these components behind the old `BaseStrategy` interface so that backtesting and live trading engines can continue to call `calculate_indicators`, `check_entry_conditions`, and related hooks.【F:src/strategies/adapters/legacy_adapter.py†L1-L200】 The long-term design documented in the strategy system redesign specification expects all runtime engines and tests to consume the component API directly.【F:.kiro/specs/strategy-system-redesign/design.md†L1-L200】 This proposal describes the remaining changes required to finish that migration while preserving existing behaviour and runtime performance.

## Current State and Contracts

* **Legacy strategy contract** – `BaseStrategy` implementations expose lifecycle hooks such as `calculate_indicators`, `check_entry_conditions`, `check_exit_conditions`, and `update_position` that assume pre-computed indicator columns on a shared `DataFrame`. Engines drive the lifecycle and translate hook outputs into orders.【F:src/strategies/MIGRATION.md†L1-L200】
* **Component strategy contract** – Component strategies encapsulate signal, sizing, and risk components and expose a single `process_candle` that returns a `TradingDecision`. Feature calculation and state management are delegated to the components themselves.【F:src/strategies/components/strategy.py†L1-L200】
* **Bridge layer** – `LegacyStrategyAdapter` adapts component strategies back onto the legacy contract so engines and tests can continue to work while the migration completes.【F:src/strategies/adapters/legacy_adapter.py†L1-L200】

Today, engines, tests, and most tooling still rely on the legacy contract. Completing the migration requires inverting that relationship so the component contract becomes the runtime default and legacy strategies are the only ones using adapters.

## Goals and Constraints

1. **Unify on `TradingDecision`** – Backtesting, live trading, and tests should request decisions from componentised strategies instead of calling legacy hook methods.
2. **Maintain behavioural parity** – Results produced by the engines must remain equivalent to the current legacy path for each supported strategy during the migration.
3. **Preserve backtest throughput** – Bulk indicator calculation must stay vectorised so that processing remains comparable to the existing pre-computation model.
4. **Facilitate incremental rollout** – Legacy strategies should continue to function (via adapters) until they are ported, but new code paths must be default for component strategies.

## Proposed Architecture Adjustments

### 1. Strategy Runtime Contract

* Introduce a `StrategyRuntime` wrapper responsible for orchestrating strategy preparation and candle-by-candle execution. The runtime exposes:
  * `prepare_data(df: pd.DataFrame) -> StrategyDataset` – resolves any feature requirements before iteration.
  * `process(index: int, context: RuntimeContext) -> TradingDecision` – delegates to `Strategy.process_candle` with the prepared dataset.
  * `finalize()` – records post-run metrics and emits audit information.
* Update `Strategy` components with optional hooks:
  * `warmup_period` property – declares minimum history length required before reliable decisions.
  * `get_feature_generators()` – returns callables describing vectorised feature construction (see Section 2).
  * `prepare_runtime(dataset)` – optional state initialisation for live/streaming use.

Legacy component strategies can implement these hooks gradually. The runtime will supply defaults that maintain current behaviour if a hook is absent.

### 2. Feature Preparation and Caching

To keep backtesting fast, indicator work must remain vectorised. Introduce a reusable feature pipeline consisting of:

* **Feature generator contracts** – Each signal generator and risk manager declares the columns it needs and provides a vectorised callable (`FeatureGenerator`) that takes a `DataFrame` slice and returns the derived columns. Feature generators run once during `prepare_data` to extend the shared dataset.
* **Feature cache descriptors** – The runtime builds a `StrategyDataset` containing the enriched `DataFrame`, metadata about warmup length, and per-component caches (e.g., ML model predictions) to avoid re-computation.
* **Incremental updaters** – For live trading, each feature generator can optionally provide an `update` method to compute the next row based on the previous cache, allowing O(1) incremental updates without re-running vectorised code on the full history.

This approach reproduces the "precompute all indicators" performance profile during backtests while still serving components that expect streaming data.

### 3. Engine Integration

* **Backtesting** – Refactor `Backtester` to accept either `BaseStrategy` implementations or a `StrategyRuntime` instance. When a raw `Strategy` (component) is supplied, the engine will:
  1. Call `prepare_data` once to enrich the market data.
  2. Iterate through the dataset using the warmup offset, requesting `TradingDecision`s and translating them into trade operations (entries/exits, stop loss, trailing stop checks) mirroring the legacy logic.
  3. Use runtime-provided metadata to maintain behaviour (e.g., risk metrics, stop-loss suggestions) so that PnL calculations and logging remain unchanged.
* **Live Trading** – Mirror the same runtime contract in the live engine: maintain a `RuntimeContext` containing the rolling feature cache, open positions, and balance; request decisions per new candle; translate them into exchange orders using the existing execution stack. The adapter layer should continue to expose the old interface temporarily for any remaining legacy strategies.
* **Adapter Simplification** – Once engines use the runtime, `LegacyStrategyAdapter` can be simplified to forward `BaseStrategy` hooks into the runtime for backwards compatibility. Eventually, when no code requires the old hooks, the adapter can be removed entirely.

### 4. Testing Strategy Alignment

* Provide shared fixtures that build a runtime-backed strategy and deliver `TradingDecision`s to both unit tests and regression suites.
* Convert existing strategy-specific tests to validate decision outputs (signal direction, size, stop-loss metadata) instead of indirectly asserting on side effects of `calculate_indicators` or `check_entry_conditions`.
* Extend regression tooling in `src/strategies/migration` to compare legacy engine runs with runtime-driven runs over the same historical data, ensuring behaviour parity within statistical tolerance.

## Extensibility for Future Strategy Work

The migration should accommodate new strategy ideas without additional framework churn:

* **New strategies and ML models** – Strategy authors declare component dependencies and feature generators. Models can register themselves as feature generators that batch-predict during `prepare_data` and provide incremental update hooks for live use. The runtime keeps the prediction cache alongside other derived columns.
* **Position and risk management** – Position sizing, risk controls, and stop systems integrate as components that emit structured instructions in `TradingDecision`. Centralising these responsibilities inside the runtime keeps engines agnostic to strategy-specific logic while allowing new modules (e.g., volatility targeting) to plug in.
* **Feature engineering adjustments** – Feature generators are isolated, versioned callables so iterating on engineering pipelines only requires updating the component and its generator metadata. Backtests automatically receive the new features through the runtime without bespoke wiring.

## Benchmarking and Artifact Capture

Before modifying runtime code, run representative backtests (including CPU and wall-clock timing) through the existing legacy pathway. Store artefacts—logs, performance summaries, profiling traces—in the repository's `artifacts/strategy-migration/` directory (or equivalent configured location). Repeat the same benchmark suite after each major milestone to confirm parity and surface regressions early.

## Implementation Plan

Each phase below includes context, architectural focus, and concrete deliverables so individual agents can work independently while staying aligned.

### Phase 0 – Baseline Benchmarking

* **Context** – Engines currently depend on legacy hooks; we need hard data on performance and behaviour before refactoring.
* **Architecture Notes** – Exercise both backtester and live-simulation pathways using the legacy contract to capture current throughput and trade sequences.
* **Deliverables**
  * Benchmark scripts and documented commands for replaying representative datasets.
  * Stored artefacts (timing, CPU, trade logs, decision traces) under `artifacts/strategy-migration/baseline/`.
  * Summary report comparing strategies and datasets to serve as regression targets.

### Phase 1 – Runtime Foundations

* **Context** – Establish the `StrategyRuntime` orchestration layer and feature pipeline while keeping existing strategies functional.
* **Architecture Notes** – Implement runtime scaffolding, feature generator contracts, and dataset abstractions that sit between engines and component strategies without altering trade semantics.
* **Deliverables**
  * `StrategyRuntime` class with `prepare_data`, `process`, and `finalize` methods plus default behaviours.
  * Feature generator interface supporting vectorised batch execution and optional incremental updates.
  * Updates to component strategies declaring `warmup_period`, feature requirements, and any runtime initialisation hooks.
  * Unit tests covering runtime lifecycle and feature caching, alongside updated documentation outlining runtime responsibilities.

### Phase 2 – Engine Integration

* **Context** – Backtesting and live engines must consume `TradingDecision`s directly while retaining parity with legacy behaviour.
* **Architecture Notes** – Introduce runtime-aware execution paths in both engines, keeping the legacy adapter available for un-migrated strategies.
* **Deliverables**
  * Backtester changes invoking `StrategyRuntime` when supplied with component strategies, including translation of decisions into orders, fills, and portfolio updates mirroring legacy logic.
  * Live trading engine modifications maintaining a shared `RuntimeContext`, executing decisions, and falling back to legacy hooks when required.
  * Documentation and guardrails ensuring short entries are only executed when component strategies explicitly opt in via `decision.metadata["enter_short"]`, preventing unintended sell-to-short conversions.
  * Side-by-side regression harness that replays datasets through both engines (legacy vs runtime) and asserts identical trade sequences, PnL, and risk metrics.
  * Benchmark artefacts demonstrating that backtest throughput remains within agreed bounds compared with Phase 0.

### Phase 3 – Test Suite Migration

* **Context** – Unit, integration, and regression tests still target legacy hooks; they must validate `TradingDecision` semantics instead.
* **Architecture Notes** – Provide shared fixtures and helpers that exercise strategies through `StrategyRuntime`, enabling consistent assertions across suites.
* **Deliverables**
  * Updated unit and integration tests asserting on decision payloads, component interactions, and risk instructions.
  * Regression tooling in `src/strategies/migration` capable of snapshotting decisions and comparing them with legacy baselines.
  * CI workflow updates running runtime-based backtests and decision diffing as part of pull requests.
  * Artefacts capturing test run timings to ensure coverage remains practical.

### Phase 4 – Cleanup and Decommissioning

* **Context** – After engines and tests fully adopt the runtime, remaining legacy-only pathways can be retired.
* **Architecture Notes** – Simplify the codebase so the component contract is the sole public strategy interface.
* **Deliverables**
  * Removal of redundant compatibility layers and deprecation of `BaseStrategy` hooks.
  * Simplified `LegacyStrategyAdapter` (or its removal) with clear upgrade guidance for any downstream consumers.
  * Documentation updates in `src/strategies/README.md` and `MIGRATION.md` describing the final architecture and onboarding flow for new strategies.
  * Final benchmark artefacts demonstrating parity or improvement relative to the baseline.

## Risk Mitigation

* **Performance regressions** – Benchmark backtests before and after runtime integration using representative datasets; profile feature generation to ensure vectorised operations remain dominant.
* **Behavioural drift** – Use regression tests and audit trail tooling to compare trade sequences and risk metrics for each strategy pre- and post-migration.
* **Live trading stability** – Roll out the runtime in shadow mode first: run runtime decisions alongside legacy decisions in the live engine without executing trades, compare outputs, then promote once parity is verified.

## Expected Outcomes

Completing this plan will align backtesting, live trading, and tests with the component-based strategy architecture, maintain existing performance characteristics, and prepare the codebase for future strategy innovation consistent with the redesign goals.【F:.kiro/specs/strategy-system-redesign/design.md†L1-L200】【F:src/strategies/components/strategy.py†L1-L200】 The migration removes the adapter bottleneck, ensures all environments operate on the same `TradingDecision` contract, and makes feature computation both explicit and optimised for bulk processing.
