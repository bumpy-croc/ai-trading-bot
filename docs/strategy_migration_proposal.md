# Strategy Migration Completion Proposal

## Background

Recent work introduced a component-oriented `Strategy` class that produces a `TradingDecision` per candle by orchestrating signal generation, risk management, position sizing, and regime analysis components.【F:src/strategies/components/strategy.py†L1-L200】 The interim `LegacyStrategyAdapter` currently wraps these components behind the old `BaseStrategy` interface so that backtesting and live trading engines can continue to call `calculate_indicators`, `check_entry_conditions`, and related hooks.【F:src/strategies/adapters/legacy_adapter.py†L1-L200】 The long-term design documented in the strategy system redesign specification expects all runtime engines and tests to consume the component API directly.【F:.kiro/specs/strategy-system-redesign/design.md†L1-L200】 This proposal describes the remaining changes required to finish that migration while preserving existing behaviour and runtime performance.

## Goals and Constraints

1. **Unify on `TradingDecision`** – Backtesting, live trading, and tests should request decisions from componentised strategies instead of calling legacy hook methods.
2. **Maintain behavioural parity** – Results produced by the engines must remain equivalent to the current legacy path for each supported strategy during the migration.
3. **Preserve backtest throughput** – Bulk indicator calculation must stay vectorised so that processing remains comparable to the existing pre-computation model.
4. **Facilitate incremental rollout** – Legacy strategies should continue to function (via adapters) until they are ported, but new code paths must be default for component strategies.

## Proposed Architecture Adjustments

### 1. Strategy Runtime Contract

* Introduce a `ComponentStrategyRuntime` (CSR) wrapper responsible for orchestrating strategy preparation and candle-by-candle execution. CSR exposes:
  * `prepare_data(df: pd.DataFrame) -> StrategyDataset` – resolves any feature requirements before iteration.
  * `process(index: int, context: RuntimeContext) -> TradingDecision` – delegates to `Strategy.process_candle` with the prepared dataset.
  * `finalize()` – records post-run metrics and emits audit information.
* Update `Strategy` components with optional hooks:
  * `warmup_period` property – declares minimum history length required before reliable decisions.
  * `get_feature_generators()` – returns callables describing vectorised feature construction (see Section 2).
  * `prepare_runtime(dataset)` – optional state initialisation for live/streaming use.

Legacy component strategies can implement these hooks gradually. CSR will supply defaults that maintain current behaviour if a hook is absent.

### 2. Feature Preparation and Caching

To keep backtesting fast, indicator work must remain vectorised. Introduce a reusable feature pipeline consisting of:

* **Feature generator contracts** – Each signal generator and risk manager declares the columns it needs and provides a vectorised callable (`FeatureGenerator`) that takes a `DataFrame` slice and returns the derived columns. Feature generators run once during `prepare_data` to extend the shared dataset.
* **Feature cache descriptors** – CSR builds a `StrategyDataset` containing the enriched `DataFrame`, metadata about warmup length, and per-component caches (e.g., ML model predictions) to avoid re-computation.
* **Incremental updaters** – For live trading, each feature generator can optionally provide an `update` method to compute the next row based on the previous cache, allowing O(1) incremental updates without re-running vectorised code on the full history.

This approach reproduces the "precompute all indicators" performance profile during backtests while still serving components that expect streaming data.

### 3. Engine Integration

* **Backtesting** – Refactor `Backtester` to accept either `BaseStrategy` implementations or a CSR instance. When a raw `Strategy` (component) is supplied, the engine will:
  1. Call `prepare_data` once to enrich the market data.
  2. Iterate through the dataset using the warmup offset, requesting `TradingDecision`s and translating them into trade operations (entries/exits, stop loss, trailing stop checks) mirroring the legacy logic.
  3. Use CSR-provided metadata to maintain behaviour (e.g., risk metrics, stop-loss suggestions) so that PnL calculations and logging remain unchanged.
* **Live Trading** – Mirror the same runtime contract in the live engine: maintain a `RuntimeContext` containing the rolling feature cache, open positions, and balance; request decisions per new candle; translate them into exchange orders using the existing execution stack. The adapter layer should continue to expose the old interface temporarily for any remaining legacy strategies.
* **Adapter Simplification** – Once engines use CSR, `LegacyStrategyAdapter` can be simplified to forward `BaseStrategy` hooks into CSR for backwards compatibility. Eventually, when no code requires the old hooks, the adapter can be removed entirely.

### 4. Testing Strategy Alignment

* Provide shared fixtures that build a CSR-backed strategy and deliver `TradingDecision`s to both unit tests and regression suites.
* Convert existing strategy-specific tests to validate decision outputs (signal direction, size, stop-loss metadata) instead of indirectly asserting on side effects of `calculate_indicators` or `check_entry_conditions`.
* Extend regression tooling in `src/strategies/migration` to compare legacy engine runs with CSR runs over the same historical data, ensuring behaviour parity within statistical tolerance.

## Implementation Plan

1. **Foundation (Sprint 1)**
   * Add CSR runtime, feature generator contracts, and dataset abstractions.
   * Implement default feature generators for existing component strategies by lifting logic currently embedded in `calculate_indicators` or helper utilities.
   * Provide incremental updater scaffolding (no-op default) to avoid breaking live trading.

2. **Engine Refactor (Sprint 2)**
   * Update backtester to detect component strategies, invoke CSR, and translate `TradingDecision`s into trade events while preserving risk, partial fills, and trailing stop logic.
   * Introduce integration tests that replay a fixed dataset through both legacy and CSR paths and assert identical trade sequences.
   * Update live engine service layer to consume CSR decisions for component strategies, falling back to legacy behaviour when necessary.

3. **Testing Migration (Sprint 3)**
   * Refactor unit and integration tests to operate on `TradingDecision`s.
   * Extend regression testing utilities in `src/strategies/migration` to snapshot component decisions and compare against legacy results, guarding against behavioural drift.
   * Update CI workflows to exercise both backtesting and live-decision simulations through the new runtime, ensuring parity.

4. **Cleanup (Sprint 4)**
   * Remove temporary compatibility code once all in-repo strategies are componentised.
   * Deprecate `BaseStrategy` hooks and simplify adapters.
   * Document the new workflow in `src/strategies` README and migration guides.

## Risk Mitigation

* **Performance regressions** – Benchmark backtests before and after CSR integration using representative datasets; profile feature generation to ensure vectorised operations remain dominant.
* **Behavioural drift** – Use regression tests and audit trail tooling to compare trade sequences and risk metrics for each strategy pre- and post-migration.
* **Live trading stability** – Roll out CSR in shadow mode first: run CSR decisions alongside legacy decisions in the live engine without executing trades, compare outputs, then promote once parity is verified.

## Expected Outcomes

Completing this plan will align backtesting, live trading, and tests with the component-based strategy architecture, maintain existing performance characteristics, and prepare the codebase for future strategy innovation consistent with the redesign goals.【F:.kiro/specs/strategy-system-redesign/design.md†L1-L200】【F:src/strategies/components/strategy.py†L1-L200】 The migration removes the adapter bottleneck, ensures all environments operate on the same `TradingDecision` contract, and makes feature computation both explicit and optimised for bulk processing.
