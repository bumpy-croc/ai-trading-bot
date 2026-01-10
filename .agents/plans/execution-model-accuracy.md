# Add Order-Type Aware Execution Modeling for Live and Backtest

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must be kept up to date as work proceeds.

This plan must be maintained in accordance with `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

This work makes trade execution modeling more faithful to real trading, especially for take-profit and stop orders, by introducing shared order-type aware execution logic that both live and backtest engines use. After this change, the default fill policy is conservative for OHLC data: take-profit limit orders fill at the limit price when the bar crosses them, no price improvement is allowed without quote data, and stop orders behave like stop-market orders with adverse slippage and taker fees. Live trading can still reconcile with actual exchange fills when available. You can see it working by running the new unit tests that encode these rules and by running a backtest with the configured execution fill policy that produces deterministic fill prices.

## Progress

- [x] (2025-12-29 22:42Z) Surveyed existing live/backtest execution flow, exit handling, and exchange interface capabilities.
- [x] (2025-12-29 22:53Z) Define shared execution model types, fill policies, and interfaces in `src/engines/shared/execution/`.
- [x] (2025-12-29 22:53Z) Integrate shared execution model into backtest entry/exit flow and update cost modeling for order type.
- [x] (2025-12-30 12:35Z) Integrate shared execution model into live entry/exit flow and reconcile with exchange fills.
- [x] (2025-12-30 12:35Z) Add unit tests, update docs, and run validation commands.

## Surprises & Discoveries

- Observation: A merge conflict in `src/engines/live/execution/exit_handler.py` surfaced after pulling develop and was resolved by keeping the shared execution-model fill decision logic.
  Evidence: conflict markers removed and the execution-model decision flow restored in `execute_exit`.

- Observation: Backtest entry processing needed the candle threaded through `_process_entry_signal` after adding execution-model snapshots.
  Evidence: unit tests failed with `NameError: name 'candle' is not defined` until `candle` was passed into `_process_entry_signal`.

## Decision Log

- Decision: Introduce a shared execution model layer that is called by both backtest and live handlers, rather than duplicating logic per engine.
  Rationale: The current architecture already splits entry/exit handlers and a cost calculator; a shared layer lets us keep parity between live and backtest while keeping cost and exchange interactions in the existing execution engines.
  Date/Author: 2025-12-29 22:42Z / Codex

- Decision: Reuse `OrderSide` and `OrderType` from `src/data_providers/exchange_interface.py` and extend cost modeling to accept order type or liquidity hints.
  Rationale: Reusing existing enums avoids duplicate concepts and keeps live exchange mapping straightforward; cost modeling needs to account for maker versus taker behavior to be realistic.
  Date/Author: 2025-12-29 22:42Z / Codex

- Decision: Keep `use_high_low_for_stops` as the trigger detection toggle, but move fill price selection into the shared execution model.
  Rationale: Triggering and filling are different concerns; separating them makes it possible to refine fills without changing exit detection rules.
  Date/Author: 2025-12-29 22:42Z / Codex

- Decision: Default the execution fill policy to `ohlc_conservative`, which fills limit orders at the limit price when crossed, disallows price improvement without quote data, and treats stop orders as stop-market with adverse slippage and taker fees.
  Rationale: Conservative OHLC assumptions avoid overstated performance and align with common professional backtesting defaults when higher-fidelity data is unavailable.
  Date/Author: 2025-12-29 22:53Z / Codex

- Decision: Expose the execution fill policy via configuration only, without adding CLI flags.
  Rationale: Keeps the interface stable and matches the requested configuration-only approach.
  Date/Author: 2025-12-29 22:53Z / Codex

- Decision: Use `EXECUTION_FILL_POLICY` as the configuration key and `DEFAULT_EXECUTION_FILL_POLICY` as the default constant value name.
  Rationale: The key is explicit and matches other env-style config entries, while the constant mirrors existing default naming patterns.
  Date/Author: 2025-12-29 22:53Z / Codex

## Outcomes & Retrospective

Shared execution modeling now drives backtest and live fill decisions, with maker/taker cost handling and live reconciliation when exchange fills are available. Unit tests cover limit/stop fill behavior and maker/taker cost rules, and configuration docs include `EXECUTION_FILL_POLICY`.

## Context and Orientation

The live trading path is orchestrated by `src/engines/live/trading_engine.py`, which wires `LiveEntryHandler`, `LiveExitHandler`, and `LiveExecutionEngine`. `LiveExitHandler` currently selects a base exit price using candle high/low and passes that to `LiveExecutionEngine`, which applies fees and slippage via `src/engines/shared/cost_calculator.py`. The backtest path is orchestrated by `src/engines/backtest/engine.py`, which uses `src/engines/backtest/execution/EntryHandler`, `ExitHandler`, and `ExecutionEngine`. Backtest exit handling uses candle high/low to trigger exits and sets a base exit price (often the limit price for take profit). Both engines share the same cost calculator but do not currently model order types explicitly.

`src/data_providers/exchange_interface.py` defines `OrderType`, `OrderSide`, and a rich `Order` structure with `average_price`, `filled_quantity`, and `commission`, which can be used to reconcile actual exchange fills in live trading. The data provider interface in `src/data_providers/data_provider.py` only exposes trade price data (OHLCV), so quote-level fills must be optional and only used when an exchange provider or specialized data provider supports them.

The plan introduces a shared execution modeling layer under `src/engines/shared/execution/` that is invoked by both live and backtest handlers. It does not replace the execution engines; instead, it determines fill eligibility and base pricing based on order type and available market data, then existing engines apply costs or reconcile real fills.

## Plan of Work

Milestone 1 establishes the shared execution model layer with explicit data structures and policies. This layer defines what an order intent is, what market data it needs, and how to decide if and at what price a fill occurs under different fidelity settings (OHLC-only, quote-based, or order book based). The output is a deterministic `ExecutionDecision` that backtest and live engines can consume. A small adapter constructs a `MarketSnapshot` from available data sources. This milestone ends with new modules under `src/engines/shared/execution/` and no behavior changes in live or backtest yet.

Milestone 2 integrates the shared execution model into backtesting. `src/engines/backtest/execution/exit_handler.py` and `src/engines/backtest/execution/entry_handler.py` will build `OrderIntent` objects and call the shared execution model to decide fill pricing, with the `ohlc_conservative` policy as the default. `src/engines/backtest/execution/execution_engine.py` will be updated to accept order type or liquidity hints so that limit orders can be modeled with zero adverse slippage and maker fee rates when configured. This milestone ends with backtests using the new model, and with unit tests that validate TP limit fills do not exceed the limit price when only OHLC is available.

Milestone 3 integrates the shared execution model into live trading. `src/engines/live/execution/entry_handler.py` and `src/engines/live/execution/exit_handler.py` will use the shared execution model to compute base prices and fill expectations, but when `enable_live_trading` is true, `src/engines/live/execution/execution_engine.py` will attempt to reconcile fills using `exchange_interface.get_order` (or recent trades) and override simulated prices, fees, and quantities with actual values when available. This milestone ends with live mode using real fills when possible and retaining deterministic simulation in paper mode.

Milestone 4 adds tests, documentation, and configuration exposure. Tests will validate the shared execution model in isolation and confirm that backtest and live handlers pass order intents correctly. Documentation in `docs/` will explain execution fidelity options, what data is required, and how to interpret results. Configuration values will be exposed through `src/config/constants.py` and `ConfigManager` only (no CLI flags).

Throughout the work, commit after each milestone with an imperative, present-tense message that describes the behavior change.

## Concrete Steps

From `/Users/alex/Sites/ai-trading-bot`, review the current exit handling and cost modeling to confirm touch points and to update references for new modules. Use ripgrep to locate references and avoid touching unrelated conflicts:

    rg -n "execute_exit|calculate_exit_costs|take profit" src/engines

Create the shared execution module directory and add one class per file to satisfy the local coding conventions. For example, create `src/engines/shared/execution/market_snapshot.py`, `order_intent.py`, `execution_decision.py`, `fill_policy.py`, and one fill model class per file.

Wire the shared execution model into backtest entry and exit handlers and update `src/engines/shared/cost_calculator.py` so it can compute costs for limit orders with zero adverse slippage when configured. Run unit tests after these changes:

    python tests/run_tests.py unit

Wire the shared execution model into live entry and exit handlers and update `src/engines/live/execution/execution_engine.py` to reconcile with exchange order fills when live trading is enabled and the exchange returns `average_price` and `commission` data. Run unit tests again:

    python tests/run_tests.py unit

Optionally run a short backtest to confirm that TP fills are capped at the limit price under the configured fill policy and that logs include the selected execution policy:

    atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30 --no-db

## Validation and Acceptance

The work is accepted when the following behavior is observable:

A new unit test for the shared execution model demonstrates that a long take-profit limit order fills at the limit price (not candle high) when only OHLC data is available, and a short take-profit limit order fills at its limit price when candle low crosses it. The test should fail before the change and pass after.

A new unit test for cost modeling demonstrates that limit orders apply zero adverse slippage and use a maker fee rate when configured, while market or stop orders retain the existing adverse slippage behavior and taker fees. The test should fail before the change and pass after.

An additional unit test demonstrates that the `ohlc_conservative` policy never allows price improvement for limit orders without quote data, and that stop orders fill as stop-market orders with adverse slippage when triggered. The test should fail before the change and pass after.

Running `python tests/run_tests.py unit` completes successfully. If an integration test is added for backtest execution flow, `python tests/run_tests.py integration` also completes successfully.

## Idempotence and Recovery

The changes are additive and safe to reapply. If a step fails, rerun the last command after fixing the specific error. If reconciliation with exchange fills is not possible due to missing exchange support, the live execution engine should log a warning and fall back to simulated pricing without breaking existing behavior.

## Artifacts and Notes

Expected test evidence after implementing the shared execution model includes a short excerpt like:

    tests/unit/engines/shared/test_execution_model.py ... PASSED
    tests/unit/engines/shared/test_cost_calculator_order_types.py ... PASSED

Keep the new tests concise and focused on the specific fill rules.

## Interfaces and Dependencies

In `src/engines/shared/execution/market_snapshot.py`, define a `MarketSnapshot` dataclass that captures the minimum market data needed for fill decisions. It should include fields for symbol, timestamp, last price, high, low, close, volume, and optional bid and ask quotes. It should include a doc comment describing how bid and ask are used when available.

In `src/engines/shared/execution/order_intent.py`, define an `OrderIntent` dataclass that represents the desired order. It should include symbol, side (`OrderSide`), order type (`OrderType`), quantity, limit price, stop price, time in force, reduce-only flag, and an exit reason string for traceability. It should include a doc comment describing what constitutes a limit order versus a stop or market order in this system.

In `src/engines/shared/execution/execution_decision.py`, define an `ExecutionDecision` dataclass that captures whether a fill should occur, the fill price, the filled quantity, the liquidity type (maker or taker), and a reason string. It should include a doc comment describing how downstream engines should interpret the decision.

In `src/engines/shared/execution/fill_policy.py`, define a `FillPolicy` class that captures execution fidelity (OHLC-only, quote-based, order-book based), whether price improvement is allowed for limit orders, and a default policy value of `ohlc_conservative`. The OHLC conservative policy should be documented as: limit orders fill at the limit price when high/low crosses; no price improvement without quote data; stop orders are treated as stop-market fills with adverse slippage and taker fees.

In `src/engines/shared/execution/ohlc_fill_model.py`, define an `OhlcFillModel` class with a method `decide_fill(order_intent: OrderIntent, snapshot: MarketSnapshot, policy: FillPolicy) -> ExecutionDecision`. It should treat limit orders as filled at the limit price when high/low crosses the limit and should not improve the price unless the policy explicitly allows it.

In `src/engines/shared/execution/execution_model.py`, define an `ExecutionModel` class that selects the appropriate fill model based on the `FillPolicy` and provides a single method for handlers to call. This class should be a thin coordinator and should not contain fee or slippage math.

Update `src/engines/shared/cost_calculator.py` to accept an optional `order_type` or `liquidity` argument for entry and exit costs, using maker fees and zero adverse slippage for limit orders when configured. Document the behavior in the doc comments and add parameter validation to avoid invalid combinations.

Update `src/engines/backtest/execution/entry_handler.py` and `src/engines/backtest/execution/exit_handler.py` to construct `OrderIntent` values for entries and exits and call the shared execution model to decide fill price. The handlers should pass the resulting base price to `ExecutionEngine` for cost calculation.

Update `src/engines/live/execution/entry_handler.py` and `src/engines/live/execution/exit_handler.py` to construct `OrderIntent` values and call the shared execution model to determine the base price. When `enable_live_trading` is true, `src/engines/live/execution/execution_engine.py` should attempt to retrieve the `Order` from the exchange and use its `average_price`, `commission`, and `filled_quantity` to populate the execution result, logging a warning and falling back to simulated costs when unavailable.

If new configuration is needed to select `FillPolicy`, add it to `src/config/constants.py` (for example, `DEFAULT_EXECUTION_FILL_POLICY = \"ohlc_conservative\"`) and use `ConfigManager` to read `EXECUTION_FILL_POLICY` in `src/engines/backtest/engine.py` and `src/engines/live/trading_engine.py`. Do not add CLI flags. Document the setting in an appropriate file under `docs/`.

Plan Change Note: 2025-12-29 22:42Z initial ExecPlan created to outline shared execution modeling and integration steps.
Plan Change Note: 2025-12-29 22:53Z updated default fill policy to `ohlc_conservative` and clarified configuration-only exposure.
Plan Change Note: 2025-12-29 22:53Z recorded config key naming decision and marked Phase 1 modules complete.
Plan Change Note: 2025-12-29 22:53Z marked backtest integration and cost-model updates complete for Phase 2.
Plan Change Note: 2025-12-29 22:53Z documented backtest candle threading fix in Surprises & Discoveries.
Plan Change Note: 2025-12-30 12:35Z marked live execution integration and test/documentation updates complete.
