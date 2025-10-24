# Component Risk Integration Guidance

This note summarises recommended architectural adjustments for aligning `position_management`, `risk_manager`, and `strategy_manager` across the trading engine and strategy components.

## Keep the Core Modules Authoritative

`src/position_management`, `src/risk/risk_manager.py`, and the live `StrategyManager` encapsulate portfolio-wide context: aggregate exposure tracking, correlation limits, and position lifecycle controls. They maintain state that spans strategies and sessions, so relocating them wholly into component code would fragment canonical risk data. Retain these modules as the source of truth while making their capabilities consumable elsewhere.

## Surface Engine Policies to Components

### 1. Introduce a Core Risk Adapter

- **Module**: `src/strategies/components/risk_adapter.py`
- **Purpose**: Wrap `src/risk/risk_manager.RiskManager` and expose the existing component contract (`calculate_position_size`, `should_exit`, `get_stop_loss`, `get_take_profit`).
- **Implementation steps**:
  1. Inject the engine `RiskManager` plus portfolio context (`PositionTracker`, daily exposure state) into the adapter constructor so the adapter works in simulations and live trading.
  2. Delegate calculations directly to the engine manager, passing through instrument identifiers and risk parameters supplied by the strategy component.
  3. Propagate any exceptions or risk-limit breaches (e.g., daily loss limits) so components can surface them in diagnostics instead of silently diverging.
  4. Export the adapter from `src/strategies/components/__init__.py` and update factories (e.g., `src/strategies/ml_basic.py`, `src/strategies/ensemble.py`) to instantiate it.

### 2. Emit Policy Descriptors from Components

- **Data classes**: Create serialisable descriptors in `src/strategies/components/policies.py` that mirror engine policies (`PartialExitPolicy`, `TrailingStopPolicy`, dynamic risk scalers in `src/position_management`).
- **Strategy integration**: Extend `TradingDecision` (or its equivalent DTO) with an optional `policies` field containing the descriptors. Teach component strategies to request the policies from the adapter and attach them to emitted decisions.
- **Engine consumption**: Update `src/live/trading_engine.py` and the backtester to recognise the descriptors. When present, the engine should hydrate full policy objects from the descriptors instead of recalculating them, guaranteeing consistency between environments.

### 3. Share Portfolio State Across Contexts

- Provide the adapter with read/write access to the same position ledger (`src/position_management/position_tracker.py`) that the engine uses. For backtests, inject a simulated ledger; for live trading, inject the real one.
- Mirror correlation and exposure caches by sharing the risk manager's stateful collaborators (e.g., `ExposureMonitor`, `CorrelationMatrix`). Avoid duplicating these structures in component land—always pass the canonical instances.
- Expose hooks (such as `on_fill` or `on_position_closed`) on the adapter so component harnesses can notify the engine modules about simulated fills, keeping drawdown tracking aligned.

## Clarify Strategy Management Responsibilities

### Rename and Align Managers

1. Rename `src/strategies/components/strategy_manager.py:StrategyManager` to `ComponentStrategyManager`, updating imports across `src/strategies/components/__init__.py`, tests, and any factory modules.
2. Extract shared versioning helpers (e.g., semantic version comparison, component registry caching) into `src/strategies/management/versioning.py`.
3. Update `src/live/strategy_manager.py` to import the helpers instead of re-implementing them. Where behaviour differs (such as hot-swap orchestration), document the divergence in module-level docstrings to avoid future confusion.

### Provide a Consistent Adapter Wiring Path

- The renamed component manager should accept adapters (risk, execution, data) as constructor dependencies. When creating strategy instances, it should inject the `CoreRiskAdapter` so every component has a consistent risk surface.
- The live manager should likewise receive adapter instances (or factories) so that hot-swapped strategies inherit the same wiring. This also makes it easier to reuse the managers in integration tests.

### Documentation and Migration Notes

- Add module docstrings describing the division of responsibilities and the shared helpers.
- Update any developer onboarding docs referencing the old `StrategyManager` name.
- Provide a short migration checklist in `docs/architecture/strategy_management.md` (or similar) so teams know which imports to update and how to plug the new helpers into custom strategies.

## Testing Implications

1. **Unit coverage** – Add focused tests under `tests/strategies/components/test_risk_adapter.py` that mock `RiskManager` responses and assert that the adapter forwards calls, propagates exceptions, and preserves policy descriptors.
2. **Integration scenarios** – Extend `tests/integration/live/test_trading_engine.py` (or create a new file) to execute a full decision lifecycle where components emit policy descriptors and the engine enforces them. Include regression cases for trailing stops and partial exits.
3. **Backtest parity** – Add a regression test in `tests/integration/backtesting` ensuring backtest results match live-engine calculations when the adapter is in place.
4. **Automation** – Update CI pipelines or local scripts to run `make code-quality` and `pytest tests/strategies -k risk_adapter` during development. For full verification, run `pytest tests/integration/live -k policy_descriptor` before release.

Document the expected command sequence in team runbooks so engineers know how to validate risk-control changes consistently.

By exposing engine-level risk and position management through adapters and shared utilities, the architecture preserves a single source of truth while empowering components to simulate and reason about the same controls they encounter in live trading.
