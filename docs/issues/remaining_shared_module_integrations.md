# GitHub Issue: Remaining Shared Module Integrations

**Title:** Complete integration of remaining shared engine modules

**Labels:** `enhancement`, `refactoring`, `engines`

---

## Summary

During Issue #454 (Extract Shared Logic Between Engines), several shared modules were created but integration was deferred due to complexity. This issue tracks the remaining work.

## Modules Requiring Integration

### 1. Shared Models (`src/engines/shared/models.py`)

**Status:** Available but not integrated

**Challenge:**
- Backtest uses `ActiveTrade`, `Trade` from `src.engines.backtest.models`
- Live uses `LivePosition`, `PositionSide` from position_tracker
- 10+ files would need import changes
- Different attribute names and behaviors

**Files Affected:**
- `src/engines/backtest/engine.py`
- `src/engines/backtest/execution/entry_handler.py`
- `src/engines/backtest/execution/execution_engine.py`
- `src/engines/backtest/execution/exit_handler.py`
- `src/engines/backtest/execution/position_tracker.py`
- `src/engines/backtest/logging/event_logger.py`
- `src/engines/live/execution/execution_engine.py`
- `src/engines/live/health/health_monitor.py`
- `src/engines/live/logging/event_logger.py`

**Suggested Approach:**
1. Define common protocol/interface for Position and Trade
2. Have existing models implement the protocol
3. Gradually migrate to shared implementations

**Risk Level:** High (affects core data structures)

---

## Completed Integrations (Reference)

These modules were successfully integrated:

| Module | Integration Commit | Reference |
|--------|-------------------|-----------|
| `TrailingStopManager` | Integrated into both exit handlers | Issue #454 |
| `PolicyHydrator` | Replaces `_apply_policies_from_decision` | Issue #454 |
| `RiskConfiguration` | Replaces `_merge_dynamic_risk_config`, `_build_trailing_policy` | Issue #454 |
| `DynamicRiskHandler` | Integrated into both entry handlers | Issue #454 |
| `CostCalculator` | Unified fee/slippage calculation in both engines | PR #466 (2025-12-26) |
| `PerformanceTracker` | Moved to `src/performance/`, integrated into both engines with 30+ metrics | `docs/execplans/performance_tracker_integration.md` (2025-12-26) |

## Testing Requirements

For each integration:
1. Run unit tests: `python -m pytest tests/unit/ -v`
2. Run integration tests: `python -m pytest tests/integration/ -v`
3. Run backtest: `atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30`
4. Test live engine (paper): `atb live ml_basic --symbol BTCUSDT --paper-trading`

## Related

- Parent Issue: #454 (Extract shared logic between engines)
- PartialOperationsManager Issue: `docs/issues/partial_operations_manager_integration.md`
- Branch: `claude/extract-shared-engine-logic-vh4lQ`
