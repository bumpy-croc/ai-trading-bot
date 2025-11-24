# Documentation Audit Report - 2025-11-24

**Date**: November 24, 2025  
**Auditor**: AI Agent (Nightly Maintenance)  
**Branch**: `HEAD` (detached)

## Executive Summary

Tightened the nightly documentation set by bringing programmatic examples in line with
the current component architecture, ensuring prediction registry snippets use the real
APIs, refreshing the tests README to reflect the actual runner/marker layout, and adding
a self-contained backtesting walkthrough that works offline.

## Scope

- Main `README.md` spot-check via dependency commands (unchanged)  
- `docs/backtesting.md`  
- `src/position_management/README.md`  
- `src/trading/README.md`  
- `src/prediction/README.md`  
- `tests/README.md`  
- Cross-reference with `docs/README.md`, `docs/prediction.md`, `pytest.ini`

## Findings

### 1. Tests README referenced non-existent files ⚠️

The previous version described scripts such as `tests/performance/automated_performance_monitor.py`
and `tests/performance/test_component_performance_regression.py` that are no longer part of
the repository. The quick-start commands also leaned on a "legacy" BaseStrategy flow that
has been removed. This misled contributors who need to understand how to run or triage tests.

### 2. Component documentation still pointed to `BaseStrategy` ⚠️

`src/trading/README.md` told readers to subclass `BaseStrategy` from `src/strategies/base.py`,
which does not exist in the component architecture. New strategies should compose
`Strategy`, `SignalGenerator`, and `RiskManager` from `src.strategies.components`.

### 3. Prediction registry usage was inaccurate ⚠️

`src/prediction/README.md` suggested calling `PredictionModelRegistry.predict()`, but that
method does not exist; inference happens via the bundle’s `OnnxRunner`. The example also did
not honor the feature schema, making it impossible to reproduce without guesswork.

### 4. Backtesting example required live Binance data ℹ️

`docs/backtesting.md` instantiated `BinanceProvider` + `CachedDataProvider`. In environments
without API keys (or the `python-binance` dependency) the snippet fails, so nightly checks
could not prove that the walkthrough works.

### 5. Position management snippet lacked database context ℹ️

`src/position_management/README.md` referenced an undefined `database_manager` variable and
implicitly required PostgreSQL even when the example only needs dynamic risk adjustments.

## Changes Made

| File | Description |
| ---- | ----------- |
| `docs/backtesting.md` | Swapped the programmatic example to `MockDataProvider`, disabled DB logging, and documented when to re-enable the cached Binance provider so the snippet runs offline. |
| `src/position_management/README.md` | Added the missing `DatabaseManager` import, defensive initialization (falls back to `None` when `DATABASE_URL` isn’t configured), and demonstrated how to apply the resulting adjustments back into `RiskParameters`. |
| `src/trading/README.md` | Rewrote the overview to describe the component-first architecture and added a runnable breakout example built on `Strategy`, `SignalGenerator`, `FixedRiskManager`, and `FixedFractionSizer`. |
| `src/prediction/README.md` | Updated the usage section to load a real bundle from the registry, honor the `feature_schema`, and run inference via the bundle’s `OnnxRunner`. |
| `tests/README.md` | Replaced legacy instructions with the actual runner commands, directory map, marker taxonomy, and performance benchmarking guidance (`tests/performance/performance_benchmark.py`). |

## Validation

Executed each updated example to confirm it runs against the current codebase:

- `python - <<'PY'` (docs/backtesting) – ran the new `MockDataProvider` snippet and printed
  `{'total_return': 0.0, 'max_drawdown': 0.0}`.
- `python - <<'PY'` (src/position_management) – exercised the dynamic risk sample; the
  `DatabaseManager` fallback triggered (no PostgreSQL), and the script printed
  `{'size': 6.25, 'position_factor': 0.8, 'scaled_risk': 0.016}`.
- `python - <<'PY'` (src/trading) – executed the breakout strategy example and produced the
  expected `SignalDirection.BUY 140.0` output.
- `python - <<'PY'` (src/prediction) – loaded the real BTCUSDT bundle and emitted
  `{'model': '2025-10-30_12h_v1', 'price': 0.6395, 'confidence': 1.0, 'direction': 1}`.

See the shell log in this session for the exact command bodies. No runtime code was modified
outside of documentation.
