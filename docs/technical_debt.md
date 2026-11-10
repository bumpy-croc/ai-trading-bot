# Technical Debt

> **Verified**: 2026-10-04 against `develop` @ `812b5701`.
> **Living source**: open work is tracked in GitHub issues (label `type:chore`, see
> [GitHub labels](github_labels.md)). This page records measured state and the standing
> backlog themes only; re-measure before acting on a number. The 2025-11-21 snapshot it replaces is
> kept in [archive/technical_debt_2025-11-21.md](archive/technical_debt_2025-11-21.md) and should not be
> read as current.

## Measured state

| Check | Result | How to re-measure |
|-------|--------|-------------------|
| Size | 330 Python files / ~120k lines in `src/`, 28 files in `cli/` | `find src -name '*.py' \| wc -l` |
| Ruff | clean on `src/` and `cli/` | `ruff check src cli` |
| Black | clean | `black --check src cli` |
| Bandit | 0 medium/high, 5 low; 10 findings suppressed with `# nosec` | `bandit -c pyproject.toml -r src` |
| `TODO` / `FIXME` markers | none in `src/` or `cli/` | `grep -rnE "TODO\|FIXME" src cli` |
| `shell=True` | none in `src/` or `cli/` | `grep -rn "shell=True" src cli` |
| `pickle` | one module, `src/prediction/utils/caching.py` (local cache keys, `nosec B403`) | `grep -rn "import pickle" src cli` |
| Functions over 50 lines | 544 of 3,563 (15%) | AST scan of `src/` and `cli/` |
| Functions without a return annotation (excluding `__init__`) | 270 | AST scan of `src/` and `cli/` |
| `except Exception` | 810 occurrences in 125 files | `grep -rn "except Exception" src cli` |
| Mypy | no type errors over `src/` and `cli/`; the `type-check` CI job (`.github/workflows/tests.yml`) fails any PR that introduces one | `python bin/run_mypy.py` |

## Open items

### Oversized modules and functions

The live and backtest engines were split since the 2025 snapshot: `BacktestingEngine.run` is now 87
lines (it was 1,248), and the live engine delegates entry, exit and market-data work to coordinators
under `src/engines/live/execution/`. Size has moved rather than gone away:

- `src/database/manager.py` is 4,568 lines (2,492 in 2025). Longest methods: `log_trade` (243),
  `atomic_position_reconciliation` (184), `get_dynamic_risk_performance_metrics` (166),
  `log_position` (159). Data access and analytics are still in one class.
- `src/engines/live/trading_engine.py` is 3,002 lines; `_trading_loop` is 308 lines and `__init__` 215.
- `src/engines/backtest/engine.py` is 2,083 lines; `__init__` is 355 lines and `_process_position_exit` 209.
- The longest functions in the repo are now in the live money path:
  `entry_coordinator.execute_entry_locked` (613), `reconciliation._reconcile_filled_entry` (473),
  `reconciliation._run_reconcile_cycle` (471), `exit_coordinator.execute_exit_locked` (407),
  `recovery.reconcile_positions_with_exchange` (388), `entry_coordinator.check_entry_conditions` (385),
  `binance_provider.place_stop_loss_order` (360).

Splitting these is high-risk work (CODE.md lock and state-management rules apply) and should be done
one function per PR with parity tests, not as a sweep.

### Broad exception handling

810 `except Exception` sites across 125 files. Many are deliberate fault isolation (monitoring,
logging, best-effort cleanup), but none has been audited against CODE.md's error-handling rules.
The money path (`src/engines/live/`, `src/database/manager.py`) is the place to start.

### Missing return annotations

270 functions. `src/database/manager.py` (12) and `src/engines/live/trading_engine.py` (10) are the
public surfaces worth finishing first.

### Flat `src/strategies/components/`

Still one flat package of 33 entries (signal generators, risk and sizing managers, performance
monitoring, registry and switching). Splitting into `core/`, `generators/`, `managers/` remains an
option; it is churn-heavy and has no functional payoff on its own.

## Resolved since the 2025-11-21 snapshot

- Ruff and Black are clean, the logging `ContextVar` no longer has a mutable default, and there are no bare `except:` clauses.
- `BacktestingEngine.run` is decomposed.
- Duplicated risk-parameter merging and policy hydration now live in `src/engines/shared/`
  (`risk_configuration.py`, `policy_hydration.py`) and `src/engines/live/strategy_runtime.py`. Thin
  per-engine wrappers such as `_apply_policies_from_decision` and `_merge_dynamic_risk_config`
  remain and delegate to the shared code.
- Shared position and trade models, the partial-operations manager and the performance tracker are
  integrated into both engines (`src/engines/shared/models.py`, `partial_operations_manager.py`,
  `src/performance/tracker.py`).
- The proposed `src/risk/parameter_merger.py`, `src/position_management/policy_applicator.py`,
  `src/data_providers/validation.py` and `src/regime/parser.py` were never created; the shared modules
  above took their place.

## Dropped from the 2025 snapshot

Generic advice with no verifiable state is not tracked here: DataFrame `.copy()` review, N+1 query
review, cache hit-rate review, test fixture and test-data organization, docstring tense. File an issue
with evidence if one of them turns out to matter.
