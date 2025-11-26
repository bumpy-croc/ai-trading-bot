# Test Suite

> **Last Updated**: 2025-11-26  
> **Related Documentation**: [Development workflow](../docs/development.md#tests-and-diagnostics)

The repository ships a single pytest-based suite with helper wrappers so engineers and automation hit the exact same commands. Use
`tests/run_tests.py` for the canonical flows or invoke pytest directly when iterating on a specific file.

## Quick start

```bash
# Fast smoke/import validation
python tests/run_tests.py smoke

# Parallelised unit slice (excludes integration + slow markers)
python tests/run_tests.py unit

# Sequential integration suite (database, live engine, CLI)
python tests/run_tests.py integration

# Unit phase followed by integration phase (stops on first failure)
python tests/run_tests.py all

# Coverage with HTML report
python tests/run_tests.py --coverage
```

`atb test unit|integration|all` delegates to the same pytest markers if you prefer the CLI entrypoint.

## Directory layout

- `tests/unit/` – fast coverage for strategies, risk, prediction, data providers, configuration, dashboards, etc. Most files are tagged with `not integration and not slow`.
- `tests/integration/` – contract tests for backtesting, live trading, CLI commands, monitoring dashboards, and prediction pipelines. Subdirectories mirror runtime packages (`backtesting/`, `cli/`, `live/`, `monitoring/`, `predictions/`, ...).
- `tests/performance/performance_benchmark.py` – lightweight timing harness triggered via `python tests/run_tests.py benchmark`.
- `tests/mocks/` – stub services (e.g., `mock_database`) shared between unit and integration tests.
- `tests/run_tests.py` – orchestrator that handles dependency checks, parallelism, environment hints, and grouped execution.

## `run_tests.py` commands

| Command | Description |
| --- | --- |
| `smoke` | Import + wiring validation (no pytest). |
| `unit` | `pytest tests/ -m "not integration and not slow"` with xdist (`-n` determined by environment). |
| `integration` | `pytest -m integration` sequentially (no xdist). |
| `critical` | `pytest -m "(live_trading or risk_management) and not slow"` for high-risk paths. |
| `database` | Targets `tests/test_database.py` only. |
| `fast` / `slow` | Marker-filtered slices: `(fast or mock_only)` vs `slow or computation`. |
| `grouped` | Three curated pytest invocations (fast/medium/heavy) for manual triage. |
| `benchmark` | Executes `tests/performance/performance_benchmark.py` (timing harness). |
| `all` | Runs `unit` then `integration` (short-circuits on failure). |
| `validate` | Ensures pytest config + directory structure exist before running anything else. |

Every command accepts `--pytest-args ...` to pass additional flags (e.g., `python tests/run_tests.py unit --pytest-args "-k ml_basic"`).

## Pytest markers & targeting

Key markers used throughout the suite:

- `integration` – DB/live-engine tests; always run sequentially.
- `slow`, `computation` – long-running jobs skipped by the default unit command.
- `fast`, `mock_only` – explicitly fast tests (consumed by the `fast` runner).
- `live_trading`, `risk_management` – grouped under the `critical` command for focused regressions.

Examples:

```bash
# Single unit file
pytest tests/unit/strategies/components/test_signal_generator.py -v

# Marker expression
pytest tests/ -m "live_trading and not slow" -v

# Integration CLI coverage
pytest tests/integration/cli/test_backtest_integration.py -v

# Disable xdist for better debugging
python tests/run_tests.py unit --pytest-args "-n 1 -k regression"
```

## Coverage & reports

- `python tests/run_tests.py --coverage` writes HTML output to `htmlcov/index.html` and prints term summaries.
- To capture JUnit XML, forward `--pytest-args "--junitxml tests/reports/unit.xml"` through the runner.
- The performance benchmark prints timing deltas directly to stdout; capture the log in CI if you need persisted metrics.

## Troubleshooting

1. **Dependency check fails** – install the listed packages (`pytest`, `numpy`, `pandas`) or rerun with `--no-deps-check` if you are managing dependencies manually.
2. **Parallel hangs** – add `--pytest-args "-n 1"` (or run pytest directly) to turn off xdist for the current run.
3. **Database state drift** – integration tests expect a PostgreSQL URL; reset via `atb db nuke` or point `DATABASE_URL` at a fresh instance.
4. **Marker confusion** – run `pytest --markers` to inspect the authoritative list before tagging new tests.

CI uses the same runner commands, so keeping this document and `tests/run_tests.py` in sync ensures the nightly and release workflows behave exactly like local development.
