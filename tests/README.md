# Test Suite

> **Last Updated**: 2025-11-24  
> **Related Documentation**: [Development workflow](../docs/development.md#tests-and-diagnostics)

The repository ships a single runner (`tests/run_tests.py`) plus focused pytest
targets that cover every subsystem—from component strategies through live trading
orchestration and CLI helpers.

## Running the suite

```bash
# Fast unit surface (skips integration + slow markers)
python tests/run_tests.py unit

# Integration surface (sequential, exercises PostgreSQL + engines)
python tests/run_tests.py integration

# Full pipeline: unit first, integration second
python tests/run_tests.py all

# Targeted helpers
python tests/run_tests.py fast        # (fast | mock_only) markers
python tests/run_tests.py slow        # long-running computations
python tests/run_tests.py grouped     # optimized waves for CI
python tests/run_tests.py benchmark   # wraps performance_benchmark.py
```

All runner commands accept `--pytest-args` to forward additional flags (markers,
`-k` expressions, coverage options) directly to pytest.

## Directory map

- `tests/unit/` – fast, parallelizable coverage of individual modules. Notable
  areas:
  - `strategies/components/` – signal generators, risk adapters, strategy runtime,
    performance tracking, registry integrations.
  - `data_providers/`, `risk/`, `position_management/`, `config/`, `cli/` – core
    services wired exactly as the application does.
  - `ml_training/`, `training_pipeline/`, `predictions/` – feature pipeline and
    registry validation.
- `tests/integration/` – end-to-end workflows:
  - `test_component_trading_workflows.py` and `test_error_handling_workflows.py`
    simulate backtest + live decision cycles.
  - `test_account_sync.py`, `test_order_lifecycle.py`, `test_order_management_methods.py`
    validate database + exchange abstractions.
  - Subpackages (`live/`, `predictions/`, `monitoring/`, `cli/`) focus on domain-
    specific flows.
- `tests/performance/performance_benchmark.py` – measures elapsed time + test
  counts for curated pytest commands and persists summaries to
  `tests/performance/performance_baseline.json`.
- `tests/data/` and `tests/mocks/` – fixtures used by both unit and integration
  layers.

## Core pytest markers

Markers declared in `pytest.ini` and enforced via `--strict-markers`:

- `unit`, `integration`, `slow`, `fast`, `medium`, `computation`
- Domain markers: `live_trading`, `risk_management`, `database`, `strategy`,
  `data_provider`, `monitoring`, `performance`
- Utility markers: `mock_only`, `smoke`, `timeout`

Examples:

```bash
# Run everything except integration + slow suites
pytest -m "not integration and not slow" -n auto --dist=loadgroup

# Focus on database-heavy coverage
pytest -m "database" tests/integration -v
```

## Targeted pytest commands

```bash
# Component system focus
pytest tests/unit/strategies/components/test_signal_generator.py -v
pytest tests/unit/strategies/components/test_risk_manager.py -v

# Strategy composition + registry workflows
pytest tests/unit/strategies/test_ml_basic_unit.py -v
pytest tests/integration/test_component_trading_workflows.py -v

# CLI + runner checks
pytest tests/unit/cli -v

# Prediction/ML pipeline
pytest tests/unit/predictions -v
pytest tests/unit/training_pipeline -v
```

Use `-n auto --dist=loadgroup` for unit scopes. Integration suites intentionally
run sequentially because they share PostgreSQL fixtures.

## Performance benchmarking

`tests/performance/performance_benchmark.py` wraps a curated set of pytest
commands, measures runtime/test counts, and updates
`tests/performance/performance_baseline.json`:

```bash
# Create or refresh benchmark data
python tests/performance/performance_benchmark.py

# Compare against the last recorded runs
python tests/performance/performance_benchmark.py --compare
```

The runner’s `benchmark` command simply shells out to the same script, so either
entry point keeps baseline data in sync.

## Troubleshooting and docs

- `tests/run_tests.py --help` lists every helper (`smoke`, `critical`, `database`,
  `coverage`, etc.) plus optional arguments like `--file` or `--markers`.
- `tests/COMPONENT_TESTING_GUIDE.md`, `tests/TEST_TROUBLESHOOTING_GUIDE.md`, and
  `tests/unit/strategies/TEST_MIGRATION_GUIDE.md` provide deeper context for the
  component system, migrations, and debugging broken suites.
- Coverage-friendly runs: `python tests/run_tests.py --coverage` or
  `pytest --cov=src --cov-report=html`.

When adding new tests, mirror the existing marker taxonomy and keep fixtures
stateless so both the runner and raw pytest commands behave the same way locally
and in CI.
