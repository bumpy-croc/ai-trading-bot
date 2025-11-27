# Test Suite

> **Last Updated**: 2025-11-27  
> **Related Documentation**: [Development workflow](../docs/development.md#tests-and-diagnostics)

The repository ships a comprehensive suite that exercises the component-based strategy runtime, data pipeline, prediction engine, and live trading integrations. Use the helper runner (`tests/run_tests.py`) for the most common combinations or call `pytest` directly when you need full control.

## Quick start

### Helper commands
```bash
python tests/run_tests.py smoke         # Fast validation (minutes)
python tests/run_tests.py unit          # All unit tests (xdist enabled)
python tests/run_tests.py integration   # Integration tests (serial, DB required)
python tests/run_tests.py all           # Unit -> integration cascade
python tests/run_tests.py grouped       # Curated fast/medium/heavy batches
```

### Direct pytest invocations
```bash
# Skip integration + slow markers for local iterations
pytest -m "not integration and not slow" tests/

# Component system focus
pytest tests/unit/strategies/components -n auto -m "not slow"

# Live trading integration slice
pytest tests/integration/live_trading -m integration

# Specific file with verbose tracebacks
pytest tests/unit/strategies/components/test_signal_generator.py -vv --tb=long
```

## Suite layout

- `tests/unit/` – fast, isolated tests organised by subsystem (strategies, risk, prediction, optimizer, training pipeline, position management, etc.).
- `tests/integration/` – database and engine contracts covering CLI flows, live trading, monitoring dashboards, and backtesting regression harnesses.
- `tests/performance/` – `performance_benchmark.py` plus `performance_baseline.json` for tracking test-suite runtime trends.
- `tests/mocks/` – shared fixtures (e.g., `mock_database.py`) consumed by both unit and integration suites.
- `tests/run_tests.py` – convenience wrapper that groups common pytest invocations, marker combinations, and coverage helpers.

## Test markers

Pytest markers are defined in `pytest.ini` and enforced via `--strict-markers`. Key markers include:

- `unit`, `integration`, `smoke`, `fast`, `slow`, `medium`, `computation`
- `live_trading`, `risk_management`, `strategy`, `data_provider`, `monitoring`
- `performance`, `database`, `mock_only`, `io`, `timeout`

Filter tests with `pytest -m "<expression>"` or pass `--markers "<expression>"` to `tests/run_tests.py`.

## Performance benchmarking

Use the dedicated harness to baseline end-to-end runtimes:

```bash
python tests/performance/performance_benchmark.py          # Full benchmark suite
python tests/performance/performance_benchmark.py --compare # Compare against saved baseline
python tests/performance/performance_benchmark.py --report  # Print aggregated stats only
```

- Results are stored in `tests/performance/performance_baseline.json` with timestamps, averages, and tests/sec calculations.
- Provide `--output <path>` to write a separate snapshot, or `--append` to keep historical runs in the same file.
- The optional `benchmark` target inside `tests/run_tests.py` is a thin wrapper around the same script; until its path is updated, run the commands above directly from the repository root.

## Coverage

```bash
# Full suite with HTML + terminal coverage
python tests/run_tests.py --coverage

# Direct pytest invocation (unit slice)
pytest tests/unit --cov=src --cov-report=term-missing --cov-report=html
```

Coverage artefacts are emitted under `htmlcov/` unless overridden via `.coveragerc`.

## Troubleshooting & references

- `tests/unit/strategies/TEST_MIGRATION_GUIDE.md` – expectations for the legacy-to-component migration harness.
- `docs/development.md#tests-and-diagnostics` – CLI helpers (`atb test`, `atb tests heartbeat`, etc.) that mirror CI.
- Use `pytest -vv --maxfail=1` plus `--pdb` when chasing intermittent failures.
- When debugging flaky integration tests, export `DATABASE_URL`, run `docker compose up -d postgres`, and invoke the target file directly to keep logs focused.
