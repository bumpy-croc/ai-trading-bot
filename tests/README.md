# Test Suite

> **Last Updated**: 2025-11-22  
> **Related Documentation**: [Development workflow](../docs/development.md#tests-and-diagnostics)

The repository ships a single unified test suite that exercises the component-based strategy stack, engines, CLI commands, and
infrastructure glue.

## Quick start

```bash
# Fast unit tests (default parallelism inside the runner)
python tests/run_tests.py unit

# Database and end-to-end checks
python tests/run_tests.py integration

# Everything (unit + integration)
python tests/run_tests.py all

# Direct pytest usage when you need fine-grained control
pytest -m "not integration and not slow"
```

`tests/run_tests.py` honours the same markers used in CI, so running it locally mirrors CI behaviour (including environment variables for
PostgreSQL containers when integration tests are selected).

## Directory layout

- `tests/unit/` – Pure-Python unit tests broken down by domain (`strategies/components`, `risk`, `prediction`, `live`, `data_providers`, etc.).
- `tests/integration/` – Backtester/live-engine scenarios, CLI harnesses, and database contract checks. Subpackages match subsystems
  (e.g., `integration/live/`, `integration/backtesting/`).
- `tests/performance/` – Lightweight benchmarking harness (`performance_benchmark.py`) plus the JSON baselines it reads/writes.
- `tests/mocks/` – Shared mock objects (database doubles, fake providers) reused across both unit and integration tests.

## Common commands

- `pytest tests/unit/strategies/components -v` – Focus on the component architecture.
- `pytest tests/integration/live/test_full_position_lifecycle.py -v` – Exercise the live engine order/position flow.
- `pytest tests/integration/cli/test_backtest_integration.py -v` – Validate the `atb backtest` command wiring.
- `pytest tests/performance/performance_benchmark.py -k sharpe` – Run the quick performance benchmark subset.
- `pytest -m "slow" -n 4` – Opt into slow markers with parallel workers.

## Performance benchmarks

`tests/performance/performance_benchmark.py` compares recent backtest artefacts against the committed baseline in
`tests/performance/performance_baseline.json`. Update the baseline only when deliberate strategy changes warrant it:

```bash
# Run benchmarks and write an updated baseline file
python tests/performance/performance_benchmark.py --update-baseline
```

Commit the modified JSON alongside the code change so CI observes the same expectations.

## Tips

- Use markers from `pytest.ini` (`integration`, `slow`, `live_trading`, etc.) to slice the suite instead of ad-hoc path filters.
- Keep PostgreSQL running (see `docs/development.md`) before invoking `python tests/run_tests.py integration`; the runner will bail early
  if it cannot reach the database URL.
- Pass `-k <expr>` to `pytest` when iterating on a single test case. Example:

  ```bash
  pytest tests/unit/risk/test_risk_manager.py -k "calculate_position_fraction"
  ```

- When debugging flaky concurrency tests, add `-s --maxfail=1 --lf` to re-run only the last failures with stdout enabled.

## Reporting issues

- File bugs with the failing command plus the exact marker/filter you used.
- Attach `tests/performance/performance_benchmark.py` output when reporting performance regressions so reviewers can diff against the
  baseline JSON.
