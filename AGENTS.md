# Repository Guidelines

## Project Structure & Module Organization
Application code lives in `src/`, grouped by domain: trading flows in `src/trading`, executable strategies in `src/strategies`, analytics in `src/performance`, and shared utilities in `src/utils`. Experimentation and ML assets sit under `src/ml` and `examples/`. Dash components are in `src/dashboards`, while database migrations reside in `migrations/` with settings in `alembic.ini`. Tests mirror the source tree: fast checks in `tests/unit`, scenario coverage in `tests/integration`, and latency/throughput checks in `tests/performance`.

## Build, Test, and Development Commands
Use `make dev-setup` to bootstrap a Python 3.11 virtualenv and install editable dependencies. `make install` upgrades `pip` and installs the CLI (`atb`) locally. `make deps` and `make deps-server` pull dev and runtime requirements. Run `make backtest STRATEGY=ml_basic DAYS=30` to replay a strategy, or `make optimizer STRATEGY=ml_basic` to sweep parameters. `make test` executes `pytest -n 4`, while `make code-quality` chains Black, Ruff, MyPy, and Bandit. For ad-hoc dashboards, `make dashboard-monitoring PORT=8090` exposes the monitoring UI.

## Coding Style & Naming Conventions
Follow four-space indentation and Python typing throughout. Black and Ruff enforce a 100-character line limit; run `black .` and `ruff check .` before committing. Keep modules and functions in `snake_case`, classes in `PascalCase`, and constants in `UPPER_CASE`. Configuration files (`*.yaml`, `.json`) should mirror the project’s existing naming, e.g., `feature_flags.json`. Prefer dependency injection via constructors to keep components testable.

## Testing Guidelines
Name new tests `test_<feature>.py` and place them beside the module they exercise. Reuse fixtures from `tests/conftest.py` and sample data from `tests/data/`. Extend integration suites when coordinating multiple services (e.g., `tests/integration/backtesting`). Run `pytest tests/unit -k <keyword>` during development, then `make test` and review `coverage.xml` before submitting. Include backtest artifacts under `artifacts/` when they help reviewers validate behaviour.

## Commit & Pull Request Guidelines
Write imperative, present-tense commits (`Add short-entry guardrails`) and use optional scope prefixes (`docs:`, `fix:`) when helpful. Reference issues or tickets with `(#123)` when relevant. Each PR should describe user-facing impact, outline testing (`make test`, `make code-quality`), and attach strategy metrics or dashboard screenshots when behaviour changes. Request review only after CI is green and migrations or scripts are documented in `docs/` if they change operational steps.

## Security & Configuration Tips
Never commit API keys or live trading credentials; load them via environment variables consumed by `src/config` loaders. Keep `.env` files out of version control and document required settings in PRs. For local databases, run `make migrate` after schema updates and ensure backups in `backups/` are encrypted or excluded from commits.
