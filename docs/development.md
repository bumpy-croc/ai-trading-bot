# Development workflow

This project ships a command-line interface and Makefile targets that standardise local setup, quality checks, and diagnostics.

## Environment setup

```bash
python -m venv .venv && source .venv/bin/activate
make install            # install the CLI in editable mode
make deps               # install development dependencies (pytest, ruff, mypy, etc.)
```

Run `make dev-setup` to execute helper scripts (pre-commit hooks, git config) used by maintainers.

## Tests and diagnostics

- `pytest -q` – run the entire unit/integration suite.
- `make test` – parallel test run (`pytest -n 4`).
- `atb tests heartbeat` – insert a `SystemEvent` row for monitoring pipelines.
- `atb tests db` – verify database connectivity end-to-end.
- `atb tests download` – smoke test data downloads via CCXT.

## Code quality

- `make code-quality` – run Black formatting, Ruff linting, MyPy type checks, and Bandit security scans.
- `ruff check .` and `ruff format .` – apply lint fixes manually.
- `python bin/run_mypy.py` – strict type checking without formatting.
- `bandit -c pyproject.toml -r src` – security audit focusing on runtime code.

The repository enforces Ruff/Black style in CI, so commit formatted code to avoid failures.

## Helpful shortcuts

- `make backtest STRATEGY=ml_basic DAYS=30` – quick simulations while iterating on strategies.
- `make live` / `make live-health` – start the live runner (paper trading by default) from the shell.
- `make optimizer` – trigger the optimisation CLI with the default configuration.

Use these wrappers to mirror CI behaviour locally before opening pull requests.
