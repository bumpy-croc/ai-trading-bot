# Development workflow

> **Last Updated**: 2025-10-24

This project ships a command-line interface and Makefile targets that standardise local setup, quality checks, and diagnostics.

## Environment setup

**Requirements**: Python 3.9+ (Python 3.11+ recommended)

```bash
python -m venv .venv && source .venv/bin/activate
make install            # install the CLI in editable mode
make deps               # install development dependencies (pytest, ruff, mypy, etc.)
```

Run `make dev-setup` to execute helper scripts (pre-commit hooks, git config) used by maintainers.

**Related**: See [Configuration](configuration.md) for environment variable setup.

## Railway deployment quick start

1. Install the Railway CLI (`npm install -g @railway/cli`) and authenticate with `railway login`.
2. From the project root run `railway init` and select the target environment, then provision PostgreSQL with `railway add postgresql`.
3. Set required variables (`BINANCE_API_KEY`, `BINANCE_API_SECRET`, `TRADING_MODE`, `INITIAL_BALANCE`, `DATABASE_URL`) via `railway variables set <KEY>=<VALUE>`.
4. Deploy the service with `railway up`; the workflow builds the container and applies environment variables automatically.
5. After the deploy succeeds, verify connectivity from your workstation:
   - `railway run atb db setup-railway --verify`
   - `railway run atb db backup --env production --backup-dir ./backups --retention 7`
6. Monitor logs (`railway logs --environment production`) and dashboards exposed by `atb live-health` to confirm the strategy is processing market data.

## Tests and diagnostics

- `pytest -q` – run the entire unit/integration suite.
- `make test` – parallel test run (`pytest -n 4`).
- `atb tests heartbeat` – insert a `SystemEvent` row for monitoring pipelines.
- `atb tests db` – verify database connectivity end-to-end.
- `atb tests download` – smoke test data downloads via CCXT.

## Code quality

- `make code-quality` – run Black formatting, Ruff linting, MyPy type checks, and Bandit security scans.
- `black .` and `ruff check . --fix` – apply formatting and lint fixes manually.
- `python bin/run_mypy.py` – strict type checking without formatting.
- `bandit -c pyproject.toml -r src` – security audit focusing on runtime code.

The repository enforces Ruff/Black style in CI, so commit formatted code to avoid failures.

## Strategy versioning

Run `atb strategies version` after modifying any file in `src/strategies/`. The helper inspects staged changes, prompts for a
succinct changelog, bumps the semantic version, and auto-stages the updated manifests under `src/strategies/store/`. Add
`--yes` when scripting the workflow; the bundled pre-commit hook simply delegates to this command when the helper is available.

## Helpful shortcuts

- `make backtest STRATEGY=ml_basic DAYS=30` – quick simulations while iterating on strategies.
- `make live` / `make live-health` – start the live runner (paper trading by default) from the shell.
- `make optimizer` – trigger the optimisation CLI with the default configuration.

Use these wrappers to mirror CI behaviour locally before opening pull requests.

## Codex auto-review workflow

The repository includes a Codex-driven loop that keeps running fast validations, requests a structured review, and lets Codex apply fixes until the review comes back clean.

```bash
python -m cli codex auto-review \
  --plan-path docs/execplans/codex_auto_review.md \
  --max-iterations 3
```

Key behaviour:

- Validation commands (`--check`) run before every review iteration. If you omit the flag, no tests or linters run and the workflow relies solely on Codex review/fix cycles. Add only the fast checks you actually need.
- The workflow automatically diffs your current branch against `develop` (override with `--compare-branch <name>` or `--compare-branch ""` to disable) so Codex focuses on the recent changes.
- Codex is explicitly told to focus on issues introduced by that diff; it can inspect other files for context but only reports regressions tied to the diff, so keep the diff scoped to the work you want touched.
- The review step enforces `cli/core/schemas/codex_review.schema.json`, so Codex replies with machine-readable findings. When the findings array is empty and validations pass, the command exits with status 0.
- Fix iterations run in `--full-auto` mode by default. Pass `--dangerous-fix` to let Codex bypass sandboxing/approvals entirely (recommended only in a disposable environment).
- Artifacts live under `.codex/workflows/<timestamp>/` and include validation logs, structured review JSON, and Codex fix transcripts for auditability.
- Run the command through the project’s Python 3.11 environment (`python3.11 -m cli ...` or `.venv/bin/python -m cli ...`) because the codebase relies on 3.10+ typing features. The loop also injects that interpreter as the `PYTHON` environment variable so Makefile targets like `make test` work even if `python` is not on your PATH (override with `--python-bin`).

You can point `--plan-path` to the ExecPlan that guided the change so Codex understands the intended milestones, but the flag is optional—leaving it out simply tells Codex to review the diff/validations. Use `--profile <name>` to select an alternate Codex configuration, or `--max-iterations 0` for a dry run that just prints the help/exit path without calling Codex.
