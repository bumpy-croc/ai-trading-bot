# Repository Guidelines

## Project Structure & Module Organization
- `cli` - CLI commands for interacting with the codebase and trading bot. Use instead of scripts.
- `docs` - Core Documentation
- `migrations` - Database migrations
- `src` - Application Code
    - `src/backtesting` - Backtesting engine
    - `src/config` - Configuration options
    - `src/dashboards` - Dashboards
    - `src/data_providers` - Data providers for communicating with crypto exchanges
    - `src/database` - Database manager and ui
    - `src/infrastructure` - Cross-cutting logging/config/runtime helpers (logging config/context/events, path+geo+secret utilities)
    - `src/live` - Live trading engine
    - `src/ml` - ML models
    - `src/optimizer` - Parameter optimization and strategy tuning tools for systematic strategy improvement.
    - `src/position_management` - Position Management policies
    - `src/prediction` - ONNX model registry and caching
    - `src/risk` - Global risk management applied across entire system
    - `src/regime` - Regime detection
    - `src/sentiment` - Sentiment adapters that merge provider data onto market series
    - `src/tech` - Shared indicator math, adapters, and feature builders (feeds prediction, risk, dashboards)
    - `src/trading` - Trading-specific helpers (strategies, symbols, etc.)
- `tests`
    - `unit` - Unit tests. Keep fast and robust.
    - `integration` - Integration tests for larger surface area.

## Operational Guidelines

- use the `atb` cli for interacting with the application when necessary. (`atb --help`)
- If you create temporary scripts, delete them after.
- For any change, make sure changes are covered by unit/integration tests and relevant tests all pass.

## Build, Test, and Development Commands
```bash
# Quick backtest (development)
atb backtest <STRATEGY_NAME> --symbol <SYMBOL> --timeframe 1h --days 30

# Run tests
python tests/run_tests.py unit          # Unit tests
python tests/run_tests.py integration   # Integration tests

# Code quality
ruff check . --fix
black .

# Live trading (paper mode)
atb live <STRATEGY_NAME> --symbol <SYMBOL> --paper-trading

# Run monitoring Dashboard
atb dashboards run monitoring

```

## ExecPlans

When writing complex features or significant refactors, use an ExecPlan (as described in .agent/PLANS.md) from design to implementation.

## Coding Style & Naming Conventions

### Key Conventions
1. Prioritize **readability, simplicity, and maintainability**.
2. Design for **change**: isolate business logic and minimize framework lock-in.
3. Emphasize clear **boundaries** and **dependency inversion**.
4. Ensure all behavior is **observable, testable, and documented**.
5. **Automate workflows** for testing, building, and deployment.

### General

- Use descriptive variable names
- Use early returns and guard clauses
- Avoid magic numbers. Define constants at module level
- Avoid overly large files.
- Avoid obvious within-file duplication
- Avoid deep nesting levels
- Remove debugging and temporary code before commits
- Code should be transparent in its intent.
- Keep lines at a readable length
- Avoid unreachable dead code
- Avoid using magic numbers. All contants should be declared with a descriptive name before its use.
- Favor composition over inheritance
- Avoid use of goto statements
- Don't use break in inner loops (break statements in deeply nested loops make control flow hard to follow without clear documentation.)
- Release locks even on exception paths (every lock acquisition must have a guaranteed release, even when exceptions occur)
- Place all user-customizable configuration variables at the beginning of scripts.

### Functions
- Keep functions concise
- Don't override function arguments
- Make a function's purpose self-evident
- Don't overuse undocumented anonymous functions
- Functions should always have a doc comment explaining what it does


### Regular expressions
- Avoid slow regular expressions (nested quantifiers or ambiguous patterns can cause catastrophic backtracking and performance issues.)

### Error Handling
- Handle errors in catch blocks (no empty catch blocks)
- Implement robust error handling.
- Prioritize specific exception types over generic ones. 
- Log errors with sufficient context (e.g., relevant variables, operation attempted). 
- Avoid silencing errors unless explicitly requested and justified. 
- Proactively include input validation and checks for null/undefined/unexpected values.

### Classes
- Classes should have single responsibility
- Use one class per file

### Databases
- Avoid SELECT * in SQL queries
- Avoid redundant database indexes

### Math
- Check divisor before division operations (division by zero causes runtime crashes and must be prevented with explicit checks)

### Comments
- Comment on the goal (why), not the mechanics (what)
- Don't ever use words like "new", "updated", etc in comments or filenames. 
- For complex algorithms or non-obvious logic, include a high-level comment explaining the approach before the code block
- Comments must describe the code's current state and purpose, not the history of changes made to it. All comments should be written in the simple present tense to describe what the code *does*, not what it *used to do* or *now does*. Examples:
	- **Bad:** `// New enhanced v2 API.`
	- **Good:** `// * Fetches user data from the v2 API.`
	- **Bad:** `// TODO: This was a temporary fix, will rewrite later.`
	- **Good:** `// TODO: Refactor this logic to be more efficient.`

### Types
- Avoid `Any` where possible in type systems
- **Type definitions properly**, especially when dealing with public APIs.

### Security

- Never embed actual sensitive information (API keys, passwords, personal data, specific user-dependent URLs/paths) directly in code.
- Always use clear, conventional placeholders (e.g., `YOUR_API_KEY`, `DATABASE_CONNECTION_STRING`, `PATH_TO_YOUR_FILE`).
- Apply **input validation and sanitization** rigorously, especially on inputs from external sources.
- Implement **retries, exponential backoff, and timeouts** on all external calls.

### Documentation 

- Maintain a `CONTRIBUTING.md` and `ARCHITECTURE.md` to guide team practices.

### Tests
- **Keep tests stateless**: Use fixtures, avoid global state.
- When writing tests, use the Arrange - Act - Assert (AAA) pattern
- Unit tests should be FIRST (fast, isolated, repeatable, self-validating and timely)

## Testing Guidelines

- Use `python tests/run_tests.py` to run tests; do not add custom runners.
- Default parallelism is 4 workers locally; CI uses 2.
- Categories: smoke, unit, integration, critical, fast, slow, grouped, benchmark.
- Prefer markers over fragile selection: e.g., `-m "not integration and not slow"`.
- Keep tests stateless; use fixtures; avoid global state.
- Mock external APIs; do not hit real exchanges in unit tests.
- Integration tests are sequential.
- Keep test data under `tests/data/`; avoid large new binaries.

## Git Guidelines
- Write imperative, present-tense commits (`Add short-entry guardrails`) and use optional scope prefixes (`docs:`, `fix:`) when helpful.
- Only commit the changes you made during a session, not the whole working tree.
- Verify correct branch before committing.
- When performiung a merge or rebase, fetch from remote first. Use non interactive mode.

## Pull Request Guidelines
- Always follow the `.github/pull_request_template.md`.
- Reference issues or tickets with `(#123)` when relevant. - - Each PR should describe user-facing impact, outline testing (`make test`, `make code-quality`), and attach strategy metrics or dashboard screenshots when behaviour changes. 
- Request review only after CI is green and migrations or scripts are documented in `docs/` if they change operational steps.

## GitHub Interaction Guidelines

- If available prefer the GitHub MCP server for interacting with GitHub. Otherwise use the `gh` cli command (authenticated with the GITHUB_TOKEN env variable).

## Security & Configuration Tips
Never commit API keys or live trading credentials; load them via environment variables consumed by `src/config` loaders. Keep `.env` files out of version control and document required settings in PRs. For local databases, run `make migrate` after schema updates and ensure backups in `backups/` are encrypted or excluded from commits.
