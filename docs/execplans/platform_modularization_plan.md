# Platform Modularization and Trading Shared Cleanup

This ExecPlan is a living document. Maintain Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective per `.agents/PLANS.md`. Treat this file as the sole source of truth for the restructuring effort described below.

## Purpose / Big Picture

`src/trading/shared` and `src/utils` have become catch-alls mixing logging glue, sentiment adapters, and position-sizing helpers. This plan consolidates those pieces into purpose-driven packages (`src/infrastructure`, `src/sentiment`, `src/position_management`, `src/trading/symbols`) so future contributors can immediately tell where to add or adjust infrastructure code. Completion means there are no leftover shims or unused directories, all imports target the new namespaces, and the backtester (`make backtest STRATEGY=ml_basic DAYS=5`) plus live engine (`make live-health STRATEGY=ml_basic PORT=8090`) continue to run on Python 3.11 with `python3.11 -m pytest -n 4` passing.

## Progress

- [x] (2025-10-26 18:40Z) Captured scope and current layout; authored this ExecPlan.
- [x] (2025-10-26 19:30Z) Stood up `src/infrastructure/logging` and `src/infrastructure/runtime` with READMEs; moved logging/context/decision logging and runtime helpers into the new namespaces and rewired imports.
- [x] (2025-10-26 19:40Z) Relocated sentiment adapters, position sizing, and `SymbolFactory` into `src/sentiment`, `src/position_management/sizing.py`, and `src/trading/symbols/factory.py` with updated imports/tests.
- [x] (2025-10-26 19:45Z) Removed `src/trading/shared` after confirming no remaining imports.
- [x] (2025-10-26 19:50Z) Removed the legacy `src/utils` directory, added READMEs for new packages, and refreshed `AGENTS.md`/`docs/development.md` to describe the updated structure.
- [ ] Re-run automated checks (`python3.11 -m pytest -n 4`, `make backtest STRATEGY=ml_basic DAYS=5`, `make live-health STRATEGY=ml_basic PORT=8090`) and update docs to describe the new structure. (Blocked: python3.11 interpreter segfaults because wheel dependencies are only built for Python 3.9, and `make` currently assumes a `python` binary that does not exist in the sandbox.)

## Surprises & Discoveries

- Observation: `python3.11 -m pytest -n 4` segfaults immediately (Signal 11) because the shared environment only has pandas/numpy wheels compiled for macOS Python 3.9; importing them under 3.11 causes interpreter crashes before tests run.
  Evidence: Running `python3.11 -m pytest tests/unit/trading/test_shared_indicators.py -q` exits with `Sandbox(Signal(11))` on 2025-10-26.
- Observation: `make backtest`/`make live-health` fail early because the sandbox lacks a `python` binary, and the Makefile's `install` target hardcodes `python -m pip ...`.
  Evidence: Commands exit with `python -m pip install --upgrade pip` followed by `make: python: No such file or directory` on 2025-10-26.

## Decision Log

- Decision: Move logging and runtime/environment helpers under a new `src/infrastructure` namespace to separate them from domain-specific code.
  Rationale: These modules cross-cut the entire system and should live in a clearly named infrastructure package instead of `src/utils`.
  Date/Author: 2025-10-26 / Codex agent.

- Decision: Relocate sentiment and sizing helpers next to their owning domains (sentiment providers, position management) and treat `trading.shared` as deprecated.
  Rationale: Enhances discoverability for engineers working on those features and prevents future duplication.
  Date/Author: 2025-10-26 / Codex agent.

## Outcomes & Retrospective

To be filled in after completion: summarize what shipped, note any regressions avoided, and list remaining follow-ups (if any).

## Context and Orientation

- `src/trading/shared/decision_logger.py` writes trade decisions to the DB via `db_manager.log_strategy_execution`. Strategies/backtesting/live engines import it through `trading.shared`.
- `src/trading/shared/sentiment.py` merges historical or live sentiment fields into OHLCV frames using provider interfaces.
- `src/trading/shared/sizing.py` normalizes strategy position sizes to fractions of balance.
- `src/trading/shared/indicators.py` now forwards to `src.tech.adapters.row_extractors` and can be removed once imports are updated.
- `src/utils` contains heterogeneous helpers: logging config/context/events, cache TTL utilities, geo-detection for Binance routing, secret lookup, symbol formatting, and project path helpers.
- There is no dedicated `src/infrastructure` package; infrastructure code is scattered. The new structure must include READMEs explaining responsibilities per `.agents/PLANS.md` guidance.

## Plan of Work

1. **Create `src/infrastructure` skeleton** with subpackages:
   - `src/infrastructure/logging/` for `logging_config.py`, `logging_context.py`, `logging_events.py`, and a moved `decision_logger.py` (rename to `decision_logger.py` and importable as `src.infrastructure.logging.decision_logger`).
   - `src/infrastructure/runtime/` for `project_paths.py` (rename to `paths.py`), `secrets.py`, `geo_detection.py` (rename to `geo.py`), and `cache_utils.py` (rename to `cache.py`). Include README files for the root and each subpackage, documenting scope and extension points.
   Update all imports (`src/utils/...`, `src/backtesting/engine.py`, `src/live/trading_engine.py`, etc.) to the new paths.

2. **Relocate trading-specific helpers**:
   - Move `src/trading/shared/sentiment.py` to `src/sentiment/adapters.py`; adjust references in backtester, live engine, dashboards, and tests.
   - Move `normalize_position_size` into `src/position_management/sizing.py` (a new module) so strategies/risk components use it from that namespace.
   - Move `SymbolFactory` into `src/trading/symbols/factory.py`, with a README describing naming conventions; update CLI/tools referencing the old import.

3. **Delete `src/trading/shared` and `src/utils`**:
   - Once all modules move, remove the directories entirely rather than leaving shims. Ensure `python -m compileall` (implicitly via tests) has no dangling references.

4. **Documentation updates**:
   - Add sections to `docs/development.md` (or `docs/architecture/*.md`) explaining the new `src/infrastructure`, `src/sentiment`, and `src/trading/symbols` locations.
   - Update any README/examples that previously referenced `src.trading.shared` or `src.utils`.

## Concrete Steps

1. Create directories with READMEs:
       mkdir -p src/infrastructure/logging src/infrastructure/runtime src/sentiment src/trading/symbols src/position_management
       touch src/infrastructure/README.md src/infrastructure/logging/README.md src/infrastructure/runtime/README.md src/sentiment/README.md src/trading/symbols/README.md src/position_management/README.md
   Populate READMEs with clear scope statements.

2. Move logging modules:
       mv src/utils/logging_config.py src/infrastructure/logging/config.py
       mv src/utils/logging_context.py src/infrastructure/logging/context.py
       mv src/utils/logging_events.py src/infrastructure/logging/events.py
       mv src/trading/shared/decision_logger.py src/infrastructure/logging/decision_logger.py
   Update imports (e.g., `from src.utils.logging_config import configure_logging` → `from src.infrastructure.logging.config import configure_logging`).

3. Move runtime helpers:
       mv src/utils/project_paths.py src/infrastructure/runtime/paths.py
       mv src/utils/secrets.py src/infrastructure/runtime/secrets.py
       mv src/utils/geo_detection.py src/infrastructure/runtime/geo.py
       mv src/utils/cache_utils.py src/infrastructure/runtime/cache.py
   Fix all import sites accordingly.

4. Sentiment + sizing + symbols:
       mv src/trading/shared/sentiment.py src/sentiment/adapters.py
       mv src/trading/shared/sizing.py src/position_management/sizing.py
       mkdir -p src/trading/symbols && mv src/utils/symbol_factory.py src/trading/symbols/factory.py
   Update import statements in strategies, engines, dashboards, and tests.

5. Remove now-empty directories:
       rm -rf src/trading/shared
       rm -rf src/utils
   Verify `git status` to ensure only expected files are deleted.

6. Run validations:
       python3.11 -m pytest -n 4
       make backtest STRATEGY=ml_basic DAYS=5
       make live-health STRATEGY=ml_basic PORT=8090
   Capture outputs in the Surprises section if failures occur and fix issues before continuing.

7. Update docs/logs:
       edit docs/development.md docs/architecture/*.md README references to point to the new modules
   Summarize the directory tree in `docs/platform.md` if helpful.

## Validation and Acceptance

- `python3.11 -m pytest -n 4` passes without using deprecated modules.
- `make backtest STRATEGY=ml_basic DAYS=5` finishes successfully and logs indicator extraction via the new modules.
- `make live-health STRATEGY=ml_basic PORT=8090` starts, serves health metrics, and cleanly exits with Ctrl+C.
- `rg "src\.trading\.shared"` and `rg "src\.utils"` return no matches outside of changelog/docs referencing history.

## Idempotence and Recovery

- File moves can be repeated safely; use `git restore <file>` to revert mistakes.
- If directory removal happens prematurely, recover with `git checkout -- src/trading/shared` (before committing) or `git restore -SW src/trading/shared`.
- Validation commands are non-destructive; rerun as needed.

## Artifacts and Notes

Record noteworthy command outputs (e.g., backtest summaries, live-health logs) once validation completes. Include short indented snippets for future readers troubleshooting the process.

## Interfaces and Dependencies

- New modules must expose the same public functions/classes:
  - `src.infrastructure.logging.config.configure_logging`, logging context/events, and `log_strategy_execution` from `src.infrastructure.logging.decision_logger`.
  - `src.infrastructure.runtime.paths.get_project_root`, `src.infrastructure.runtime.cache.get_cache_ttl_for_provider`, `src.infrastructure.runtime.geo.get_binance_api_endpoint`, `src.infrastructure.runtime.secrets.get_secret_key`.
  - `src.sentiment.adapters.merge_historical_sentiment` / `apply_live_sentiment`.
  - `src.position_management.sizing.normalize_position_size`.
  - `src.trading.symbols.factory.SymbolFactory`.
- No compatibility shims remain; callers import from the new paths directly.
- Documentation explicitly links to these modules so engineers know where to extend runtime/logging infrastructure.

Document future decisions or deviations inside this plan as work progresses.
