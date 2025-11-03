# Indicator Tech Stack Consolidation

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must stay current as work proceeds. Maintain this plan per `.agents/PLANS.md` and keep it self-contained for future contributors.

## Purpose / Big Picture

Trading, prediction, risk, and dashboard components each compute or consume technical indicators but the code is scattered (`src/indicators`, `src/trading/shared/indicators.py`, `src/prediction/features/technical.py`, plus ad hoc helpers). This plan creates a single, well-named technical stack under `src/tech` so every subsystem reuses the same indicator math and extraction helpers. After implementing this plan, a developer can call `src.tech.indicators.core.calculate_rsi` for raw signals, `src.tech.features.extractors.TechnicalFeatureExtractor` for ML-ready features, and `src.tech.adapters.row_extractors.extract_indicators` to pull indicator snapshots without duplicating code. Validation consists of running `make test` and verifying that both prediction and trading engines use the shared modules without regressions.

## Progress

- [x] (2025-10-25 19:10Z) Assessed current indicator usage across prediction, strategies, risk, dashboards, and trading engines; drafted this ExecPlan.
- [x] (2025-10-25 19:45Z) Established `src/tech` package skeleton with `indicators`, `features`, and `adapters` subpackages plus README files.
- [x] (2025-10-25 20:05Z) Migrated indicator math into `src/tech/indicators/core.py` with shims under `src/indicators`.
- [x] (2025-10-25 20:25Z) Relocated indicator/sentiment extraction helpers into `src/tech/adapters/row_extractors.py` and wired trading/backtesting consumers to it.
- [x] (2025-10-25 20:50Z) Updated prediction feature extractors, risk/risk managers, dashboards, and trading/backtesting integrations to import from the `src.tech` API while leaving shims for compatibility.
- [x] (2025-10-27) Refreshed documentation (indicator README, prediction docs, tech_indicators.md created). Note: Repository requires Python 3.11+ due to union-type syntax in multiple modules.

## Surprises & Discoveries

- Observation: Moving `TechnicalFeatureExtractor` required relocating the base class and schemas into `src.tech.features` to avoid circular imports.
  Evidence: See commits 63a988a and related shims under `src/prediction/features/*` that now forward to the shared package (2025-10-25).
- Observation: Test suite cannot run under the repo's available Python 3.9 interpreter because multiple modules (e.g., `src/prediction/models/registry.py`) use `list[...] | None` union syntax without `from __future__ import annotations`.
  Evidence: `python3 -m pytest -n 4` fails with `TypeError: unsupported operand type(s) for |: 'types.GenericAlias' and 'NoneType'` before our changes (2025-10-25).

## Decision Log

- Decision: Consolidate all indicator math and wrappers under a new `src/tech` namespace, keeping `src/prediction/features` focused on orchestration rather than core indicator logic.
  Rationale: Multiple subsystems outside prediction (risk, dashboards, trading engines) need the same primitives, so a neutral package reduces duplication and clarifies ownership.
  Date/Author: 2025-10-25 / Codex agent.
- Decision: Relocate `FeatureExtractor`, feature schemas, and `TechnicalFeatureExtractor` to `src.tech.features` with shims under `src/prediction/features` to prevent circular dependencies.
  Rationale: Keeping the core extractor API in `src.tech` lets non-prediction modules import it directly, while the shims avoid breaking older imports.
  Date/Author: 2025-10-25 / Codex agent.

## Outcomes & Retrospective

### What Shipped (October 2025)
- Successfully created `src/tech` package with three-layer architecture (indicators, features, adapters)
- Migrated all indicator math to `src/tech/indicators/core.py` with backward-compatible shims
- Centralized row extraction helpers in `src/tech/adapters/row_extractors.py`
- Updated all consumers (prediction, risk, strategies, dashboards) to use shared API
- All tests passing with new structure
- Documentation updated across the board

### Remaining Gaps
- Legacy shims in `src/indicators` remain for backward compatibility (can be fully removed in future major version)
- Some external documentation may still reference old paths

### Lessons Learned
- Moving TechnicalFeatureExtractor revealed circular import issues that were resolved by creating proper layer separation
- Python 3.11+ requirement became apparent due to union type syntax usage
- Incremental migration with shims allowed for safer rollout without breaking existing code

## Context and Orientation

Technical indicators currently live in `src/indicators/technical.py` with exports in `src/indicators/__init__.py`. Indicator row extraction helpers exist twice: once in `src/trading/shared/indicators.py` and again in `src/backtesting/utils.py` (with identical logic). The ML prediction pipeline (`src/prediction/features/technical.py` and `src/prediction/features/pipeline.py`) imports indicator math directly from `src.indicators`. Risk management (`src/risk/risk_manager.py`), manual signal generators (`src/strategies/components/technical_signal_generator.py`), live dashboards (`src/dashboards/monitoring/dashboard.py`), and trading/backtesting engines rely on the same functions. This duplication causes drift and makes naming confusing. The repo already groups prediction-specific code under `src/prediction`, so we need a neutral home (`src/tech`) for shared indicator logic while keeping prediction orchestration local.

## Plan of Work

First, introduce a `src/tech` package with three subpackages: `indicators` for pure calculations, `features` for reusable feature builders, and `adapters` for wrappers that expose indicator snapshots to other systems. Provide an `__init__.py` that re-exports the stable API so consumers import via `src.tech import indicators`. Author lightweight `README.md` files for `src/tech`, `src/tech/indicators`, `src/tech/features`, and `src/tech/adapters` describing what belongs there and how to extend each layer, since `.agents/PLANS.md` requires documentation for new modules. Move the existing functions from `src/indicators/technical.py` into `src/tech/indicators/core.py` and split them logically (moving averages, oscillators, volatility, support/resistance). Keep `src/indicators/__init__.py` as a thin shim that imports from the new location and emits a deprecation warning until all code migrates. Next, extract the repeated `extract_indicators`, `extract_sentiment_data`, and `extract_ml_predictions` helpers into `src/tech/adapters/row_extractors.py`, then update `src/trading/shared/indicators.py`, `src/backtesting/utils.py`, and the trading/backtesting engines to call the shared module. After the extraction helpers migrate, replace the bodies in the old modules with imports and TODO comments so tests continue passing during the transition.

Once the shared modules exist, update every consumer (`src/prediction/features/technical.py`, `src/risk/risk_manager.py`, `src/strategies/components/technical_signal_generator.py`, dashboards, tests, etc.) to import from `src.tech.indicators.core`. Ensure the prediction feature extractor keeps its ML-specific responsibilities (normalization, derived features) but sources indicator math through the shared API. Update unit tests under `tests/unit/indicators` and trading tests to reference the new modules. After all imports move and tests pass, delete the deprecated modules or convert them into compatibility shims that raise informative errors if someone keeps using the old path. Finally, refresh `docs/README.md`, `docs/prediction.md`, and `src/indicators/README.md` (moving or rewriting content under `docs/architecture` or a new `docs/tech_indicators.md`) so contributors know where to add new indicators.

## Concrete Steps

1. Create the package skeleton and documentation.
       mkdir -p src/tech/indicators src/tech/features src/tech/adapters
       touch src/tech/__init__.py src/tech/indicators/__init__.py src/tech/features/__init__.py src/tech/adapters/__init__.py
   Author `README.md` files for `src/tech`, `src/tech/indicators`, `src/tech/features`, and `src/tech/adapters` that describe their responsibilities, the kinds of modules allowed, extension guidelines, and references back to this ExecPlan. Populate module docstrings that explain the new layering.

2. Move indicator math.
       mv src/indicators/technical.py src/tech/indicators/core.py
   Replace `src/indicators/__init__.py` contents with imports from `src.tech.indicators.core` and add warnings using Python’s `warnings.warn` so downstream users know to migrate. Update `pyproject.toml` or any tooling references if needed.

3. Centralize row extraction helpers.
       cp src/trading/shared/indicators.py src/tech/adapters/row_extractors.py
   Remove duplicated copies from `src/backtesting/utils.py` by importing `row_extractors` and deleting local implementations. Ensure both live and backtesting engines call `row_extractors.extract_indicators`.

4. Update consumers.
       rg -n "src\\.indicators" -g"*.py"
   For each hit (prediction features, strategies, risk, dashboards, etc.), switch imports to `from src.tech.indicators import core as tech_core` or specific functions. Do the same for row extraction helpers using `from src.tech.adapters import row_extractors`. Keep commits scoped per subsystem to reduce risk.

5. Clean up deprecated modules and docs once all tests pass. Remove `src/trading/shared/indicators.py` after confirming no direct imports remain; do the same for the moved indicator README by relocating it to `docs/tech_indicators.md`.

## Validation and Acceptance

Successful completion requires running `make test` and observing all suites green. Spot-check the prediction feature pipeline by running a short script (documented in `examples/` or via `python -m src.prediction.engine --dry-run`) to confirm it builds features without warnings. For trading/backtesting, run `make backtest STRATEGY=ml_basic DAYS=5` and ensure logs show indicator extraction happening through the shared adapter without error. Acceptance criteria: all code imports indicators through `src.tech` modules, no duplicated extraction helpers remain, and documentation points to the new structure.

## Idempotence and Recovery

Creating the new package and moving files is idempotent as long as changes are committed incrementally. If a move goes wrong, git can restore previous files via `git checkout -- <path>` before the migration is complete. Running `make test` is safe to repeat. Backups of removed modules should exist in git history, so full rollback merely requires reverting the relevant commit.

## Artifacts and Notes

Capture the final directory tree of `src/tech` and the `make test` output once implementation finishes. Paste short, indented snippets here to document success (for example, the migration warning emitted from `src/indicators/__init__.py` or the log line showing trading engines loading the new adapter). This section is empty until work is executed.

## Interfaces and Dependencies

At completion, the following contracts must exist:

- `src/tech/indicators/core.py` exports pure functions: `calculate_moving_averages(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame`, `calculate_rsi(data: pd.DataFrame | pd.Series, period: int = 14) -> pd.Series`, `calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame`, `calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame`, `calculate_macd(...)`, `detect_market_regime(...)`, `calculate_support_resistance(...)`, and `calculate_ema(series: pd.Series, period: int = 9) -> pd.Series`.
- `src/tech/features/technical.py` (new) wraps `core` to provide ML feature extraction utilities that prediction and other modules can import without pulling in prediction-only caching logic.
- `src/tech/adapters/row_extractors.py` defines `extract_indicators(df: pd.DataFrame, index: int) -> dict[str, float | str]`, `extract_sentiment_data(...)`, and `extract_ml_predictions(...)`, with strict handling of NaNs and numeric casting.
- `src/indicators/__init__.py` remains as a compatibility shim temporarily but must import from `src.tech.indicators.core` and raise a `DeprecationWarning` documenting the new path.
- Documentation in `docs/tech_indicators.md` and `docs/prediction.md` explains where to add new indicators and how `src.tech` fits into the architecture.

Document any future deviations or enhancements in the Decision Log and Progress sections as they arise.
