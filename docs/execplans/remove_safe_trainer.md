# Remove Safe Model Trainer and Legacy Keras Artifacts

## Purpose / Big Picture

Production model deployments currently rely on symbolic links under `src/ml/models/**/latest`, so once a new bundle is written there every strategy and live runner immediately starts consuming it. The older `SafeModelTrainer` staged models in `/tmp/ai-trading-bot-staging` and copied flat `.onnx/.h5` files into `src/ml`, but the new training pipeline already publishes versioned bundles and updates the `latest` links, making the staging flow redundant. This plan explains how to retire `src/ml/safe_model_trainer.py`, remove its CLI hooks, and delete the leftover `.h5` artifacts and backup logic. When finished, there will be a single training pathway (`atb train`), no unused files in `src/ml`, and the live-control tooling will detect new models straight from the registry.

## Scope and Non-Goals

In scope:
- Remove `SafeModelTrainer` and its CLI entry (`atb train safe`) plus any call sites (notably the live-control helpers in `cli/commands/live.py`).
- Eliminate references to `.h5` artifacts in backups or docs, and purge the stale files from `src/ml`.
- Confirm that live and backtest code paths consult only the registry (`src/ml/models`) so automatic pickup works without the safe trainer.
- Update documentation and scripts that previously mentioned the safe trainer.

Out of scope:
- Changing the training pipeline introduced in this branch—only integration points and cleanup are targeted.
- Altering live deployment mechanics beyond removing the staging flow; the plan assumes the live stack already reads `latest` from the registry.

## Context and Definitions

- **Model registry**: `src/ml/models/{SYMBOL}/{TYPE}/{VERSION}` plus the `latest` symlink used by the prediction engine (`PredictionModelRegistry`) and live/backtest workflows.
- **SafeModelTrainer**: legacy script that backed up flat artifacts (`.onnx`, `.keras`, `.h5`, metadata) and ran legacy scripts from `scripts/`. Those scripts are deprecated; the new CLI trainer saves straight into the registry.
- **Live-control tooling**: commands under `cli/commands/live.py` that allow operators to stage or deploy models. They currently import `SafeModelTrainer` and need to be reworked or removed.

## Progress

- [x] (2025-10-27 21:10Z) Authored this ExecPlan, captured goals, scope, and background.
- [x] (2025-10-27 22:05Z) Inventory every reference to `SafeModelTrainer` and `.h5` artifacts across CLI, docs, and tests; confirmed only `cli/commands/live.py`, `cli/commands/train.py`, and documentation depended on them.
- [x] (2025-10-27 22:10Z) Designed a registry-first live-control workflow that calls the modern trainer, repoints `latest` symlinks, and exposes registry listings.
- [x] (2025-10-27 22:40Z) Removed `SafeModelTrainer`, updated CLI/live commands, pointed `StrategyManager` at the registry, and deleted the legacy `.h5` assets.
- [x] (2025-10-27 22:45Z) Updated docs to describe the new live-control flow and note that only the registry and symlinks are required.
- [ ] Run regression tests (`python tests/run_tests.py unit`, targeted integration checks) and verify live/backtest commands still discover new bundles automatically.

## Surprises & Discoveries

- Python 3.9 cannot import `src.prediction.models.registry` because that module uses the `dict[...] | None` union syntax. Full unit test runs therefore need Python 3.11 (the version required by `pyproject.toml`). Targeted tests were executed under Python 3.11 for validation.

## Decision Log

Decision: Retire SafeModelTrainer entirely and rely on the registry `latest` symlink for live deployment; live-control commands now invoke the modern trainer directly and repoint symlinks when asked to "deploy" a version.
Rationale: The new training pipeline already writes versioned bundles and updates `latest`, making the staging flow redundant while the registry satisfies rollback/rollback requirements by preserving old versions.
Date/Author: 2025-10-27 / Codex agent.

## Plan of Work

1. **Reference audit**: Use `rg` to list every import of `safe_model_trainer` and every `.h5` mention. Note which commands (`atb train safe`, live deployment helpers, docs) would break after removal. Confirm no automation still calls `scripts/train_model.py`.

2. **Live-control redesign**: Decide how operators will trigger safe deployments. Options:
   - Call the new `atb train model` with proper flags and rely on versioned bundles plus `latest` symlink.
   - Introduce a thin wrapper in `cli/commands/live.py` that invokes the new trainer and performs backups inside the registry (copying `latest` to `backup/`). Document the chosen approach.

3. **Implementation**:
   - Delete `src/ml/safe_model_trainer.py` and remove `_handle_safe` registration in `cli/commands/train.py` and related CLI entries.
   - Update live-control commands (`cli/commands/live.py`, `StrategyManager` usage) to use the chosen replacement or drop staging functionality if unnecessary.
   - Remove backup code copying `.h5` files; if backups are still desired, switch to copying the registry version directory.
   - Delete the `.h5` artifacts from `src/ml` and ensure no code expects them.

4. **Documentation**: Edit `docs/prediction.md`, `README.md`, and any runbooks mentioning the safe trainer or `.h5` exports. Explain the new workflow for swapping models in production and how to roll back (e.g., repoint `latest` to a previous version).

5. **Validation**:
   - Run unit tests and targeted integration tests (`python tests/run_tests.py unit`, `python tests/run_tests.py integration` if available).
   - Manually run `atb train model` and verify the registry updates, ensuring live-control commands still display or deploy models correctly.

6. **Cleanup & Follow-up**: Ensure no `import safe_model_trainer` remains. Confirm `.h5` assets are gone from the repo. Update the plan’s Progress, Surprises, and Decision sections with outcomes.

## Outcomes & Retrospective

- SafeModelTrainer and the `atb train safe` CLI entry were removed; live-control commands now invoke the modern trainer, repoint registry symlinks, and surface registry listings.
- Legacy `.h5` artifacts were deleted, and legacy flat `.onnx/.keras/metadata` files remain only as symlinks pointing into the versioned registry.
- Documentation (`docs/prediction.md`, `docs/live_trading.md`, `src/ml/README.md`) now references the single training workflow; a new unit test guards the registry path configuration.
- Validation: `python3.11 -m pytest tests/unit/training_pipeline/test_config_paths.py` and `tests/unit/training_pipeline/test_datasets.py` passed; full unit suites require Python ≥3.10 due to union typing in `src/prediction.models.registry`.

## Revision History

2025-10-27: Initial plan drafted to retire the safe trainer and tidy legacy artifacts.
2025-10-27: Executed plan—removed SafeModelTrainer, updated live-control CLI, deleted `.h5` assets, refreshed docs, and validated with targeted tests.
