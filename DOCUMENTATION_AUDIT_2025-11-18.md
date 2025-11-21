# Documentation Audit Report - 2025-11-18

**Date**: November 18, 2025  
**Auditor**: AI Agent (Nightly Maintenance)  
**Branch**: `cursor/nightly-documentation-audit-and-update-0a03`

## Executive Summary

Focused this cycle on the data access and model registry docs after discovering they still referenced the pre-Parquet cache format
and legacy model layout. Updated the affected guides/READMEs to match the current `CachedDataProvider` implementation and structured
bundle contents. Verified the `atb models` workflows that the docs now highlight.

## Scope

### Files Reviewed
- `docs/data_pipeline.md`
- `src/data_providers/README.md`
- `docs/prediction.md`
- `src/ml/models/README.md`
- Spot-check: `cli/commands/data.py`, `src/data_providers/cached_data_provider.py`, `src/prediction/models/registry.py`

### Areas Assessed
1. Cache storage format and CLI coverage
2. Model registry structure and management commands
3. Consistency between CLI examples and actual argument order
4. TODO/FIXME sweep for user-facing docs

## Findings

### 1. CachedDataProvider docs lagged actual Parquet implementation ⚠️
- **Observation**: `docs/data_pipeline.md` and `src/data_providers/README.md` still described yearly pickle files, omitted the hash-
  based Parquet layout, and failed to mention the `cache-manager clear` / `populate-dummy` subcommands.
- **Resolution**: Documented the Parquet format, deterministic hashes, fallback directory behaviour, and the full CLI surface
  (including `--force-refresh`, `--test-offline`, `clear`, and `populate-dummy`).

### 2. Model registry guides referenced deprecated structure ⚠️
- **Observation**: `docs/prediction.md` and `src/ml/models/README.md` claimed bundles only stored `model.onnx` + `metadata`,
  suggested `model_type=price` in CLI examples, and implied legacy directories were still scanned.
- **Resolution**: Refreshed both files with the actual artifact list (`model.keras`, `saved_model/`, `feature_schema.json`,
  optional `metrics.json`), corrected the CLI examples to use `basic`, and clarified that the registry exclusively loads the
  structured layout with `latest/` symlinks.

### 3. TODO/FIXME scan ✅
- No actionable TODO/FIXME markers found in user-facing docs; remaining references live inside planning/constitution guides.

## Changes Made

| File | Description |
| ---- | ----------- |
| `docs/data_pipeline.md` | Updated cached access section for Parquet storage, documented deterministic hashes/fallbacks, and expanded CLI coverage (`clear`, `populate-dummy`, `--force-refresh`, `--test-offline`). |
| `src/data_providers/README.md` | Synced summary with the Parquet implementation and reminded readers to share cache directories with the CLI tools. |
| `docs/prediction.md` | Documented full bundle contents, corrected CLI examples (`atb models compare/promote`), and noted optional `metrics.json` usage. |
| `src/ml/models/README.md` | Rebuilt the registry README to reflect the structured hierarchy (artifacts + `latest` symlink) and clarified that legacy flat directories are no longer scanned. |

## Validation

- ✅ `python -m cli models list` – ensured the registry loader enumerates the bundles referenced in the docs.
- ✅ `python -m cli models compare BTCUSDT 1h basic` – confirmed the compare example runs (returns `{}` when `metrics.json` is absent).
- ✅ Markdown preview / lint – verified formatting and relative links.

## Recommendations

1. Consider adding `metrics.json` generation to the training pipeline so the CLI compare output carries richer summaries.
2. Keep documentation edits co-located with future cache or registry refactors to avoid format drift.
3. Re-run the cache-manager subcommands after the next cache schema change to make sure user-facing messaging stays accurate.

---

**Audit Completed**: 2025-11-18  
**Status**: Passed (documentation updates only)  
**Risk Level**: None – documentation-only changes
