# Documentation Audit Report - 2025-11-27

**Audit Type:** Nightly Maintenance  
**Repository:** AI Trading Bot  
**Branch:** local workspace (detached HEAD)

---

## Executive Summary

Focused this pass on two problem spots uncovered during the registry refactor follow-up: the component strategy guide still referenced the removed `model_path` workflow and the top-level test README pointed to files/scripts that no longer exist. Both documents now describe the registry-based selection flow and the current test layout/commands. All other docs under `docs/` and the module READMEs were spot-checked—no additional discrepancies were found.

### Overall Status: ✅ PASS (documentation-only updates)

---

## Scope & Findings

| Area | Finding | Resolution |
| ---- | ------- | ---------- |
| `src/strategies/components/README.md` | Examples still instructed readers to load ONNX artifacts via `model_path` and `use_prediction_engine`, contradicting the registry-only code. | Rewrote the ML signal generator examples to select bundles via `PredictionModelRegistry`, documented the `StrategyModel.key` format, and refreshed the usage snippets + metadata stamp (2025-11-27). |
| `tests/README.md` | Referenced non-existent files (`tests/test_indicators.py`, `automated_performance_monitor.py`, etc.) and stale performance commands, causing broken copy/paste examples. | Re-authored the guide to mirror the actual folder structure, supported helper commands, available markers, coverage workflow, and the current `tests/performance/performance_benchmark.py` harness. |
| Remaining docs (`docs/*.md`, module READMEs) | Verified links, setup steps, and CLI snippets against the latest CLI commands and strategy APIs. | No changes required; kept "Last Updated" metadata untouched to avoid implying edits that did not occur. |

---

## Validation

- `rg "model_path" src/strategies/components/README.md` → no matches (ensures legacy instructions were removed).
- `rg "automated_performance_monitor" tests/README.md` → no matches (confirms stale references dropped).
- Manual review of `PredictionModelRegistry` helpers and `tests/run_tests.py` to make sure the documented commands exist.

---

## Recommendations

1. Update the `benchmark` command inside `tests/run_tests.py` to point at `tests/performance/performance_benchmark.py` so the helper matches the documentation again.
2. Keep module README snippets co-located with future strategy or CLI refactors to prevent drift.
3. Continue nightly sweeps for TODO/FIXME markers in user-facing docs (none were found today).

---

**Audit Completed:** 2025-11-27  
**Status:** Complete (docs only)  
**Risk:** None
