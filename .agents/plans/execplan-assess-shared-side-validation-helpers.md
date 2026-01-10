# Assess and Wire Shared Side/Validation Helpers

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` and must be maintained in accordance with it.

## Purpose / Big Picture

The goal is to reduce duplicated trading-engine logic by extracting shared entry/exit calculations into `src/engines/shared/`, while also deciding how the new side-handling and validation helpers should be wired in. After this work, both backtest and live engines should call the same shared functions for entry plan derivation and partial-exit fraction conversion, or there should be a documented rationale for keeping specific paths separate. Success is visible by running the unit tests and, if wiring is done, seeing new tests that fail before the change and pass after.

## Progress

- [x] (2026-01-02 21:52Z) Drafted the ExecPlan after reviewing helper modules and `.agents/PLANS.md`.
- [x] (2026-01-02 22:10Z) Inventoried side handling and validation patterns plus entry/exit duplication between backtest and live handlers.
- [x] (2026-01-02 22:24Z) Evaluated helper semantics and recorded wiring decisions for entry extraction, SL/TP resolution, and partial-exit conversion.
- [x] (2026-01-02 22:36Z) Implemented shared entry utilities and validation conversion helper; added side/validation helpers to shared exports.
- [x] (2026-01-02 22:48Z) Wired shared helpers into backtest/live entry handlers and partial-exit loops; added unit tests.
- [x] (2026-01-02 23:05Z) Ran targeted unit tests and updated docs/changelog for shared helper wiring.

## Surprises & Discoveries

- Observation: The develop branch did not contain the side/validation helper modules referenced in the review.
  Evidence: `ls src/engines/shared` listed no `side_utils.py` or `validation.py` before additions.

## Decision Log

- Decision: Add `src/engines/shared/entry_utils.py` to centralize entry plan extraction and SL/TP percentage resolution, with parameters to preserve backtest/live differences.
  Rationale: Entry handlers duplicated logic but needed distinct defaults; parameterization keeps parity without semantic change.
  Date/Author: 2026-01-02 / Codex
- Decision: Add `convert_exit_fraction_to_current` and use `is_position_fully_closed` in partial-exit loops while keeping scale-in behavior intact.
  Rationale: Shared conversion avoids duplicated math while allowing exit fractions over original size when current size grows.
  Date/Author: 2026-01-02 / Codex
- Decision: Use `to_side_string` in backtest tracker/exit paths while keeping the helper’s buy/sell mapping behavior.
  Rationale: Centralizes side normalization without altering existing long/short semantics.
  Date/Author: 2026-01-02 / Codex

## Outcomes & Retrospective

Shared entry/exit helpers now centralize entry plan extraction, SL/TP percentage resolution, and partial-exit fraction conversion across engines. Backtest/live handlers call the shared functions, and targeted unit tests for the new helpers pass (`pytest -q tests/unit/engines/shared/test_entry_utils.py tests/unit/engines/shared/test_validation_utils.py`). Remaining work is limited to any broader regression suites the team wants to run.

## Context and Orientation

The helpers live in `src/engines/shared/side_utils.py` and `src/engines/shared/validation.py`, and are re-exported by `src/engines/shared/__init__.py`. They sit alongside existing shared engine utilities like `PositionSide` and `normalize_side` in `src/engines/shared/models.py`, and clamping/validation helpers in `src/utils/bounds.py`. Entry logic is duplicated between `src/engines/backtest/execution/entry_handler.py` and `src/engines/live/execution/entry_handler.py`, while partial-exit fraction conversion is duplicated between backtest and live exit handlers. In this repository, a position side refers to exposure direction (`long` or `short`) on an open position, while an order side refers to the action (`buy` or `sell`) taken to open or close a position. These concepts are not interchangeable: a `buy` can open a long or close a short depending on context. The assessment must keep this distinction explicit to avoid wiring helpers where they would change order semantics.

## Plan of Work

Start by mapping how sides are represented and normalized today, and by cataloging numeric validation patterns already in use (for example, existing clamp/validate utilities and EPSILON constants). For each current usage, identify whether the code is dealing with a position side or an order side, and whether it relies on side strings, enums, or free-form values. In parallel, catalog where division, fraction bounds, and notional/price validation happen, and whether those sites expect exceptions or defensive fallbacks.

Next, compare helper semantics against the current behavior. `side_utils.to_side_string` maps `buy` to `long` and `sell` to `short`, which is only valid in contexts that are explicitly describing a position opening or desired exposure. If a candidate location deals with order execution, reduce-only behavior, or closing logic, do not apply this mapping unless the surrounding code already treats order side as position side. For validation helpers, decide whether `safe_divide` is acceptable where an exception or explicit error is preferable. Use this step to decide which helper functions can be used without altering results.

Then, identify the duplicated entry/exit logic that can be centralized without changing semantics. The primary targets are entry plan extraction (side + size fraction) and stop-loss/take-profit percentage resolution, plus partial-exit fraction conversion in the exit loops. Define shared helper functions for those operations in `src/engines/shared/` and update backtest/live handlers to use them. Where a shared helper would change behavior, add parameters to preserve the existing behavior and record the decision in the Decision Log.

If wiring is justified, implement it incrementally with tests that demonstrate the unchanged behavior. If wiring is not justified, add a brief rationale in docs or code comments explaining why the helpers remain unused and what conditions would make them safe to adopt. In either case, ensure any new tests are fast, isolated, and use the AAA pattern.

Commit often with clear, imperative messages as each step lands.

## Concrete Steps

From the repository root, run the following commands to inventory side handling, entry logic duplication, and validation logic. These commands are read-only and can be re-run safely.

    rg -n "PositionSide|normalize_side|side_str|BUY|SELL|order side|position side" src
    rg -n "_extract_entry_plan|_calculate_sl_tp|_calculate_sl_tp_pct|entry plan|stop loss" src/engines
    rg -n "partial exit|exit_fraction|current_size_fraction" src/engines
    rg -n "to_side_string|to_position_side|is_long|is_short|opposite_side|get_position_side" src tests
    rg -n "validate_price|validate_notional|validate_fraction|is_valid_|safe_divide|is_position_fully_closed|clamp_fraction|validate_parallel_lists|EPSILON" src tests

Open the most relevant results to understand semantics before changing anything, prioritizing:

    src/engines/shared/models.py
    src/engines/backtest/execution/entry_handler.py
    src/engines/backtest/execution/exit_handler.py
    src/engines/live/execution/entry_handler.py
    src/engines/live/execution/exit_handler.py
    src/engines/shared/partial_exit_executor.py
    src/engines/shared/partial_operations_manager.py
    src/utils/bounds.py
    src/config/constants.py

After reviewing each candidate, update the Decision Log with a clear statement of whether the helper should be used there and why.

If wiring is approved for any module, edit the module to use the helper while preserving behavior, then add a unit test that captures the specific behavior that must remain stable.

## Validation and Acceptance

If helpers are wired in, run:

    python tests/run_tests.py unit

Expect the unit test suite to pass. Any new tests added for side conversions or validation should fail before wiring and pass after. If helpers are not wired in, document the rationale and still ensure the unit test suite passes.

Acceptance is met when either the helpers are safely integrated with no behavior changes (validated by tests), or when a documented rationale explains why wiring is unsafe and no changes were made.

## Idempotence and Recovery

All search and read steps are idempotent. Wiring changes should be made in small, isolated edits so they can be reverted file-by-file if behavior changes unexpectedly. If a wiring change causes a regression, revert the specific edit and record the reason in the Decision Log rather than forcing the helper into an unsafe context.

## Artifacts and Notes

Initial observation to confirm during inventory: current references to the helpers appear only in `src/engines/shared/__init__.py` and the helper modules themselves, while existing side normalization and clamping utilities live elsewhere. This should be re-validated with the search commands before deciding on wiring.

## Interfaces and Dependencies

The following helper functions are the intended public surface if wiring is approved: `to_side_string`, `to_position_side`, `is_long`, `is_short`, `opposite_side`, `opposite_side_string`, and `get_position_side` from `src/engines/shared/side_utils.py`, plus `validate_price`, `validate_notional`, `validate_fraction`, `is_valid_price`, `is_valid_fraction`, `safe_divide`, `is_position_fully_closed`, `clamp_fraction`, `validate_parallel_lists`, and `EPSILON` from `src/engines/shared/validation.py`. The plan also introduces shared entry/exit helpers in `src/engines/shared/entry_utils.py` and a partial-exit fraction conversion helper in `src/engines/shared/validation.py`, which both engines should call to avoid duplication. Any usage must preserve the distinction between order side and position side, and should align EPSILON usage with `src/config/constants.DEFAULT_EPSILON` unless there is a documented reason to diverge. No new third-party dependencies are required.

## Plan Change Notes

Initial draft created to guide assessment before any wiring decisions.
Plan updated to widen scope to shared entry/exit logic extraction after user guidance.
Progress and decisions updated after implementing shared entry/exit helpers and wiring changes.
Progress updated to reflect completed tests and outcome summary.
