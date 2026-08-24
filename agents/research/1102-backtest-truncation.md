# #1102 — Backtests silently truncate at the drawdown cap with no marker

Status: fixed. Branch `fix/1102-backtest-drawdown-truncation`, worktree
`.claude/worktrees/fix-1102-drawdown-truncation`.

## The defect (as found)

PR #1073 made `RiskParameters()` default-hydrate `max_drawdown` from the
ratified `src/config/risk-limits.json` (`portfolio.max_drawdown_pct = 0.20`).
The backtest engine (`src/engines/backtest/engine.py`, `_run_main_loop`) has
always early-stopped a run the instant running drawdown crosses
`risk_manager.params.max_drawdown` — it breaks the main loop, and the results
dict only carried `early_stop_reason` (`str | None`), `early_stop_date`, and
`early_stop_candle_index`. Nothing in the payload said, in an unambiguous
boolean, "this run is a partial result." A caller that didn't specifically
check `if results["early_stop_reason"]:` — including the reporting layer in
`src/experiments/reporter.py`, which never looked at it at all — had no way
to distinguish a complete run from one that silently stopped 40% of the way
through the requested window.

Before #1073, `RiskParameters()`'s bare default was a hardcoded ~50% (see
`_early_stop_max_drawdown`'s `else 0.5` branch, still present for the
`risk_parameters is None` case — see "Related but not fixed" below), so most
backtests never got close enough to trip the stop and the gap went
unnoticed. After #1073, the default is 0.20, and any strategy whose honest
drawdown exceeds that — HyperGrowth's 365d backtest measures ~21.8% MaxDD —
hits the cap. This was root-caused during the #1081 Step-2 short-suppression
re-run (PR #1101): two of thirteen re-run arms didn't match their originals,
because the original had run through a code path (bare
`ExperimentRunner.run()` → `RiskParameters()` with no override) that used to
tolerate ~50% drawdown and, after #1073, tolerates 20% — silently cutting the
re-run short partway through.

## Why "just remove the cap" is wrong

The early stop is *correct* behaviour for a live-representative simulation —
prod really does latch close-only once drawdown crosses the ratified limit
(`src/engines/live/monitoring/drawdown_guard.py`, shipped in #849/#851). A
backtest meant to answer "would this be safe to promote" should reproduce
that halt. But a backtest meant to answer "what does this strategy's
drawdown profile actually look like" needs the *opposite* behaviour — running
straight through the point prod would have stopped, so the researcher can see
how bad it gets and whether it recovers. One behaviour, applied
unconditionally with no signal about which question was being answered, was
the actual defect: not that the cap fires, but that firing and not firing
look identical from the output.

## The fix

### `src/engines/backtest/engine.py`

- New constructor parameter `drawdown_cap_mode: Literal["enforce", "measure"] = "enforce"`.
  - `"enforce"` (default): unchanged behaviour — halts the loop the instant
    drawdown crosses the threshold. This is the safe default: every existing
    caller that doesn't pass the new parameter keeps today's live-representative
    behaviour with zero silent change.
  - `"measure"`: does not break the loop. Instead it records the first
    breach (`drawdown_cap_breached`, `drawdown_cap_breach_date`,
    `drawdown_cap_breach_candle_index`) and keeps running to the end of the
    requested window.
- Invalid values raise `ValueError` immediately (fail fast, not silent
  fallback).
- Every result dict (`_build_final_results` and `_build_empty_results`) now
  always carries:
  - `early_stopped: bool` — the actual fix. Check this, not whether
    `early_stop_reason` is truthy.
  - `drawdown_cap_mode: str` — which mode produced this result, so a
    consumer never has to guess.
  - `drawdown_cap_threshold: float` — the cap value actually in force
    (mirrors what `_early_stop_max_drawdown` resolved to).
  - `drawdown_cap_breached: bool`, `drawdown_cap_breach_date`,
    `drawdown_cap_breach_candle_index` — populated in both modes; in
    `"enforce"` mode this is redundant with `early_stopped` (the run stopped
    at first breach), in `"measure"` mode this is the answer the research run
    needed.
- `_reset_run_state()` (used when a `Backtester` instance is `run()` more than
  once) resets all of the above so results never leak across runs.

### `cli/commands/backtest.py`

- New `--drawdown-cap-mode {enforce,measure}` flag, default `enforce`.
- The results banner now prints `Drawdown Cap Mode: ...` always, and when
  `early_stopped` is true prints a loud
  `WARNING: RUN WAS TRUNCATED` block with the reason, stop time, and a
  pointer to `--drawdown-cap-mode measure`. When the run wasn't truncated but
  did cross the cap under `measure` mode, it prints a one-line note instead.
  This also fixes a stale claim in `docs/backtesting.md` ("The run prints the
  stop reason") that was never actually true — the CLI never printed
  `early_stop_reason` before this change.

### `src/experiments/schemas.py` / `src/experiments/runner.py`

This is the actual repro path for #1081: `ExperimentRunner.run()` builds a
bare `RiskParameters()` whenever `config.risk_parameters` doesn't specify
`max_drawdown`, which now hydrates to the ratified 0.20.

- `ExperimentConfig` gained `drawdown_cap_mode: str = "enforce"` so a
  preregistered research study can opt into the full-window measurement
  explicitly and record that choice in its preregistration
  (`docs/research/experiments/...`), per `experiment-preregister`.
- `ExperimentRunner.run()` forwards `config.drawdown_cap_mode` to the
  `Backtester`.
- `ExperimentResult` gained `early_stopped`, `drawdown_cap_mode`,
  `drawdown_cap_breached`, `drawdown_cap_breach_date`,
  `drawdown_cap_threshold`, populated from the backtest results — mirroring
  the existing `effective_sizing` pattern from the #1088 fix (reported ==
  enforced, by construction, not by convention).

### `src/experiments/reporter.py`

- New `_detect_truncation_warnings(result, name, baseline=None)` helper,
  wired into `ExperimentReporter.render()` for both the baseline row and
  every variant row (reusing the existing `VariantReport.warnings` seam
  built for the #1088-class G6/G7 "identical to baseline" detector).
  - Emits a `TRUNCATED` warning on any row whose result was truncated.
  - Emits a `TRUNCATION MISMATCH vs baseline` warning when a variant's
    `early_stopped` disagrees with the baseline's — this is exactly the
    #1081 Step-2 failure mode: comparing a truncated re-run against an
    untruncated original with no signal that either was partial.
- Both `render_text()` and `to_dict()` (JSON) already surface
  `VariantReport.warnings`, so this required no further plumbing.

### Docs

- `docs/backtesting.md` "Safety limits" section rewritten — it previously
  claimed a 50% default and that the CLI prints the stop reason, neither of
  which was true post-#1073. Now documents `--drawdown-cap-mode` and both
  modes' semantics.
- `docs/changelog.md` — Unreleased/Fixed entry.

## Callers checked

Grepped every `Backtester(` construction site (`grep -rn "Backtester("`)
across `src/`, `cli/`, and `tests/`: ~100 call sites, all in test files plus
`src/experiments/runner.py`, `cli/commands/backtest.py`, and
`cli/commands/migration.py`. None pass a `drawdown_cap_mode` keyword prior to
this change, so all of them keep exactly today's behaviour (the new
parameter defaults to `"enforce"`, identical to the pre-#1102 unconditional
early stop). `cli/commands/migration.py`'s `Backtester(...)` call was left
untouched deliberately — it's a data-migration utility, not a research or
promotion path, and doesn't need the new flag.

## Other silent-truncation / silent-cap paths found (not fixed here)

Per the request to grep for siblings of this defect class:

1. **`#1089` (open, P2, already tracked)** —
   `Backtester._early_stop_max_drawdown` falls back to a **hardcoded 0.5**
   when `risk_parameters is None` (`engine.py`, the
   `if risk_parameters is not None else 0.5` branch), instead of reading
   `self.risk_manager.params.max_drawdown` — which, because
   `RiskManager(None)` internally constructs a bare `RiskParameters()`, would
   actually resolve to the same ratified 0.20 the enforced path uses
   everywhere else. This is the same "reported diverges from enforced"
   defect class as #1088 (fixed) and #1102 (this fix), just for the *default
   threshold value* rather than the *truncation signal*. Deliberately left
   alone here: #1089 is already filed, correctly scoped separately, and
   fixing it changes actual default backtest behaviour (a bare `Backtester()`
   would start halting at 20% instead of 50%), which deserves its own
   backtest-behavior-change review rather than folding into this
   truncation-visibility fix. Recommend picking it up next — it interacts
   directly with `drawdown_cap_mode`: today a bare `Backtester()` in
   `"measure"` mode would report `drawdown_cap_threshold: 0.5`, which is a
   misleading "cap" to measure against once #1089 exists as a known
   divergence.
2. **`AccountCircuitBreaker`** (`src/risk/circuit_breaker.py`, wired into the
   backtest engine at `engine.py:508`) governs a different kind of halt
   (new-entry suppression under a `dry_run`/`live` mode flag,
   `account_circuit_breakers`, default `off`). It does not truncate the
   run — the backtest continues to the end of the window either way — so it
   is not the same defect class. Not investigated further; flagging only
   because it's the other "risk gate that can silently change what a
   backtest measures" mechanism in the engine, worth a second look if
   similar "reproducibility" confusion shows up again.
3. Grepped for other `is not None else <hardcoded literal>` risk-parameter
   fallbacks and other unconditional `break` statements in
   `_run_main_loop` — found none beyond the one entry above. The
   `max_holding_hours` forced-exit path and `TimeExitPolicy` close individual
   trades, not the run; they don't truncate the window.
4. **Re-checked after the finding was escalated** (2026-08-24, once #1108
   close-to-free-balance and #1109 stop-loss-to-free-balance confirmed this
   "reports success while silently doing less" shape is systemic in the live
   path): grepped every `break` statement in the backtest engine package
   (`src/engines/backtest/**/*.py`, excluding tests). Only one whole-*run*
   `break` exists — the one this PR fixes, in `_run_main_loop`. The other
   `break` statements found (`src/engines/backtest/execution/exit_handler.py`,
   inside `_process_partial_operations`) terminate a bounded inner
   `while iteration_count < MAX_PARTIAL_EXITS_PER_CYCLE` loop over partial
   exits/scale-ins for the *current trade only* — they stop iterating once
   `should_exit` is false, the position is fully closed, or the remaining
   size rounds to ~0, all logged at `debug`. These do not truncate the
   backtest window or silently drop the rest of the run; they are ordinary
   loop termination, not the reported-vs-actual divergence class. Also
   grepped for `clamp` — the one hit
   (`src/engines/backtest/execution/position_tracker.py`, partial-exit size
   clamped to remaining position size) is logged at `warning` with both the
   requested and clamped values, so it's already visible, not silent.
   **No other silent-cap or silent-early-stop path was found in the backtest
   engine.** The one instance of the class that exists here (#1102 itself)
   is fixed by this PR; #1089 (below) is the one *adjacent* defect
   (wrong-value-used, not silent-truncation) still open.
5. **`system_events` severity casing** (asked about directly, since a
   miscased column value would make a fix for a silent-failure bug itself
   fail silently): this fix does not write to `system_events` at all — it's
   backtest-engine-only, no DB writes, no live-path code touched. Grepped
   `git diff` for `system_events` across every file this PR changes;
   zero matches. Not applicable here, but noted for whoever picks up the
   `#1108`/`#1109` live-path fixes: the column is lowercase (`'critical'`)
   and a query built with a different case will silently match nothing.

## How verified

- New deterministic unit test
  (`tests/unit/backtesting/test_backtesting_comprehensive_edge_cases.py::TestRiskManagementEdgeCases::test_drawdown_cap_mode_measure_does_not_truncate`)
  using a scripted `Strategy` (fixed BUY/SELL cycle signal generator, full
  balance allocation, zero fees/slippage) against synthetic price data
  engineered so a 2nd trade close crosses the 20% cap. Confirms: `"enforce"`
  stops there (`early_stopped=True`); `"measure"` on identical data does not
  stop (`early_stopped=False`), reports the same breach date
  (`drawdown_cap_breach_date` equals the enforced run's `early_stop_date`),
  and executes strictly more trades than the enforced run (proving it
  actually kept going, not just that a flag flipped).
- Updated `test_max_drawdown_exceeded_early_stop` to assert the new
  `early_stopped` boolean and the threshold/breach fields alongside the
  existing (flaky-by-design, ML-strategy-dependent) reason-string check.
- New `test_drawdown_cap_mode_rejects_invalid_value` — fail-fast on garbage
  input.
- New `TestRunnerDrawdownCapModePlumbing` class in
  `tests/unit/experiments/test_runner_risk_seeding.py` (mocked-`Backtester`
  harness, mirroring the existing `effective_sizing` tests): confirms the
  harness default is `"enforce"`, an explicit `"measure"` config value is
  forwarded verbatim, and `ExperimentResult` correctly mirrors a truncated
  vs. a complete `Backtester` result.
- New tests in `tests/unit/experiments/test_reporter.py`: a truncated
  variant gets a `TRUNCATED` warning (row and rendered text), a
  baseline/variant truncation mismatch gets a `MISMATCH` warning, and neither
  fires when nothing was truncated.
- `atb test unit` (full suite): 5918 passed, 1 skipped, 1 failed
  (`tests/unit/ml/training_pipeline/test_models_tft.py::TestModelFactoryIntegration::test_create_model_lightgbm_dispatches_to_directional_classifier`
  — `Failed: DID NOT RAISE <class 'ImportError'>`). This is a known
  pre-existing lightgbm-related failure unconnected to this change: it lives
  in the ML training pipeline, imports none of the files this PR touches,
  and was flagged as pre-existing independently of this work.
- `atb dev quality --changed` (black, ruff, mypy) on the 8 changed source/test
  files: all pass. `bandit -r` on the same 5 source files: clean.
- `atb test unit` also caught two real regressions from an earlier pass of
  this fix, both corrected before the run above: (1) `cli/commands/backtest.py`
  read `ns.drawdown_cap_mode` unconditionally, which raised `AttributeError`
  in `tests/unit/cli/test_backtest.py`'s hand-built `argparse.Namespace`
  fixtures that predate the new flag — changed to
  `getattr(ns, "drawdown_cap_mode", "enforce")`; (2) `mypy` caught
  `ExperimentConfig.drawdown_cap_mode: str` being passed to `Backtester`'s
  `Literal["enforce", "measure"]` parameter — narrowed the field's type to
  the same `Literal` and added an explicit `cast` at the one point a
  `Backtester` result dict's string value flows back into
  `ExperimentResult`.

## Follow-ups suggested to `pm`

- Pick up `#1089` next (see above) — now that `drawdown_cap_mode="measure"`
  exists and reports `drawdown_cap_threshold`, its bare-`Backtester()` 0.5
  fallback is a more visible inconsistency than it was before.
- `experiment-preregister` skill should be updated to require a study state
  which `drawdown_cap_mode` it used, per the original issue's proposed fix
  #2 — not done here since it's a skill-content change outside this PR's
  code diff, flagged for `pm`/skill owner.
