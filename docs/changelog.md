# Changelog

All notable changes to the AI Trading Bot project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Maintainer Note**: This is a living document. Update after completing features, bug fixes, or significant changes. Use the `/update-docs` command to auto-populate entries.

---

## [Unreleased]

### Fixed
- **Exposure-governor pre-enablement fixes** (#802 follow-ups, from the PR merge
  note): (P2) `src/engines/shared/exposure.py::position_notional` now uses a
  position's `current_size` (the live fraction after partial exits / scale-ins)
  instead of the original `size`, so gross exposure is no longer overstated after
  a partial exit. (P3) scale-ins now respect the regime gross-exposure cap too:
  the exposure governor is shared with both engines' exit handlers and clamps a
  scale-in's added exposure to `scale_in_gross_cap_headroom` (conservative cap −
  current gross). Both remain inert unless `enable_exposure_governor` is on. This
  clears the two conditions the PM flagged before the governor can be enabled live.
- **P0: trading symbol now reaches the ML signal generator; cross-symbol model
  substitution is guarded** (2026-07-04 ml-engineer audit finding): the live
  runner and the backtest CLI constructed strategies with zero arguments, so
  `--symbol ETHUSDT` never reached `MLBasicSignalGenerator`, which defaulted to
  `BTCUSDT` for model registry selection — HyperGrowth live on ETHUSDT silently
  scored every bar with the BTCUSDT basic model. Both runners now thread the
  symbol through `call_strategy_factory()` (`src/strategies/__init__.py`) into
  every factory that composes `MLBasicSignalGenerator` (`hyper_growth`,
  `ml_basic`, `leveraged_regime`, `ensemble_weighted`, `StrategyFactory`
  presets); the generator normalizes it to the registry's Binance-style form
  (invalid symbols raise a clear `Invalid trading symbol` error at init). The
  hot-swap path is covered too: `StrategyManager` accepts the trading symbol
  (assigned by the startup sequencer at session start) and threads it through
  `_instantiate_strategy`, so regime-switcher hot-swaps select models for the
  traded pair — and a swap-time missing-model failure rejects the swap and
  keeps the current strategy instead of killing the trading loop.
  Guards at the generator/registry seam (identical in backtest and live):
  - **Fail fast at startup** when no model bundle exists for
    `(symbol, model_type, timeframe)` — the error lists available bundles
    instead of silently substituting the default model.
  - **`FEATURE_ALLOW_CROSS_SYMBOL_MODEL=true`** explicitly opts into the
    substitution: startup logs CRITICAL and pins a deterministic fallback
    bundle (same type/timeframe, `BTCUSDT` preferred), and every resolution
    logs a rate-limited WARNING. **Prod transition path**: production ETHUSDT
    has no `basic` model yet, so promoting this fix requires setting the flag
    temporarily — behavior is then *unchanged but loud* — until an ETHUSDT
    basic model ships, at which point the flag must be unset.
  - **Rate-limited ERROR on mismatch** (separate rate-limit clock per guard
    condition) whenever the resolved bundle's symbol differs from the trading
    symbol, and `Signal.metadata` is stamped with `trading_symbol` +
    `model_symbol` on every branch — including HOLD paths
    (`insufficient_history`, `prediction_failed`,
    `invalid_prediction_or_price`) — for auditability.
  - If the bundle vanishes after startup (registry reload), predictions fail
    safe (HOLD) instead of falling back to another symbol's model.
  Direct constructions without a symbol keep the `BTCUSDT` default.

### Added
- **Account circuit-breaker loop enforcement** (#807 follow-up): a new
  `CircuitBreakerEnforcer` (`src/engines/live/monitoring/circuit_breaker_enforcer.py`)
  runs the #807 `AccountCircuitBreaker` on every trading-loop iteration (mirroring
  `MaxDrawdownEnforcer`), completing the follow-ups flagged as human-sign-off:
  (1) **restart-safe daily baseline** — seeds the daily-loss baseline from the
  day's first `account_history` snapshot (`get_first_snapshot_of_day`) on boot, so
  an intraday restart no longer disarms the halt; (2) **halt on trip** — in
  `active` mode a trip flips the engine's existing **close-only mode** (new entries
  and scale-ins stop; exits/stop-losses keep running — nothing is liquidated,
  matching the `MaxDrawdownGuard` precedent), and in `dry_run` it logs "would
  halt"; (3) **surfacing** — emits a `risk_event` + a CRITICAL `system_events`
  ALERT (`ACCOUNT_CIRCUIT_BREAKER_TRIP`) for the dashboard/alerting, with the log
  signatures added to `.claude/LESSONS.md §5`. Fault-isolated (never crashes the
  loop). Still gated by `account_circuit_breakers` (default `off`). Deliberate
  non-goal: literal force-liquidation of open positions (the codebase does not
  liquidate into a dip; close-only is the safe halt).
- **Account-level circuit breakers** (#807): new `AccountCircuitBreaker`
  (`src/risk/circuit_breaker.py`) enforces hard, account-level safety limits
  independent of strategy logic — a **daily-loss halt** (default 2.5% below a
  UTC-day-anchored baseline → halt new entries for the day, latched) and a
  **drawdown halt** (default 15% peak-to-trough → halt until recovery within 5%
  of peak). Graduated drawdown throttling stays with dynamic-risk (no
  double-count). Wired into the shared `apply_pre_order_gates` seam so a halt
  blocks new entries in both engines and the legacy short path. Controlled by the
  **string** flag `account_circuit_breakers` ∈ `off` / `dry_run` (evaluate + log
  "would halt", no action) / `active` (block entries), read via `get_flag` and
  resolved once at build. Ships `off`. **Follow-ups requiring human sign-off**
  (money-mover / live integration): force-flatten of open positions on trip (vs
  the safe entry-block delivered here), DB-persisted daily baseline reload across
  restarts (a `seed_daily_baseline` hook is provided), and dashboard surfacing.
- **Event-aware de-risking windows** (#806): around high-impact macro events
  (FOMC, CPI) the bot now blocks new entries and halves regime exposure caps.
  New `MacroEventCalendar` / `MacroEventGuard`
  (`src/position_management/macro_events.py`) load a maintained calendar from
  `config/macro_events.json` (dates + per-event N-hours-before / M-hours-after
  window — config, not hardcoded; stale/empty is fail-safe). The guard plugs
  into the shared `apply_pre_order_gates` seam alongside the #802 exposure
  governor, so it applies identically in the backtest and live engines (and the
  legacy short path): inside a window `entry_allowed` is False (block) and
  `exposure_factor` is 0.5 (halves the governor's cap via `extra_factor`).
  Behind `enable_macro_event_guard` (**default OFF**).
- **Sentiment-extreme mean-reversion overlay + short block** (#804): at Fear &
  Greed extremes, fading beats following. New `SentimentExtremeOverlay`
  (`src/strategies/components/sentiment_overlay.py`) wraps the `ml_sentiment`
  signal generator and, when F&G < 15, **blocks new SHORT entries** (capitulation
  shorts get squeezed) and permits new LONGs only within a configurable band of a
  **structural support level** — a config *parameter* (`DEFAULT_SENTIMENT_SUPPORT_LEVEL`,
  default None = no band restriction), since market levels go stale. When F&G > 70
  in a downtrend it permits small fade shorts. Implemented as a `SignalGenerator`
  decorator, so it composes with the ETF flow gate (#803) — most-restrictive-wins,
  any veto → HOLD — and applies in both engines via the strategy. F&G comes from
  `FearGreedProvider` (degrades to neutral offline → overlay inert). Behind
  `enable_sentiment_extreme_overlay` (**default OFF**).
- **Volatility-targeted position sizing** (#805): a new `VolatilityTargetSizer`
  (`src/strategies/components/position_sizer.py`) wraps a base sizer and scales
  its output by `target_atr_percentile / atr_percentile` (from the regime
  detector) so per-position dollar-vol is roughly constant — smaller in high vol,
  larger in calm, bounded to avoid blow-ups. Passes through unchanged when the
  regime/ATR-percentile is unavailable (never guesses). Wired into
  `ml_basic`/`ml_adaptive` behind `enable_vol_target_sizing` (**default OFF**,
  requires regime detection). Also: `kelly_momentum` now clamps fractional Kelly
  to `DEFAULT_MAX_KELLY_FRACTION` (0.5) for bear safety (full/half Kelly
  over-sizes into drawdowns); and `momentum_leverage` / `hyper_growth` emit a
  startup warning that they are not recommended for bear/high-vol regimes.
- **ETF net-flow signal + flow gate** (#803): US spot BTC/ETH ETF net flows are
  the marginal buyer/seller this cycle, but the bot had no flow awareness. New
  `ETFFlowProvider` (`src/data_providers/etf_flow_provider.py`) ingests daily net
  flows, caches to parquet (atomic write), and degrades gracefully
  (fetch → cache → bundled seed) so it never hard-fails a loop. Derived features:
  5d/20d net-flow z-scores (regime-aware — a sustained outflow streak reads
  strongly negative) and consecutive-outflow-day count. A rule-based **gate**
  (`FlowGatedSignalGenerator`) vetoes NEW LONG entries while the 5-day flow
  z-score is below a configurable threshold (default −1.0), implemented as a
  signal-generator decorator so it applies in both engines via the strategy with
  no per-engine wiring (SELL/HOLD pass through; unknown flow does not block).
  Wired into `ml_basic`/`ml_adaptive` behind `enable_etf_flow_gate` (**default
  OFF**). A separate `ETFFlowFeatureExtractor` exposes the same features as
  optional model inputs but is **inert until a compatible model is retrained**
  (it changes the feature schema) — registered only behind
  `etf_flows_features.enabled`. See `docs/data_pipeline.md`.
- **Regime-gated gross exposure caps** (#802): a new `ExposureGovernor`
  (`src/strategies/components/exposure_governor.py`) caps *total gross open
  exposure* (sum of |entry notional| / current equity) by market regime — in a
  bear, exposure itself is the primary risk lever. Defaults (config, overridable):
  trend_down+high_vol 15%, trend_down+low_vol 20%, range 20–30%, trend_up 35–50%;
  an unknown/None regime uses the most-conservative 15% cap. Applied after
  position sizing / dynamic risk and before order placement, as an **absolute
  cap** (min-wins, never double-counting dynamic risk's graduated throttle). The
  gate lives once on `SharedEntryHandlerMixin.apply_pre_order_gates` and is
  invoked identically by the backtest and live runtime entry handlers **and** the
  legacy short path (no bypass); gross exposure is computed by the shared
  `src/engines/shared/exposure.py` from both engines' `BasePosition` objects, so
  the arithmetic can't drift (backtest-live parity). Behind the
  `enable_exposure_governor` feature flag, **default OFF**. Requires regime
  detection (`enable_regime_detection`) on for non-conservative caps.
- **Bear-market model-validation gate** (#801): ML model promotion is now gated
  on a fixed set of historical bear/crash/chop windows. A candidate model's
  `latest` symlink is flipped only after it keeps max-drawdown at or below a
  per-window threshold (`config/validation_windows.json`). New
  `src/ml/validation/` package: `BearValidationHarness` scores a model per
  window (Sharpe / max-drawdown / win-rate / trades) via the backtest engine
  (reusing `ExperimentRunner`, so `mock`/`fixture` providers give deterministic
  CI runs); `promote_version_if_valid` makes the promotion decision and writes
  an auditable `validation_audit.json` next to the model version. Because the
  prediction registry resolves models purely by the `latest` symlink, the gate
  scores the *candidate* via flip → validate → roll-back-on-failure (a
  canary-with-rollback: a failing model is reverted to the previously-live
  version). Wired into `atb live-control deploy-model` (now validation-gated,
  `--skip-validation` human override) and the training `--auto-deploy` path
  (training promotes as before; on validation failure `latest` rolls back to the
  pre-training model). New `atb live-control validate-model` scores a model
  (flip/validate/always-roll-back) without changing what is live.
  Un-runnable validation (e.g. missing data) is *inconclusive* → soft-pass with
  a loud warning unless `VALIDATION_REQUIRED` is set. Thresholds are config, not
  code. See `docs/prediction.md` → "Bear-market validation gate".
- **Live enforcement of the portfolio max-drawdown hard cap** (risk-officer
  2026-07-04 finding, corroborating the 2026-06-08 observability audit —
  `RiskManager.check_drawdown()` had zero callers, so nothing halted the live
  engine at `max_drawdown_pct`): new `MaxDrawdownGuard` + `MaxDrawdownEnforcer`
  (`src/engines/live/monitoring/drawdown_guard.py`) measure drawdown from the
  session peak balance on every trading-loop iteration and, at the cap (0.20),
  trip the existing close-only mode — entries, legacy shorts, and scale-ins
  stop (close-only now also gates the `execute_entry_locked` chokepoint and
  the scale-in branch); exits/stop-losses keep running, nothing is
  liquidated. Peak baseline = peak true equity since the last reconciled
  reset (session-scoped; phantom-era ledger history deliberately excluded;
  durable cross-session peak tracked in #847). Emits a CRITICAL `system_events` row
  (`MAX_DRAWDOWN_BREACH`), a structured `risk_event`, and the alert webhook;
  latched (no re-trigger spam) and restart-safe (peak recomputed from
  `account_history` on boot via `DatabaseManager.get_session_peak_balance`).
  Warning tiers per risk-limits.json escalation: WARNING at 50% of the cap,
  CRITICAL at 80%, rate-limited. Operators clear a trip by restarting with
  `FEATURE_MAX_DRAWDOWN_RESET_PEAK=true` (re-baselines the peak; remove the
  flag afterwards). See `docs/live_trading.md` → "Max-drawdown hard cap".
- `FEATURE_ENTRY_PAUSE` feature flag: when truthy the live engine blocks all
  exposure INCREASES — new positions (long, short, and the legacy duck-typed
  short path) AND scale-ins — while exits, partial exits, stop-loss
  management, reconciliation, and monitoring continue untouched. Lets a human
  flatten risk ahead of macro events (FOMC/CPI) with a single env var and no
  code redeploy. Skips log one rate-limited WARNING per
  `ENTRY_PAUSE_WARNING_INTERVAL_SECONDS` (300s) via the shared
  `EntryPauseGate` (`src/engines/live/execution/entry_pause.py`), consulted by
  `LiveEntryCoordinator` (entry evaluation, entry execution defense-in-depth,
  legacy short path) and `LiveExitHandler` (scale-in decision). Discoverable
  default (`"entry_pause": false`) lives in `feature_flags.json`; the
  `FEATURE_ENTRY_PAUSE` env var remains the override path.

### Fixed
- **Stop-loss re-placement could arm a naked margin position for an
  externally-closed position** (Codex review of #852, finding 2 — pre-existing):
  when a position is closed or liquidated externally while the bot is offline,
  its DB row stays OPEN and is re-loaded on the next startup. Stop-loss
  verification deliberately runs *before* the asset-holdings check (so an offline
  SL *fill* can book its realized P&L first), so for an externally-closed
  position the tracked stop looks missing/cancelled and was **re-placed with
  `AUTO_REPAY`** before the holdings check could remove the phantom — on margin,
  the naked-position (fund-loss) path. The periodic cycle had the same risk (it
  iterates a stale snapshot copy while step 1b removes phantoms from the live
  tracker). A new `_position_holding_is_gone(exchange, use_margin, position)`
  guard now gates all five stop re-placement sites (startup `_verify_stop_loss`
  not-found + cancelled/expired/rejected branches, startup `reconcile_position`
  step-3 placement, periodic step-2 re-placement, and periodic
  `_place_missing_stop_loss`): it positively confirms the asset is gone using the
  same 50%-of-tracked thresholds as `_verify_asset_holdings` /
  `_verify_margin_position_exists` (margin short → borrowed, long → netAsset;
  short-circuits on `exchange_close_pending`), and **fails safe** (returns
  `False`, keep protecting) on a transient API error. Ordering was **not**
  changed, so offline SL-fill P&L booking is preserved. A follow-up (Codex review
  of #881) extended the guard to two more paths — the crash-recovery stop in
  `_reconcile_filled_entry` (a pending entry that filled then closed externally
  while offline is now handled as an external close: no stop, no emergency-sell)
  and the startup partial-exit `_resize_stop_loss_after_partial_exit` — and made
  margin-**long** liveness robust: `get_balance` returns `None` for **both** an
  absent asset and a transient error, so a fully-closed margin long could not be
  distinguished from an API blip; a new `_margin_net_asset`
  (`get_margin_account_asset`: zeros for an absent asset, `None` only on error)
  now backs the guard and both margin-long phantom-removers. Spot is unaffected
  (AUTO_REPAY is a no-op on spot and an oversell is exchange-rejected). Adds 22
  unit tests.
- **Max-drawdown guard mis-seeded its peak from the configured balance**
  (prod 2026-07-04: guard armed at $100.00 vs true session equity $84.42 and
  immediately warned at a phantom 15.60% drawdown): the seed took
  `max(db_peak, tracker_peak, balance)` and the PerformanceTracker peak
  initializes from `INITIAL_BALANCE` (the optimistic book value from the June
  phantom-balance pathology). The `account_history` session max is now
  authoritative — the tracker peak is no longer a seed candidate; fallback is
  the current recovered balance. A failed DB read now defers seeding to the
  next loop cycle (bounded by `MAX_SEED_ATTEMPTS`) instead of latching a
  half-seeded baseline. Deeper fix included: the live engine now constructs
  `PerformanceTracker` from the RESUMED session balance rather than the
  configured amount, which also fixes the phantom ~15% in
  `account_history.drawdown` and dynamic-risk drawdown after restarts.
- **Kelly sizer never received trade outcomes — Kelly sizing was permanently
  in cold-start fallback** (#840): `KellyCriterionSizer.record_trade` had zero
  engine callers, so `has_sufficient_history` stayed `False` forever and any
  Kelly-sized strategy silently traded its `fallback_fraction` in both
  backtest and live. Realized outcomes now flow through shared seams: final
  closes via `PerformanceTracker.add_trade_listener` (the same `record_trade`
  choke point both engines already call on every close, including live
  crash-recovery closes) and each banked partial-exit slice via an identical
  `on_partial_exit` hook on both position trackers — all funneling into
  `Strategy.on_trade_closed` → duck-typed `position_sizer.record_trade`, so
  backtest/live parity is structural and a position that banks partials
  before stopping out counts its wins, not just a final-slice loss. Outcomes
  are UNSIZED R-multiples (directional price move), so past sizing decisions
  cannot bias Kelly's reward:risk statistics; breakeven and near-zero-size
  bookkeeping closes are skipped. `LeveragedPositionSizer` forwards
  `record_trade` to its base sizer, the legacy `KellySizer` gains a
  `record_trade` adapter onto the same seam, and `kelly_momentum`'s
  `fallback_fraction` default now uses `DEFAULT_KELLY_FALLBACK_FRACTION`
  (0.02) instead of a divergent local 0.03.
- **Backtest partial-exit accounting booked fraction-of-position as
  fraction-of-balance** — a units-collision family that fabricated returns in
  every backtest with partial exits (a kelly_momentum ETHUSDT 30d run booked
  +$14.19 of phantom credits on $0.07–0.29 of notional and reported +16.67%
  where reality was ~0%):
  - Both engines' exit handlers now convert the policy's fraction-of-original
    to balance-fraction units (`fraction_of_original × original_size`) before
    P&L computation AND the size decrement; the shared
    `PartialExitExecutor` docstring now pins this units contract.
  - Phantom position zeroing fixed: `current_size` decrements in consistent
    units, and the backtest tracker clamps the exit to the remaining size
    (mirroring live), so final closes no longer book `Trade.pnl = 0.0`
    (0%-win-rate artifacts).
  - Zombie scale-ins guarded in both engines: a position fully consumed by
    partial exits can no longer be revived by a scale-in. When partials fully
    consume a backtest position, the engine now closes it immediately
    ("Partial exits complete @ level N", parity with live).
  - Live scale-ins now convert policy units the same way (dev-flagged path,
    #734 — no production behavior change; `live_partial_operations` is off).
  - Live DB persistence now records the same balance-fraction delta the
    runtime tracker applies (`apply_partial_exit_update` /
    `apply_scale_in_update` previously subtracted/added the raw
    fraction-of-original from the balance-fraction `Position.current_size`,
    phantom-closing rows and corrupting crash-recovery `daily_risk_used`).
    `PartialTrade.size` is likewise recorded in balance-fraction units.
  - Backtest max drawdown now marks open positions to market: the equity
    series fed to the performance tracker includes open-position unrealized
    P&L, so adverse excursions appear in MaxDD (previously invisible —
    a −9.4% excursion read as 0.026% MaxDD).
  - Strategy-declared `partial_operations` overrides now hydrate in backtests
    (previously `DEFAULT_PARTIAL_EXIT_TARGETS` always won); hydration moved to
    a shared `build_partial_exit_policy` used by both engines.
  - Backtest partial-exit fees/slippage now use the engine's configured rates
    (previously always the defaults, ignoring `fee_rate`/`slippage_rate`).
  - Backtest scale-ins now respect the max-position cap with live's
    never-shrink semantics (#835 parity): growth clamps to headroom,
    over-cap positions are never shrunk.
  - NOTE: deterministic backtest fingerprints change — the old numbers were
    fabricated. All historical backtest results with partial exits are suspect.

### Changed
- HyperGrowth default sizing raised: `risk_fraction` / `base_fraction`
  0.20 → 0.25 (`stop_loss_pct` stays 0.10). Board-approved 2026-07-03 with the
  risk-officer condition of ≤2% realized risk per trade: live confidence
  scaling lands realized notional at ~0.46–0.80 of base (≈11–20% of balance),
  so the 10% stop bounds loss at ≈1.1–2.0% per trade.
- Live max-position cap made explicit and enforced end-to-end:
  `railway.json` `startCommand` (the value prod actually runs) and the
  Dockerfile CMD now pass `--max-position 0.20` (prod previously ran an
  implicit `0.5`). The engine now wires `max_position_size` into
  `LiveExitHandler`, scale-ins are clamped to the remaining max-position
  headroom (previously only the daily-risk budget bounded them, which resets
  daily and allowed an at-cap position to keep growing), and
  `LivePositionTracker.apply_scale_in` caps `current_size` growth at the cap
  (was hardcoded 1.0) without shrinking already-over-cap adopted positions.
  Consciously accepted gaps: (a) the backtest engine does not yet enforce the
  scale-in max-position clamp — the parity clamp + test land in the sibling
  backtest PR (`fix/backtest-partial-exit-units`); (b) HyperGrowth's
  strategy-level `max_fraction` override is 0.25 while live is pinned to 0.20
  via `railway.json` — default backtests of HyperGrowth should pass
  `--max-position-size 0.20` to match prod.
- `LiveTradingEngine.start()` bootstrap sequence extracted into a new
  `LiveStartupSequencer` (`engines/live/startup.py`, #486 follow-up): the public
  `start()` is now a thin delegator to `LiveStartupSequencer.run()`, and the
  seven bootstrap phase helpers (session recover/create + wiring, #668
  open-position carry-forward, #657 self-heal, account sync, runtime services,
  main-loop launch) move verbatim behind an engine-backref `Protocol`
  (mechanical `self.` → `state.`). The capital-critical startup ordering and the
  public `start()` signature are preserved exactly. Adds a direct
  `LiveStartupSequencer` unit test file. Pure refactor — backtest determinism
  fingerprint byte-identical. (#486)
- Legacy duck-typed short-entry path moved off `LiveTradingEngine` into
  `LiveEntryCoordinator.process_legacy_short_entry` (#486 follow-up): the
  ~100-line `_process_legacy_short_entry` body is now a verbatim coordinator
  method (mechanical `self.` → `state.`, with the entry execution routed through
  the coordinator's own `execute_entry`); the trading loop calls the coordinator
  directly (the engine wrapper was removed — no test-mock seam or other caller).
  Hardens a carried-over bare `except Exception` around `get_risk_overrides()` to
  log at WARNING (was a silent swallow on a live short-entry path). Adds seven
  direct `LiveEntryCoordinator` unit tests for the moved path. Pure refactor —
  backtest determinism fingerprint byte-identical. (#486)
- Live entry/exit coordinator `Protocol` types tightened (#486 Step D): the
  engine-state backref `Protocol`s (`LiveEntryEngineState`,
  `LiveExitEngineState`) now declare `data_provider: DataProvider`,
  `db_manager: DatabaseManager`, `_base_asset_locks: BaseAssetLockRegistry`, and
  `_component_strategy: ComponentStrategy | None` (were bare `Any`), matching the
  concrete-typing standard set by the later coordinators / `LiveSessionRecoverer`.
  `exchange_interface` / `strategy` stay `Any` (genuinely duck-typed). Their unit
  tests now build the mocked backref with `create_autospec(..., instance=True)`
  instead of `MagicMock(spec=...)`, so call-signature drift on the spec'd engine
  helpers is caught, not just attribute-name drift. Typing/test-only — no runtime
  change; backtest determinism fingerprint byte-identical. (#486)
- `LiveTradingEngine._trading_loop` readability: extracted the ~100-line legacy
  duck-typed short-entry path into `_process_legacy_short_entry()` and the
  periodic account snapshot + exchange-sync block into
  `_log_periodic_account_state()`. Both are behavior-preserving moves (the loop
  body shrinks ~390 → ~250 lines); the loop, its per-iteration control flow, and
  the capital-critical error-handling/backoff block stay inline on the engine.
  Pure refactor — backtest determinism fingerprint byte-identical. (#486)
- `LiveTradingEngine.start()` decomposed from a ~327-line bootstrap monolith into
  a thin ~18-line phase orchestrator that calls cohesive, single-purpose private
  helpers (`_begin_session_runtime`, `_bootstrap_trading_session`,
  `_carry_forward_open_positions`, `_self_heal_terminal_positions`,
  `_synchronize_account_on_start`, `_start_runtime_services`,
  `_run_main_loop_until_stopped`). Each block moved verbatim — the
  capital-critical startup ordering (recover → create session → #668
  carry-forward → #657 self-heal → account sync → runtime services → loop
  kickoff) and the public `start()` signature are preserved exactly. Pure
  refactor — backtest determinism fingerprint byte-identical. (#486)
- `LiveTradingEngine.__init__` decomposed from a ~534-line monolith into a thin
  ~110-line orchestrator that calls cohesive, single-purpose private
  initializer helpers (`_validate_inputs`, `_resolve_settings`,
  `_init_coordinators`, `_init_risk_manager`, `_init_risk_policies`,
  `_init_partial_operations`, `_init_correlation`, `_init_database`,
  `_init_dynamic_risk_manager`, `_init_exchange_interface`,
  `_resume_balance_from_snapshot`, `_init_strategy_manager`,
  `_seed_trading_state`, `_init_time_exit_policy`, `_install_signal_handlers`).
  Each block moved verbatim — construction ordering, the full 35-param public
  constructor signature, and every public attribute are preserved exactly.
  Aligned `LiveLoopTimingEngineState.data_freshness_threshold` to `int` (was
  `float`) so it matches the engine attribute (`MarketDataHandler` and the
  sibling interval fields are already `int`); this latent inconsistency was
  previously masked by mypy's same-`__init__` deferral. Pure refactor — backtest
  determinism fingerprint byte-identical. (#486)
- `LiveTradingEngine` dynamic-risk adjustment extracted into
  `LiveDynamicRiskCoordinator` (`engines/live/dynamic_risk_coordinator.py`):
  `_apply_dynamic_risk_adjustment` and `_log_dynamic_risk_adjustments` move
  verbatim (mechanical `self.` → `state.` against an engine backref
  `Protocol`); the engine keeps thin delegating wrappers (still called by the
  trading loop and by `LiveEntryCoordinator` via `state._apply_dynamic_risk_adjustment`).
  Adds direct `LiveDynamicRiskCoordinator` unit tests. Pure refactor — backtest
  determinism fingerprint byte-identical; engine `trading_engine.py` ~2,570 →
  ~2,490 lines. (#486)
- `LiveTradingEngine` trading-loop timing helpers extracted into
  `LiveLoopTimingCoordinator` (`engines/live/loop_timing.py`):
  `_sleep_with_interrupt`, `_calculate_adaptive_interval`, and `_is_data_fresh`
  move verbatim (mechanical `self.` → `state.` against an engine backref
  `Protocol`); the engine keeps thin delegating wrappers (still called by the
  trading loop, and `_is_data_fresh` by `LiveMarketDataCoordinator`). Leaf
  helpers — no order placement or balance mutation. Pure refactor — backtest
  determinism fingerprint byte-identical; engine `trading_engine.py` ~2,630 →
  ~2,570 lines. (#486)
- `LiveTradingEngine` per-candle market-data + context read path extracted into
  `LiveMarketDataCoordinator` (`engines/live/execution/market_data_coordinator.py`):
  `_is_context_ready`, `_get_latest_data`, `_add_sentiment_data`, and
  `_build_correlation_context` move verbatim (mechanical `self.` → `state.`
  against an engine backref `Protocol`); the engine keeps thin delegating
  wrappers (still called by the trading loop and, for the correlation context,
  by `StrategyRuntimeCoordinator`). Read-only path — no order placement or
  balance mutation. Pure refactor — backtest determinism fingerprint
  byte-identical; engine `trading_engine.py` ~2,880 → ~2,630 lines. (#486)
- `LiveTradingEngine` order-fill callbacks extracted into
  `LiveOrderFillCoordinator` (`engines/live/execution/order_fill_coordinator.py`).
  The `OrderTracker` callbacks — `_handle_order_fill`, `_handle_partial_fill`,
  `_handle_order_cancel` (+ its `_handle_stop_loss_cancelled` escalation), and
  `_handle_order_tracking_lost` — move verbatim (mechanical `self.` → `state.`
  against an engine backref `Protocol`); the engine keeps thin delegating
  wrappers and still registers those wrappers with `OrderTracker`. These run on
  the OrderTracker poll thread; the coordinator holds no state of its own, so
  the single-writer / thread-safe-handoff discipline (stop-loss fills enqueued
  on `_pending_fill_exits`, #631; atomic tracker mutations) is unchanged. Pure
  refactor — backtest determinism fingerprint byte-identical; engine
  `trading_engine.py` ~3,117 → ~2,880 lines. (#486)
- `LiveTradingEngine` exit pipeline extracted into `LiveExitCoordinator`
  (`engines/live/execution/exit_coordinator.py`), mirroring the entry
  extraction. `_check_exit_conditions`, `_execute_exit`, and
  `_execute_exit_locked` move verbatim (a mechanical `self.` → `state.`
  rewrite against an engine backref `Protocol`); the engine keeps thin
  delegating wrappers so all call sites and test mock points are unchanged.
  `check_exit_conditions` invokes the close through the engine's
  `_execute_exit` wrapper so existing engine-level test mocks still intercept;
  the base-asset close lock (#703) and the resting-stop cancel-before-close
  ordering (#710) are preserved. Pure refactor — backtest determinism
  fingerprint byte-identical before/after; engine `trading_engine.py`
  3,574 → ~3,130 lines. (#486)

### Fixed
- Hardened `LiveOrderFillCoordinator` CODE.md compliance after the #486
  order-fill extraction (issues carried over verbatim from the engine): the
  order-fill `logger.info` uses lazy `%s` formatting (was an f-string); the
  cancel-refund's original-quantity fallback uses an explicit
  `quantity is not None` check instead of `or 0.0` (Position-Fields rule — a
  legitimate `0.0` is valid state, not "unset"; behaviour-neutral here given
  the downstream `> 0` guard); and the entry-fee-refund-failure
  `logger.critical` now passes `exc_info=True` (balance-integrity failure
  where the traceback matters most). Adds direct `LiveOrderFillCoordinator`
  unit tests covering the deferred stop-loss-close queue handoff, the
  stop-loss-cancel escalation seam, the cancel-refund full/partial/`None`-qty
  branches, and the fail-closed tracking-lost contract. (#486 follow-up)
- Hardened `LiveExitCoordinator` CODE.md compliance after the #486 exit
  extraction (issues carried over verbatim from the engine): the exit-logging
  `logger.error` calls use lazy `%s` formatting (were f-strings); the
  position-age log reason uses `datetime.now(UTC)` instead of the deprecated
  `datetime.utcnow()` (behaviour-preserving — both sides compared tz-naive);
  the realized-P&L balance-failure `logger.error` now passes `exc_info=True`
  (financial-state failure where the traceback matters most); and
  `check_exit_conditions` / `execute_exit` / `execute_exit_locked` gained
  `Any` annotations on `runtime_decision`/`candle` and a `-> None` return.
  Adds direct `LiveExitCoordinator` unit tests covering the close-routing seam
  (`state._execute_exit`), the base-asset-lock delegation, and the
  `execute_exit_locked` early-return guards. (#486 follow-up)
- Live entry stop-loss gate no longer silently skips a misconfigured `0.0`
  stop. `LiveEntryCoordinator.execute_entry_locked` now keys the server-side
  stop-loss placement on `stop_loss is not None` (was a truthy check), so a
  `0.0` stop enters the placement path and fails there → emergency-close,
  rather than leaving the position open and unprotected. Also hardened the
  surrounding CODE.md issues carried over in the #486 entry extraction: the
  stop-loss-calc and risk-override failure paths now log at WARNING (were
  silent / `debug`), the emergency-close error log uses lazy `%s` formatting,
  and the entry-reason `stop_loss`/`take_profit` checks use `is not None`.
  Adds direct `LiveEntryCoordinator.execute_entry_locked` unit-test coverage
  for the guard, tracking-failure, balance-failure, risk-failure, ambiguous,
  and stop-loss-placement branches. (#486 follow-up)
- Backtests are now bit-reproducible — `Backtester.run` pins BLAS/OpenMP thread
  pools to 1 (`threadpoolctl`) for the duration of the run. Multi-threaded
  parallel float reduction is non-associative, so its run-to-run ordering could
  perturb a feature value enough to flip a near-threshold ML signal, changing
  the trade count (49 vs 50 vs 51 observed on the same inputs) and breaking the
  backtest↔live parity fingerprint that refactor work relies on. Investigation
  ruled out ONNX inference (byte-identical within and across processes,
  multi- and single-threaded), `PYTHONHASHSEED` (10 fixed seeds identical), and
  the prediction cache (varied with caching disabled); the cause was BLAS thread
  scheduling. ONNX keeps its own (deterministic) thread pool, so inference stays
  multi-threaded — measured backtest wall-time is neutral-to-faster, since
  pinning also avoids thread oversubscription across concurrent backtest
  processes. A new `tests/integration/parity/test_backtest_determinism.py`
  guards the guarantee. (#486 parity work)

### Changed
- Entry decision + execution pipeline extracted from `LiveTradingEngine` into
  `src/engines/live/execution/entry_coordinator.py` (`LiveEntryCoordinator`,
  #486): `check_entry_conditions` (signal/sizing/SL-TP derivation) and the
  base-asset-locked `execute_entry` → `execute_entry_locked` order path
  (duplicate/limit guards, balance + fee accounting, position tracking, risk
  re-registration, server-side stop-loss placement, and the emergency-close
  fallbacks). This is a real-money path, so the move is verbatim — the methods
  are unchanged except for `self.`→`state.` against an engine backref; the
  base-asset locking and ordering (#703) are preserved. The two engine methods
  callers mock (`_execute_exit`, `_record_event`) are invoked through the
  backref so test mocks still intercept. The engine keeps thin delegating
  wrappers, dropping ~620 lines (to ~3,575). The deterministic backtest↔live
  parity fingerprint is byte-identical before and after the extraction.
- WebSocket stream-health subsystem extracted from `LiveTradingEngine` into
  `src/engines/live/ws_health.py` (`WebSocketHealthMonitor`, #486): WS stream
  startup, the background health-monitor thread and its loop, kline/user-stream
  staleness detection, reconnect/probe decisions, degraded-user hard-reconnect +
  primary restore, and draining the order-fill exit queue on the trading-loop
  thread. The lock-free single-writer threading model is preserved byte-for-byte
  — the daemon-thread handle, the reconnect-failure counters, the
  `_ws_kline_active` flag, and the thread-safe `_pending_fill_exits` queue all
  stay on the engine and are accessed by the monitor via a narrow `Protocol`
  backref, so the single writer (the health thread) and the single reader (the
  trading loop) are unchanged. The engine keeps thin delegating wrappers so the
  loop call sites and all test mock points are unchanged; the three test-mocked
  sibling calls route back through the engine wrappers. The deterministic
  backtest↔live parity fingerprint is byte-identical before and after the
  extraction.
- Strategy hot-swap / model-update lifecycle extracted from `LiveTradingEngine`
  into `src/engines/live/strategy_hot_swap.py` (`StrategyHotSwapCoordinator`,
  #486): the public `hot_swap_strategy` / `update_model` entry points, the
  `StrategyManager` callbacks, the loop-applied `_apply_pending_strategy_update`,
  and the post-swap refresh of all strategy-derived engine state (trailing-stop
  / partial-operations / time-exit policies, component risk re-binding,
  correlation-handler strategy reference). The engine keeps thin delegating
  wrappers so the public API, the `__init__` callback registrations, the
  trading-loop call site, and test mock points are unchanged. The coordinator
  reads/writes engine state at call time via a narrow `Protocol`; all mutation
  runs on the single trading-loop thread (the entry points/callbacks only queue
  a `StrategyManager`-locked pending update), so the lock-free design is
  preserved. Pure refactor (live-engine only; no backtest/shared code touched);
  full unit suite incl. the hot-swap behavior tests stays green. Engine: 5,107
  → 4,790 lines.
- Strategy-runtime coordination extracted from `LiveTradingEngine` into
  `src/engines/live/strategy_runtime.py` (`StrategyRuntimeCoordinator`, #486):
  strategy normalization (`_configure_strategy`), the component risk-context
  provider (correlation hydration), runtime dataframe prep, `RuntimeContext`
  construction from live positions, per-candle runtime decision processing, and
  the construction-time risk-parameter merge/clone helpers. The engine keeps
  thin delegating wrappers so all call sites and test mock points are unchanged;
  the coordinator reads/writes the engine's strategy-runtime state at call time
  via a narrow `Protocol`, all on the single trading-loop thread. Pure refactor
  (no behavior change): full unit suite, parity suite, and the deterministic
  backtest fingerprint are byte-identical before/after. Engine: 5,383 → 5,105
  lines.
- `LiveTradingEngine` construction-time settings resolution (feature flags /
  env / app config) moved to `src/engines/live/config.py`
  (`LiveEngineSettings`, #486 step d): the #734 `live_partial_operations`
  gate, the `FEATURE_ENABLE_REGIME_DETECTION` env flag, and the
  `EXECUTION_FILL_POLICY` fill-policy read. `runner.py` resolves and injects
  settings explicitly; the engine self-resolves when not injected (using its
  module-level lookups so existing test patch points keep working).
  Runtime-dynamic flag reads (`ws_user_hard_reconnect`, hot-swap partial-ops
  re-check, heartbeat steps) intentionally stay runtime.
- Live trading engine refactor, steps 1–3 of #486 (pure refactor, no behavior
  change; verified by the full unit suite, the parity suite, and a
  deterministic backtest fingerprint that is byte-identical before/after):
  - Exchange-facing stop-loss lifecycle (placement with retry, cancellation,
    fill/held-inventory queries, re-protection, offline-fill detection for the
    legacy reconciliation fallback) moved from `LiveTradingEngine` into
    `src/engines/live/execution/stop_loss_manager.py` (`LiveStopLossManager`).
    The engine no longer calls `place_stop_loss_order`, `cancel_order`,
    `get_open_orders`, or `get_order` directly — it orchestrates via thin
    delegating wrappers.
  - Account monitoring glue (balance/equity snapshots, status lines,
    performance summaries, dataframe extraction helpers) moved to
    `src/engines/live/monitoring/` (`LiveAccountMonitor`).
  - Session/crash-recovery startup sequence (balance recovery, persisted
    position reload with stale-OPEN self-healing, risk-manager
    re-registration, startup exchange reconciliation incl. the legacy
    SL-based fallback) moved to `src/engines/live/recovery.py`
    (`LiveSessionRecoverer`). Close-accounting helpers shared by the exit and
    recovery paths moved to `src/engines/live/trade_close_accounting.py`
    (re-exported from `trading_engine`). Engine across all four extractions:
    6,558 → 5,368 lines.
  - The three byte-identical entry-handler methods (`_extract_entry_plan`,
    `_apply_dynamic_risk`, `get_dynamic_risk_adjustments`) now live once in
    `src/engines/shared/execution/entry_handler_mixin.py`
    (`SharedEntryHandlerMixin`), inherited by both the backtest and live
    entry handlers so this slice of backtest-live parity holds by
    construction. Divergent orchestration (`process_runtime_decision`,
    `execute_entry`, exit checks) is intentionally left engine-specific.

### Fixed
- Periodic reconciler now persists a balance-neutral audit `trades` row when it
  detects an externally-closed position (margin and spot branches of
  `PeriodicReconciler._reconcile_cycle`), extending the startup external-close
  audit row to the periodic cycle. Each branch delegates to the startup
  reconciler's `_log_external_close_trade` (so it books identically: GROSS
  `Trade.pnl` at a proxy mark-to-market price, dedup key
  `reconcile_ext_<db_position_id>`, `balance_realized=False`), popping the
  position only if still tracked and gating the row on the DB `close_position`
  call actually returning `True` (it swallows DB errors to `False`) so a failed
  close is re-reconciled rather than logged for a still-open position.
  The periodic spot path also **self-heals** the session balance the same cycle
  via a new `_reconcile_spot_balance`, which values a **fresh** position snapshot
  — fixing a stale-snapshot over-correction where the periodic balance check
  counted a just-closed position's notional and over-corrected the balance by it
  for ~one cycle (~2 min) before self-healing. Margin balance stays owned by
  `AccountSynchronizer._sync_margin_equity`.
- Backtest risk tracking now covers next-bar (pending) entries (#757):
  the post-fill `RiskManager.update_position` call passed the `PositionSide`
  enum, whose string validation (`side in VALID_SIDES`) raised `ValueError`
  on every call — swallowed with a warning — so `daily_risk_used` and
  position tracking for correlation control silently omitted every pending
  entry. Backtests could take position sequences a correctly-accounted run
  would have rejected. The side now converts via `to_side_string`, like the
  immediate-entry path.
- Live daily P&L survives restarts now (#766): day-start balance recovery
  queried `DatabaseManager.get_first_snapshot_of_day`, which was never
  implemented (`AttributeError` swallowed as a "graceful fallback"), and the
  recovery helper itself was never invoked — so every intraday restart reset
  the daily P&L baseline to the restart-time balance. The method now exists
  (earliest `account_history` row of the UTC day for the session) and is
  wired into the first snapshot after engine start. Trading-day semantics are
  explicitly UTC throughout the event logger (was local `date.today()`,
  skewing day boundaries on non-UTC hosts).
- Backtest trades persist the correct `pnl_percent` for longs (#758):
  the backtest event logger passed the engines' `PositionSide` enum into
  `log_trade`, which compares against the **database** `PositionSide` —
  cross-enum equality is always False, so every long backtest trade was
  stored with the short formula (sign-flipped). `log_trade` now normalizes
  any Enum side/source by value before classification (hardens all callers)
  and the backtest call site converts via `to_side_string`.
- Correlation control no longer silently drops peer symbols (#759): the
  no-window fallback omitted the required `start` argument, so every call
  raised `TypeError` (swallowed) and the peer vanished from correlated-
  exposure accounting in BOTH engines. A failed window computation now falls
  back to the default correlation window; when no time window is derivable
  at all (non-datetime index), peers are skipped with an explicit WARNING
  instead of fabricating a wall-clock window (backtest lookahead).
- Backtest strategies can finally see their open position (#756):
  `_build_runtime_context` passed the `PositionSide` enum into
  `ComponentPosition`, whose validation expects "long"/"short" strings, so
  construction raised `ValueError` on every candle (swallowed) and component
  strategies always received `current_positions=None` — while live populated
  it correctly. Position-aware logic (pyramiding guards, exposure checks) was
  silently inert in backtests. The side now converts via `to_side_string`,
  exactly like live.
- CoinbaseProvider no longer submits every order as MARKET (#762):
  `_convert_to_cb_type` was keyed by lowercase strings while `OrderType`
  enum values are uppercase, so the lookup always fell back to "market" —
  limit orders lost price protection and stop orders fired immediately.
  Mapping is now enum-keyed, unknown types raise instead of defaulting to
  the most dangerous order type, and GTD time_in_force is rejected before
  the API call (it requires an end_time this client cannot send).
- `LivePositionTracker.recover_positions` actually recovers positions now
  (#764): it called `DatabaseManager.get_open_positions`, a method that does
  not exist, so the swallowed `AttributeError` made it always return `[]` —
  a silent fail-open trap for any future recovery caller. It now maps the
  dict rows from the real `get_active_positions` API with the same
  normalization and hydration as the engine's `_recover_active_positions`
  (uppercase DB side, tracker key fallback to row id, partial-op state,
  reconciliation ids), skips invalid-entry-price rows with a CRITICAL log,
  and isolates per-row failures.
- `TradeProtocol` members are now read-only properties (#767), so concrete
  trade classes with narrower types (non-Optional datetimes, `PositionSide`
  enum side) conform structurally — the three `cast("TradeProtocol", ...)`
  workarounds at the engines' `record_trade` call sites are gone. The `side`
  member is honestly typed `str | Enum | None` (record_trade stringifies it).
- Backtest trailing-stop updates no longer crash the run when trailing
  activates without a stop improvement (#761): `TrailingStopManager.update`
  legitimately returns `updated=True, new_stop_price=None` (e.g. ATR
  unavailable on the activation candle), and the backtest tracker compared
  that `None` against the current stop (`TypeError`, unwrapped). The tracker
  now mirrors the live tracker: flag updates apply, price comparison skipped.
- `build_time_exit_policy` (engines/shared) can now actually build a policy
  (#760): it passed `exit_time`/`exit_days` kwargs that `TimeExitPolicy` does
  not accept, so it always raised `TypeError` internally and returned `None`.
  It now maps the same `time_exits` config shape as both engines' builders
  (max holding, end-of-day/weekend flat, timezone, restrictions) and honors
  both `params.time_exits` and the legacy `params.max_holding_hours` fallback.
- `StrategyManager.update_model` with no strategy loaded now fails with the
  intended descriptive `ValueError` (#765) instead of an `AttributeError`
  raised while formatting the error message itself (`self.current_strategy.name`
  on `None`), which surfaced as a misleading generic failed update.
- `atb data populate-dummy` works again (#763): `log_trade` was called with the
  nonexistent `order_id` kwarg (the parameter is `exit_order_id`), so the first
  generated trade raised `TypeError` and the command always failed. Same bug
  class as #732; an autospec'd regression test now enforces the real signature.
- A REJECTED stop-loss is now re-placed and an unexpected stop-loss
  termination escalates (#741). The reconciler's re-placement branches
  (periodic loop and startup `_verify_stop_loss`) matched only
  CANCELLED/EXPIRED/missing, so a triggered STOP_LOSS_LIMIT whose limit
  leg was rejected by margin checks (-2010 class) fell through every
  cycle — position permanently unprotected, no escalation. Both branches
  now treat REJECTED as terminal. The engine's `_handle_order_cancel`
  also no longer ignores stop-loss order ids: when a tracked position's
  stop terminates unexpectedly it clears the stale id (so the
  reconciler's missing-stop path re-protects next cycle), logs CRITICAL,
  and emits a `system_events` row (`STOP_LOSS_CANCELLED`) with webhook
  alert. Deliberate close-path cancels are unaffected (they stop
  tracking the order before the callback can fire).
- Repo-wide static-analysis debt cleared — `atb dev quality` (black, ruff,
  mypy, bandit) now passes from a red baseline of 26 unformatted files, 171
  ruff errors, ~700 mypy errors across ~90 files, and 25 bandit findings.
  All fixes are type/lint-level with no runtime behavior change: annotations
  (`X | None` lazy-init attributes, honest container/dict types, SQLAlchemy
  `Mapped[...]` column annotations), justified `cast()`s at untyped
  numpy/onnx/keras/exchange boundaries, removal of ~45 stale `type: ignore`
  comments, and isinstance tuples → PEP 604 unions. Bandit
  try/except-pass/continue sites now log (breadth unchanged; debug in
  per-candle hot loops, WARNING elsewhere); false positives carry justified
  `# nosec` markers; the one `assert` in `src/` is an explicit raise.
  Repaired silently-dead tooling config: `mypy.ini` per-module ignore
  sections stopped matching after the `src/` layout migration (modules
  gained the `src.` prefix) and its `exclude` regex had a trailing `|`
  matching every path; `types-PyYAML` is now pinned so `yaml` imports
  type-check. TensorFlow guarded imports use a `TYPE_CHECKING`-stable
  pattern so mypy results match whether TF is installed or not.
  Static analysis exposed several pre-existing behavioral defects which are
  deliberately NOT fixed here — each is marked with a `KNOWN BUG` comment at
  the site: backtest strategies never see open positions in
  `RuntimeContext` (enum-vs-string side validation always raises, swallowed
  per candle); risk tracking silently skipped for next-bar entries;
  persisted `pnl_percent` uses the short formula for long backtest trades;
  correlation control silently drops peer symbols on a missing-argument
  `TypeError`; `build_time_exit_policy` can never produce a policy;
  Coinbase enum order types map to lowercase keys that never match (limit
  orders would submit as market); `atb data populate-dummy` crashes on a
  nonexistent `log_trade(order_id=...)` kwarg.
- Periodic reconciler now books realized P&L when it detects a filled
  stop-loss. Both detection paths previously corrupted tracked capital:
  the stop-verification branch closed the DB row with NO balance update
  (and no trade record), and the margin holdings check misclassified a
  just-filled short stop-loss (AUTO_REPAY zeroes the borrow) as
  "externally closed" (the spot holdings check had the identical flaw:
  a filled stop also empties the held balance), closing the row with no
  exit price at all. Every
  SL loss the reconciler processed before the engine's deferred-exit
  drain (~equal ~2-minute cadences, so a large fraction) silently never
  hit the balance → overstated capital → oversized subsequent positions.
  Both the margin and spot holdings checks now consult the tracked stop
  order before classifying an external close.
  Both paths now delegate to the startup reconciler's filled-SL handler
  (#731): DB close first (a failed close leaves the position tracked for
  retry), P&L with USD-normalized commission and margin interest, plus a
  deduplicated `trades` row. The periodic wrapper skips when the engine's
  deferred-exit drain already processed the fill (no double-booking) and
  defers classification (fail-closed) when the stop's state cannot be
  confirmed.
- OrderTracker no longer converts an API outage into a position deletion.
  After `MAX_API_ERROR_RETRIES` (10) consecutive failed/`None` polls
  (~50 s at the live 5 s interval) the tracker fired `on_cancel`, and
  `_handle_order_cancel` popped the (possibly live) position from the
  tracker and refunded its entry fee — manufacturing untracked exchange
  exposure, a corrupted balance, and room for a double entry on the next
  signal, exactly during exchange API degradations (LESSONS §1.8 fail-open
  class). Polling give-up now routes to a new `on_tracking_lost` callback;
  the engine's `_handle_order_tracking_lost` keeps the position tracked,
  leaves the balance untouched, and escalates with a critical
  `system_events` row (`ORDER_TRACKING_LOST`) + webhook alert so the
  periodic reconciler resolves the order's true state from the exchange.
  `on_cancel` now fires only for exchange-confirmed terminal states.
- Closed live `trades` rows now persist `commission` and `quantity` (previously
  always `0` / `NULL`). The live close path (`LiveTradingEngine._close_position`
  and the offline stop-loss reconciliation path) now passes `commission` and
  `quantity` to `DatabaseManager.log_trade`, which already supported both.
  `trades.commission` is the round-trip fee in **USD** (`entry_fee + exit_fee`) —
  the same values booked to `account_balances` (entry as the `entry_fee_<symbol>`
  ledger event, exit folded into `realized_pnl_<symbol>`), **not** the raw
  `orders.actual_commission` (which is denominated in the received asset and
  populated asynchronously, so unit-ambiguous and unreliable at close time).
  `trades.quantity` is the actual filled base quantity, scaled by
  `current_size/original_size` for partially-exited positions (NULL for scale-in
  positions, whose held quantity is not derivable, and for corrupt sizing).
  `DatabaseManager._trade_net_pnl` now also subtracts `commission`, so true net P&L
  (`pnl - commission - margin_interest_cost`) flows through performance metrics and
  `recover_last_balance` reconstruction — correcting a latent overstatement now that
  commission is populated (historical rows carry `commission = 0` and are unaffected).
  For positions recovered after a restart, the entry-fee leg is reconstructed from the
  fee model (the `positions` table does not persist entry fee) rather than dropped, and
  scaled to the closed portion so a partial final close's commission matches its
  portion-level pnl/quantity. The `PositionReconciler` offline stop-loss path
  (`_realize_pnl_on_close`) now also inserts a `trades` row — previously it
  balance-corrected and DB-closed the position but recorded **no trade at all** (deduped
  via the exit order id + `uq_trade_order_session`). `LiveExecutionEngine` now converts an
  exchange fill commission to USD via its `commission_asset` (a base-asset commission on a
  buy, e.g. ETH, is priced into USD; an unconvertible asset like BNB falls back to the
  modelled fee) — fixing a latent bug where a base-asset commission could be booked as if
  it were USD. Relatedly, `_recover_active_positions` now hydrates
  `original_size`/`current_size` and partial-operation counters from the DB, so a position
  partially exited before a restart closes at its remaining size. The commission→USD
  conversion is shared via `src/engines/shared/commission.py` and applied on the
  reconciler offline-SL path too (a short's stop-loss is a base-asset buy), so it is
  never booked wrong-unit. The reconciler logs its trade row only after the DB position
  is actually closed and with a stable, non-NULL dedup key (real exit order id, else a
  synthetic id from the position) so a re-run cannot insert a duplicate
  (`uq_trade_order_session`; NULL≠NULL in Postgres) — guarding the #657/#668 phantom-trade
  class; a failure to persist the row after the balance was corrected now escalates to
  CRITICAL rather than a silent warning. See the "Trade fee accounting" note in
  `docs/live_trading.md`.
- Reconciler accounting hardening (review follow-ups): the offline stop-loss close now
  realizes P&L **only after** the DB position is actually closed (a failed close no longer
  double-subtracts P&L on the next reconcile), and a failed balance write skips the audit +
  trade row (no `trades`/`account_balances` divergence). Fees route through the shared
  `CostCalculator` (no duplicated fee modelling); the SL exit-fee fallback and the recovered
  entry/exit reconciler bookings now normalize commission to USD via `commission_asset` like
  the rest of the change. A scaled-in position closed by the reconciler stores NULL quantity
  and an un-inflated entry fee, matching the engine close path. `_extract_base_asset` now
  delegates to the shared `split_base_quote`. The mock DB enforces `uq_trade_order_session`
  so the dedup path is unit-tested.
- Live engine hard-disables partial exits / scale-ins behind the default-OFF
  `live_partial_operations` feature flag (#734). The live engine executed
  partial operations as bookkeeping only — `_execute_partial_exit` /
  `_execute_scale_in` mutate the tracker/DB but **never place an exchange
  order** — and with mismatched units (policy fractions of the original
  position applied to fraction-of-balance state), so on a real account a
  winner reaching the default +2%/+3% triggers desynced tracked size from
  actual holdings (stranded inventory, un-repaid margin borrows, -2010 close
  failures), booked phantom realized PnL, and freed daily-risk budget that
  was still deployed. All three activation paths are gated (constructor,
  strategy hot-swap overrides, runtime policy hydration via the existing
  opt-in state). Re-enable only for development of the #734 fix.
- Reconciler no longer places a DUPLICATE stop-loss when an order lookup
  fails transiently (#713). `BinanceProvider.get_order` swallows every
  exception into `None`, and both stop-loss verifiers (startup
  `PositionReconciler._verify_stop_loss` and the periodic reconciler's
  stop-verification loop) treated `None` as "stop missing" — clearing the
  tracked `stop_loss_order_id` and re-placing a new stop while the original
  could still be resting on the exchange (reserving base/margin, able to
  cause -2010 on a later close, and able to flip the position if both
  stops fill). Added a fail-closed `ExchangeInterface.get_order_checked`
  (Binance override returns `None` only on a confirmed -2013
  "order does not exist" and raises `OrderLookupError` on any unconfirmed
  lookup), and both verifiers now skip the cycle on an unconfirmed lookup
  instead of re-placing. Confirmed-missing stops are still re-placed.
- Live trade recovery on the `emergency_sync` path no longer silently fails.
  `AccountSynchronizer.recover_missing_trades` called
  `DatabaseManager.log_trade(order_id=...)`, but `log_trade` has no `order_id`
  parameter (the field is `exit_order_id`) and no `**kwargs`, so every recovered
  trade raised `TypeError` — swallowed by the per-trade `except` — and was never
  persisted to the ledger. Maps `trade.order_id` onto `exit_order_id` (which feeds
  the `Trade.order_id` column). Adds a regression test that drives
  `recover_missing_trades` with an autospec'd `DatabaseManager`, so the real
  `log_trade` signature is enforced. Also clears pre-existing mypy loop-variable
  and ruff `UP038` debt on `account_sync.py` (behaviour-neutral).
- Reconciler `trades` rows now cover two more close paths that previously corrected state
  but recorded no trade row (extending #731's offline stop-loss trade-row logging). (1) The **crash-recovery
  FULL_EXIT** path (`_reconcile_filled_exit`) opts into `log_trade=True` with a stable dedup
  key — the real exchange exit order id, else a synthetic `reconcile_exit_<position_id>` — and
  realizes P&L **only after** the DB position is actually closed (a failed close no longer
  double-corrects the balance on the next reconcile pass). (2) **External/manual closes**
  (operator sells on the exchange UI, or a liquidation) detected by `_verify_asset_holdings`
  (spot) and `_remove_phantom_position` (margin) now persist a **balance-neutral** audit trade
  row — commission (reconstructed entry leg) + quantity + GROSS pnl, priced mark-to-market via
  the data provider (degrading to entry price → pnl 0 when no price source) — deduped by a
  synthetic `reconcile_ext_<position_id>` key, gated on the DB close succeeding. These paths
  deliberately do **not** realize P&L: a spot external close's capital is already reconciled by
  startup Step C (`_reconcile_balance`), and a margin external close's by
  `AccountSynchronizer._sync_margin_equity`, so writing the balance here too would double-book
  the `account_balances` ledger. `PositionReconciler` now accepts an optional `data_provider`
  for the mark-to-market estimate. Hardening on all reconciler close paths: the DB-close gate now
  reads `close_position`'s actual return (it returns `False` **without raising** on a missing row
  or a rolled-back commit, so "did not raise" was not "closed"); the external-close paths use
  `pop_position` so a position already reconciled earlier in the same run is not logged twice (the
  `reconcile_exit_`/`reconcile_ext_` keys do not collide); and a failed trade-row write on a
  balance-neutral path now escalates as a missing-audit-row alert rather than a false
  "account_balances/trades DIVERGED" page. (Known follow-up: `PeriodicReconciler._reconcile_cycle`
  has parallel inline external-close detection that still records no trade row — its capital is
  reconciled each cycle by the periodic balance step (notional check, like startup Step C), so this
  is an audit-row gap, not a balance gap.)
- Margin-equity balance corrections are now audited and alertable.
  `margin_equity_sync_correction` book-downs (written by
  `AccountSynchronizer._sync_margin_equity`) previously updated the balance ledger
  without recording a `reconciliation_audit_events` row or a warning-level
  `system_events` row, so the single largest capital event a margin session can
  produce was invisible to monitoring/auditing — a −$15.75 (−15.8%) production
  book-down on 2026-06-03 (and a second −$1.37 on 2026-06-05) left zero audit
  trail. The path now emits both records via a new best-effort
  `_record_equity_correction_audit` helper: an immutable audit row
  (`entity_type='balance'`, `field='total_balance'`, before/after values, severity
  `HIGH`, escalating to `CRITICAL` when divergence ≥ 5%) and a `BALANCE_ADJUSTMENT`
  system event at `warning` severity (`critical` when ≥ 5%) so alerting can see large
  book-downs. Both writes are independently guarded so a logging failure can neither
  raise into the sync loop nor unwind the already-persisted correction; emission is
  skipped entirely if `update_balance` itself reports failure (no audit for a
  correction that never persisted). The audit binds to the same session the balance
  write used — resolved via `update_balance`'s own `_current_session_id` fallback — so
  the first post-restart correction (when `AccountSynchronizer.session_id` has not yet
  been assigned) is captured too, not just steady-state periodic syncs.
- Live position/trade recovery no longer crashes on `Decimal`-vs-`float`
  arithmetic. `DatabaseManager.get_active_positions` and `get_recent_trades`
  now coerce SQLAlchemy `Numeric(18,8)` columns (which read back from
  PostgreSQL as `Decimal`) to `float` at the source — `float()` for
  non-nullable columns, `_to_optional_float()` for nullable ones — mirroring
  the existing `orders_data` block and `LivePositionTracker.recover_positions`.
  Previously these raw `Decimal`s flowed through `_recover_active_positions`
  into recovered `Position` objects and raised `unsupported operand type(s)
  for *: 'decimal.Decimal' and 'float'` in reconciliation's default
  stop-loss branches (`entry_price * (1.0 ± DEFAULT_STOP_LOSS_PCT)`), which
  run *before* the `place_stop_loss_order` boundary that PR #653 had patched.
  Also keeps dashboard consumers JSON-serializable (`json.dumps` raises on
  `Decimal`).
- Live restart balance recovery no longer crashes or silently resets on
  `Decimal`-vs-`float` arithmetic (same `Numeric` class as above, balance path).
  `DatabaseManager.recover_last_balance`'s trades fallback computed
  `initial_balance + net PnL`, raising `TypeError` on `Decimal + float` — swallowed
  by `_recover_existing_session`, which then returned `None` and reset the engine to
  its default balance on restart. With no trades it returned a raw `Decimal` that
  later broke `_print_final_stats`' float arithmetic on shutdown
  (`unsupported operand -: Decimal and float`). The fallback now coerces
  `float(initial_balance)`; `_recover_existing_session` coerces the recovered value
  to `float` and fails fast (raises) on a non-finite balance *before* its `> 0`
  positivity filter, so corrupt persisted state can never reach position sizing or
  silently fall back to the default balance.
- Backtest-live engine parity: closed nine silent divergences. Backtest now
  propagates `TimeExitPolicy`-specific exit reasons (`"Max holding period"`,
  `"Weekend flat"`, etc.) instead of hardcoding `"Time limit"`; gained an
  optional `annual_margin_interest_rate` parameter on `Backtester` mirroring
  live's `MarginInterestTracker` (default `0.0` preserves spot-mode
  behaviour); now sums `entry_fee + exit_fee + margin_interest_cost` into
  `PerformanceTracker.record_trade` matching live's total-fee semantics;
  persists `margin_interest_cost` to the `trades` DB column via
  `EventLogger.log_completed_trade`. Live now wires `CorrelationHandler`
  into `LiveEntryHandler` and threads the full `symbol/timeframe/df/index`
  context through `_check_entry_conditions` so correlation-driven sizing
  reduction actually fires; backfills historical sentiment over the full
  buffer before overlaying the live snapshot so ML strategies get
  equivalent inputs; sweeps the position tracker after reconciliation to
  register reconciler-created positions with the risk manager (using
  `current_size` to preserve partial-exit accounting); passes the live
  positions list to the direct `ComponentStrategy.process_candle` path.
  Documented tick-size rounding, margin interest, and single-vs-multi-position
  as known parity caveats on the `Backtester` docstring.
- Live trading engine no longer shuts itself down during transient database
  outages. Transient DB-connectivity errors (DNS resolution failures, dropped
  connections, brief Postgres unavailability) are now classified and *ridden
  out* with a bounded backoff instead of counting toward
  `max_consecutive_errors`. This was the root cause of the 2026-05-19 incident:
  a multi-hour Railway internal-DNS outage made `postgres.railway.internal`
  unresolvable, every loop iteration raised `OperationalError`, the
  consecutive-error limit tripped, and **both the staging and production bots
  went offline — silently — for ~12 days**. `pool_pre_ping` reconnects
  automatically once the database returns. Permanent faults (bad credentials,
  missing role/database, permission denied) are excluded and still fail fast,
  and an outage lasting more than 30 minutes drops the engine into close-only
  mode (new entries suspended; exits and server-side stop-losses continue).
- Prediction-cache performance test (`test_cache_performance_characteristics`)
  is no longer timing-flaky on loaded CI runners. It previously took the *mean*
  of `time.time()` over 100 cold, fully-mocked operations and asserted cache-hit
  was within 5× a tiny (~0.13ms/op) noise-dominated cache-miss baseline, so a
  single GC/scheduler pause inflating the mean would trip it (it failed twice on
  PR #637). It now warms up, samples many ops with `perf_counter`, and asserts
  the *median* (immune to those outliers) against a generous absolute budget. It
  is also marked `@pytest.mark.performance` so it runs in the nightly performance
  workflow rather than the blocking PR integration gate.

### Added
- Heartbeat staleness monitor (`scripts/check_heartbeat.py` +
  `.github/workflows/heartbeat-monitor.yml`): a scheduled, read-only CI job that
  fails (notifying maintainers) when an active trading session's
  `account_history` snapshot goes stale beyond a threshold (default 2h) — the
  canonical liveness signal. Requires the `RAILWAY_STAGING_DATABASE_URL` /
  `RAILWAY_PRODUCTION_DATABASE_URL` repository secrets.

### Changed
- `railway.json`: raised the Trading Bot `restartPolicyMaxRetries` from 3 to 10
  so Railway keeps retrying through longer transient infrastructure failures.
- `/deploy-staging` and `/deploy-prod` slash-command skills rewritten to match
  the actual `develop` → `staging` → `main` promotion workflow (Railway
  development → staging → production). `/deploy-prod` no longer does
  `git reset --hard origin/staging` + force-push to `main` (which would rewrite
  production history and drop the "Promote to production" commits that live only
  on `main`); it now opens an additive **"Promote to production" PR**
  (`staging`→`main`), reconciles changelog conflicts by merging `main` into
  `staging`, waits for green CI, and merges with a merge commit. `/deploy-staging`
  syncs the long-running `staging` branch to `develop`.
- Protected the `staging` branch (`allow_deletions: false`, force-push still
  allowed) so the repo-wide `delete_branch_on_merge: true` no longer
  auto-deletes it when a `staging`→`main` promotion PR is merged. `staging` is a
  long-running branch bound to the Railway staging environment and must persist.

### Security
- Hardened a batch of security findings from a repo-wide scan (bandit + manual
  audit):
  - Monitoring dashboard: added a token auth guard (`MONITORING_DASHBOARD_TOKEN`)
    on state-changing/data-leaking endpoints (`POST /api/balance`,
    `POST /api/config`, `POST /api/debug/fix-positions`, `GET /api/debug/positions`).
    Fails closed in production when no token is set; warns-and-allows only in
    explicit dev/test envs. Restricted Socket.IO CORS from `"*"` to same-origin
    (override via `MONITORING_CORS_ALLOWED_ORIGINS`).
  - SageMaker artifact extraction now validates tar members (rejects path
    traversal / zip-slip and escaping symlinks). Model-registry sync validates
    `version_id`/`model_type` from `metadata.json` and asserts the resolved path
    stays inside the registry before any `rmtree`/`copytree`/`symlink`.
    S3 artifact download skips object keys that escape the target directory.
  - `get_secret_key()` now fails closed: an unset `ENV`/`FLASK_ENV` is treated as
    production instead of silently returning the public dev key.
  - Admin UI login compares the username in constant time (`hmac.compare_digest`).
  - `atb data cache-manager --detailed` uses a restricted unpickler (allowlisted
    pandas/numpy types) instead of raw `pickle.load` on legacy `.pkl` files.
  - JUnit XML parsing uses `defusedxml` (XXE / billion-laughs hardening).
  - Tightened temp-shim permissions to `0o700`; quoted/validated table
    identifiers in the DB integrity check; marked the dashboard's `0.0.0.0`
    bind intentional.

### Added
- **Monitoring dashboard mobile layout**: V2 dashboard reflows below 768px to a
  bottom tab bar + stacked content + inline inspector. Reuses the same React
  store and data flow; layout swap driven by `useIsMobile()` hook backed by
  `window.matchMedia` with a resize listener so it adapts live. iOS safe-area
  insets respected via `viewport-fit=cover` + `env(safe-area-inset-*)`.
- **Monitoring dashboard V2 redesign**: chart-led layout with left-rail nav
  (Dash / Pos / Strat / Trades / Risk / Logs), KPI strip, hero equity chart
  with overlay toggles (benchmark / trades / drawdown), positions strip, and
  a swappable right inspector. Light + dark themes (toggle persisted to
  `localStorage`). Tech stack swap: Bootstrap + Chart.js → React 18 (UMD) +
  Babel-standalone + socket.io-client. CDN scripts pinned with SRI hashes.
- New `GET /api/dashboard/state` endpoint bundles metrics + positions +
  trades + bot meta in a single request to keep first paint snappy. Accepts
  `?trades_limit=` (clamped to 1..500). Falls back to per-resource fetches
  in the JS adapter if the bundled endpoint is unavailable.
- New `MonitoringDashboard._get_bot_meta()` reads strategy / symbols /
  timeframe / mode / `max_open_positions` from the most recent **running**
  `trading_sessions` row (falls back to the most recent overall row),
  matching the "Exchange Mode & Account Type Safety" guidance so a stale
  paper-mode session can't mask an active live one.
- `.claude/launch.json` — preview-server configurations for all three
  dashboards plus live-health.
- Experimentation framework (`src/experiments/`) with declarative YAML suites,
  `atb experiment run|list|show|promote` CLI, ranked reporter with statistical
  verdicts, file-based ledger under `experiments/.history/`, and promotion
  writer for `StrategyVersionRecord`/`ChangeRecord` plus patch YAML emission.
- ML signal generators now expose `long_entry_threshold`, `short_entry_threshold`,
  `confidence_multiplier`, and regime-specific thresholds as overridable instance
  attributes (class constants remain as defaults).
- `ConfidenceWeightedSizer` gained `min_confidence_floor` parameter.
- `create_ml_basic_strategy`, `create_ml_adaptive_strategy`, and
  `create_ml_sentiment_strategy` accept the new tuning knobs.

### Removed
- Deleted the unused first-attempt optimizer layer: `src/optimizer/analyzer.py`,
  `validator.py`, `strategy_drift.py`, the `atb optimizer` CLI, the
  `OptimizationCycle` DB model/table, `DatabaseManager.record_optimization_cycle`,
  `fetch_optimization_cycles`, and the `/api/optimizer/cycles` dashboard route.
  Alembic migration `0011_drop_optimization_cycles` drops the table.
- Renamed `src/optimizer/` → `src/experiments/` now that the package reflects
  its actual purpose. `atb walk-forward` continues to work via
  `src/experiments/walk_forward.py`.

### Fixed
- Binance margin-WS keepalive noise + user-stream watchdog gap (#608).
  `python-binance==1.0.36` multiplexes margin user-data subscriptions over a
  shared `ws_api` connection that Binance closes every ~2 min with WS code
  1011 'keepalive ping timeout'. The library's reconnect machinery recovers
  but each cycle surfaces an unretrieved-task exception on the asyncio
  default handler (~720/day on prod). Added
  `BinanceWSKeepaliveFilter` (rate-limits to one full traceback per 60s
  window with a periodic suppression summary) and extended
  `BinanceProvider.ws_healthy` to fail when the user/margin stream is
  configured but stale or non-PRIMARY (was previously kline-only, masking a
  permanently-dark user stream). New `user_ws_healthy` property exposes the
  user-stream status directly.
- `BinanceWSKeepaliveFilter` now also matches the ws_api subscribe-timeout
  signature (GH #608 follow-up). #609's filter only matched the 1011
  'keepalive ping timeout' close code, which never fires on prod — the
  actual ~2-min churn is the margin `userDataStream.subscribe` request
  timing out after 10s (`BinanceWebsocketUnableToConnect: Request timed
  out`), which carries no 'keepalive ping timeout' text and so was never
  suppressed. Replaced the single fingerprint with `KEEPALIVE_MARKER_GROUPS`
  (match all markers in any group); a `binance/ws/` anchor prevents
  swallowing connection errors raised by our own code.
- Add ban-aware retry to Binance client startup — parses `-1003` ban expiry and sleeps until lifted instead of crashing (#590)
- `hyper_growth`: fix silent-SELL bug caused by feature-shape mismatch
  (#603). The factory wired `MLBasicSignalGenerator(model_type="sentiment")`
  but fed the sentiment model the 5-column price-only feature tensor
  instead of the 10 columns it was trained on. The model returned 0.0 on
  every bar, which the generator converted to `predicted_return=-1.0` and
  emitted as a constant SELL with confidence=1.0. Swapped to
  `model_type="basic"` (real directional edge of 55-57% BUY accuracy at
  12-24h horizons). Also tightened the default `stop_loss_pct` from 0.20
  to 0.10. On BTCUSDT 1h 2024: 14.16% → 99.80% return, 7.24% → 4.74% max
  drawdown, 0.055 → 0.259 Sharpe.

---

## 2026-02-18

### Infrastructure
- Added minimal CI dependencies and enabled tests in Claude GitHub workflow (#551)
- Added Claude Code GitHub Workflow (#543)

---

## 2026-01-15

### Added
- Automated cloud training with auto data download/upload (#532)
- CoinGecko data provider as Binance alternative (#538)
- Feature schema saving with trained ML models (#530)
- `--changed` flag to run quality checks only on modified files (#529)
- Code review agents and deployment slash commands
- Automated quality checks hook for Python files
- Side utilities and validation utility modules (#500)
- Order-type execution modeling for live and backtest (#493)

### Changed
- Consolidated backtesting and live engines into unified architecture (#527)
- Removed deprecated `src/indicators` directory (#515)
- Refactored strategies for improved code quality and maintainability (#501)
- Improved ML training and cloud module code quality (#502)
- Used shared `pnl_percent` function for engine parity (#505)

### Fixed
- Prevented race conditions in position tracking (#528)
- Addressed infrastructure code quality and safety issues (#513)
- Resolved database manager bugs and improved financial data safety (#512)
- Comprehensive position management code quality and safety improvements (#507)
- Critical issues in risk management module (#509)
- Comprehensive input validation for performance module (#508)
- Made regime regression test deterministic with dependency injection (#504)
- Used relative comparison in cache performance test (#540)

### Documentation
- Comprehensive risk management architecture documentation (#518)
- Updated docs and CLI commands for cache and migrations (#533)
- Added common PR review issues to CLAUDE.md (#499)
- Added instructions to run review agents after significant changes

---

## 2025-12-28

### Added
- Stop hook with completion detection for Claude Code Web
- PSB system analysis documentation (`docs/PSB_SYSTEM_ANALYSIS.md`)
- Automated documentation system (changelog.md, project_status.md, architecture.md)
- `/update-docs` slash command for documentation maintenance
- Shared entry utilities and validation helpers for consistent engine behavior
- Comprehensive engine parity test coverage (#487)
- Correlation sizing adjustments for runtime entries (#483)

### Changed
- Enhanced CLAUDE.md with Railway environment guidelines
- Unified backtest/live entry and partial-exit logic via shared helpers
- Refactored live entry execution to use LiveEntryHandler & LiveExecutionEngine (#482)
- Routed filled live exits through LiveExitHandler (#485)
- Completed shared engine models consolidation (#475)

### Fixed
- Fixed post-fee entry balance in live entry paths (#491)
- Aligned live engine dynamic risk handling (#490)
- Honored take-profit limit pricing (#489)
- Added missing order tracking columns to positions table migration
- Recorded live exits even when filled prices exceed deviation thresholds

### Documentation
- Updated documentation links in READMEs (#488)
- Added comprehensive backtesting engine audit report (#476)
- Added performance tracker integration execplan (#467)

---

## 2025-12-22

### Changed
- Removed outdated workflows for cursor reviews and nightly code quality

---

## 2025-12-21

### Added
- Nightly performance test workflow for CI (#438)

### Changed
- Optimized ML training pipeline with performance improvements (#439)
  - Batch processing enhancements
  - Memory efficiency improvements

### Documentation
- Clarified merge-develop command in documentation
- Updated AGENTS.md with detailed execplan storage guidelines
- Enhanced PR creation guidelines for clarity

---

## 2025-12-20

### Changed
- Refactored trading bot for better code quality (#437)
  - Code organization improvements
  - Enhanced maintainability

### Documentation
- Updated CLI command consistency and accuracy across docs
- Clarified live-health invocation across guides (#429)
- Fixed broken link in prediction README (#428)

---

## 2025-12-19

### Changed
- Refactored prediction model registry and usage (#421)
  - Improved model loading patterns
  - Enhanced registry structure

### Documentation
- Updated data pipeline and model registry docs (#416)
- Refreshed nightly documentation set (#427)
- Changed documentation scan workflow from nightly to weekly

---

## Earlier Changes

For changes prior to December 2025, see the git history:
```bash
git log --oneline --since="2025-01-01"
```

---

## Categories

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Features to be removed in future versions
- **Removed**: Features that have been removed
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes
- **Documentation**: Documentation-only changes
- **Infrastructure**: CI/CD, deployment, and tooling changes
