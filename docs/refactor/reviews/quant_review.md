# Adversarial Review: Backtest ↔ Live Parity Plan — Quant / Backtest-Fidelity Lens

> Reviewer: quant-researcher (adversarial pass, read-only). Target: `docs/refactor/backtest_live_parity_plan.md` v1 (2026-06-15 draft).
> All file:line citations verified against the worktree at
> `/Users/alex/Sites/ai-trading-bot/.claude/worktrees/backtest-live-parity` on 2026-07-05. No source files were changed to produce this review.

---

## Finding 1 — BLOCKER: §P2.5's core premise ("live decisions are gated per-candle too") is FALSE

**Rationale:** The plan's entire argument that partial-op/decision cadence isn't "parity-critical for candle-driven strategies" rests on live being gated to one decision per closed candle, same as backtest. It is not. Live re-runs the *entire* decision pipeline — entries, exits, partial-ops — on a wall-clock poll timer (30–300s), independent of candle boundaries, and can (and normally does) re-evaluate the same still-open candle many times before it closes.

**Evidence:**
- `src/engines/live/trading_engine.py:1554-1560` — `_runtime_process_decision(...)` (the live analogue of `strategy.process_candle()`) runs unconditionally every loop iteration, gated only by `safety_mode`/data-readiness — no `if new_candle_closed:` anywhere in the file (confirmed by grep, zero matches).
- `src/engines/live/trading_engine.py:1586-1593` — `_check_exit_conditions(...)` and `src/engines/live/trading_engine.py:1597-1604` — `check_partial_operations(...)` run at the identical uncontrolled cadence, also with no candle-close gate.
- `src/engines/live/loop_timing.py:94-133` (`is_data_fresh`) is an **age-threshold** check (`age_seconds <= state.data_freshness_threshold`, line 133), not a "did the candle change" check. The same closed (or still-forming) candle stays "fresh" and is reprocessed repeatedly until it ages out.
- `src/config/constants.py` (`DEFAULT_CHECK_INTERVAL=60`, `DEFAULT_MIN_CHECK_INTERVAL=30`, `DEFAULT_MAX_CHECK_INTERVAL=300`) and `loop_timing.py:63-92` (`calculate_adaptive_interval`) confirm the poll cadence is seconds-scale, not candle-period-scale (e.g. 3600s for 1h). For a 1h timeframe, live evaluates the same bar up to ~60-120 times before it closes.
- Verified independently twice: once via direct read of `trading_engine.py:1442-1621`, once via a fresh isolated subagent — both concur.

**Suggested plan revision:** Rewrite §P2.5 entirely. Drop the "not parity-critical... live decisions are gated per-candle too" claim. Reframe the cadence gap as a **first-class, high-priority item**, not an optional afterthought: (a) it means live's *effective* decision rate is far higher than backtest's, so any per-candle-only backtest strategy is not actually being tested against the loop cadence live runs at; (b) it means strategies whose signal logic is not idempotent across repeated evaluation of the same candle (e.g. state that mutates on each `process_candle` call) will behave differently in live than backtest assumes. Add this as its own ledger row in §7 with "live re-evaluates every 30-300s vs backtest once per closed bar" as the description, and make quantifying its effect a **P0/P1** priority (harness scenario), not a "later, optional" P2 line.

---

## Finding 2 — BLOCKER: Three-way (not two-way) exit-timing asymmetry; live can act on a mutating, still-forming candle

**Rationale:** §2 item #4 says "live SL can fill mid-bar at the exchange... backtest checks the SL level once per candle against high/low" — correctly identifying that SL is a resting order, but it understates the mechanism and misses that live's own bot-side TP/SL polling loop reads a candle buffer whose **last row mutates in place from partial (unclosed) WebSocket kline events**. This is not just "SL fills mid-bar" — it's that live's own `candle_high`/`candle_low` inputs to exit-condition checks can be a live-updating, not-yet-final value, structurally different from backtest's frozen, closed-bar high/low.

**Evidence:**
- `src/engines/live/execution/market_data_coordinator.py:127-170` (`get_latest_data`) — no `is_closed`/`close_time` filter anywhere in the file.
- `kline_buffer.py:205-218` (`_update_current_candle`) — docstring: *"Update the tail row's OHLCV in-place from a partial kline event."* `on_kline` treats `event_ts == tail_ts` as "current candle (open or closed — same OHLCV write)."
- `src/engines/live/execution/stop_loss_manager.py:95` → `place_stop_loss_order` — SL is a real resting exchange order, confirmed independent of the bot loop (cancel-before-close handling at `exit_coordinator.py:388-424` exists precisely because the order lives on the exchange, not in-process).
- No `place_take_profit_order` exists on `ExchangeInterface` — TP is bot-side only, polled via `exit_handler.py:648-680` (`_check_take_profit`) against `candle_high`/`candle_low`, which per the kline-buffer finding above can be a still-forming candle's live-mutating high/low.
- `src/engines/backtest/execution/exit_handler.py:534-620` (`check_exit_conditions`) reads `candle["high"]`/`candle["low"]` exactly once per call (lines 577-578, 596, 603, 616, 618) from a frozen, closed historical bar — strictly discrete, coarse-grained.
- No mention of "resting order," "forming candle," or "intrabar" timing asymmetry anywhere in `docs/architecture.md`, `docs/live_trading.md`, or `docs/backtesting.md` today.

**Suggested plan revision:** In §2, replace item #4's framing with a three-tier taxonomy: (1) SL = continuous, exchange-side, any timestamp; (2) bot-side TP (and any bot-side SL re-check) = discrete-but-frequent, loop-cadence, **potentially against a still-forming candle**; (3) backtest = discrete-and-coarse, always closed data. Add a dedicated Parity Gap Ledger (§7) row for "live TP/exit polling can read a mutating forming candle" — this is a distinct mechanism from "SL fills mid-bar" and needs its own scenario in the P0.3 matrix (e.g., a candle whose high briefly exceeds TP mid-formation, then retraces before close — live may exit, backtest never sees it because backtest only sees the final closed high, which by construction would also breach TP... but the exit *price and timing* would differ). Also feeds directly into P0.1's `SimulatedExchange` design: it must model a mutating current-bar buffer if it's to reproduce this behavior, not just closed-bar candles.

---

## Finding 3 — BLOCKER: L3's "trade sequences equal" invariant will be flaky/unachievable as stated — wall-clock order-ids and exit timestamps are structurally non-reproducible

**Rationale:** The plan's canonical `TradeRecord` tuple and L3 Hypothesis invariant "(i) trade sequences equal" imply exact equality across fields including timestamps. Several fields are generated from `time.time()`/`uuid4()` directly in live's execution path, with no clock-injection seam threaded through yet — meaning two runs of the identical scripted scenario through a `SimulatedExchange`-backed live engine will not produce identical records, even before comparing to backtest.

**Evidence:**
- `src/engines/live/execution/execution_engine.py:658` and `:452` — paper order id: `f"paper_{int(time.time() * 1000)}"`.
- `src/engines/live/execution/execution_engine.py:746-748` — entry `client_order_id`: `timestamp_ms = int(time.time() * 1000); unique_suffix = uuid.uuid4().hex[:8]`.
- `src/engines/live/execution/execution_engine.py:897-899` — exit `client_order_id`, same pattern.
- `src/engines/live/execution/position_tracker.py:365`, `exit_coordinator.py:564,620`, `recovery.py:751,778`, `reconciliation.py:2958` — all stamp `exit_time = datetime.now(UTC)` (wall clock at code-execution instant), vs. backtest's `src/engines/backtest/execution/exit_handler.py:809,821` — `exit_time=current_time`, which is the **candle timestamp** passed in from `engine.py`.
- `entry_time` fares better: `trading_engine.py:1544-1552` derives it from `current_candle.name` (candle-derived), falling back to `datetime.now(UTC)` only on an anomalous path — likely safe, but the fallback should be asserted-against in the harness, not assumed unreachable.
- Backtest has no order-id concept at all (`src/engines/backtest/models.py`, `execution_engine.py`) — trades are identified positionally, so any future extension of the tuple to include order id is a non-starter by construction.

**Suggested plan revision:** In §6 (L3 row) and §P0.1, add an explicit prerequisite: retrofit the four `datetime.now(UTC)` call sites above to consume the injected clock from `LiveLoopTimingCoordinator` before any exact-equality trade-sequence assertion is attempted — otherwise L3 will be red by construction, independent of whether the strategy logic is actually correct. Redefine the invariant itself as **tiered**, not blanket `==`:
- **Exact equality**: `symbol`, `side`, `entry_px`, `qty`, `exit_px`, `reason`, `entry_ts` (after verifying the fallback path is unreachable in the harness).
- **Exact equality contingent on clock-injection retrofit**: `exit_ts`.
- **Tolerance-based** (`math.isclose`, e.g. 1e-9 absolute): `fee_entry`, `fee_exit`, `slippage`, `interest`, `gross_pnl`, `net_pnl`, `balance_after` — these are chained float arithmetic through two independently-coded wrapper paths (e.g. live's `scaled_entry_fee = entry_fee * close_portion` at `exit_coordinator.py:546` vs backtest's direct sum at `engine.py:1402-1408`); a few ULPs of difference is plausible and economically meaningless.
- **Excluded from the tuple entirely**: any order-id field (already correctly absent, but state this explicitly as a design decision, not an oversight).

---

## Finding 4 — MAJOR: P4.1 production shadow replay is not well-posed today — model version resolved via `latest` symlink is never persisted per session/trade

**Rationale:** §P4.1 claims replay "requires persisting the live session's input snapshot (candles already cached; config already in `trading_sessions`)" — implying only candles need work. In fact the ML model version is a silent, unlogged input: `latest` is a mutable symlink, `_load_bundle` resolves it to a concrete `version_id` at load time, but that resolved id is never written to any table. If `latest` gets repointed between when a session ran and when someone replays it, the replay will silently use the WRONG model and produce a divergence that looks like a live/backtest bug but is actually a resolved-model mismatch.

**Evidence:**
- `src/prediction/models/registry.py:109-128` — `_load()` loads concrete version dirs first, then applies `latest` last so "latest symlink assignment wins" (comment, line 112); `_load_bundle` at line 138-152 does `real_dir = vdir.resolve()` and extracts `version_id = real_dir.name` — the concrete version IS known in-process at load time.
- `src/database/models.py:543-594` (`TradingSession`) — has `strategy_name`, `strategy_config` (JSONType), `symbol`, `timeframe`, but **no** `model_version`/`version_id` column.
- `src/database/models.py:134-203` (`Trade`) — has `strategy_config` (JSONType, strategy hyperparameters only) and `confidence_score`, but **no** `model_version`/`version_id` column anywhere. Confirmed via grep across `models.py`: zero matches for `version_id`/`model_version`.
- This is exactly the trap the review brief warned about: "is the model version even recorded per session?" — verified: **no**.

**Suggested plan revision:** Add an explicit P4.1 sub-task (before the replay CLI is built): stamp the resolved concrete `version_id` (per symbol/model_type) into `TradingSession` (new column or a JSON field) at session start, and ideally per-trade if hot-swap (`strategy_hot_swap.py`) can change the model mid-session — otherwise a session that hot-swaps models mid-flight cannot be replayed at all, since there's no record of *which trades used which model version*. Without this, P4.1's replay is unimplementable as scoped; it needs its own short design note, not a one-line "requires persisting... config already in trading_sessions" hand-wave.

---

## Finding 5 — MAJOR: Candle cache is a mutable, overwritten-in-place file — "candles already cached" does not mean "candles as-seen-live are preserved"

**Rationale:** P4.1 assumes the candle window for a past session can be reconstructed from cache. The cache is not an immutable ledger — it's a single parquet file per key that gets atomically **overwritten** on every refresh, and live's own loop re-pulls the latest candle via REST on every tick and merges it in. There is no versioning of "what the cache contained at time T" vs. "what it contains now." For the most recent candles in a session (which were still forming or freshly closed when live acted on them), replaying today may read a value that was revised after the fact — silently breaking the replay's well-posedness for exactly the trades where timing matters most.

**Evidence:**
- `src/data_providers/cached_data_provider.py:182-218` (`_save_to_cache`) — atomic write via `os.replace(temp_path, cache_path)` (line 218): each refresh **replaces** the file in place; no historical snapshot retained.
- Live's `update_live_data` re-pulls the latest candle via REST every loop tick and merges by index (dedup + `sort_index`) regardless of what the WS stream delivered — meaning the cache's most recent rows are a moving target during the exact window a replay would need to be stable.

**Suggested plan revision:** P4.1 must explicitly snapshot (not just "reference") the candle window at trade-decision time — e.g., append-only per-session candle archive keyed by `(session_id, symbol, timeframe)` written once at first use, immune to later cache overwrites — rather than relying on the live cache-manager's current file as a stand-in for history. State this as a required schema/storage addition in §P4.1, not assume it's already covered by "candles already cached."

---

## Finding 6 — MAJOR: Sensitivity/impact-scaling and dust/residual gaps missing from §2 and the (not-yet-created) ledger

**Rationale:** Two real, currently-unlisted divergence sources were found in the cost model and quantization logic:

**(a) Order-book impact / size-dependent slippage is absent from both engines identically.** `CostCalculator` (`src/engines/shared/cost_calculator.py:110-214`) and `execution_engine.py:294-297` (`calculate_slippage_cost`) apply a flat `slippage_rate` × notional with no order-size, book-depth, or ADV term. Since both engines share this blind spot, it's not a live-vs-backtest *divergence* per se — but it means neither engine's slippage number is trustworthy for larger position sizes, and the plan's "measured, bounded, monitored difference" framing (§1) implicitly assumes the *model* itself is sound, which it isn't for size-scaling. This is a calibration-risk gap, distinct from the mechanism gaps already ledgered, and belongs in §7 as its own row so it doesn't get silently conflated with "parity is proven" once L0-L3 go green.

**(b) Dust/residual base-asset from partial-close rounding is not tracked or fed back into sizing.** `_normalize_quantity` (`execution_engine.py:986-1030`) recomputes `quantity = value / price` fresh at each call site — it does not take a prior-position residual as input, and no `dust`/`residual` field exists on `position_tracker`. Rounding residue from a partial close (leftover base qty that doesn't round cleanly to `step_size`) silently sits as unaccounted holdings; over many partial-exit cycles, live's actual position/exposure can drift from backtest's exact-float assumption in ways P2.1 ("ExchangeRules... quantize_to_step") does not address, since P2.1 only covers forward-rounding, not residual carry-forward.

**Evidence:** as cited above; also confirmed via grep that no `market_impact`/`depth` size-scaling logic exists anywhere in `src/engines/`, and the only "dust" references found are a `$1 free-balance dust threshold` guard (`execution_engine.py:730`, unrelated) and `BORROW_DUST_EPSILON` (`stop_loss_manager.py:209-228`, a borrow-repayment completeness epsilon, not position-size dust).

**Suggested plan revision:** Add two new rows to §7's seed list: "flat slippage model — no size/impact scaling (shared blind spot, not a divergence)" and "partial-close rounding residual not tracked or fed to subsequent sizing (dust accumulation)." Scope the latter into P2.1 or split it into its own item — as written, P2.1's "Removes caveat #1" claim is only true for forward quantization, not the residual-carry failure mode.

---

## Finding 7 — MINOR: T1 (bps/trade) and T2 (%/30d cumulative) thresholds can hide sign-cancelling divergence; propose a distribution/worst-case metric alongside the mean

**Rationale:** §P4.2's headline metrics — per-trade bps delta and cumulative %/30d delta — are both susceptible to a classic offsetting-errors failure mode: N trades with large, opposite-signed per-trade deltas (e.g., +40bps, −38bps, +41bps, −39bps...) can net to a cumulative delta well under T2 while every individual trade blows past T1, or conversely a handful of trades with consistent one-directional bias could stay under a naive "average bps/trade" if it's computed as a mean rather than checked per-trade. The plan does gate on **per-trade** bps (good), but the cumulative %/30d check as the *only* aggregate statistic will not surface a systematic bias that's small per-trade but persistent in one direction (e.g., a small, permanent live-slippage-model miscalibration) if T1 happens to bound each individual instance just under threshold while T2's cumulative check is satisfied by chance cancellation with unrelated noise from other trades.

**Suggested plan revision:** In §P4.2, augment the two scalar thresholds with: (i) the full distribution of `|delta|` per trade (not just a pass/fail against T1) so a human can see if the divergence is concentrating in a subset of trade types (e.g., all partial-exits, or all short entries); (ii) a **sign-preserving cumulative sum** (not just net %) to distinguish "small consistent bias, same direction, every trade" from "large but random, cancelling" — these have very different implications for whether backtest is trustworthy going forward. A rolling worst-case (max |delta| in the window) alongside the mean/median closes the "diluted by cancellation" hole. This is a metric-design fix, not a blocker — the mechanism (weekly replay + ledger) is sound, only the choice of summary statistic needs sharpening.

---

## Finding 8 — MINOR: Funding rates correctly out of scope; confirm this explicitly rather than leaving silent

**Rationale:** The review brief asked whether funding-rate accrual (perpetual futures) is a missed divergence source. It is not applicable: this system trades spot/cross-margin only. `src/data_providers/binance_provider.py` routes exclusively via `BINANCE_ACCOUNT_TYPE` (spot/margin); no perpetual-futures client, and no `fundingRate`/`premiumIndex` API calls exist anywhere in `src/`. The single "futures" hit found (`src/data_providers/exchange_interface.py:189`) is a stale, generic docstring phrase ("Get open positions (for futures/margin trading)"), not a real code path.

**Suggested plan revision:** No action needed on funding rates themselves. But add one sentence to §2 or §9 explicitly stating "this system trades spot/cross-margin only; perpetual-futures funding-rate accrual is out of scope by design, not an oversight" — a future reader auditing the plan without this context (as I was asked to do) would otherwise reasonably flag its total absence as a gap. Making the scope boundary explicit costs one sentence and forecloses a repeat of this exact audit question.

---

## Finding 9 — MINOR: Spread/bid-ask reference price is correctly a non-issue today, but the plan should say so explicitly to close off a plausible-sounding false lead

**Rationale:** Neither engine ever references a live bid/ask quote or order-book snapshot — both apply slippage as `price × (1 ± slippage_rate)` off a candle-derived close/open (`cost_calculator.py:135-160`; live's `execution_engine.py` `apply_entry_slippage`/`apply_exit_slippage`, lines 236-278, same candle-derived price). So "spread" is not an *extra* divergence source beyond what's already modeled by the shared slippage rate — but this also means the flat slippage constant is not validated against real spread-widening in thin books/volatile regimes, which is really the same calibration-risk theme as Finding 6(a).

**Suggested plan revision:** Merge this into Finding 6's ledger row rather than treating as separate — "flat slippage/spread model, not validated against real bid-ask behavior in thin or volatile conditions" is one calibration-risk item, not two. Flagging here only so the plan doesn't need a separate investigation later; no structural change required beyond what Finding 6 already proposes.

---

## Summary table

| # | Severity | One-line issue |
|---|---|---|
| 1 | BLOCKER | §P2.5's "gated per-candle too" claim is false — live polls every 30-300s regardless of candle state |
| 2 | BLOCKER | Exit timing is a 3-way asymmetry (continuous SL / frequent-on-forming-candle TP / discrete backtest), not the 2-way the plan describes |
| 3 | BLOCKER | L3 exact trade-sequence equality is unachievable as stated — wall-clock order-ids/exit-timestamps need a clock-injection retrofit + tiered tolerance, not blanket `==` |
| 4 | MAJOR | P4.1 replay is unimplementable today — resolved `latest`-symlink model version is never persisted per session/trade |
| 5 | MAJOR | Candle cache is mutable/overwritten-in-place — "candles already cached" ≠ a stable replay input |
| 6 | MAJOR | Flat slippage (no size/impact scaling) and untracked rounding-residual dust are real gaps, absent from §2/§7 |
| 7 | MINOR | T1/T2 thresholds can hide sign-cancelling divergence — add distribution + sign-preserving cumulative metrics |
| 8 | MINOR | Funding rates correctly out of scope (spot/margin only) — state this explicitly |
| 9 | MINOR | Spread/bid-ask correctly a non-issue today — fold into the slippage-calibration ledger row (Finding 6) |
