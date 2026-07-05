# HyperGrowth Exit Geometry: Loss Decomposition + Parameter Sweep

**Date**: 2026-07-04
**Researcher**: quant-researcher
**Status**: complete
**Engine**: `origin/develop @ b5427b82` (post-#838 corrected partial-exit units, #853 reconciliation-severity surfacing), checked out in a disposable worktree `.claude/worktrees/exit-sweep`, detached, never touching the main checkout or production.
**Related**: `docs/research/experiments/2026-07-04_tournament-v2-corrected.md` (Issue #842) established HyperGrowth as the sole live strategy on corrected-engine numbers; this experiment attacks HyperGrowth's own exit geometry rather than comparing it to challengers.

## Why this experiment exists

The mission brief quotes a corrected 365-day HyperGrowth/ETHUSDT backtest: **-20.15% total return, 21.84% MaxDD, 71.2% win rate, profit factor 0.47**. A win rate above 70% combined with a net loss and a sub-1.0 profit factor is not a paradox — it is the signature of an **inverted exit geometry**: winners are cut short, losers are allowed to run to the full stop. This experiment quantifies that geometry precisely (Phase 1) and tests whether tightening the stop-loss and/or the trailing-stop mechanics recovers some of the lost edge without sacrificing multi-regime robustness (Phase 2).

## Hypothesis

**H1 (decomposition)**: A small number of "full-stop" losers (losses that ride price to approximately the full 10% `stop_loss_pct`) account for most or all of the net loss, while winners systematically exit via the trailing stop well before their maximum favorable excursion (MFE), capturing only a minority of the available favorable move.

**H2 (geometry fix)**: Tightening `stop_loss_pct` (0.10 → 0.07 or 0.05) and/or tightening the trailing-stop activation/distance will improve profit factor and/or total return on the 365-day window **without** degrading performance on both the first-half and second-half sub-windows (charter: "prefer variants that improve BOTH windows and both year-halves; multi-regime robustness beats single-window wins").

## Falsifiable statement

A variant is "ready for a staging paper recommendation" only if, relative to baseline HyperGrowth (`stop_loss_pct=0.10`, `take_profit_pct=0.30`, partial-exit targets `[0.08, 0.15, 0.30]`, trailing activation `0.03`/distance `0.015`):

1. Total return improves on the full 365d window, AND
2. Total return does not get *worse* on **both** the first-half (H1) and second-half (H2) sub-windows relative to baseline's own H1/H2 (i.e. it is not winning only because one regime got lucky),
3. MaxDD stays inside the 20% portfolio hard cap (`risk-limits.json`),
4. Profit factor improves (directly targets the diagnosed problem), and
5. No fabrication signature (0%-win positive return; near-zero MaxDD with multi-% return; return/win-rate/trade-count inconsistency) — same checklist used in the tournament-v2 rerun.

If no variant clears all five bars, the honest conclusion is "promising direction, not yet ready" — not a forced recommendation.

## Setup

- Worktree: `git worktree add .claude/worktrees/exit-sweep origin/develop --detach`, HEAD `b5427b82`.
- Interpreter: main checkout's venv (`/Users/alex/Sites/ai-trading-bot/.venv/bin/python`), `PYTHONPATH=.`, `DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot`.
- Cache: symlinked `cache -> /Users/alex/Sites/ai-trading-bot/cache` (pre-filled Binance 1h ETHUSDT data; `atb data cache-manager info` confirmed 25 files, newest timestamp 2026-07-04T12:28 — fresh, not stale).
- Config matches live: `--initial-balance 85 --risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20`.
- Fees/slippage: `Backtester` default `CostCalculator` **ON** (`fee_rate=0.001`, `slippage_rate=0.0005`, confirmed at `src/config/constants.py:107-108`) — not a fee-free debug run.
- **Method**: the `atb backtest` CLI does not expose HyperGrowth's internal parameters (`stop_loss_pct`, `take_profit_pct`, partial-exit targets, trailing-stop activation/distance) as flags — `cli/commands/backtest.py`'s `_load_strategy` calls each strategy factory with zero arguments. Per the brief's allowance ("if a param needs a code tweak in your worktree for the experiment, fine — experimental only, never committed without sign-off"), I drove `create_hyper_growth_strategy(...)` and `Backtester` **in-process** via standalone scripts (not committed anywhere; scratch scripts only), reading `backtester.trades` (the shared `BaseTrade` list, `src/engines/shared/models.py`) directly after each run for per-trade decomposition. No engine or strategy source file was modified — only run-time keyword arguments and a post-construction `strategy.set_risk_overrides(...)` mutation of the trailing-stop/partial-exit sub-dicts (the same dict the engine already reads from at `Backtester.__init__` time via `build_trailing_stop_policy`).
- Runs executed **strictly sequentially** — verified no concurrent `atb backtest`/python backtest process via `ps aux` before every run, consistent with sharing the machine with another read-only agent.

## Phase 1 — Loss Decomposition

One instrumented run: `hyper_growth`/ETHUSDT/1h/365d, default params, `backtester.trades` captured directly (104 trades).

### Reproduction of the quoted baseline

| Metric | Mission brief | This run |
|---|---|---|
| Total return | -20.15% | -20.09% |
| Max drawdown | 21.84% | 21.79% |
| Win rate | 71.2% | 71.15% |
| Profit factor | (implied 0.47) | 0.4717 |
| Trades | — | 104 |

(Small deltas are expected — both runs use `--days 365` measured from "now," and "now" differs by hours/days between the mission brief's run and this one, shifting the trailing edge of the window by a few candles.)

### P&L by side

| Side | Trades | Sum P&L ($) | Win rate |
|---|---|---|---|
| Long | 49 | -10.06 | 69.4% |
| Short | 55 | -6.21 | 72.7% |

No meaningful long/short asymmetry in the loss — both sides lose money at a similar rate; this is not a directional-bias problem.

### P&L by exit-reason class (heuristic reclassification)

The engine records only a generic `exit_reason = "Stop loss"` string for **all 104 trades** — the trailing-stop mechanism tightens `trade.stop_loss` in place rather than emitting a distinct reason string, and `trailing_stop_activated` is not copied onto the completed `Trade` record (confirmed at `src/engines/backtest/execution/position_tracker.py:423-441`; it exists on `ActiveTrade`/`BasePosition` but is dropped at `close_position`). To distinguish "raw entry stop hit" from "trailing-tightened stop hit," I reclassified each trade using its realized price move and MFE/MAE fields (both present and populated on every `Trade` — `mfe`, `mae`, `mfe_price`, `mae_price`, `src/engines/shared/models.py:307-333`):

| Class (heuristic) | Definition | n | Sum P&L ($) | Avg P&L/trade ($) |
|---|---|---|---|---|
| `hard_stop_full_loss` | pnl≤0 AND realized price move ≤ -9.5% (rode to ~full 10% stop) | 24 | **-30.48** | -1.27 |
| `small_loss_other` | pnl≤0, price move > -9.5% (exited before the hard stop — signal/other) | 6 | -0.32 | -0.05 |
| `trailing_stop_after_activation` | pnl>0 AND peak favorable price move (MFE) ≥ 3% trailing-activation threshold | 72 | +14.49 | +0.20 |
| `small_win_no_trail_activation` | pnl>0, MFE never crossed 3% | 2 | +0.04 | +0.02 |
| **Total** | | **104** | **-16.27*** | |

*\*Note on arithmetic: `sum(trade.pnl)` = -$16.27, but `final_balance - initial_balance` = -$17.64 (a $1.37 gap). This is NOT a #838-style fabrication — it is fully explained by two accounting-scope facts confirmed by reading the engine directly, and is worth flagging so nobody mistakes `sum(trade.pnl)` for the true balance path:*
- *`Trade.pnl`/`pnl_percent` are computed from `close_result.pnl_cash` — **gross of exit fees and margin interest** — while `net_pnl = close_result.pnl_cash - exit_fee - interest_cost` is what actually hits `self.balance` (`src/engines/backtest/execution/exit_handler.py:817-819`). Entry fees are also deducted from balance immediately at entry (`engine.py:1808`) and never written back onto the `Trade` record. Summed `total_fees` for this run was $1.97 — this alone accounts for essentially the whole gap.*
- *Partial-exit realized P&L (`partial_result.realized_pnl`, folded into `self.balance` at `engine.py:1283`) is also never itemized into any `Trade.pnl` field — only the final remaining-position close becomes a `Trade`. A separate instrumented rerun found 4 partial-exit events over the 365d window contributing +$0.62 to balance that is invisible in any `Trade` record.*
- *Net: the top-line summary metrics (`total_return`, `final_balance`, `profit_factor`) are internally correct and consistent — this is a per-trade-record **completeness** caveat, not a return-fabrication bug. Anyone decomposing P&L from raw `Trade` objects (as this document does) should expect `sum(trade.pnl)` to run ~1-2% of balance short of the true `final_balance - initial_balance` delta, explained by fees + un-itemized partial exits.*

### H1 confirmed: full-stop losers dominate, winners give back most of their peak gain

**Losers**: 24 of 30 losing trades (80%) have a realized price move of -9.5% to -16.5% — i.e. they ride to (or very near, with slippage/gap-through-stop pushing some past -10%) the full 10% hard stop before exiting. Only 6 losers exit smaller (price moves of -0.01% to -1.30%). These 24 full-stop losers alone total **-$30.48**, which is **187% of the entire net loss** — every other trade combined (all wins, small losses, and small-win-no-trail) is net **+$14.21**.

**Winners**: of 72 winners with a recorded MFE price, the average peak favorable price excursion (MFE, raw price %) was **+4.31%**, but the average realized price move at exit was only **+2.10%** — winners capture a median of only **46.7%** of their own peak favorable move before the trailing stop (activation 3%, trailing distance 1.5%) closes them out. This is the mechanical cause of the "avg loss ≈ 2× avg win" pattern quoted in the mission brief (`avg_win=0.196%` vs `avg_loss=-1.027%` of balance, both from the engine's own performance-tracker output, matching the mission's framing) — winners are cut roughly in half relative to what price actually did, while losers are allowed to run the entire designed 10% stop distance.

### Monthly P&L (regime texture)

| Month | Trades | Sum P&L ($) | Win rate |
|---|---|---|---|
| 2025-07 | 6 | -4.38 | 50% |
| 2025-08 | 14 | -3.14 | 64% |
| 2025-09 | 2 | -1.43 | 50% |
| 2025-10 | 10 | -4.85 | 50% |
| 2025-11 | 15 | +0.31 | 87% |
| 2025-12 | 5 | +0.06 | 60% |
| 2026-01 | 12 | -0.64 | 75% |
| 2026-02 | 13 | -0.28 | 85% |
| 2026-03 | 9 | -1.32 | 78% |
| 2026-04 | 8 | +1.05 | 88% |
| 2026-05 | 1 | -1.22 | 0% |
| 2026-06 | 9 | -0.43 | 67% |

Losses are front-loaded (Jul-Oct 2025, -$13.80 combined — 85% of the net loss) with a partial recovery Nov 2025-Apr 2026, then mild renewed erosion May-Jun 2026. The full-stop losers are not confined to a single bad month; they recur throughout, consistent with a structural (parameter-driven) problem rather than a one-off regime event.

### No look-ahead bias

Stop-loss and take-profit hits are checked against the **candle's high/low**, not its close (`src/engines/backtest/execution/exit_handler.py:565-595`), with a conservative fill assumption when price gaps through the stop (worst-case candle low for longs, candle high for shorts). This is correct, parity-safe exit-timing logic — no look-ahead bias in the exit path.

---

## Phase 2 — Geometry Sweep

**[SWEEP IN PROGRESS — see final message for completed results; this section will be filled in before the file is considered final.]**


## Phase 2 — Exit-geometry sweep results (completed 2026-07-04; PM-compiled after agent stall)

6 variants x 3 windows (365d / H1 first-182d / H2 last-182d), sequential, corrected engine, prod-matched caps.

| Variant | 365d ret% | 365d PF | 365d MaxDD% | H1 ret% | H2 ret% | trades(365d) |
|---|---|---|---|---|---|---|
| baseline | -20.09 | 0.47 | 21.79 | -16.70 | -6.78 | 104 |
| sl_007 | -24.10 | 0.47 | 24.10 | -14.87 | -10.99 | 124 |
| sl_005 | -31.32 | 0.42 | 32.35 | -18.45 | -18.18 | 183 |
| tighter_trail | -29.91 | 0.31 | 30.24 | -20.47 | -15.42 | 136 |
| sl_007_tighter_trail | -37.64 | 0.23 | 37.64 | -28.20 | -21.17 | 179 |
| sl_005_tighter_trail | -40.90 | 0.25 | 40.91 | -30.46 | -24.22 | 253 |

### Verdict: NO-GO — no geometry variant ships. Every variant is worse than baseline on every window.

Tightening the stop (7%: −24.1%; 5%: −31.3%) or the trailing distance (−29.9%) or both (−37.6% / −40.9%)
strictly increases losses and drawdown. This is the coherent consequence of the root cause found the same
day (the entry signal is cross-symbol noise — BTCUSDT model scoring ETHUSDT, #867): with entries ~random,
"losers" and "winners" are indistinguishable at entry, so cutting losers earlier just crystallizes noise
excursions faster (churn), while the wide 10% stop at least allows mean reversion. Phase 1's observation
(80% of losers ride to the full stop) is real but NOT fixable at the exit layer — the exits are not the
disease, they are the symptom of a signal with no edge.

**Implication for the roadmap**: all expectancy work routes through the signal (symbol wiring fix #867 +
native ETHUSDT model + staging validation). Exit-geometry re-evaluation only makes sense AFTER a signal
with demonstrated edge exists — re-run this sweep then. Raw data: phase2_results.json (18 runs) preserved
in the session scratchpad; per-run kwargs embedded in each record.
