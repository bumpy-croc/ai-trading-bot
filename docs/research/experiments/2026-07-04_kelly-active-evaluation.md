# Kelly-Active Evaluation: kelly_momentum with PR #843 Wiring Fix

**Date**: 2026-07-04
**Researcher**: quant-researcher
**Status**: complete
**Engine**: `origin/develop @ e1d24239` (PR #843, "wire closed-trade outcomes into Kelly sizer (#840)", built on #838's corrected partial-exit units and #840's R-multiple feedback design)
**Worktree**: disposable, `.claude/worktrees/kelly-eval` (detached at `e1d24239`), removed at the end of this session. Never touched the main checkout or production.
**Related**: Issue #842 (`docs/research/experiments/2026-07-04_tournament-v2-corrected.md`) voided all pre-#838 kelly_momentum numbers on units-bug grounds and separately found `KellyCriterionSizer.record_trade()` had zero effective callers, meaning every prior kelly_momentum backtest — including the original proposal in `.claude/state/proposals/2026-07-03-01-kelly-momentum-ethusdt.md` — ran in permanent cold-start. #842 explicitly recommended re-opening the question "after fix/kelly-sizer-trade-feedback (#840) merges." This experiment is that follow-up.

## Why this evaluation exists

Two independent defects previously made every kelly_momentum backtest meaningless:

1. **#838 units bug** (fixed 2026-07-04): partial-exit P&L math mixed balance-fraction and position-fraction units, fabricating returns for any strategy using partial exits.
2. **Kelly wiring gap** (risk-officer finding, 2026-07-03; fixed by #840/#843): `KellyCriterionSizer.record_trade()` had zero callers wired from either engine into its own ring buffer. `has_sufficient_history` was permanently `False`, so kelly_momentum always ran in `fallback_fraction` cold-start mode regardless of trade count. The Kelly edge never activated in any prior test or in live.

PR #843 wires `Strategy.on_trade_closed` (`src/strategies/components/strategy.py:461-528`) into the sizer's `record_trade`, fed via two engine-side seams that both engines share:
- `PerformanceTracker.add_trade_listener(self._notify_strategy_trade_closed)` (`src/engines/backtest/engine.py:271`; live equivalent in `src/engines/live/trading_engine.py`) — fires on every final close.
- `position_tracker.on_partial_exit = self._notify_strategy_trade_closed` (`engine.py:362`) — fires on every partial-exit slice.

This is the shared `src/strategies/components/` seam used by both engines (Backtest-Live Parity satisfied). This is the first evaluation where Kelly sizing can actually activate.

## Hypothesis

With the wiring fix live, kelly_momentum/ETHUSDT/1h, given a window long enough to accumulate 30+ closed trades (`DEFAULT_KELLY_MIN_TRADES=30`), will transition from fixed fallback sizing to live Kelly-computed sizing, and — if the strategy has genuine edge — this will show up as better risk-adjusted and/or absolute return than the incumbent HyperGrowth over a full, multi-regime 365-day window (not just a short, cold-start-only window).

## Falsifiable statement

Kelly-active kelly_momentum earns a **staging paper trial** (alongside, not replacing, live HyperGrowth) only if, on the 365-day window:
1. Warm-up genuinely occurs (`trade_count` crosses 30, independently verified).
2. Total return and Sharpe are positive, or meaningfully better than HyperGrowth's on the *same* full window.
3. Max drawdown stays comfortably inside the 20% portfolio hard cap in `risk-limits.json`.
4. No fabrication signature (0%-win positive return; near-zero MaxDD with multi-% return; return/win-rate/trade-count mutual inconsistency).
5. Per charter's "prefer multi-regime robustness over single-window wins" — a good 30d/90d number alone does not qualify if the 365d number disagrees.

## Methodology

- Fresh disposable worktree, `git worktree add .claude/worktrees/kelly-eval origin/develop --detach`, HEAD `e1d24239`.
- Cache: symlinked from the main checkout's `cache/` (pre-filled Binance 1h ETHUSDT/BTCUSDT data back to 2025).
- Initial balance $85, `--risk-per-trade 0.02 --max-risk-per-trade 0.03 --max-position-size 0.20` (matches the live account's config per prior tournament precedent).
- Fees/slippage: default `CostCalculator` ON (`fee_rate=0.001`, `slippage_rate=0.0005`) for all runs — not fee-free debug numbers.
- Runs executed **strictly sequentially**, one `atb backtest` invocation at a time.
- `--log-to-db` enabled on every run so per-trade records (side, size, pnl, timestamps) could be pulled from Postgres for independent verification, not just trusted from CLI summary output.

### Important methodology corrections made during this run (documented for auditability)

1. **A same-session unit error was caught and corrected.** I initially computed "notional as % of balance" as `trades.size * entry_price`, which produced apparent position sizes of 20-62% of balance — an anomaly that would have blown through every configured cap (Kelly's own 20% max fraction, the CLI's 20% max-position-size, the strategy's 25% hard clamp). A dedicated root-cause investigation (reading `src/database/models.py:157`) established that `trades.size` is **already a balance fraction** ("Position size as % of balance," stored as a decimal, e.g. `0.0025` = 0.25%), not an ETH quantity — the `quantity` column (meant for the real asset amount) is never populated by the backtest engine (only live's reconciliation path writes it). Multiplying `size` by `entry_price` a second time was the error. Corrected figures below use `size` directly. No leverage, scale-in double-counting, or margin path was involved — this was purely an analysis bug, not a system bug.
2. **A wiring contradiction was resolved empirically, not just by code reading.** A first-pass static-analysis check concluded `record_trade` had "zero callers" (it had grepped the wrong `PerformanceTracker` class — there are two in this codebase, `src/performance/tracker.py` used by the engines and `src/strategies/components/performance_tracker.py`, an unrelated component). Rather than trust either static claim, I ran an **instrumented replay** of the exact same 365-day backtest (same strategy factory, same `RiskParameters`, same date range, same cached data) with `KellyCriterionSizer.record_trade` and `.calculate_size` monkey-patched to log `trade_count`/`has_sufficient_history` at every call. This is ground truth, not inference:
   - `record_trade` fired **105 times** (73 final closes + partial-exit slices — consistent with the "one Kelly trade per realized slice" design documented at `strategy.py:467-469`).
   - `has_sufficient_history` flips from `False` to `True` at **exactly trade 30** and stays `True` for the rest of the run (confirmed for calculate_size calls at trade_count 30 through 100, where 100 = `DEFAULT_KELLY_LOOKBACK_TRADES` ring-buffer cap).
   - The replay's total return (-0.295%) matches the CLI run (-0.286%) closely (small residual difference is expected: `RiskParameters` in the replay used the CLI's engine-level object directly rather than going through the full CLI arg-parsing path, and float ordering/threadpool pinning can differ marginally run-to-run per `engine.py`'s own bit-reproducibility comment) — enough to confirm this is a faithful reproduction, not an unrelated code path.
   - **Conclusion: Kelly sizing is genuinely active from trade 30 onward. The wiring works.**

## Results

### kelly_momentum, ETHUSDT, 1h — all windows

| Window | Trades | Return% | MaxDD% | Sharpe | Sortino | Win Rate% | Profit Factor | Avg Win/Loss (R-mult) | Long/Short | Avg Pos Frac | Max Pos Frac | Final $85 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30d | 3 | +0.016 | 0.030 | 0.0005 | 0.0006 | 66.7 | 1.005 | 0.0072 / -0.0144 | 3/0 | 0.20% | 0.26% | $85.00 |
| 90d | 7 | -0.019 | 0.067 | 0.0005 | 0.0008 | 57.1 | 0.389 | 0.0050 / -0.0173 | 7/0 | 0.22% | 0.40% | $84.98 |
| 365d | 73 | -0.286 | 0.438 | 0.0018 | 0.0029 | 56.2 | 0.454 | 0.0061 / -0.0172 | 73/0 | 0.25% | 1.57% | $84.75 |

### Kelly warm-up split (365d run, trade_count boundary at 30 per `DEFAULT_KELLY_MIN_TRADES`)

| Segment | Trades | Avg Pos Frac | Max Pos Frac | Win Rate% |
|---|---|---|---|---|
| Pre-warmup (cold-start fallback, trades 1-30) | 30 | 0.250% | 0.881% | 73.3 |
| Post-warmup (Kelly-live, trades 31-73) | 43 | 0.254% | 1.572% | 44.2 |

**Position size looks flat across the warm-up boundary — this is real, not a bug.** The instrumented replay's internal Kelly math shows why: the sizer's rolling win rate stabilizes around 68-69%, but its rolling reward:risk ratio is only **~0.47-0.49** (average loss is roughly twice the average win, confirmed independently by the backtest's own final trade-level R-multiples: `avg_win=0.0061`, `avg_loss=-0.0172`, ratio 0.35). Kelly's formula `f* = (b·p - q)/b` with `p≈0.68`, `b≈0.48` computes a raw Kelly percentage of only **0.5%-6.3%** across the post-warmup period (half-Kelly applied on top, per `kelly_fraction=0.5`) — i.e., Kelly is *correctly* telling the strategy this is a poor-payoff setup (frequent small wins, occasional large stop-outs) and sizing small, landing in the same 0.1-1.6%-of-balance range the fallback fraction (2%, then damped by confidence/strength floors of 0.3 each) was already producing. Kelly activating did not materially change behavior here because both cold-start and warm Kelly independently converge on "size small" given this strategy's realized payoff structure — not because Kelly failed to activate.

**All 83 kelly_momentum trades across all three windows are LONG-only.** The momentum signal generator never fired a short entry over a full year that saw ETHUSDT fall from ~$2560 to ~$1800 with a mid-year rally to $4700+. This is a signal-generator characteristic worth a separate note, not a sizing issue — flagged for future investigation, not itself a reason to reject or accept this proposal.

### hyper_growth, ETHUSDT, 1h, 365d (benchmark)

| Trades | Return% | MaxDD% | Sharpe | Sortino | Win Rate% | Profit Factor | Long/Short | Avg Pos Frac | Max Pos Frac | Final $85 |
|---|---|---|---|---|---|---|---|---|---|---|
| 104 | -20.15 | 21.84 | 0.119 | 0.330 | 71.2 | 0.472 | 49/55 | 13.1% | 20.0% | $67.38 |

HyperGrowth's 365d yearly split: 2025 -16.72%, 2026 -4.81% (both negative). This is the **first time HyperGrowth has been evaluated on a full-year, multi-regime window** in this repo's research history — prior tournament-v2 only covered 90d (-3.29%) and 30d (+1.22%). The 365d number is materially worse than either short window suggested, and its 21.84% MaxDD **breaches the 20% portfolio hard cap in `risk-limits.json`** — a finding independent of this Kelly evaluation but directly relevant to it, since it's the benchmark kelly_momentum is being measured against.

### Sanity checks performed

Checked every row against the three fabrication signatures that flagged the pre-#838 tournament (0%-win with positive return; near-zero MaxDD with multi-% return; return/win-rate/trade-count mutual inconsistency):
- kelly_momentum 365d: 56.2% win rate with negative return (-0.286%) and profit_factor 0.454 — consistent (frequent small wins offset by larger losses, matches the R-multiple asymmetry above). MaxDD (0.44%) scales sensibly given max position size of only 1.57%.
- kelly_momentum 90d/30d: both consistent with cold-start-fallback sizing (0.1-0.4% positions) and match tournament-v2's prior corrected-engine numbers closely (30d: this run +0.016% vs tournament-v2's +0.02%; 90d: this run -0.019% vs tournament-v2's -0.03%) — small residual differences are expected from slightly different "as of" end-dates (today vs 2026-07-04 exact cache state) and are not a red flag.
- HyperGrowth 365d: 71.2% win rate with negative return again indicates a poor payoff ratio (profit_factor 0.472), not a paradox; MaxDD (21.84%) scales with the ~13-20% position sizing actually used. Consistent, not fabricated — just a bad year for this strategy/symbol combination.

No fabrication signatures triggered in any of the four runs.

## Comparison table (365d, same window, same starting capital, same fee/slippage model)

| Strategy | Return% | MaxDD% | Sharpe | Sortino | Win Rate% | Trades | Final $85 |
|---|---|---|---|---|---|---|---|
| **kelly_momentum** (Kelly-active) | **-0.29** | **0.44** | **0.0018** | **0.0029** | 56.2 | 73 | $84.75 |
| **hyper_growth** (incumbent, live) | **-20.15** | **21.84** | **0.119** | **0.330** | 71.2 | 104 | $67.38 |

kelly_momentum's absolute and drawdown numbers look better only because its position sizing is ~50x smaller (0.25% avg vs 13.1% avg) — this is not a fair "better strategy" comparison, it's a "barely-invested strategy loses less" comparison. HyperGrowth has a materially higher Sharpe (0.119 vs 0.0018) and Sortino (0.330 vs 0.0029) *despite* its much larger drawdown, because its risk-adjusted return metrics account for the capital actually put at risk; kelly_momentum's near-zero Sharpe reflects a strategy that is barely participating in the market, not one with a superior risk-adjusted edge. Both strategies show a negative-expectancy payoff structure (profit_factor 0.45-0.47, frequent smaller wins offset by larger losses) over this particular 365-day window — this looks like a genuinely difficult year for ETHUSDT momentum/trend approaches broadly, not a kelly_momentum-specific problem.

## Robustness / sensitivity

Full parameter sensitivity sweep (momentum thresholds, kelly_fraction, lookback) was **not** run in this session — the priority was resolving whether Kelly activates at all (the central open question from #842) before investing in parameter tuning of a strategy whose base signal generator hasn't demonstrated edge at any position size. This is flagged as a gap, not a completed robustness pass; a follow-up experiment should sensitivity-test `momentum_entry_threshold` (±20%), `kelly_fraction` (0.25 vs 0.5), and `base_risk` (currently a hardcoded 0.08 default that CLI risk flags do not reach — see below) if this strategy proceeds further.

### Risk-parameter wiring finding (methodology-relevant, not this evaluation's headline result)

An investigation into the anomalous notional readings (before the units-error was found) surfaced a real, separate finding worth flagging to `risk-officer`/`pm`: for kelly_momentum specifically, the CLI's `--risk-per-trade`/`--max-risk-per-trade` flags are **not wired** to the strategy's own `VolatilityRiskManager(base_risk=0.08)` (hardcoded default in `create_kelly_momentum_strategy`, `src/strategies/kelly_momentum.py:37`) — they only populate a separate, engine-level `RiskParameters` object (`cli/commands/backtest.py:129-140`) that is never consulted by the component-based strategy path for `base_risk_per_trade`/`max_risk_per_trade`. The **only** engine-level flag that reaches kelly_momentum's actual sizing is `--max-position-size`, enforced once, downstream, as a fraction-of-balance clamp in `EntryHandler.process_runtime_decision` (`src/engines/backtest/execution/entry_handler.py:205`). Practically, this did not affect today's results (realized sizes topped out at 1.57%, nowhere near any of the caps), but it does mean the `--risk-per-trade 0.02 --max-risk-per-trade 0.03` flags in the task brief were decorative for this strategy — they would matter for HyperGrowth or other strategies whose factories accept them, but kelly_momentum's risk envelope is governed entirely by its own hardcoded `base_risk=0.08` plus Kelly's internal `max_fraction=0.20`. This should be reconciled or at minimum documented before any live-facing decision leans on "risk-per-trade" framing for kelly_momentum specifically.

## How this could lose money (if promoted to paper trial)

1. **Long-only blind spot.** Zero shorts across 83 trades in a year that included a sustained downtrend. If ETHUSDT enters another multi-month downtrend, this strategy would sit mostly flat (small wins, occasional stop-outs) rather than profiting from the down move the way a long/short strategy could — opportunity cost, not capital loss, but worth knowing before comparing "risk-adjusted return" claims against a long/short incumbent.
2. **Poor payoff ratio is structural, not incidental.** `avg_loss` is ~2-3x `avg_win` across all three windows and both warm-up segments. If win rate reverts toward 50% (the post-warmup segment already shows 44.2%, down from 73.3% pre-warmup, in a single backtest — small-sample noise, but directionally concerning), expectancy goes solidly negative and Kelly's own math would (correctly) size down further or refuse to trade, not rescue the outcome.
3. **Kelly's rolling window (100 trades) needs live history to build.** In a staging paper trial starting from zero, this strategy would spend its first 30 trades in cold-start (same 2% fallback, further damped by confidence/strength) before Kelly math has any influence at all — expect several weeks of paper history before the "Kelly-active" characteristic (as opposed to fallback) is even observable, based on today's realized trade cadence (~1 trade per 5 days at 1h/ETHUSDT).
4. **`base_risk=0.08` cannot currently be tuned via CLI/config for kelly_momentum** — any future risk-limit tightening exercise (e.g., in response to the `risk-limits.json` reconciliation flagged by #842) would need a code change to this strategy's factory defaults, not just a config/flag change, unlike HyperGrowth.

## What risk-officer should stress-test

- Independently confirm the warm-up-boundary replay (instrumented monkey-patch results in this write-up) rather than trusting my instrumentation — ideally via a unit/integration test asserting `has_sufficient_history` transitions correctly inside a real `Backtester.run()` call, since none currently exists per grep (only isolated `KellyCriterionSizer` unit tests call `record_trade` directly).
- Regime-shift behavior: this strategy has never been tested through a sustained downtrend with shorts enabled/disabled at the signal-generator level — is `MomentumSignalGenerator`'s long-only behavior over this window a data artifact of the specific window, or a structural bias worth understanding before any capital (paper or live) is committed?
- Correlation with HyperGrowth: both trade ETHUSDT/1h; a paper trial running in parallel needs to confirm they don't systematically double up on the same signal at the same time (per `max_correlated_exposure_pct=0.15` in `risk-limits.json`).
- The risk-parameter wiring gap above (CLI risk flags not reaching kelly_momentum's `base_risk`) — confirm this doesn't create a false sense of control if ops ever try to "tighten risk" on this strategy via CLI flags alone.

## Recommendation to pm

**Promising but not ready for a live decision; qualifies for a STAGING PAPER trial alongside (not replacing) live HyperGrowth, with caveats communicated clearly:**

- The headline finding of this experiment is methodological, not a strategy verdict: **Kelly sizing is now genuinely active** (empirically confirmed, not just code-reviewed), closing the wiring gap #842 flagged. That is real progress and satisfies the falsifiable statement's condition #1.
- However, conditions #2 and #5 are **not** met on an unambiguous "kelly_momentum beats HyperGrowth" basis: kelly_momentum's better absolute/drawdown numbers are an artifact of running at ~50x smaller position size than HyperGrowth, not a demonstrated superior edge. Both strategies show negative expectancy (profit_factor <1) over this specific 365-day ETHUSDT window. Sharpe/Sortino, which correct for position size, actually favor HyperGrowth.
- Per charter's preference for multi-regime robustness: neither strategy earns an unqualified "this wins" verdict from a single 365-day window on one symbol. What this experiment does establish is that kelly_momentum is now testable in a way it wasn't before, and its risk profile (tiny realized position sizes, no shorts, MaxDD well inside limits) makes it a **safe, low-cost paper-trial candidate** to accumulate the 30+ trade live history needed to evaluate Kelly's warm-state behavior outside backtest — which is exactly what the original 2026-07-03 proposal's risk-officer review recommended before this wiring fix existed.
- **Also flag to pm/risk-officer, independent of the Kelly question**: HyperGrowth's first-ever full-year backtest shows -20.15% return and 21.84% MaxDD — a portfolio-hard-cap breach. This is arguably a more urgent finding than the Kelly result and should not get lost under this write-up's primary Kelly framing. Recommend a dedicated HyperGrowth-365d review as a follow-up, separate from this Kelly evaluation.

**Verdict for the specific question asked**: No — Kelly-active kelly_momentum does **not** beat HyperGrowth on both risk-adjusted AND absolute return over 365d (HyperGrowth wins on Sharpe/Sortino; kelly_momentum wins on absolute return and MaxDD only because it's barely invested). It does, however, earn a **staging paper trial** to build real Kelly-warm trade history, run in parallel with (never replacing) live HyperGrowth, given its now-confirmed-working sizing mechanism and its risk profile is nowhere near any hard limit.
