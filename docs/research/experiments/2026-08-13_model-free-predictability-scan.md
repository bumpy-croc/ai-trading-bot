# Experiment: Model-Free Predictability/Inefficiency Scan (Phase 0)

**Author**: quant-researcher
**Status**: PREREGISTERED — locked before any statistic is computed. Do not edit thresholds after seeing results; corrections are new dated sections.
**Program**: Board-directed Phase 0, "make the system profitable across all market conditions" (2026-08-13)
**Board directive** (verbatim intent): a model-free scan across symbols and timeframes to decide WHERE to point expensive model machinery, before building any model.

---

## 0. Why this experiment, and what it is not

`docs/research/2026-07-12_returns-levers-synthesis.md` established that six independent ETHUSDT/1h
experiments (window tournament #898, architecture tournament #939, target-redesign #933,
linear input screen #967, nonlinear input screen #973, exit-geometry rounds 1-2 #970/#1013) all
converged on a ~51-53% directional-accuracy (DA) ceiling, and that the honest 365-day backtest of
the live strategy is -20.15% / 21.84% MaxDD, including a real 185-day bear window that was net
negative. That whole line of work varied *model* levers (window, architecture, target, features)
while holding symbol=ETHUSDT and timeframe=1h fixed. It never asked whether the ceiling is a
property of ETHUSDT-1h specifically, or of liquid crypto generally.

This experiment does **not** train any model. It is a pre-model, model-free statistical scan
whose only job is to tell us which (symbol, timeframe) cells have enough raw exploitable
structure to justify pointing model machinery at them next. A cell that shows no structure here
would not magically produce a trainable edge from a fancier model — the July synthesis already
demonstrated that model sophistication does not manufacture edge where none exists in the
underlying series.

## 1. Hypotheses

**H0 (null, the default we must fail to reject to keep going)**: the ~51-53% DA ceiling
and lack of tradeable structure found on ETHUSDT-1h is a general property of liquid crypto
spot markets, and does not vary meaningfully by symbol, timeframe, or statistical framing.
Structure detected across the grid, after correcting for multiple comparisons and testing against
a shuffled-return null, is statistically indistinguishable from what noise produces at the same
sample sizes, or is real but smaller than round-trip transaction costs (5-6bps/side, measured).

**H1a (symbol-specific)**: the ceiling is specific to ETHUSDT. Some other symbol in the basket
shows structure (autocorrelation, variance-ratio deviation, regime persistence, or a
directional-accuracy baseline) that both (i) beats its shuffled-return surrogate after BH
correction and (ii) is large enough to survive 2x round-trip costs (~11bps at limit-order fills,
using the measured 5-6bps/side, not the 10bps assumption).
- *Mechanism if true*: liquidity, market-maker structure, or dominant participant behavior differs
  enough by symbol (mega-cap vs mid-cap vs meme-asset) that some symbols carry more exploitable
  microstructure than ETHUSDT.
- *Falsified if*: no symbol clears both bars above.

**H1b (timeframe-specific)**: the ceiling is specific to 1h resolution. A different timeframe
(15m, 4h, or 1d) shows structure that clears the same two bars.
- *Mechanism if true*: 1h sits in a "no-man's-land" between the very-short-horizon microstructure
  effects that intraday market-makers exploit (15m) and the longer-horizon macro/trend effects
  that persist over days (1d/4h); 1h nets out to noise while the extremes don't.
- *Falsified if*: no timeframe clears both bars for any symbol.

**H1c (framing-specific)**: the ceiling is specific to *directional next-bar prediction* as a
question. Structure is real but shows up in a different statistical shape (vol clustering, mean
reversion at a specific half-life, regime persistence) that a directional classifier would never
surface, because a point directional forecast is the wrong lens.
- *Mechanism if true*: crypto returns are close to a random walk in *sign* but not in *magnitude*
  (vol) or in mean level over longer windows (mean reversion) — DA tournaments only ever tested
  the sign question.
- *Falsified if*: no vol-clustering, mean-reversion, or regime-persistence statistic clears its
  surrogate null by a margin large enough to plausibly translate into a costed edge, on any
  symbol/timeframe.

**H1d (general liquid-crypto property, i.e. H0 restated as a positive claim)**: none of the above.
No tradeable structure survives the full grid. This is a legitimate, useful outcome — it tells the
Board to stop pointing model machinery at raw-price prediction and to look at cost structure,
position sizing, portfolio construction, or genuinely different information sources instead.

The four hypotheses are not mutually exclusive (e.g., structure could be both symbol- and
timeframe-specific); the grid is designed to let evidence land on more than one cell.

## 2. Symbol selection rule (fixed BEFORE looking at any performance number)

Stratify for **testable diversity of market structure**, never for past returns. Selecting
"fast-growing" or "high-Sharpe" symbols after the fact would be selection bias and would flatter
any long-biased or momentum-biased result — this is stated explicitly so a later reader can check
we didn't cherry-pick post hoc.

| Symbol | Tier | Rationale |
|---|---|---|
| BTCUSDT | Mega-cap anchor | Deepest liquidity, most mature price discovery on Binance; the closest thing crypto has to an efficient-market baseline. |
| ETHUSDT | Mega-cap, control/reference | Already exhaustively studied (six prior nulls) — included so this scan's methodology can be validated against known results before trusting the other four cells. |
| SOLUSDT | Large-cap alt, different architecture | High-beta L1, different market-maker/derivatives ecosystem than BTC/ETH, listed since mid-2020 — enough history for a 2-year window. |
| LINKUSDT | Mid-cap alt, different sector | Oracle/infra token, not an L1 — deliberately picked to diversify away from "L1 blockchain" as a category, listed since 2019. |
| DOGEUSDT | High-volatility / low fundamentals-linkage | Meme-asset, sentiment- and flow-driven rather than fundamentals-driven, listed since 2019 (full 2-year history available) — the basket's "low-maturity behavior" proxy even though it is not literally a new listing. |

**Non-crypto-correlated reference — not included, and this is a disclosed gap, not a silent
omission.** The data providers available in this repo (`src/data_providers/`: Binance, Coinbase,
CoinGecko, Fear&Greed) are all crypto-native. There is no equities/forex/commodities provider
wired up. Adding one is out of scope for a "no new infrastructure" model-free scan. If the Board
wants a true non-crypto reference series, that needs a new data-provider integration, which is a
separate, larger piece of work.

## 3. Timeframes

15m, 1h, 4h, 1d — per Board directive, treated as a primary axis, not an afterthought, because
the 1h forming-bar issue and the DA ceiling documented in the July synthesis may both be
resolution artifacts specific to 1h.

## 4. Data window

2 years (730 days) of OHLCV per symbol/timeframe, ending at the most recent complete UTC day
before this experiment's run date, pulled via `atb data prefill-cache`. Rationale: long enough for
1d bars to have ~730 observations (adequate for Hurst/VR estimation), short enough that 15m bars
(~70,000 observations) and the full 5×4 grid stay within a single-session, sequential-download
compute budget. All five symbols have full Binance history well before this window opens, so the
window is symmetric across the basket — no symbol gets a shorter lookback than another.

## 5. Statistics, and why each is in scope

Computed per (symbol, timeframe) cell on **log returns** unless stated:

1. **Variance ratio test** (Lo-MacKinlay, VR(q) for q = 2, 4, 8, 16) — the standard test of the
   random-walk null; VR(q) != 1 indicates return predictability at that scale (VR>1 momentum,
   VR<1 mean reversion). Heteroskedasticity-robust z-stat used (returns are not homoskedastic).
2. **Return autocorrelation** at lags 1, 5, 10 — Ljung-Box Q-test for joint significance, plus the
   raw lag-1 coefficient (the single number most directly interpretable as "does yesterday's
   sign/magnitude predict today's").
3. **ARCH / vol-clustering** — Ljung-Box Q-test on **squared** returns at lags 1, 5, 10. Tests
   whether volatility itself is autocorrelated (a real, well-documented crypto phenomenon) even
   when raw returns are not — this is the primary test for hypothesis H1c.
4. **Hurst exponent** via rescaled-range (R/S) analysis on the return series. H > 0.5 indicates
   trending/persistent behavior, H < 0.5 mean-reverting, H = 0.5 a random walk. No clean analytic
   null exists for R/S at finite sample sizes, so this statistic is evaluated purely against the
   bootstrap surrogate (below).
5. **Regime persistence** — classify each bar into a high/low realized-vol regime (rolling 20-bar
   realized vol above/below its own median), compute the Markov transition probability
   P(stay in same regime | in that regime). A value well above 0.5 indicates regimes cluster in
   time (exploitable for regime-conditional sizing even without directional edge); evaluated
   against the bootstrap surrogate.
6. **Naive directional-accuracy ceiling** — two baselines: (a) momentum ("predict same sign as
   previous bar"), (b) mean-reversion ("predict opposite sign as previous bar"). Reported with a
   Wilson 95% CI and an exact binomial test against 0.5. This is the model-free analogue of the
   51-53% DA ceiling the July tournaments found with actual trained models — if even the dumbest
   possible baseline can't beat the shuffled-return null here, no model will manufacture an edge
   the raw series doesn't contain.

Spectral/seasonality check: a coarse day-of-week / hour-of-day mean-return breakdown (ANOVA F-test
across buckets) is included as a cheap secondary check, reported but not part of the primary
grading — seasonality effects this simple would likely already be arbitraged away in a market this
liquid, and are included mainly to rule out an obvious miss rather than as a primary candidate.

## 6. Null construction (the part that must be right)

**Method: i.i.d. bootstrap shuffle of the return series**, not phase randomization. Phase
randomization preserves the linear autocorrelation/power spectrum of the original series by
construction, which would make it a *useless* null for statistics 1-3 above (VR, autocorrelation,
ARCH) — those statistics are exactly what phase randomization holds fixed. An i.i.d. shuffle
destroys all temporal structure (autocorrelation, vol clustering, regime persistence) while
preserving the marginal distribution (mean, variance, skew, kurtosis) of returns exactly. This is
the correct null for "is there structure beyond what an i.i.d. draw from this same
return-distribution would produce."

- **N = 500 shuffles** per (symbol, timeframe) cell for statistics 4-6 (Hurst, regime persistence,
  directional-accuracy baselines), which lack a clean closed-form null.
- For statistics 1-3 (VR, autocorrelation, ARCH), the asymptotic analytic p-values (Lo-MacKinlay
  z-stat; Ljung-Box chi-squared) are the primary read, **cross-validated** against a reduced
  N = 200 bootstrap null on the two headline sub-statistics (VR(2), lag-1 autocorrelation) to
  confirm the analytic approximation isn't miscalibrated at our sample sizes (it can be, for
  fat-tailed, heteroskedastic crypto returns at short horizons) — this is a deliberate
  compute-scoping decision (full bootstrap on every lag of every VR order across 20 cells would be
  the expensive part of an otherwise cheap experiment) and is disclosed as such, not hidden.
- p-value convention: two-sided, `p = (1 + #{|surrogate stat| >= |observed stat|}) / (N + 1)`
  where a surrogate null is used.

**Multiple-comparison correction**: Benjamini-Hochberg FDR at q = 0.05 across the full test grid
(5 symbols x 4 timeframes x 6 primary statistics = 120 tests; VR's 4 sub-orders and
autocorrelation/ARCH's 3 lags are pooled into one representative p-value per statistic per cell
via the joint Ljung-Box / lowest-q-order VR test, to keep the correction grid at 120 rather than
several hundred near-duplicate tests). BH-FDR is chosen over Bonferroni deliberately: this is a
**screening** experiment whose job is to rank candidates for further work, not a confirmatory
test defending a single go/no-go decision (unlike the exit-geometry and input-screen tournaments,
which correctly used Bonferroni). Using Bonferroni here would bias the scan toward false negatives
exactly when the goal is "don't miss a cell worth investigating."

## 7. Cost-reality translation (the step that turns "statistically significant" into "tradeable")

For every statistic that clears its null at FDR q=0.05, translate it into an approximate
per-trade edge in basis points and compare against **measured** round-trip cost of ~11bps
(5-6bps/side, per `docs/research/experiments/2026-07-12_exit-geometry-honest.md`'s measured
figure, not the CostCalculator's default assumption). For the directional-accuracy baselines this
is direct: `EV_bps ~= (2*accuracy - 1) * avg_abs_return_bps - round_trip_cost_bps`. For VR/Hurst/
regime-persistence, the translation is necessarily rougher (these don't map to a single trade
rule) — reported as an order-of-magnitude sanity check, not a precise EV number, and labeled as
such. Any statistic that clears its statistical null but implies an edge smaller than round-trip
cost is reported as "real but not tradeable," not folded into the "structure found" count.

## 8. Success thresholds / decision each outcome triggers

Pre-committed, all branches:

- **A cell "graduates" to a model-investment candidate** if: at least one statistic in that cell
  clears BH-FDR q=0.05 against its surrogate/analytic null **AND** the cost-reality translation in
  §7 implies a per-trade edge that is positive after the measured ~11bps round-trip cost (not just
  statistically nonzero). Graduating cells get ranked and handed to pm/ml-engineer as
  model-investment candidates, ordered by translated edge size.
- **A cell shows "structure but not tradeable"** if it clears the statistical null but the
  implied edge is smaller than round-trip cost. Reported explicitly, not silently dropped — this
  distinguishes "no structure" from "structure too small to matter at our size," which are
  different findings with different implications (the latter could matter at different position
  sizes/holding periods later).
- **A cell is "clean null"** if nothing clears BH-FDR. No further action on that cell.
- **If zero cells graduate** (full-grid null): the write-up states this as the headline finding.
  Decision: recommend to pm that raw-price directional/statistical prediction be deprioritized as
  a return lever across the whole liquid-crypto/mainstream-timeframe space we can test, and that
  Phase 0's next move should be the levers the July synthesis already ranked above "smarter model"
  — trade management (lever a), diversification-for-variance rather than diversification-for-edge
  (lever b, now informed by this scan's actual per-symbol numbers), and the live/backtest parity
  gap (lever c) — rather than opening a Phase 1 model-training program on any of these cells.
- **If H1a/b/c is supported for specific cells**: recommend those specific (symbol, timeframe,
  framing) combinations to pm/ml-engineer as the Phase 1 target, in ranked order, with the
  explicit caveat set from §9.
- **Inconclusive** (e.g., analytic and bootstrap nulls disagree materially on the same statistic,
  or a cell's data has known gaps/quality issues): flagged per-cell, not smoothed into a verdict;
  next step is a targeted re-check of that cell's data quality before it counts either way.

## 9. Risks of false positive (named up front)

- **Regime-single-draw risk**: 2 years is one draw of recent market history (roughly one full
  crypto cycle leg). A statistic significant in this window may not replicate in an earlier or
  later window. This scan does not claim regime-robustness — it claims "worth a closer look,"
  which a graduating cell would still need to earn via a proper multi-fold study before any model
  or capital decision.
- **Multiple-comparison residual risk even after BH-FDR**: with 120 correlated tests (many share
  the same underlying return series across statistics and lags), BH-FDR's independence assumption
  is only approximately met; a graduating cell should be treated as "promising," never as
  "proven," consistent with how lever (a)'s `tp_06` result was treated in the July synthesis.
- **Analytic-null miscalibration**: Lo-MacKinlay and Ljung-Box asymptotics assume enough
  regularity that fat-tailed, volatility-clustered crypto returns can violate at short horizons
  (15m especially) — this is exactly why §6 cross-validates the two headline statistics against a
  bootstrap null; if they disagree, the bootstrap number governs, not the analytic one.
  Given anything that graduates is downstream policy, not a live-capital decision.
- **Cache/provider drift**: OHLCV pulled fresh via `atb data prefill-cache` in this session; any
  provider-side gaps, forward-fills, or stitching across contract changes (e.g., a symbol's
  earliest history sourced from a different venue) will be spot-checked for gap count/size and
  disclosed per symbol, not silently interpolated over.
- **Look-ahead**: none of the statistics in §5 use future information to make a same-bar decision
  — this is explicitly a backward-looking statistical characterization of realized series, not a
  live signal, so the usual look-ahead risk in signal design doesn't directly apply here, but is
  worth stating because a careless implementation of the "regime" or "vol clustering" stats could
  accidentally use a centered rolling window (using future bars) instead of a trailing one. All
  rolling calculations in the implementation must be trailing-only.

## 10. Protocol / compute discipline

- Worktree: `/Users/alex/Sites/ai-trading-bot/.claude/worktrees/model-free-scan-0813`
  (branch `claude/model-free-scan-0813`, from `origin/develop`).
- Data pulled via `atb data prefill-cache --symbols BTCUSDT ETHUSDT SOLUSDT LINKUSDT DOGEUSDT
  --timeframes 15m 1h 4h 1d --years 2`, run **once**, sequentially (no parallel downloads).
- All statistics computed in a single Python script under
  `docs/research/experiments/scripts/2026-08-13_model_free_scan.py`, using
  `src/data_providers` / `src/engines/shared` for any OHLCV access — no ad hoc duplicate data
  loading logic. Bootstrap loops run sequentially, single-process, no parallel workers, per the
  Board's thermal-load instruction for this session.
- Engine version: whatever is on `develop` at worktree-creation time (`c4a97f36`). This experiment
  does not touch backtest/live trading code, so engine-version drift is not a validity concern the
  way it is for money-metric tournaments.
- Determinism guard: the shuffle bootstrap uses a fixed seed (`numpy.random.default_rng(42)`), and
  the full grid is re-run once after the first pass completes to confirm bit-identical results
  before the write-up is finalized (cheap here since nothing is stochastic besides the seeded
  shuffle).

## 11. What this experiment explicitly does not do

- Does not train any ML model.
- Does not touch `src/` production code, `RiskParameters`, `ExitHandler`, or any live/paper
  trading path.
- Does not propose a live-affecting change. Any graduating cell is a *research* recommendation for
  where to spend the next modeling session, not a strategy-change proposal — `risk_review_required`
  does not apply to this document itself.

---

*Locked 2026-08-13, before any statistic was computed. Results appended below in a dated section
per the anti-p-hacking rule — this header and everything above it is not edited after results are
known; corrections are new dated sections.*
