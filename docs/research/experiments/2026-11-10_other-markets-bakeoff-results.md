# Other-markets bake-off (forex, ETFs, volatility): results

**Date**: 2026-11-10
**Status**: COMPLETE. All 15 gated candidates FAIL. H1 (forex is easier to predict) is falsified at 1h and 1d. H2 (a published rule still adds something over passive holding) is not supported.
**Prereg**: [2026-11-10_other-markets-bakeoff-prereg.md](2026-11-10_other-markets-bakeoff-prereg.md), committed as `8adb4dc` before either script existed. Parameters, costs and gates unchanged.
**Scripts**: `experiments/fx_predictability.py` (Part A), `experiments/markets_bakeoff.py` (Part B), shared code in `experiments/markets_common.py`. Re-run: `pip install lightgbm scikit-learn`, then `python experiments/markets_bakeoff.py` and `python experiments/fx_predictability.py`.
**Raw output**: `experiments/markets_results/markets_bakeoff.json`, `experiments/markets_results/fx_predictability.json`.
**Data**: cached under `experiments/markets_data/` (git-ignored, about 500 MB, re-downloadable). histdata.com 1-minute bid bars for 7 pairs (about 6.1 million bars each, no missing periods), ECB reference rates 1999-2026, BIS policy rates, Yahoo daily for 20 ETFs plus `^VIX`, `^VIX3M`, `^IRX`, data.binance.vision 1h klines for BTC and ETH.
**Determinism**: each script was run twice; the output JSON was byte-identical both times.
**Candidates that have now faced the 2015-2026 holdout**: 15.

## Verdict in plain words

1. **Forex is calmer than crypto, not easier to predict.** Majors move 8 to 11% a year against 65 to 85% for BTC and ETH. On daily bars the model's out-of-sample accuracy over 25,158 bars is 49.9%: a coin flip. On hourly bars it is 51.6 to 52.5%, which is statistically real but worth under 1 bp per trade against a round trip that costs about 1.8 bps, and most of it is not tradable at all (see the diagnostic below). Net result: minus 3.5 to minus 9.5% a year.
2. **The same pipeline finds more pattern in crypto than in forex**, not less (53.4 to 54.1% hourly accuracy), and still loses 60 to 70% a year after costs. Costs relative to the size of a typical hourly move are the same in both markets (0.17 to 0.31 of a move in FX, 0.20 to 0.27 in crypto). Moving to forex does not lower the bar.
3. **None of the nine published rules cleared a single headline gate.** Holdout Sharpe ranges from -0.19 to 0.42 against a 0.5 threshold. None has a measurable return of its own once the stock/bond market is accounted for (best alpha t-statistic 0.62, threshold 1.65).
4. **What did make money from 2015 to 2026 was holding, not trading.** S&P 500 buy and hold: 13.5% a year with a 34% drawdown. 60/40 stocks and bonds: 8.7% a year, 22% drawdown. Stocks, long bonds, gold and cash in equal parts: 6.6% a year, 18% drawdown. T-bills returned 2.1% a year over the same window. Every trading rule tested earned less than 60/40 on a risk-adjusted basis.

## Part A: can a model predict direction?

| Cell | Out-of-sample span | Bars | Accuracy (95% interval) | Break-even accuracy | Accuracy on traded bars | Trades/yr | Gross Sharpe | Net Sharpe | Net excess/yr | Max DD | Always-in net Sharpe | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FX-1h-logit | 2013-01-01 to 2026-10-30 | 596,415 | 51.59% (51.46 to 51.71) | 62.1% | 53.3% | 9,937 | 1.23 | -3.17 | -9.5% | 74.7% | -4.15 | FAIL |
| FX-1h-GBM | 2013-01-01 to 2026-10-30 | 596,415 | 52.54% (52.42 to 52.67) | 62.1% | 57.9% | 10,600 | 5.50 | -1.64 | -3.5% | 38.8% | -5.60 | FAIL |
| FX-1d-logit | 2013-01-02 to 2026-10-29 | 25,158 | 49.95% (49.34 to 50.57) | 52.2% | 51.9% | 271 | 0.19 | -0.20 | -0.5% | 10.0% | -0.65 | FAIL |
| FX-1d-GBM | 2013-01-02 to 2026-10-29 | 25,158 | 49.94% (49.32 to 50.56) | 52.2% | 50.6% | 426 | 0.05 | -0.60 | -1.1% | 14.6% | -0.90 | FAIL |
| ETF-1d-logit | 2010-01-04 to 2026-10-29 | 29,624 | 53.30% (52.73 to 53.87) | 56.8% | 52.5% | 339 | 0.61 | 0.14 | 0.7% | 19.4% | 0.58 | FAIL |
| ETF-1d-GBM | 2010-01-04 to 2026-10-29 | 29,624 | 52.45% (51.88 to 53.02) | 56.8% | 52.9% | 422 | 0.69 | -0.08 | -0.5% | 23.6% | 0.00 | FAIL |
| crypto-1h-logit | 2021-01-01 to 2026-10-31 | 102,210 | 53.39% (53.08 to 53.70) | 62.4% | 56.0% | 3,904 | -0.31 | -3.81 | -71.0% | 99.9% | -4.13 | reference |
| crypto-1h-GBM | 2021-01-01 to 2026-10-31 | 102,210 | 54.06% (53.76 to 54.37) | 62.4% | 58.0% | 4,926 | 1.65 | -3.60 | -62.3% | 99.7% | -5.39 | reference |

Gates (G1 Sharpe >= 0.5, G2 drawdown <= 25%, G4 positive since 2022, G5 t >= 2.71): no forex cell passes any of G1, G4, G5. The two FX-1d cells pass only G2, because they barely move. The ETF cells pass G2 and G4 and fail G1 and G5. "Break-even accuracy" is 0.5 + round-trip cost / (2 x mean absolute bar move); "trades/yr" counts position changes across all instruments.

Net return by year, FX-1h-GBM (the best-looking gross cell): 2013 -4.7%, 2014 -5.9%, 2015 -5.2%, 2016 -3.2%, 2017 -2.4%, 2018 -3.0%, 2019 -4.4%, 2020 +0.2%, 2021 -1.0%, 2022 0.0%, 2023 -6.7%, 2024 -7.4%, 2025 -2.7%, 2026 -1.5%. One positive year in fourteen.

**ETF accuracy of 52 to 53% is drift, not skill.** These seven ETFs rose on 53.5% of days since 2010, so a model that always says "up" scores the same or better. The always-in variant's 0.58 net Sharpe for the logit is the model sitting long most of the time; the decile rule that has to pick days earns 0.14.

### Descriptive: stability, autocorrelation, cost against move size

| Bar | Instrument | Annual vol | Lag-1 autocorr | VR(5) | VR(20) | Mean bar move (bps) | Round-trip cost / move |
|---|---|---|---|---|---|---|---|
| 1h | EURUSD | 8.5% | -0.011 | 0.98 | 0.95 | 6.9 | 0.17 |
| 1d | EURUSD | 8.3% | -0.004 | 0.97 | 0.92 | 38.4 | 0.03 |
| 1h | GBPUSD | 9.0% | -0.017 | 0.98 | 0.95 | 7.2 | 0.25 |
| 1d | GBPUSD | 8.8% | +0.013 | 1.00 | 0.88 | 39.8 | 0.05 |
| 1h | USDJPY | 9.3% | -0.004 | 0.99 | 0.98 | 7.4 | 0.19 |
| 1d | USDJPY | 9.3% | -0.004 | 0.97 | 0.96 | 40.7 | 0.03 |
| 1h | USDCHF | 9.6% | -0.000 | 1.00 | 1.00 | 7.3 | 0.27 |
| 1d | USDCHF | 10.2% | -0.015 | 0.95 | 0.81 | 40.2 | 0.05 |
| 1h | AUDUSD | 11.0% | -0.020 | 0.96 | 0.94 | 9.4 | 0.19 |
| 1d | AUDUSD | 10.6% | -0.014 | 0.98 | 0.91 | 49.0 | 0.04 |
| 1h | USDCAD | 7.8% | -0.024 | 0.94 | 0.91 | 6.5 | 0.31 |
| 1d | USDCAD | 7.5% | -0.039 | 0.92 | 0.84 | 34.7 | 0.06 |
| 1h | NZDUSD | 11.5% | -0.027 | 0.95 | 0.92 | 9.9 | 0.24 |
| 1d | NZDUSD | 10.9% | -0.016 | 0.97 | 0.92 | 51.5 | 0.05 |
| 1h | BTCUSDT | 66.3% | -0.026 | 0.92 | 0.93 | 41.5 | 0.27 |
| 1d | BTCUSDT | 64.9% | -0.063 | 0.97 | 1.05 | 223.1 | 0.05 |
| 1h | ETHUSDT | 84.7% | -0.012 | 0.94 | 0.96 | 54.9 | 0.20 |
| 1d | ETHUSDT | 85.4% | -0.065 | 0.98 | 1.08 | 301.0 | 0.04 |

Variance ratios are all within a few percent of 1.0, which is what a random walk gives. Forex is six to nine times less volatile than crypto and exactly as random.

### Post-hoc diagnostic (not preregistered, cannot promote anything): why FX-1h-GBM has a gross Sharpe of 5.5

A gross Sharpe of 5.5 invites the thought "this would work with a cheaper broker". It would not.

- 68% of the cell's gross profit comes from decisions taken at hours 15, 16 and 17 on the feed's clock, the three bars around the 17:00 New York rollover. The feed is **bid prices only**. At the rollover, spreads widen and the bid drops, then recovers as spreads normalise. A model with hour-of-day features learns to "buy the dip" in the bid. A real trader buys at the ask, which did not dip.
- Gross profit per traded bar is 0.95 bps over all hours and 0.41 bps outside those four hours. A position lasts 1.65 bars on average and costs about 1.8 bps to open and close. Even a broker charging a quarter of the assumed spread would not make the outside-rollover edge pay.
- The logit cell shows the same shape: 50% of gross profit from hours 15 to 17.

So the honest reading of the hourly forex numbers is "a little mean reversion, much smaller than the spread, plus a quote artifact".

## Part B: rule-based sources of return

Holdout = 2015-01-01 to 2026-10-31. Sharpe, t-statistic, excess return and drawdown are on excess return over T-bills, net of all costs. "Total/yr" is what the account balance would have shown (CAGR, cash interest included).

| Strategy | Design Sharpe | Holdout Sharpe | t-stat | Excess/yr | Total/yr | Max DD | Alpha t vs 60/40 | 2022+ excess/yr | G1 | G2 | G3 | G4 | G5 | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 FX carry | 0.35 | 0.04 | 0.12 | 0.0% | 2.1% | 15.9% | -0.54 | 1.4% | fail | pass | fail | pass | fail | FAIL |
| B2 FX trend | 0.29 | -0.19 | -0.64 | -1.4% | 0.7% | 23.9% | -0.53 | -1.3% | fail | pass | fail | fail | fail | FAIL |
| B3 Multi-asset trend, long/cash | 0.48 | 0.40 | 1.39 | 2.6% | 4.7% | 16.6% | -0.05 | 0.4% | fail | pass | fail | pass | fail | FAIL |
| B4 Multi-asset trend, long/short | 0.51 | -0.08 | -0.26 | -0.5% | 1.6% | 17.6% | -0.49 | 0.8% | fail | pass | fail | pass | fail | FAIL |
| B5 Dual momentum | 0.68 | 0.41 | 1.41 | 5.4% | 7.6% | 33.8% | -1.02 | 4.1% | fail | fail | fail | pass | fail | FAIL |
| B6 Sector rotation | 0.65 | 0.42 | 1.44 | 4.8% | 7.0% | 23.6% | 0.09 | 5.5% | fail | pass | fail | pass | fail | FAIL |
| B7 RSI-2 mean reversion | 0.44 | 0.41 | 1.39 | 2.5% | 4.6% | 12.8% | 0.62 | 2.9% | fail | pass | fail | pass | fail | FAIL |
| B8 Turn of the month | 0.29 | 0.15 | 0.53 | 0.9% | 3.0% | 12.4% | -0.45 | 0.5% | fail | pass | fail | pass | fail | FAIL |
| B9 Volatility risk premium | 0.28 | 0.35 | 1.21 | 3.6% | 5.8% | 25.8% | 0.03 | 0.1% | fail | fail | fail | pass | fail | FAIL |
| Control: SPY buy and hold | 0.42 | 0.69 | 2.38 | 11.1% | 13.5% | 33.8% | 0.69 | 8.5% | pass | fail | fail | pass | fail | control |
| Control: 60/40 SPY/IEF | 0.66 | 0.66 | 2.28 | 6.5% | 8.7% | 22.1% | n/a | 2.8% | pass | pass | n/a | pass | fail | control |
| Control: permanent portfolio | 0.86 | 0.62 | 2.15 | 4.4% | 6.6% | 17.8% | 0.86 | 2.7% | pass | pass | fail | pass | fail | control |

B7's secondary variant (fill at the same close the signal is computed on, reported only): holdout Sharpe 0.57, excess 3.6%/yr, drawdown 11.2%, alpha t 1.23. Better, as expected for a rule that depends on acting instantly, and still short of the alpha gate.

No strategy reached PASS or PROVISIONAL, so the robustness runs and the blend the prereg reserves for survivors were not run.

Even the passive controls fail G5. Eleven years is not enough to prove at the multiple-testing bar that stocks beat cash, which is a fair measure of how demanding that bar is and of how little an eleven-year backtest proves on its own.

### Total return by calendar year (%)

| Strategy | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (to Oct) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 FX carry | -9.2 | 6.4 | -1.3 | -1.3 | 3.0 | -0.3 | 3.3 | -0.5 | 3.8 | 12.1 | 0.9 | 10.2 |
| B2 FX trend | 5.5 | -3.1 | -2.8 | -1.2 | 1.1 | 5.9 | -8.6 | 10.8 | 1.8 | -2.4 | -0.2 | 2.5 |
| B3 Multi-asset trend, long/cash | -4.0 | 3.9 | 9.0 | -2.2 | 7.3 | 1.7 | 21.9 | -5.6 | 2.0 | 5.1 | 8.7 | 10.7 |
| B4 Multi-asset trend, long/short | 1.9 | -6.4 | 5.9 | -7.4 | 2.5 | -0.2 | 1.1 | 4.6 | 0.1 | 4.1 | 9.2 | 4.9 |
| B5 Dual momentum | -7.9 | 5.9 | 19.2 | -7.8 | 17.5 | 4.9 | 28.7 | -15.6 | 8.4 | 24.9 | 14.2 | 7.2 |
| B6 Sector rotation | -9.5 | -4.5 | 14.3 | -10.1 | 12.2 | 20.8 | 21.9 | -2.6 | 3.5 | 14.5 | 4.0 | 26.7 |
| B7 RSI-2 mean reversion | 7.3 | -1.2 | 4.9 | -2.7 | 2.3 | 3.3 | 8.1 | -1.4 | 6.3 | 17.0 | 7.4 | 4.9 |
| B8 Turn of the month | -3.3 | -2.2 | 4.0 | -1.4 | 3.5 | 15.0 | 0.5 | -2.2 | 9.6 | -2.1 | 3.2 | 12.8 |
| B9 Volatility risk premium | -3.1 | 23.2 | 37.7 | -22.0 | 11.4 | 0.7 | 14.1 | -4.7 | 25.3 | -9.9 | 7.1 | 3.5 |
| Control: SPY buy and hold | 1.2 | 12.0 | 21.7 | -4.6 | 31.2 | 18.3 | 28.7 | -18.2 | 26.2 | 24.9 | 17.7 | 11.7 |
| Control: 60/40 SPY/IEF | 1.7 | 7.7 | 13.7 | -2.0 | 21.8 | 16.6 | 15.1 | -16.6 | 16.8 | 14.2 | 13.9 | 6.3 |
| Control: permanent portfolio | -2.8 | 5.3 | 11.1 | -1.6 | 16.3 | 15.4 | 5.1 | -12.0 | 11.6 | 12.1 | 22.3 | 0.4 |

Nothing here "works in all market conditions". 2022 is the test: stocks and bonds fell together, 60/40 lost 16.6%, and the only rules that made money were FX trend (+10.8%) and long/short trend (+4.6%), both of which lose or stand still in most other years.

### Activity and overlap with the stock market (holdout)

| Strategy | Data starts | Turnover/yr | Avg gross exposure | Weekly correlation with 60/40 |
|---|---|---|---|---|
| B1 FX carry | 1999-02 | 1.4x | 1.78 | 0.17 |
| B2 FX trend | 1999-12 | 12.3x | 1.23 | -0.20 |
| B3 Multi-asset trend, long/cash | 1993-01 | 2.1x | 0.67 | 0.64 |
| B4 Multi-asset trend, long/short | 1993-01 | 3.4x | 0.81 | 0.16 |
| B5 Dual momentum | 2004-09 | 3.4x | 1.00 | 0.85 |
| B6 Sector rotation | 1999-12 | 5.9x | 0.82 | 0.58 |
| B7 RSI-2 mean reversion | 1999-03 | 17.5x | 0.12 | 0.38 |
| B8 Turn of the month | 1993-02 | 24.0x | 0.19 | 0.41 |
| B9 Volatility risk premium | 1993-01 | 3.1x | 0.28 | 0.53 |

For B7 and B8, "turnover" is entries plus exits per year per unit of capital, and "gross" is the share of days invested.

## How the prereg was implemented

Choices the prereg left open, all fixed in code before any real result was produced. Mechanics (fill timing, cost charging, turn-of-month membership, RSI-2 fills) were checked on synthetic series first, and the Part A trading code was checked with random predictions (it loses its costs, as it should).

- **Timing, Part B.** Signal on the close (or ECB fix) of day t, fill at the close of day t+1, first return earned on day t+2. B7 fills at the open of t+1. B8 is calendar-only and trades at the closes the rule names.
- **ETF holdings drift** between rebalances; costs are charged on the actual weight change at each rebalance. A sleeve with too little history for its lookback sits in cash, so B3 and B4 run partly in cash before 2007. The holdout is unaffected.
- **FX strategies** hold target weights constant between rebalances (unfunded positions). Policy rates are applied with a one-day lag. Ties in the carry ranking keep the fixed order USD, EUR, JPY, GBP, CHF, AUD, NZD, CAD, SEK, NOK.
- **T-bill accrual** uses the previous day's `^IRX` yield over actual calendar days.
- **G3** uses daily excess returns on the dates both series share, as preregistered. Because ECB fixes and US closes are hours apart, a weekly version was also computed as a check; no verdict changes (the largest move is B2, from t = -0.53 to -0.15; nothing approaches 1.65).
- **Part A features for ETFs** use dividend-adjusted open, high, low and close.
- **Part A returns** are summed per calendar day (the feed's clock) before Sharpe and drawdown are computed. Thresholds for the decile rule come from the model's predictions on its own training rows, refit each year. A training row is dropped if its target bar ends after the cutoff.
- **Part A forex positions** earn the policy-rate differential and pay the 1%/yr markup pro rata by time held, and the portfolio pays the 25% idle-margin drag.
- **Crypto reference** cells are treated as unfunded long/short positions with no financing cost, which is generous to crypto. They still lose.
- **Bad ticks.** One 1-minute bar was dropped, in USDCHF (the January 2015 Swiss franc move). Dropping a single bar does not remove the move itself, which stays in the next bar's return.

## Caveats

- **False negative risk, Part A.** One feature set and two fixed models. A FAIL means "not with the kind of model the bot uses on price history", not "nobody can". Firms that do profit from intraday forex do it with order-flow data and execution at a fraction of retail spreads, neither of which a retail account has.
- **Bid-only forex data** overstates gross predictability (the diagnostic above). It cannot understate net losses by much, because the modelled costs are at the cheap end of what retail brokers charge.
- **ECB fixes are not tradable prices.** B1 and B2 are close to zero before and after costs, so the conclusion does not hinge on a few basis points of fill quality.
- **Policy rates approximate swap rates.** Real broker swaps are usually worse than "differential minus 1%".
- **Yahoo adjusted prices** assume dividends are reinvested with no tax. Non-US holders lose 15 to 30% of US dividends to withholding. UK and EU retail accounts cannot buy the US-listed ETFs used here and would hold UCITS equivalents.
- **The holdout is one draw**: 2015 to 2026 contained one fast crash (2020), one stock-and-bond bear market (2022) and otherwise a long US equity bull market, which flatters the passive controls. Their Sharpe of 0.6 to 0.7 is above the long-run figure for stocks (about 0.4).
- **Not blind.** As the prereg says, the author knew roughly how several of these had done. The rules were taken as published.

## Decision implied by the prereg

- Nothing passes Part A: **H1 is falsified at 1h and 1d. Do not port the prediction model to forex.** Per the prereg, no retuned rerun against this window without a new preregistration; the candidate count (15) carries over.
- Nothing passes Part B: the measured best option in these markets is to hold a diversified portfolio and not trade it. That is reported to the Board as the finding.
- No paper trial is proposed. No account is opened. No live or staging change.

## What a move to forex or ETFs would take in the bot (for the record)

Not recommended on these results, but scoped so the question does not need re-asking.

- **Data**: a new `DataProvider` (the base class in `src/data_providers/data_provider.py` is exchange-neutral). Forex history is free (histdata, broker APIs); ETF daily history is free.
- **Execution**: a new `ExchangeInterface` implementation. OANDA's v20 REST API is the natural forex target (trade size down to one unit, so a small account works; free practice account). Alpaca or Interactive Brokers for ETFs (fractional shares, paper accounts).
- **Engine changes**: market hours and the weekend gap (the live engine assumes a 24/7 market), daily financing charges in P&L and reconciliation, position sizing in units with pip precision, margin and leverage caps by jurisdiction, and removing the `USDT` quote-currency assumption that about 57 source files carry.
- **Estimate**: two to three weeks of engineering plus a paper-trading period. The blocker is not engineering. It is that there is nothing in this file worth executing.
