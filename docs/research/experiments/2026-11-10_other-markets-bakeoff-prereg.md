# Other-markets bake-off (forex, ETFs, volatility): preregistration

**Date**: 2026-11-10
**Status**: PREREGISTERED. Hypotheses, parameters, costs and gates below are locked before any result is computed. Corrections go in new dated sections, never edits.
**Why**: the Board asked (2026-11-10) whether the bot's approach can make money outside crypto, starting with forex ("more stable and can be predicted a bit easier?"), and for outside-the-box ideas. The crypto bake-off the same day (`2026-11-10_strategy-bakeoff-*.md`) left nothing worth building. A separate thread covers crypto strategy mixes; this file does not.
**Scripts**: `experiments/fx_predictability.py` (Part A), `experiments/markets_bakeoff.py` (Part B). Neither existed when this file was written.

## Hypotheses

**H1 (the Board's belief).** Forex direction is predictable enough from price history that the bot's method (a model trained on past price/indicator features, predicting the next bar) earns money after retail costs.
Mechanism claimed: forex is less volatile and less news-driven hour to hour than crypto, so patterns persist.
**H0.** Forex majors are the most heavily traded and most efficiently priced market there is. Lower volatility means smaller moves to capture, not more predictable ones. Out-of-sample accuracy sits within noise of 50% and no cell clears costs.
H1 is supported if any Part A forex cell passes the gate below. H1 is falsified if none does. "Easier than crypto" is judged by running the identical pipeline on BTC and ETH as a reference.

**H2.** Some rule-based source of return that is known from the published literature, and implementable by a small retail account, has kept working since it was published, after costs, and adds something over simply holding a passive stock/bond portfolio.
**H0.** After retail costs and post-publication decay, none of them beats passive holding on a risk-adjusted basis.

## Data (all free, all reachable from the research machine)

- **FX intraday**: histdata.com 1-minute bid bars, EURUSD GBPUSD USDJPY USDCHF AUDUSD USDCAD NZDUSD, 2010-01 to 2026-10. Resampled to 1h bars and to daily bars cut at 17:00 EST (the feed's fixed clock). Bid only, so spread is modelled as a cost.
- **FX daily (long history)**: ECB euro reference rates, 1999-01 to 2026-10, converted to USD crosses for EUR JPY GBP CHF AUD NZD CAD SEK NOK.
- **Short rates**: BIS central bank policy rates (daily), same ten currencies. Used as the interest earned or paid on a currency position.
- **ETFs, VIX, T-bills**: Yahoo Finance daily, adjusted for splits and dividends (total return). SPY EFA EEM QQQ IEF TLT AGG GLD DBC VNQ, the nine original sector SPDRs (XLB XLE XLF XLI XLK XLP XLU XLV XLY), SVXY, `^VIX`, `^VIX3M`, `^IRX` (13-week T-bill yield).
- **Crypto reference (Part A only)**: data.binance.vision BTCUSDT and ETHUSDT 1h klines, 2018-01 to 2026-10.

## Costs and account assumptions (small retail account)

- **FX**, per side (half spread plus slippage), in bps of notional: EUR 0.6, JPY 0.7, GBP 0.9, AUD 0.9, CAD 1.0, CHF 1.0, NZD 1.2, SEK 2.5, NOK 2.5. For the seven histdata pairs the same numbers apply to the pair's non-USD currency. These are good-broker, liquid-hours figures; real hourly trading through the 17:00 New York rollover is worse, so hourly results are flattered, not penalised.
- **FX financing (swap)**: a position earns the policy-rate differential and pays a broker markup of 1.0%/yr on pair notional. Part B carry is implemented as currency-against-currency pairs, so the markup is charged on pair notional (1.0x capital). Part A daily cells and Part B trend pay the markup on gross notional. Part A hourly cells pay it too, pro rata by bars held.
- **FX capital**: 25% of capital sits at the broker as margin earning nothing; 75% earns the T-bill rate. So excess return of an FX strategy = position P&L minus 0.25 x T-bill. Gross exposure never exceeds 2.0x capital (regulatory caps are 30:1 EU/UK and 50:1 US, so leverage limits never bind).
- **ETFs**: 5 bps per side. No commission. Cash earns T-bill minus 0.20%/yr. No leverage. Shorts (only in B4 and the Part A ETF cells) pay 1.5%/yr on short notional.
- **Crypto reference**: 11 bps round trip (the repo's measured figure).
- Not modelled: taxes, dividend withholding for non-US holders, the fact that UK/EU retail accounts cannot buy US-listed ETFs and would use UCITS equivalents.

## Part A: can a model predict forex (and ETF) direction?

The bot's method, reduced to its testable core.

- **Features** (all computed from bars up to and including t): log returns over 1, 2, 3, 6, 12, 24, 48, 120 bars; RSI(14); MACD(12,26,9) histogram / close; close/EMA(20) - 1, close/EMA(50) - 1, close/EMA(200) - 1; ATR(14)/close; realised vol over 24 and 120 bars; Bollinger z-score (20); position of close in the trailing 24-bar range; for hourly bars, sin/cos of hour of day and day of week.
- **Target**: sign of the log return from close t to close t+1.
- **Models**, fixed hyperparameters, no tuning: (a) L2 logistic regression (C = 1.0) on standardised features; (b) LightGBM classifier, 300 trees, learning rate 0.03, 15 leaves, min 500 samples per leaf (200 for daily), subsample 0.8, colsample 0.8, seed 7.
- **Walk-forward**: one model per market family, pooled across its instruments. Train on all data before 1 January of year Y (at least three years), predict year Y, repeat. FX out-of-sample span: 2013-01 to 2026-10. ETF daily (SPY QQQ EFA EEM IEF TLT GLD, from 2005): 2010-01 to 2026-10. Crypto reference: 2021-01 to 2026-10.
- **Trading rule**: thresholds are the 10th and 90th percentile of the model's predicted probabilities on its own training set. Long when the prediction is above the 90th percentile, short when below the 10th, flat otherwise. Decision at close t, position held over bar t+1. Cost charged on every change of position. Portfolio = equal weight across instruments, each instrument at 1.0x its share of capital.
- **Cells (gated)**: FX-1h-logit, FX-1h-GBM, FX-1d-logit, FX-1d-GBM, ETF-1d-logit, ETF-1d-GBM. **Reference (not gated)**: crypto-1h-logit, crypto-1h-GBM.
- **Primary metric**: net annualised Sharpe of the pooled portfolio over the out-of-sample span. **Secondary**: pooled directional accuracy with a 95% interval, break-even accuracy implied by costs, gross Sharpe, trades per year, net return by calendar year, and the always-in-the-market variant (sign of p - 0.5).
- **Descriptive only, cannot promote anything**: per-pair lag-1 autocorrelation and variance ratios (5 and 20 bars) at 1h and 1d; realised volatility of FX vs crypto; cost as a fraction of the typical bar move.

## Part B: rule-based sources of return (canonical published rules, no sweeps)

Timing unless stated: signal on the close of day t, fill at the close of day t+1. Monthly rules use the last trading day of the month as t.

- **B1 FX carry (G10).** Month end: rank the ten currencies by policy rate. Long the top three, short the bottom three, one third of capital each side per currency (gross 2.0). Return = spot move + rate differential - markup - trading costs.
- **B2 FX trend.** Nine currencies vs USD. Weekly (last fix of the week): direction = sign of the trailing 252-day spot return. Size = (10% / 60-day realised vol) / 9 per currency, total gross capped at 2.0. Earns or pays the rate differential, pays markup and trading costs.
- **B3 Multi-asset trend, long or cash (Faber GTAA 5, published 2007).** SPY EFA IEF DBC VNQ, 20% each. Month end: hold a sleeve if its total-return price is above its 10-month average (210 trading days), else T-bills.
- **B4 Multi-asset trend, long/short (managed-futures style, after Moskowitz, Ooi, Pedersen 2012).** SPY EFA EEM IEF TLT GLD DBC VNQ. Month end: long if the trailing 252-day total return exceeds the T-bill return over the same days, else short. Size = (10% / 60-day realised vol) / 8, total gross capped at 1.0.
- **B5 Dual momentum (Antonacci GEM, published 2014).** Month end: if SPY's 252-day return beats T-bills, hold whichever of SPY and EFA has the higher 252-day return; otherwise hold AGG.
- **B6 Sector rotation (Faber 2010).** Month end: hold the three of nine sector SPDRs with the highest 252-day return, equal weight, only while SPY is above its 210-day average; else T-bills.
- **B7 Short-term mean reversion (Connors RSI-2, published 2009).** SPY and QQQ, half of capital each. Enter long when RSI(2) < 10 and close > 200-day average; exit when close > 5-day average. Primary: signal at close t, fill at open t+1 (strictly causal). Secondary, reported only: fill at close t.
- **B8 Turn of the month.** Hold SPY from the close of the second-to-last trading day of the month to the close of the third trading day of the next month; T-bills otherwise.
- **B9 Volatility risk premium.** Hold 25% of capital in SVXY when `^VIX` < `^VIX3M`, else T-bills. (SVXY from 2011-10; its February 2018 collapse is inside the holdout.)
- **Controls, run through the same gates but not candidates**: T-bills; SPY buy and hold; 60/40 (SPY/IEF, rebalanced monthly); four-way permanent portfolio (SPY TLT GLD T-bills, 25% each, rebalanced yearly).

## Windows

- **Design window**: start of data to 2014-12-31. Reported, not gated. Every Part B rule was published or in wide use before 2015, so its authors had seen this period.
- **Holdout**: 2015-01-01 to 2026-10-31 (11.8 years). All Part B gates are measured here. For Part A the whole walk-forward span is out-of-sample and the gates are measured on it.

## Pass gate

Measured net of all costs, on excess return over T-bills.

1. **G1 worth having**: annualised Sharpe >= 0.5.
2. **G2 survivable**: max drawdown <= 25%.
3. **G3 not just the stock market** (Part B only): intercept of a regression of daily excess returns on the 60/40 control's excess returns is positive with t >= 1.65.
4. **G4 still alive**: mean excess return over 2022-01-01 to 2026-10-31 is positive.
5. **G5 not luck from trying many things**: the t-statistic of mean excess return over the gate window (Sharpe x sqrt(years)) is at least the one-sided 5% critical value Bonferroni-adjusted for all **15** gated candidates in this file (6 in Part A, 9 in Part B): z = 2.71. Over 11.8 years that needs a Sharpe of about 0.79.

**PASS** = all of G1 to G5. **PROVISIONAL** = G1 to G4 but not G5: real enough to look at, not proven, and its Sharpe should be assumed to halve going forward. **FAIL** = anything else.

Robustness, reported for every PASS or PROVISIONAL strategy: change its main lookback by +/-50% and report the holdout Sharpe; a sign flip downgrades it to FAIL.

## Decision each outcome triggers

- **Any PASS**: propose to the Board a paper-trading trial of that strategy on a broker demo account. Opening any account, demo or real, needs the Board's go-ahead.
- **PROVISIONAL only**: report with the haircut. Recommend a paper trial only if the haircut net return still beats T-bills by 2%/yr.
- **Nothing passes Part A**: H1 is falsified at these horizons. Recommend against porting the prediction model to forex. Do not rerun with tuned features, thresholds or models against the same window; that needs a new preregistration and the candidate count carries over.
- **Nothing passes Part B**: say so. Report what the passive controls earned, since "hold a diversified portfolio and stop trading" is then the measured best option.
- A blend of whatever is PASS or PROVISIONAL (equal capital, monthly rebalanced) is reported as descriptive only, because choosing its members on the holdout biases it upward.

## Risks of a false positive (and of a false negative)

- **Not blind.** The author knows roughly how several of these did after 2015 (trend had a strong 2022, short volatility blew up in 2018). The rules are the published canonical versions precisely so that knowledge cannot shape them.
- **Fifteen candidates.** G5 handles this. PROVISIONAL results are flagged as possibly luck.
- **Data.** histdata is bid-only with gaps around weekends and holidays; ECB fixes are once-a-day reference prices, not tradable quotes; Yahoo adjusted prices are occasionally revised. Policy rates stand in for the true tom-next swap rates. Bad ticks: any 1-minute return above 3% in absolute value is dropped and counted in the results.
- **Same-close fills** flatter B7's secondary variant; it is not the gate.
- **False negative.** Fixed hyperparameters and one feature set may miss a real edge that a better model would find. A FAIL here means "not with the bot's current kind of model", not "impossible for anyone". The results will say this.
- **Determinism.** Each script is run twice; results must be identical before they are written up.
