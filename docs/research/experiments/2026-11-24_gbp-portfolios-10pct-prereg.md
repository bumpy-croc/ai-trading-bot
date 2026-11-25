# Which plain portfolios returned 10% a year in pounds: preregistration

**Date**: 2026-11-24
**Status**: PREREGISTERED. Definitions fixed and pushed before the script exists or any figure below has been computed. Corrections go in new dated sections, never edits.
**Authority**: Alex, 2026-11-24 12:47 UTC, project thread: "Let's expand the scope a bit outside crypto. Let's say I have £5000 in crypto at the moment. Can you find some good investments or trading strategies that can be profitable with 10% + returns a year? I've heard forex can be great for this as there are quite predictable patterns".
**Script**: `experiments/gbp_portfolios.py` (output `experiments/markets_results/gbp_portfolios.json`), using the loaders in `experiments/markets_common.py`.
**Follows**: `2026-11-10_other-markets-bakeoff-results.md`. That file already answers the forex part: daily direction 49.9%, hourly patterns smaller than the spread, forex carry and trend at 2.1% and 0.7% a year, and every trading rule behind a passive 60/40 portfolio. Nothing in it is rerun here.

## 0. What is measured, and what it is not

The other-markets bake-off found that from 2015 to 2026 holding made money and trading did not. It measured holding in dollars, on US-listed funds, over eleven years that were unusually good for US stocks. Alex's question is about pounds, a small account, and a target of 10% a year. This file measures, over about thirty years that include two halvings of the stock market, what seven portfolios that a UK retail account can hold would have returned in pounds, how often they reached 10%, and what they lost on the way.

It is a measurement of known sources of return, not a search for a new edge. The author is not blind: it is common knowledge that US stocks returned roughly 10% a year over long periods and that leverage raised that in the last fifteen years. The point is to put the pound figures, the bad stretches and the cost of leverage in one table, with the rules written down first so that the portfolios cannot be chosen after seeing which did best.

## 1. Hypotheses

**H1**: at least one of the seven portfolios returned 10% a year or more in pounds over the full window and in each half of it, with a largest fall of no more than 40%.

**H0**: 10% a year in pounds was reached only in the good half, or only by accepting falls of more than 40%, or not at all.

## 2. Data

- **US stocks**: S&P 500 total-return index (`^SP500TR`, Yahoo, daily).
- **Stocks outside the US**: Vanguard Total International Stock Index fund (`VGTSX`, dividends reinvested, from 1996).
- **Bonds**: Vanguard Intermediate-Term Treasury fund (`VFITX`) and Long-Term Treasury fund (`VUSTX`), dividends reinvested.
- **Dollar cash rate**: 13-week Treasury bill yield (`^IRX`).
- **Pounds per dollar and UK cash**: Bank of England daily series `XUDLUSS` (dollars per pound) and `IUDBEDR` (Bank Rate), downloaded 2026-11-24.
- **Check on the leverage model**: ProShares Ultra S&P500 (`SSO`), a real 2x fund, from 2006.
- **Long look**: Robert Shiller's monthly US stock series from 1871 (prices and dividends, dollars).

## 3. Window

- **Common window**: 1997-01-01 to 2026-10-31, the longest that every series covers. First half 1997-01-01 to 2011-12-31, second half 2012-01-01 to 2026-10-31.
- Reported beside it, never decisive: the portfolios that need only US data from 1991-01-01, and the Shiller series from 1871.

## 4. Portfolios (fixed)

All returns are in pounds, with dividends reinvested, before any UK tax (as inside a stocks and shares ISA).

| Id | Portfolio | Rule |
|---|---|---|
| P1 | US stocks | S&P 500, held. |
| P2 | World stocks | 60% S&P 500, 40% stocks outside the US, rebalanced at each month end. |
| P3 | 60/40 | 60% P2, 40% intermediate Treasuries hedged to pounds, rebalanced at each month end. |
| P4 | US stocks with a trend filter | At each month end: if the S&P 500 total-return index is above the average of its last ten month-end values, hold P1 for the next month, otherwise hold pound cash. The switch is made at the close of the first trading day of the new month. |
| P5 | US stocks at 2x | A fund that returns twice the S&P 500's daily return, held. |
| P6 | US stocks at 2x with the trend filter | P4's signal and timing, holding P5 instead of P1. |
| P7 | Stocks and bonds at 2x | Twice the daily return of a mix of 60% S&P 500 and 40% long Treasuries (the mix reset to 60/40 daily), held. |

Definitions:

- **Pounds.** A dollar asset's daily return in pounds is (1 + its dollar return) x (pounds per dollar today / pounds per dollar on the previous trading day) − 1. The Bank of England rate is carried forward over days it is not published.
- **Hedged bonds (P3)**: the bond fund's dollar return plus UK cash rate minus dollar cash rate, accrued daily. No currency conversion is applied.
- **Pound cash**: Bank Rate less 0.10% a year, accrued over calendar days.
- **2x fund (P5 to P7)**: daily dollar return = 2 x the underlying's daily return − one day of (dollar cash rate + 0.50% a year) on the borrowed half − one day of a 0.60% a year fee. Converted to pounds as above.
- **Costs.** S&P 500 holdings: 0.30% a year (0.07% fund charge plus the 15% US withholding tax on dividends that a UK or Irish fund pays). The other Vanguard funds are already net of their own charges. Each switch or rebalance: 0.10% of the amount moved. A platform fee is not included; it is discussed in the results because it ranges from nothing to about 0.45% a year depending on the platform.
- **The trend filter uses dollar prices**, as the published versions of the rule do, and is not tuned: ten months is the published length. No other length is run.

## 5. Metrics and bar

For each portfolio, over the common window, each half, and each calendar year:

- compound annual return in pounds;
- largest fall from a peak on daily values, and the longest time below a previous peak;
- worst calendar year;
- the share of rolling five-year periods (starting each month) that returned 10% a year or more, the worst five-year period, and the same for ten years;
- what £5,000 became.

**Verdict per portfolio**:

- **MEETS**: 10% a year or more over the common window and in both halves, and a largest fall of 40% or less.
- **MEETS WITH DEEP LOSSES**: the three return conditions hold and the largest fall exceeded 40%.
- **SHORT**: anything else.

There is no significance test. These are not discoveries to be defended against chance; they are records of what happened, and thirty years is one draw.

## 6. Decision per outcome

- **A portfolio MEETS**: reported to Alex as an option, with its falls, how it would be held in a UK account and at what cost. Nothing is built into the bot and no money is moved.
- **Only MEETS WITH DEEP LOSSES**: reported as such, with the size and length of the falls stated before the return.
- **All SHORT**: reported plainly, with what return each fall bought.

In every case the report says that this is a record of the past and not a forecast or personal advice, and the bot stays paused.

## 7. Risks of a misleading result

- **Hindsight in the choice of market.** US stocks were the best large market of the period. P2 is the fairer guide to what a buyer in 1997 could have expected; P1 and everything built on it carry that hindsight.
- **One draw.** The window holds two falls of about half (2000-02, 2007-09), one fast crash (2020) and one year when stocks and bonds fell together (2022), then a long rise. The second half flatters everything, leverage most of all.
- **The 2x model is an approximation.** It is checked against a real 2x fund from 2006 and the gap is reported. Real UK-listed leveraged products are swaps with their own costs and can be closed by the issuer after a large one-day fall.
- **The trend filter is a well-known rule** that has been looked at by many people over this same history; its record here is not out-of-sample evidence. The bake-off's trend rules, tested out of sample on 2015-2026, did not beat holding.
- **Fund proxies.** The Vanguard US mutual funds stand in for the UK-listed index funds a UK account would hold; the two track the same indexes but differ slightly in charges and tax.
- **No tax and no inflation.** Returns are before UK tax (true inside an ISA) and before inflation, which averaged about 2.8% a year over the window.
- **Determinism**: the script is run twice; outputs must match.
