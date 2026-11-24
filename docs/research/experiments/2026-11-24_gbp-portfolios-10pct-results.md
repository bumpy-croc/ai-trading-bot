# Which plain portfolios returned 10% a year in pounds: results

**Date**: 2026-11-24
**Status**: COMPLETE. No portfolio MEETS. Two MEET WITH DEEP LOSSES, both at 2x leverage. H0.
**Prereg**: [2026-11-24_gbp-portfolios-10pct-prereg.md](2026-11-24_gbp-portfolios-10pct-prereg.md), pushed as `33c77b1` before the script existed. Portfolios, costs, window and bar unchanged.
**Script**: `experiments/gbp_portfolios.py`. Re-run: `pip install xlrd`, then `python experiments/gbp_portfolios.py` from `experiments/`.
**Raw output**: `experiments/markets_results/gbp_portfolios.json`.
**Determinism**: run twice, output byte-identical (daily-return hash `e90ea7db…2f286`).

## Verdict in plain words

1. **Holding US stocks did return 10% a year in pounds over thirty years, and it was not steady.** £5,000 put into the S&P 500 at the start of 1997 was £102,000 at the end of October 2026 (10.6% a year). The first fifteen years returned 5.9% a year and the second fifteen 15.7%. It fell 52% between September 2000 and October 2002 and did not get back above its 2000 level, in pounds, until March 2012.
2. **No portfolio reached 10% a year in both halves without a fall of more than 40%.** The two that reached 10% in both halves borrow to hold twice the exposure, and fell 44% and 53%.
3. **A simple trend filter came closest among the unleveraged ones**: 9.9% a year (9.0% and 10.8% in the two halves) with a largest fall of 21%, and no losing five-year period. It misses the 10% bar by a small margin, it is a rule that many people have fitted to this same history, and since 2012 it has cost five points a year against simply holding (10.8% against 15.7%).
4. **Leverage alone was the worst trade in the table for the first half.** Twice the S&P 500, held, fell 86% and was under water for fourteen years. Its 13.9% a year over the full window comes entirely from the second half.
5. **Mixing in bonds cut the falls and the return together.** A 60/40 portfolio returned 7.7% a year with a largest fall of 25%. It never returned 10% a year over any ten-year period in the window.
6. **Over 150 years the picture is the same.** US stocks returned 9.2% a year in dollars from 1871 to 2023. Ten-year periods returned 10% a year or more 39% of the time; the median was 8.7%.

So 10% a year has been available from plain index investing roughly half the time over five- and ten-year periods, at the price of occasionally losing half and waiting a decade. Nothing tested here, or in the forex bake-off of 2026-11-10, delivers 10% a year reliably.

## Common window: 1997-01-01 to 2026-10-31, in pounds

Dividends reinvested, after fund costs and switching costs, before any platform fee, UK tax or inflation.

| | Portfolio | Return a year | 1997-2011 | 2012-2026 | Largest fall | Longest time below a previous peak | Worst year | £5,000 became | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| P1 | US stocks | 10.6% | 5.9% | 15.7% | 51.6% | 11.5 years | -29.8% (2002) | £102,000 | SHORT |
| P2 | World stocks | 9.2% | 5.5% | 13.0% | 50.8% | 9.5 years | -27.0% (2002) | £68,700 | SHORT |
| P3 | 60/40 | 7.7% | 7.0% | 8.4% | 24.7% | 4.7 years | -10.8% (2002) | £45,600 | SHORT |
| P4 | US stocks with a trend filter | 9.9% | 9.0% | 10.8% | 20.7% | 2.7 years | -15.7% (2022) | £83,000 | SHORT |
| P5 | US stocks at 2x | 13.9% | 2.9% | 26.2% | 86.4% | 14.0 years | -54.8% (2008) | £240,000 | SHORT |
| P6 | US stocks at 2x with the trend filter | 14.5% | 12.7% | 16.4% | 44.1% | 2.9 years | -35.5% (2022) | £286,000 | MEETS WITH DEEP LOSSES |
| P7 | Stocks and bonds at 2x | 13.6% | 10.5% | 16.8% | 53.2% | 9.5 years | -36.0% (2022) | £223,000 | MEETS WITH DEEP LOSSES |
| | Pound cash | 2.8% | 4.2% | 1.5% | none | none | 0.0% (2021) | £11,500 | reference |

### How often 10% a year was reached

Periods start at each month end.

| | Portfolio | Calendar years at 10% or more | Losing years | Five-year periods at 10% a year or more | Five-year periods that lost money | Worst five years (a year) | Ten-year periods at 10% a year or more | Worst ten years (a year) |
|---|---|---|---|---|---|---|---|---|
| P1 | US stocks | 18 of 30 | 5 | 56% | 17% | -7.0% | 54% | -2.9% |
| P2 | World stocks | 19 of 30 | 7 | 47% | 13% | -5.6% | 44% | -1.0% |
| P3 | 60/40 | 14 of 30 | 5 | 7% | 0% | +0.9% | 0% | +3.0% |
| P4 | Trend filter | 13 of 30 | 6 | 35% | 0% | +2.8% | 33% | +4.4% |
| P5 | 2x | 20 of 30 | 7 | 58% | 35% | -16.1% | 58% | -13.7% |
| P6 | 2x with trend filter | 17 of 30 | 6 | 61% | 0% | +2.5% | 85% | +5.5% |
| P7 | Stocks and bonds at 2x | 18 of 30 | 6 | 55% | 12% | -4.2% | 73% | -1.6% |

### Return by calendar year (%, pounds)

| | 97 | 98 | 99 | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 (to Oct) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | 38 | 27 | 25 | -2 | -10 | -30 | 15 | 3 | 17 | 1 | 3 | -13 | 12 | 18 | 3 | 11 | 29 | 20 | 6 | 34 | 11 | 1 | 27 | 14 | 30 | -9 | 19 | 27 | 9 | 12 |
| P2 | 23 | 22 | 28 | -5 | -13 | -27 | 20 | 7 | 22 | 5 | 7 | -17 | 16 | 17 | -4 | 12 | 22 | 12 | 4 | 31 | 13 | -3 | 23 | 12 | 22 | -7 | 15 | 19 | 15 | 11 |
| P3 | 19 | 19 | 15 | 3 | -4 | -11 | 14 | 7 | 15 | 4 | 9 | -4 | 9 | 14 | 2 | 8 | 12 | 9 | 3 | 18 | 8 | -2 | 16 | 11 | 12 | -9 | 10 | 12 | 12 | 6 |
| P4 | 38 | 14 | 17 | 9 | 5 | -4 | 12 | -1 | 17 | 1 | 3 | 4 | 25 | 9 | -6 | 8 | 29 | 20 | -3 | 24 | 11 | -3 | 10 | 12 | 30 | -16 | 9 | 27 | 9 | 4 |
| P5 | 68 | 48 | 38 | -21 | -27 | -50 | 42 | 11 | 17 | 10 | 1 | -55 | 31 | 31 | -1 | 25 | 67 | 34 | 4 | 46 | 32 | -9 | 59 | 18 | 63 | -31 | 39 | 47 | 19 | 19 |
| P6 | 68 | 21 | 24 | -1 | 5 | -10 | 34 | 5 | 17 | 10 | 1 | 2 | 49 | 4 | -12 | 16 | 67 | 34 | -13 | 26 | 32 | -14 | 15 | 29 | 63 | -36 | 10 | 47 | 16 | 1 |
| P7 | 53 | 39 | 13 | 3 | -13 | -26 | 22 | 9 | 19 | 0 | 7 | -8 | 6 | 30 | 29 | 17 | 21 | 47 | 5 | 38 | 21 | -4 | 45 | 36 | 31 | -36 | 20 | 18 | 12 | 7 |
| Cash | 7 | 7 | 5 | 6 | 5 | 4 | 4 | 4 | 5 | 5 | 6 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 5 | 5 | 4 | 3 |

No portfolio made money in every year. 2022 is the year the two "safer" leveraged versions lost most (35% and 36%): stocks and bonds fell together, and the trend filter was whipsawed.

## Reading the two that reached 10% in both halves

**P6, twice the S&P 500 with the trend filter.** Its first-half figure (12.7% a year against 5.9% for holding) comes from two events: being in cash for most of 2000-02 and of 2008. Those are the two events the ten-month rule is famous for, which is why the rule is famous; its record on them is not evidence about the next fall. In the second half, where there was no long bear market, P6 returned 16.4% a year against 15.7% for simply holding the S&P 500 without leverage, and its largest fall was 44% against 26% for holding over the same years. Twice the exposure bought 0.7 points a year. The filter was in stocks 77% of the time and switched 42 times in thirty years.

**P7, stocks and long bonds at 2x.** Its first-half figure is 10.5%, half a point over the bar, and the leverage model flatters by about that much (below). It relied on bonds rising when stocks fell, which held from 2000 to 2020 and failed in 2022 (-36%). It was below its 2000 peak until 2010.

**The leverage model against a real fund.** From June 2006 the model's 2x S&P 500 returned 16.1% a year in dollars; the real ProShares fund (SSO) returned 15.4%. Largest falls 84.3% and 84.7%, daily correlation 0.995. So the 2x figures above are optimistic by about 0.7 points a year, mostly the difference between the 0.60% fee assumed and the 0.89% that fund charges.

## Beside the main window (reported, never decisive)

**From 1991, US-only portfolios, in pounds**: US stocks 12.1% a year; trend filter 11.4%; 2x 16.7%; 2x with filter 16.9%; stocks and bonds at 2x 15.6%; cash 3.6%. Largest falls unchanged. The six extra years (1991-96) were all good ones.

**US stocks since 1871, in dollars, monthly, no costs** (Shiller, to mid-2023): 9.2% a year.

| Period length | Share at 10% a year or more | Share that lost money | Worst (a year) | Median (a year) |
|---|---|---|---|---|
| 5 years | 48% | 10% | -17.3% | 9.6% |
| 10 years | 39% | 3% | -4.0% | 8.7% |
| 20 years | 35% | 0% | +2.0% | 8.1% |

**The pound.** A dollar bought 58p at the start of 1997 and 75p at the end of October 2026. That fall added 0.8 points a year to every unhedged dollar holding in the table. It is not a return to count on again.

## What this means for a UK account of £5,000

Checked on 2026-11-24 from public sources; fees and rules change, so each should be confirmed on the provider's own page.

- **Wrapper.** A stocks and shares ISA takes up to £20,000 in the 2026/27 tax year; gains and dividends inside it are free of UK tax ([AJ Bell](https://www.ajbell.co.uk/isa/isa-allowances), [Fidelity](https://www.fidelity.co.uk/markets-insights/personal-finance/personal-finance/20262027-tax-allowances-whats-changed/)).
- **Cost.** Which? (fees checked March 2026) names Freetrade, Trading 212 and InvestEngine as the cheapest platforms for funds traded on the exchange below £50,000, with no account fee ([Which?](https://www.which.co.uk/money/investing/investment-platforms/compare-investment-platform-fees-and-charges-aqzb27v3lwsp)). An S&P 500 or world index fund charges about 0.07% to 0.22% a year. P1 to P3 are each one or two such funds.
- **Selling crypto to fund it is a disposal for capital gains tax.** The annual exempt amount is £3,000 of gain in 2026/27. Whether that matters depends on what the £5,000 cost.
- **Leveraged funds** (P5 to P7) are listed in London and can usually be held in an ISA, but platforms treat them as complex products and ask for an appropriateness test, and the issuers' own documents say they are built for one day and not for holding ([interactive investor](https://ii.co.uk/analysis-commentary/risks-holding-leveraged-etfs-more-one-day-ii513155)). There is no 2x fund of P7's mix; it would have to be assembled from separate leveraged products.
- **Forex and CFD accounts.** UK brokers must publish the share of retail accounts that lose money. A third-party count of fourteen FCA-authorised brokers' own figures in April 2026 averaged 70% ([The Investors Centre](https://www.theinvestorscentre.co.uk/trading/statistics/cfd-trading/)); the FCA's own earlier samples found 76% to 82% ([FCA](https://www.fca.org.uk/news/press-releases/fca-warns-investors-cfds-risk-losing-out-protections)). That agrees with the bake-off of 2026-11-10.

Alex's country has been inferred from the pound sign and is not confirmed.

## How the prereg was implemented

Choices the prereg left open, fixed in code before any real figure was produced. The mechanics (rebalancing, the timing of the trend signal, one cost per switch, drawdown and compound-return arithmetic) were checked on synthetic series first.

- **P3** is held as three sleeves (36% S&P 500, 24% other stocks, 40% hedged bonds) reset monthly, which is the same thing as 60% P2 and 40% bonds when both reset on the same day.
- **Month-end rebalancing** happens at the month-end close; the cost is charged that day on the weight moved.
- **Trend filter**: the signal uses month-end values of the total-return index including the current month; the switch cost is charged on the day of the trade; the new position earns from the next trading day.
- **Compound annual return** uses calendar time from the first to the last day of each window.
- **The 2x funds** carry the fee and borrowing cost written in the prereg and no separate dividend-withholding charge, as written. The comparison with SSO shows the size of what that leaves out.
- **Rolling periods** are measured between month-end values.

## Caveats

- **One draw, and a good one for the US.** See the prereg, section 7. P2 is the fairer guide to what a buyer could have expected; it returned 9.2%.
- **Not blind and not out of sample.** These are records. The trend filter in particular has been examined on this history many times; the bake-off's trend rules, tested out of sample on 2015-2026, did not beat holding, and P4's second half agrees.
- **Before inflation.** UK prices rose about 2.8% a year over the window, so 10.6% was about 7.6% in buying power.
- **Before platform fees and UK tax.** True in an ISA on a platform with no account fee.
- **The Shiller series uses monthly average prices**, which smooths returns a little, and ends in mid-2023.
- **Starting valuation.** At the end of October 2026 US stocks stand after fifteen strong years, as they did in 1999. The window's own history shows what the following decade can look like. This file does not forecast.

## Decision implied by the prereg

- No portfolio MEETS; two MEET WITH DEEP LOSSES. Per the prereg this is reported to Alex with the size and length of the falls stated before the return.
- Nothing is built into the bot and no money is moved. The bot stays paused.
- This is a record of the past, not a forecast and not personal advice.
