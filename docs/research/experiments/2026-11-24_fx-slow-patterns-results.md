# Slow forex patterns (weeks, months, years): results

**Date**: 2026-11-24
**Status**: COMPLETE at Stage A. No candidate qualified for the exam, so Stage B was not run and the 2015-2026 window was not touched. H0.
**Prereg**: [2026-11-24_fx-slow-patterns-prereg.md](2026-11-24_fx-slow-patterns-prereg.md), pushed as `be5c1b3` before the script existed. Rules, costs and thresholds unchanged.
**Script**: `experiments/fx_slow_patterns.py scan`. Re-run from `experiments/` with pandas, numpy, scipy and requests installed.
**Raw output**: `experiments/markets_results/fx_slow_scan.json`, `fx_slow_shortlist.json` (empty).
**Determinism**: the scan was run twice, output byte-identical.

## Verdict in plain words

1. **None of the 31 candidates was good enough in the period it was searched in.** The scan covered 1987 to 2014: 28 years, nine major pairs, 7,077 trading days. The shortlist needed a Sharpe ratio of 0.40 after retail costs and a profit in both halves. The best candidate reached 0.28. So nothing went forward to the 2015-2026 test, as the preregistration required.
2. **Before costs there is something, and it is small.** The textbook blend of carry, three-month winners and cheapness (K1) has a Sharpe ratio of 0.63 before costs; buying five-year losers against five-year winners (V2) has 0.50. These are the premiums the academic studies describe. After a retail account's spread, financing markup and idle margin they fall to 0.28 and 0.27, which is 1.2% and 2.0% a year above cash, and neither can be told apart from luck even over 28 years (t-statistics 1.48 and 1.44).
3. **Sized up to move 10% a year, the best pattern earned under 3% a year above cash** in the very period chosen to flatter it. Alex's target is 10%.
4. **The raw statistics look like chance.** Across nine pairs and twelve calendar months there are 108 seasonal averages; 5 are beyond two standard errors, and chance alone gives 4.9. Across nine pairs and five weekdays, none of 45 is, where chance gives 2. One period's return says almost nothing about the next at one week, one month, three months or twelve months.
5. **The pattern with the strongest claim, carry, is the one already tested forward.** On 1987-2014 the carry rule of 2026-11-10 has a Sharpe ratio of 0.37 after costs (0.58 before). On 2015-2026 it had 0.04 and returned 2.1% a year including cash interest. A pattern that clears the bar in the past and then stops is what this method exists to catch.

Patterns over weeks, months and years in the major currencies are visible in the academic sense, too small for a retail account to keep, and the best-established of them has faded since it was published.

## Stage A: every candidate, 1987-01-01 to 2014-12-31

Return in excess of dollar cash, after spread, a 1% a year financing markup and a quarter of capital idle as margin. "Before costs" removes all three. "+" is the rule as written in the prereg; "−" is its opposite.

| Id | Pattern | Sharpe after costs | Sharpe before costs | Excess a year | 1987-2000 | 2001-2014 | Largest fall | Trades a year (x capital) |
|---|---|---|---|---|---|---|---|---|
| K1+ | Carry + 3-month winners + cheapness, a third each | 0.28 | 0.63 | 1.2% | 1.5% | 0.8% | 14.6% | 6.7 |
| V2+ | Five-year losers against five-year winners | 0.27 | 0.50 | 2.0% | 3.3% | 0.7% | 23.6% | 5.2 |
| T3+ | Each pair, direction of the last 3 months | 0.08 | 0.39 | 0.3% | -0.6% | 1.2% | 23.9% | 5.9 |
| X3+ | Last 3 months' winners against losers | 0.04 | 0.27 | 0.0% | -0.3% | 0.2% | 24.6% | 16.8 |
| W1− | Last week's losers against winners | 0.02 | 0.41 | -0.3% | -1.2% | 0.7% | 37.1% | 129.7 |
| S2+ | All currencies against the dollar by calendar month | -0.01 | 0.26 | -0.4% | -0.2% | -0.6% | 36.5% | 13.4 |
| T6+ | Each pair, direction of the last 6 months | -0.02 | 0.28 | -0.3% | -1.3% | 0.6% | 31.0% | 4.2 |
| T1+ | Each pair, direction of the last month | -0.02 | 0.31 | -0.3% | -0.4% | -0.2% | 25.6% | 11.6 |
| T12+ | Each pair, direction of the last 12 months | -0.03 | 0.26 | -0.4% | -0.2% | -0.7% | 40.6% | 2.7 |
| X12+ | Last 12 months' winners against losers | -0.13 | 0.09 | -1.7% | -1.5% | -1.8% | 44.9% | 8.9 |
| S1− | Worst against best currencies for the coming calendar month | -0.15 | 0.14 | -1.5% | -3.7% | 0.8% | 55.4% | 27.6 |
| X1+ | Last month's winners against losers | -0.15 | 0.10 | -1.8% | -1.5% | -2.2% | 49.9% | 29.7 |
| X6+ | Last 6 months' winners against losers | -0.17 | 0.05 | -2.0% | -1.3% | -2.7% | 52.2% | 13.2 |
| V1+ | Cheap against dear on real exchange rates | -0.20 | 0.04 | -1.9% | -0.6% | -3.2% | 46.9% | 3.7 |
| W4− | Last four weeks' losers against winners | -0.25 | 0.04 | -2.7% | -4.8% | -0.7% | 67.8% | 65.5 |
| X6− | reversed | -0.28 | -0.05 | -3.0% | -4.7% | -1.3% | 66.3% | 13.2 |
| V1− | reversed | -0.29 | -0.04 | -2.6% | -4.8% | -0.4% | 63.1% | 3.7 |
| X12− | reversed | -0.30 | -0.09 | -3.3% | -4.4% | -2.2% | 71.0% | 8.9 |
| W4+ | Last four weeks' winners against losers | -0.34 | -0.04 | -3.5% | -2.4% | -4.6% | 69.6% | 65.5 |
| X1− | reversed | -0.35 | -0.10 | -3.6% | -4.8% | -2.3% | 69.7% | 29.7 |
| S1+ | Best against worst currencies for the coming calendar month | -0.43 | -0.14 | -3.6% | -2.4% | -4.8% | 66.5% | 27.6 |
| X3− | reversed | -0.49 | -0.27 | -5.0% | -5.8% | -4.3% | 77.4% | 16.8 |
| S2− | reversed | -0.53 | -0.26 | -4.3% | -5.3% | -3.3% | 75.5% | 13.4 |
| T12− | reversed | -0.55 | -0.26 | -3.9% | -5.0% | -2.8% | 75.6% | 2.7 |
| T6− | reversed | -0.58 | -0.28 | -4.0% | -4.0% | -4.1% | 69.3% | 4.2 |
| T1− | reversed | -0.64 | -0.31 | -4.2% | -4.9% | -3.4% | 71.3% | 11.6 |
| T3− | reversed | -0.69 | -0.39 | -4.7% | -4.6% | -4.7% | 74.0% | 5.9 |
| V2− | reversed | -0.73 | -0.50 | -6.5% | -8.7% | -4.4% | 87.1% | 5.2 |
| W1+ | Last week's winners against losers | -0.79 | -0.41 | -7.3% | -7.3% | -7.4% | 88.8% | 129.7 |

Qualifying (Sharpe at least 0.40 and both halves positive): none. Shortlist: empty.

What the table shows, by horizon:

- **Weeks.** Last week's losers do bounce a little (0.41 before costs), but the rule turns the account over 130 times a year and the spread takes all of it.
- **Months.** Following the direction of the last one to twelve months earns 0.26 to 0.39 before costs and nothing after. Ranking currencies against each other by recent return earns less.
- **Calendar.** Trading each currency on its own seasonal record (S1) lost money in both directions and in both halves. Month-of-year effects did not survive being traded even inside the period in which they were measured.
- **Years.** Five-year reversal (V2) is the best single rule: 3.3% a year in 1987-2000, 0.7% in 2001-2014. The real-exchange-rate version (V1) earned nothing.
- **Blend.** The textbook mix is the steadiest (largest fall 14.6%) and still earns only 1.2% a year over cash after costs.

### Where the costs go

For K1: about 2.8% a year before costs on a position that moves 4.5% a year. Idle margin takes roughly 0.9 points (a quarter of capital earning nothing while cash paid 3.7% on average), the financing markup roughly 0.5 points, and spreads under 0.1. What is left is 1.2%. Leverage does not change the ratio: sized to move 10% a year the same rule earns about 2.8% a year above cash after costs.

## Descriptive statistics, 1987-2014

**Does one period's return predict the next?** Correlation of a pair's return with its own previous return, averaged over the nine pairs. A value inside the band is indistinguishable from zero.

| Horizon | Periods | Average correlation | Band for "no pattern" (two standard errors) | Pairs outside the band |
|---|---|---|---|---|
| 1 week | 1,461 | -0.026 | ±0.052 | 1 of 9 |
| 1 month | 335 | +0.044 | ±0.109 | 1 of 9 |
| 3 months | 111 | +0.041 | ±0.190 | 1 of 9 |
| 12 months | 27 | -0.102 | ±0.385 | 0 of 9 |

**Calendar months.** Of 108 pair-and-month averages, 5 are beyond two standard errors; 4.9 are expected if there is no seasonality at all. The five: the euro and the Swiss franc weak against the dollar in January (-1.5% and -1.6% on average), the euro strong in December (+1.3%), the pound and the New Zealand dollar strong in April (+0.9% and +1.1%). Each of these has been written up somewhere as a tradable regularity. A table of 108 numbers will always contain about five such entries.

**Weekdays.** Of 45 pair-and-weekday averages, none is beyond two standard errors (2.0 expected by chance). The largest is 1.4.

## Reference, not a candidate: the carry rule of 2026-11-10 on this data

B1 (long the three highest-interest currencies, short the three lowest, monthly) on 1987-2014: Sharpe 0.37 after costs and 0.58 before, 3.1% a year over cash, largest fall 37.7%, positive in both halves (0.42 and 0.33). It would have fallen just short of this file's shortlist bar, and it would have looked like the best pattern in forex to anyone studying the period. Its record on 2015-2026, established on 2026-11-10 and not rerun: Sharpe 0.04, 2.1% a year including cash interest, with -9.2% in 2015.

## How the prereg was implemented

Choices the prereg left open, fixed in code before any real figure was produced. Mechanics were checked on synthetic series first: the ranking rule; that weights computed on data cut at a date equal those computed on the full series for every earlier signal day (no look-ahead) for all fifteen rules; interest accrual and costs against a hand calculation; and that planted momentum and planted seasonality are found.

- **Markup basis.** Rules that hold currencies against each other (W, X, S1, V1, V2, K1) are charged the markup once per pair, as B1 was. Rules that hold each currency against the dollar (T, S2) are charged on gross notional, as B2 was.
- **The dollar in V1** is ranked on its own real exchange rate, not at zero. In every other rule it is ranked at a return of zero as the prereg says.
- **Seasonal averages** use every earlier occurrence of the coming calendar month from 1975 up to the signal day.
- **The cut.** The script loads each file and cuts prices, interest rates and real exchange rates at 2014-12-31 before any signal or return is computed, and asserts it.
- **Cash** is the US policy rate.

## Caveats

- **False negative risk.** The cost model is the one fixed on 2026-11-10: a retail broker with tight spreads, a 1% financing markup and a quarter of capital idle. An institution paying none of that keeps the before-costs column, where two or three of these are real. A retail account cannot get those terms.
- **Majors only**, for the reason in the prereg. Higher-interest emerging currencies would show a larger carry before costs and larger, less knowable costs and devaluations.
- **Fifteen fixed rules** taken from the published literature. This does not show that no slow pattern exists; it shows that the well-known ones, at their standard settings, do not pay a retail account even in-sample.
- **Policy rates** stand in for deposit rates, and a 20% cap removes the 1992 defence rates.
- **Stage B was not run**, so nothing here says how these did after 2014, except carry, which was tested earlier.

## Decision implied by the prereg

- No shortlist: **H0**. The exam was not run; the 2015-2026 window stays at fifteen candidates.
- No further forex patterns at these horizons on this data. A new idea would need information that is not in the price and interest-rate history.
- Nothing is built and no account is opened. The bot stays paused.
