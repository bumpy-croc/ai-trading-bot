# Slow forex patterns (weeks, months, years): preregistration

**Date**: 2026-11-24
**Status**: PREREGISTERED. Written and pushed before the script exists. The data files were downloaded first to check how far back they go; no return, signal or pattern has been computed from them. Corrections go in new dated sections, never edits.
**Authority**: Alex, 2026-11-24 21:17 UTC, project thread: "Can you investigate Forex some more. Can you spot any patterns over weeks/months/years that would make consistent profitable trades".
**Script**: `experiments/fx_slow_patterns.py` (`scan`, then `exam`), outputs `experiments/markets_results/fx_slow_scan.json`, `fx_slow_shortlist.json`, `fx_slow_exam.json`.
**Follows**: `2026-11-10_other-markets-bakeoff-results.md`, which tested hourly and daily prediction, a three-against-three carry rule (B1) and a twelve-month trend rule (B2) on the major currencies. All failed on 2015-2026. Those are not rerun or retuned here.

## 0. The question and the method

Alex asks whether patterns can be spotted at slow horizons. Spotting patterns in a price history is easy: any history contains some. The test of a pattern is whether it is still there in data that was not used to find it. So the work is in two stages with a wall between them.

- **Stage A (the scan)** looks for patterns in 1987-2014 only. The script cuts every series at 2014-12-31 before it computes anything.
- **Stage B (the exam)** takes at most five patterns that the scan ranks highest, fixed by a rule written here, and runs each once on 2015-01-01 to 2026-10-31 with retail costs.

Both stages are reported in full, whichever way they come out.

## 1. Hypotheses

**H1**: at least one slow pattern in the major currencies that is visible in 1987-2014 kept earning in 2015-2026, after the costs a retail account pays, by enough to tell it apart from luck.

**H0**: the patterns visible in 1987-2014 are either chance, or were real and have since been competed away, and none earns more than its costs in 2015-2026.

## 2. Data

- **Exchange rates**: Bank of England daily spot rates against the dollar from 1975 (`XUDLUSS`, `XUDLERD`, `XUDLJYD`, `XUDLSFD`, `XUDLADD`, `XUDLNDD`, `XUDLCDD`, `XUDLSKD`, `XUDLNKD`; `XUDLDMD` for the German mark before 1999, joined to the euro by returns at 1999-01-04). The Bank's Canadian dollar series stops in June 2021; from then the ECB reference rates are used for it, joined by returns.
- **Interest rates**: BIS central bank policy rates, daily (Germany's before 1999 for the euro). All ten are available from March 1986. Any rate above 20% is treated as 20% (the 1992 Swedish defence rate of 500% was not a rate anyone could earn).
- **Real exchange rates**: BIS real effective exchange rate, narrow basket, monthly from 1964, as published today.
- **Currencies**: USD, EUR, JPY, GBP, CHF, AUD, NZD, CAD, SEK, NOK. Nine pairs against the dollar.
- **Periods**: scan 1987-01-01 to 2014-12-31 (first half to 2000-12-31, second half from 2001-01-01); exam 2015-01-01 to 2026-10-31. Signals may use prices from 1975.

## 3. How a pattern is turned into a trade (fixed)

Every pattern is a rule that, on a signal day, sets a weight for each of the nine currencies against the dollar. The simulator is the one used for B1 and B2 (`fx_run` in `experiments/markets_bakeoff.py`), unchanged:

- signal at the close of day t, traded at the close of day t+1, first return on day t+2;
- return of a position = change in the exchange rate plus the interest-rate difference, less a 1% a year broker markup on the notional;
- cost per trade: the per-currency half-spread table in `experiments/markets_common.py` (0.6 to 2.5 bp a side);
- a quarter of the capital sits idle at the broker as margin and earns nothing.

Signals use spot returns (no interest). Ranks include the dollar itself with a return of zero. "Top three against bottom three" means +1/3 in each of the three highest-ranked currencies and −1/3 in each of the three lowest, the dollar leg being implied, as in B1.

## 4. The patterns scanned (all of them; nothing is added after the scan)

| Id | Horizon | Rule |
|---|---|---|
| W1, W4 | weeks | At each week end, rank the ten currencies by their return over the last 1 week (W1) or 4 weeks (W4). Top three against bottom three, held one week. |
| T1, T3, T6, T12 | months | At each month end, for each pair: long if its return over the last 1, 3, 6 or 12 months is positive, short otherwise, 1/9 each. |
| X1, X3, X6, X12 | months | At each month end, rank by return over the last 1, 3, 6 or 12 months. Top three against bottom three. |
| S1 | calendar | At each month end, rank the currencies by their average return in the coming calendar month over all earlier years since 1975 (at least ten). Top three against bottom three. |
| S2 | calendar | At each month end: long all nine currencies against the dollar (1/9 each) if their average return in the coming calendar month over all earlier years is positive, short all nine if negative. |
| V1 | years | At each month end, rank by how cheap the currency is: minus (log real exchange rate last month, less its average over the previous 60 months). Top three against bottom three. |
| V2 | years | At each month end, rank by minus the return over the last 60 months. Top three against bottom three. |
| K1 | blend | One third each of the weights of B1 (carry, as specified on 2026-11-10), X3 and V1. This is the textbook combination and is fixed here, not chosen from the scan. |

That is sixteen rules. For each of W1, W4, the four T, the four X, S1, S2, V1 and V2 the scan also considers the opposite sign (for example "last week's losers bounce" instead of "last week's winners continue"), because either direction would count as a pattern. K1 has one sign. So the scan looks at 31 candidates.

**Also reported from the scan, descriptive only**: for each pair, the correlation of one period's return with the next at 1 week, 1 month, 3 months and 12 months; the average return by calendar month and by weekday with t-statistics, and how many of those are beyond 2 against the number expected by chance.

## 5. Stage A: shortlist rule

For each candidate, on 1987-2014: Sharpe ratio of the daily return in excess of dollar cash, after all costs.

A candidate is shortlisted if its Sharpe is at least 0.40 over 1987-2014 **and** its return after costs is positive in both halves. If more than five qualify, the five with the highest Sharpe are taken, and a rule and its opposite cannot both be taken. **If none qualifies, the exam is not run and the answer is H0.**

The threshold is set at 0.40 because published studies of these same patterns report ratios of 0.4 to 0.7 before retail costs over roughly this period; a pattern weaker than that in the data it was found in has no claim to a test.

## 6. Stage B: one look at 2015-01-01 to 2026-10-31

Each shortlisted candidate is run once, unchanged. The exam window has already faced the fifteen candidates of 2026-11-10, so the significance bar is shared across 15 + n candidates, where n is the number shortlisted (n = 5 gives a t-statistic of 2.81, which over this window is a Sharpe ratio of about 0.82).

**PASS**, all of:

1. t-statistic of the mean daily excess return at or above the shared bar;
2. Sharpe ratio at least 0.5;
3. largest fall at most 25%;
4. positive from 2022-01-01 on.

**PROVISIONAL**: 2, 3 and 4 hold and 1 does not.

**FAIL**: anything else.

Reported for each: return a year as run, and the return a year it would have given if sized to move 10% a year (the usual way these are compared), each calendar year, turnover, and the same figures before costs.

## 7. Decision per outcome

- **No shortlist, or all FAIL**: H0. Reported to Alex with both tables. No further forex patterns at these horizons on this data; a new idea would need information that is not in the price and interest-rate history.
- **PROVISIONAL**: reported as "held up but not distinguishable from luck". Proposal to Alex: a forward paper record (monthly positions written down before they are taken), no money.
- **PASS**: proposal to Alex of a forward paper record first, and a costed plan for what running it on a small account would take (broker, minimum sizes, financing charges as actually quoted). No money moves on a backtest.

The bot stays paused in every case.

## 8. Risks of a false result

- **The scan will find something.** Thirty-one candidates on 28 years: several will clear 0.40 by chance alone. That is the point of Stage B and of the shared bar.
- **Not blind.** The author knows the published record: carry, momentum and value in currencies were documented on data ending around 2010 and are widely reported to have weakened since, and B1 and B2 failed on 2015-2026. The author therefore expects H0. The rules above are taken from the published definitions without tuning so that expectation cannot shape them.
- **Policy rates are not the rates a trader earns.** They approximate deposit rates; the 1% markup is at the cheap end of retail financing.
- **Spot fixes are not tradable prices**, and the Bank of England and ECB fix at different times of day. At weekly and monthly horizons this is small against the moves.
- **Real exchange rates are today's vintage.** The BIS revises them. The one-month lag covers publication delay, not revisions.
- **Majors only.** Emerging-market currencies pay higher interest and have wider retail spreads, larger financing markups and occasional devaluations. They are not tested here because the retail cost of holding them cannot be established from free data.
- **Determinism**: each stage is run twice; outputs must match.
