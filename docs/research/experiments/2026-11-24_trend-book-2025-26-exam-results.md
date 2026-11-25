# The trend book on 2025-26, once: results

**Date**: 2026-11-24
**Preregistration**: `2026-11-24_trend-book-2025-26-exam-prereg.md` (commit `9d3b792`, pushed before either book was run on any day after 2024-12-31).
**Script**: `experiments/strategy_search/adaptive.py exam`. Output: `results/trend_exam.json`, `results/trend_exam_daily_returns.csv`. Two runs, identical output (daily-return hash `73161316…`).
**Window**: 2025-01-01 to 2026-10-31, 669 days. Candidates ten and eleven on this window.

## Verdict

**Both books FAIL. H0 is supported for B0 and for Q1.**

The plain trend book made 4.6% a year after costs, which is what idle dollars earn (4%). It did not lose money, in a period when the coins it trades lost most of their value, but it did not earn anything for the risk either. The quiet-market version did somewhat better over the whole window and lost money in 2025.

| | B0 plain trend | Q1 quiet markets only | Idle dollars | BTC held | Top 30 coins held, equal weight |
|---|---|---|---|---|---|
| Return per year | 4.6% | 7.9% | 4.0% | −9.7% | −48.4% |
| Total over the window | +8.6% | +14.9% | +7.5% | −17.1% | −70.3% |
| Sharpe above idle dollars | 0.12 | 0.36 | 0 | −0.12 | −0.48 |
| t-statistic (2.61 required) | 0.16 | 0.48 | | | |
| Maximum drawdown | 14.1% | 12.9% | 0 | 53.0% | 81.6% |
| 2025 | +4.2% | −5.4% | | | |
| 2026 (Jan-Oct) | +4.3% | +21.5% | | | |
| Worst day | −5.3% | −4.8% | | | |

### Gates

| Gate | B0 | Q1 |
|---|---|---|
| 1. t-statistic at least 2.61 | fail (0.16) | fail (0.48) |
| 2. Return at least 10% a year | fail (4.6%) | fail (7.9%) |
| 3. Drawdown at most 20% | pass | pass |
| 4. 2025 and 2026 both positive | pass | fail (2025 negative) |
| 5. Exposure on at least 80% of days (B0 only) | pass (100%) | n/a (on in 58% of weeks) |
| CONSISTENT: Sharpe at least 0.5, above idle dollars, drawdown at most 20% | no (Sharpe 0.12) | no (Sharpe 0.36) |
| Verdict | **FAIL** | **FAIL** |

Q1's Sharpe beat B0's by 0.24, more than the 0.15 margin, but it had to be at least CONSISTENT as well, and it was not. By the preregistered rule it is closed and is not added to the forward recorder.

### Secondaries (never decisive)

| | B0 | Q1 |
|---|---|---|
| Volatility | 16.8% | 12.4% |
| Turnover per year | 10.1x | 7.9x |
| Trading cost per year | 1.4% | 1.1% |
| Funding per year | received 0.8% | received 0.2% |
| Average gross exposure | 0.31 | 0.18 |
| Average net exposure | −0.04 | +0.02 |
| Correlation with BTC | −0.35 | −0.01 |
| Sharpe if filled at the signal close | 0.13 | 0.27 |

B0's 2020-24 figures in the same output reproduce the design run exactly (Sharpe 0.86 fit, 0.52 validation).

## Reading it

- **The decay continued.** B0's Sharpe was 0.86 in 2020-22, 0.52 in 2023-24 and 0.12 in 2025-26. Its return went from 20% a year to 12% to under 5%. The result is on the same line as the design numbers, pointing down.
- **It did what trend following is for, and that was not enough.** The 30 coins it trades lost 70% held equally, and BTC lost 17%. B0 was net short on average, correlated −0.35 with BTC, and finished up 8.6%. It protected capital in a falling market. It did not turn the fall into a return worth having: the shorts' gains were offset by losses in the sharp rebounds.
- **Q1 is not rescued by its one good year.** It earned 21.5% in 2026 and lost 5.4% in 2025. One year out of two is what chance gives, which is the same reading applied to the machine-learning ranker on 2026-11-11.
- **A 22-month window is short.** A Sharpe of 0.12 does not prove the true figure is zero. It does mean there is no evidence for an edge here, on top of a declining design record.

## Decision (as preregistered)

- **B0**: recorded as failing its out-of-sample period. The forward paper recording continues to its fixed checkpoints (it costs nothing and its rules were set on 2026-11-11). The odds that it leads to a live trial are now low, and no work starts on a futures executor.
- **Q1**: closed. No further versions of the switch on historical data.
- The bot stays paused.

## Window accounting

The 2025-01 to 2026-10 window has now faced eleven daily-bar trading candidates (and the two yield mixes of the same date). None passed.
