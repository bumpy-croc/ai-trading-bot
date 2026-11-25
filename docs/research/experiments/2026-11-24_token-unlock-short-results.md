# Shorting coins around large scheduled token unlocks: results

**Date**: 2026-11-24
**Preregistration**: `2026-11-24_token-unlock-short-prereg.md` (commit `2acd983`, pushed before the simulator saw a real price).
**Script**: `experiments/strategy_search/unlocks.py design`. Output: `results/unlock_design.json`, `results/unlock_shortlist.json` (empty), `results/unlock_events.csv`. Two runs, identical output.
**Data**: prices and events cut at 2024-12-31. Fit 2022-23, validation 2024. **The 2025-26 window was not touched**: no window reached the shortlist, so Stage B did not run.

## Verdict

**H0 is supported at the design stage. No window qualifies, no exam, no build.**

Short the coin, long a basket of similar coins (liquidity ranks 31 to 100), 1.0% round-trip cost plus actual funding, 30% stop:

| Window | Period | Positions | Mean net per position | Median net | Winners | Stopped at −30% | Portfolio Sharpe | Return / yr | Drawdown |
|---|---|---|---|---|---|---|---|---|---|
| U1: 14 days before to the unlock day | 2022-23 | 132 | −1.7% | +1.2% | 57% | 10 | −0.76 | −4.7% | 19.5% |
| | 2024 | 166 | +0.9% | +1.3% | 57% | 9 | −0.11 | −0.7% | 16.7% |
| U2: 7 days before to 7 days after | 2022-23 | 128 | −0.8% | +0.7% | 52% | 8 | −0.60 | −3.0% | 17.1% |
| | 2024 | 170 | −0.4% | −0.1% | 49% | 8 | −0.57 | −10.6% | 23.5% |
| U3: 30 days before to 14 days after | 2022-23 | 38 | −8.1% | −10.9% | 32% | 13 | −1.02 | −9.7% | 24.4% |
| | 2024 | 21 | +16.3% | +6.4% | 62% | 7 | 0.82 | +21.8% | 21.0% |

The shortlist needed a Sharpe above 0.7 in both periods, drawdown under 25% in both and a positive mean per position in both. U1 and U2 fail on Sharpe in both periods. U3 clears the Sharpe bar in 2024 and loses heavily in 2022-23. U3 has few positions because coins that unlock monthly merge into one long position.

## Reading it

- **There is a small drift, and the squeezes eat it.** In the two short windows a little over half of the positions win and the median position makes about 1%, before the losers are counted. Then about one position in fifteen hits the 30% stop in the two short windows, and one in three in the long one. A coin with a known unlock coming is a crowded short, and crowded shorts get squeezed. The mean is negative or near zero.
- **Against BTC the same trade looks excellent in 2024, and that is the trap.** With a BTC hedge the three windows show 51%, 58% and 100% for 2024. With the basket hedge they show −1%, −11% and +22%. The difference is small coins falling against BTC that year, unlock or no unlock. A study that compares unlock coins with BTC, or with their own earlier price, measures that drift and calls it an unlock effect. In 2022-23 the BTC-hedged versions made nothing either (−5%, 0%, −7% a year).
- **Costs are a third of the story.** Before the 1.0% round trip, the mean position in U1 was −0.7% (2022-23) and +1.9% (2024), and in U2 +0.2% and +0.6%. Even the gross numbers do not show an effect that holds in both periods.

## Limits

As preregistered: schedules are DefiLlama's current record; the 2022-23 period has 125 events concentrated in a few monthly unlockers (ENS, CELO, IMX); re-sizing costs are not modelled. None of these would turn a loss into a pass.

## Decision (as preregistered)

No shortlist at Stage A: H0, no exam, no build. The 2025-26 window was not used for this idea, so it stays at eleven trading candidates and two yield mixes. No further windows, thresholds or hedges for unlock shorts on this data.
