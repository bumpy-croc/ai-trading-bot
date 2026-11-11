# Wide strategy search (Phase 1b): results

**Date**: 2026-11-10
**Prereg**: [2026-11-10_strategy-search-prereg.md](2026-11-10_strategy-search-prereg.md) (families, parameters, windows and gates unchanged). Issue #1324.
**Scripts**: `experiments/strategy_search/` (`python run_search.py discover`, then `python run_search.py holdout`). Raw numbers: `experiments/strategy_search/results/*.json`.

## Part 1: discovery (2020-07 → 2024-12 only), recorded before the holdout was read

The discovery run truncates the data at 2024-12-31 in code. Two consecutive runs produced byte-identical `discovery.json` (sha256 `e93694bb…`).

### Data actually used

- 918 USDT perpetuals in the archive; 166 of them stopped trading before 2026-10-31 (delisted coins are in). 175 distinct coins were in the top-30 universe at some point in 2020-24.
- Universe size: 30 coins from early 2021 on; it averaged 21 in 2020 while Binance still had few seasoned perps. The quintile rule (minimum 3 per leg) covers that period.
- Funding coverage inside the universe: 100% in discovery.

### Implementation notes and deviations from the prereg (all fixed before the first strategy run)

1. **Start date.** The archive's perp klines start 2020-01-01, not 2019-09-01. The fit window (from 2020-07) is unaffected.
2. **Non-crypto perps.** Binance now lists about 240 tokenised equities, commodities, FX and pre-IPO perps. As the prereg anticipated, they are excluded: the list is everything `exchangeInfo` marks as `contractType != PERPETUAL` or `underlyingType != COIN` (`non_crypto_symbols.json`, 242 symbols), plus the hand list in `engine.EXCLUDE`. All of them listed after 2024, so discovery is unaffected.
3. **Settled contracts.** A delisted perp keeps printing flat zero-volume klines in the archive. Each series is cut after its last day with real volume, so a delisting is treated as one (56,682 phantom symbol-days removed).
4. **Archive holes.** 1,076 missing symbol-days (2026-10-28 is absent for every symbol; a few dozen coins miss 2-5 days in early 2022). 947 were filled from the exchange's REST klines, the other 129 by carrying the last close forward (zero volume). 1,378 missing funding days were filled from REST; 805 remain, none inside the universe in discovery.
5. **Simulator checks.** `selfcheck.py` verifies fill timing, costs, funding sign and delisting on synthetic data, and that no family earns anything on i.i.d. noise (worst |t| 2.4 across 19 variants, against > 10 for a deliberately peeking control).

### Discovery table

Sharpe is net, in excess of 4%/yr idle USDT, one-day-delayed fills, 15% vol target (CC at natural size). `*` = the family's representative (best fit Sharpe).

| Variant | Fit Sharpe | Fit CAGR | Fit DD | Val Sharpe | Val CAGR | Val DD | Cost/yr | Same-close Sharpe (fit / val) |
|---|---|---|---|---|---|---|---|---|
| T fast (7,14,28) | 0.66 | 14.6% | 16.0% | -0.12 | 0.4% | 28.0% | 2.5% | 0.41 / -0.18 |
| *T slow (30,60,120) | 0.86 | 19.7% | 15.1% | 0.52 | 12.2% | 17.5% | 1.8% | 0.75 / 0.45 |
| XM 7 | -0.02 | 2.2% | 13.4% | -0.89 | -11.2% | 33.1% | 6.3% | 0.27 / -0.93 |
| XM 14 | 0.69 | 16.5% | 15.7% | 0.46 | 10.8% | 18.1% | 5.2% | 0.49 / 0.23 |
| *XM 28 | 0.74 | 16.3% | 13.8% | 0.14 | 5.0% | 18.6% | 4.1% | 0.48 / 0.13 |
| XM 56 | -0.16 | -0.5% | 26.7% | 0.25 | 7.0% | 17.8% | 3.1% | -0.30 / 0.33 |
| XR 1 | -2.83 | -34.8% | 65.7% | -1.90 | -24.3% | 48.0% | 39.9% | -2.80 / -2.83 |
| XR 3 | -2.76 | -35.2% | 67.2% | -1.11 | -13.9% | 27.0% | 24.5% | -2.83 / -1.60 |
| *XR 7 | -0.71 | -9.3% | 24.8% | -0.89 | -11.5% | 27.9% | 6.3% | -1.17 / -0.89 |
| ***XC 3** | **1.23** | 26.3% | 24.9% | **1.09** | 23.7% | 13.0% | 5.4% | 1.32 / 1.65 |
| XC 7 | 0.81 | 17.8% | 27.9% | 1.53 | 33.0% | 12.4% | 4.9% | 0.51 / 1.87 |
| XC 14 | 0.80 | 17.7% | 28.0% | 1.17 | 24.6% | 11.0% | 3.6% | 0.65 / 1.91 |
| CC 7 | 3.77 | 10.3% | 0.9% | -2.75 | 1.6% | 1.0% | 3.2% | 3.85 / -2.70 |
| *CC 14 | 4.20 | 10.8% | 0.9% | -2.81 | 0.8% | 2.2% | 2.7% | 4.29 / -2.71 |
| *AB dollar-neutral | -0.22 | -1.0% | 30.6% | 0.30 | 7.7% | 13.9% | 1.1% | -0.11 / 0.34 |
| AB beta-neutral | -0.28 | -1.9% | 28.5% | 0.81 | 16.9% | 14.9% | 1.1% | -0.16 / 0.90 |
| LV vol | -0.20 | -2.5% | 35.5% | -0.47 | -5.3% | 25.9% | 2.1% | 0.02 / -0.63 |
| *LV vol, equal risk | 0.46 | 11.0% | 19.8% | 0.96 | 21.1% | 21.0% | 2.2% | 0.51 / 1.06 |
| LV max | -0.39 | -5.8% | 35.7% | -0.53 | -6.4% | 25.8% | 2.7% | -0.09 / -0.80 |
| NL 7-60 | -0.65 | -8.5% | 32.8% | -0.21 | -2.3% | 28.2% | 1.7% | -0.83 / -0.11 |
| *NL 7-120 | -0.20 | -0.5% | 23.8% | 0.30 | 7.8% | 20.9% | 1.2% | -0.19 / 0.19 |
| BTC buy and hold | 0.63 | 26.7% | 76.7% | 1.93 | 137.6% | 26.3% | | |
| Top-30 equal weight, long | 0.56 | 8.8% | 89.0% | 0.77 | 40.0% | 59.0% | | |

### Selection (prereg section 4)

| Family | Representative | Fit > 0.7 | Val > 0.7 | Val DD < 25% | All variants positive | Result |
|---|---|---|---|---|---|---|
| T trend L/S | T slow | ok | FAIL (0.52) | ok | ok | dropped |
| XM momentum | XM 28 | ok | FAIL (0.14) | ok | FAIL | dropped |
| XR reversal | XR 7 | FAIL | FAIL | FAIL | FAIL | dropped |
| **XC funding carry across perps** | **XC 3** | ok (1.23) | ok (1.09) | ok (13.0%) | ok | **SHORTLISTED** |
| CC cash-and-carry rotation | CC 14 | ok | FAIL (1.6% and 0.8% a year, below idle USDT) | ok | ok | dropped |
| AB BTC vs alts | AB dollar | FAIL | FAIL | ok | ok | dropped |
| LV low-vol / lottery | LV equal risk | FAIL (0.46) | ok | ok | FAIL | dropped |
| NL short new listings | NL 7-120 | FAIL | FAIL | ok | FAIL | dropped |

**Shortlist: one candidate, `XC_3`.** With a single family there is no mix candidate (the prereg needs two or more). XC_3 becomes candidate number 4 on the 2025-01 → 2026-10 exam window.

What XC_3 is: every Monday, among the 30 most liquid crypto perps, go long the six with the lowest average funding over the last 3 days and short the six with the highest, equal weight, dollar-neutral, sized to 15% annual volatility.

Things to know about it before the holdout is read:

- Its correlation with BTC is -0.02 over 2020-24. It is market-neutral in practice, not only on paper.
- Calendar years: 2020 (Jul-Dec) +48.5%, 2021 +48.5%, **2022 -19.6%**, 2023 +13.5%, 2024 +34.9%. All three lookbacks lost about 20% in 2022. That is the year LUNA and FTX failed; coins with deeply negative funding (heavily shorted) were on the long side.
- Funding collected is 10 points a year in fit and 12 in validation, at an average gross exposure of 0.5-0.6x; the rest of the return is the high-funding coins underperforming the low-funding ones. Trading costs take 4.3% a year in fit and 5.4% in validation (31x and 37x turnover).
- Worst single day: -5.8%.

### What the dropped families say

- **Trend long/short** made money in both periods (Sharpe 0.86 then 0.52) but not enough to clear the bar. It is the nearest miss and the natural diversifier for carry; under the prereg it does not go forward.
- **Momentum** between coins is weak and unstable across lookbacks. **Reversal** loses at every horizon once fills are delayed a day (and also with same-close fills): the bounce is not there at daily frequency in liquid perps.
- **Cash-and-carry rotation** earned 10% a year in 2020-22 and about 1% in 2023-24: the same decay #1315 found for BTC/ETH, now shown for the highest-funding coins too. The spot-hedged version of carry is arbitraged out; the unhedged cross-sectional version (XC) is not, because it carries real price risk.
- **BTC vs alts**, **low-vol** and **short new listings** each worked in one period and not the other.

## Part 2: holdout (2025-01-01 → 2026-10-31), read once, after the shortlist was committed (1c4667d)

Two consecutive runs produced byte-identical `holdout.json`. The fit and validation numbers recomputed on the full data match the discovery file exactly, so nothing shifted when the holdout rows were added.

| Candidate | Sharpe | CAGR | Max DD | Worst day | 2025 | 2026 (Jan-Oct) | Corr. to BTC | Verdict |
|---|---|---|---|---|---|---|---|---|
| **XC_3** funding carry across perps | **-1.21** | **-48.4%** | **74.0%** | **-49.2%** | -73.1% | +10.6% | -0.03 | **FAIL** (gates 1, 2, 3, 4) |
| BTC buy and hold | -0.12 | -9.7% | 53.0% | | | | | |
| Top-30 equal weight, long | -0.48 | -48.4% | 81.6% | | | | | |

Same-close-fill sensitivity: Sharpe -0.94. The failure is not an execution-timing artefact.

### Verdict against the prereg

**H0 is supported. No candidate clears the holdout gate.** Of 8 families and 21 variants, one reached the holdout and it failed on four of five gates. By section 6 of the prereg this is the "shortlist non-empty, nothing passes" branch: **no build**, and the 2025-01 → 2026-10 exam window is retired for strategies on this price panel (4 candidates used).

### What happened to XC_3

On 2025-09-01 the rule shorted MYX at about 5% of equity: it had the highest 3-day funding in the universe. MYX then went from $1.31 to $13.89 in two days (2025-09-07 and 09-08, a 10.6x move on $6bn of daily volume). An unhedged short with no stop turned a 5% position into a 49% one-day loss, after 9.5% the day before and 28.5% the day after. That is a real event in real data, not a data fault, and it is exactly the short-squeeze tail the prereg listed as a risk.

### Exploratory, post-hoc diagnostics (`explore_tail.py`). These cannot change the verdict.

Every choice below was made after the holdout was read, so none of it is evidence a strategy works. It answers one question: was XC_3 a good strategy with one unlucky day, or had the premium gone?

| Version | Fit Sharpe | Val Sharpe | Holdout Sharpe | Holdout CAGR | Holdout DD | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| As preregistered | 1.23 | 1.09 | -1.21 | -48.4% | 74.0% | -73.1% | +10.6% |
| Shorts closed at +100% from the rebalance price (what 1x isolated margin does) | 1.16 | 1.00 | -0.51 | -6.1% | 23.9% | -17.0% | +7.3% |
| Shorts closed at +50% | 1.08 | 1.09 | -0.78 | -9.9% | 27.1% | -22.5% | +6.6% |
| Shorts closed at +25% | 0.90 | 0.81 | -0.84 | -10.9% | 30.6% | -24.1% | +6.7% |

- With the three MYX days simply deleted, the unlevered holdout Sharpe is 0.13 (rf = 0) and the total return is -10%.
- A loss cap removes the catastrophe but not the loss. **The premium itself was gone in 2025-26.** Funding was still collected (16-18% a year in the capped versions, more than in 2020-24), but the price leg lost more than that: the high-funding coins kept rising instead of fading.
- Squeezes became more frequent: shorts hit the +100% cap once or twice a year in 2020-23, never in 2024, and five times in each of 2025 and 2026.

Reading: through 2024, crowded longs paid shorts to take the other side and then usually deflated. In 2025-26 the crowded longs were right more often, or shorts were squeezed before they could collect. Either the trade got crowded from the short side, or the market's microstructure changed (thin-float tokens whose perp volume dwarfs their float). In both readings it is not something to put money on now.

## What this experiment established

1. **Nothing in the eight families survives.** Long/short and market-neutral daily strategies on the full Binance perp universe do not clear a realistic bar once costs, funding, a one-day delay and an out-of-sample period are applied.
2. **The one real-looking premium (cross-sectional funding carry) decayed and then blew up.** Its 2020-24 record was Sharpe 1.1-1.2 with a correlation to BTC of zero. Anyone who deployed it at the end of 2024 on that evidence would have lost about half their capital in 2025, or 6-11% with loss caps.
3. **Spot-hedged carry on the highest-funding coins decayed in the same way as BTC/ETH carry** (10% a year in 2020-22, about 1% in 2023-24). This confirms #1315 on a wider set.
4. **Long/short trend is the nearest miss** (Sharpe 0.86 in fit, 0.52 in validation). It did not qualify and was not run on the holdout.
5. **Daily reversal and momentum between coins are not there** in liquid perps at these costs.
6. Method note for future work: any strategy that shorts individual coins must be simulated with an explicit per-position loss cap from the start, and its worst day must be a gate, not a secondary.

## Decision implied by the prereg

No build and no deployment. The bot stays halted. Any further crypto work needs a mechanism that is not in this list, and a clean test for it (new data, a different venue pair, or forward paper trading), not a re-tuned version of these families.
