# Strategy bake-off (Phase 1): results

**Date**: 2026-11-10
**Prereg**: [2026-11-10_strategy-bakeoff-prereg.md](2026-11-10_strategy-bakeoff-prereg.md) (parameters and gates unchanged)
**Script**: `experiments/strategy_bakeoff.py` (re-run: `python experiments/strategy_bakeoff.py`; `--refresh` re-downloads)
**Data**: Binance spot daily klines, USDⓈ-M perp daily klines (BTC, ETH) and USDⓈ-M funding history, 2019-01-01 to 2026-10-31, cached under `experiments/bakeoff_data/` (not committed, 4.4 MB, re-downloadable).

## Verdict

| Gate | S1 carry | S2 trend+vol | S3 XS momentum | BTC buy&hold (control) |
|---|---|---|---|---|
| G1 IS Sharpe > 1.0 | PASS (7.78) | PASS (1.64) | PASS (1.67) | PASS (1.12) |
| G2 IS max DD < 25% | PASS (1.1%) | PASS (15.8%) | FAIL (50.6%) | FAIL (76.6%) |
| G3 >= 4/5 positive years 2020-24 | PASS (4/5) | PASS (4/5) | PASS (4/5) | PASS (4/5) |
| G4 Holdout Sharpe > 0.5 and DD < 25% | PASS (3.80, 0.5%) | FAIL (-0.11, 12.5%) | FAIL (-0.02, 32.4%) | FAIL (-0.02, 53.0%) |
| G5 ±50% lookbacks keep IS Sharpe >= 0 | PASS | PASS | PASS | n/a |
| **Final** | **PASS** | **FAIL** | **FAIL** | FAIL |

S1 (funding carry) is the only strategy that clears every preregistered gate, so by the prereg it goes to Phase 2 (8 weeks of paper trading on staging). S2 and S3 fail on the 2025-2026 holdout.

**Read before acting on the S1 pass.** It passes on Sharpe, which the prereg measures with rf = 0. Its absolute return is small and shrinking: 6.7% CAGR in 2020-2024, 0.9% CAGR in the holdout. BTC funding averaged 14.4%/yr on notional in 2020-2024 and 4.2%/yr in 2025-2026 (ETH: 17.4% to 3.6%). With half the capital as collateral, the strategy earns about half of that, minus costs. A 0.9% return is below what idle USDT earns in a savings product, so on a cash-adjusted basis the holdout is roughly break-even or negative. The gate passed as written; whether it is worth eight weeks of paper trading is a Board call, not something these gates answer.

## Metrics

IS = 2020-01-01 to 2024-12-31. Holdout = 2025-01-01 to 2026-10-31. Fees are total fees and slippage over the window as a fraction of the window's starting capital. Turnover is one-sided traded notional / equity per year.

### S1 Funding carry (BTC + ETH, long spot / short perp)

| | IS | Holdout | Full 2019-2026 |
|---|---|---|---|
| CAGR | 6.7% | 0.9% | 4.5% |
| Sharpe | 7.78 | 3.80 | 5.96 |
| Max drawdown | 1.1% | 0.5% | 1.1% |
| Fees paid | 7.6% | 1.7% | 10.7% |
| Turnover / yr | 5.1x | 4.4x | 4.8x |
| Avg gross exposure | 0.83 | 0.70 | 0.71 |

Years: 2019 0.0%, 2020 10.0%, 2021 16.8%, 2022 -0.8%, 2023 3.2%, 2024 5.5%, 2025 1.4%, 2026 (to Oct) 0.2%.
Robustness (IS Sharpe): funding lookback 11 prints (3.5d) 7.11, 32 prints (10.5d) 8.06.
Time in position, IS: BTC 82%, ETH 83%.

### S2 Trend with volatility targeting

| | IS | Holdout | Full |
|---|---|---|---|
| CAGR | 22.9% | -1.8% | 13.2% |
| Sharpe | 1.64 | -0.11 | 1.12 |
| Max drawdown | 15.8% | 12.5% | 15.8% |
| Fees paid | 4.3% | 0.8% | 6.5% |
| Turnover / yr | 3.1x | 3.1x | 2.7x |
| Avg gross exposure | 0.16 | 0.16 | 0.14 |

Years: 2019 -2.3%, 2020 35.6%, 2021 50.7%, 2022 -10.6%, 2023 19.0%, 2024 29.0%, 2025 0.8%, 2026 -4.0%.
Robustness (IS Sharpe): SMA 50 1.73, SMA 150 1.69, vol 15 1.46, vol 45 1.66.

### S3 Cross-sectional momentum

| | IS | Holdout | Full |
|---|---|---|---|
| CAGR | 244.2% | -9.1% | 110.6% |
| Sharpe | 1.67 | -0.02 | 1.26 |
| Max drawdown | 50.6% | 32.4% | 50.6% |
| Fees paid | 2401% | 6.4% | 4636% |
| Turnover / yr | 24.3x | 23.2x | 21.6x |
| Avg gross exposure | 0.62 | 0.46 | 0.52 |

Years: 2019 -15.9%, 2020 288.2%, 2021 3460.6%, 2022 -16.1%, 2023 143.9%, 2024 72.0%, 2025 7.4%, 2026 -21.8%.
Robustness (IS Sharpe): SMA 50 1.62, SMA 150 1.62, momentum 14 1.98, momentum 42 1.48.
Fees look huge only because they are measured against starting capital while equity compounded ~485x; as a drag they are about 24x turnover x 15 bps = 3.6%/yr. The 2021 number is dominated by holding DOGE/SOL-type moves, which is where survivorship bias bites hardest (see caveats).

### Control: BTC buy and hold

| | IS | Holdout | Full |
|---|---|---|---|
| CAGR | 66.9% | -9.7% | 46.6% |
| Sharpe | 1.12 | -0.02 | 0.94 |
| Max drawdown | 76.6% | 53.0% | 76.6% |

Years: 2019 86.2%, 2020 302.0%, 2021 59.8%, 2022 -64.2%, 2023 155.6%, 2024 121.3%, 2025 -6.3%, 2026 -11.5%.

## How the prereg was implemented

Choices the prereg left open, fixed before any results were looked at:

- **Timing.** Signals use day t's close; fills happen at day t+1's close. Daily candles are Binance UTC days.
- **Annualisation.** 365 days for Sharpe and vol (crypto trades every day).
- **Eligibility.** A coin is eligible once it has 200 daily closes on Binance spot. This rule was not varied in the robustness runs (it is an eligibility filter, not a strategy lookback).
- **S1 funding.** "Trailing 7-day average" = mean of the last 21 funding prints known at the day's close. A print stamped 00:00 UTC on day d+1 is counted as settling at day d's close. Hysteresis: start flat; enter when the average is above +0.005%, exit when below 0. Pair P&L per day = notional x (spot return - perp return + funding received). Robustness lookbacks: 21 prints ±50% → 11 and 32 (rounded half up).
- **S1 capital.** Each coin gets 50% of capital; half of that buys spot and half is perp margin, so each coin's hedged notional is 25% of capital. Entry/exit cost = 15 bps on the spot leg + 10 bps on the perp leg.
- **S1 rebalancing cost (added, conservative).** Keeping the short at 1x and the margin topped up means moving each day's spot P&L across legs. The script charges |spot return| x notional x 15 bps per day held. The prereg does not list this cost; adding it makes S1 slightly worse, not better. Without it, liquidation risk on the 1x short is ignored entirely, which would be the less honest choice.
- **S2.** Rebalance every Monday close (filled Tuesday close). N = number of eligible coins (not the number with a long signal). Vol = 30-day stdev of daily simple returns x √365. If the sum of weights exceeds 1.0, all weights are scaled down to sum to 1.0. Holdings drift with prices between rebalances.
- **S3.** Rebalance every Monday close. Regime: BTC close > BTC 100-day SMA. Ranking among eligible coins by 28-day return. Robustness varies the SMA (50/150) and the momentum lookback (14/42) one at a time.
- **Costs.** Spot 15 bps per side on traded notional (10 taker + 5 slippage); perp 10 bps per side. Rebalances pay cost on the full weight change. Cash earns nothing.

## Caveats

- **Survivorship bias.** The universe is today's large caps, picked with hindsight. Coins that were large in 2019-2021 and then collapsed (e.g. LUNA, FTT, EOS, BCH-era names) are absent. This flatters S2 and especially S3, which buys the biggest winners. It does not affect S1, which only trades BTC and ETH.
- **Data coverage.** No missing days in any spot series after listing. Late listings: SOL spot from 2020-08-11 (eligible 2021-02-26), DOT from 2020-08-18 (eligible 2021-03-05), AVAX from 2020-09-22 (eligible 2021-04-09), DOGE from 2019-07-05 (eligible 2020-01-20), LINK from 2019-01-16. Perp/funding history starts 2019-09-10 (BTC) and 2019-11-27 (ETH), so S1 cannot trade before then; that does not touch the gate windows. One day in the funding history has 2 prints instead of 3. Funding history for the other ten coins was downloaded but is not used (S1 is BTC/ETH only).
- **S1 risks not in the numbers.** Exchange counterparty risk (spot and perp both on Binance), funding regime change (already visible in the holdout), and margin mechanics: with daily rebalancing between legs assumed, a fast spot rally within a day could still force a margin call on the 1x short. The daily-close model cannot see intraday basis blowouts.
- **Low exposure in S2.** The prereg's per-coin vol target (20% / N) gives an average gross exposure of only ~15%. Its risk-adjusted IS numbers are decent, but in absolute terms it is a mostly-cash strategy.
- **IS is not truly out-of-sample.** As the prereg says, these designs were picked by people who knew 2020-2024. The holdout is the only clean test; S2 and S3 fail it, S1 passes it with a small absolute return.

## Decision implied by the prereg

S1 passes → it is the Phase 2 candidate (8 weeks of paper trading on staging). S2 and S3 fail and are dropped. Given the holdout return of 0.9%/yr, the Board should decide whether a carry strategy with that return level is worth Phase 2, or whether the Phase 1 fail branch is the better reading in practice.
